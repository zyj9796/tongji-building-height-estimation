from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import contains_xy, distance, points
from shapely.geometry import Polygon
from shapely.ops import unary_union

from geometry import StrictRadarProjector, clean_ring, ear_clip_triangulation, llh_to_ecef


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


def load_local_rows(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    return {int(row.fid): float(row.applied_row_shift) for row in table.itertuples()}


def search_bounds(prior_m: float, cfg: dict) -> tuple[float, float]:
    minimum = max(float(cfg["minimum_height_m"]), math.floor(float(cfg["prior_lower_factor"]) * prior_m))
    upper_prior = max(
        float(cfg["prior_upper_factor"]) * prior_m + float(cfg["prior_upper_margin_m"]),
        prior_m + float(cfg["prior_minimum_span_above_m"]),
    )
    maximum = min(float(cfg["maximum_height_m"]), math.ceil(upper_prior))
    return minimum, maximum


@dataclass
class PreparedBuilding:
    fid: int
    clean_id: int
    prior_m: float
    ring: np.ndarray
    bottom_ecef: np.ndarray
    bottom_xy: np.ndarray
    footprint_triangles: np.ndarray
    minimum_m: float
    maximum_m: float
    row_shift: float
    col_shift: float
    corridor: object


def project_tops(projector: StrictRadarProjector, item: PreparedBuilding, heights_m: np.ndarray, base_m: float) -> np.ndarray:
    heights = np.asarray(heights_m, dtype=np.float64)
    n = item.ring.shape[0]
    lon = np.tile(item.ring[:, 0], heights.size)
    lat = np.tile(item.ring[:, 1], heights.size)
    elevation = np.repeat(base_m + heights, n)
    ecef = llh_to_ecef(lon, lat, elevation)
    xy = projector.project_ecef(ecef) + np.asarray([item.col_shift, item.row_shift])
    return xy.reshape(heights.size, n, 2)


def surfaces(item: PreparedBuilding, top_xy: np.ndarray) -> tuple[object, object, object]:
    roof = Polygon(top_xy).buffer(0)
    wall_triangles = []
    n = item.ring.shape[0]
    for index in range(n):
        following = (index + 1) % n
        wall_triangles.append(Polygon([item.bottom_xy[index], item.bottom_xy[following], top_xy[following]]).buffer(0))
        wall_triangles.append(Polygon([item.bottom_xy[index], top_xy[following], top_xy[index]]).buffer(0))
    walls = unary_union([triangle for triangle in wall_triangles if not triangle.is_empty]).buffer(0)
    volume = unary_union([roof, walls]).buffer(0)
    return roof, walls, volume


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, float(np.std(values)), 1e-9)
    return (values - median) / scale


def score_grid(
    item: PreparedBuilding,
    projector: StrictRadarProjector,
    heights: np.ndarray,
    ps_xy: np.ndarray,
    ps_weight: np.ndarray,
    base_m: float,
    cfg: dict,
) -> pd.DataFrame:
    top_grids = project_tops(projector, item, heights, base_m)
    point_geometry = points(ps_xy[:, 0], ps_xy[:, 1])
    rows: list[dict] = []
    roof_sigma = float(cfg["roof_distance_sigma_px"])
    edge_sigma = float(cfg["roof_edge_sigma_px"])
    for height, top_xy in zip(heights, top_grids, strict=True):
        roof, walls, _ = surfaces(item, top_xy)
        roof_inside = contains_xy(roof, ps_xy[:, 0], ps_xy[:, 1])
        wall_inside = contains_xy(walls, ps_xy[:, 0], ps_xy[:, 1])
        roof_distance = np.asarray(distance(point_geometry, roof), dtype=np.float64)
        edge_distance = np.asarray(distance(point_geometry, roof.boundary), dtype=np.float64)
        roof_kernel = np.exp(-0.5 * (roof_distance / roof_sigma) ** 2)
        edge_kernel = np.exp(-0.5 * (edge_distance / edge_sigma) ** 2)
        roof_density = float(np.sum(ps_weight * roof_kernel) / max(roof.area, 1.0))
        edge_support = float(np.sum(ps_weight * edge_kernel) / max(roof.length, 1.0))
        wall_density = float(np.sum(ps_weight[wall_inside]) / max(walls.area, 1.0))
        rows.append(
            {
                "height_m": float(height),
                "roof_density": roof_density,
                "roof_edge_support": edge_support,
                "wall_density": wall_density,
                "roof_inside_ps": int(np.sum(roof_inside)),
                "wall_inside_ps": int(np.sum(wall_inside)),
                "roof_near_ps": int(np.sum(roof_distance <= roof_sigma)),
                "roof_edge_near_ps": int(np.sum(edge_distance <= edge_sigma)),
            }
        )
    table = pd.DataFrame(rows)
    weights = cfg["component_weights"]
    table["roof_density_z"] = robust_z(table.roof_density.to_numpy())
    table["roof_edge_z"] = robust_z(table.roof_edge_support.to_numpy())
    table["wall_density_z"] = robust_z(table.wall_density.to_numpy())
    table["score"] = (
        float(weights["roof_density"]) * table.roof_density_z
        + float(weights["roof_edge"]) * table.roof_edge_z
        + float(weights["wall_density"]) * table.wall_density_z
    )
    return table


def infer_quality(
    curve: pd.DataFrame,
    best_index: int,
    minimum_m: float,
    maximum_m: float,
    fine_step: float,
    candidate_ps: int,
) -> dict:
    best = curve.iloc[best_index]
    best_height = float(best.height_m)
    separated = curve.loc[np.abs(curve.height_m - best_height) >= 1.0, "score"]
    second_score = float(separated.max()) if len(separated) else float(best.score)
    gap = float(best.score - second_score)
    near_peak = curve.loc[curve.score >= float(best.score) - 0.5, "height_m"]
    uncertainty = max(float(fine_step), 0.5 * float(near_peak.max() - near_peak.min())) if len(near_peak) else float("nan")
    boundary = bool(abs(best_height - minimum_m) <= fine_step or abs(best_height - maximum_m) <= fine_step)
    support = int(best.roof_near_ps + best.roof_edge_near_ps)
    if boundary or candidate_ps < 3 or support < 2 or gap < 0.10:
        quality = "low"
    elif support >= 8 and gap >= 0.35 and uncertainty <= 2.0:
        quality = "high"
    else:
        quality = "medium"
    return {
        "height_uncertainty_m": uncertainty,
        "peak_gap": gap,
        "boundary_hit": boundary,
        "quality": quality,
    }


def run(
    config_path: Path,
    max_buildings: int = 0,
    ps_points_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inv = config["height_inversion"]
    base_m = float(config["base_elevation_m"])
    inputs = config["inputs"]
    resolved_ps_points = (
        ps_points_path.resolve()
        if ps_points_path is not None
        else resolve(inputs["ps_points"])
    )
    output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else resolve(config["outputs"]["height_estimates"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    buildings = gpd.read_file(resolve(inputs["buildings"])).reset_index(drop=True)
    if buildings.crs is None:
        raise ValueError("Building CRS is missing")
    buildings_lonlat = buildings.to_crs("EPSG:4326").copy()
    buildings_lonlat["fid"] = np.arange(len(buildings_lonlat), dtype=np.int64)
    if max_buildings > 0:
        buildings_lonlat = buildings_lonlat.iloc[:max_buildings].copy()

    ps = pd.read_csv(resolved_ps_points)
    required_ps_columns = {"azimuth_pixel", "range_pixel", "coherence"}
    missing_ps_columns = sorted(required_ps_columns.difference(ps.columns))
    if missing_ps_columns:
        raise ValueError(
            f"PS input {resolved_ps_points} is missing required columns: "
            + ", ".join(missing_ps_columns)
        )
    input_ps_count = int(len(ps))
    ps = ps.loc[pd.to_numeric(ps.coherence, errors="coerce") >= float(config["minimum_ps_coherence"])].copy()
    offset = 1.0 if config["ps_pixel_indexing"] == "one_based" else 0.0
    ps["row0"] = pd.to_numeric(ps.azimuth_pixel, errors="coerce") - offset
    ps["col0"] = pd.to_numeric(ps.range_pixel, errors="coerce") - offset
    ps = ps.loc[np.isfinite(ps.row0) & np.isfinite(ps.col0)].reset_index(drop=True)
    ps_points = gpd.GeoDataFrame(ps, geometry=gpd.points_from_xy(ps.col0, ps.row0), crs=None)

    projector = StrictRadarProjector(resolve(inputs["rslc_par"]))
    local_rows = load_local_rows(resolve(inputs["local_registration"]))
    registration = config["registration"]
    prepared: list[PreparedBuilding] = []
    failures: list[dict] = []
    for building in buildings_lonlat.itertuples():
        fid = int(building.fid)
        try:
            prior = float(getattr(building, config["height_field"]))
            minimum_m, maximum_m = search_bounds(prior, inv)
            ring = clean_ring(np.asarray(building.geometry.exterior.coords))
            bottom_ecef = llh_to_ecef(ring[:, 0], ring[:, 1], np.full(len(ring), base_m))
            row_shift = float(registration["global_row_shift_px"])
            if registration["use_local_row_shift"]:
                row_shift += local_rows.get(fid, 0.0)
            col_shift = float(registration["global_col_shift_px"])
            bottom_xy = projector.project_ecef(bottom_ecef) + np.asarray([col_shift, row_shift])
            temporary = PreparedBuilding(
                fid=fid,
                clean_id=int(building.clean_id),
                prior_m=prior,
                ring=ring,
                bottom_ecef=bottom_ecef,
                bottom_xy=bottom_xy,
                footprint_triangles=ear_clip_triangulation(ring),
                minimum_m=minimum_m,
                maximum_m=maximum_m,
                row_shift=row_shift,
                col_shift=col_shift,
                corridor=None,
            )
            endpoint_tops = project_tops(projector, temporary, np.asarray([minimum_m, maximum_m]), base_m)
            endpoint_volumes = [surfaces(temporary, top_xy)[2] for top_xy in endpoint_tops]
            temporary.corridor = unary_union(endpoint_volumes).convex_hull.buffer(float(inv["corridor_buffer_px"]))
            prepared.append(temporary)
        except Exception as exc:
            failures.append({"fid": fid, "reason": str(exc)})

    corridors = gpd.GeoDataFrame(
        {"fid": [item.fid for item in prepared], "geometry": [item.corridor for item in prepared]},
        geometry="geometry",
        crs=None,
    )
    corridor_index = corridors.sindex
    ps_to_corridors: dict[int, list[int]] = {}
    ambiguity = np.zeros(len(ps_points), dtype=np.int32)
    for ps_index, point in enumerate(ps_points.geometry):
        positions = list(corridor_index.query(point, predicate="intersects"))
        ambiguity[ps_index] = len(positions)
        for position in positions:
            ps_to_corridors.setdefault(int(corridors.iloc[int(position)].fid), []).append(ps_index)

    estimates: list[dict] = []
    curves: list[pd.DataFrame] = []
    coarse_step = float(inv["coarse_step_m"])
    fine_step = float(inv["fine_step_m"])
    for index, item in enumerate(prepared, start=1):
        pool_indexes = np.asarray(ps_to_corridors.get(item.fid, []), dtype=np.int64)
        if pool_indexes.size == 0:
            estimates.append(
                {
                    "fid": item.fid,
                    "clean_id": item.clean_id,
                    "height_prior_m": item.prior_m,
                    "height_est_m": np.nan,
                    "height_uncertainty_m": np.nan,
                    "quality": "no_ps",
                    "candidate_ps": 0,
                    "search_min_m": item.minimum_m,
                    "search_max_m": item.maximum_m,
                }
            )
            continue
        pool = ps.iloc[pool_indexes]
        ps_xy = pool[["col0", "row0"]].to_numpy(dtype=np.float64)
        ps_weight = pool.coherence.to_numpy(dtype=np.float64) / np.sqrt(np.maximum(ambiguity[pool_indexes], 1))
        coarse = np.arange(item.minimum_m, item.maximum_m + 0.5 * coarse_step, coarse_step)
        coarse_curve = score_grid(item, projector, coarse, ps_xy, ps_weight, base_m, inv)
        coarse_best = float(coarse_curve.loc[coarse_curve.score.idxmax(), "height_m"])
        fine_min = max(item.minimum_m, coarse_best - float(inv["fine_half_window_m"]))
        fine_max = min(item.maximum_m, coarse_best + float(inv["fine_half_window_m"]))
        fine = np.arange(fine_min, fine_max + 0.5 * fine_step, fine_step)
        heights = np.unique(np.round(np.r_[coarse, fine], 6))
        curve = score_grid(item, projector, heights, ps_xy, ps_weight, base_m, inv)
        best_index = int(curve.score.to_numpy().argmax())
        best = curve.iloc[best_index]
        diagnostics = infer_quality(curve, best_index, item.minimum_m, item.maximum_m, fine_step, int(pool_indexes.size))
        estimates.append(
            {
                "fid": item.fid,
                "clean_id": item.clean_id,
                "height_prior_m": item.prior_m,
                "height_est_m": float(best.height_m),
                **diagnostics,
                "candidate_ps": int(pool_indexes.size),
                "best_roof_inside_ps": int(best.roof_inside_ps),
                "best_wall_inside_ps": int(best.wall_inside_ps),
                "best_roof_near_ps": int(best.roof_near_ps),
                "best_roof_edge_near_ps": int(best.roof_edge_near_ps),
                "best_score": float(best.score),
                "search_min_m": item.minimum_m,
                "search_max_m": item.maximum_m,
                "local_row_shift_px": item.row_shift - float(registration["global_row_shift_px"]),
            }
        )
        curve.insert(0, "fid", item.fid)
        curves.append(curve)
        if index % 100 == 0:
            print(f"processed {index}/{len(prepared)}", flush=True)

    estimates_table = pd.DataFrame(estimates)
    estimates_table["height_recommended_m"] = estimates_table["height_est_m"].where(
        estimates_table["quality"].isin(["medium", "high"])
    )
    curves_table = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    estimates_csv = output_dir / "building_heights_from_ps.csv"
    curves_csv = output_dir / "building_height_ps_score_curves.csv"
    failures_csv = output_dir / "height_inversion_failures.csv"
    estimates_table.to_csv(estimates_csv, index=False)
    curves_table.to_csv(curves_csv, index=False)
    pd.DataFrame(failures, columns=["fid", "reason"]).to_csv(failures_csv, index=False)

    vector = buildings.copy().reset_index(drop=True)
    vector["building_fid"] = np.arange(len(vector), dtype=np.int64)
    vector = vector.rename(columns={config["height_field"]: "height_shp_m"})
    estimate_attributes = estimates_table.drop(columns=["clean_id"], errors="ignore").rename(columns={"fid": "building_fid"})
    vector = vector.merge(estimate_attributes, on="building_fid", how="left")
    vector_path = output_dir / "building_heights_from_ps.gpkg"
    vector.to_file(vector_path, layer="ps_height_estimates", driver="GPKG")
    finite = estimates_table.loc[np.isfinite(estimates_table.height_est_m)]
    recommended = estimates_table.loc[
        np.isfinite(estimates_table.height_recommended_m)
    ]
    summary = {
        "method": "strict_candidate_triangle_projection_scored_only_by_ps_radar_coordinates_and_coherence",
        "uses_ps_height_m": False,
        "uses_ps_z_dsm_m": False,
        "input_ps_points": input_ps_count,
        "quality_filtered_ps_points": int(len(ps)),
        "ps_points_path": str(resolved_ps_points),
        "input_buildings": int(len(buildings_lonlat)),
        "prepared_buildings": int(len(prepared)),
        "estimated_buildings": int(len(finite)),
        "recommended_buildings": int(len(recommended)),
        "no_ps_buildings": int(np.sum(estimates_table.quality == "no_ps")),
        "projection_failures": int(len(failures)),
        "quality_counts": estimates_table.quality.value_counts().to_dict(),
        "boundary_hit_buildings": int(finite.boundary_hit.sum()) if len(finite) else 0,
        "uncertainty_median_m": float(finite.height_uncertainty_m.median()) if len(finite) else None,
        "uncertainty_p90_m": float(finite.height_uncertainty_m.quantile(0.9)) if len(finite) else None,
        "height_m": {
            "min": float(finite.height_est_m.min()) if len(finite) else None,
            "median": float(finite.height_est_m.median()) if len(finite) else None,
            "mean": float(finite.height_est_m.mean()) if len(finite) else None,
            "max": float(finite.height_est_m.max()) if len(finite) else None,
        },
        "recommended_height_m": {
            "min": float(recommended.height_recommended_m.min()) if len(recommended) else None,
            "median": float(recommended.height_recommended_m.median()) if len(recommended) else None,
            "mean": float(recommended.height_recommended_m.mean()) if len(recommended) else None,
            "max": float(recommended.height_recommended_m.max()) if len(recommended) else None,
        },
        "prior_is_only_search_bound": True,
        "warning": "This is an internal PS-geometry inversion without independent height truth. Quality measures peak identifiability, not external accuracy.",
        "outputs": {
            "estimates_csv": str(estimates_csv),
            "curves_csv": str(curves_csv),
            "vector_gpkg": str(vector_path),
            "failures_csv": str(failures_csv),
        },
    }
    summary_path = output_dir / "height_estimation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate building height by strict candidate projection scored only with PS radar coordinates and coherence.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-buildings", type=int, default=0)
    parser.add_argument(
        "--ps-points",
        type=Path,
        help="Override config inputs.ps_points without modifying the shared config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write this run to a separate directory instead of config outputs.height_estimates.",
    )
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.max_buildings,
        ps_points_path=args.ps_points,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
