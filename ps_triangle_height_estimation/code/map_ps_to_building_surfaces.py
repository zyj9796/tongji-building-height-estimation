from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.ndimage import binary_closing, binary_dilation, label
from shapely.geometry import Point, Polygon

from geometry import StrictRadarProjector, barycentric_weights, ecef_to_llh


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


def read_sar_amplitude(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Read a GAMMA complex-int16 RSLC and return normalized log amplitude."""
    raw = np.fromfile(path, dtype=">i2")
    expected = shape[0] * shape[1] * 2
    if raw.size != expected:
        raise ValueError(f"Unexpected RSLC size: {raw.size}, expected {expected}")
    iq = raw.reshape(*shape, 2).astype(np.float32)
    amplitude = np.log1p(np.hypot(iq[:, :, 0], iq[:, :, 1]))
    positive = amplitude[np.isfinite(amplitude) & (amplitude > 0)]
    low, high = (
        np.percentile(positive, [2.0, 98.0])
        if positive.size
        else (0.0, 1.0)
    )
    return np.clip(
        (amplitude - low) / max(float(high - low), 1e-6),
        0.0,
        1.0,
    ).astype(np.float32)


def rasterize_projected_triangles(
    projected_triangles: list[np.ndarray],
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize radar-coordinate triangles at SAR pixel centres."""
    rows, cols = shape
    mask = np.zeros(shape, dtype=bool)
    for xy in projected_triangles:
        if len(xy) != 3 or not np.all(np.isfinite(xy)):
            continue
        c0 = max(0, int(math.floor(float(np.min(xy[:, 0])))) - 1)
        c1 = min(cols - 1, int(math.ceil(float(np.max(xy[:, 0])))) + 1)
        r0 = max(0, int(math.floor(float(np.min(xy[:, 1])))) - 1)
        r1 = min(rows - 1, int(math.ceil(float(np.max(xy[:, 1])))) + 1)
        if c1 < c0 or r1 < r0:
            continue
        rr, cc = np.mgrid[r0 : r1 + 1, c0 : c1 + 1]
        inside = MplPath(xy).contains_points(
            np.column_stack([cc.ravel(), rr.ravel()]),
            radius=1e-9,
        ).reshape(rr.shape)
        mask[r0 : r1 + 1, c0 : c1 + 1] |= inside
    return mask


def _retain_connected_components(
    mask: np.ndarray,
    minimum_pixels: int,
    minimum_fraction_of_largest: float,
    maximum_components: int,
) -> np.ndarray:
    components, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return mask
    sizes = np.bincount(components.ravel())[1:]
    largest = int(sizes.max())
    eligible = np.flatnonzero(
        (sizes >= minimum_pixels)
        & (sizes >= largest * minimum_fraction_of_largest)
    )
    order = eligible[np.argsort(sizes[eligible])[::-1]][:maximum_components]
    return np.isin(components, order + 1)


def refine_projected_mask(
    amplitude: np.ndarray,
    surface_triangles: dict[str, list[np.ndarray]],
    refinement: dict,
) -> tuple[np.ndarray, dict]:
    """Implement paper section 3.5 under a strict projected-model constraint."""
    surfaces = tuple(refinement.get("surfaces", ["roof", "wall"]))
    triangles = [
        np.asarray(triangle, dtype=np.float64)
        for surface in surfaces
        for triangle in surface_triangles.get(surface, [])
        if len(triangle) == 3 and np.all(np.isfinite(triangle))
    ]
    if not triangles:
        return np.zeros_like(amplitude, dtype=bool), {
            "initial_pixels": 0,
            "refined_pixels": 0,
            "threshold": np.nan,
            "background_mean": np.nan,
            "background_std": np.nan,
            "fallback_used": 0,
        }

    gap = max(0, int(refinement.get("background_gap_px", 1)))
    width = max(1, int(refinement.get("background_buffer_px", 5)))
    closing_iterations = max(
        0, int(refinement.get("closing_iterations", 1))
    )
    padding = gap + width + closing_iterations + 2
    all_xy = np.vstack(triangles)
    c0 = max(0, int(math.floor(float(np.min(all_xy[:, 0])))) - padding)
    c1 = min(
        amplitude.shape[1],
        int(math.ceil(float(np.max(all_xy[:, 0])))) + padding + 1,
    )
    r0 = max(0, int(math.floor(float(np.min(all_xy[:, 1])))) - padding)
    r1 = min(
        amplitude.shape[0],
        int(math.ceil(float(np.max(all_xy[:, 1])))) + padding + 1,
    )
    working_amplitude = amplitude[r0:r1, c0:c1]
    offset = np.asarray([c0, r0], dtype=np.float64)
    shifted_by_surface = {
        surface: [
            np.asarray(triangle, dtype=np.float64) - offset
            for triangle in surface_triangles.get(surface, [])
        ]
        for surface in surfaces
    }
    initial_by_surface = {
        surface: rasterize_projected_triangles(
            shifted_by_surface[surface],
            working_amplitude.shape,
        )
        for surface in surfaces
    }
    initial = np.logical_or.reduce(list(initial_by_surface.values()))
    if not np.any(initial):
        return np.zeros_like(amplitude, dtype=bool), {
            "initial_pixels": 0,
            "refined_pixels": 0,
            "threshold": np.nan,
            "background_mean": np.nan,
            "background_std": np.nan,
            "fallback_used": 0,
        }

    inner = binary_dilation(initial, iterations=gap) if gap else initial
    outer = binary_dilation(initial, iterations=gap + width)
    background = outer & ~inner
    background_values = working_amplitude[background]
    background_values = background_values[np.isfinite(background_values)]
    if not background_values.size:
        background_values = working_amplitude[np.isfinite(working_amplitude)]
    background_mean = float(np.mean(background_values))
    background_std = float(np.std(background_values))
    threshold = background_mean + float(
        refinement.get("threshold_sigma", 0.35)
    ) * background_std

    refined = np.zeros_like(initial)
    minimum_pixels = max(
        1, int(refinement.get("minimum_component_pixels", 2))
    )
    minimum_fraction = float(
        refinement.get("minimum_component_fraction_of_largest", 0.08)
    )
    maximum_components = max(
        1, int(refinement.get("maximum_components_per_surface", 3))
    )
    for surface_mask in initial_by_surface.values():
        selected = surface_mask & (working_amplitude > threshold)
        if closing_iterations:
            selected = binary_closing(
                selected,
                structure=np.ones((3, 3), dtype=bool),
                iterations=closing_iterations,
            )
            selected &= surface_mask
        refined |= _retain_connected_components(
            selected,
            minimum_pixels,
            minimum_fraction,
            maximum_components,
        )

    fallback_used = 0
    fallback_pixels = max(
        0, int(refinement.get("minimum_fallback_pixels", 3))
    )
    if not np.any(refined) and fallback_pixels:
        rr, cc = np.nonzero(initial)
        values = working_amplitude[rr, cc]
        take = min(fallback_pixels, len(values))
        if take:
            selected = np.argpartition(values, -take)[-take:]
            refined[rr[selected], cc[selected]] = True
            fallback_used = 1

    # Morphology may reconnect pixels, but never expand beyond the projected model.
    refined &= initial
    full_refined = np.zeros_like(amplitude, dtype=bool)
    full_refined[r0:r1, c0:c1] = refined
    return full_refined, {
        "initial_pixels": int(initial.sum()),
        "refined_pixels": int(refined.sum()),
        "threshold": threshold,
        "background_mean": background_mean,
        "background_std": background_std,
        "fallback_used": fallback_used,
    }


def load_local_shifts(path: Path) -> dict[int, tuple[float, float, bool, float]]:
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    return {
        int(row.fid): (
            float(row.applied_row_shift),
            float(row.applied_col_shift),
            bool(getattr(row, "apply_local_shift_to_roof_only", False)),
            float(getattr(row, "registration_reliability", 0.35)),
        )
        for row in table.itertuples()
    }


def load_height_overrides(path: Path | None) -> dict[int, float]:
    if path is None:
        return {}
    table = pd.read_csv(path)
    if "fid" not in table:
        raise ValueError(f"Height override table has no fid column: {path}")
    height_column = next(
        (name for name in ("height_current_m", "height_est_m", "height_m") if name in table),
        None,
    )
    if height_column is None:
        raise ValueError(
            f"Height override table must contain height_current_m, height_est_m, or height_m: {path}"
        )
    fid = pd.to_numeric(table["fid"], errors="coerce")
    height = pd.to_numeric(table[height_column], errors="coerce")
    valid = np.isfinite(fid)
    return {int(f): float(h) for f, h in zip(fid[valid], height[valid], strict=True)}


def triangle_record(fid: int, clean_id: int, height: float, mesh, index: int) -> dict:
    vertices = mesh.triangles[index]
    xy = mesh.projected_xy[vertices]
    polygon = Polygon(xy)
    return {
        "fid": fid,
        "clean_id": clean_id,
        "height_prior_m": height,
        "surface": str(mesh.surfaces[index]),
        "triangle_index": index,
        "near_range_col": float(np.min(xy[:, 0])),
        "geometry": polygon,
    }


def choose_match(matches: list[dict], surface_rank: dict[str, int]) -> dict:
    # Paper section 3.6: an overlapping pixel belongs to the foreground building,
    # represented by the smallest near-range column. Within one building, prefer
    # visible wall triangles, then roof and bottom, as described in section 3.7.
    best_per_building: list[dict] = []
    for fid in sorted({item["fid"] for item in matches}):
        group = [item for item in matches if item["fid"] == fid]
        best_per_building.append(min(group, key=lambda item: (surface_rank[item["surface"]], item["violation"], item["triangle_index"])))
    return min(best_per_building, key=lambda item: (item["near_range_col"], surface_rank[item["surface"]], item["violation"]))


def run(
    config_path: Path,
    max_buildings: int = 0,
    height_overrides_path: Path | None = None,
    local_registration_path: Path | None = None,
    output_root: Path | None = None,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inputs, outputs = config["inputs"], config["outputs"]
    if output_root is None:
        output_paths = {key: resolve(value) for key, value in outputs.items()}
    else:
        output_root = output_root.resolve()
        output_paths = {
            "tables": output_root / "tables",
            "vectors": output_root / "vectors",
            "triangles": output_root / "triangles",
            "summary": output_root / "mapping_summary.json",
            "height_estimates": output_root / "height_estimation",
        }
    for key, path in output_paths.items():
        (path if key != "summary" else path.parent).mkdir(parents=True, exist_ok=True)

    buildings = gpd.read_file(resolve(inputs["buildings"])).reset_index(drop=True)
    if buildings.crs is None:
        raise ValueError("Building CRS is missing")
    buildings = buildings.to_crs("EPSG:4326")
    buildings["fid"] = np.arange(len(buildings), dtype=np.int64)
    if max_buildings > 0:
        buildings = buildings.iloc[:max_buildings].copy()

    ps = pd.read_csv(resolve(inputs["ps_points"]))
    ps = ps.loc[pd.to_numeric(ps["coherence"], errors="coerce") >= float(config["minimum_ps_coherence"])].copy()
    offset = 1.0 if config["ps_pixel_indexing"] == "one_based" else 0.0
    ps["row0"] = pd.to_numeric(ps["azimuth_pixel"], errors="coerce") - offset
    ps["col0"] = pd.to_numeric(ps["range_pixel"], errors="coerce") - offset
    ps = ps.loc[np.isfinite(ps["row0"]) & np.isfinite(ps["col0"])].copy()

    registration = config["registration"]
    local_path = local_registration_path.resolve() if local_registration_path else resolve(inputs["local_registration"])
    local = load_local_shifts(local_path)
    height_overrides = load_height_overrides(height_overrides_path.resolve() if height_overrides_path else None)
    projector = StrictRadarProjector(resolve(inputs["rslc_par"]))
    surface_rank = {name: rank for rank, name in enumerate(config["surface_priority"])}
    refinement = config.get("mask_refinement", {})
    refinement_enabled = bool(refinement.get("enabled", False))
    amplitude = None
    refined_pixels_by_fid: dict[int, set[int]] = {}
    mask_rows: list[dict] = []
    if refinement_enabled:
        shape = (
            int(projector.par["azimuth_lines"]),
            int(projector.par["range_samples"]),
        )
        amplitude = read_sar_amplitude(
            resolve(inputs["rslc_par"]).with_suffix(""),
            shape,
        )

    mesh_by_fid = {}
    triangle_rows: list[dict] = []
    failures: list[dict] = []
    for building in buildings.itertuples():
        fid = int(building.fid)
        try:
            height = height_overrides.get(fid, float(getattr(building, config["height_field"])))
            if not np.isfinite(height) or height <= 0:
                raise ValueError(f"invalid {config['height_field']}={height}")
            if building.geometry.geom_type != "Polygon" or building.geometry.is_empty:
                raise ValueError("only non-empty Polygon footprints are supported")
            local_row, local_col, roof_only, registration_reliability = local.get(
                fid, (0.0, 0.0, False, 0.35)
            )
            global_row = float(registration["global_row_shift_px"])
            global_col = float(registration["global_col_shift_px"])
            applied_local_row = local_row if registration["use_local_row_shift"] else 0.0
            applied_local_col = local_col if registration["use_local_col_shift"] else 0.0
            row_shift = global_row if roof_only else global_row + applied_local_row
            col_shift = global_col if roof_only else global_col + applied_local_col
            top_row_shift = global_row + applied_local_row
            top_col_shift = global_col + applied_local_col
            mesh = projector.build_mesh(
                np.asarray(building.geometry.exterior.coords),
                float(config["base_elevation_m"]),
                height,
                row_shift,
                col_shift,
                top_row_shift,
                top_col_shift,
            )
            mesh_by_fid[fid] = (
                mesh,
                height,
                int(building.clean_id),
                row_shift,
                col_shift,
                top_row_shift,
                top_col_shift,
                roof_only,
                registration_reliability,
            )
            triangle_rows.extend(triangle_record(fid, int(building.clean_id), height, mesh, index) for index in range(len(mesh.triangles)))
        except Exception as exc:
            failures.append({"fid": fid, "reason": str(exc)})

    triangles = gpd.GeoDataFrame(triangle_rows, geometry="geometry", crs=None)
    if refinement_enabled and amplitude is not None:
        image_cols = amplitude.shape[1]
        for fid, group in triangles.groupby("fid", sort=True):
            surface_triangles = {
                str(surface): [
                    np.asarray(geometry.exterior.coords)[:3]
                    for geometry in part.geometry
                    if geometry is not None and not geometry.is_empty
                ]
                for surface, part in group.groupby("surface")
            }
            refined_mask, mask_stats = refine_projected_mask(
                amplitude,
                surface_triangles,
                refinement,
            )
            rr, cc = np.nonzero(refined_mask)
            refined_pixels_by_fid[int(fid)] = set(
                (rr.astype(np.int64) * image_cols + cc).tolist()
            )
            mask_rows.append({"fid": int(fid), **mask_stats})
    mask_stats_by_fid = {row["fid"]: row for row in mask_rows}

    spatial_index = triangles.sindex
    matches_by_ps: dict[int, list[dict]] = {}
    raw_matches_by_ps: dict[int, list[dict]] = {}
    for ps_index, row in ps.iterrows():
        point = Point(float(row.col0), float(row.row0))
        candidate_indexes = list(spatial_index.query(point, predicate="intersects"))
        for triangle_position in candidate_indexes:
            record = triangles.iloc[int(triangle_position)]
            (
                mesh,
                height,
                clean_id,
                row_shift,
                col_shift,
                top_row_shift,
                top_col_shift,
                roof_only,
                registration_reliability,
            ) = mesh_by_fid[int(record.fid)]
            vertices = mesh.triangles[int(record.triangle_index)]
            weights, violation = barycentric_weights(
                np.asarray([row.col0, row.row0]),
                mesh.projected_xy[vertices],
            )
            if not np.all(np.isfinite(weights)) or violation > 1e-7:
                continue
            match = {
                "fid": int(record.fid),
                "clean_id": clean_id,
                "height_prior_m": height,
                "surface": str(record.surface),
                "triangle_index": int(record.triangle_index),
                "near_range_col": float(record.near_range_col),
                "weights": weights,
                "violation": violation,
                "row_shift": row_shift,
                "col_shift": col_shift,
                "top_row_shift": top_row_shift,
                "top_col_shift": top_col_shift,
                "roof_only_registration": roof_only,
                "registration_reliability": registration_reliability,
            }
            raw_matches_by_ps.setdefault(int(ps_index), []).append(match)
            keep = True
            if refinement_enabled:
                pixel_key = (
                    int(round(float(row.row0))) * amplitude.shape[1]
                    + int(round(float(row.col0)))
                )
                keep = pixel_key in refined_pixels_by_fid.get(
                    int(record.fid), set()
                )
            if keep:
                matches_by_ps.setdefault(int(ps_index), []).append(match)

    mapped: list[dict] = []
    for ps_index, matches in matches_by_ps.items():
        selected = choose_match(matches, surface_rank)
        mesh, height, clean_id, *_ = mesh_by_fid[selected["fid"]]
        vertex_indexes = mesh.triangles[selected["triangle_index"]]
        weights = selected["weights"]
        surface_ecef = weights @ mesh.vertices_ecef[vertex_indexes]
        surface_llh = ecef_to_llh(surface_ecef)[0]
        vertical_fraction = float(weights @ mesh.vertex_is_top[vertex_indexes])
        roof_llh = np.asarray([surface_llh[0], surface_llh[1], float(config["base_elevation_m"]) + height])
        source = ps.loc[ps_index]
        mapped.append(
            {
                "ps_id": int(source.ps_id),
                "fid": selected["fid"],
                "clean_id": selected["clean_id"],
                "surface": selected["surface"],
                "triangle_index": selected["triangle_index"],
                "coherence": float(source.coherence),
                "azimuth_pixel_1based": float(source.azimuth_pixel),
                "range_pixel_1based": float(source.range_pixel),
                "row0": float(source.row0),
                "col0": float(source.col0),
                "bary_alpha": float(weights[0]),
                "bary_beta": float(weights[1]),
                "bary_gamma": float(weights[2]),
                "vertical_fraction_of_prior": vertical_fraction,
                "base_elevation_m": float(config["base_elevation_m"]),
                "height_prior_m": height,
                "surface_lon": float(surface_llh[0]),
                "surface_lat": float(surface_llh[1]),
                "surface_elevation_m": float(surface_llh[2]),
                "corresponding_roof_lon": float(roof_llh[0]),
                "corresponding_roof_lat": float(roof_llh[1]),
                "corresponding_roof_elevation_m": float(roof_llh[2]),
                "overlapping_building_candidates": len({item["fid"] for item in matches}),
                "overlapping_triangle_candidates": len(matches),
                "global_plus_local_row_shift_px": selected["row_shift"],
                "global_plus_local_col_shift_px": selected["col_shift"],
                "roof_global_plus_local_row_shift_px": selected["top_row_shift"],
                "roof_global_plus_local_col_shift_px": selected["top_col_shift"],
                "local_shift_applied_to_roof_only": int(selected["roof_only_registration"]),
                "registration_reliability": float(
                    selected["registration_reliability"]
                ),
                "sar_log_amplitude_normalized": (
                    float(
                        amplitude[
                            int(round(float(source.row0))),
                            int(round(float(source.col0))),
                        ]
                    )
                    if amplitude is not None
                    else np.nan
                ),
                "mask_refinement_threshold": (
                    float(mask_stats_by_fid[selected["fid"]]["threshold"])
                    if refinement_enabled
                    else np.nan
                ),
                "mask_refinement_pass": int(refinement_enabled),
            }
        )

    mapped_table = pd.DataFrame(mapped)
    mapped_csv = output_paths["tables"] / "ps_building_surface_coordinates.csv"
    mapped_table.to_csv(mapped_csv, index=False)
    failures_csv = output_paths["tables"] / "projection_failures.csv"
    pd.DataFrame(failures, columns=["fid", "reason"]).to_csv(failures_csv, index=False)
    mask_summary = pd.DataFrame(mask_rows)
    if refinement_enabled:
        candidate_before: dict[int, int] = {}
        candidate_after: dict[int, int] = {}
        audit_rows = []
        for ps_index, raw_matches in raw_matches_by_ps.items():
            kept_matches = matches_by_ps.get(ps_index, [])
            source = ps.loc[ps_index]
            raw_fids = sorted({int(item["fid"]) for item in raw_matches})
            kept_fids = sorted({int(item["fid"]) for item in kept_matches})
            for fid in raw_fids:
                candidate_before[fid] = candidate_before.get(fid, 0) + 1
            for fid in kept_fids:
                candidate_after[fid] = candidate_after.get(fid, 0) + 1
            audit_rows.append(
                {
                    "ps_id": int(source.ps_id),
                    "row0": float(source.row0),
                    "col0": float(source.col0),
                    "sar_log_amplitude_normalized": float(
                        amplitude[
                            int(round(float(source.row0))),
                            int(round(float(source.col0))),
                        ]
                    ),
                    "candidate_buildings_before": ";".join(map(str, raw_fids)),
                    "candidate_buildings_after": ";".join(map(str, kept_fids)),
                    "mask_refinement_pass": int(bool(kept_fids)),
                }
            )
        mask_summary["candidate_ps_before"] = mask_summary.fid.map(
            candidate_before
        ).fillna(0).astype(int)
        mask_summary["candidate_ps_after"] = mask_summary.fid.map(
            candidate_after
        ).fillna(0).astype(int)
        mask_summary["candidate_ps_retained_fraction"] = (
            mask_summary.candidate_ps_after
            / mask_summary.candidate_ps_before.replace(0, np.nan)
        )
        pd.DataFrame(audit_rows).to_csv(
            output_paths["tables"] / "ps_mask_refinement_audit.csv",
            index=False,
        )
    mask_summary.to_csv(
        output_paths["tables"] / "mask_refinement_summary.csv",
        index=False,
    )
    triangles_path = output_paths["triangles"] / "building_surface_triangles_radar.gpkg"
    triangle_export = triangles.rename(columns={"fid": "building_fid"})
    triangle_export.to_file(triangles_path, layer="sar_triangles", driver="GPKG")

    if mapped:
        points = gpd.GeoDataFrame(
            mapped_table.rename(columns={"fid": "building_fid"}).copy(),
            geometry=gpd.points_from_xy(mapped_table.surface_lon, mapped_table.surface_lat),
            crs="EPSG:4326",
        )
    else:
        points = gpd.GeoDataFrame(mapped_table.copy(), geometry=[], crs="EPSG:4326")
    points_path = output_paths["vectors"] / "ps_points_on_building_surfaces.gpkg"
    points.to_file(points_path, layer="ps_surface_points", driver="GPKG")

    per_building = (
        mapped_table.groupby(["fid", "clean_id", "height_prior_m"], as_index=False)
        .agg(
            ps_count=("ps_id", "size"),
            coherence_mean=("coherence", "mean"),
            wall_ps_count=("surface", lambda values: int(np.sum(values == "wall"))),
            roof_ps_count=("surface", lambda values: int(np.sum(values == "roof"))),
            vertical_fraction_median=("vertical_fraction_of_prior", "median"),
        )
        if mapped
        else pd.DataFrame(columns=["fid", "clean_id", "height_prior_m", "ps_count"])
    )
    building_csv = output_paths["tables"] / "building_ps_support_summary.csv"
    per_building.to_csv(building_csv, index=False)

    summary = {
        "method": (
            "literature_strict_triangle_projection_local_amplitude_mask_"
            "refinement_and_barycentric_ps_surface_mapping"
            if refinement_enabled
            else "literature_strict_triangle_projection_and_barycentric_ps_surface_mapping"
        ),
        "paper_sections": (
            ["2.4", "3.3", "3.4", "3.5", "3.6", "3.7"]
            if refinement_enabled
            else ["2.4", "3.3", "3.4", "3.6", "3.7"]
        ),
        "base_elevation_m": float(config["base_elevation_m"]),
        "roof_elevation_definition": (
            f"{config['base_elevation_m']} m + per-building iterative height"
            if height_overrides
            else f"{config['base_elevation_m']} m + buildings.{config['height_field']}"
        ),
        "height_override_buildings": int(len(height_overrides)),
        "local_registration_path": str(local_path),
        "input_buildings": int(len(buildings)),
        "projected_buildings": int(len(mesh_by_fid)),
        "projection_failures": int(len(failures)),
        "input_ps": int(len(pd.read_csv(resolve(inputs["ps_points"])))),
        "coherence_filtered_ps": int(len(ps)),
        "mapped_ps": int(len(mapped_table)),
        "candidate_ps_before_mask_refinement": int(len(raw_matches_by_ps)),
        "candidate_ps_after_mask_refinement": int(len(matches_by_ps)),
        "mask_refinement": {
            "enabled": refinement_enabled,
            "configuration": refinement if refinement_enabled else None,
            "initial_mask_pixels": (
                int(mask_summary.initial_pixels.sum())
                if refinement_enabled and len(mask_summary)
                else None
            ),
            "refined_mask_pixels": (
                int(mask_summary.refined_pixels.sum())
                if refinement_enabled and len(mask_summary)
                else None
            ),
            "fallback_buildings": (
                int(mask_summary.fallback_used.sum())
                if refinement_enabled and len(mask_summary)
                else None
            ),
        },
        "buildings_with_ps": int(mapped_table.fid.nunique()) if mapped else 0,
        "surface_counts": mapped_table.surface.value_counts().to_dict() if mapped else {},
        "overlap_resolution": (
            "minimum near-range column building; within-building surface priority "
            + "-".join(config["surface_priority"])
        ),
        "independence_warning": "Mapped elevations depend on the input SHP height used to build the reference mesh. They are geometric PS features for a later height inversion, not independent height truth.",
        "outputs": {
            "ps_surface_csv": str(mapped_csv),
            "building_support_csv": str(building_csv),
            "ps_surface_gpkg": str(points_path),
            "sar_triangle_gpkg": str(triangles_path),
            "failures_csv": str(failures_csv),
            "mask_refinement_summary_csv": str(
                output_paths["tables"] / "mask_refinement_summary.csv"
            ),
            "ps_mask_refinement_audit_csv": (
                str(output_paths["tables"] / "ps_mask_refinement_audit.csv")
                if refinement_enabled
                else None
            ),
        },
    }
    output_paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 4 m + SHP-height building meshes and map PS pixels to 3-D surfaces by barycentric coordinates.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-buildings", type=int, default=0)
    parser.add_argument("--height-overrides", type=Path)
    parser.add_argument("--local-registration", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.max_buildings,
        args.height_overrides,
        args.local_registration,
        args.output_root,
    )


if __name__ == "__main__":
    main()
