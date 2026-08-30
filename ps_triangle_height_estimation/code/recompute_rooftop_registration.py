from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    gaussian_filter,
    maximum_filter,
)

from geometry import StrictRadarProjector
from map_ps_to_building_surfaces import load_height_overrides, resolve
from recompute_iterative_local_registration import (
    rasterize_triangles,
    read_sar_features,
    robust_mean,
    shifted_values,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"
FEATURE_NAMES = (
    "top_region_contrast",
    "top_support_contrast",
    "roof_boundary_support",
    "normal_edge_response",
    "top_texture_contrast",
)


def robust_unit_scale(values: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float32)
    valid = finite[np.isfinite(finite)]
    if not valid.size:
        return np.zeros_like(finite, dtype=np.float32)
    low, high = np.percentile(valid, [lower, upper])
    return np.clip((finite - low) / max(float(high - low), 1e-6), 0.0, 1.0).astype(
        np.float32
    )


def read_rooftop_features(
    path: Path,
    shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Create speckle-robust SAR features aimed at projected roof surfaces."""
    amplitude, edges, gradient_x, gradient_y = read_sar_features(path, shape)
    roof_scale = gaussian_filter(amplitude, sigma=1.15)
    background = gaussian_filter(roof_scale, sigma=4.5)
    residual = roof_scale - background
    local_noise = np.sqrt(
        np.maximum(gaussian_filter(residual * residual, sigma=4.5), 1e-6)
    )
    cfar = robust_unit_scale(residual / (local_noise + 0.04), 3.0, 97.0)
    local_mean = gaussian_filter(amplitude, sigma=1.0)
    local_second = gaussian_filter(amplitude * amplitude, sigma=1.0)
    texture = robust_unit_scale(
        np.sqrt(np.maximum(local_second - local_mean * local_mean, 0.0))
    )
    roof_likelihood = robust_unit_scale(0.78 * cfar + 0.22 * texture)
    edge_proximity = maximum_filter(edges, size=5, mode="reflect")
    return {
        "amplitude": amplitude,
        "edges": edges,
        "gradient_x": gradient_x,
        "gradient_y": gradient_y,
        "roof_likelihood": roof_likelihood,
        "texture": texture,
        "edge_proximity": edge_proximity,
    }


def rooftop_samples(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Use an eroded roof core and a separated control ring to suppress wall layover."""
    core = binary_erosion(mask, iterations=1)
    if int(core.sum()) < max(6, int(mask.sum() * 0.25)):
        core = mask.copy()
    ring = binary_dilation(mask, iterations=5) & ~binary_dilation(mask, iterations=2)
    boundary = mask & ~binary_erosion(mask, iterations=1)
    if not np.any(boundary):
        boundary = binary_dilation(mask, iterations=1) & ~mask
    core_rr, core_cc = np.nonzero(core)
    ring_rr, ring_cc = np.nonzero(ring)
    boundary_rr, boundary_cc = np.nonzero(boundary)

    # The gradient of a lightly smoothed binary roof points towards its interior.
    soft = gaussian_filter(mask.astype(np.float32), sigma=0.9)
    normal_y, normal_x = np.gradient(soft)
    norm = np.hypot(normal_x, normal_y)
    boundary_nx = normal_x[boundary_rr, boundary_cc] / np.maximum(
        norm[boundary_rr, boundary_cc], 1e-6
    )
    boundary_ny = normal_y[boundary_rr, boundary_cc] / np.maximum(
        norm[boundary_rr, boundary_cc], 1e-6
    )
    return {
        "core_rr": core_rr,
        "core_cc": core_cc,
        "ring_rr": ring_rr,
        "ring_cc": ring_cc,
        "boundary_rr": boundary_rr,
        "boundary_cc": boundary_cc,
        "boundary_nx": boundary_nx,
        "boundary_ny": boundary_ny,
    }


def raw_rooftop_metrics(
    features: dict[str, np.ndarray],
    samples: dict[str, np.ndarray],
    dr: int,
    dc: int,
    support_threshold: float,
) -> dict[str, float]:
    core_like = shifted_values(
        features["roof_likelihood"], samples["core_rr"], samples["core_cc"], dr, dc
    )
    ring_like = shifted_values(
        features["roof_likelihood"], samples["ring_rr"], samples["ring_cc"], dr, dc
    )
    if core_like.size < 6 or ring_like.size < 6:
        return {name: float("nan") for name in FEATURE_NAMES}

    core_texture = shifted_values(
        features["texture"], samples["core_rr"], samples["core_cc"], dr, dc
    )
    ring_texture = shifted_values(
        features["texture"], samples["ring_rr"], samples["ring_cc"], dr, dc
    )
    boundary_support = shifted_values(
        features["edge_proximity"],
        samples["boundary_rr"],
        samples["boundary_cc"],
        dr,
        dc,
    )
    ring_edges = shifted_values(
        features["edge_proximity"], samples["ring_rr"], samples["ring_cc"], dr, dc
    )

    rows = samples["boundary_rr"] + dr
    cols = samples["boundary_cc"] + dc
    keep = (
        (rows >= 0)
        & (cols >= 0)
        & (rows < features["roof_likelihood"].shape[0])
        & (cols < features["roof_likelihood"].shape[1])
    )
    if np.any(keep):
        gx = features["gradient_x"][rows[keep], cols[keep]]
        gy = features["gradient_y"][rows[keep], cols[keep]]
        directional = np.abs(
            gx * samples["boundary_nx"][keep] + gy * samples["boundary_ny"][keep]
        )
        normal_response = robust_mean(directional)
    else:
        normal_response = float("nan")

    return {
        "top_region_contrast": robust_mean(core_like) - robust_mean(ring_like),
        "top_support_contrast": float(np.mean(core_like >= support_threshold))
        - float(np.mean(ring_like >= support_threshold)),
        "roof_boundary_support": robust_mean(boundary_support)
        - 0.35 * robust_mean(ring_edges),
        "normal_edge_response": normal_response,
        "top_texture_contrast": robust_mean(core_texture) - robust_mean(ring_texture),
    }


def robust_candidate_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    if not valid.size:
        return np.full_like(values, np.nan)
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    scale = max(1.4826 * mad, float(np.std(valid)) * 0.25, 1e-6)
    return (values - median) / scale


def evaluate_rooftop_grid(
    mask: np.ndarray,
    features: dict[str, np.ndarray],
    max_shift: int,
    weights: dict[str, float],
) -> dict:
    samples = rooftop_samples(mask)
    roof_pixels = int(mask.sum())
    if roof_pixels < int(weights["minimum_roof_pixels"]):
        return {"roof_pixels": roof_pixels, "valid": False}

    search_region = binary_dilation(mask, iterations=max_shift + 5)
    local_values = features["roof_likelihood"][search_region]
    support_threshold = (
        float(np.quantile(local_values, float(weights["support_quantile"])))
        if local_values.size
        else 0.65
    )
    shifts = [(dr, dc) for dr in range(-max_shift, max_shift + 1) for dc in range(-max_shift, max_shift + 1)]
    metric_rows = [
        raw_rooftop_metrics(features, samples, dr, dc, support_threshold)
        for dr, dc in shifts
    ]
    raw = {
        name: np.asarray([row[name] for row in metric_rows], dtype=np.float64)
        for name in FEATURE_NAMES
    }
    score = np.zeros(len(shifts), dtype=np.float64)
    for name in FEATURE_NAMES:
        score += float(weights[f"{name}_weight"]) * robust_candidate_z(raw[name])
    displacement = np.asarray(
        [(dr * dr + dc * dc) / max(max_shift, 1) for dr, dc in shifts],
        dtype=np.float64,
    )
    score -= float(weights["shift_penalty"]) * displacement
    score[~np.isfinite(score)] = -1e9
    return {
        "roof_pixels": roof_pixels,
        "valid": True,
        "support_threshold": support_threshold,
        "shifts": shifts,
        "raw": raw,
        "score": score,
    }


def select_scene_consensus(grids: list[dict], max_shift: int) -> tuple[int, int, np.ndarray]:
    eligible = [grid["score"] for grid in grids if grid.get("valid")]
    if not eligible:
        return 0, 0, np.full((2 * max_shift + 1) ** 2, np.nan)
    scene_score = np.nanmedian(np.vstack(eligible), axis=0)
    shifts = next(grid["shifts"] for grid in grids if grid.get("valid"))
    allowed = np.asarray(
        [abs(dr) < max_shift and abs(dc) < max_shift for dr, dc in shifts],
        dtype=bool,
    )
    scene_score = np.where(allowed, scene_score, -1e9)
    best_index = int(np.argmax(scene_score))
    return (*shifts[best_index], scene_score)


def finalize_grid(
    grid: dict,
    scene_shift: tuple[int, int],
    scene_score: np.ndarray,
    weights: dict[str, float],
) -> dict:
    scene_dr, scene_dc = scene_shift
    if not grid.get("valid"):
        return {
            "roof_pixels": int(grid.get("roof_pixels", 0)),
            "base_score": np.nan,
            "best_score": np.nan,
            "score_gain": np.nan,
            "candidate_row_shift": scene_dr,
            "candidate_col_shift": scene_dc,
            "applied_row_shift": scene_dr,
            "applied_col_shift": scene_dc,
            "accepted": int(scene_dr != 0 or scene_dc != 0),
            "scene_consensus_only": 1,
            "local_refinement_accepted": 0,
            "feature_improvement_count": 0,
            "boundary_limited": 0,
            "peak_margin": np.nan,
        }

    shifts = grid["shifts"]
    index_by_shift = {shift: index for index, shift in enumerate(shifts)}
    base_index = index_by_shift[(0, 0)]
    radius = int(weights["local_refinement_radius_px"])
    local_indexes = [
        index
        for index, (dr, dc) in enumerate(shifts)
        if abs(dr - scene_dr) <= radius and abs(dc - scene_dc) <= radius
    ]
    best_index = max(local_indexes, key=lambda index: grid["score"][index])
    best_shift = shifts[best_index]
    alternatives = [
        grid["score"][index]
        for index in local_indexes
        if max(
            abs(shifts[index][0] - best_shift[0]),
            abs(shifts[index][1] - best_shift[1]),
        )
        >= 2
    ]
    peak_margin = float(
        grid["score"][best_index] - max(alternatives)
        if alternatives
        else grid["score"][best_index] - grid["score"][base_index]
    )
    improvement_count = int(
        sum(
            grid["raw"][name][best_index] > grid["raw"][name][base_index]
            for name in FEATURE_NAMES[:4]
        )
    )
    scene_index = index_by_shift[(scene_dr, scene_dc)]
    refinement_gain = float(grid["score"][best_index] - grid["score"][scene_index])
    local_accepted = int(
        best_shift != scene_shift
        and refinement_gain >= float(weights["minimum_local_score_gain"])
        and (
            grid["score"][best_index] - grid["score"][base_index]
            >= float(weights["minimum_total_score_gain"])
        )
        and peak_margin >= float(weights["minimum_peak_margin"])
        and improvement_count >= int(weights["minimum_improved_feature_count"])
        and grid["raw"]["top_region_contrast"][best_index] > 0.0
    )
    applied_shift = best_shift if local_accepted else scene_shift
    applied_index = index_by_shift[applied_shift]
    result = {
        "roof_pixels": int(grid["roof_pixels"]),
        "base_score": float(grid["score"][base_index]),
        "best_score": float(grid["score"][applied_index]),
        "score_gain": float(grid["score"][applied_index] - grid["score"][base_index]),
        "candidate_row_shift": int(best_shift[0]),
        "candidate_col_shift": int(best_shift[1]),
        "applied_row_shift": int(applied_shift[0]),
        "applied_col_shift": int(applied_shift[1]),
        "accepted": int(applied_shift != (0, 0)),
        "scene_consensus_only": int(not local_accepted),
        "local_refinement_accepted": local_accepted,
        "feature_improvement_count": improvement_count,
        "boundary_limited": 0,
        "peak_margin": peak_margin,
        "scene_score": float(scene_score[scene_index]),
        "support_threshold": float(grid["support_threshold"]),
    }
    for name in FEATURE_NAMES:
        result[f"base_{name}"] = float(grid["raw"][name][base_index])
        result[f"best_{name}"] = float(grid["raw"][name][applied_index])
    return result


def run(
    config_path: Path,
    height_overrides_path: Path,
    output_csv: Path,
    max_buildings: int = 0,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = config["iterative_adjustment"]["registration_feature_v3"]
    weights = {
        **{f"{name}_weight": float(cfg[f"{name}_weight"]) for name in FEATURE_NAMES},
        "shift_penalty": float(cfg["shift_penalty"]),
        "support_quantile": float(cfg["support_quantile"]),
        "minimum_roof_pixels": int(cfg["minimum_roof_pixels"]),
        "local_refinement_radius_px": int(cfg["local_refinement_radius_px"]),
        "minimum_local_score_gain": float(cfg["minimum_local_score_gain"]),
        "minimum_total_score_gain": float(cfg["minimum_total_score_gain"]),
        "minimum_peak_margin": float(cfg["minimum_peak_margin"]),
        "minimum_improved_feature_count": int(cfg["minimum_improved_feature_count"]),
    }
    buildings = (
        gpd.read_file(resolve(config["inputs"]["buildings"]))
        .reset_index(drop=True)
        .to_crs("EPSG:4326")
    )
    buildings["fid"] = np.arange(len(buildings), dtype=np.int64)
    if max_buildings > 0:
        buildings = buildings.iloc[:max_buildings].copy()
    heights = load_height_overrides(height_overrides_path)
    projector = StrictRadarProjector(resolve(config["inputs"]["rslc_par"]))
    rows = int(projector.par["azimuth_lines"])
    cols = int(projector.par["range_samples"])
    rslc_path = resolve(config["inputs"]["rslc_par"]).with_suffix("")
    features = read_rooftop_features(rslc_path, (rows, cols))
    max_shift = int(cfg["scene_search_max_shift_px"])
    global_row = float(config["registration"]["global_row_shift_px"])
    global_col = float(config["registration"]["global_col_shift_px"])

    grids: list[dict] = []
    metadata: list[dict] = []
    for building in buildings.itertuples():
        fid = int(building.fid)
        try:
            height = heights.get(fid, float(getattr(building, config["height_field"])))
            mesh = projector.build_mesh(
                np.asarray(building.geometry.exterior.coords),
                float(config["base_elevation_m"]),
                height,
                global_row,
                global_col,
            )
            roof_triangles = [
                mesh.projected_xy[mesh.triangles[index]]
                for index, surface in enumerate(mesh.surfaces)
                if surface == "roof"
            ]
            mask = rasterize_triangles(roof_triangles, features["amplitude"].shape)
            grids.append(evaluate_rooftop_grid(mask, features, max_shift, weights))
            metadata.append({"fid": fid, "height_projection_m": height})
        except Exception as exc:
            grids.append({"roof_pixels": 0, "valid": False})
            metadata.append(
                {
                    "fid": fid,
                    "height_projection_m": heights.get(fid, np.nan),
                    "failure_reason": str(exc),
                }
            )

    scene_dr, scene_dc, scene_score = select_scene_consensus(grids, max_shift)
    records = []
    for meta, grid in zip(metadata, grids, strict=True):
        records.append(
            {
                **meta,
                "registration_feature_mode": "scene_consensus_rooftop_shape_v3",
                "scene_consensus_row_shift": scene_dr,
                "scene_consensus_col_shift": scene_dc,
                **finalize_grid(grid, (scene_dr, scene_dc), scene_score, weights),
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register projected vector roofs to SAR rooftop features in two dimensions."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--height-overrides", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--max-buildings", type=int, default=0)
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.height_overrides.resolve(),
        args.output_csv.resolve(),
        args.max_buildings,
    )


if __name__ == "__main__":
    main()
