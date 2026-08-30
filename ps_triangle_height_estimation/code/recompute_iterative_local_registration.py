from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    sobel,
)

from geometry import StrictRadarProjector
from map_ps_to_building_surfaces import load_height_overrides, resolve


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"


def read_sar_features(
    path: Path,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return robust log-amplitude, edge strength, and normalized image gradients."""
    raw = np.fromfile(path, dtype=">i2")
    expected = shape[0] * shape[1] * 2
    if raw.size != expected:
        raise ValueError(f"Unexpected RSLC size: {raw.size}, expected {expected}")
    iq = raw.reshape(*shape, 2).astype(np.float32)
    amplitude = np.log1p(np.hypot(iq[:, :, 0], iq[:, :, 1]))
    positive = amplitude[amplitude > 0]
    low, high = np.percentile(positive, [2, 98]) if positive.size else (0.0, 1.0)
    amplitude = np.clip((amplitude - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    smoothed = gaussian_filter(amplitude, sigma=0.8)
    gx = sobel(smoothed, axis=1, mode="reflect") / 8.0
    gy = sobel(smoothed, axis=0, mode="reflect") / 8.0
    edge = np.hypot(gx, gy)
    positive = edge[edge > 0]
    high = float(np.percentile(positive, 98)) if positive.size else 1.0
    scale = max(high, 1e-6)
    return (
        amplitude.astype(np.float32),
        np.clip(edge / scale, 0.0, 1.0).astype(np.float32),
        (gx / scale).astype(np.float32),
        (gy / scale).astype(np.float32),
    )


def rasterize_triangles(triangles: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    mask = np.zeros(shape, dtype=bool)
    for xy in triangles:
        c0 = max(0, int(math.floor(np.min(xy[:, 0]))) - 1)
        c1 = min(cols - 1, int(math.ceil(np.max(xy[:, 0]))) + 1)
        r0 = max(0, int(math.floor(np.min(xy[:, 1]))) - 1)
        r1 = min(rows - 1, int(math.ceil(np.max(xy[:, 1]))) + 1)
        if c1 < c0 or r1 < r0:
            continue
        yy, xx = np.mgrid[r0 : r1 + 1, c0 : c1 + 1]
        inside = MplPath(xy).contains_points(
            np.column_stack([xx.ravel(), yy.ravel()]), radius=1e-9
        ).reshape(yy.shape)
        mask[r0 : r1 + 1, c0 : c1 + 1] |= inside
    return mask


def shifted_values(
    image: np.ndarray,
    rr: np.ndarray,
    cc: np.ndarray,
    dr: int,
    dc: int,
) -> np.ndarray:
    rows, cols = rr + dr, cc + dc
    keep = (rows >= 0) & (cols >= 0) & (rows < image.shape[0]) & (cols < image.shape[1])
    return image[rows[keep], cols[keep]] if np.any(keep) else np.zeros(0, dtype=np.float32)


def robust_mean(values: np.ndarray, trim_fraction: float = 0.10) -> float:
    """Trim speckle extremes before averaging a feature response."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan")
    if values.size < 10:
        return float(np.mean(values))
    values = np.sort(values)
    trim = min(int(values.size * trim_fraction), (values.size - 1) // 2)
    return float(np.mean(values[trim : values.size - trim])) if trim else float(np.mean(values))


def mask_feature_samples(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Build interior, control-ring, boundary, and outward-normal samples."""
    rr, cc = np.nonzero(mask)
    ring = binary_dilation(mask, iterations=5) & ~binary_dilation(mask, iterations=2)
    ring_rr, ring_cc = np.nonzero(ring)
    boundary = binary_dilation(mask, iterations=1) & ~binary_erosion(mask, iterations=1)
    boundary_rr, boundary_cc = np.nonzero(boundary)
    signed_distance = distance_transform_edt(mask) - distance_transform_edt(~mask)
    normal_y, normal_x = np.gradient(signed_distance.astype(np.float32))
    normal_norm = np.hypot(normal_x, normal_y)
    boundary_nx = normal_x[boundary_rr, boundary_cc] / np.maximum(
        normal_norm[boundary_rr, boundary_cc], 1e-6
    )
    boundary_ny = normal_y[boundary_rr, boundary_cc] / np.maximum(
        normal_norm[boundary_rr, boundary_cc], 1e-6
    )
    return {
        "rr": rr,
        "cc": cc,
        "ring_rr": ring_rr,
        "ring_cc": ring_cc,
        "boundary_rr": boundary_rr,
        "boundary_cc": boundary_cc,
        "boundary_nx": boundary_nx,
        "boundary_ny": boundary_ny,
    }


def score_shift(
    amplitude: np.ndarray,
    edges: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    samples: dict[str, np.ndarray],
    dr: int,
    dc: int,
    max_shift: int,
    weights: dict[str, float],
) -> dict:
    rr, cc = samples["rr"], samples["cc"]
    ring_rr, ring_cc = samples["ring_rr"], samples["ring_cc"]
    inside_amp = shifted_values(amplitude, rr, cc, dr, dc)
    ring_amp = shifted_values(amplitude, ring_rr, ring_cc, dr, dc)
    boundary_edge = shifted_values(
        edges, samples["boundary_rr"], samples["boundary_cc"], dr, dc
    )
    ring_edge = shifted_values(edges, ring_rr, ring_cc, dr, dc)
    if inside_amp.size < 4:
        return {
            "score": -1e9,
            "amplitude_contrast": np.nan,
            "boundary_edge_contrast": np.nan,
            "normal_edge_response": np.nan,
            "bright_fraction_contrast": np.nan,
        }
    ia = robust_mean(inside_amp)
    ra = robust_mean(ring_amp) if ring_amp.size else robust_mean(amplitude.ravel())
    be = robust_mean(boundary_edge)
    re = robust_mean(ring_edge) if ring_edge.size else robust_mean(edges.ravel())
    rows = samples["boundary_rr"] + dr
    cols = samples["boundary_cc"] + dc
    keep = (
        (rows >= 0)
        & (cols >= 0)
        & (rows < amplitude.shape[0])
        & (cols < amplitude.shape[1])
    )
    if np.any(keep):
        gx = gradient_x[rows[keep], cols[keep]]
        gy = gradient_y[rows[keep], cols[keep]]
        directional = np.abs(
            gx * samples["boundary_nx"][keep] + gy * samples["boundary_ny"][keep]
        )
        normal_edge = robust_mean(directional)
    else:
        normal_edge = 0.0
    amplitude_contrast = ia - ra
    boundary_edge_contrast = be - re
    normal_edge_response = normal_edge - 0.5 * re
    bright_fraction_contrast = float(np.mean(inside_amp >= 0.72)) - (
        float(np.mean(ring_amp >= 0.72)) if ring_amp.size else 0.0
    )
    penalty = weights["shift_penalty"] * (dr * dr + dc * dc) / max(max_shift, 1)
    score = (
        weights["amplitude_contrast"] * amplitude_contrast
        + weights["boundary_edge_contrast"] * boundary_edge_contrast
        + weights["normal_edge_response"] * normal_edge_response
        + weights["bright_fraction_contrast"] * bright_fraction_contrast
        - penalty
    )
    return {
        "score": float(score),
        "amplitude_contrast": float(amplitude_contrast),
        "boundary_edge_contrast": float(boundary_edge_contrast),
        "normal_edge_response": float(normal_edge_response),
        "bright_fraction_contrast": float(bright_fraction_contrast),
    }


def optimize_mask(
    mask: np.ndarray,
    amplitude: np.ndarray,
    edges: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    max_shift: int,
    coarse_step: int,
    min_score_gain: float,
    search_col: bool,
    weights: dict[str, float],
) -> dict:
    samples = mask_feature_samples(mask)
    rr = samples["rr"]
    empty = {
        "roof_pixels": int(rr.size), "base_score": -1e9, "best_score": -1e9,
        "score_gain": 0.0, "candidate_row_shift": 0, "candidate_col_shift": 0,
        "applied_row_shift": 0, "applied_col_shift": 0, "accepted": 0,
    }
    if rr.size < 4:
        return empty
    base = score_shift(
        amplitude, edges, gradient_x, gradient_y, samples, 0, 0, max_shift, weights
    )
    best = {**base, "row_shift": 0, "col_shift": 0}
    col_values = range(-max_shift, max_shift + 1, coarse_step) if search_col else (0,)
    for dr in range(-max_shift, max_shift + 1, coarse_step):
        for dc in col_values:
            candidate = score_shift(
                amplitude, edges, gradient_x, gradient_y, samples, dr, dc, max_shift, weights
            )
            if candidate["score"] > best["score"]:
                best = {**candidate, "row_shift": dr, "col_shift": dc}
    for dr in range(max(-max_shift, best["row_shift"] - coarse_step), min(max_shift, best["row_shift"] + coarse_step) + 1):
        fine_cols = (
            range(max(-max_shift, best["col_shift"] - coarse_step), min(max_shift, best["col_shift"] + coarse_step) + 1)
            if search_col else (0,)
        )
        for dc in fine_cols:
            candidate = score_shift(
                amplitude, edges, gradient_x, gradient_y, samples, dr, dc, max_shift, weights
            )
            if candidate["score"] > best["score"]:
                best = {**candidate, "row_shift": dr, "col_shift": dc}
    gain = float(best["score"] - base["score"])
    feature_improvement_count = int(
        (best["amplitude_contrast"] > base["amplitude_contrast"])
        + (best["boundary_edge_contrast"] > base["boundary_edge_contrast"])
        + (best["normal_edge_response"] > base["normal_edge_response"])
    )
    boundary_limited = int(
        abs(int(best["row_shift"])) == max_shift
        or (search_col and abs(int(best["col_shift"])) == max_shift)
    )
    accepted = int(
        gain >= min_score_gain
        and (best["row_shift"] != 0 or best["col_shift"] != 0)
        and feature_improvement_count >= int(weights["minimum_improved_feature_count"])
        and not (bool(weights["reject_search_boundary"]) and boundary_limited)
    )
    return {
        "roof_pixels": int(rr.size),
        "base_score": float(base["score"]),
        "best_score": float(best["score"]),
        "score_gain": gain,
        "candidate_row_shift": int(best["row_shift"]),
        "candidate_col_shift": int(best["col_shift"]),
        "applied_row_shift": int(best["row_shift"]) if accepted else 0,
        "applied_col_shift": int(best["col_shift"]) if accepted else 0,
        "accepted": accepted,
        "feature_improvement_count": feature_improvement_count,
        "boundary_limited": boundary_limited,
        "base_amplitude_contrast": float(base["amplitude_contrast"]),
        "best_amplitude_contrast": float(best["amplitude_contrast"]),
        "base_boundary_edge_contrast": float(base["boundary_edge_contrast"]),
        "best_boundary_edge_contrast": float(best["boundary_edge_contrast"]),
        "base_normal_edge_response": float(base["normal_edge_response"]),
        "best_normal_edge_response": float(best["normal_edge_response"]),
        "base_bright_fraction_contrast": float(base["bright_fraction_contrast"]),
        "best_bright_fraction_contrast": float(best["bright_fraction_contrast"]),
    }


def run(
    config_path: Path,
    height_overrides_path: Path,
    output_csv: Path,
    max_buildings: int = 0,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    iteration = config["iterative_adjustment"]
    buildings = gpd.read_file(resolve(config["inputs"]["buildings"])).reset_index(drop=True).to_crs("EPSG:4326")
    buildings["fid"] = np.arange(len(buildings), dtype=np.int64)
    if max_buildings > 0:
        buildings = buildings.iloc[:max_buildings].copy()
    heights = load_height_overrides(height_overrides_path)
    projector = StrictRadarProjector(resolve(config["inputs"]["rslc_par"]))
    rows = int(projector.par["azimuth_lines"])
    cols = int(projector.par["range_samples"])
    rslc_path = resolve(config["inputs"]["rslc_par"]).with_suffix("")
    amplitude, edges, gradient_x, gradient_y = read_sar_features(rslc_path, (rows, cols))
    feature_cfg = iteration["registration_feature_v2"]
    weights = {
        "amplitude_contrast": float(feature_cfg["amplitude_contrast_weight"]),
        "boundary_edge_contrast": float(feature_cfg["boundary_edge_contrast_weight"]),
        "normal_edge_response": float(feature_cfg["normal_edge_response_weight"]),
        "bright_fraction_contrast": float(feature_cfg["bright_fraction_contrast_weight"]),
        "shift_penalty": float(feature_cfg["shift_penalty"]),
        "minimum_improved_feature_count": int(feature_cfg["minimum_improved_feature_count"]),
        "reject_search_boundary": bool(feature_cfg["reject_search_boundary"]),
    }
    global_row = float(config["registration"]["global_row_shift_px"])
    global_col = float(config["registration"]["global_col_shift_px"])
    records: list[dict] = []
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
            mask = rasterize_triangles(roof_triangles, amplitude.shape)
            result = optimize_mask(
                mask,
                amplitude,
                edges,
                gradient_x,
                gradient_y,
                int(iteration["local_registration_max_shift_px"]),
                int(iteration["local_registration_coarse_step_px"]),
                float(iteration["local_registration_min_score_gain"]),
                bool(config["registration"]["use_local_col_shift"]),
                weights,
            )
            records.append(
                {
                    "fid": fid,
                    "height_projection_m": height,
                    "registration_feature_mode": "log_amplitude_boundary_orientation_v2",
                    **result,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "fid": fid, "height_projection_m": heights.get(fid, np.nan),
                    "applied_row_shift": 0, "applied_col_shift": 0, "accepted": 0,
                    "failure_reason": str(exc),
                }
            )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute roof-mask local registration for iterative building heights.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--height-overrides", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--max-buildings", type=int, default=0)
    args = parser.parse_args()
    run(args.config.resolve(), args.height_overrides.resolve(), args.output_csv.resolve(), args.max_buildings)


if __name__ == "__main__":
    main()
