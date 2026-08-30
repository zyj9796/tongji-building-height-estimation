from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-roof-only-height")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter1d, sobel
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union


WORK_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = WORK_ROOT / "config.json"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans", "Arial", "sans-serif"],
        "svg.fonttype": "none",
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def resolve(text: str) -> Path:
    return (WORK_ROOT / text).resolve()


def load_shared_projection(config: dict):
    shared = resolve(config["inputs"]["shared_projection_code"])
    sys.path.insert(0, str(shared))
    from strict_candidate_geometry import StrictRadarProjector, clean_ring_lonlat

    return StrictRadarProjector, clean_ring_lonlat


def parse_shape(path: Path) -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        if key.strip() in {"range_samples", "azimuth_lines"}:
            values[key.strip()] = int(float(rest.split()[0]))
    return values["azimuth_lines"], values["range_samples"]


def read_rslc_amplitude(path: Path, shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    raw = np.fromfile(str(path), dtype=">i2")
    if raw.size != rows * cols * 2:
        raise ValueError(f"Unexpected RSLC size: {path}")
    parts = raw.reshape(rows, cols, 2).astype(np.float32)
    amplitude = np.hypot(parts[:, :, 0], parts[:, :, 1])
    positive = amplitude[amplitude > 0]
    lo, hi = np.percentile(positive, (2.0, 98.0)) if positive.size else (0.0, 1.0)
    return np.clip((amplitude - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)


def load_evidence(config: dict) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    root = resolve(config["inputs"]["rslc_dir"])
    evidence: dict[str, dict[str, np.ndarray]] = {}
    shape = None
    for date in config["scenes"]:
        current_shape = parse_shape(root / f"{date}.rslc.par")
        if shape is None:
            shape = current_shape
        elif current_shape != shape:
            raise ValueError("Coregistered RSLC dimensions differ")
        amplitude = read_rslc_amplitude(root / f"{date}.rslc", current_shape)
        edge = np.hypot(sobel(amplitude, axis=1, mode="nearest"), sobel(amplitude, axis=0, mode="nearest"))
        edge = np.clip(edge / max(float(np.percentile(edge, 95.0)), 1e-6), 0.0, 1.0)
        evidence[date] = {"amplitude": amplitude.astype(np.float32), "edge": edge.astype(np.float32)}
    median_amplitude = np.median(np.stack([item["amplitude"] for item in evidence.values()]), axis=0)
    median_edge = np.median(np.stack([item["edge"] for item in evidence.values()]), axis=0)
    return evidence, median_amplitude, median_edge


def load_ps(path: Path) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    return {
        "row": frame["azimuth_pixel"].to_numpy(dtype=np.float64) - 1.0,
        "col": frame["range_pixel"].to_numpy(dtype=np.float64) - 1.0,
        "coh": frame["coherence"].to_numpy(dtype=np.float64),
    }


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, float(np.std(values)) * 0.25, 1e-6)
    return np.clip((values - median) / scale, -4.0, 4.0)


def local_mask(polygon, c0: int, r0: int, width: int, height: int) -> np.ndarray:
    if polygon.is_empty:
        return np.zeros((height, width), dtype=bool)
    return rasterize(
        [(mapping(polygon), 1)],
        out_shape=(height, width),
        transform=Affine.translation(c0, r0),
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)


def project_roofs(projector, ring: np.ndarray, absolute_heights: np.ndarray, row_shift: float, col_shift: float):
    rows, cols = projector.project_height_grid(ring, absolute_heights)
    roofs = []
    for index in range(len(absolute_heights)):
        xy = np.column_stack([cols[index] + col_shift, rows[index] + row_shift])
        roofs.append(Polygon(xy).buffer(0))
    return roofs


def score_building(
    fid: int,
    ring: np.ndarray,
    prior: float,
    projector,
    evidence: dict[str, dict[str, np.ndarray]],
    median_amplitude: np.ndarray,
    median_edge: np.ndarray,
    ps: dict[str, np.ndarray],
    config: dict,
) -> tuple[dict, list[dict]]:
    search = config["height_search"]
    base = float(config["base_elevation_m"])
    row_shift = float(config["registration"]["global_row_shift_px"])
    col_shift = float(config["registration"]["global_col_shift_px"])
    if search.get("strategy", "legacy_asymmetric") == "symmetric_prior_window":
        half_window = max(
            float(search["symmetric_half_window_minimum_m"]),
            float(search["symmetric_half_window_prior_factor"]) * prior,
        )
        low = max(float(search["minimum_m"]), math.floor(prior - half_window))
        high = min(float(search["maximum_m"]), math.ceil(prior + half_window))
    else:
        low = max(float(search["minimum_m"]), math.floor(float(search["prior_lower_factor"]) * prior))
        high = min(
            float(search["maximum_m"]),
            math.ceil(max(float(search["prior_upper_factor"]) * prior + float(search["prior_upper_margin_m"]), prior + float(search["prior_absolute_margin_m"]))),
        )
    endpoints = project_roofs(projector, ring, base + np.asarray([low, high]), row_shift, col_shift)
    envelope = unary_union(endpoints).convex_hull.buffer(7.0)
    rows, cols = median_amplitude.shape
    c0 = max(0, int(math.floor(envelope.bounds[0])))
    c1 = min(cols - 1, int(math.ceil(envelope.bounds[2])))
    r0 = max(0, int(math.floor(envelope.bounds[1])))
    r1 = min(rows - 1, int(math.ceil(envelope.bounds[3])))
    width, height = c1 - c0 + 1, r1 - r0 + 1
    if width <= 1 or height <= 1:
        raise ValueError("candidate roof envelope lies outside SAR image")
    scene_crops = {
        date: {key: value[r0 : r1 + 1, c0 : c1 + 1] for key, value in arrays.items()}
        for date, arrays in evidence.items()
    }
    amp_crop = median_amplitude[r0 : r1 + 1, c0 : c1 + 1]
    edge_crop = median_edge[r0 : r1 + 1, c0 : c1 + 1]
    nearby = (
        (ps["col"] >= c0)
        & (ps["col"] <= c1)
        & (ps["row"] >= r0)
        & (ps["row"] <= r1)
        & (ps["coh"] >= 0.55)
    )
    ps_cols = np.rint(ps["col"][nearby]).astype(np.int64) - c0
    ps_rows = np.rint(ps["row"][nearby]).astype(np.int64) - r0
    ps_coh = ps["coh"][nearby]
    keep = (ps_cols >= 0) & (ps_cols < width) & (ps_rows >= 0) & (ps_rows < height)
    ps_cols, ps_rows, ps_coh = ps_cols[keep], ps_rows[keep], ps_coh[keep]
    image_extent = box(0.0, 0.0, float(cols), float(rows))

    def evaluate(candidates: np.ndarray) -> tuple[np.ndarray, list[dict], dict[str, np.ndarray]]:
        roofs = project_roofs(projector, ring, base + candidates, row_shift, col_shift)
        metrics: list[dict] = []
        for candidate, roof in zip(candidates, roofs):
            mask = local_mask(roof, c0, r0, width, height)
            pixels = int(mask.sum())
            coverage_fraction = (
                float(roof.intersection(image_extent).area / roof.area) if not roof.is_empty and roof.area > 0 else 0.0
            )
            row: dict[str, float | int] = {
                "height_m": float(candidate),
                "roof_pixels": pixels,
                "roof_coverage_fraction": coverage_fraction,
            }
            if pixels < 3 or coverage_fraction < 0.98:
                row.update({"bright": 0.0, "contrast": -1.0, "edge": 0.0, "ps": 0.0})
                for date in evidence:
                    row.update({f"bright_{date}": 0.0, f"contrast_{date}": -1.0, f"edge_{date}": 0.0})
                metrics.append(row)
                continue
            boundary = binary_dilation(mask, iterations=1) & ~binary_erosion(mask, iterations=1)
            outside = binary_dilation(mask, iterations=4) & ~binary_dilation(mask, iterations=1)

            def image_metrics(amplitude: np.ndarray, edge: np.ndarray) -> dict[str, float]:
                values = amplitude[mask]
                if search.get("brightness_statistic") == "trimmed_p50_p90":
                    lower, upper = np.percentile(values, (50.0, 90.0))
                    trimmed = values[(values >= lower) & (values <= upper)]
                    bright = float(np.mean(trimmed)) if trimmed.size else float(np.median(values))
                else:
                    cutoff = max(0, int(math.floor(0.70 * values.size)))
                    bright = float(np.mean(np.partition(values, cutoff)[cutoff:]))
                outside_mean = float(np.mean(amplitude[outside])) if np.any(outside) else float(np.mean(amplitude))
                contrast = float(np.mean(values) - outside_mean)
                edge_alignment = float(np.mean(edge[boundary])) if np.any(boundary) else 0.0
                lower, upper = np.percentile(values, (10.0, 90.0))
                ordered = np.sort(np.maximum(values.astype(np.float64), 0.0))[::-1]
                top_count = max(1, int(math.ceil(0.05 * ordered.size)))
                energy_sum = float(ordered.sum())
                return {
                    "bright": bright,
                    "contrast": contrast,
                    "edge": edge_alignment,
                    # These scene-level diagnostics are consumed by V10.  They are
                    # descriptive only here and do not alter the legacy V1--V9 score.
                    "shadow_fraction": float(np.mean(values <= 0.02)),
                    "texture_p90_p10": float(upper - lower),
                    "extreme_bright_fraction": float(np.mean(values >= 0.98)),
                    "top5_energy_fraction": float(ordered[:top_count].sum() / max(energy_sum, 1e-9)),
                }

            median_metrics = image_metrics(amp_crop, edge_crop)
            ps_selector = mask[ps_rows, ps_cols] if ps_rows.size else np.zeros(0, dtype=bool)
            ps_density = float(np.sum(ps_coh[ps_selector])) / math.sqrt(max(pixels, 1)) if ps_rows.size else 0.0
            row.update(
                {
                    "bright": median_metrics["bright"],
                    "contrast": median_metrics["contrast"],
                    "edge": median_metrics["edge"],
                    "ps": ps_density,
                }
            )
            for date, crops in scene_crops.items():
                scene_metrics = image_metrics(crops["amplitude"], crops["edge"])
                row.update(
                    {
                        f"bright_{date}": scene_metrics["bright"],
                        f"contrast_{date}": scene_metrics["contrast"],
                        f"edge_{date}": scene_metrics["edge"],
                        f"shadow_fraction_{date}": scene_metrics["shadow_fraction"],
                        f"texture_p90_p10_{date}": scene_metrics["texture_p90_p10"],
                        f"extreme_bright_fraction_{date}": scene_metrics["extreme_bright_fraction"],
                        f"top5_energy_fraction_{date}": scene_metrics["top5_energy_fraction"],
                    }
                )
            metrics.append(row)

        valid = np.asarray(
            [item["roof_pixels"] >= 3 and item["roof_coverage_fraction"] >= 0.98 for item in metrics],
            dtype=bool,
        )
        if not np.any(valid):
            raise ValueError("no fully covered candidate roof intersects SAR image")

        def valid_z(values: np.ndarray) -> np.ndarray:
            result = np.zeros(len(values), dtype=np.float64)
            result[valid] = robust_z(np.asarray(values)[valid])
            return result

        weights = config["score_weights"]
        ps_values = np.asarray([item["ps"] for item in metrics])
        ps_term = valid_z(ps_values) if np.max(ps_values[valid]) > 0 else np.zeros(len(metrics))
        scene_raw_scores: dict[str, np.ndarray] = {}
        for date in evidence:
            scene_raw_scores[date] = (
                float(weights["roof_bright_scatter"]) * valid_z(np.asarray([item[f"bright_{date}"] for item in metrics]))
                + float(weights["roof_boundary_edge"]) * valid_z(np.asarray([item[f"edge_{date}"] for item in metrics]))
                + float(weights["roof_inside_outside_contrast"]) * valid_z(np.asarray([item[f"contrast_{date}"] for item in metrics]))
                + float(weights["roof_ps_support"]) * ps_term
            )
        prior_scale = max(0.5 * prior + 6.0, 8.0)
        prior_penalty = float(search["prior_penalty_weight"]) * ((candidates - prior) / prior_scale) ** 2
        fused = np.median(np.stack(list(scene_raw_scores.values())), axis=0) - prior_penalty
        fused = gaussian_filter1d(fused, sigma=0.8, mode="nearest")
        scene_scores = {
            date: gaussian_filter1d(values - prior_penalty, sigma=0.8, mode="nearest")
            for date, values in scene_raw_scores.items()
        }
        fused[~valid] = -1e9
        for values in scene_scores.values():
            values[~valid] = -1e9
        for index, item in enumerate(metrics):
            item["score"] = float(fused[index])
            for date in evidence:
                item[f"score_{date}"] = float(scene_scores[date][index])
        return fused, metrics, scene_scores

    coarse = np.arange(low, high + 0.5 * float(search["coarse_step_m"]), float(search["coarse_step_m"]))
    coarse_scores, coarse_metrics, coarse_scene_scores = evaluate(coarse)
    coarse_valid = np.asarray([item["roof_coverage_fraction"] >= 0.98 and item["roof_pixels"] >= 3 for item in coarse_metrics])
    valid_search_min = float(np.min(coarse[coarse_valid]))
    valid_search_max = float(np.max(coarse[coarse_valid]))
    coarse_best = float(coarse[int(np.argmax(coarse_scores))])
    fine_low = max(low, coarse_best - float(search["fine_half_window_m"]))
    fine_high = min(high, coarse_best + float(search["fine_half_window_m"]))
    fine = np.arange(fine_low, fine_high + 0.5 * float(search["fine_step_m"]), float(search["fine_step_m"]))
    fine_scores, fine_metrics, fine_scene_scores = evaluate(fine)
    best_index = int(np.argmax(fine_scores))
    estimate = float(fine[best_index])
    best_score = float(fine_scores[best_index])
    fine_valid = np.asarray(
        [item["roof_coverage_fraction"] >= 0.98 and item["roof_pixels"] >= 3 for item in fine_metrics],
        dtype=bool,
    )
    alternatives = (np.abs(fine - estimate) >= 1.0) & fine_valid
    margin = best_score - float(np.max(fine_scores[alternatives])) if np.any(alternatives) else 0.0
    accepted = fine[fine_scores >= best_score - 0.35]
    uncertainty = max(0.5, 0.5 * float(np.ptp(accepted))) if accepted.size else 2.0
    boundary_guard = float(search["fine_half_window_m"])
    boundary_hit = estimate <= valid_search_min + boundary_guard or estimate >= valid_search_max - boundary_guard
    scene_heights: dict[str, float] = {}
    scene_fine_metrics: dict[str, list[dict]] = {}
    for date, coarse_values in coarse_scene_scores.items():
        scene_coarse_best = float(coarse[int(np.argmax(coarse_values))])
        scene_fine_low = max(low, scene_coarse_best - float(search["fine_half_window_m"]))
        scene_fine_high = min(high, scene_coarse_best + float(search["fine_half_window_m"]))
        scene_fine = np.arange(
            scene_fine_low,
            scene_fine_high + 0.5 * float(search["fine_step_m"]),
            float(search["fine_step_m"]),
        )
        _, date_metrics, date_scores = evaluate(scene_fine)
        scene_heights[date] = float(scene_fine[int(np.argmax(date_scores[date]))])
        scene_fine_metrics[date] = date_metrics
    scene_range = float(np.ptp(list(scene_heights.values())))
    scene_height_values = list(scene_heights.values())
    scene_pairs = [
        (abs(scene_height_values[left] - scene_height_values[right]), 0.5 * (scene_height_values[left] + scene_height_values[right]))
        for left, right in ((0, 1), (0, 2), (1, 2))
    ]
    closest_pair_range, closest_pair_median = min(scene_pairs, key=lambda item: item[0])
    fused_to_pair_difference = abs(estimate - closest_pair_median)
    acceptance = config.get("acceptance", {})
    all_scene_consensus = scene_range <= float(acceptance.get("maximum_scene_range_m", 10.0))
    two_scene_consensus = bool(
        acceptance.get("allow_two_scene_consensus", False)
        and closest_pair_range <= float(acceptance.get("maximum_closest_pair_range_m", 2.0))
        and fused_to_pair_difference <= float(acceptance.get("maximum_fused_to_pair_difference_m", 3.0))
    )
    scene_consensus = "all_three" if all_scene_consensus else "two_of_three" if two_scene_consensus else "inconsistent"
    if not boundary_hit and margin >= 0.35 and scene_range <= 5.0:
        quality = "high"
    elif not boundary_hit and margin >= 0.15 and scene_consensus in {"all_three", "two_of_three"}:
        quality = "medium"
    else:
        quality = "low"
    best = fine_metrics[best_index]
    rejection_reason = ""
    if bool(acceptance.get("reject_search_boundary", False)) and boundary_hit:
        rejection_reason = "search_boundary"
    elif scene_consensus == "inconsistent":
        rejection_reason = "scene_inconsistent"
    elif margin < float(acceptance.get("minimum_score_margin", -float("inf"))):
        rejection_reason = "weak_peak"
    accepted_solution = rejection_reason == ""
    output_quality = quality if accepted_solution else f"rejected_{rejection_reason}"
    result = {
        "fid": fid,
        "height_prior_m": prior,
        "height_raw_m": estimate,
        "height_est_m": estimate if accepted_solution else float("nan"),
        "base_elevation_m": base,
        "roof_elevation_raw_m": base + estimate,
        "roof_elevation_m": base + estimate if accepted_solution else float("nan"),
        "height_uncertainty_m": uncertainty,
        "score": best_score,
        "score_margin": margin,
        "search_min_m": low,
        "search_max_m": high,
        "valid_search_min_m": valid_search_min,
        "valid_search_max_m": valid_search_max,
        "boundary_hit": int(boundary_hit),
        "quality_raw": quality,
        "quality": output_quality,
        "accepted_solution": int(accepted_solution),
        "rejection_reason": rejection_reason,
        "roof_pixels": int(best["roof_pixels"]),
        "roof_coverage_fraction": float(best["roof_coverage_fraction"]),
        "roof_bright_scatter": float(best["bright"]),
        "roof_contrast": float(best["contrast"]),
        "roof_boundary_edge": float(best["edge"]),
        "roof_ps_support": float(best["ps"]),
        "n_ps_search": int(ps_rows.size),
        "height_scene_range_m": scene_range,
        "closest_scene_pair_range_m": float(closest_pair_range),
        "closest_scene_pair_median_m": float(closest_pair_median),
        "fused_to_scene_pair_difference_m": float(fused_to_pair_difference),
        "scene_consensus": scene_consensus,
        "geometry_mode": "strict_roof_only",
        "search_strategy": str(search.get("strategy", "legacy_asymmetric")),
        "brightness_statistic": str(search.get("brightness_statistic", "top_30_percent_mean")),
        "walls_used": 0,
        "bottom_surface_used": 0,
        "stretch_projection_used": 0,
        "local_building_shift_used": 0,
        "ps_height_fields_used": 0,
    }
    for date, height_value in scene_heights.items():
        result[f"height_est_{date}_m"] = height_value
    curve = [{"fid": fid, "stage": "coarse", **item} for item in coarse_metrics]
    curve.extend({"fid": fid, "stage": "fine", **item} for item in fine_metrics)
    for date, date_metrics in scene_fine_metrics.items():
        curve.extend({"fid": fid, "stage": f"fine_{date}", **item} for item in date_metrics)
    return result, curve


def plot_diagnostic(
    fid: int,
    result: pd.Series,
    curve: pd.DataFrame,
    ring: np.ndarray,
    projector,
    amplitude: np.ndarray,
    config: dict,
    path: Path,
) -> None:
    best = float(result["height_est_m"])
    low = max(float(result["valid_search_min_m"]), best - 10.0)
    high = min(float(result["valid_search_max_m"]), best + 10.0)
    candidates = np.asarray([low, best, high])
    roofs = project_roofs(
        projector,
        ring,
        float(config["base_elevation_m"]) + candidates,
        float(config["registration"]["global_row_shift_px"]),
        float(config["registration"]["global_col_shift_px"]),
    )
    envelope = unary_union(roofs).bounds
    margin = 12
    c0 = max(0, int(envelope[0]) - margin)
    c1 = min(amplitude.shape[1], int(math.ceil(envelope[2])) + margin)
    r0 = max(0, int(envelope[1]) - margin)
    r1 = min(amplitude.shape[0], int(math.ceil(envelope[3])) + margin)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), gridspec_kw={"width_ratios": [1.15, 1.0]})
    axes[0].imshow(amplitude[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=1, origin="upper")
    palette = ["#4C78A8", "#E45756", "#72B7B2"]
    labels = [f"低候选 {low:.1f} m", f"最优 {best:.1f} m", f"高候选 {high:.1f} m"]
    for roof, color, label in zip(roofs, palette, labels):
        x, y = roof.exterior.xy
        axes[0].plot(np.asarray(x) - c0, np.asarray(y) - r0, color=color, lw=1.8, label=label)
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].set_title(f"a  建筑 {fid}：仅顶面候选投影", loc="left", fontweight="bold")
    axes[0].set_xlabel("距离向像素")
    axes[0].set_ylabel("方位向像素")
    fine = curve[curve["stage"] == "fine"].sort_values("height_m")
    coarse = curve[curve["stage"] == "coarse"].sort_values("height_m")
    axes[1].plot(coarse["height_m"], coarse["score"], color="#9E9E9E", lw=1.0, label="粗搜索")
    axes[1].plot(fine["height_m"], fine["score"], color="#E45756", lw=2.0, label="精搜索")
    axes[1].axvline(best, color="#E45756", ls="--", lw=1.0)
    axes[1].scatter([best], [float(result["score"])], color="#E45756", s=28, zorder=3)
    axes[1].set_title("b  顶面匹配得分曲线", loc="left", fontweight="bold")
    axes[1].set_xlabel("候选建筑高度 / m")
    axes[1].set_ylabel("三景融合得分")
    axes[1].legend(fontsize=7)
    axes[1].text(
        0.02,
        0.03,
        f"估计高度 {best:.1f} m\n质量 {result['quality']}\n三景极差 {result['height_scene_range_m']:.1f} m",
        transform=axes[1].transAxes,
        va="bottom",
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_search_process(
    fid: int,
    result: pd.Series,
    curve: pd.DataFrame,
    ring: np.ndarray,
    projector,
    amplitude: np.ndarray,
    config: dict,
    output: Path,
) -> None:
    low = float(result["valid_search_min_m"])
    high = float(result["valid_search_max_m"])
    best = float(result["height_est_m"])
    candidates = np.asarray([low, low + 0.5 * (best - low), best, high])
    roofs = project_roofs(
        projector,
        ring,
        float(config["base_elevation_m"]) + candidates,
        float(config["registration"]["global_row_shift_px"]),
        float(config["registration"]["global_col_shift_px"]),
    )
    bounds = unary_union(roofs).bounds
    margin = 10
    c0 = max(0, int(bounds[0]) - margin)
    c1 = min(amplitude.shape[1], int(math.ceil(bounds[2])) + margin)
    r0 = max(0, int(bounds[1]) - margin)
    r1 = min(amplitude.shape[0], int(math.ceil(bounds[3])) + margin)
    crop = amplitude[r0:r1, c0:c1]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0))
    letters = "abcd"
    for index, (candidate, roof) in enumerate(zip(candidates, roofs)):
        ax = axes.flat[index]
        ax.imshow(crop, cmap="gray", vmin=0, vmax=1, origin="upper")
        x, y = roof.exterior.xy
        color = "#E45756" if abs(candidate - best) < 1e-6 else "#4C78A8"
        ax.plot(np.asarray(x) - c0, np.asarray(y) - r0, color=color, lw=2.0)
        ax.set_title(f"{letters[index]}  候选高度 {candidate:.1f} m", loc="left", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
    coarse = curve[(curve["stage"] == "coarse") & (curve["score"] > -1e8)].sort_values("height_m")
    fine = curve[curve["stage"] == "fine"].sort_values("height_m")
    ax_score = axes[1, 1]
    ax_score.plot(coarse["height_m"], coarse["score"], color="#777777", lw=1.2, label="三景融合粗搜索")
    ax_score.plot(fine["height_m"], fine["score"], color="#E45756", lw=2.0, label="三景融合精搜索")
    ax_score.axvline(best, color="#E45756", ls="--", lw=1.0)
    ax_score.set_title("e  从完整范围定位最优峰", loc="left", fontweight="bold")
    ax_score.set_xlabel("候选建筑高度 / m")
    ax_score.set_ylabel("融合得分")
    ax_score.legend(fontsize=7)
    ax_scene = axes[1, 2]
    palette = ["#4C78A8", "#F28E2B", "#59A14F"]
    for date, color in zip(config["scenes"], palette):
        ax_scene.plot(coarse["height_m"], coarse[f"score_{date}"], color=color, lw=1.2, label=date)
    ax_scene.set_title("f  三景独立得分用于稳定性检查", loc="left", fontweight="bold")
    ax_scene.set_xlabel("候选建筑高度 / m")
    ax_scene.set_ylabel("单景得分")
    ax_scene.legend(fontsize=7)
    fig.suptitle(
        f"仅顶面高度搜索过程示例（建筑 {fid}）：高度改变顶面在 SAR 中的位置",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_full_map(buildings: gpd.GeoDataFrame, table: pd.DataFrame, output: Path) -> None:
    mapped = buildings.to_crs("EPSG:32651").merge(table, on="fid", how="left", validate="one_to_one")
    finite = mapped[np.isfinite(mapped["height_est_m"])].copy()
    outside = mapped[mapped["quality"] == "outside_sar"].copy()
    rejected_boundary = mapped[mapped["quality"] == "rejected_search_boundary"].copy()
    rejected_scene = mapped[mapped["quality"] == "rejected_scene_inconsistent"].copy()
    rejected_weak = mapped[mapped["quality"] == "rejected_weak_peak"].copy()
    vmax = max(20.0, float(finite["height_est_m"].quantile(0.98)))
    norm = colors.Normalize(0.0, vmax)
    fig, ax = plt.subplots(figsize=(11.0, 10.0))
    if not outside.empty:
        outside.plot(ax=ax, color="#E8E8E8", edgecolor="#D0D0D0", linewidth=0.2)
    if not rejected_boundary.empty:
        rejected_boundary.plot(ax=ax, color="#F6D7A7", edgecolor="#D5A45A", linewidth=0.2)
    if not rejected_scene.empty:
        rejected_scene.plot(ax=ax, color="#F2B6B6", edgecolor="#CE7F7F", linewidth=0.2)
    if not rejected_weak.empty:
        rejected_weak.plot(ax=ax, color="#D7D0E8", edgecolor="#A99BC7", linewidth=0.2)
    finite.plot(ax=ax, column="height_est_m", cmap="viridis", norm=norm, edgecolor="#FFFFFF", linewidth=0.15)
    for row in finite.itertuples():
        point = row.geometry.representative_point()
        ax.text(point.x, point.y, f"{row.height_est_m:.0f}", ha="center", va="center", fontsize=1.6, color="#151515")
    ax.set_title("仅顶面严格投影的全区建筑高度估计", fontsize=13, pad=8)
    ax.set_xlabel("Easting / m (UTM 51N)")
    ax.set_ylabel("Northing / m (UTM 51N)")
    ax.set_aspect("equal")
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax, fraction=0.028, pad=0.015, extend="max")
    cbar.set_label("估计建筑高度 / m")
    ax.text(0.01, 0.01, "灰：范围外｜橙：边界峰｜红：跨景冲突｜紫：弱峰；均未填充高度", transform=ax.transAxes, fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9})
    fig.tight_layout()
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_simple_height_map(
    buildings: gpd.GeoDataFrame,
    table: pd.DataFrame,
    output: Path,
    preview_png: Path | None = None,
    title: str = "V3全区建筑高度估计",
) -> None:
    """Plot final height availability only: missing gray, finite values colored by height."""
    mapped = buildings.to_crs("EPSG:32651").merge(table, on="fid", how="left", validate="one_to_one")
    finite = mapped[mapped["height_est_m"].notna()].copy()
    missing = mapped[mapped["height_est_m"].isna()].copy()
    vmax = max(20.0, float(finite["height_est_m"].quantile(0.98)))
    norm = colors.Normalize(vmin=0.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(10.2, 10.0))
    missing.plot(ax=ax, color="#D9D9D9", edgecolor="#C4C4C4", linewidth=0.20)
    finite.plot(
        ax=ax,
        column="height_est_m",
        cmap="viridis",
        norm=norm,
        edgecolor="#FFFFFF",
        linewidth=0.18,
    )
    for row in finite.itertuples():
        point = row.geometry.representative_point()
        text_color = "#FFFFFF" if norm(float(row.height_est_m)) < 0.58 else "#111111"
        ax.text(
            point.x,
            point.y,
            f"{row.height_est_m:.0f}",
            ha="center",
            va="center",
            fontsize=1.65,
            color=text_color,
            clip_on=True,
        )
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel("Easting / m (UTM 51N)")
    ax.set_ylabel("Northing / m (UTM 51N)")
    ax.set_aspect("equal")
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap="viridis"),
        ax=ax,
        fraction=0.030,
        pad=0.016,
        extend="max",
    )
    cbar.set_label("估计建筑高度 / m")
    ax.text(
        0.01,
        0.01,
        f"灰色：无最终高度（{len(missing)}栋）｜彩色：有最终高度（{len(finite)}栋）",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#AAAAAA", "alpha": 0.94},
    )
    fig.tight_layout()
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    if preview_png is not None:
        fig.savefig(preview_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_summary(buildings: gpd.GeoDataFrame, table: pd.DataFrame, output: Path) -> None:
    mapped = buildings.to_crs("EPSG:32651").merge(table, on="fid", how="left", validate="one_to_one")
    finite = mapped[np.isfinite(mapped["height_est_m"])].copy()
    missing = mapped[~np.isfinite(mapped["height_est_m"])].copy()
    fig = plt.figure(figsize=(11.5, 8.2))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.0, 1.0], hspace=0.32, wspace=0.3)
    ax_map = fig.add_subplot(grid[:, 0])
    if not missing.empty:
        missing.plot(ax=ax_map, color="#E7E7E7", edgecolor="none")
    finite.plot(ax=ax_map, column="height_est_m", cmap="viridis", edgecolor="none")
    ax_map.set_title("a  全区空间分布", loc="left", fontweight="bold")
    ax_map.set_axis_off()
    ax_hist = fig.add_subplot(grid[0, 1:])
    ax_hist.hist(finite["height_est_m"], bins=np.arange(0, 151, 5), color="#4C78A8", edgecolor="white")
    ax_hist.axvline(float(finite["height_est_m"].median()), color="#E45756", ls="--", lw=1.5,
                    label=f"中位数 {finite['height_est_m'].median():.1f} m")
    ax_hist.set_title("b  建筑高度分布", loc="left", fontweight="bold")
    ax_hist.set_xlabel("估计高度 / m")
    ax_hist.set_ylabel("建筑数量")
    ax_hist.legend()
    ax_quality = fig.add_subplot(grid[1, 1])
    order = [
        "high",
        "medium",
        "low",
        "rejected_search_boundary",
        "rejected_scene_inconsistent",
        "rejected_weak_peak",
        "outside_sar",
    ]
    counts = table["quality"].value_counts().reindex(order, fill_value=0)
    bars = ax_quality.bar(
        ["高", "中", "低", "边界", "跨景", "弱峰", "范围外"],
        counts,
        color=["#59A14F", "#F2CF5B", "#E15759", "#E6A756", "#D98383", "#9C8AC4", "#BDBDBD"],
    )
    ax_quality.tick_params(axis="x", labelrotation=35)
    ax_quality.bar_label(bars, fontsize=7)
    ax_quality.set_title("c  内部质量等级", loc="left", fontweight="bold")
    ax_quality.set_ylabel("建筑数量")
    ax_scene = fig.add_subplot(grid[1, 2])
    ax_scene.hist(finite["height_scene_range_m"], bins=np.arange(0, 31, 1), color="#72B7B2", edgecolor="white")
    ax_scene.axvline(5.0, color="#E45756", ls="--", lw=1.2)
    ax_scene.set_title("d  三景高度极差", loc="left", fontweight="bold")
    ax_scene.set_xlabel("三景极差 / m")
    ax_scene.set_ylabel("建筑数量")
    fig.suptitle("仅顶面严格投影高度搜索：结果与内部稳定性", fontsize=14, y=0.995)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_recovery_summary(
    buildings: gpd.GeoDataFrame,
    table: pd.DataFrame,
    reference: pd.DataFrame,
    output: Path,
) -> None:
    mapped = buildings.to_crs("EPSG:32651").merge(table, on="fid", how="left", validate="one_to_one")
    reference_fields = reference[["fid", "quality", "rejection_reason"]].rename(
        columns={"quality": "reference_quality", "rejection_reason": "reference_rejection_reason"}
    )
    mapped = mapped.merge(reference_fields, on="fid", how="left", validate="one_to_one")
    preserved = mapped[mapped["solution_source"] == "preserved_v2"].copy()
    recovered = mapped[mapped["solution_source"] == "recovered_v3"].copy()
    background = mapped[~mapped["solution_source"].isin(["preserved_v2", "recovered_v3"])].copy()
    fig = plt.figure(figsize=(11.5, 7.6))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.55, 1.0, 1.0], hspace=0.34, wspace=0.32)
    ax_map = fig.add_subplot(grid[:, 0])
    background.plot(ax=ax_map, color="#ECECEC", edgecolor="white", linewidth=0.15)
    preserved.plot(ax=ax_map, color="#4C78A8", edgecolor="white", linewidth=0.2, label=f"保留V2（{len(preserved)}栋）")
    recovered.plot(ax=ax_map, color="#F28E2B", edgecolor="#8C4D00", linewidth=0.35, label=f"V3新增（{len(recovered)}栋）")
    ax_map.set_title("a  V3新增建筑的空间位置", loc="left", fontweight="bold")
    ax_map.set_axis_off()
    ax_map.legend(loc="lower left", fontsize=8, frameon=True)

    ax_count = fig.add_subplot(grid[0, 1])
    bars = ax_count.bar(["V2", "V3"], [len(preserved), len(preserved) + len(recovered)], color=["#4C78A8", "#59A14F"])
    ax_count.bar_label(bars, fontsize=9)
    ax_count.set_title("b  可靠覆盖数量", loc="left", fontweight="bold")
    ax_count.set_ylabel("有高度建筑数量")

    ax_reason = fig.add_subplot(grid[0, 2])
    reason_order = ["search_boundary", "scene_inconsistent", "weak_peak"]
    reason_counts = recovered["reference_rejection_reason"].value_counts().reindex(reason_order, fill_value=0)
    reason_bars = ax_reason.bar(["边界峰", "跨景冲突", "弱峰"], reason_counts, color=["#E6A756", "#D98383", "#9C8AC4"])
    ax_reason.bar_label(reason_bars, fontsize=8)
    ax_reason.set_title("c  新增项原拒绝原因", loc="left", fontweight="bold")
    ax_reason.set_ylabel("建筑数量")

    ax_delta = fig.add_subplot(grid[1, 1])
    delta = recovered["height_est_m"] - recovered["height_prior_m"]
    ax_delta.hist(delta, bins=np.arange(-42, 44, 4), color="#F28E2B", edgecolor="white")
    ax_delta.axvline(0.0, color="#555555", ls="--", lw=1.0)
    ax_delta.axvline(float(delta.median()), color="#E45756", ls="--", lw=1.4,
                     label=f"中位数 {delta.median():.1f} m")
    ax_delta.set_title("d  新增高度相对粗略高度", loc="left", fontweight="bold")
    ax_delta.set_xlabel("新增高度 − 粗略高度 / m")
    ax_delta.set_ylabel("建筑数量")
    ax_delta.legend(fontsize=7)

    ax_consensus = fig.add_subplot(grid[1, 2])
    consensus_counts = recovered["scene_consensus"].value_counts().reindex(["all_three", "two_of_three"], fill_value=0)
    consensus_bars = ax_consensus.bar(["三景一致", "两景一致"], consensus_counts, color=["#59A14F", "#72B7B2"])
    ax_consensus.bar_label(consensus_bars, fontsize=8)
    ax_consensus.set_title("e  新增项跨景共识", loc="left", fontweight="bold")
    ax_consensus.set_ylabel("建筑数量")
    fig.suptitle("仅顶面高度搜索 V3：保留 V2 基线并可靠补充", fontsize=14, y=0.995)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison(comparison: pd.DataFrame, output: Path) -> None:
    common = comparison.dropna(subset=["roof_only_height_m", "strict_joint_height_m"]).copy()
    difference = common["roof_only_height_m"] - common["strict_joint_height_m"]
    limit = max(float(common[["roof_only_height_m", "strict_joint_height_m"]].max().max()), 20.0)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    axes[0].scatter(
        common["strict_joint_height_m"],
        common["roof_only_height_m"],
        s=10,
        alpha=0.55,
        color="#4C78A8",
        edgecolors="none",
    )
    axes[0].plot([0, limit], [0, limit], color="#555555", ls="--", lw=1.0)
    axes[0].set_xlim(0, limit)
    axes[0].set_ylim(0, limit)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("a  两种方法的逐建筑高度", loc="left", fontweight="bold")
    axes[0].set_xlabel("严格墙面 + 联合归属高度 / m")
    axes[0].set_ylabel("仅顶面高度 / m")
    correlation = float(common[["roof_only_height_m", "strict_joint_height_m"]].corr().iloc[0, 1])
    axes[0].text(0.03, 0.96, f"共同建筑 n={len(common)}\nr={correlation:.3f}", transform=axes[0].transAxes, va="top")
    axes[1].hist(difference, bins=np.arange(-60, 62, 2), color="#72B7B2", edgecolor="white")
    axes[1].axvline(0.0, color="#555555", ls="--", lw=1.0)
    axes[1].axvline(float(np.median(difference)), color="#E45756", ls="--", lw=1.5)
    axes[1].set_title("b  仅顶面相对严格联合结果的差值", loc="left", fontweight="bold")
    axes[1].set_xlabel("仅顶面高度 − 严格联合高度 / m")
    axes[1].set_ylabel("建筑数量")
    axes[1].text(
        0.03,
        0.96,
        f"差值中位数 {np.median(difference):.1f} m\n绝对差中位数 {np.median(np.abs(difference)):.1f} m",
        transform=axes[1].transAxes,
        va="top",
    )
    fig.suptitle("仅顶面方案与既有严格联合方案的内部对照（不代表精度验证）", fontsize=13)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_version_comparison(
    comparison: pd.DataFrame,
    output: Path,
    reference_label: str,
    current_label: str,
) -> None:
    common_raw = comparison.dropna(subset=["reference_height_m", "current_raw_height_m"]).copy()
    accepted = comparison.dropna(subset=["reference_height_m", "current_height_m"]).copy()
    limit = max(float(common_raw[["reference_height_m", "current_raw_height_m"]].max().max()), 20.0)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    rejected = common_raw[common_raw["current_height_m"].isna()]
    axes[0].scatter(rejected["reference_height_m"], rejected["current_raw_height_m"], s=10, alpha=0.35,
                    color="#BDBDBD", label=f"{current_label}拒绝")
    axes[0].scatter(accepted["reference_height_m"], accepted["current_height_m"], s=12, alpha=0.65,
                    color="#4C78A8", label=f"{current_label}接受")
    axes[0].plot([0, limit], [0, limit], color="#555555", ls="--", lw=1.0)
    axes[0].set_xlim(0, limit)
    axes[0].set_ylim(0, limit)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title(f"a  {reference_label}与{current_label}逐建筑候选高度", loc="left", fontweight="bold")
    axes[0].set_xlabel(f"{reference_label}仅顶面高度 / m")
    axes[0].set_ylabel(f"{current_label}原始候选高度 / m")
    axes[0].legend(fontsize=7)
    raw_delta = common_raw["current_raw_height_m"] - common_raw["height_prior_m"]
    accepted_delta = accepted["current_height_m"] - accepted["height_prior_m"]
    bins = np.arange(-45, 47, 2)
    axes[1].hist(raw_delta, bins=bins, color="#BDBDBD", alpha=0.65, label=f"全部原始候选 n={len(raw_delta)}")
    axes[1].hist(accepted_delta, bins=bins, color="#59A14F", alpha=0.75, label=f"最终接受 n={len(accepted_delta)}")
    axes[1].axvline(0.0, color="#555555", ls="--", lw=1.0)
    axes[1].set_title(f"b  {current_label}相对粗略高度的差值", loc="left", fontweight="bold")
    axes[1].set_xlabel(f"{current_label}高度 − 粗略高度 / m")
    axes[1].set_ylabel("建筑数量")
    axes[1].legend(fontsize=7)
    fig.suptitle(f"仅顶面高度搜索：{current_label}与{reference_label}内部对照", fontsize=13)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def run(config_path: Path, max_buildings: int = 0, output_tag: str = "") -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    StrictRadarProjector, clean_ring_lonlat = load_shared_projection(config)
    outputs = config["outputs"]
    directories = {key: resolve(value) / output_tag if output_tag else resolve(value) for key, value in outputs.items()}
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    figure_dirs = {name: directories["figures"] / name for name in ["process", "diagnostics", "final"]}
    for path in figure_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
        for pattern in ("*.png", "*.svg", "*.tiff"):
            for old_figure in path.glob(pattern):
                old_figure.unlink()

    buildings = gpd.read_file(resolve(config["inputs"]["buildings"]), engine="pyogrio").reset_index(drop=True)
    if buildings.crs is None:
        raise ValueError("Building vectors have no CRS")
    buildings["fid"] = np.arange(len(buildings), dtype=np.int64)
    lonlat = buildings.to_crs("EPSG:4326")
    if not bool(lonlat.geometry.is_valid.all()):
        raise ValueError("Invalid building geometry found")
    evidence, median_amplitude, median_edge = load_evidence(config)
    ps = load_ps(resolve(config["inputs"]["ps_points"]))
    master_par = resolve(config["inputs"]["rslc_dir"]) / f"{config['master_scene']}.rslc.par"
    projector = StrictRadarProjector(master_par)
    fids = list(range(len(buildings)))
    if max_buildings > 0:
        fids = fids[:max_buildings]
    results: list[dict] = []
    curves: list[dict] = []
    rings: dict[int, np.ndarray] = {}
    failures: list[dict] = []
    for index, fid in enumerate(fids, start=1):
        ring = clean_ring_lonlat(np.asarray(lonlat.iloc[fid].geometry.exterior.coords))
        rings[fid] = ring
        prior = float(buildings.iloc[fid]["height"])
        try:
            result, curve = score_building(fid, ring, prior, projector, evidence, median_amplitude, median_edge, ps, config)
            results.append(result)
            curves.extend(curve)
        except Exception as exc:
            outside = (
                "outside SAR image" in str(exc)
                or "no candidate roof intersects" in str(exc)
                or "no fully covered candidate roof intersects" in str(exc)
            )
            row = {
                "fid": fid,
                "height_prior_m": prior,
                "height_raw_m": np.nan,
                "height_est_m": np.nan,
                "base_elevation_m": float(config["base_elevation_m"]),
                "roof_elevation_m": np.nan,
                "roof_elevation_raw_m": np.nan,
                "height_uncertainty_m": np.nan,
                "score": np.nan,
                "score_margin": 0.0,
                "search_min_m": np.nan,
                "search_max_m": np.nan,
                "valid_search_min_m": np.nan,
                "valid_search_max_m": np.nan,
                "boundary_hit": 1,
                "quality": "outside_sar" if outside else "failed",
                "quality_raw": "outside_sar" if outside else "failed",
                "accepted_solution": 0,
                "rejection_reason": "outside_sar" if outside else "processing_failure",
                "roof_pixels": 0,
                "roof_coverage_fraction": np.nan,
                "roof_bright_scatter": np.nan,
                "roof_contrast": np.nan,
                "roof_boundary_edge": np.nan,
                "roof_ps_support": np.nan,
                "n_ps_search": 0,
                "height_scene_range_m": np.nan,
                "closest_scene_pair_range_m": np.nan,
                "closest_scene_pair_median_m": np.nan,
                "fused_to_scene_pair_difference_m": np.nan,
                "scene_consensus": "no_observation",
                "geometry_mode": "strict_roof_only",
                "search_strategy": str(config["height_search"].get("strategy", "legacy_asymmetric")),
                "brightness_statistic": str(config["height_search"].get("brightness_statistic", "top_30_percent_mean")),
                "walls_used": 0,
                "bottom_surface_used": 0,
                "stretch_projection_used": 0,
                "local_building_shift_used": 0,
                "ps_height_fields_used": 0,
            }
            for date in config["scenes"]:
                row[f"height_est_{date}_m"] = np.nan
            results.append(row)
            failures.append({"fid": fid, "reason": str(exc), "outside_sar": int(outside)})
        if index % 100 == 0 or index == len(fids):
            print(f"roof-only height search {index}/{len(fids)} failures={len(failures)}", flush=True)

    table = pd.DataFrame(results).sort_values("fid").reset_index(drop=True)
    table["v3_search_height_est_m"] = table["height_est_m"]
    table["v3_search_quality"] = table["quality"]
    table["v3_search_rejection_reason"] = table["rejection_reason"]
    table["solution_source"] = np.where(
        table["height_est_m"].notna(), "recovered_v3", "not_accepted"
    )
    if bool(config.get("acceptance", {}).get("anchor_reference_accepted", False)):
        anchor_path = resolve(config["inputs"]["roof_only_v1_reference"])
        anchor = pd.read_csv(anchor_path)
        anchor = anchor[anchor["height_est_m"].notna()].set_index("fid")
        anchor_mask = table["fid"].isin(anchor.index)
        anchor_rows = table.loc[anchor_mask, "fid"].map(anchor["height_est_m"])
        table.loc[anchor_mask, "height_est_m"] = anchor_rows.to_numpy()
        table.loc[anchor_mask, "roof_elevation_m"] = (
            float(config["base_elevation_m"]) + anchor_rows.to_numpy()
        )
        for column in ["height_uncertainty_m", "quality", "quality_raw"]:
            if column in anchor.columns:
                table.loc[anchor_mask, column] = table.loc[anchor_mask, "fid"].map(anchor[column]).to_numpy()
        table.loc[anchor_mask, "accepted_solution"] = 1
        table.loc[anchor_mask, "rejection_reason"] = ""
        table.loc[anchor_mask, "scene_consensus"] = "preserved_v2"
        table.loc[anchor_mask, "solution_source"] = "preserved_v2"
    curve_table = pd.DataFrame(curves)
    table.to_csv(directories["tables"] / "roof_only_building_heights.csv", index=False)
    curve_table.to_csv(directories["tables"] / "roof_only_score_curves.csv", index=False)
    pd.DataFrame(failures, columns=["fid", "reason", "outside_sar"]).to_csv(
        directories["tables"] / "roof_only_failures.csv", index=False
    )
    selected = buildings[buildings["fid"].isin(fids)].copy()
    vector = selected.merge(table, on="fid", how="left", validate="one_to_one")
    vector.to_file(directories["vectors"] / "roof_only_building_heights.gpkg", layer="roof_only_heights", driver="GPKG")

    comparison = pd.DataFrame()
    reference_path = resolve(config["inputs"]["strict_joint_reference"])
    if reference_path.exists():
        reference = pd.read_csv(reference_path, usecols=["fid", "height_est_m"]).rename(
            columns={"height_est_m": "strict_joint_height_m"}
        )
        comparison = table[["fid", "height_est_m", "quality", "height_scene_range_m"]].rename(
            columns={"height_est_m": "roof_only_height_m", "quality": "roof_only_quality"}
        ).merge(reference, on="fid", how="left", validate="one_to_one")
        comparison["roof_only_minus_strict_joint_m"] = (
            comparison["roof_only_height_m"] - comparison["strict_joint_height_m"]
        )
        comparison["absolute_difference_m"] = comparison["roof_only_minus_strict_joint_m"].abs()
        comparison.to_csv(directories["tables"] / "roof_only_vs_strict_joint_comparison.csv", index=False)
    version_comparison = pd.DataFrame()
    comparison_config = config.get("version_comparison", {})
    reference_label = str(comparison_config.get("reference_label", "V1"))
    current_label = str(comparison_config.get("current_label", "V2"))
    comparison_stem = str(comparison_config.get("output_stem", "roof_only_v2_vs_v1_comparison"))
    v1_reference_text = config["inputs"].get("roof_only_v1_reference")
    if v1_reference_text:
        v1_reference_path = resolve(v1_reference_text)
        if v1_reference_path.exists():
            reference_version_full = pd.read_csv(v1_reference_path)
            reference_version = reference_version_full[["fid", "height_est_m"]].rename(
                columns={"height_est_m": "reference_height_m"}
            )
            version_comparison = table[
                ["fid", "height_prior_m", "height_raw_m", "height_est_m", "quality", "rejection_reason"]
            ].rename(columns={
                "height_raw_m": "current_raw_height_m",
                "height_est_m": "current_height_m",
                "quality": "current_quality",
            })
            version_comparison = version_comparison.merge(
                reference_version, on="fid", how="left", validate="one_to_one"
            )
            version_comparison.to_csv(directories["tables"] / f"{comparison_stem}.csv", index=False)

    finite = table[np.isfinite(table["height_est_m"])].copy()
    diagnostic_count = int(config["visualization"]["diagnostic_buildings"])
    diagnostic_pool = finite[finite["solution_source"] == "recovered_v3"]
    if diagnostic_pool.empty:
        diagnostic_pool = finite
    diagnostics = diagnostic_pool.sort_values(["quality", "score_margin"], ascending=[True, False]).head(diagnostic_count)
    for row in diagnostics.itertuples(index=False):
        fid = int(row.fid)
        plot_diagnostic(
            fid,
            pd.Series(row._asdict()),
            curve_table[curve_table["fid"] == fid],
            rings[fid],
            projector,
            median_amplitude,
            config,
            figure_dirs["diagnostics"] / f"building_{fid:04d}_roof_only_diagnostic",
        )
    if not finite.empty:
        newly_recovered = finite[finite["solution_source"] == "recovered_v3"]
        finite_for_process = newly_recovered if not newly_recovered.empty else finite
        high_quality = finite_for_process[finite_for_process["quality"] == "high"]
        process_candidates = (high_quality if not high_quality.empty else finite_for_process).copy()
        process_candidates["absolute_prior_difference_m"] = (
            process_candidates["height_est_m"] - process_candidates["height_prior_m"]
        ).abs()
        process_row = process_candidates.sort_values(
            ["height_scene_range_m", "absolute_prior_difference_m", "score_margin"],
            ascending=[True, True, False],
        ).iloc[0]
        process_fid = int(process_row["fid"])
        plot_search_process(
            process_fid,
            process_row,
            curve_table[curve_table["fid"] == process_fid],
            rings[process_fid],
            projector,
            median_amplitude,
            config,
            figure_dirs["process"] / "roof_only_height_search_process",
        )
    plot_full_map(selected, table, figure_dirs["final"] / "roof_only_full_area_height_map")
    plot_simple_height_map(selected, table, figure_dirs["final"] / "roof_only_simple_height_availability_map")
    plot_summary(selected, table, figure_dirs["final"] / "roof_only_height_summary")
    if not comparison.empty:
        plot_method_comparison(comparison, figure_dirs["final"] / "roof_only_vs_strict_joint_comparison")
    if not version_comparison.empty:
        plot_version_comparison(
            version_comparison,
            figure_dirs["final"] / comparison_stem,
            reference_label,
            current_label,
        )
        if {"quality", "rejection_reason"}.issubset(reference_version_full.columns):
            plot_recovery_summary(
                selected,
                table,
                reference_version_full,
                figure_dirs["final"] / "roof_only_v3_recovery_summary",
            )

    outside_count = int((table["quality"] == "outside_sar").sum())
    failed_count = int((table["quality"] == "failed").sum())
    summary = {
        "method": config["method"],
        "buildings": int(len(table)),
        "finite_heights": int(table["height_est_m"].notna().sum()),
        "outside_sar": outside_count,
        "processing_failures": failed_count,
        "base_elevation_m": float(config["base_elevation_m"]),
        "walls_used": False,
        "bottom_surface_used": False,
        "stretch_projection_used": False,
        "local_building_shift_used": False,
        "ps_height_fields_used": False,
        "height_median": float(table["height_est_m"].median(skipna=True)),
        "height_mean": float(table["height_est_m"].mean(skipna=True)),
        "quality_counts": {str(key): int(value) for key, value in table["quality"].value_counts().items()},
        "solution_source_counts": {
            str(key): int(value) for key, value in table["solution_source"].value_counts().items()
        },
        "accepted_scene_consensus_counts": {
            str(key): int(value)
            for key, value in table.loc[table["height_est_m"].notna(), "scene_consensus"].value_counts().items()
        },
        "outputs": {key: str(value) for key, value in directories.items()},
    }
    if not comparison.empty:
        common = comparison.dropna(subset=["roof_only_height_m", "strict_joint_height_m"])
        summary["strict_joint_internal_comparison"] = {
            "common_buildings": int(len(common)),
            "correlation": float(common[["roof_only_height_m", "strict_joint_height_m"]].corr().iloc[0, 1]),
            "median_absolute_difference_m": float(common["absolute_difference_m"].median()),
            "p90_absolute_difference_m": float(common["absolute_difference_m"].quantile(0.9)),
            "accuracy_validation": False,
        }
    if not version_comparison.empty:
        raw = version_comparison.dropna(subset=["current_raw_height_m"])
        accepted = version_comparison.dropna(subset=["current_height_m"])
        summary[comparison_stem] = {
            "reference_label": reference_label,
            "current_label": current_label,
            "current_raw_candidates": int(len(raw)),
            "current_accepted": int(len(accepted)),
            "current_rejected": int(len(raw) - len(accepted)),
            "raw_minus_prior_mean_m": float((raw["current_raw_height_m"] - raw["height_prior_m"]).mean()),
            "raw_minus_prior_median_m": float((raw["current_raw_height_m"] - raw["height_prior_m"]).median()),
            "accepted_minus_prior_mean_m": float((accepted["current_height_m"] - accepted["height_prior_m"]).mean()),
            "accepted_minus_prior_median_m": float((accepted["current_height_m"] - accepted["height_prior_m"]).median()),
        }
    summary_path = directories["tables"] / "roof_only_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict roof-only SAR building height search")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--max-buildings", type=int, default=0)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()
    run(args.config.resolve(), args.max_buildings, args.output_tag)


if __name__ == "__main__":
    main()
