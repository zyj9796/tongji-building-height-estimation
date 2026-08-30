from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-shape-adaptive-sar")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter, sobel, uniform_filter
from shapely.geometry import box

from run_pixel_offset_height import ROOT, estimate_one, load_module, mpl_patch, resolve


STRATEGY_COLORS = {
    "elongated": "#E69F00",
    "compact": "#0072B2",
    "small": "#CC79A7",
    "large": "#009E73",
    "regular": "#D55E00",
}
STRATEGY_LABELS = {
    "elongated": "长条形",
    "compact": "近方形",
    "small": "小建筑",
    "large": "大建筑",
    "regular": "常规建筑",
}


def robust_unit(values, lower=2.0, upper=98.0):
    lo, hi = np.percentile(values[np.isfinite(values)], (lower, upper))
    return np.clip((values - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0).astype(np.float32)


def enhance_scene(amplitude):
    log_amplitude = np.log1p(20.0 * np.clip(amplitude, 0.0, None)) / np.log(21.0)
    local_mean = uniform_filter(log_amplitude, size=5, mode="nearest")
    local_second = uniform_filter(log_amplitude * log_amplitude, size=5, mode="nearest")
    local_variance = np.maximum(local_second - local_mean * local_mean, 0.0)
    noise_variance = float(np.percentile(local_variance, 35.0))
    lee_weight = np.clip((local_variance - noise_variance) / np.maximum(local_variance, 1e-6), 0.0, 1.0)
    lee = local_mean + lee_weight * (log_amplitude - local_mean)

    broad_mean = gaussian_filter(lee, sigma=4.0, mode="nearest")
    broad_variance = gaussian_filter((lee - broad_mean) ** 2, sigma=4.0, mode="nearest")
    local_z = np.clip((lee - broad_mean) / np.sqrt(broad_variance + 2.5e-3), -2.5, 2.5)
    enhanced = robust_unit(0.68 * robust_unit(lee) + 0.32 * robust_unit(local_z))

    fine = gaussian_filter(enhanced, sigma=0.65, mode="nearest")
    coarse = gaussian_filter(enhanced, sigma=1.45, mode="nearest")
    edge_fine = np.hypot(sobel(fine, axis=1, mode="nearest"), sobel(fine, axis=0, mode="nearest"))
    edge_coarse = np.hypot(sobel(coarse, axis=1, mode="nearest"), sobel(coarse, axis=0, mode="nearest"))
    edge = robust_unit(0.65 * robust_unit(edge_fine, 2.0, 97.0) + 0.35 * robust_unit(edge_coarse, 2.0, 97.0), 1.0, 99.0)
    return enhanced, edge


def enhance_evidence(raw_evidence):
    enhanced = {}
    for date, arrays in raw_evidence.items():
        amplitude, edge = enhance_scene(arrays["amplitude"])
        enhanced[date] = {"amplitude": amplitude, "edge": edge}
    median_amplitude = np.median(np.stack([item["amplitude"] for item in enhanced.values()]), axis=0)
    median_edge = np.median(np.stack([item["edge"] for item in enhanced.values()]), axis=0)
    return enhanced, median_amplitude.astype(np.float32), median_edge.astype(np.float32)


def shape_descriptor(geometry):
    rectangle = geometry.minimum_rotated_rectangle
    coordinates = np.asarray(rectangle.exterior.coords)
    sides = np.hypot(np.diff(coordinates[:, 0]), np.diff(coordinates[:, 1]))
    sides = sides[sides > 1e-6]
    short_side = float(np.min(sides)) if sides.size else 0.0
    long_side = float(np.max(sides)) if sides.size else 0.0
    aspect = long_side / max(short_side, 1e-6)
    area = float(geometry.area)
    if aspect >= 3.0:
        strategy = "elongated"
    elif area < 150.0:
        strategy = "small"
    elif area > 1200.0:
        strategy = "large"
    elif aspect <= 1.6:
        strategy = "compact"
    else:
        strategy = "regular"
    return {
        "footprint_area_m2": area,
        "oriented_long_side_m": long_side,
        "oriented_short_side_m": short_side,
        "aspect_ratio": aspect,
        "shape_strategy": strategy,
    }


def adaptive_config(base_config, descriptor):
    config = copy.deepcopy(base_config)
    strategy = descriptor["shape_strategy"]
    settings = {
        "elongated": {
            "weights": (0.70, 0.23, 0.07, 0.10), "minimum_pixels": 6,
            "boundary_step": 0.35, "perpendicular_limit": 0.75,
            "maximum_parallel": 16.0, "minimum_margin": 0.12,
        },
        "compact": {
            "weights": (0.47, 0.36, 0.17, 0.06), "minimum_pixels": 6,
            "boundary_step": 0.45, "perpendicular_limit": 1.50,
            "maximum_parallel": 16.0, "minimum_margin": 0.15,
        },
        "small": {
            "weights": (0.44, 0.24, 0.32, 0.08), "minimum_pixels": 3,
            "boundary_step": 0.30, "perpendicular_limit": 1.00,
            "maximum_parallel": 12.0, "minimum_margin": 0.18,
        },
        "large": {
            "weights": (0.64, 0.31, 0.05, 0.09), "minimum_pixels": 12,
            "boundary_step": 0.70, "perpendicular_limit": 1.00,
            "maximum_parallel": 16.0, "minimum_margin": 0.12,
        },
        "regular": {
            "weights": (0.57, 0.30, 0.13, 0.07), "minimum_pixels": 6,
            "boundary_step": 0.50, "perpendicular_limit": 1.25,
            "maximum_parallel": 16.0, "minimum_margin": 0.15,
        },
    }[strategy]
    edge, contrast, bright, penalty = settings["weights"]
    config["score_weights"].update(
        {
            "roof_boundary_edge": edge,
            "roof_inside_outside_contrast": contrast,
            "roof_bright_scatter": bright,
            "perpendicular_shift_penalty": penalty,
        }
    )
    config["search"]["minimum_roof_pixels"] = settings["minimum_pixels"]
    config["search"]["boundary_sampling_step_px"] = settings["boundary_step"]
    config["search"]["maximum_parallel_correction_px"] = settings["maximum_parallel"]
    config["registration"]["local_perpendicular_correction_limit_px"] = settings["perpendicular_limit"]
    config["acceptance"]["minimum_score_margin"] = settings["minimum_margin"]
    return config


def add_processed_sar(ax, image):
    lo, hi = np.percentile(image, (1.0, 99.5))
    ax.imshow(image, cmap="gray", vmin=lo, vmax=hi, origin="upper", interpolation="nearest", rasterized=True)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_aspect("equal")
    ax.set_xlabel("Range pixel")
    ax.set_ylabel("Azimuth pixel")


def plot_enhancement(raw_median, enhanced_median, edge_median, output, preview):
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), sharex=True, sharey=True)
    raw_display = np.sqrt(np.clip(raw_median, 0.0, 1.0))
    for ax, image, title, cmap in (
        (axes[0], raw_display, "a  原始多景中值幅度", "gray"),
        (axes[1], enhanced_median, "b  保边去斑与局部对比度增强", "gray"),
        (axes[2], edge_median, "c  多尺度建筑边缘响应", "magma"),
    ):
        lo, hi = np.percentile(image, (1.0, 99.5))
        ax.imshow(image, cmap=cmap, vmin=lo, vmax=hi, origin="upper", interpolation="nearest", rasterized=True)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.set_aspect("equal")
        ax.set_xlabel("Range pixel")
        ax.set_title(title, loc="left", fontweight="bold")
    axes[0].set_ylabel("Azimuth pixel")
    fig.suptitle("用于建筑轮廓匹配的SAR特征增强", fontsize=13, y=0.985)
    fig.text(0.5, 0.93, "多景分别处理后再中值融合；增强结果仅用于匹配评分，不修改原始RSLC", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(preview, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_correction(initial, corrected, table, enhanced_median, output, preview):
    reliable = table[table.accepted == 1].copy()
    reliable_ids = set(reliable.fid.astype(int))
    image_box = box(0.0, 0.0, float(enhanced_median.shape[1]), float(enhanced_median.shape[0]))
    visible = initial[initial.geometry.intersects(image_box)].copy()
    rejected = visible[~visible.fid.isin(reliable_ids)].copy()
    accepted_geometry = corrected[corrected.fid.isin(reliable_ids)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0), sharex=True, sharey=True)
    ax = axes[0]
    add_processed_sar(ax, enhanced_median)
    if len(visible):
        ax.add_collection(PatchCollection([mpl_patch(g) for g in visible.geometry], facecolor="none", edgecolor="#00B8D4", linewidth=0.42, alpha=0.82))
    ax.set_title("a  增强SAR上的height初始投影", loc="left", fontweight="bold")
    ax.text(0.01, 0.99, f"青色：校正前投影（{len(visible)}栋与SAR相交）", transform=ax.transAxes, va="top", fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9})

    ax = axes[1]
    add_processed_sar(ax, enhanced_median)
    if len(rejected):
        ax.add_collection(PatchCollection([mpl_patch(g) for g in rejected.geometry], facecolor="none", edgecolor="#B8B8B8", linewidth=0.28, alpha=0.45))
    handles = [Patch(facecolor="none", edgecolor="#B8B8B8", label="未通过质量控制")]
    counts = reliable.shape_strategy.value_counts()
    for strategy in STRATEGY_COLORS:
        part = accepted_geometry[accepted_geometry.shape_strategy == strategy]
        if len(part):
            ax.add_collection(PatchCollection([mpl_patch(g) for g in part.geometry], facecolor="none", edgecolor=STRATEGY_COLORS[strategy], linewidth=0.66, alpha=0.96))
        handles.append(Patch(facecolor="none", edgecolor=STRATEGY_COLORS[strategy], label=f"{STRATEGY_LABELS[strategy]}（{int(counts.get(strategy, 0))}栋）"))
    ax.set_title("b  SAR增强与形态自适应校正后", loc="left", fontweight="bold")
    ax.text(0.01, 0.99, f"彩色：可靠逐建筑校正（{len(reliable)}栋）｜灰色：保留初始位置", transform=ax.transAxes, va="top", fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9})
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=True, bbox_to_anchor=(0.5, 0.005), fontsize=6.8)
    fig.suptitle("SAR特征增强与建筑形态自适应的逐建筑局部校正", fontsize=13, y=0.985)
    fig.text(0.5, 0.935, "长条形强调长边连续性；近方形强调闭合边缘与内外对比；大小建筑采用不同像素与权重参数", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.89))
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(preview, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    roof_module = load_module("roof_evidence_shape_adaptive", resolve(config["inputs"]["roof_evidence_code"]))
    projection = load_module("strict_projection_shape_adaptive", resolve(config["inputs"]["projection_code"]))
    buildings = gpd.read_file(resolve(config["inputs"]["buildings"]), engine="pyogrio").to_crs(4326).reset_index(drop=True)
    metric_buildings = buildings.to_crs(32651)
    descriptors = [shape_descriptor(geometry) for geometry in metric_buildings.geometry]
    projector = projection.StrictRadarProjector(resolve(config["inputs"]["rslc_dir"]) / f"{config['master_scene']}.rslc.par")
    raw_evidence, raw_median, _ = roof_module.load_evidence({**config, "inputs": {**config["inputs"], "rslc_dir": config["inputs"]["rslc_dir"]}})
    evidence, enhanced_median, edge_median = enhance_evidence(raw_evidence)

    processed_dir = ROOT / "results/processed_sar"
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(processed_dir / "building_feature_enhanced_sar.npz", enhanced_amplitude=enhanced_median, multiscale_edge=edge_median)

    results, initial_rows, corrected_rows = [], [], []
    for fid, building in buildings.iterrows():
        descriptor = descriptors[fid]
        local_config = adaptive_config(config, descriptor)
        result, _, _, prior, corrected = estimate_one(fid, building, projector, projection.clean_ring_lonlat, evidence, enhanced_median.shape, local_config)
        result.update(descriptor)
        result["corrected_absolute_elevation_m"] = float(config["base_elevation_m"]) + result.get("height_raw_m", np.nan)
        result["height_change_m"] = float(result.get("height_raw_m", np.nan) - result["height_prior_m"]) if np.isfinite(result.get("height_raw_m", np.nan)) else np.nan
        result["correction_magnitude_px"] = float(math.hypot(result.get("prior_correction_col_px", np.nan), result.get("prior_correction_row_px", np.nan))) if np.isfinite(result.get("prior_correction_col_px", np.nan)) else np.nan
        results.append(result)
        initial_rows.append({"fid": int(fid), "internal_fid": int(fid), "clean_id": int(building.clean_id), "source_height_m": float(building["height"]), **descriptor, "geometry": prior})
        if corrected is not None and not corrected.is_empty:
            corrected_rows.append({"fid": int(fid), "internal_fid": int(fid), "clean_id": int(building.clean_id), "source_height_m": float(building["height"]), "corrected_absolute_elevation_m": result["corrected_absolute_elevation_m"], "height_change_m": result["height_change_m"], "correction_col_px": result.get("prior_correction_col_px", np.nan), "correction_row_px": result.get("prior_correction_row_px", np.nan), "correction_magnitude_px": result["correction_magnitude_px"], "accepted": int(result["accepted"]), "quality": str(result["quality"]), "score": result.get("score", np.nan), "score_margin": result.get("score_margin", np.nan), **descriptor, "geometry": corrected})
        if (fid + 1) % 25 == 0 or fid + 1 == len(buildings):
            print(f"shape-adaptive {fid + 1}/{len(buildings)} accepted={sum(int(item['accepted']) for item in results)}", flush=True)

    table = pd.DataFrame(results)
    initial_gdf = gpd.GeoDataFrame(initial_rows, geometry="geometry", crs=None)
    corrected_gdf = gpd.GeoDataFrame(corrected_rows, geometry="geometry", crs=None)
    reliable_gdf = corrected_gdf[corrected_gdf.accepted == 1].copy()
    table_path = ROOT / "results/tables/shape_adaptive_enhanced_sar_correction.csv"
    table.to_csv(table_path, index=False)
    gpkg = ROOT / "results/vectors/shape_adaptive_enhanced_sar_correction.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    initial_gdf.to_file(gpkg, layer="shp_height_initial", driver="GPKG")
    corrected_gdf.to_file(gpkg, layer="adaptive_best_all_calculable", driver="GPKG")
    reliable_gdf.to_file(gpkg, layer="adaptive_corrected_reliable", driver="GPKG")

    enhancement_svg = ROOT / "results/picall/正式图件/08_合成孔径雷达建筑特征增强.svg"
    correction_svg = ROOT / "results/picall/正式图件/09_形态自适应增强雷达校正.svg"
    plot_enhancement(raw_median, enhanced_median, edge_median, enhancement_svg, Path("/tmp/sar_feature_enhancement_qa.png"))
    plot_correction(initial_gdf, corrected_gdf, table, enhanced_median, correction_svg, Path("/tmp/shape_adaptive_correction_qa.png"))
    for output in (enhancement_svg, correction_svg):
        (ROOT / "results/picall/过程图件" / output.name).write_bytes(output.read_bytes())

    baseline_path = ROOT / "results/tables/shp_height_local_sar_correction.csv"
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    reliable = table[table.accepted == 1]
    strategy_counts = table.shape_strategy.value_counts()
    accepted_by_strategy = reliable.shape_strategy.value_counts()
    summary = {
        "method": "sar_feature_enhancement_and_shape_adaptive_local_correction",
        "buildings": int(len(table)),
        "calculable_local_best": int(len(corrected_gdf)),
        "reliable_local_corrections": int(len(reliable)),
        "baseline_reliable_local_corrections": int(baseline.accepted.sum()) if len(baseline) else None,
        "quality_counts": {str(k): int(v) for k, v in table.quality.value_counts().items()},
        "strategy_counts": {str(k): int(v) for k, v in strategy_counts.items()},
        "accepted_by_strategy": {str(k): int(v) for k, v in accepted_by_strategy.items()},
        "reliable_median_correction_px": float(reliable.correction_magnitude_px.median()) if len(reliable) else None,
        "reliable_median_abs_height_change_m": float(reliable.height_change_m.abs().median()) if len(reliable) else None,
        "unreliable_results_forced_to_move": False,
        "processed_sar": str(processed_dir / "building_feature_enhanced_sar.npz"),
        "table": str(table_path),
        "vector": str(gpkg),
        "enhancement_svg": str(enhancement_svg),
        "correction_svg": str(correction_svg),
    }
    summary_path = ROOT / "results/tables/shape_adaptive_enhanced_sar_correction_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
