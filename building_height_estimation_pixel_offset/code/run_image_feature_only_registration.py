from __future__ import annotations

import json
import math
import os
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-image-feature-registration")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates, sobel
from shapely import affinity
from shapely.geometry import box, mapping

from run_pixel_offset_height import ROOT, load_module, mpl_patch, resolve
from run_shape_adaptive_enhanced_sar_correction import (
    STRATEGY_COLORS,
    STRATEGY_LABELS,
    add_processed_sar,
    enhance_scene,
    shape_descriptor,
)


def robust_z(values):
    values = np.asarray(values, dtype=float)
    center = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - center)))
    scale = max(1.4826 * mad, float(np.nanstd(values)) * 0.25, 1e-6)
    return np.clip((values - center) / scale, -4.0, 4.0)


def prepare_evidence(raw_evidence):
    evidence = {}
    for date, arrays in raw_evidence.items():
        amplitude, edge = enhance_scene(arrays["amplitude"])
        smoothed = gaussian_filter(amplitude, sigma=0.8, mode="nearest")
        gx = sobel(smoothed, axis=1, mode="nearest")
        gy = sobel(smoothed, axis=0, mode="nearest")
        evidence[date] = {
            "amplitude": amplitude,
            "edge": edge,
            "gx": gx.astype(np.float32),
            "gy": gy.astype(np.float32),
        }
    median_amplitude = np.median(np.stack([item["amplitude"] for item in evidence.values()]), axis=0)
    return evidence, median_amplitude.astype(np.float32)


def boundary_samples(geometry, strategy, maximum=96):
    ring = geometry.exterior
    count = min(maximum, max(16, int(math.ceil(ring.length / 0.65))))
    distances = np.linspace(0.0, ring.length, count, endpoint=False)
    points = np.asarray([(ring.interpolate(float(distance)).x, ring.interpolate(float(distance)).y) for distance in distances])
    previous = np.roll(points, 1, axis=0)
    following = np.roll(points, -1, axis=0)
    tangent = following - previous
    tangent /= np.maximum(np.hypot(tangent[:, 0], tangent[:, 1])[:, None], 1e-6)
    normals = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    weights = np.ones(len(points), dtype=float)
    if strategy == "elongated":
        rectangle = geometry.minimum_rotated_rectangle
        coordinates = np.asarray(rectangle.exterior.coords)
        vectors = np.diff(coordinates, axis=0)
        lengths = np.hypot(vectors[:, 0], vectors[:, 1])
        dominant = vectors[int(np.argmax(lengths))]
        dominant /= max(float(np.hypot(dominant[0], dominant[1])), 1e-6)
        alignment = np.abs(tangent @ dominant)
        weights = np.where(alignment >= 0.75, 1.8, 0.45)
    return points, normals, weights


def area_samples(geometry, maximum_inside=56, maximum_outside=72):
    minx, miny, maxx, maxy = geometry.bounds
    c0, r0 = int(math.floor(minx)) - 4, int(math.floor(miny)) - 4
    c1, r1 = int(math.ceil(maxx)) + 4, int(math.ceil(maxy)) + 4
    width, height = max(1, c1 - c0 + 1), max(1, r1 - r0 + 1)
    mask = rasterize(
        [(mapping(geometry), 1)], out_shape=(height, width),
        transform=Affine.translation(c0, r0), fill=0, all_touched=True, dtype="uint8",
    ).astype(bool)
    outside = binary_dilation(mask, iterations=4) & ~binary_dilation(mask, iterations=1)
    inside_rows, inside_cols = np.where(mask)
    outside_rows, outside_cols = np.where(outside)

    def reduce(rows, cols, maximum):
        if len(rows) > maximum:
            indices = np.linspace(0, len(rows) - 1, maximum).astype(int)
            rows, cols = rows[indices], cols[indices]
        return np.column_stack([cols + c0 + 0.5, rows + r0 + 0.5]).astype(float)

    return reduce(inside_rows, inside_cols, maximum_inside), reduce(outside_rows, outside_cols, maximum_outside)


def strategy_settings(strategy):
    return {
        "elongated": {"limit": 16.0, "weights": (0.58, 0.24, 0.14, 0.04), "margin": 0.10},
        "compact": {"limit": 16.0, "weights": (0.40, 0.16, 0.30, 0.14), "margin": 0.12},
        "small": {"limit": 10.0, "weights": (0.34, 0.14, 0.16, 0.36), "margin": 0.16},
        "large": {"limit": 18.0, "weights": (0.52, 0.26, 0.18, 0.04), "margin": 0.10},
        "regular": {"limit": 16.0, "weights": (0.46, 0.20, 0.25, 0.09), "margin": 0.12},
    }[strategy]


def candidate_grid(limit, step, center=(0.0, 0.0), half_window=None):
    half = limit if half_window is None else half_window
    xs = np.arange(max(-limit, center[0] - half), min(limit, center[0] + half) + 0.5 * step, step)
    ys = np.arange(max(-limit, center[1] - half), min(limit, center[1] + half) + 0.5 * step, step)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def sample_array(array, points, shifts):
    cols = points[:, 0][None, :] + shifts[:, 0][:, None]
    rows = points[:, 1][None, :] + shifts[:, 1][:, None]
    values = map_coordinates(array, [rows.ravel(), cols.ravel()], order=1, mode="nearest")
    return values.reshape(len(shifts), len(points))


def score_candidates(shifts, boundary, normals, boundary_weights, inside, outside, evidence, strategy):
    rows, cols = next(iter(evidence.values()))["amplitude"].shape
    all_points = np.vstack([boundary, inside, outside])
    shifted_cols = all_points[:, 0][None, :] + shifts[:, 0][:, None]
    shifted_rows = all_points[:, 1][None, :] + shifts[:, 1][:, None]
    valid_fraction = np.mean(
        (shifted_cols >= 0.0) & (shifted_cols <= cols - 1.0)
        & (shifted_rows >= 0.0) & (shifted_rows <= rows - 1.0), axis=1,
    )
    edge_weight, continuity_weight, contrast_weight, bright_weight = strategy_settings(strategy)["weights"]
    scene_scores, diagnostics = [], {}
    for date, arrays in evidence.items():
        edge = sample_array(arrays["edge"], boundary, shifts)
        gx = sample_array(arrays["gx"], boundary, shifts)
        gy = sample_array(arrays["gy"], boundary, shifts)
        magnitude = np.hypot(gx, gy)
        orientation = np.abs(gx * normals[:, 0][None, :] + gy * normals[:, 1][None, :]) / np.maximum(magnitude, 1e-6)
        oriented_edge = np.average(edge * (0.25 + 0.75 * orientation), axis=1, weights=boundary_weights)
        continuity = np.average(edge >= 0.38, axis=1, weights=boundary_weights)
        inside_values = sample_array(arrays["amplitude"], inside, shifts)
        outside_values = sample_array(arrays["amplitude"], outside, shifts)
        contrast = np.mean(inside_values, axis=1) - np.mean(outside_values, axis=1)
        bright = np.percentile(inside_values, 82.0, axis=1)
        score = (
            edge_weight * robust_z(oriented_edge)
            + continuity_weight * robust_z(continuity)
            + contrast_weight * robust_z(contrast)
            + bright_weight * robust_z(bright)
        )
        score[valid_fraction < 0.90] = -1e9
        scene_scores.append(score)
        diagnostics[date] = {
            "score": score, "oriented_edge": oriented_edge,
            "continuity": continuity, "contrast": contrast, "bright": bright,
        }
    fused = np.median(np.stack(scene_scores), axis=0)
    return fused, diagnostics, valid_fraction


def best_registration(initial, evidence, strategy, limit_override=None):
    settings = dict(strategy_settings(strategy))
    if limit_override is not None:
        settings["limit"] = float(limit_override)
    boundary, normals, boundary_weights = boundary_samples(initial, strategy)
    inside, outside = area_samples(initial)
    if len(inside) < 3 or len(outside) < 3:
        return None
    coarse = candidate_grid(settings["limit"], 2.0)
    coarse_score, _, _ = score_candidates(coarse, boundary, normals, boundary_weights, inside, outside, evidence, strategy)
    if not np.isfinite(coarse_score).any() or float(np.max(coarse_score)) <= -1e8:
        return None
    coarse_best = coarse[int(np.argmax(coarse_score))]
    fine = candidate_grid(settings["limit"], 0.25, center=tuple(coarse_best), half_window=2.0)
    score, diagnostics, valid_fraction = score_candidates(fine, boundary, normals, boundary_weights, inside, outside, evidence, strategy)
    best_index = int(np.argmax(score))
    best = fine[best_index]
    separation = np.hypot(fine[:, 0] - best[0], fine[:, 1] - best[1]) >= 1.5
    margin = float(score[best_index] - np.max(score[separation])) if np.any(separation) else 0.0
    scene_best = {}
    for date, values in diagnostics.items():
        index = int(np.argmax(values["score"]))
        scene_best[date] = fine[index]
    scene_values = list(scene_best.values())
    pairs = [
        (float(np.hypot(*(first - second))), 0.5 * (first + second))
        for first, second in combinations(scene_values, 2)
    ]
    if pairs:
        pair_distance, pair_center = min(pairs, key=lambda item: item[0])
    else:
        # A single-scene run cannot test inter-scene agreement; retain the
        # fused solution while allowing the remaining observability tests to
        # determine acceptance.
        pair_distance, pair_center = 0.0, best
    fused_to_pair = float(np.hypot(*(best - pair_center)))
    best_edges = [float(values["oriented_edge"][best_index]) for values in diagnostics.values()]
    best_continuity = [float(values["continuity"][best_index]) for values in diagnostics.values()]
    boundary_hit = bool(
        abs(float(best[0])) >= settings["limit"] - 0.75
        or abs(float(best[1])) >= settings["limit"] - 0.75
    )
    accepted = bool(
        margin >= settings["margin"]
        and pair_distance <= 4.0
        and fused_to_pair <= 3.0
        and not boundary_hit
        and float(np.median(best_edges)) >= 0.10
        and float(np.median(best_continuity)) >= 0.10
        and valid_fraction[best_index] >= 0.90
    )
    quality = "accepted" if accepted else "rejected_weak_peak"
    if boundary_hit:
        quality = "rejected_search_boundary"
    elif pair_distance > 4.0 or fused_to_pair > 3.0:
        quality = "rejected_scene_inconsistent"
    elif float(np.median(best_edges)) < 0.10 or float(np.median(best_continuity)) < 0.10:
        quality = "rejected_low_observability"
    _, comparison, _ = score_candidates(
        np.asarray([[0.0, 0.0], best], dtype=float),
        boundary, normals, boundary_weights, inside, outside, evidence, strategy,
    )
    feature_values = {}
    for feature in ("oriented_edge", "continuity", "contrast", "bright"):
        initial_value = float(np.median([values[feature][0] for values in comparison.values()]))
        registered_value = float(np.median([values[feature][1] for values in comparison.values()]))
        feature_values[f"initial_{feature}"] = initial_value
        feature_values[f"registered_{feature}"] = registered_value
        feature_values[f"gain_{feature}"] = registered_value - initial_value
    return {
        "dx": float(best[0]), "dy": float(best[1]), "score": float(score[best_index]),
        "score_margin": margin, "pair_distance_px": pair_distance,
        "fused_to_pair_px": fused_to_pair, "edge_strength": float(np.median(best_edges)),
        "edge_continuity": float(np.median(best_continuity)),
        "registration_accepted": int(accepted), "registration_quality": quality,
        "search_limit_px": float(settings["limit"]),
        **feature_values,
        **{f"dx_{date}": float(value[0]) for date, value in scene_best.items()},
        **{f"dy_{date}": float(value[1]) for date, value in scene_best.items()},
    }


def plot_audit(table, output, preview):
    reliable = table[table.registration_accepted == 1].copy()
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2))
    ax = axes[0, 0]
    edge_data = [reliable.initial_oriented_edge.dropna(), reliable.registered_oriented_edge.dropna()]
    ax.boxplot(edge_data, tick_labels=["配准前", "配准后"], widths=0.55, patch_artist=True,
               boxprops={"facecolor": "#76B7B2", "alpha": 0.75}, medianprops={"color": "#111111"})
    ax.set_ylabel("方向一致边缘强度")
    ax.set_title("a  建筑边缘方向匹配", loc="left", fontweight="bold")

    ax = axes[0, 1]
    ax.scatter(reliable.initial_continuity, reliable.registered_continuity, s=10, alpha=0.55, color="#4E79A7", edgecolors="none")
    limit = max(0.25, float(np.nanmax(np.r_[reliable.initial_continuity, reliable.registered_continuity])))
    ax.plot([0, limit], [0, limit], ls="--", lw=0.9, color="#777777")
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="配准前轮廓连续率", ylabel="配准后轮廓连续率")
    ax.set_title("b  轮廓连续率变化", loc="left", fontweight="bold")

    ax = axes[1, 0]
    for strategy, color in STRATEGY_COLORS.items():
        part = reliable[reliable.shape_strategy == strategy]
        ax.scatter(part.dx, part.dy, s=11, alpha=0.58, color=color, edgecolors="none", label=STRATEGY_LABELS[strategy])
    ax.axhline(0, color="#999999", lw=0.7)
    ax.axvline(0, color="#999999", lw=0.7)
    ax.set(xlabel="Range方向校正 / pixel", ylabel="Azimuth方向校正 / pixel")
    ax.set_title("c  自由二维校正向量", loc="left", fontweight="bold")
    ax.legend(fontsize=6.5, ncol=2)

    ax = axes[1, 1]
    values = reliable.perpendicular_residual_px.abs().dropna()
    ax.hist(values, bins=np.arange(0, 18.5, 1.0), color="#9C755F", edgecolor="white")
    ax.axvline(3.0, color="#59A14F", ls="--", lw=1.0, label="高/中置信界线")
    ax.axvline(8.0, color="#F28E2B", ls="--", lw=1.0, label="中/低置信界线")
    ax.set(xlabel="相对高程方向的垂直残差 / pixel", ylabel="建筑数量")
    ax.set_title("d  高度几何一致性", loc="left", fontweight="bold")
    ax.legend(fontsize=6.5)
    fig.suptitle("纯图像特征二维配准的质量审计", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(preview, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_registration(initial, registered, table, image, output, preview):
    accepted = table[table.registration_accepted == 1]
    accepted_ids = set(accepted.clean_id.astype(int))
    image_box = box(0.0, 0.0, float(image.shape[1]), float(image.shape[0]))
    visible = initial[initial.geometry.intersects(image_box)]
    rejected = visible[~visible.clean_id.isin(accepted_ids)]
    reliable = registered[registered.clean_id.isin(accepted_ids)]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0), sharex=True, sharey=True)
    ax = axes[0]
    add_processed_sar(ax, image)
    ax.add_collection(PatchCollection([mpl_patch(g) for g in visible.geometry], facecolor="none", edgecolor="#00B8D4", linewidth=0.42, alpha=0.82))
    ax.set_title("a  图像特征配准前：height投影", loc="left", fontweight="bold")
    ax.text(0.01, 0.99, f"青色：4 m底面+先验建筑高度的屋顶初始投影（{len(visible)}栋与SAR相交）", transform=ax.transAxes, va="top", fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9})
    ax = axes[1]
    add_processed_sar(ax, image)
    if len(rejected):
        ax.add_collection(PatchCollection([mpl_patch(g) for g in rejected.geometry], facecolor="none", edgecolor="#B8B8B8", linewidth=0.27, alpha=0.42))
    counts = accepted.shape_strategy.value_counts()
    handles = [Patch(facecolor="none", edgecolor="#B8B8B8", label="未通过图像质量控制")]
    for strategy, color in STRATEGY_COLORS.items():
        part = reliable[reliable.shape_strategy == strategy]
        if len(part):
            ax.add_collection(PatchCollection([mpl_patch(g) for g in part.geometry], facecolor="none", edgecolor=color, linewidth=0.67, alpha=0.96))
        handles.append(Patch(facecolor="none", edgecolor=color, label=f"{STRATEGY_LABELS[strategy]}（{int(counts.get(strategy, 0))}栋）"))
    ax.set_title("b  完全基于SAR图像特征的二维配准后", loc="left", fontweight="bold")
    ax.text(0.01, 0.99, f"彩色：可靠二维配准（{len(accepted)}栋）｜灰色：保留初始位置", transform=ax.transAxes, va="top", fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9})
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=True, bbox_to_anchor=(0.5, 0.005), fontsize=6.8)
    fig.suptitle("Shapefile height投影的纯图像特征二维局部配准", fontsize=13, y=0.985)
    fig.text(0.5, 0.935, "二维窗口自由搜索；评分仅使用多景SAR边缘方向、轮廓连续性、内外对比和亮散射；无高程方向约束", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.89))
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(preview, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_height_map(buildings, table, output, preview):
    mapped = buildings.to_crs(32651).copy().reset_index(drop=True)
    mapped = mapped.merge(table[["clean_id", "height_accepted", "height_est_m", "height_confidence"]], on="clean_id", validate="one_to_one")
    reliable = mapped[mapped.height_accepted == 1]
    missing = mapped[mapped.height_accepted == 0]
    vmax = max(40.0, float(reliable.height_est_m.quantile(0.98))) if len(reliable) else 40.0
    norm = Normalize(0.0, vmax)
    fig, ax = plt.subplots(figsize=(10.4, 10.0))
    missing.plot(ax=ax, color="#DEDEDE", edgecolor="#C7C7C7", linewidth=0.18)
    for confidence, edgecolor, linewidth, alpha in (
        ("low", "#D55E00", 0.52, 0.78),
        ("medium", "#F0E442", 0.38, 0.90),
        ("high", "white", 0.18, 1.00),
    ):
        part = reliable[reliable.height_confidence == confidence]
        if len(part):
            part.plot(ax=ax, column="height_est_m", cmap="viridis", norm=norm, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha)
    for row in reliable.itertuples():
        point = row.geometry.representative_point()
        ax.text(point.x, point.y, f"{row.height_est_m:.0f}", ha="center", va="center", fontsize=1.65,
                color="white" if norm(row.height_est_m) < 0.58 else "#111111")
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax, fraction=0.030, pad=0.016, extend="max")
    colorbar.set_label("纯图像配准像素偏移估计高度 / m")
    ax.set_title("纯图像特征二维配准后的建筑高度估计", fontsize=13)
    ax.set_xlabel("Easting / m (UTM 51N)")
    ax.set_ylabel("Northing / m (UTM 51N)")
    ax.set_aspect("equal")
    ax.text(0.01, 0.99, "配准完成后才投影到高程敏感方向：H = Δp平行 / s", transform=ax.transAxes, va="top", fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "#AAAAAA", "alpha": 0.94})
    counts = reliable.height_confidence.value_counts()
    handles = [
        Patch(facecolor="#888888", edgecolor="white", label=f"高置信（垂直残差≤3 px，{int(counts.get('high', 0))}栋）"),
        Patch(facecolor="#888888", edgecolor="#F0E442", label=f"中置信（3–8 px，{int(counts.get('medium', 0))}栋）"),
        Patch(facecolor="#888888", edgecolor="#D55E00", label=f"低置信（>8 px，{int(counts.get('low', 0))}栋）"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, facecolor="white", framealpha=0.94, fontsize=6.5)
    ax.text(0.01, 0.01, f"彩色：图像配准后可计算高度（{len(reliable)}栋）｜灰色：无高度（{len(missing)}栋）｜未用height先验填充", transform=ax.transAxes, fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "#AAAAAA", "alpha": 0.94})
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(preview, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    roof_module = load_module("image_only_evidence", resolve(config["inputs"]["roof_evidence_code"]))
    raw_evidence, _, _ = roof_module.load_evidence({**config, "inputs": {**config["inputs"], "rslc_dir": config["inputs"]["rslc_dir"]}})
    evidence, median_image = prepare_evidence(raw_evidence)

    initial = gpd.read_file(ROOT / "results/vectors/all_buildings_shp_height_projection.gpkg").sort_values("clean_id").reset_index(drop=True)
    base_projection = gpd.read_file(ROOT / "results/vectors/building_base_4m_projection.gpkg").sort_values("clean_id").reset_index(drop=True)
    buildings = gpd.read_file(resolve(config["inputs"]["buildings"]), engine="pyogrio").sort_values("clean_id").reset_index(drop=True)
    metric = buildings.to_crs(32651)
    image_box = box(0.0, 0.0, float(median_image.shape[1]), float(median_image.shape[0]))

    results, registered_rows = [], []
    for index, (initial_row, base_row, building_row) in enumerate(zip(initial.itertuples(), base_projection.itertuples(), buildings.itertuples())):
        descriptor = shape_descriptor(metric.geometry.iloc[index])
        initial_geometry = initial_row.geometry
        base_geometry = base_row.geometry
        visible_fraction = float(initial_geometry.intersection(image_box).area / max(initial_geometry.area, 1e-9))
        result = None if visible_fraction < 0.20 else best_registration(initial_geometry, evidence, descriptor["shape_strategy"])
        base = {
            "fid": int(index), "clean_id": int(initial_row.clean_id),
            "source_height_m": float(building_row.height),
            "initial_visible_fraction": visible_fraction, **descriptor,
        }
        if result is None:
            base.update({"registration_accepted": 0, "registration_quality": "outside_or_unidentifiable", "dx": np.nan, "dy": np.nan})
            corrected = None
        else:
            base.update(result)
            corrected = affinity.translate(initial_geometry, xoff=result["dx"], yoff=result["dy"])

        height_accepted, height_est, parallel_px, perpendicular_px, pixels_per_m = 0, np.nan, np.nan, np.nan, np.nan
        height_confidence = "unavailable"
        if corrected is not None:
            prior_height = float(building_row.height)
            dcol = float((initial_geometry.centroid.x - base_geometry.centroid.x) / prior_height)
            drow = float((initial_geometry.centroid.y - base_geometry.centroid.y) / prior_height)
            pixels_per_m = float(math.hypot(dcol, drow))
            offset_col = float(corrected.centroid.x - base_geometry.centroid.x)
            offset_row = float(corrected.centroid.y - base_geometry.centroid.y)
            parallel_px = float((offset_col * dcol + offset_row * drow) / max(pixels_per_m, 1e-9))
            perpendicular_px = float((-offset_col * drow + offset_row * dcol) / max(pixels_per_m, 1e-9))
            height_est = float(parallel_px / max(pixels_per_m, 1e-9))
            height_accepted = int(
                int(base.get("registration_accepted", 0)) == 1
                and 3.0 <= height_est <= 120.0
                and pixels_per_m >= 0.2
            )
            if height_accepted:
                height_confidence = "high" if abs(perpendicular_px) <= 3.0 else "medium" if abs(perpendicular_px) <= 8.0 else "low"
            registered_rows.append({**base, "height_accepted": height_accepted, "height_est_m": height_est if height_accepted else np.nan,
                                    "height_confidence": height_confidence,
                                    "parallel_offset_from_base_px": parallel_px, "perpendicular_residual_px": perpendicular_px,
                                    "pixels_per_m": pixels_per_m, "geometry": corrected})
        base.update({"height_accepted": height_accepted, "height_est_m": height_est if height_accepted else np.nan,
                     "height_confidence": height_confidence,
                     "height_raw_m": height_est, "parallel_offset_from_base_px": parallel_px,
                     "perpendicular_residual_px": perpendicular_px, "pixels_per_m": pixels_per_m,
                     "prior_height_used_as_final_fill": False})
        results.append(base)
        if (index + 1) % 25 == 0 or index + 1 == len(initial):
            print(f"image-feature-only {index + 1}/{len(initial)} registered={sum(r['registration_accepted'] for r in results)} heights={sum(r['height_accepted'] for r in results)}", flush=True)

    table = pd.DataFrame(results)
    registered = gpd.GeoDataFrame(registered_rows, geometry="geometry", crs=None)
    registration_reliable = registered[registered.registration_accepted == 1].copy()
    height_reliable = registered[registered.height_accepted == 1].copy()
    height_high_confidence = registered[registered.height_confidence == "high"].copy()
    table_path = ROOT / "results/tables/image_feature_only_registration_heights.csv"
    table.to_csv(table_path, index=False)
    gpkg = ROOT / "results/vectors/image_feature_only_registration.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    initial.to_file(gpkg, layer="shp_height_initial", driver="GPKG")
    registration_reliable.to_file(gpkg, layer="image_registered_reliable", driver="GPKG")
    height_reliable.to_file(gpkg, layer="height_available", driver="GPKG")
    height_high_confidence.to_file(gpkg, layer="height_high_confidence", driver="GPKG")

    registration_svg = ROOT / "results/picall/正式图件/12_纯影像特征局部配准.svg"
    height_svg = ROOT / "results/picall/正式图件/13_纯影像特征建筑高度图.svg"
    audit_svg = ROOT / "results/picall/正式图件/14_纯影像特征配准审计.svg"
    plot_registration(initial, registered, table, median_image, registration_svg, Path("/tmp/image_feature_registration_qa.png"))
    plot_height_map(buildings, table, height_svg, Path("/tmp/image_feature_height_map_qa.png"))
    plot_audit(table, audit_svg, Path("/tmp/image_feature_registration_audit_qa.png"))
    for output in (registration_svg, height_svg, audit_svg):
        (ROOT / "results/picall/过程图件" / output.name).write_bytes(output.read_bytes())

    heights = table[table.height_accepted == 1]
    summary = {
        "method": "unconstrained_2d_local_registration_from_sar_image_features_then_height_from_parallel_offset",
        "buildings": int(len(table)),
        "registration_reliable": int(table.registration_accepted.sum()),
        "height_reliable": int(table.height_accepted.sum()),
        "height_confidence_counts": {str(k): int(v) for k, v in table.height_confidence.value_counts().items()},
        "registration_quality_counts": {str(k): int(v) for k, v in table.registration_quality.value_counts().items()},
        "height_mean_m": float(heights.height_est_m.mean()) if len(heights) else None,
        "height_median_m": float(heights.height_est_m.median()) if len(heights) else None,
        "height_range_m": [float(heights.height_est_m.min()), float(heights.height_est_m.max())] if len(heights) else None,
        "accepted_median_oriented_edge_gain": float(table.loc[table.registration_accepted == 1, "gain_oriented_edge"].median()),
        "accepted_median_continuity_gain": float(table.loc[table.registration_accepted == 1, "gain_continuity"].median()),
        "registration_uses_height_direction_constraint": False,
        "registration_uses_displacement_prior_penalty": False,
        "height_direction_used_after_registration_only": True,
        "prior_height_used_as_final_fill": False,
        "table": str(table_path), "vector": str(gpkg),
        "registration_svg": str(registration_svg), "height_svg": str(height_svg), "audit_svg": str(audit_svg),
        "external_accuracy_validated": False,
    }
    summary_path = ROOT / "results/tables/image_feature_only_registration_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
