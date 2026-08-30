from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pixel-offset-height")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Polygon as MplPolygon
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import binary_dilation, map_coordinates
from shapely import affinity
from shapely.geometry import Polygon, box, mapping


ROOT = Path(__file__).resolve().parents[1]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans", "Arial", "sans-serif"],
        "svg.fonttype": "none",
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


def resolve(text: str) -> Path:
    return (ROOT / text).resolve()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def center_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = max(1.4826 * mad, float(np.std(values)) * 0.25, 1e-6)
    return center, scale


def fixed_z(values: np.ndarray, stats: tuple[float, float]) -> np.ndarray:
    return np.clip((np.asarray(values, dtype=float) - stats[0]) / stats[1], -4.0, 4.0)


def local_mask(polygon, c0: int, r0: int, width: int, height: int) -> np.ndarray:
    return rasterize(
        [(mapping(polygon), 1)],
        out_shape=(height, width),
        transform=Affine.translation(c0, r0),
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)


def densified_boundary(polygon: Polygon, step: float) -> np.ndarray:
    ring = polygon.exterior
    count = max(8, int(math.ceil(ring.length / max(step, 0.1))))
    distances = np.linspace(0.0, ring.length, count, endpoint=False)
    return np.asarray([(ring.interpolate(float(distance)).x, ring.interpolate(float(distance)).y) for distance in distances])


def project_roof(projector, ring, absolute_height, row_shift, col_shift):
    rows, cols = projector.project_height_grid(ring, np.asarray([absolute_height], dtype=float))
    return Polygon(np.column_stack([cols[0] + col_shift, rows[0] + row_shift])).buffer(0)


def translated_polygon(prior_polygon, height, prior_height, dcol_per_m, drow_per_m, perpendicular_px):
    magnitude = math.hypot(dcol_per_m, drow_per_m)
    pcol, prow = -drow_per_m / magnitude, dcol_per_m / magnitude
    dx = (height - prior_height) * dcol_per_m + perpendicular_px * pcol
    dy = (height - prior_height) * drow_per_m + perpendicular_px * prow
    return affinity.translate(prior_polygon, xoff=dx, yoff=dy), dx, dy


def raw_candidate_metrics(
    prior_polygon,
    prior_height,
    dcol_per_m,
    drow_per_m,
    heights,
    perpendiculars,
    evidence,
    image_shape,
    settings,
):
    rows, cols = image_shape
    extreme_polygons = []
    for height in (float(np.min(heights)), float(np.max(heights))):
        for perpendicular in (float(np.min(perpendiculars)), float(np.max(perpendiculars))):
            extreme_polygons.append(
                translated_polygon(prior_polygon, height, prior_height, dcol_per_m, drow_per_m, perpendicular)[0]
            )
    bounds = np.asarray([polygon.bounds for polygon in extreme_polygons])
    c0 = max(0, int(math.floor(float(bounds[:, 0].min()) - 6.0)))
    c1 = min(cols - 1, int(math.ceil(float(bounds[:, 2].max()) + 6.0)))
    r0 = max(0, int(math.floor(float(bounds[:, 1].min()) - 6.0)))
    r1 = min(rows - 1, int(math.ceil(float(bounds[:, 3].max()) + 6.0)))
    width, height_px = c1 - c0 + 1, r1 - r0 + 1
    if width <= 2 or height_px <= 2:
        return pd.DataFrame()
    crops = {
        date: {key: values[r0 : r1 + 1, c0 : c1 + 1] for key, values in arrays.items()}
        for date, arrays in evidence.items()
    }
    image_box = box(0.0, 0.0, float(cols), float(rows))
    records = []
    for height in heights:
        for perpendicular in perpendiculars:
            polygon, dx, dy = translated_polygon(
                prior_polygon,
                float(height),
                prior_height,
                dcol_per_m,
                drow_per_m,
                float(perpendicular),
            )
            coverage = float(polygon.intersection(image_box).area / polygon.area) if polygon.area > 0 else 0.0
            mask = local_mask(polygon, c0, r0, width, height_px) if coverage > 0 else np.zeros((height_px, width), bool)
            pixels = int(mask.sum())
            row = {
                "height_m": float(height),
                "perpendicular_shift_px": float(perpendicular),
                "translation_col_px": float(dx),
                "translation_row_px": float(dy),
                "roof_pixels": pixels,
                "coverage_fraction": coverage,
            }
            valid = pixels >= int(settings["minimum_roof_pixels"]) and coverage >= float(settings["minimum_coverage_fraction"])
            if not valid:
                for date in evidence:
                    row.update({f"edge_{date}": np.nan, f"contrast_{date}": np.nan, f"bright_{date}": np.nan})
                records.append(row)
                continue
            outside = binary_dilation(mask, iterations=4) & ~binary_dilation(mask, iterations=1)
            boundary = densified_boundary(polygon, float(settings["boundary_sampling_step_px"]))
            sample_rows = boundary[:, 1] - r0
            sample_cols = boundary[:, 0] - c0
            for date, arrays in crops.items():
                amplitude = arrays["amplitude"]
                edge = arrays["edge"]
                edge_value = float(map_coordinates(edge, [sample_rows, sample_cols], order=1, mode="nearest").mean())
                values = amplitude[mask]
                lower, upper = np.percentile(values, (50.0, 90.0))
                trimmed = values[(values >= lower) & (values <= upper)]
                bright = float(trimmed.mean()) if trimmed.size else float(np.median(values))
                outside_mean = float(amplitude[outside].mean()) if np.any(outside) else float(amplitude.mean())
                row.update(
                    {
                        f"edge_{date}": edge_value,
                        f"contrast_{date}": float(values.mean() - outside_mean),
                        f"bright_{date}": bright,
                    }
                )
            records.append(row)
    return pd.DataFrame(records)


def score_metrics(frame, weights, stats=None):
    frame = frame.copy()
    valid = frame[[f"edge_{date}" for date in weights["scenes"]]].notna().all(axis=1).to_numpy()
    if not np.any(valid):
        return frame, None
    if stats is None:
        stats = {}
        for date in weights["scenes"]:
            for feature in ("edge", "contrast", "bright"):
                key = f"{feature}_{date}"
                stats[key] = center_scale(frame.loc[valid, key].to_numpy())
    scene_scores = []
    for date in weights["scenes"]:
        score = (
            float(weights["roof_boundary_edge"]) * fixed_z(frame[f"edge_{date}"], stats[f"edge_{date}"])
            + float(weights["roof_inside_outside_contrast"]) * fixed_z(frame[f"contrast_{date}"], stats[f"contrast_{date}"])
            + float(weights["roof_bright_scatter"]) * fixed_z(frame[f"bright_{date}"], stats[f"bright_{date}"])
            - float(weights["perpendicular_shift_penalty"]) * frame["perpendicular_shift_px"].to_numpy(dtype=float) ** 2
        )
        score[~valid] = -1e9
        frame[f"score_{date}"] = score
        scene_scores.append(score)
    frame["score"] = np.median(np.stack(scene_scores), axis=0)
    return frame, stats


def best_per_height(scored: pd.DataFrame) -> pd.DataFrame:
    indices = scored.groupby("height_m")["score"].idxmax()
    return scored.loc[indices].sort_values("height_m").reset_index(drop=True)


def estimate_one(fid, building, projector, clean_ring, evidence, image_shape, config):
    base = float(config["base_elevation_m"])
    reference_height = float(config["reference_building_height_m"])
    prior = float(building["height"])
    row_shift = float(config["registration"]["global_row_shift_px"])
    col_shift = float(config["registration"]["global_col_shift_px"])
    ring = clean_ring(np.asarray(building.geometry.exterior.coords))
    reference = project_roof(projector, ring, base + reference_height, row_shift, col_shift)
    prior_roof = project_roof(projector, ring, base + prior, row_shift, col_shift)
    calibration_height = prior if abs(prior - reference_height) >= 1.0 else reference_height + 1.0
    calibration_roof = prior_roof if calibration_height == prior else project_roof(
        projector, ring, base + calibration_height, row_shift, col_shift
    )
    denominator = calibration_height - reference_height
    dcol = float((calibration_roof.centroid.x - reference.centroid.x) / denominator)
    drow = float((calibration_roof.centroid.y - reference.centroid.y) / denominator)
    px_per_m = math.hypot(dcol, drow)
    rows, cols = image_shape
    image_box = box(0.0, 0.0, float(cols), float(rows))
    base_result = {
        "fid": int(fid),
        "clean_id": int(building["clean_id"]),
        "height_prior_m": prior,
        "reference_height_m": reference_height,
        "pixels_per_m": px_per_m,
        "dcol_per_m": dcol,
        "drow_per_m": drow,
        "reference_centroid_col": float(reference.centroid.x),
        "reference_centroid_row": float(reference.centroid.y),
        "prior_centroid_col": float(prior_roof.centroid.x),
        "prior_centroid_row": float(prior_roof.centroid.y),
    }
    if px_per_m < 0.2 or reference.intersection(image_box).area / max(reference.area, 1e-9) < 0.98:
        return {**base_result, "height_est_m": np.nan, "accepted": 0, "quality": "outside_or_unidentifiable"}, [], reference, prior_roof, None
    search = config["search"]
    perpendicular_limit = float(config["registration"]["local_perpendicular_correction_limit_px"])
    correction_height_half_window = float(search["maximum_parallel_correction_px"]) / px_per_m
    local_minimum_height = max(float(search["minimum_height_m"]), prior - correction_height_half_window)
    local_maximum_height = min(float(search["maximum_height_m"]), prior + correction_height_half_window)
    coarse_heights = np.arange(
        local_minimum_height,
        local_maximum_height + 0.5 * float(search["coarse_step_m"]),
        float(search["coarse_step_m"]),
    )
    coarse_perpendiculars = np.arange(
        -perpendicular_limit,
        perpendicular_limit + 0.5 * float(search["coarse_perpendicular_step_px"]),
        float(search["coarse_perpendicular_step_px"]),
    )
    coarse_raw = raw_candidate_metrics(
        prior_roof, prior, dcol, drow, coarse_heights, coarse_perpendiculars, evidence, image_shape, search
    )
    if coarse_raw.empty:
        return {**base_result, "height_est_m": np.nan, "accepted": 0, "quality": "outside_or_unidentifiable"}, [], reference, prior_roof, None
    score_settings = {**config["score_weights"], "scenes": config["scenes"]}
    coarse_scored, stats = score_metrics(coarse_raw, score_settings)
    if stats is None or "score" not in coarse_scored.columns:
        return {**base_result, "height_est_m": np.nan, "accepted": 0, "quality": "outside_or_unidentifiable"}, [], reference, prior_roof, None
    coarse_curve = best_per_height(coarse_scored)
    coarse_best = coarse_curve.loc[coarse_curve["score"].idxmax()]
    fine_half = float(search["fine_half_window_m"])
    fine_heights = np.arange(
        max(local_minimum_height, float(coarse_best["height_m"]) - fine_half),
        min(local_maximum_height, float(coarse_best["height_m"]) + fine_half) + 0.5 * float(search["fine_step_m"]),
        float(search["fine_step_m"]),
    )
    fine_perpendiculars = np.arange(
        max(-perpendicular_limit, float(coarse_best["perpendicular_shift_px"]) - 0.5),
        min(perpendicular_limit, float(coarse_best["perpendicular_shift_px"]) + 0.5) + 0.5 * float(search["fine_perpendicular_step_px"]),
        float(search["fine_perpendicular_step_px"]),
    )
    fine_raw = raw_candidate_metrics(
        prior_roof, prior, dcol, drow, fine_heights, fine_perpendiculars, evidence, image_shape, search
    )
    fine_scored, _ = score_metrics(fine_raw, score_settings, stats=stats)
    if "score" not in fine_scored.columns:
        return {**base_result, "height_est_m": np.nan, "accepted": 0, "quality": "outside_or_unidentifiable"}, [], reference, prior_roof, None
    fine_curve = best_per_height(fine_scored)
    best = fine_curve.loc[fine_curve["score"].idxmax()]
    estimated = float(best["height_m"])
    alternative = np.abs(fine_curve["height_m"].to_numpy(dtype=float) - estimated) >= float(
        config["acceptance"]["alternative_separation_m"]
    )
    margin = float(best["score"] - fine_curve.loc[alternative, "score"].max()) if np.any(alternative) else 0.0
    scene_heights = {}
    for date in config["scenes"]:
        scene_best = fine_scored.loc[fine_scored[f"score_{date}"].idxmax()]
        scene_heights[date] = float(scene_best["height_m"])
    scene_values = list(scene_heights.values())
    pairs = [
        (abs(scene_values[a] - scene_values[b]), 0.5 * (scene_values[a] + scene_values[b]))
        for a, b in itertools.combinations(range(len(scene_values)), 2)
    ]
    if not pairs:
        pairs = [(0.0, scene_values[0])]
    pair_range, pair_center = min(pairs, key=lambda item: item[0])
    fused_to_pair = abs(estimated - pair_center)
    acceptance = config["acceptance"]
    height_boundary = estimated <= local_minimum_height + float(acceptance["search_boundary_guard_m"]) or estimated >= local_maximum_height - float(acceptance["search_boundary_guard_m"])
    perpendicular_boundary = abs(float(best["perpendicular_shift_px"])) >= perpendicular_limit - float(
        acceptance["perpendicular_boundary_guard_px"]
    )
    accepted = bool(
        not height_boundary
        and not perpendicular_boundary
        and margin >= float(acceptance["minimum_score_margin"])
        and pair_range <= float(acceptance["maximum_closest_scene_pair_range_m"])
        and fused_to_pair <= float(acceptance["maximum_fused_to_scene_pair_difference_m"])
    )
    corrected, correction_col, correction_row = translated_polygon(
        prior_roof, estimated, prior, dcol, drow, float(best["perpendicular_shift_px"])
    )
    corrected_centroid = corrected.centroid
    total_col = float(corrected_centroid.x - reference.centroid.x)
    total_row = float(corrected_centroid.y - reference.centroid.y)
    projected_height = reference_height + (total_col * dcol + total_row * drow) / max(px_per_m**2, 1e-9)
    quality = "accepted" if accepted else "rejected"
    if height_boundary:
        quality = "rejected_height_boundary"
    elif perpendicular_boundary:
        quality = "rejected_perpendicular_boundary"
    elif margin < float(acceptance["minimum_score_margin"]):
        quality = "rejected_weak_peak"
    elif pair_range > float(acceptance["maximum_closest_scene_pair_range_m"]) or fused_to_pair > float(
        acceptance["maximum_fused_to_scene_pair_difference_m"]
    ):
        quality = "rejected_scene_inconsistent"
    result = {
        **base_result,
        "height_raw_m": projected_height,
        "height_est_m": projected_height if accepted else np.nan,
        "accepted": int(accepted),
        "quality": quality,
        "score": float(best["score"]),
        "score_margin": margin,
        "perpendicular_shift_px": float(best["perpendicular_shift_px"]),
        "prior_correction_col_px": correction_col,
        "prior_correction_row_px": correction_row,
        "total_offset_from_4m_px": math.hypot(total_col, total_row),
        "parallel_offset_from_4m_px": (total_col * dcol + total_row * drow) / px_per_m,
        "corrected_centroid_col": float(corrected_centroid.x),
        "corrected_centroid_row": float(corrected_centroid.y),
        "closest_scene_pair_range_m": pair_range,
        "fused_to_scene_pair_difference_m": fused_to_pair,
        "local_search_minimum_m": local_minimum_height,
        "local_search_maximum_m": local_maximum_height,
        **{f"height_{date}_m": value for date, value in scene_heights.items()},
    }
    curve = fine_curve.copy()
    curve.insert(0, "fid", int(fid))
    curve.insert(1, "clean_id", int(building["clean_id"]))
    return result, curve.to_dict("records"), reference, prior_roof, corrected


def mpl_patch(geometry):
    return MplPolygon(np.asarray(geometry.exterior.coords), closed=True)


def plot_height_map(buildings, table, output):
    mapped = buildings.to_crs(32651).copy()
    mapped["fid"] = np.arange(len(mapped))
    mapped = mapped.merge(table[["fid", "height_est_m"]], on="fid", validate="one_to_one")
    finite = mapped[mapped.height_est_m.notna()]
    missing = mapped[mapped.height_est_m.isna()]
    vmax = max(20.0, float(finite.height_est_m.quantile(0.98))) if len(finite) else 20.0
    norm = Normalize(0, vmax)
    fig, ax = plt.subplots(figsize=(10.2, 10.0))
    missing.plot(ax=ax, color="#D9D9D9", edgecolor="#C4C4C4", linewidth=0.18)
    if len(finite):
        finite.plot(ax=ax, column="height_est_m", cmap="viridis", norm=norm, edgecolor="white", linewidth=0.16)
        for row in finite.itertuples():
            point = row.geometry.representative_point()
            ax.text(point.x, point.y, f"{row.height_est_m:.0f}", ha="center", va="center", fontsize=1.65,
                    color="white" if norm(row.height_est_m) < 0.58 else "#111111")
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax, fraction=0.030, pad=0.016, extend="max")
    cbar.set_label("像素偏移反演高度 / m")
    ax.set_title("基于校正像素偏移的全区域建筑高度估计", fontsize=13)
    ax.set_xlabel("Easting / m (UTM 51N)")
    ax.set_ylabel("Northing / m (UTM 51N)")
    ax.set_aspect("equal")
    ax.text(0.01, 0.01, f"灰色：无可靠高度（{len(missing)}栋）｜彩色：有可靠高度（{len(finite)}栋）",
            transform=ax.transAxes, fontsize=8, bbox={"facecolor":"white","edgecolor":"#AAAAAA","alpha":0.94})
    fig.tight_layout();fig.savefig(output,bbox_inches="tight");plt.close(fig)


def add_sar(ax, amplitude):
    display = np.sqrt(np.clip(amplitude, 0, 1));lo,hi=np.percentile(display,(2,99.5))
    ax.imshow(display,cmap="gray",vmin=lo,vmax=hi,origin="upper",interpolation="nearest",rasterized=True)
    ax.set_xlim(0,amplitude.shape[1]);ax.set_ylim(amplitude.shape[0],0);ax.set_aspect("equal")
    ax.set_xlabel("Range pixel");ax.set_ylabel("Azimuth pixel")


def plot_projection_correction(initial, prior, corrected, table, amplitude, output):
    fig,ax=plt.subplots(figsize=(11.3,8.6));add_sar(ax,amplitude)
    ax.add_collection(PatchCollection([mpl_patch(g) for g in initial.geometry],facecolor="none",edgecolor="#D0D0D0",linewidth=.25,alpha=.55))
    accepted=table[table.accepted==1]
    ids=set(accepted.fid.astype(int));p=prior[prior.fid.isin(ids)];c=corrected[corrected.fid.isin(ids)]
    ax.add_collection(PatchCollection([mpl_patch(g) for g in p.geometry],facecolor="none",edgecolor="#00D5E8",linewidth=.48,alpha=.75))
    ax.add_collection(PatchCollection([mpl_patch(g) for g in c.geometry],facecolor="none",edgecolor="#FFB000",linewidth=.65,alpha=.95))
    joined=accepted.set_index('fid')
    for fid in sorted(ids):
        row=joined.loc[fid]
        ax.plot([row.prior_centroid_col,row.corrected_centroid_col],[row.prior_centroid_row,row.corrected_centroid_row],color="#FFB000",lw=.25,alpha=.65)
    ax.legend(handles=[Patch(facecolor='none',edgecolor='#D0D0D0',label='建筑高度4 m投影'),Patch(facecolor='none',edgecolor='#00D5E8',label='Shapefile先验高度投影'),Patch(facecolor='none',edgecolor='#FFB000',label=f'SAR校正后投影（{len(accepted)}栋）')],loc='lower left',frameon=True,facecolor='white',framealpha=.9)
    ax.set_title('先验高度投影的SAR校正与像素位移',fontsize=13)
    ax.text(.01,.99,'灰：4 m参考｜青：height先验｜橙：校正结果；橙色连线为校正位移',transform=ax.transAxes,va='top',fontsize=7,bbox={'facecolor':'white','edgecolor':'#BBBBBB','alpha':.9})
    fig.tight_layout();fig.savefig(output,bbox_inches='tight');plt.close(fig)


def plot_audit(table, output):
    finite=table[table.accepted==1]
    fig,axes=plt.subplots(2,2,figsize=(10.5,8.0))
    axes[0,0].scatter(finite.height_prior_m,finite.height_est_m,s=10,alpha=.6,color='#4E79A7',edgecolors='none')
    limit=max(60,float(np.nanmax(np.r_[finite.height_prior_m,finite.height_est_m]))+3) if len(finite) else 60
    axes[0,0].plot([0,limit],[0,limit],ls='--',color='#777777',lw=1);axes[0,0].set(xlim=(0,limit),ylim=(0,limit),xlabel='Shapefile height / m',ylabel='像素偏移估计 / m',title='a  先验与反演高度')
    axes[0,1].hist(finite.parallel_offset_from_4m_px,bins=30,color='#76B7B2',edgecolor='white');axes[0,1].set(xlabel='相对4 m投影的平行像素偏移',ylabel='建筑数量',title='b  高度方向像素偏移')
    order=['accepted','rejected_weak_peak','rejected_scene_inconsistent','rejected_height_boundary','rejected_perpendicular_boundary','outside_or_unidentifiable']
    counts=table.quality.value_counts().reindex(order,fill_value=0)
    axes[1,0].barh(range(len(order)),counts,color=['#59A14F','#B07AA1','#E15759','#F28E2B','#EDC948','#BAB0AC']);axes[1,0].set_yticks(range(len(order)),['接受','弱峰','跨景冲突','高度边界','垂直校正边界','范围外/不可辨识']);axes[1,0].invert_yaxis();axes[1,0].set_xscale('symlog',linthresh=10);axes[1,0].set_xlabel('建筑数量');axes[1,0].set_title('c  接受与拒绝原因')
    for i,v in enumerate(counts):axes[1,0].text(v+2,i,str(int(v)),va='center',fontsize=7)
    axes[1,1].scatter(finite.score_margin,finite.closest_scene_pair_range_m,s=10,alpha=.6,color='#F28E2B',edgecolors='none');axes[1,1].axvline(.15,ls='--',color='#777777',lw=1);axes[1,1].axhline(3,ls='--',color='#777777',lw=1);axes[1,1].set(xlabel='峰值间隔',ylabel='最接近两景高度差 / m',title='d  峰值与跨景稳定性')
    fig.suptitle('像素偏移高度反演的全区域质量审计',fontsize=13);fig.tight_layout();fig.savefig(output,bbox_inches='tight');plt.close(fig)


def plot_example(initial, prior, corrected, table, curves, amplitude, output):
    finite = table[(table.accepted == 1) & (table.perpendicular_shift_px.abs() <= 0.5)].copy()
    if finite.empty:
        finite = table[table.accepted == 1].copy()
    target_height = float(finite.height_est_m.median())
    finite["example_rank"] = (finite.height_est_m - target_height).abs() - 0.15 * finite.score_margin
    row = finite.sort_values("example_rank").iloc[0]
    fid = int(row.fid)
    g0 = initial.set_index("fid").loc[fid].geometry
    gp = prior.set_index("fid").loc[fid].geometry
    gc = corrected.set_index("fid").loc[fid].geometry
    bounds = np.asarray([g0.bounds, gp.bounds, gc.bounds])
    c0 = max(0, int(np.floor(bounds[:, 0].min() - 15)))
    c1 = min(amplitude.shape[1], int(np.ceil(bounds[:, 2].max() + 15)))
    r0 = max(0, int(np.floor(bounds[:, 1].min() - 15)))
    r1 = min(amplitude.shape[0], int(np.ceil(bounds[:, 3].max() + 15)))
    crop = np.sqrt(np.clip(amplitude[r0:r1, c0:c1], 0, 1))
    lo, hi = np.percentile(crop, (2, 99.5))
    curve = curves[curves.fid == fid].sort_values("height_m")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axes[0]
    ax.imshow(crop, cmap="gray", vmin=lo, vmax=hi, extent=(c0, c1, r1, r0), interpolation="nearest", rasterized=True)
    for geometry, color, label, linewidth in (
        (g0, "#D9D9D9", "建筑高度4 m", 1.1),
        (gp, "#00D5E8", f"先验 {row.height_prior_m:.1f} m", 1.2),
        (gc, "#FFB000", f"校正后 {row.height_est_m:.1f} m", 1.5),
    ):
        xy = np.asarray(geometry.exterior.coords)
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=linewidth, label=label)
    ax.annotate("", xy=(row.corrected_centroid_col, row.corrected_centroid_row), xytext=(row.prior_centroid_col, row.prior_centroid_row), arrowprops={"arrowstyle": "->", "color": "#FFB000", "lw": 1.2})
    ax.set(xlim=(c0, c1), ylim=(r1, r0), xlabel="Range pixel", ylabel="Azimuth pixel")
    ax.set_aspect("equal");ax.legend(loc="lower left",frameon=True,facecolor="white",framealpha=.88,fontsize=7)
    ax.set_title(f"a  clean_id={int(row.clean_id)} 的投影校正",loc="left",fontweight="bold")
    ax = axes[1]
    scene_columns = [column for column in curve.columns if column.startswith("score_")]
    colors = ("#4E79A7", "#59A14F", "#B07AA1", "#E15759", "#76B7B2")
    for column, color in zip(scene_columns, colors):
        date = column.removeprefix("score_")
        ax.plot(curve.height_m, curve[column], lw=.8, alpha=.75, color=color, label=date)
    ax.plot(curve.height_m, curve.score, color="#111111", lw=1.5, label="多景中位融合")
    ax.axvline(4.0, color="#BAB0AC", ls="--", lw=1, label="4 m参考")
    ax.axvline(float(row.height_prior_m), color="#00AFC4", ls="--", lw=1, label="Shapefile先验")
    ax.axvline(float(row.height_est_m), color="#FF9D00", ls="-", lw=1.2, label="像素偏移估计")
    ax.set(xlabel="由平行像素位移换算的高度 / m", ylabel="SAR匹配得分")
    ax.set_title("b  校正位移搜索曲线",loc="left",fontweight="bold")
    ax.legend(fontsize=6.5,ncol=2)
    fig.suptitle("4 m投影—先验投影—SAR校正—像素偏移高度反演",fontsize=13)
    fig.tight_layout();fig.savefig(output,bbox_inches="tight");plt.close(fig)


def comparison_audit(table, config, output):
    v10 = pd.read_csv(resolve(config["inputs"]["v10_audit_table"]))[["fid", "height_est_m"]].rename(
        columns={"height_est_m": "v10_height_m"}
    )
    strict = pd.read_csv(resolve(config["inputs"]["strict_joint_audit_table"]))[
        ["fid", "height_est_m", "quality", "height_scene_range_m"]
    ].rename(columns={"height_est_m": "strict_height_m", "quality": "strict_quality"})
    audit = table.merge(v10, on="fid", how="left", validate="one_to_one").merge(
        strict, on="fid", how="left", validate="one_to_one"
    )
    audit["strict_reliable"] = audit.strict_quality.isin(["high", "medium"]) & (
        audit.height_scene_range_m <= 5.0
    )
    audit["pixel_offset_to_v10_abs_m"] = (audit.height_est_m - audit.v10_height_m).abs()
    audit["pixel_offset_to_strict_abs_m"] = (audit.height_est_m - audit.strict_height_m).abs()
    audit["v10_to_strict_abs_m"] = (audit.v10_height_m - audit.strict_height_m).abs()
    audit.to_csv(output, index=False)
    accepted = audit[audit.accepted == 1]
    reliable = accepted[accepted.strict_reliable & accepted.strict_height_m.notna()]
    common = reliable[reliable.v10_height_m.notna()]
    return {
        "accepted_with_v10": int(accepted.v10_height_m.notna().sum()),
        "pixel_offset_to_v10_mae_m": float(accepted.pixel_offset_to_v10_abs_m.mean()),
        "reliable_strict_common": int(len(reliable)),
        "pixel_offset_to_reliable_strict_mae_m": float(reliable.pixel_offset_to_strict_abs_m.mean()),
        "pixel_offset_to_reliable_strict_median_abs_m": float(reliable.pixel_offset_to_strict_abs_m.median()),
        "three_method_common": int(len(common)),
        "pixel_offset_common_strict_mae_m": float(common.pixel_offset_to_strict_abs_m.mean()),
        "v10_common_strict_mae_m": float(common.v10_to_strict_abs_m.mean()),
        "accuracy_interpretation": "internal_cross_method_audit_not_external_truth_validation",
    }


def run(config_path: Path, max_buildings: int | None = None):
    config=json.loads(config_path.read_text(encoding='utf-8'))
    roof_module=load_module('roof_evidence_pixel_offset',resolve(config['inputs']['roof_evidence_code']))
    projection=load_module('strict_projection_pixel_offset',resolve(config['inputs']['projection_code']))
    buildings=gpd.read_file(resolve(config['inputs']['buildings']),engine='pyogrio').to_crs(4326).reset_index(drop=True)
    if max_buildings is not None:buildings=buildings.iloc[:max_buildings].copy()
    projector=projection.StrictRadarProjector(resolve(config['inputs']['rslc_dir'])/f"{config['master_scene']}.rslc.par")
    evidence,median_amplitude,_=roof_module.load_evidence({**config,'inputs':{**config['inputs'],'rslc_dir':config['inputs']['rslc_dir']}})
    results=[];curves=[];initial_rows=[];prior_rows=[];corrected_rows=[]
    for fid,building in buildings.iterrows():
        result,curve,initial,prior,corrected=estimate_one(fid,building,projector,projection.clean_ring_lonlat,evidence,median_amplitude.shape,config)
        results.append(result);curves.extend(curve)
        for rows,geometry in ((initial_rows,initial),(prior_rows,prior),(corrected_rows,corrected)):
            if geometry is not None and not geometry.is_empty:rows.append({'fid':fid,'internal_fid':fid,'clean_id':int(building.clean_id),'geometry':geometry})
        if (fid+1)%25==0 or fid+1==len(buildings):print(f"pixel-offset {fid+1}/{len(buildings)} accepted={sum(r['accepted'] for r in results)}",flush=True)
    table=pd.DataFrame(results);curve_table=pd.DataFrame(curves)
    outputs={k:resolve(v) for k,v in config['outputs'].items()}
    for path in outputs.values():path.mkdir(parents=True,exist_ok=True)
    table.to_csv(outputs['tables']/'building_heights.csv',index=False);curve_table.to_csv(outputs['tables']/'score_curves.csv',index=False)
    comparison = comparison_audit(table, config, outputs['tables']/'cross_method_comparison_audit.csv')
    initial_gdf=gpd.GeoDataFrame(initial_rows,geometry='geometry');prior_gdf=gpd.GeoDataFrame(prior_rows,geometry='geometry');corrected_gdf=gpd.GeoDataFrame(corrected_rows,geometry='geometry')
    gpkg=outputs['vectors']/'pixel_offset_projections.gpkg'
    if gpkg.exists():gpkg.unlink()
    initial_gdf.to_file(gpkg,layer='building_height_4m',driver='GPKG');prior_gdf.to_file(gpkg,layer='shp_prior_height',driver='GPKG');corrected_gdf.to_file(gpkg,layer='sar_corrected_prior',driver='GPKG')
    if max_buildings is None:
        geo=gpd.read_file(resolve(config['inputs']['buildings']),engine='pyogrio').reset_index(drop=True)
        plot_height_map(geo,table,outputs['picall']/'01_像素偏移建筑高度图.svg')
        plot_projection_correction(initial_gdf,prior_gdf,corrected_gdf,table,median_amplitude,outputs['picall']/'02_先验投影与合成孔径雷达校正.svg')
        plot_audit(table,outputs['picall']/'03_像素偏移质量审计.svg')
        plot_example(initial_gdf,prior_gdf,corrected_gdf,table,curve_table,median_amplitude,outputs['picall']/'04_像素偏移单体建筑诊断.svg')
        for name in ('01_像素偏移建筑高度图.svg','02_先验投影与合成孔径雷达校正.svg','03_像素偏移质量审计.svg','04_像素偏移单体建筑诊断.svg'):(outputs['figures']/name).write_bytes((outputs['picall']/name).read_bytes())
    finite=table[table.accepted==1]
    summary={'method':'corrected_prior_projection_pixel_offset','buildings':len(table),'accepted':int(len(finite)),'quality_counts':{str(k):int(v) for k,v in table.quality.value_counts().items()},'mean_height_m':float(finite.height_est_m.mean()) if len(finite) else None,'median_height_m':float(finite.height_est_m.median()) if len(finite) else None,'pixels_per_m_median':float(table.pixels_per_m.median()),'prior_height_used_as_final_fill':False,'walls_used':False,'stretch_projection_used':False,'arbitrary_2d_local_shift_used':False,'cross_method_audit':comparison}
    (outputs['tables']/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2));return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'config.json');parser.add_argument('--max-buildings',type=int);args=parser.parse_args();run(args.config.resolve(),args.max_buildings)


if __name__=='__main__':main()
