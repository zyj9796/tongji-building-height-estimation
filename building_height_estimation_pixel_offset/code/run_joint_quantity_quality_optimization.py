from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-joint-quantity-quality")

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
from scipy.spatial import cKDTree
from shapely import affinity
from shapely.geometry import box

from run_image_feature_only_registration import (
    best_registration,
    prepare_evidence,
)
from run_pixel_offset_height import ROOT, load_module, mpl_patch, project_roof, resolve
from run_shape_adaptive_enhanced_sar_correction import add_processed_sar, shape_descriptor


CONFIDENCE_RANK = {"high": 4, "medium": 3, "low": 2, "supplemental": 1, "none": 0}
CONFIDENCE_COLORS = {"high": "#009E73", "medium": "#0072B2", "low": "#E69F00", "supplemental": "#CC79A7"}
CONFIDENCE_LABELS = {"high": "高置信", "medium": "中置信", "low": "低置信", "supplemental": "补充结果"}


def weighted_median(values, weights):
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), 0.5 * weights.sum())])


def spatial_residual_field(control_points, control_values, targets, target_ids, control_ids, neighbors=60):
    tree = cKDTree(control_points)
    k = min(neighbors + 1, len(control_points))
    distances, indices = tree.query(targets, k=k)
    if distances.ndim == 1:
        distances, indices = distances[:, None], indices[:, None]
    predictions = []
    for distance, index, target_id in zip(distances, indices, target_ids):
        keep = control_ids[index] != target_id
        distance, index = distance[keep], index[keep]
        if len(index) == 0:
            predictions.append(0.0)
            continue
        values = control_values[index]
        prediction = weighted_median(values, 1.0 / (distance + 8.0))
        predictions.append(float(np.clip(prediction, -3.0, 3.0)))
    return np.asarray(predictions)


def strict_refine(projector, projection, ring, target, center_height, base_elevation,
                  minimum_building_height, maximum_building_height, row_shift, col_shift):
    if not np.isfinite(center_height) or target is None or target.is_empty:
        return None
    minimum = max(base_elevation + minimum_building_height, center_height - 15.0)
    maximum = min(base_elevation + maximum_building_height, center_height + 15.0)
    coarse = np.arange(minimum, maximum + 0.5, 1.0)

    def evaluate(heights):
        values = []
        geometries = []
        for height in heights:
            geometry = project_roof(projector, ring, float(height), row_shift, col_shift)
            if geometry is None or geometry.is_empty:
                continue
            centroid_distance = geometry.centroid.distance(target.centroid)
            boundary_distance = geometry.hausdorff_distance(target)
            if not np.isfinite(centroid_distance + boundary_distance):
                continue
            values.append(float(centroid_distance + 0.15 * boundary_distance))
            geometries.append(geometry)
        return np.asarray(values), geometries

    coarse_score, _ = evaluate(coarse)
    if len(coarse_score) == 0:
        return None
    coarse_best = float(coarse[int(np.argmin(coarse_score))])
    fine = np.arange(max(minimum, coarse_best - 1.0), min(maximum, coarse_best + 1.0) + 0.05, 0.1)
    fine_score, geometries = evaluate(fine)
    if len(fine_score) == 0:
        return None
    index = int(np.argmin(fine_score))
    roof_elevation = float(fine[index])
    boundary_hit = bool(roof_elevation <= minimum + 0.15 or roof_elevation >= maximum - 0.15)
    return roof_elevation, geometries[index], float(fine_score[index]), boundary_hit


def recover_boundary_candidates(table, initial, buildings_metric, evidence):
    recovered = []
    candidates = table[table.registration_quality == "rejected_search_boundary"]
    limit_by_strategy = {"small": 16.0, "elongated": 24.0, "compact": 24.0, "regular": 24.0, "large": 28.0}
    initial_lookup = initial.set_index("clean_id")
    metric_lookup = buildings_metric.set_index("clean_id")
    for number, row in enumerate(candidates.itertuples(), start=1):
        strategy = str(row.shape_strategy)
        geometry = initial_lookup.loc[int(row.clean_id)].geometry
        result = best_registration(geometry, evidence, strategy, limit_override=limit_by_strategy[strategy])
        if result is not None and int(result["registration_accepted"]) == 1:
            recovered.append(
                {
                    "fid": int(row.fid), "clean_id": int(row.clean_id),
                    "source_height_m": float(row.source_height_m), "shape_strategy": strategy,
                    "registration_source": "adaptive_boundary_rescue", **result,
                    "geometry": affinity.translate(geometry, xoff=result["dx"], yoff=result["dy"]),
                }
            )
        if number % 25 == 0 or number == len(candidates):
            print(f"boundary-rescue {number}/{len(candidates)} recovered={len(recovered)}", flush=True)
    return recovered


def resolve_duplicate_targets(frame, table):
    if len(frame) < 2:
        return set()
    index = frame.sindex
    removed = set()
    attributes = table.set_index("clean_id")
    rows = list(frame.itertuples())
    for i, row in enumerate(rows):
        if row.clean_id in removed:
            continue
        for j in index.query(row.geometry, predicate="intersects"):
            if j <= i:
                continue
            other = rows[int(j)]
            if other.clean_id in removed:
                continue
            intersection = row.geometry.intersection(other.geometry).area
            overlap = intersection / max(min(row.geometry.area, other.geometry.area), 1e-9)
            centroid_distance = row.geometry.centroid.distance(other.geometry.centroid)
            if overlap < 0.65 or centroid_distance > 3.0:
                continue
            a, b = attributes.loc[row.clean_id], attributes.loc[other.clean_id]
            priority_a = (CONFIDENCE_RANK[a.final_confidence], float(a.final_score_margin))
            priority_b = (CONFIDENCE_RANK[b.final_confidence], float(b.final_score_margin))
            removed.add(int(other.clean_id if priority_a >= priority_b else row.clean_id))
    return removed


def plot_registration(initial, final_radar, table, image, output, preview):
    image_box = box(0.0, 0.0, float(image.shape[1]), float(image.shape[0]))
    visible = initial[initial.geometry.intersects(image_box)]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0), sharex=True, sharey=True)
    ax = axes[0]
    add_processed_sar(ax, image)
    ax.add_collection(PatchCollection([mpl_patch(g) for g in visible.geometry], facecolor="none", edgecolor="#00B8D4", linewidth=0.40, alpha=0.78))
    ax.set_title("a  4 m底面+先验建筑高度的屋顶初始投影", loc="left", fontweight="bold")
    ax.text(0.01, 0.99, f"青色：与SAR相交的初始投影（{len(visible)}栋）", transform=ax.transAxes, va="top", fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9})
    ax = axes[1]
    add_processed_sar(ax, image)
    counts = table[table.final_accepted == 1].final_confidence.value_counts()
    handles = []
    for confidence, color in CONFIDENCE_COLORS.items():
        part = final_radar[final_radar.final_confidence == confidence]
        if len(part):
            ax.add_collection(PatchCollection([mpl_patch(g) for g in part.geometry], facecolor="none", edgecolor=color, linewidth=0.68, alpha=0.96))
        handles.append(Patch(facecolor="none", edgecolor=color, label=f"{CONFIDENCE_LABELS[confidence]}（{int(counts.get(confidence, 0))}栋）"))
    ax.set_title("b  扩窗、残差场与严格重投影联合优化后", loc="left", fontweight="bold")
    ax.text(0.01, 0.99, f"彩色：最终可用高度投影（{int(table.final_accepted.sum())}栋）", transform=ax.transAxes, va="top", fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9})
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("数量—质量联合优化后的建筑屋顶投影", fontsize=13, y=0.985)
    fig.text(0.5, 0.935, "二维图像定位 → 空间残差场校正 → 每候选高度严格距离—多普勒重投影 → 多分支与重叠冲突仲裁", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.89))
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(preview, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_height_map(buildings, table, output, preview):
    mapped = buildings.to_crs(32651).merge(table[["clean_id", "final_accepted", "final_height_m", "final_confidence"]], on="clean_id", validate="one_to_one")
    reliable = mapped[mapped.final_accepted == 1]
    missing = mapped[mapped.final_accepted == 0]
    vmax = max(40.0, float(reliable.final_height_m.quantile(0.98)))
    norm = Normalize(0.0, vmax)
    fig, ax = plt.subplots(figsize=(10.4, 10.0))
    missing.plot(ax=ax, color="#DEDEDE", edgecolor="#C7C7C7", linewidth=0.18)
    for confidence in ("supplemental", "low", "medium", "high"):
        part = reliable[reliable.final_confidence == confidence]
        if len(part):
            part.plot(ax=ax, column="final_height_m", cmap="viridis", norm=norm, edgecolor=CONFIDENCE_COLORS[confidence], linewidth=0.42 if confidence != "high" else 0.18, alpha=0.78 if confidence == "supplemental" else 1.0)
    for row in reliable.itertuples():
        point = row.geometry.representative_point()
        ax.text(point.x, point.y, f"{row.final_height_m:.0f}", ha="center", va="center", fontsize=1.65,
                color="white" if norm(row.final_height_m) < 0.58 else "#111111")
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax, fraction=0.030, pad=0.016, extend="max")
    colorbar.set_label("联合优化建筑高度 / m")
    counts = reliable.final_confidence.value_counts()
    handles = [Patch(facecolor="#888888", edgecolor=color, label=f"{CONFIDENCE_LABELS[key]}（{int(counts.get(key, 0))}栋）") for key, color in CONFIDENCE_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", frameon=True, facecolor="white", framealpha=0.94, fontsize=6.5)
    ax.set_title("数量—质量联合优化的建筑高度估计", fontsize=13)
    ax.set_xlabel("Easting / m (UTM 51N)")
    ax.set_ylabel("Northing / m (UTM 51N)")
    ax.set_aspect("equal")
    ax.text(0.01, 0.99, "严格几何优先；多分支一致则升级；旧可靠结果仅作补充；灰色不填充", transform=ax.transAxes, va="top", fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "#AAAAAA", "alpha": 0.94})
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(preview, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    image_table = pd.read_csv(ROOT / "results/tables/image_feature_only_registration_heights.csv")
    hybrid_table = pd.read_csv(ROOT / "results/tables/hybrid_shape_adaptive_local_correction.csv")
    initial = gpd.read_file(ROOT / "results/vectors/all_buildings_shp_height_projection.gpkg").sort_values("clean_id").reset_index(drop=True)
    base_projection = gpd.read_file(ROOT / "results/vectors/building_base_4m_projection.gpkg").sort_values("clean_id").reset_index(drop=True)
    buildings = gpd.read_file(resolve(config["inputs"]["buildings"]), engine="pyogrio").sort_values("clean_id").reset_index(drop=True)
    buildings_metric = buildings.to_crs(32651).copy()
    buildings_metric["clean_id"] = buildings.clean_id.to_numpy()

    roof = load_module("joint_evidence", resolve(config["inputs"]["roof_evidence_code"]))
    projection = load_module("joint_projection", resolve(config["inputs"]["projection_code"]))
    raw_evidence, _, _ = roof.load_evidence({**config, "inputs": {**config["inputs"], "rslc_dir": config["inputs"]["rslc_dir"]}})
    evidence, median_image = prepare_evidence(raw_evidence)
    recovered_rows = recover_boundary_candidates(image_table, initial, buildings_metric, evidence)

    initial_lookup = initial.set_index("clean_id")
    base_lookup = base_projection.set_index("clean_id")
    building_lookup = buildings.set_index("clean_id")
    candidate_rows = []
    for row in image_table[image_table.registration_accepted == 1].itertuples():
        geometry = affinity.translate(initial_lookup.loc[int(row.clean_id)].geometry, xoff=float(row.dx), yoff=float(row.dy))
        candidate_rows.append({**row._asdict(), "registration_source": "image_feature_primary", "geometry": geometry})
    candidate_rows.extend(recovered_rows)
    candidates = gpd.GeoDataFrame(candidate_rows, geometry="geometry", crs=None).drop_duplicates("clean_id", keep="first")

    controls = candidates[candidates.registration_source == "image_feature_primary"].copy()
    control_table = image_table.set_index("clean_id").loc[controls.clean_id]
    control_points = np.asarray([[g.centroid.x, g.centroid.y] for g in controls.geometry])
    control_values = control_table.perpendicular_residual_px.to_numpy(dtype=float)
    target_points = np.asarray([[g.centroid.x, g.centroid.y] for g in candidates.geometry])
    predicted = spatial_residual_field(control_points, control_values, target_points, candidates.clean_id.to_numpy(), controls.clean_id.to_numpy())
    candidates["predicted_perpendicular_field_px"] = predicted

    projector = projection.StrictRadarProjector(resolve(config["inputs"]["rslc_dir"]) / f"{config['master_scene']}.rslc.par")
    base_elevation = float(config["base_elevation_m"])
    minimum_building_height = float(config["search"]["minimum_height_m"])
    maximum_building_height = float(config["search"]["maximum_height_m"])
    row_shift = float(config["registration"]["global_row_shift_px"])
    col_shift = float(config["registration"]["global_col_shift_px"])
    strict_rows = []
    for number, candidate in enumerate(candidates.itertuples(), start=1):
        clean_id = int(candidate.clean_id)
        prior = initial_lookup.loc[clean_id].geometry
        base_geometry = base_lookup.loc[clean_id].geometry
        source_height = float(building_lookup.loc[clean_id, "height"])
        ring = projection.clean_ring_lonlat(np.asarray(building_lookup.loc[clean_id].geometry.exterior.coords))
        if np.isfinite(source_height) and abs(source_height) > 1e-6:
            dcol = float((prior.centroid.x - base_geometry.centroid.x) / source_height)
            drow = float((prior.centroid.y - base_geometry.centroid.y) / source_height)
        else:
            one_meter = project_roof(projector, ring, base_elevation + 1.0, row_shift, col_shift)
            dcol = float(one_meter.centroid.x - base_geometry.centroid.x)
            drow = float(one_meter.centroid.y - base_geometry.centroid.y)
        pixels_per_m = float(math.hypot(dcol, drow))
        if not np.isfinite(pixels_per_m) or pixels_per_m <= 1e-9:
            strict_rows.append({
                "clean_id": clean_id, "strict_roof_elevation_m": np.nan, "strict_height_m": np.nan,
                "strict_residual_px": np.nan, "strict_height_boundary_hit": False,
                "strict_confidence": "none", "registration_source": candidate.registration_source,
                "score_margin": float(getattr(candidate, "score_margin", 0.0)),
                "predicted_perpendicular_field_px": float(candidate.predicted_perpendicular_field_px),
                "geometry": None,
            })
            continue
        pcol, prow = -drow / pixels_per_m, dcol / pixels_per_m
        target = affinity.translate(candidate.geometry, xoff=-candidate.predicted_perpendicular_field_px * pcol, yoff=-candidate.predicted_perpendicular_field_px * prow)
        offset_col = float(target.centroid.x - base_geometry.centroid.x)
        offset_row = float(target.centroid.y - base_geometry.centroid.y)
        center_height = base_elevation + float((offset_col * dcol + offset_row * drow) / max(pixels_per_m**2, 1e-9))
        refined = strict_refine(
            projector, projection, ring, target, center_height, base_elevation,
            minimum_building_height, maximum_building_height, row_shift, col_shift,
        )
        if refined is None:
            strict_rows.append({
                "clean_id": clean_id, "strict_roof_elevation_m": np.nan, "strict_height_m": np.nan,
                "strict_residual_px": np.nan, "strict_height_boundary_hit": False,
                "strict_confidence": "none", "registration_source": candidate.registration_source,
                "score_margin": float(getattr(candidate, "score_margin", 0.0)),
                "predicted_perpendicular_field_px": float(candidate.predicted_perpendicular_field_px),
                "geometry": None,
            })
            continue
        roof_elevation, geometry, residual, height_boundary_hit = refined
        pair_distance = float(getattr(candidate, "pair_distance_px", 99.0))
        if height_boundary_hit:
            confidence = "none"
        elif residual <= 2.0 and pair_distance <= 2.5:
            confidence = "high"
        elif residual <= 4.0:
            confidence = "medium"
        elif residual <= 7.0:
            confidence = "low"
        else:
            confidence = "none"
        strict_rows.append({
            "clean_id": clean_id, "strict_roof_elevation_m": roof_elevation,
            "strict_height_m": roof_elevation - base_elevation,
            "strict_residual_px": residual, "strict_height_boundary_hit": height_boundary_hit,
            "strict_confidence": confidence, "registration_source": candidate.registration_source,
            "score_margin": float(getattr(candidate, "score_margin", 0.0)),
            "predicted_perpendicular_field_px": float(candidate.predicted_perpendicular_field_px),
            "geometry": geometry,
        })
        if number % 25 == 0 or number == len(candidates):
            accepted = sum(CONFIDENCE_RANK[item["strict_confidence"]] >= 2 for item in strict_rows)
            print(f"strict-refine {number}/{len(candidates)} usable={accepted}", flush=True)
    strict = gpd.GeoDataFrame(strict_rows, geometry="geometry", crs=None)

    hybrid_selected = hybrid_table[hybrid_table.selected == 1].set_index("clean_id")
    hybrid_geometry = gpd.read_file(ROOT / "results/vectors/hybrid_shape_adaptive_local_correction.gpkg", layer="hybrid_corrected_reliable").set_index("clean_id")
    strict_lookup = strict.set_index("clean_id")
    final_rows, final_geometries = [], []
    for clean_id in buildings.clean_id.astype(int):
        has_strict = clean_id in strict_lookup.index and strict_lookup.loc[clean_id].strict_confidence != "none"
        has_hybrid = clean_id in hybrid_selected.index
        confidence, source, height, roof_elevation, geometry, agreement = "none", "none", np.nan, np.nan, None, np.nan
        score_margin = 0.0
        if has_strict:
            item = strict_lookup.loc[clean_id]
            confidence, source = item.strict_confidence, "strict_refined"
            height, roof_elevation, geometry = float(item.strict_height_m), float(item.strict_roof_elevation_m), item.geometry
            score_margin = float(item.score_margin)
            if has_hybrid:
                agreement = abs(roof_elevation - float(hybrid_selected.loc[clean_id].corrected_absolute_elevation_m))
                if agreement <= 3.0 and confidence == "medium":
                    confidence = "high"
                elif agreement <= 3.0 and confidence == "low":
                    confidence = "medium"
                elif agreement > 8.0 and confidence == "low":
                    confidence = "supplemental"
        elif has_hybrid:
            item = hybrid_selected.loc[clean_id]
            confidence, source = "supplemental", "legacy_hybrid_supplement"
            roof_elevation = float(item.corrected_absolute_elevation_m)
            height = roof_elevation - base_elevation
            geometry = hybrid_geometry.loc[clean_id].geometry
            score_margin = float(item.score_margin) if np.isfinite(item.score_margin) else 0.0
        if confidence != "none" and (not np.isfinite(height) or height < minimum_building_height):
            confidence, source, height, roof_elevation, geometry = "none", "nonpositive_height_rejected", np.nan, np.nan, None
        accepted = int(confidence != "none")
        final_rows.append({
            "clean_id": clean_id, "final_accepted": accepted, "final_height_m": height,
            "final_roof_elevation_m": roof_elevation, "base_elevation_m": base_elevation,
            "final_confidence": confidence, "final_source": source,
            "branch_height_agreement_m": agreement, "final_score_margin": score_margin,
            "prior_height_used_as_final_fill": False,
        })
        if geometry is not None:
            final_geometries.append({"clean_id": clean_id, "final_confidence": confidence, "final_source": source,
                                     "final_height_m": height, "final_roof_elevation_m": roof_elevation,
                                     "base_elevation_m": base_elevation, "geometry": geometry})
    final_table = pd.DataFrame(final_rows)
    final_radar = gpd.GeoDataFrame(final_geometries, geometry="geometry", crs=None)
    duplicate_removed = resolve_duplicate_targets(final_radar, final_table)
    if duplicate_removed:
        final_table.loc[final_table.clean_id.isin(duplicate_removed), ["final_accepted", "final_height_m", "final_roof_elevation_m"]] = [0, np.nan, np.nan]
        final_table.loc[final_table.clean_id.isin(duplicate_removed), "final_confidence"] = "none"
        final_table.loc[final_table.clean_id.isin(duplicate_removed), "final_source"] = "duplicate_target_rejected"
        final_radar = final_radar[~final_radar.clean_id.isin(duplicate_removed)].copy()

    table_path = ROOT / "results/tables/joint_quantity_quality_building_heights.csv"
    final_table.to_csv(table_path, index=False)
    gpkg = ROOT / "results/vectors/joint_quantity_quality_optimization.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    candidates.to_file(gpkg, layer="image_registration_candidates", driver="GPKG")
    strict.to_file(gpkg, layer="strict_height_refinement", driver="GPKG")
    final_radar.to_file(gpkg, layer="final_height_radar_roofs", driver="GPKG")
    final_buildings = buildings.merge(final_table, on="clean_id", how="left", validate="one_to_one")
    final_buildings.to_file(gpkg, layer="final_building_height_map_wgs84", driver="GPKG")

    registration_svg = ROOT / "results/picall/正式图件/15_数量质量联合配准.svg"
    height_svg = ROOT / "results/picall/正式图件/16_数量质量联合建筑高度图.svg"
    plot_registration(initial, final_radar, final_table, median_image, registration_svg, Path("/tmp/joint_quantity_quality_registration_qa.png"))
    plot_height_map(buildings, final_table, height_svg, Path("/tmp/joint_quantity_quality_height_qa.png"))
    for output in (registration_svg, height_svg):
        (ROOT / "results/picall/过程图件" / output.name).write_bytes(output.read_bytes())

    accepted = final_table[final_table.final_accepted == 1]
    summary = {
        "method": "adaptive_boundary_rescue_spatial_residual_field_strict_reprojection_multibranch_fusion",
        "buildings": int(len(final_table)), "primary_image_registrations": int(image_table.registration_accepted.sum()),
        "boundary_candidates": int((image_table.registration_quality == "rejected_search_boundary").sum()),
        "boundary_recovered": int(len(recovered_rows)), "strict_candidates": int(len(strict)),
        "strict_usable": int((strict.strict_confidence != "none").sum()),
        "final_heights": int(len(accepted)),
        "confidence_counts": {str(k): int(v) for k, v in accepted.final_confidence.value_counts().items()},
        "source_counts": {str(k): int(v) for k, v in accepted.final_source.value_counts().items()},
        "duplicate_targets_rejected": int(len(duplicate_removed)),
        "height_mean_m": float(accepted.final_height_m.mean()), "height_median_m": float(accepted.final_height_m.median()),
        "height_range_m": [float(accepted.final_height_m.min()), float(accepted.final_height_m.max())],
        "height_definition": "building_height_above_ground",
        "base_elevation_m": base_elevation,
        "roof_elevation_to_height": "final_height_m = final_roof_elevation_m - base_elevation_m",
        "search_boundary_solutions_rejected": True,
        "prior_height_used_as_final_fill": False, "external_accuracy_validated": False,
        "table": str(table_path), "vector": str(gpkg), "registration_svg": str(registration_svg), "height_svg": str(height_svg),
    }
    summary_path = ROOT / "results/tables/joint_quantity_quality_optimization_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
