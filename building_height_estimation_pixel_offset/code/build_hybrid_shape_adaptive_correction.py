from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-hybrid-shape-adaptive")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch
from shapely.geometry import box

from run_pixel_offset_height import ROOT, mpl_patch
from run_shape_adaptive_enhanced_sar_correction import add_processed_sar


def select_hybrid(baseline, enhanced):
    joined = baseline.merge(enhanced, on="fid", suffixes=("_baseline", "_enhanced"), validate="one_to_one")
    rows = []
    for row in joined.itertuples():
        baseline_ok = int(row.accepted_baseline) == 1
        enhanced_ok = int(row.accepted_enhanced) == 1
        source = "rejected"
        reason = "neither_branch_reliable"
        if baseline_ok and not enhanced_ok:
            source, reason = "baseline_raw_sar", "baseline_only_reliable"
        elif enhanced_ok and not baseline_ok:
            source, reason = "enhanced_shape_adaptive", "enhanced_only_reliable"
        elif baseline_ok and enhanced_ok:
            height_difference = abs(
                float(row.corrected_absolute_elevation_m_baseline)
                - float(row.corrected_absolute_elevation_m_enhanced)
            )
            baseline_consistency = float(row.closest_scene_pair_range_m_baseline + row.fused_to_scene_pair_difference_m_baseline)
            enhanced_consistency = float(row.closest_scene_pair_range_m_enhanced + row.fused_to_scene_pair_difference_m_enhanced)
            consistency_difference = abs(baseline_consistency - enhanced_consistency)
            if height_difference > 5.0 and consistency_difference < 1.0:
                source, reason = "rejected", "branch_conflict"
            elif enhanced_consistency < baseline_consistency:
                source, reason = "enhanced_shape_adaptive", "better_multiscene_consistency"
            else:
                source, reason = "baseline_raw_sar", "better_multiscene_consistency"

        prefix = "enhanced" if source == "enhanced_shape_adaptive" else "baseline"
        selected = {
            "fid": int(row.fid),
            "clean_id": int(row.clean_id_baseline),
            "selected": int(source != "rejected"),
            "selection_source": source,
            "selection_reason": reason,
            "shape_strategy": str(row.shape_strategy),
            "footprint_area_m2": float(row.footprint_area_m2),
            "aspect_ratio": float(row.aspect_ratio),
            "source_height_m": float(row.height_prior_m_baseline),
            "corrected_absolute_elevation_m": float(getattr(row, f"corrected_absolute_elevation_m_{prefix}")) if source != "rejected" else np.nan,
            "height_change_m": float(getattr(row, f"height_change_m_{prefix}")) if source != "rejected" else np.nan,
            "correction_magnitude_px": float(getattr(row, f"correction_magnitude_px_{prefix}")) if source != "rejected" else np.nan,
            "correction_col_px": float(getattr(row, f"prior_correction_col_px_{prefix}")) if source != "rejected" else np.nan,
            "correction_row_px": float(getattr(row, f"prior_correction_row_px_{prefix}")) if source != "rejected" else np.nan,
            "closest_scene_pair_range_m": float(getattr(row, f"closest_scene_pair_range_m_{prefix}")) if source != "rejected" else np.nan,
            "fused_to_scene_pair_difference_m": float(getattr(row, f"fused_to_scene_pair_difference_m_{prefix}")) if source != "rejected" else np.nan,
            "score_margin": float(getattr(row, f"score_margin_{prefix}")) if source != "rejected" else np.nan,
            "branch_height_difference_m": abs(float(row.corrected_absolute_elevation_m_baseline) - float(row.corrected_absolute_elevation_m_enhanced)) if baseline_ok and enhanced_ok else np.nan,
        }
        rows.append(selected)
    return pd.DataFrame(rows)


def plot_hybrid(initial, baseline_geometry, enhanced_geometry, selected_geometry, table, image, output, preview):
    image_box = box(0.0, 0.0, float(image.shape[1]), float(image.shape[0]))
    visible = initial[initial.geometry.intersects(image_box)]
    baseline_reliable = baseline_geometry.copy()
    hybrid = table[table.selected == 1]
    baseline_selected = selected_geometry[selected_geometry.selection_source == "baseline_raw_sar"]
    enhanced_selected = selected_geometry[selected_geometry.selection_source == "enhanced_shape_adaptive"]
    conflict_ids = set(table.loc[table.selection_reason == "branch_conflict", "fid"].astype(int))
    conflict_initial = visible[visible.fid.isin(conflict_ids)]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0), sharex=True, sharey=True)
    ax = axes[0]
    add_processed_sar(ax, image)
    if len(baseline_reliable):
        ax.add_collection(PatchCollection([mpl_patch(g) for g in baseline_reliable.geometry], facecolor="none", edgecolor="#0072B2", linewidth=0.60, alpha=0.92))
    ax.set_title("a  上一版：原始SAR局部校正", loc="left", fontweight="bold")
    ax.text(0.01, 0.99, f"蓝色：可靠结果（{len(baseline_reliable)}栋）", transform=ax.transAxes, va="top", fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9})

    ax = axes[1]
    add_processed_sar(ax, image)
    if len(baseline_selected):
        ax.add_collection(PatchCollection([mpl_patch(g) for g in baseline_selected.geometry], facecolor="none", edgecolor="#0072B2", linewidth=0.58, alpha=0.90))
    if len(enhanced_selected):
        ax.add_collection(PatchCollection([mpl_patch(g) for g in enhanced_selected.geometry], facecolor="none", edgecolor="#E69F00", linewidth=0.68, alpha=0.98))
    if len(conflict_initial):
        ax.add_collection(PatchCollection([mpl_patch(g) for g in conflict_initial.geometry], facecolor="none", edgecolor="#CC79A7", linewidth=0.70, alpha=0.95))
    ax.set_title("b  增强SAR与形态策略逐栋择优后", loc="left", fontweight="bold")
    ax.text(0.01, 0.99, f"可靠融合结果：{len(hybrid)}栋｜分支冲突未采用：{len(conflict_ids)}栋", transform=ax.transAxes, va="top", fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9})

    handles = [
        Patch(facecolor="none", edgecolor="#0072B2", label=f"保留原始SAR分支（{len(baseline_selected)}栋）"),
        Patch(facecolor="none", edgecolor="#E69F00", label=f"采用增强形态分支（{len(enhanced_selected)}栋）"),
        Patch(facecolor="none", edgecolor="#CC79A7", label=f"两分支冲突，暂不采用（{len(conflict_ids)}栋）"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("原始SAR与增强形态自适应校正的逐建筑质量择优融合", fontsize=13, y=0.985)
    fig.text(0.5, 0.935, "单分支可靠则采用；双分支可靠时优先多景一致性更好者；高程差>5 m且无明显优势则拒绝", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.89))
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(preview, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    baseline_table = pd.read_csv(ROOT / "results/tables/shp_height_local_sar_correction.csv")
    enhanced_table = pd.read_csv(ROOT / "results/tables/shape_adaptive_enhanced_sar_correction.csv")
    table = select_hybrid(baseline_table, enhanced_table)

    baseline_gpkg = ROOT / "results/vectors/shp_height_local_sar_correction.gpkg"
    enhanced_gpkg = ROOT / "results/vectors/shape_adaptive_enhanced_sar_correction.gpkg"
    initial = gpd.read_file(enhanced_gpkg, layer="shp_height_initial").rename(columns={"internal_fid": "fid"})
    baseline_geometry = gpd.read_file(baseline_gpkg, layer="local_corrected_reliable").rename(columns={"internal_fid": "fid"})
    enhanced_geometry = gpd.read_file(enhanced_gpkg, layer="adaptive_corrected_reliable").rename(columns={"internal_fid": "fid"})
    baseline_by_fid = baseline_geometry.set_index("fid")
    enhanced_by_fid = enhanced_geometry.set_index("fid")

    selected_rows = []
    for row in table[table.selected == 1].itertuples(index=False):
        source = enhanced_by_fid if row.selection_source == "enhanced_shape_adaptive" else baseline_by_fid
        geometry = source.loc[int(row.fid)].geometry
        selected_rows.append({**row._asdict(), "geometry": geometry})
    selected_geometry = gpd.GeoDataFrame(selected_rows, geometry="geometry", crs=None)

    table_path = ROOT / "results/tables/hybrid_shape_adaptive_local_correction.csv"
    table.to_csv(table_path, index=False)
    gpkg = ROOT / "results/vectors/hybrid_shape_adaptive_local_correction.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    initial.to_file(gpkg, layer="shp_height_initial", driver="GPKG")
    selected_geometry.to_file(gpkg, layer="hybrid_corrected_reliable", driver="GPKG")

    processed = np.load(ROOT / "results/processed_sar/building_feature_enhanced_sar.npz")
    image = processed["enhanced_amplitude"]
    output = ROOT / "results/picall/正式图件/10_混合形态自适应局部校正.svg"
    preview = Path("/tmp/hybrid_shape_adaptive_correction_qa.png")
    plot_hybrid(initial, baseline_geometry, enhanced_geometry, selected_geometry, table, image, output, preview)
    (ROOT / "results/picall/过程图件" / output.name).write_bytes(output.read_bytes())

    selected = table[table.selected == 1]
    summary = {
        "method": "per_building_quality_selection_between_raw_and_enhanced_shape_adaptive_branches",
        "buildings": int(len(table)),
        "baseline_reliable": int(len(baseline_geometry)),
        "enhanced_shape_adaptive_reliable": int(len(enhanced_geometry)),
        "hybrid_reliable": int(len(selected)),
        "selected_source_counts": {str(k): int(v) for k, v in selected.selection_source.value_counts().items()},
        "branch_conflicts_rejected": int((table.selection_reason == "branch_conflict").sum()),
        "hybrid_median_correction_px": float(selected.correction_magnitude_px.median()),
        "hybrid_median_abs_height_change_m": float(selected.height_change_m.abs().median()),
        "external_accuracy_validated": False,
        "table": str(table_path),
        "vector": str(gpkg),
        "svg": str(output),
    }
    summary_path = ROOT / "results/tables/hybrid_shape_adaptive_local_correction_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
