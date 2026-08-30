from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-hybrid-height-map")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from run_pixel_offset_height import ROOT, resolve


def derive_heights(hybrid, baseline, enhanced):
    baseline = baseline.set_index("fid")
    enhanced = enhanced.set_index("fid")
    records = []
    for row in hybrid.itertuples(index=False):
        record = {
            "fid": int(row.fid),
            "clean_id": int(row.clean_id),
            "accepted": int(row.selected),
            "selection_source": str(row.selection_source),
            "selection_reason": str(row.selection_reason),
            "shape_strategy": str(row.shape_strategy),
            "source_height_m": float(row.source_height_m),
            "height_est_m": np.nan,
            "pixels_per_m": np.nan,
            "parallel_offset_from_4m_reference_px": np.nan,
            "total_offset_from_4m_reference_px": np.nan,
            "perpendicular_residual_px": np.nan,
            "dcol_per_m": np.nan,
            "drow_per_m": np.nan,
            "prior_used_as_final_fill": False,
        }
        if int(row.selected) == 1:
            source = enhanced.loc[int(row.fid)] if row.selection_source == "enhanced_shape_adaptive" else baseline.loc[int(row.fid)]
            dcol = float(source.dcol_per_m)
            drow = float(source.drow_per_m)
            pixels_per_m = float(source.pixels_per_m)
            offset_col = float(source.corrected_centroid_col - source.reference_centroid_col)
            offset_row = float(source.corrected_centroid_row - source.reference_centroid_row)
            parallel_px = float((offset_col * dcol + offset_row * drow) / pixels_per_m)
            perpendicular_px = float((-offset_col * drow + offset_row * dcol) / pixels_per_m)
            height = float(source.reference_height_m + parallel_px / pixels_per_m)
            record.update(
                {
                    "height_est_m": height,
                    "pixels_per_m": pixels_per_m,
                    "parallel_offset_from_4m_reference_px": parallel_px,
                    "total_offset_from_4m_reference_px": float(np.hypot(offset_col, offset_row)),
                    "perpendicular_residual_px": perpendicular_px,
                    "dcol_per_m": dcol,
                    "drow_per_m": drow,
                }
            )
        records.append(record)
    return pd.DataFrame(records)


def plot_height_map(buildings, table, output, preview):
    mapped = buildings.to_crs(32651).copy().reset_index(drop=True)
    mapped["fid"] = np.arange(len(mapped), dtype=int)
    mapped = mapped.merge(table, on=["fid", "clean_id"], validate="one_to_one")
    reliable = mapped[mapped.accepted == 1].copy()
    missing = mapped[mapped.accepted == 0].copy()
    vmax = max(40.0, float(reliable.height_est_m.quantile(0.98)))
    norm = Normalize(0.0, vmax)

    fig, ax = plt.subplots(figsize=(10.4, 10.0))
    missing.plot(ax=ax, color="#DEDEDE", edgecolor="#C7C7C7", linewidth=0.18)
    reliable.plot(
        ax=ax, column="height_est_m", cmap="viridis", norm=norm,
        edgecolor="white", linewidth=0.17,
    )
    for row in reliable.itertuples():
        point = row.geometry.representative_point()
        color = "white" if norm(float(row.height_est_m)) < 0.58 else "#111111"
        ax.text(
            point.x, point.y, f"{row.height_est_m:.0f}",
            ha="center", va="center", fontsize=1.65, color=color,
        )
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap="viridis"), ax=ax,
        fraction=0.030, pad=0.016, extend="max",
    )
    colorbar.set_label("像素偏移估计建筑高度 / m")
    ax.set_title("基于吴淞4 m底面与局部校正像素偏移的建筑高度估计", fontsize=13)
    ax.set_xlabel("Easting / m (UTM 51N)")
    ax.set_ylabel("Northing / m (UTM 51N)")
    ax.set_aspect("equal")
    ax.text(
        0.01, 0.99,
        "H = 4 m参考建筑高 + Δp平行 / s；最终屋顶高程 = 吴淞4 m底面 + H",
        transform=ax.transAxes, va="top", fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "#AAAAAA", "alpha": 0.94},
    )
    ax.text(
        0.01, 0.01,
        f"彩色：可靠像素偏移高度（{len(reliable)}栋，标注单位m）｜灰色：无可靠结果（{len(missing)}栋）｜未用先验填充",
        transform=ax.transAxes, fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "#AAAAAA", "alpha": 0.94},
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(preview, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    hybrid = pd.read_csv(ROOT / "results/tables/hybrid_shape_adaptive_local_correction.csv")
    baseline = pd.read_csv(ROOT / "results/tables/shp_height_local_sar_correction.csv")
    enhanced = pd.read_csv(ROOT / "results/tables/shape_adaptive_enhanced_sar_correction.csv")
    table = derive_heights(hybrid, baseline, enhanced)

    selected = table[table.accepted == 1]
    source_selected = hybrid[hybrid.selected == 1].set_index("fid")
    difference = np.abs(
        selected.set_index("fid").height_est_m
        - (source_selected.loc[selected.fid, "corrected_absolute_elevation_m"].to_numpy()
           - float(config["base_elevation_m"]))
    )
    if float(np.nanmax(difference)) > 1e-6:
        raise RuntimeError("Pixel-offset height formula is inconsistent with selected correction geometry")

    table_path = ROOT / "results/tables/hybrid_pixel_offset_building_heights.csv"
    table.to_csv(table_path, index=False)
    buildings = gpd.read_file(resolve(config["inputs"]["buildings"]), engine="pyogrio").reset_index(drop=True)
    output_gdf = buildings.merge(table, on="clean_id", validate="one_to_one")
    gpkg = ROOT / "results/vectors/hybrid_pixel_offset_building_heights.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    output_gdf.to_file(gpkg, layer="pixel_offset_building_heights", driver="GPKG")

    output = ROOT / "results/picall/正式图件/11_混合像素偏移建筑高度图.svg"
    preview = Path("/tmp/hybrid_pixel_offset_height_map_qa.png")
    plot_height_map(buildings, table, output, preview)
    (ROOT / "results/picall/过程图件" / output.name).write_bytes(output.read_bytes())

    reliable = table[table.accepted == 1]
    summary = {
        "method": "height_from_parallel_pixel_offset_relative_to_wusong_4m_base",
        "formula": "building_height_m = roof_elevation_m - base_elevation_m",
        "base_elevation_m": float(config["base_elevation_m"]),
        "buildings": int(len(table)),
        "reliable_heights": int(len(reliable)),
        "missing_heights": int((table.accepted == 0).sum()),
        "prior_height_used_as_final_fill": False,
        "mean_height_m": float(reliable.height_est_m.mean()),
        "median_height_m": float(reliable.height_est_m.median()),
        "minimum_height_m": float(reliable.height_est_m.min()),
        "maximum_height_m": float(reliable.height_est_m.max()),
        "median_pixels_per_m": float(reliable.pixels_per_m.median()),
        "median_parallel_offset_from_4m_reference_px": float(reliable.parallel_offset_from_4m_reference_px.median()),
        "formula_consistency_max_abs_m": float(np.nanmax(difference)),
        "external_accuracy_validated": False,
        "table": str(table_path),
        "vector": str(gpkg),
        "svg": str(output),
    }
    summary_path = ROOT / "results/tables/hybrid_pixel_offset_building_heights_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
