from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-shp-height-local-correction")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch
from shapely.geometry import box

from run_pixel_offset_height import (
    ROOT,
    add_sar,
    estimate_one,
    load_module,
    mpl_patch,
    resolve,
)


def plot_correction(initial, corrected, table, amplitude, output, preview):
    reliable = table[table.accepted == 1].copy()
    reliable_ids = set(reliable.fid.astype(int))
    image = box(0.0, 0.0, float(amplitude.shape[1]), float(amplitude.shape[0]))
    visible_initial = initial[initial.geometry.intersects(image)].copy()
    reliable_corrected = corrected[corrected.fid.isin(reliable_ids)].copy()
    rejected_initial = visible_initial[~visible_initial.fid.isin(reliable_ids)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0), sharex=True, sharey=True)
    ax = axes[0]
    add_sar(ax, amplitude)
    if len(visible_initial):
        ax.add_collection(
            PatchCollection(
                [mpl_patch(g) for g in visible_initial.geometry],
                facecolor="none", edgecolor="#00B8D4", linewidth=0.42, alpha=0.82,
            )
        )
    ax.set_title("a  局部校正前：Shapefile height投影", loc="left", fontweight="bold")
    ax.text(
        0.01, 0.99, f"青色：与SAR相交的初始投影（{len(visible_initial)}栋）",
        transform=ax.transAxes, va="top", fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
    )

    ax = axes[1]
    add_sar(ax, amplitude)
    if len(rejected_initial):
        ax.add_collection(
            PatchCollection(
                [mpl_patch(g) for g in rejected_initial.geometry],
                facecolor="none", edgecolor="#B8B8B8", linewidth=0.30, alpha=0.55,
            )
        )
    if len(reliable_corrected):
        ax.add_collection(
            PatchCollection(
                [mpl_patch(g) for g in reliable_corrected.geometry],
                facecolor="none", edgecolor="#FF9D00", linewidth=0.62, alpha=0.95,
            )
        )
    by_fid = reliable.set_index("fid")
    for fid in sorted(reliable_ids):
        row = by_fid.loc[fid]
        ax.plot(
            [row.prior_centroid_col, row.corrected_centroid_col],
            [row.prior_centroid_row, row.corrected_centroid_row],
            color="#FF9D00", lw=0.26, alpha=0.60,
        )
    ax.set_title("b  逐建筑SAR局部校正后", loc="left", fontweight="bold")
    ax.text(
        0.01, 0.99,
        f"橙色：可靠局部校正（{len(reliable)}栋）｜灰色：证据不足，保留初始位置",
        transform=ax.transAxes, va="top", fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
    )

    handles = [
        Patch(facecolor="none", edgecolor="#00B8D4", label="校正前height投影"),
        Patch(facecolor="none", edgecolor="#FF9D00", label="可靠的逐建筑校正结果"),
        Patch(facecolor="none", edgecolor="#B8B8B8", label="未通过质量控制，保留原位置"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.01))
    if len(reliable):
        med_shift = float(reliable.correction_magnitude_px.median())
        med_height_change = float(reliable.height_change_m.abs().median())
    else:
        med_shift = math.nan
        med_height_change = math.nan
    fig.suptitle("Shapefile height投影的逐建筑SAR局部校正", fontsize=13, y=0.985)
    fig.text(
        0.5, 0.935,
        f"多景RSLC共同评分；每栋独立校正；可靠结果中位位移{med_shift:.2f}像元，"
        f"中位绝对高程改正{med_height_change:.1f} m",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.89))
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(preview, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run(config_path: Path, max_buildings: int | None = None):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    roof_module = load_module("roof_evidence_shp_local", resolve(config["inputs"]["roof_evidence_code"]))
    projection = load_module("strict_projection_shp_local", resolve(config["inputs"]["projection_code"]))
    buildings = (
        gpd.read_file(resolve(config["inputs"]["buildings"]), engine="pyogrio")
        .to_crs(4326)
        .reset_index(drop=True)
    )
    if max_buildings is not None:
        buildings = buildings.iloc[:max_buildings].copy()
    projector = projection.StrictRadarProjector(
        resolve(config["inputs"]["rslc_dir"]) / f"{config['master_scene']}.rslc.par"
    )
    evidence, median_amplitude, _ = roof_module.load_evidence(
        {**config, "inputs": {**config["inputs"], "rslc_dir": config["inputs"]["rslc_dir"]}}
    )

    results = []
    initial_rows = []
    corrected_rows = []
    for fid, building in buildings.iterrows():
        result, _, _, prior, corrected = estimate_one(
            fid, building, projector, projection.clean_ring_lonlat,
            evidence, median_amplitude.shape, config,
        )
        result["corrected_absolute_elevation_m"] = (
            float(config["base_elevation_m"]) + result.get("height_raw_m", np.nan)
        )
        result["height_change_m"] = (
            float(result.get("height_raw_m", np.nan) - result["height_prior_m"])
            if np.isfinite(result.get("height_raw_m", np.nan)) else np.nan
        )
        result["correction_magnitude_px"] = (
            float(math.hypot(result.get("prior_correction_col_px", np.nan), result.get("prior_correction_row_px", np.nan)))
            if np.isfinite(result.get("prior_correction_col_px", np.nan)) else np.nan
        )
        results.append(result)
        initial_rows.append(
            {
                "fid": int(fid), "internal_fid": int(fid),
                "clean_id": int(building.clean_id),
                "source_height_m": float(building["height"]),
                "geometry": prior,
            }
        )
        if corrected is not None and not corrected.is_empty:
            corrected_rows.append(
                {
                    "fid": int(fid), "internal_fid": int(fid),
                    "clean_id": int(building.clean_id),
                    "source_height_m": float(building["height"]),
                    "corrected_absolute_elevation_m": result["corrected_absolute_elevation_m"],
                    "height_change_m": result["height_change_m"],
                    "correction_col_px": result.get("prior_correction_col_px", np.nan),
                    "correction_row_px": result.get("prior_correction_row_px", np.nan),
                    "correction_magnitude_px": result["correction_magnitude_px"],
                    "accepted": int(result["accepted"]),
                    "quality": str(result["quality"]),
                    "score": result.get("score", np.nan),
                    "score_margin": result.get("score_margin", np.nan),
                    "geometry": corrected,
                }
            )
        if (fid + 1) % 25 == 0 or fid + 1 == len(buildings):
            accepted = sum(int(item["accepted"]) for item in results)
            print(f"local-correction {fid + 1}/{len(buildings)} accepted={accepted}", flush=True)

    table = pd.DataFrame(results)
    initial_gdf = gpd.GeoDataFrame(initial_rows, geometry="geometry", crs=None)
    corrected_gdf = gpd.GeoDataFrame(corrected_rows, geometry="geometry", crs=None)
    reliable_gdf = corrected_gdf[corrected_gdf.accepted == 1].copy()

    outputs = {key: resolve(value) for key, value in config["outputs"].items()}
    for path in outputs.values():
        path.mkdir(parents=True, exist_ok=True)
    table_path = outputs["tables"] / "shp_height_local_sar_correction.csv"
    table.to_csv(table_path, index=False)
    gpkg = outputs["vectors"] / "shp_height_local_sar_correction.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    initial_gdf.to_file(gpkg, layer="shp_height_initial", driver="GPKG")
    corrected_gdf.to_file(gpkg, layer="local_best_all_calculable", driver="GPKG")
    reliable_gdf.to_file(gpkg, layer="local_corrected_reliable", driver="GPKG")

    output = outputs["picall"] / "07_矢量高度局部雷达校正.svg"
    preview = Path("/tmp/shp_height_local_sar_correction_qa.png")
    if max_buildings is None:
        plot_correction(initial_gdf, corrected_gdf, table, median_amplitude, output, preview)
        (outputs["figures"] / output.name).write_bytes(output.read_bytes())

    reliable = table[table.accepted == 1]
    summary = {
        "method": "per_building_local_sar_correction_from_shp_height_projection",
        "buildings": int(len(table)),
        "calculable_local_best": int(len(corrected_gdf)),
        "reliable_local_corrections": int(len(reliable)),
        "quality_counts": {str(k): int(v) for k, v in table.quality.value_counts().items()},
        "initial_elevation_source": "Wusong base_elevation_m + Shapefile building height",
        "base_elevation_added": True,
        "base_elevation_m": float(config["base_elevation_m"]),
        "independent_per_building_correction": True,
        "maximum_parallel_correction_px": float(config["search"]["maximum_parallel_correction_px"]),
        "maximum_perpendicular_correction_px": float(config["registration"]["local_perpendicular_correction_limit_px"]),
        "reliable_median_correction_px": float(reliable.correction_magnitude_px.median()) if len(reliable) else None,
        "reliable_median_abs_height_change_m": float(reliable.height_change_m.abs().median()) if len(reliable) else None,
        "reliable_height_change_range_m": [
            float(reliable.height_change_m.min()), float(reliable.height_change_m.max())
        ] if len(reliable) else None,
        "unreliable_results_forced_to_move": False,
        "table": str(table_path),
        "vector": str(gpkg),
        "svg": str(output) if max_buildings is None else None,
    }
    summary_path = outputs["tables"] / "shp_height_local_sar_correction_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--max-buildings", type=int)
    args = parser.parse_args()
    run(args.config.resolve(), args.max_buildings)


if __name__ == "__main__":
    main()
