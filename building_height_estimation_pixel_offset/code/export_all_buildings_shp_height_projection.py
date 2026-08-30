from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-all-shp-height")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from shapely.geometry import Polygon

from export_all_buildings_elevation0m import (
    ROOT,
    add_outlines,
    classify,
    load_module,
    resolve,
)


def project_all(config, projection):
    buildings = (
        gpd.read_file(resolve(config["inputs"]["buildings"]), engine="pyogrio")
        .to_crs(4326)
        .reset_index(drop=True)
    )
    building_heights = buildings["height"].to_numpy(dtype=float)
    elevations = float(config["base_elevation_m"]) + building_heights
    if not np.isfinite(elevations).all():
        bad = buildings.loc[~np.isfinite(elevations), "clean_id"].tolist()
        raise ValueError(f"Shapefile height字段含无效值，clean_id={bad}")

    projector = projection.StrictRadarProjector(
        resolve(config["inputs"]["rslc_dir"]) / f"{config['master_scene']}.rslc.par"
    )
    row_shift = float(config["registration"]["global_row_shift_px"])
    col_shift = float(config["registration"]["global_col_shift_px"])
    rows = []
    for fid, building in buildings.iterrows():
        building_height = float(building["height"])
        absolute_elevation = float(config["base_elevation_m"]) + building_height
        ring = projection.clean_ring_lonlat(np.asarray(building.geometry.exterior.coords))
        radar_rows, radar_cols = projector.project_height_grid(
            ring, np.asarray([absolute_elevation])
        )
        geometry = Polygon(
            np.column_stack([radar_cols[0] + col_shift, radar_rows[0] + row_shift])
        ).buffer(0)
        rows.append(
            {
                "internal_fid": int(fid),
                "clean_id": int(building.clean_id),
                "source_height_m": absolute_elevation,
                "building_height_m": building_height,
                "base_elevation_m": float(config["base_elevation_m"]),
                "projection_absolute_elevation_m": absolute_elevation,
                "centroid_col_px": float(geometry.centroid.x),
                "centroid_row_px": float(geometry.centroid.y),
                "geometry": geometry,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=None)


def plot(frame, amplitude, output, preview=None):
    colors = {"inside": "#00A6C2", "partial": "#F28E2B", "outside": "#B07AA1"}
    widths = {"inside": 0.34, "partial": 0.60, "outside": 0.42}
    alphas = {"inside": 0.90, "partial": 1.00, "outside": 0.82}
    counts = frame.sar_status.value_counts()
    bounds = frame.total_bounds
    padding = 18.0
    xmin, ymin = bounds[0] - padding, bounds[1] - padding
    xmax, ymax = bounds[2] + padding, bounds[3] + padding
    fig, axes = plt.subplots(
        1, 2, figsize=(13.2, 6.2), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    ax = axes[0]
    ax.set_facecolor("#F4F4F4")
    ax.add_patch(
        Rectangle(
            (0, 0), amplitude.shape[1], amplitude.shape[0],
            facecolor="#D8D8D8", edgecolor="#555555", lw=0.8, zorder=0,
        )
    )
    add_outlines(ax, frame, colors, widths, alphas)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal")
    ax.set_xlabel("Range pixel")
    ax.set_ylabel("Azimuth pixel")
    ax.set_title(f"a  全部{len(frame)}栋的4 m+height屋顶投影范围", loc="left", fontweight="bold")
    ax.text(
        0.01, 0.99, "灰色矩形：SAR影像范围（900×630像元）",
        transform=ax.transAxes, va="top", fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
    )

    ax = axes[1]
    display = np.sqrt(np.clip(amplitude, 0.0, 1.0))
    lo, hi = np.percentile(display, (2.0, 99.5))
    ax.imshow(
        display, cmap="gray", vmin=lo, vmax=hi, origin="upper",
        interpolation="nearest", rasterized=True,
    )
    add_outlines(ax, frame[frame.sar_status != "outside"], colors, widths, alphas)
    ax.set_xlim(0, amplitude.shape[1])
    ax.set_ylim(amplitude.shape[0], 0)
    ax.set_aspect("equal")
    ax.set_xlabel("Range pixel")
    ax.set_ylabel("Azimuth pixel")
    ax.set_title("b  与SAR相交的4 m+height屋顶投影", loc="left", fontweight="bold")

    handles = [
        Patch(facecolor="none", edgecolor=colors["inside"], label=f"完全位于SAR内（{int(counts.get('inside', 0))}栋）"),
        Patch(facecolor="none", edgecolor=colors["partial"], label=f"部分与SAR相交（{int(counts.get('partial', 0))}栋）"),
        Patch(facecolor="none", edgecolor=colors["outside"], label=f"完全位于SAR外（{int(counts.get('outside', 0))}栋）"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("全部建筑屋顶的严格距离—多普勒投影", fontsize=13, y=0.985)
    fig.text(
        0.5, 0.935,
        "屋顶绝对高程 = 吴淞4 m建筑底面 + Shapefile离地建筑高度",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.89))
    fig.savefig(output, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    projection = load_module("shp_height_projection", resolve(config["inputs"]["projection_code"]))
    roof = load_module("shp_height_evidence", resolve(config["inputs"]["roof_evidence_code"]))
    _, amplitude, _ = roof.load_evidence(
        {**config, "inputs": {**config["inputs"], "rslc_dir": config["inputs"]["rslc_dir"]}}
    )
    frame = classify(project_all(config, projection), amplitude.shape[1], amplitude.shape[0])

    vector = ROOT / "results/vectors/all_buildings_shp_height_projection.gpkg"
    if vector.exists():
        vector.unlink()
    frame.to_file(vector, layer="all_buildings_shp_height", driver="GPKG")
    table = ROOT / "results/tables/all_buildings_shp_height_projection.csv"
    frame.drop(columns="geometry").to_csv(table, index=False)
    output = ROOT / "results/picall/正式图件/06_全部建筑矢量高度投影.svg"
    preview = Path("/tmp/all_buildings_shp_height_projection_qa.png")
    plot(frame, amplitude, output, preview)
    (ROOT / "results/picall/过程图件/06_全部建筑矢量高度投影.svg").write_bytes(output.read_bytes())

    elevations = frame.projection_absolute_elevation_m
    summary = {
        "buildings": int(len(frame)),
        "projection_elevation_source": "base_elevation_m + Shapefile building height",
        "height_field_used": True,
        "height_as_absolute_elevation": False,
        "base_elevation_added": True,
        "base_elevation_m": float(config["base_elevation_m"]),
        "height_min_m": float(elevations.min()),
        "height_median_m": float(elevations.median()),
        "height_mean_m": float(elevations.mean()),
        "height_max_m": float(elevations.max()),
        "sar_status_counts": {str(k): int(v) for k, v in frame.sar_status.value_counts().items()},
        "global_row_shift_px": float(config["registration"]["global_row_shift_px"]),
        "global_col_shift_px": float(config["registration"]["global_col_shift_px"]),
        "walls_used": False,
        "stretch_projection_used": False,
        "local_building_shift_used": False,
        "vector": str(vector),
        "table": str(table),
        "svg": str(output),
    }
    summary_path = ROOT / "results/tables/all_buildings_shp_height_projection_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
