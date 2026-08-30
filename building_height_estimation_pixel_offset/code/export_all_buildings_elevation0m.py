from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-all-elevation0m")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Polygon as MplPolygon, Rectangle
from shapely.geometry import Polygon, box


ROOT = Path(__file__).resolve().parents[1]


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


def patch(geometry):
    return MplPolygon(np.asarray(geometry.exterior.coords), closed=True)


def project_all(config, projection):
    buildings = gpd.read_file(resolve(config["inputs"]["buildings"]), engine="pyogrio").to_crs(4326).reset_index(drop=True)
    projector = projection.StrictRadarProjector(
        resolve(config["inputs"]["rslc_dir"]) / f"{config['master_scene']}.rslc.par"
    )
    absolute_elevation = 0.0
    row_shift = float(config["registration"]["global_row_shift_px"])
    col_shift = float(config["registration"]["global_col_shift_px"])
    rows = []
    for fid, building in buildings.iterrows():
        ring = projection.clean_ring_lonlat(np.asarray(building.geometry.exterior.coords))
        radar_rows, radar_cols = projector.project_height_grid(ring, np.asarray([absolute_elevation]))
        geometry = Polygon(
            np.column_stack([radar_cols[0] + col_shift, radar_rows[0] + row_shift])
        ).buffer(0)
        rows.append(
            {
                "internal_fid": int(fid),
                "clean_id": int(building.clean_id),
                "projection_absolute_elevation_m": absolute_elevation,
                "centroid_col_px": float(geometry.centroid.x),
                "centroid_row_px": float(geometry.centroid.y),
                "geometry": geometry,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=None)


def classify(frame, width, height):
    image = box(0.0, 0.0, float(width), float(height))
    status = []
    fraction = []
    for geometry in frame.geometry:
        overlap = float(geometry.intersection(image).area / max(geometry.area, 1e-9))
        fraction.append(overlap)
        status.append("inside" if overlap >= 1.0 - 1e-9 else "partial" if overlap > 0 else "outside")
    frame = frame.copy()
    frame["sar_overlap_fraction"] = fraction
    frame["sar_status"] = status
    return frame


def add_outlines(ax, frame, colors, widths, alphas):
    for key in ("inside", "partial", "outside"):
        part = frame[frame.sar_status == key]
        if len(part):
            ax.add_collection(
                PatchCollection(
                    [patch(geometry) for geometry in part.geometry],
                    facecolor="none",
                    edgecolor=colors[key],
                    linewidth=widths[key],
                    alpha=alphas[key],
                )
            )


def plot(frame, amplitude, output, preview=None):
    colors = {"inside": "#00A6C2", "partial": "#F28E2B", "outside": "#B07AA1"}
    widths = {"inside": 0.34, "partial": 0.60, "outside": 0.42}
    alphas = {"inside": 0.90, "partial": 1.00, "outside": 0.82}
    counts = frame.sar_status.value_counts()
    bounds = frame.total_bounds
    padding = 18.0
    xmin, ymin, xmax, ymax = bounds[0] - padding, bounds[1] - padding, bounds[2] + padding, bounds[3] + padding
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2), gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax = axes[0]
    ax.set_facecolor("#F4F4F4")
    ax.add_patch(Rectangle((0, 0), amplitude.shape[1], amplitude.shape[0], facecolor="#D8D8D8", edgecolor="#555555", lw=0.8, zorder=0))
    add_outlines(ax, frame, colors, widths, alphas)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal")
    ax.set_xlabel("Range pixel")
    ax.set_ylabel("Azimuth pixel")
    ax.set_title(f"a  全部{len(frame)}栋的0 m高程投影范围", loc="left", fontweight="bold")
    ax.text(0.01, 0.99, "灰色矩形：SAR影像范围（900×630像元）", transform=ax.transAxes, va="top", fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9})

    ax = axes[1]
    display = np.sqrt(np.clip(amplitude, 0.0, 1.0))
    lo, hi = np.percentile(display, (2.0, 99.5))
    ax.imshow(display, cmap="gray", vmin=lo, vmax=hi, origin="upper", interpolation="nearest", rasterized=True)
    add_outlines(ax, frame[frame.sar_status != "outside"], colors, widths, alphas)
    ax.set_xlim(0, amplitude.shape[1])
    ax.set_ylim(amplitude.shape[0], 0)
    ax.set_aspect("equal")
    ax.set_xlabel("Range pixel")
    ax.set_ylabel("Azimuth pixel")
    ax.set_title("b  与SAR相交的0 m高程投影", loc="left", fontweight="bold")

    handles = [
        Patch(facecolor="none", edgecolor=colors["inside"], label=f"完全位于SAR内（{int(counts.get('inside', 0))}栋）"),
        Patch(facecolor="none", edgecolor=colors["partial"], label=f"部分与SAR相交（{int(counts.get('partial', 0))}栋）"),
        Patch(facecolor="none", edgecolor=colors["outside"], label=f"完全位于SAR外（{int(counts.get('outside', 0))}栋）"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("全部建筑矢量按绝对高程0 m的严格距离—多普勒投影", fontsize=13, y=0.985)
    fig.text(0.5, 0.935, "全部顶点高程固定为0 m；未使用Shapefile height字段；全局配准偏移(+34, -1)像元", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.89))
    fig.savefig(output, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    projection = load_module("elevation0_projection", resolve(config["inputs"]["projection_code"]))
    roof = load_module("elevation0_evidence", resolve(config["inputs"]["roof_evidence_code"]))
    _, amplitude, _ = roof.load_evidence({**config, "inputs": {**config["inputs"], "rslc_dir": config["inputs"]["rslc_dir"]}})
    frame = classify(project_all(config, projection), amplitude.shape[1], amplitude.shape[0])
    vector = ROOT / "results/vectors/all_buildings_elevation0m_projection.gpkg"
    if vector.exists():
        vector.unlink()
    frame.to_file(vector, layer="all_buildings_elevation_0m", driver="GPKG")
    frame.drop(columns="geometry").to_csv(ROOT / "results/tables/all_buildings_elevation0m_projection.csv", index=False)
    output = ROOT / "results/picall/正式图件/05_全部建筑零米高程投影.svg"
    plot(frame, amplitude, output, Path("/tmp/all_buildings_elevation0m_projection_qa.png"))
    (ROOT / "results/picall/过程图件/05_全部建筑零米高程投影.svg").write_bytes(output.read_bytes())
    summary = {
        "buildings": int(len(frame)),
        "projection_absolute_elevation_m": 0.0,
        "height_field_used": False,
        "base_elevation_added": False,
        "sar_status_counts": {str(k): int(v) for k, v in frame.sar_status.value_counts().items()},
        "global_row_shift_px": float(config["registration"]["global_row_shift_px"]),
        "global_col_shift_px": float(config["registration"]["global_col_shift_px"]),
        "walls_used": False,
        "stretch_projection_used": False,
        "local_building_shift_used": False,
        "vector": str(vector),
        "svg": str(output),
    }
    (ROOT / "results/tables/all_buildings_elevation0m_projection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
