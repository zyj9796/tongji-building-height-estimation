from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-roof-only-projection")

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
from shapely.geometry import Polygon, box


ROOT = Path(__file__).resolve().parents[1]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans", "Arial", "sans-serif"],
        "svg.fonttype": "none",
        "font.size": 8,
        "axes.linewidth": 0.8,
    }
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def polygon_patch(geometry: Polygon) -> MplPolygon:
    return MplPolygon(np.asarray(geometry.exterior.coords), closed=True)


def project_buildings(buildings, table, projector, clean_ring, module, config):
    row_shift = float(config["registration"]["global_row_shift_px"])
    col_shift = float(config["registration"]["global_col_shift_px"])
    base = float(config["base_elevation_m"])
    initial_records = []
    final_records = []
    for fid, row in buildings.iterrows():
        ring = clean_ring(np.asarray(row.geometry.exterior.coords))
        initial = module.project_roofs(
            projector,
            ring,
            np.asarray([base]),
            row_shift,
            col_shift,
        )[0]
        if not initial.is_empty:
            initial_records.append(
                {
                    "fid": int(fid),
                    "clean_id": int(row.clean_id),
                    "building_height_m": 0.0,
                    "absolute_elevation_m": base,
                    "geometry": initial,
                }
            )
        height = table.loc[fid, "height_est_m"]
        if pd.notna(height):
            final = module.project_roofs(
                projector,
                ring,
                np.asarray([base + float(height)]),
                row_shift,
                col_shift,
            )[0]
            if not final.is_empty:
                final_records.append(
                    {
                        "fid": int(fid),
                        "clean_id": int(row.clean_id),
                        "building_height_m": float(height),
                        "absolute_elevation_m": base + float(height),
                        "geometry": final,
                    }
                )
    initial = gpd.GeoDataFrame(initial_records, geometry="geometry", crs=None)
    final = gpd.GeoDataFrame(final_records, geometry="geometry", crs=None)
    return initial, final


def visible(frame: gpd.GeoDataFrame, width: int, height: int) -> gpd.GeoDataFrame:
    image = box(0.0, 0.0, float(width), float(height))
    keep = frame.geometry.intersects(image)
    return frame.loc[keep].copy()


def common_extent(initial, final, width: int, height: int, padding: float = 25.0):
    bounds = np.vstack([initial.total_bounds, final.total_bounds])
    xmin = max(0.0, float(np.nanmin(bounds[:, 0])) - padding)
    ymin = max(0.0, float(np.nanmin(bounds[:, 1])) - padding)
    xmax = min(float(width), float(np.nanmax(bounds[:, 2])) + padding)
    ymax = min(float(height), float(np.nanmax(bounds[:, 3])) + padding)
    return xmin, xmax, ymin, ymax


def add_sar(ax, amplitude: np.ndarray, extent):
    xmin, xmax, ymin, ymax = extent
    c0, c1 = int(np.floor(xmin)), int(np.ceil(xmax))
    r0, r1 = int(np.floor(ymin)), int(np.ceil(ymax))
    crop = amplitude[r0:r1, c0:c1]
    display = np.sqrt(np.clip(crop, 0.0, 1.0))
    low, high = np.percentile(display[np.isfinite(display)], (2.0, 99.5))
    ax.imshow(
        display,
        cmap="gray",
        vmin=float(low),
        vmax=float(high),
        extent=(c0, c1, r1, r0),
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal")
    ax.set_xlabel("Range pixel")
    ax.set_ylabel("Azimuth pixel")


def plot_initial(initial, amplitude, extent, output: Path, preview: Path | None = None):
    fig, ax = plt.subplots(figsize=(11.2, 9.2))
    add_sar(ax, amplitude, extent)
    patches = [polygon_patch(geometry) for geometry in initial.geometry]
    collection = PatchCollection(patches, facecolor="none", edgecolor="#00E5FF", linewidth=0.42, alpha=0.90)
    ax.add_collection(collection)
    ax.legend(
        handles=[Patch(facecolor="none", edgecolor="#00E5FF", linewidth=1.2, label=f"初始顶面：绝对高程4 m（{len(initial)}栋）")],
        loc="lower left",
        frameon=True,
        framealpha=0.90,
        facecolor="white",
    )
    ax.set_title("V10建筑顶面初始严格投影（固定基底绝对高程4 m）", fontsize=13)
    ax.text(
        0.01,
        0.99,
        "三景共注册RSLC中值幅度｜严格距离—多普勒投影｜无墙面、无拉伸、无局部平移",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88},
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_final(initial, final, amplitude, extent, output: Path, preview: Path | None = None):
    fig, ax = plt.subplots(figsize=(11.2, 9.2))
    add_sar(ax, amplitude, extent)
    initial_patches = [polygon_patch(geometry) for geometry in initial.geometry]
    ax.add_collection(
        PatchCollection(initial_patches, facecolor="none", edgecolor="#F0F0F0", linewidth=0.42, alpha=0.82)
    )
    vmax = max(20.0, float(final["building_height_m"].quantile(0.98)))
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = mpl.colormaps["viridis"]
    final_patches = [polygon_patch(geometry) for geometry in final.geometry]
    facecolors = cmap(norm(final["building_height_m"].to_numpy(dtype=float)))
    facecolors[:, 3] = 0.25
    ax.add_collection(PatchCollection(final_patches, facecolor=facecolors, edgecolor="none"))
    edgecolors = cmap(norm(final["building_height_m"].to_numpy(dtype=float)))
    edgecolors[:, 3] = 1.0
    ax.add_collection(PatchCollection(final_patches, facecolor="none", edgecolor=edgecolors, linewidth=0.72))
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.030, pad=0.015, extend="max")
    cbar.set_label("V10估计建筑高度 / m")
    ax.legend(
        handles=[
            Patch(facecolor="none", edgecolor="#F0F0F0", linewidth=1.0, label="初始4 m投影位置"),
            Patch(facecolor="#2A788E", edgecolor="#2A788E", alpha=0.55, label=f"4 m + V10估计高度（{len(final)}栋）"),
        ],
        loc="lower left",
        frameon=True,
        framealpha=0.90,
        facecolor="white",
    )
    ax.set_title("V10选用高度后的建筑顶面严格投影", fontsize=13)
    ax.text(
        0.01,
        0.99,
        "浅灰：初始4 m位置｜彩色：最终屋顶位置｜仅使用有V10高度的建筑",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88},
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(preview: bool = False):
    config = json.loads((ROOT / "config_v3.json").read_text(encoding="utf-8"))
    module = load_module("roof_search_projection_plot", ROOT / "code" / "run_roof_only_height_search.py")
    buildings = gpd.read_file(module.resolve(config["inputs"]["buildings"]), engine="pyogrio").reset_index(drop=True)
    table = pd.read_csv(ROOT / "results/tables/roof_only_v10_scale_adaptive_observability/roof_only_building_heights.csv")
    if len(buildings) != len(table) or not np.array_equal(buildings["clean_id"].to_numpy(), table["clean_id"].to_numpy()):
        raise ValueError("Building order or clean_id does not match V10 table")
    projector_class, clean_ring = module.load_shared_projection(config)
    projector = projector_class(module.resolve(config["inputs"]["rslc_dir"]) / f"{config['master_scene']}.rslc.par")
    _, amplitude, _ = module.load_evidence(config)
    initial, final = project_buildings(buildings, table, projector, clean_ring, module, config)
    rows, cols = amplitude.shape
    initial = visible(initial, cols, rows)
    final = visible(final, cols, rows)
    extent = common_extent(initial, final, cols, rows)

    vector_dir = ROOT / "results/vectors/roof_only_v10_scale_adaptive_observability"
    vector_dir.mkdir(parents=True, exist_ok=True)
    gpkg = vector_dir / "roof_projection_initial_final_radar_pixels.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    initial.to_file(gpkg, layer="initial_absolute_elevation_4m", driver="GPKG")
    final.to_file(gpkg, layer="v10_selected_roofs", driver="GPKG")

    picall = ROOT / "results/picall/正式图件"
    figures = ROOT / "results/picall/过程图件/roof_only_v10_scale_adaptive_observability"
    picall.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    initial_svg = picall / "32_图件_453731326937.svg"
    final_svg = picall / "33_图件_960462484320.svg"
    preview_initial = Path("/tmp/v10_initial_projection_qa.png") if preview else None
    preview_final = Path("/tmp/v10_final_projection_qa.png") if preview else None
    plot_initial(initial, amplitude, extent, initial_svg, preview_initial)
    plot_final(initial, final, amplitude, extent, final_svg, preview_final)
    (figures / "图件_963049724273.svg").write_bytes(initial_svg.read_bytes())
    (figures / "图件_914291961526.svg").write_bytes(final_svg.read_bytes())

    summary = {
        "initial_definition": "strict roof projection at absolute elevation 4 m (building height 0 m)",
        "final_definition": "strict roof projection at absolute elevation 4 m plus V10 estimated building height",
        "initial_visible_buildings": int(len(initial)),
        "final_visible_buildings": int(len(final)),
        "rslc_background": "median normalized amplitude of three coregistered RSLC scenes",
        "global_row_shift_px": float(config["registration"]["global_row_shift_px"]),
        "global_col_shift_px": float(config["registration"]["global_col_shift_px"]),
        "walls_used": False,
        "stretch_projection_used": False,
        "local_building_shift_used": False,
        "extent_range_azimuth_pixels": [float(value) for value in extent],
        "outputs": {"initial_svg": str(initial_svg), "final_svg": str(final_svg), "projection_gpkg": str(gpkg)},
    }
    (ROOT / "results/tables/roof_only_v10_scale_adaptive_observability/roof_projection_initial_final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="Also write temporary PNG previews under /tmp for visual QA")
    args = parser.parse_args()
    run(preview=args.preview)


if __name__ == "__main__":
    main()
