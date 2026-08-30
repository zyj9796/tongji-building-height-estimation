"""Project the Wusong 4 m building ground polygons to the master SAR grid."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Polygon as MplPolygon
from shapely.geometry import Polygon, box


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def polygon_patch(geometry):
    return MplPolygon(np.asarray(geometry.exterior.coords), closed=True)


def main() -> None:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    base = float(config["base_elevation_m"])
    if base != 4.0:
        raise ValueError(f"This reproduction requires a Wusong 4 m base, got {base}")
    buildings = gpd.read_file(config["inputs"]["buildings"], engine="pyogrio")
    if buildings.crs is None:
        raise ValueError("building vector has no CRS")
    buildings = buildings.to_crs(4326).reset_index(drop=True)
    if not buildings.geometry.is_valid.all():
        buildings.geometry = buildings.geometry.make_valid()

    projection = load_module("clean_base_projection", Path(config["inputs"]["projection_code"]))
    roof = load_module("clean_base_evidence", Path(config["inputs"]["roof_evidence_code"]))
    projector = projection.StrictRadarProjector(
        Path(config["inputs"]["rslc_dir"]) / f"{config['master_scene']}.rslc.par"
    )
    _, amplitude, _ = roof.load_evidence(
        {**config, "inputs": {**config["inputs"], "rslc_dir": config["inputs"]["rslc_dir"]}}
    )
    image_box = box(0.0, 0.0, float(amplitude.shape[1]), float(amplitude.shape[0]))
    rows = []
    for fid, building in buildings.iterrows():
        ring = projection.clean_ring_lonlat(np.asarray(building.geometry.exterior.coords))
        radar_rows, radar_cols = projector.project_height_grid(ring, np.asarray([base]))
        geometry = Polygon(np.column_stack([radar_cols[0], radar_rows[0]])).buffer(0)
        overlap = float(geometry.intersection(image_box).area / max(geometry.area, 1e-9))
        rows.append({
            "internal_fid": int(fid), "clean_id": int(building.clean_id),
            "base_elevation_wusong_m": base,
            "centroid_col_px": float(geometry.centroid.x),
            "centroid_row_px": float(geometry.centroid.y),
            "sar_overlap_fraction": overlap,
            "sar_status": "inside" if overlap >= 1.0 - 1e-9 else "partial" if overlap > 0 else "outside",
            "geometry": geometry,
        })
    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=None)

    vector = ROOT / "results/vectors/building_base_4m_projection.gpkg"
    if vector.exists():
        vector.unlink()
    frame.to_file(vector, layer="building_base_wusong_4m_sar_pixels", driver="GPKG")
    table = ROOT / "results/tables/building_base_4m_projection.csv"
    frame.drop(columns="geometry").to_csv(table, index=False)

    output_dir = ROOT / "results/clean_workflow/02_base_projection"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "01_图件_459712912978.svg"
    display = np.sqrt(np.clip(amplitude, 0.0, 1.0))
    lo, hi = np.percentile(display, (2.0, 99.5))
    fig, ax = plt.subplots(figsize=(12.0, 7.8))
    ax.imshow(display, cmap="gray", vmin=lo, vmax=hi, origin="upper", interpolation="nearest", rasterized=True)
    colors = {"inside": "#00D5E8", "partial": "#FFB000", "outside": "#999999"}
    counts = frame.sar_status.value_counts()
    for status, color in colors.items():
        part = frame[frame.sar_status == status]
        if len(part):
            ax.add_collection(PatchCollection([polygon_patch(g) for g in part.geometry], facecolor="none", edgecolor=color, linewidth=0.45, alpha=0.9))
    ax.set(xlim=(0, amplitude.shape[1]), ylim=(amplitude.shape[0], 0), xlabel="Range pixel", ylabel="Azimuth pixel")
    ax.set_aspect("equal")
    ax.set_title("建筑底面（吴淞高程4 m）GAMMA严格投影到主SAR影像", fontsize=13)
    ax.legend(handles=[Patch(facecolor="none", edgecolor=colors[k], label=f"{k}: {int(counts.get(k, 0))}栋") for k in colors], loc="lower left", frameon=True)
    ax.text(0.01, 0.99, "青色仅表示4 m建筑底面投影，不表示屋顶，也不表示匹配结果", transform=ax.transAxes, va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#AAAAAA", "alpha": 0.94})
    fig.tight_layout(); fig.savefig(output, bbox_inches="tight"); plt.close(fig)

    summary = {
        "definition": "building ground/base projection",
        "vertical_datum_input": "Wusong elevation",
        "base_elevation_wusong_m": base,
        "gamma_height_conversion": "Wusong + GAMMA EGM96 proxy -> WGS84 ellipsoid height",
        "building_height_field_used": False,
        "buildings": int(len(frame)),
        "sar_status_counts": {str(k): int(v) for k, v in counts.items()},
        "radar_geometry_crs": None,
        "radar_geometry_units": "master SAR range/azimuth pixels",
        "vector": str(vector), "table": str(table), "figure": str(output),
    }
    (ROOT / "results/tables/building_base_4m_projection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
