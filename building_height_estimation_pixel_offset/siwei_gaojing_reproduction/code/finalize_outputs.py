"""Finalize the Siwei reproduction without rerunning the expensive search."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    buildings = gpd.read_file(config["inputs"]["buildings"], engine="pyogrio")
    if buildings.crs is None:
        raise RuntimeError("Building input has no CRS; refusing to assign one implicitly")
    buildings = buildings.to_crs(4326)
    if not buildings.geometry.is_valid.all():
        buildings.geometry = buildings.geometry.make_valid()

    table_path = ROOT / "results/tables/joint_quantity_quality_building_heights.csv"
    table = pd.read_csv(table_path)
    if not buildings["clean_id"].is_unique or not table["clean_id"].is_unique:
        raise RuntimeError("clean_id must be unique in both buildings and final table")
    final = buildings.merge(table, on="clean_id", how="left", validate="one_to_one")
    accepted = final[final["final_accepted"] == 1].copy()

    output = ROOT / "results/vectors/final_building_heights_wgs84.gpkg"
    if output.exists():
        output.unlink()
    final.to_file(output, layer="all_buildings", driver="GPKG")
    accepted.to_file(output, layer="accepted_heights", driver="GPKG")

    # The upstream figure templates were written for three scenes.  This run
    # uses exactly two compatible SP_527 descending scenes.
    replacements = {
        "三景": "两景",
        "全部1028栋的height字段投影范围": "全部1028栋的4 m+height屋顶投影范围",
        "与SAR相交的height字段投影": "与SAR相交的4 m+height屋顶投影",
        "第06图height初始投影": "4 m底面+先验建筑高度的屋顶初始投影",
        "第06图初始投影": "4 m底面+先验建筑高度的屋顶初始投影",
    }
    for directory in (ROOT / "results/picall/正式图件", ROOT / "results/picall/过程图件"):
        for svg in directory.glob("*.svg"):
            text = svg.read_text(encoding="utf-8")
            for old, new in replacements.items():
                text = text.replace(old, new)
            svg.write_text(text, encoding="utf-8")

    figures = sorted((ROOT / "results/picall/正式图件").glob("*.svg"))
    summary = {
        "reproduction": "Siwei Gaojing same-track SP_527 descending pair",
        "master_scene": config["master_scene"],
        "scenes": config["scenes"],
        "figures": len(figures),
        "files": [path.name for path in figures],
        "buildings": int(len(final)),
        "accepted_heights": int(len(accepted)),
        "geographic_vector": str(output),
        "geographic_crs": str(final.crs),
        "radar_geometry_layers": "pixel row/column coordinates; intentionally have no geographic CRS",
        "external_accuracy_validated": False,
    }
    (ROOT / "results/picall/正式图件_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
