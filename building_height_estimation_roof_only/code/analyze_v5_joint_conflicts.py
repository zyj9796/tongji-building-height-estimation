from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parents[1]


def load_roof_module():
    path = ROOT / "code" / "run_roof_only_height_search.py"
    spec = importlib.util.spec_from_file_location("roof_only_height_search", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(output: Path) -> pd.DataFrame:
    module = load_roof_module()
    config = json.loads((ROOT / "config_v3.json").read_text(encoding="utf-8"))
    table = pd.read_csv(ROOT / "results/tables/roof_only_v4_consensus_audit/roof_only_building_heights.csv")
    vector = gpd.read_file(
        ROOT / "results/vectors/roof_only_v4_consensus_audit/roof_only_building_heights.gpkg",
        layer="roof_only_heights",
    ).reset_index(drop=True)
    vector["fid"] = np.arange(len(vector), dtype=np.int64)
    lonlat = vector.to_crs("EPSG:4326")
    metric = vector.to_crs("EPSG:32651")
    if not bool(lonlat.geometry.is_valid.all()):
        raise ValueError("Invalid building geometry")

    StrictRadarProjector, clean_ring_lonlat = module.load_shared_projection(config)
    projector = StrictRadarProjector(
        module.resolve(config["inputs"]["rslc_dir"]) / f"{config['master_scene']}.rslc.par"
    )
    base = float(config["base_elevation_m"])
    row_shift = float(config["registration"]["global_row_shift_px"])
    col_shift = float(config["registration"]["global_col_shift_px"])
    accepted = table[table["height_est_m"].notna()].copy()
    fids = accepted["fid"].astype(int).tolist()
    roofs = []
    for fid in fids:
        ring = clean_ring_lonlat(np.asarray(lonlat.iloc[fid].geometry.exterior.coords))
        roof = module.project_roofs(
            projector,
            ring,
            base + np.asarray([float(table.loc[table["fid"] == fid, "height_est_m"].iloc[0])]),
            row_shift,
            col_shift,
        )[0]
        roofs.append(roof)

    tree = STRtree(roofs)
    max_overlap: dict[int, float] = {fid: 0.0 for fid in fids}
    overlap_partner: dict[int, int] = {fid: -1 for fid in fids}
    for index, roof in enumerate(roofs):
        fid = fids[index]
        if roof.is_empty or roof.area <= 0:
            continue
        for other_index in tree.query(roof):
            other_index = int(other_index)
            if other_index == index:
                continue
            overlap = roof.intersection(roofs[other_index]).area / roof.area
            if overlap > max_overlap[fid]:
                max_overlap[fid] = float(overlap)
                overlap_partner[fid] = fids[other_index]

    representative = metric.geometry.representative_point()
    coordinates = np.column_stack([representative.x.to_numpy(), representative.y.to_numpy()])
    spatial_tree = cKDTree(coordinates)
    areas = metric.geometry.area.to_numpy()
    heights = table.set_index("fid")["height_est_m"]
    priors = table.set_index("fid")["height_prior_m"]
    rows: list[dict] = []
    for fid in fids:
        comparable = []
        for other in spatial_tree.query_ball_point(coordinates[fid], 120.0):
            if other == fid or pd.isna(heights.get(other)):
                continue
            if abs(float(priors[other]) - float(priors[fid])) > 3.0:
                continue
            ratio = areas[other] / areas[fid]
            if not 0.67 <= ratio <= 1.50:
                continue
            comparable.append(int(other))
        group_median = float(np.median([heights[item] for item in comparable])) if len(comparable) >= 2 else np.nan
        group_deviation = abs(float(heights[fid]) - group_median) if np.isfinite(group_median) else np.nan
        rows.append(
            {
                "fid": fid,
                "height_v4_m": float(heights[fid]),
                "height_prior_m": float(priors[fid]),
                "sar_roof_max_overlap_fraction": max_overlap[fid],
                "sar_overlap_partner_fid": overlap_partner[fid],
                "similar_neighbor_count": len(comparable),
                "similar_neighbor_median_height_m": group_median,
                "similar_neighbor_deviation_m": group_deviation,
                "similar_neighbor_fids": "|".join(map(str, comparable)),
            }
        )
    audit = pd.DataFrame(rows)
    audit["v5_joint_conflict"] = (
        (audit["sar_roof_max_overlap_fraction"] >= 0.20)
        & (audit["similar_neighbor_count"] >= 2)
        & (audit["similar_neighbor_deviation_m"] > 10.0)
    ).astype(int)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False)
    print(
        json.dumps(
            {
                "accepted_roofs": len(audit),
                "overlap_ge_0_20": int((audit["sar_roof_max_overlap_fraction"] >= 0.20).sum()),
                "similar_group_available": int((audit["similar_neighbor_count"] >= 2).sum()),
                "group_outlier_gt_10m": int((audit["similar_neighbor_deviation_m"] > 10.0).sum()),
                "joint_conflicts": int(audit["v5_joint_conflict"].sum()),
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "work/v5_joint_conflict_analysis.csv")
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
