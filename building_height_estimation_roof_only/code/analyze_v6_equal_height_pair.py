from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "code" / "run_roof_only_height_search.py"
    spec = importlib.util.spec_from_file_location("roof_only_height_search", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(ids: list[int], output: Path, id_field: str = "clean_id") -> pd.DataFrame:
    module = load_module()
    config = json.loads((ROOT / "config_v3.json").read_text(encoding="utf-8"))
    config["height_search"].update(
        {
            "minimum_m": 3.0,
            "maximum_m": 80.0,
            "symmetric_half_window_minimum_m": 45.0,
            "symmetric_half_window_prior_factor": 0.0,
            "prior_penalty_weight": 0.0,
        }
    )
    StrictRadarProjector, clean_ring_lonlat = module.load_shared_projection(config)
    buildings = gpd.read_file(module.resolve(config["inputs"]["buildings"]), engine="pyogrio").reset_index(drop=True)
    if id_field not in buildings.columns:
        raise ValueError(f"Missing building identifier field: {id_field}")
    if buildings[id_field].duplicated().any():
        raise ValueError(f"Building identifier field is not unique: {id_field}")
    id_to_fid = {int(value): int(fid) for fid, value in enumerate(buildings[id_field])}
    missing_ids = [value for value in ids if value not in id_to_fid]
    if missing_ids:
        raise ValueError(f"Building identifiers not found in {id_field}: {missing_ids}")
    lonlat = buildings.to_crs("EPSG:4326")
    evidence, median_amplitude, median_edge = module.load_evidence(config)
    ps = module.load_ps(module.resolve(config["inputs"]["ps_points"]))
    projector = StrictRadarProjector(
        module.resolve(config["inputs"]["rslc_dir"]) / f"{config['master_scene']}.rslc.par"
    )
    curves: list[pd.DataFrame] = []
    for building_id in ids:
        fid = id_to_fid[building_id]
        ring = clean_ring_lonlat(np.asarray(lonlat.iloc[fid].geometry.exterior.coords))
        prior = float(buildings.iloc[fid]["height"])
        _, rows = module.score_building(
            fid,
            ring,
            prior,
            projector,
            evidence,
            median_amplitude,
            median_edge,
            ps,
            config,
        )
        curve = pd.DataFrame(rows)
        curve = curve[(curve["stage"] == "coarse") & (curve["score"] > -1e8)].copy()
        curves.append(curve[["height_m", "score", *[f"score_{date}" for date in config["scenes"]]]].rename(
            columns={"score": f"score_{id_field}_{building_id}", **{f"score_{date}": f"score_{id_field}_{building_id}_{date}" for date in config["scenes"]}}
        ))
    joint = curves[0]
    for curve in curves[1:]:
        joint = joint.merge(curve, on="height_m", how="inner", validate="one_to_one")
    normalized = []
    for building_id in ids:
        normalized.append(module.robust_z(joint[f"score_{id_field}_{building_id}"].to_numpy(dtype=float)))
        joint[f"normalized_score_{id_field}_{building_id}"] = normalized[-1]
    joint["equal_height_joint_score"] = np.mean(np.stack(normalized), axis=0)
    joint = joint.sort_values("height_m").reset_index(drop=True)
    joint["stage"] = "coarse"
    coarse_best = float(joint.loc[joint["equal_height_joint_score"].idxmax(), "height_m"])

    fine_config = json.loads(json.dumps(config))
    fine_config["height_search"].update(
        {
            "minimum_m": coarse_best - 4.0,
            "maximum_m": coarse_best + 4.0,
            "coarse_step_m": 0.1,
            "fine_step_m": 0.1,
            "fine_half_window_m": 0.5,
        }
    )
    fine_curves: list[pd.DataFrame] = []
    for building_id in ids:
        fid = id_to_fid[building_id]
        ring = clean_ring_lonlat(np.asarray(lonlat.iloc[fid].geometry.exterior.coords))
        prior = float(buildings.iloc[fid]["height"])
        _, rows = module.score_building(
            fid,
            ring,
            prior,
            projector,
            evidence,
            median_amplitude,
            median_edge,
            ps,
            fine_config,
        )
        curve = pd.DataFrame(rows)
        curve = curve[(curve["stage"] == "coarse") & (curve["score"] > -1e8)].copy()
        fine_curves.append(
            curve[["height_m", "score", *[f"score_{date}" for date in config["scenes"]]]].rename(
                columns={
                    "score": f"score_{id_field}_{building_id}",
                    **{f"score_{date}": f"score_{id_field}_{building_id}_{date}" for date in config["scenes"]},
                }
            )
        )
    fine = fine_curves[0]
    for curve in fine_curves[1:]:
        fine = fine.merge(curve, on="height_m", how="inner", validate="one_to_one")
    fine_normalized = []
    for building_id in ids:
        fine_normalized.append(module.robust_z(fine[f"score_{id_field}_{building_id}"].to_numpy(dtype=float)))
        fine[f"normalized_score_{id_field}_{building_id}"] = fine_normalized[-1]
    fine["equal_height_joint_score"] = np.mean(np.stack(fine_normalized), axis=0)
    fine["stage"] = "fine"

    result = pd.concat([joint, fine], ignore_index=True, sort=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    coarse_top = joint.sort_values("equal_height_joint_score", ascending=False).head(5)
    fine_top = fine.sort_values("equal_height_joint_score", ascending=False).head(8)
    columns = ["height_m", *[f"score_{id_field}_{building_id}" for building_id in ids], "equal_height_joint_score"]
    print("COARSE")
    print(coarse_top[columns].to_string(index=False))
    print("FINE")
    print(fine_top[columns].to_string(index=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", type=int, default=[776, 788])
    parser.add_argument("--id-field", default="clean_id")
    parser.add_argument("--output", type=Path, default=ROOT / "work/v6_equal_height_pair_clean_id_776_788.csv")
    args = parser.parse_args()
    run(args.ids, args.output.resolve(), args.id_field)


if __name__ == "__main__":
    main()
