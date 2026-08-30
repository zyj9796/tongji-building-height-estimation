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


def local_peaks(curve: pd.DataFrame) -> pd.DataFrame:
    ordered = curve.sort_values("height_m").reset_index(drop=True)
    scores = ordered["score"].to_numpy(dtype=float)
    keep = np.zeros(len(ordered), dtype=bool)
    if len(ordered):
        keep[0] = len(ordered) == 1 or scores[0] >= scores[1]
        keep[-1] = len(ordered) == 1 or scores[-1] >= scores[-2]
    if len(ordered) > 2:
        keep[1:-1] = (scores[1:-1] >= scores[:-2]) & (scores[1:-1] >= scores[2:])
    return ordered[keep].sort_values("score", ascending=False)


def run(output: Path) -> pd.DataFrame:
    module = load_module()
    config = json.loads((ROOT / "config_v3.json").read_text(encoding="utf-8"))
    table = pd.read_csv(ROOT / "results/tables/roof_only_v4_consensus_audit/roof_only_building_heights.csv")
    curves = pd.read_csv(ROOT / "results/tables/roof_only_v3_full/roof_only_score_curves.csv")
    audit = pd.read_csv(ROOT / "work/v5_joint_conflict_analysis.csv")
    vector = gpd.read_file(
        ROOT / "results/vectors/roof_only_v4_consensus_audit/roof_only_building_heights.gpkg",
        layer="roof_only_heights",
    ).reset_index(drop=True)
    vector["fid"] = np.arange(len(vector), dtype=np.int64)
    lonlat = vector.to_crs("EPSG:4326")
    StrictRadarProjector, clean_ring_lonlat = module.load_shared_projection(config)
    projector = StrictRadarProjector(
        module.resolve(config["inputs"]["rslc_dir"]) / f"{config['master_scene']}.rslc.par"
    )
    base = float(config["base_elevation_m"])
    row_shift = float(config["registration"]["global_row_shift_px"])
    col_shift = float(config["registration"]["global_col_shift_px"])

    accepted = table[table["height_est_m"].notna()]
    current_roofs: dict[int, object] = {}
    rings: dict[int, np.ndarray] = {}
    for row in accepted.itertuples():
        fid = int(row.fid)
        ring = clean_ring_lonlat(np.asarray(lonlat.iloc[fid].geometry.exterior.coords))
        rings[fid] = ring
        current_roofs[fid] = module.project_roofs(
            projector,
            ring,
            base + np.asarray([float(row.height_est_m)]),
            row_shift,
            col_shift,
        )[0]

    rows: list[dict] = []
    conflicts = audit[audit["v5_joint_conflict"] == 1]
    for conflict in conflicts.itertuples():
        fid = int(conflict.fid)
        curve = curves[
            (curves["fid"] == fid)
            & (curves["stage"] == "coarse")
            & np.isfinite(curves["score"])
            & (curves["score"] > -1e8)
            & (curves["roof_coverage_fraction"] >= 0.98)
            & (curves["roof_pixels"] >= 3)
        ].copy()
        peaks = local_peaks(curve).head(5)
        best_score = float(curve["score"].max())
        candidate_heights = peaks["height_m"].to_numpy(dtype=float)
        candidate_roofs = module.project_roofs(
            projector,
            rings[fid],
            base + candidate_heights,
            row_shift,
            col_shift,
        )
        for rank, (peak, roof) in enumerate(zip(peaks.itertuples(), candidate_roofs), start=1):
            overlap = 0.0
            partner = -1
            for other_fid, other_roof in current_roofs.items():
                if other_fid == fid or not roof.intersects(other_roof):
                    continue
                value = roof.intersection(other_roof).area / roof.area if roof.area > 0 else 0.0
                if value > overlap:
                    overlap = float(value)
                    partner = other_fid
            group_deviation = abs(float(peak.height_m) - float(conflict.similar_neighbor_median_height_m))
            rows.append(
                {
                    "fid": fid,
                    "rank": rank,
                    "candidate_height_m": float(peak.height_m),
                    "candidate_score": float(peak.score),
                    "score_loss": best_score - float(peak.score),
                    "candidate_max_overlap_fraction": overlap,
                    "candidate_overlap_partner_fid": partner,
                    "candidate_group_deviation_m": group_deviation,
                    "current_height_m": float(conflict.height_v4_m),
                    "current_max_overlap_fraction": float(conflict.sar_roof_max_overlap_fraction),
                    "current_group_deviation_m": float(conflict.similar_neighbor_deviation_m),
                    "overlap_reduction": float(conflict.sar_roof_max_overlap_fraction) - overlap,
                    "group_deviation_reduction_m": float(conflict.similar_neighbor_deviation_m) - group_deviation,
                }
            )
    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "work/v5_topk_candidate_audit.csv")
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
