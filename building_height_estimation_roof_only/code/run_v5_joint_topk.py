from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def resolve(text: str) -> Path:
    return (ROOT / text).resolve()


def load_module(name: str, filename: str):
    path = ROOT / "code" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inputs = config["inputs"]
    outputs = {key: resolve(value) for key, value in config["outputs"].items()}
    for key in ("tables", "vectors", "figures"):
        outputs[key].mkdir(parents=True, exist_ok=True)
    outputs["picall_svg"].parent.mkdir(parents=True, exist_ok=True)

    joint_module = load_module("v5_joint_analysis", "analyze_v5_joint_conflicts.py")
    topk_module = load_module("v5_topk_analysis", "explore_v5_topk_candidates.py")
    plot_module = load_module("roof_only_height_search", "run_roof_only_height_search.py")
    joint_module.run(resolve(inputs["joint_audit"]))
    topk_module.run(resolve(inputs["topk_audit"]))

    table = pd.read_csv(resolve(inputs["v4_table"]))
    curves = pd.read_csv(resolve(inputs["v3_curves"]))
    audit = pd.read_csv(resolve(inputs["joint_audit"]))
    candidates = pd.read_csv(resolve(inputs["topk_audit"]))
    vector = gpd.read_file(resolve(inputs["v4_vector"]), layer="roof_only_heights").reset_index(drop=True)
    vector["fid"] = np.arange(len(vector), dtype=np.int64)
    if vector.crs is None or not bool(vector.geometry.is_valid.all()):
        raise ValueError("Invalid or missing V4 vector CRS/geometry")
    if len(table) != len(vector) or table["fid"].nunique() != len(table):
        raise ValueError("V4 table/vector mismatch")

    table["height_v4_input_m"] = table["height_est_m"]
    table["v5_action"] = np.where(table["height_est_m"].notna(), "kept_v4", "not_available_v4")
    table["v5_joint_conflict_flag"] = 0
    for column in [
        "v5_current_overlap_fraction",
        "v5_group_median_height_m",
        "v5_current_group_deviation_m",
        "v5_selected_peak_score_loss",
        "v5_selected_peak_overlap_fraction",
        "v5_selected_peak_group_deviation_m",
    ]:
        table[column] = np.nan

    joint = config["joint_conflict"]
    conflict_audit = audit[
        (audit["sar_roof_max_overlap_fraction"] >= float(joint["minimum_sar_overlap_fraction"]))
        & (audit["similar_neighbor_count"] >= int(joint["minimum_similar_neighbor_count"]))
        & (audit["similar_neighbor_deviation_m"] > float(joint["minimum_group_height_deviation_m"]))
    ].copy()
    conflict_fids = set(conflict_audit["fid"].astype(int))
    table.loc[table["fid"].isin(conflict_fids), "v5_joint_conflict_flag"] = 1

    recovery = config["topk_recovery"]
    changes: list[dict] = []
    for conflict in conflict_audit.itertuples():
        fid = int(conflict.fid)
        index = table.index[table["fid"] == fid][0]
        table.loc[index, "v5_current_overlap_fraction"] = float(conflict.sar_roof_max_overlap_fraction)
        table.loc[index, "v5_group_median_height_m"] = float(conflict.similar_neighbor_median_height_m)
        table.loc[index, "v5_current_group_deviation_m"] = float(conflict.similar_neighbor_deviation_m)
        options = candidates[candidates["fid"] == fid].copy()
        prior = float(table.loc[index, "height_prior_m"])
        strict = float(table.loc[index, "strict_joint_height_m"])
        options["distance_to_prior_or_strict_m"] = np.minimum(
            (options["candidate_height_m"] - prior).abs(),
            (options["candidate_height_m"] - strict).abs(),
        )
        eligible = options[
            (options["score_loss"] <= float(recovery["maximum_score_loss"]))
            & (options["candidate_max_overlap_fraction"] <= float(recovery["maximum_candidate_overlap_fraction"]))
            & (options["overlap_reduction"] >= float(recovery["minimum_overlap_reduction"]))
            & (options["group_deviation_reduction_m"] >= float(recovery["minimum_group_deviation_reduction_m"]))
            & (options["candidate_group_deviation_m"] <= float(recovery["maximum_candidate_group_deviation_m"]))
            & (
                options["distance_to_prior_or_strict_m"]
                <= float(recovery["maximum_candidate_distance_to_prior_or_strict_m"])
            )
        ].copy()
        corrected = False
        if not eligible.empty:
            eligible["joint_objective"] = (
                eligible["score_loss"]
                + eligible["candidate_max_overlap_fraction"]
                + eligible["candidate_group_deviation_m"] / 20.0
            )
            selected = eligible.sort_values("joint_objective").iloc[0]
            height = float(selected["candidate_height_m"])
            if not (
                (curves["fid"] == fid)
                & (curves["stage"] == "coarse")
                & np.isclose(curves["height_m"], height)
            ).any():
                raise AssertionError(f"V5 height {height} for fid {fid} is not a SAR coarse candidate")
            table.loc[index, "height_est_m"] = height
            table.loc[index, "roof_elevation_m"] = float(table.loc[index, "base_elevation_m"]) + height
            table.loc[index, "height_uncertainty_m"] = max(float(table.loc[index, "height_uncertainty_m"]), 1.0)
            table.loc[index, "quality"] = "medium"
            table.loc[index, "quality_raw"] = "medium"
            table.loc[index, "accepted_solution"] = 1
            table.loc[index, "rejection_reason"] = ""
            table.loc[index, "scene_consensus"] = "joint_topk_sar_peak"
            table.loc[index, "solution_source"] = "corrected_v5_joint_topk"
            table.loc[index, "v5_action"] = "corrected_joint_topk_sar_peak"
            table.loc[index, "v5_selected_peak_score_loss"] = float(selected["score_loss"])
            table.loc[index, "v5_selected_peak_overlap_fraction"] = float(
                selected["candidate_max_overlap_fraction"]
            )
            table.loc[index, "v5_selected_peak_group_deviation_m"] = float(
                selected["candidate_group_deviation_m"]
            )
            corrected = True
        if not corrected:
            table.loc[index, "height_est_m"] = np.nan
            table.loc[index, "roof_elevation_m"] = np.nan
            table.loc[index, "accepted_solution"] = 0
            table.loc[index, "quality"] = "rejected_joint_overlap_group_outlier"
            table.loc[index, "rejection_reason"] = "joint_overlap_group_outlier"
            table.loc[index, "solution_source"] = "not_accepted_v5"
            table.loc[index, "v5_action"] = "rejected_joint_overlap_group_outlier"
        changes.append(
            {
                "fid": fid,
                "height_prior_m": prior,
                "strict_joint_height_m": strict,
                "height_v4_m": float(table.loc[index, "height_v4_input_m"]),
                "group_median_height_m": float(conflict.similar_neighbor_median_height_m),
                "current_overlap_fraction": float(conflict.sar_roof_max_overlap_fraction),
                "current_group_deviation_m": float(conflict.similar_neighbor_deviation_m),
                "v5_action": str(table.loc[index, "v5_action"]),
                "height_v5_m": table.loc[index, "height_est_m"],
                "selected_peak_score_loss": table.loc[index, "v5_selected_peak_score_loss"],
                "selected_peak_overlap_fraction": table.loc[index, "v5_selected_peak_overlap_fraction"],
                "selected_peak_group_deviation_m": table.loc[index, "v5_selected_peak_group_deviation_m"],
            }
        )

    table.to_csv(outputs["tables"] / "roof_only_building_heights.csv", index=False)
    pd.DataFrame(changes).to_csv(outputs["tables"] / "roof_only_v5_changes.csv", index=False)
    audit.to_csv(outputs["tables"] / "roof_only_v5_joint_conflict_audit.csv", index=False)
    candidates.to_csv(outputs["tables"] / "roof_only_v5_topk_candidate_audit.csv", index=False)
    geometry = vector[["fid", "geometry"]].merge(table, on="fid", how="left", validate="one_to_one")
    geometry.to_file(outputs["vectors"] / "roof_only_building_heights.gpkg", layer="roof_only_heights", driver="GPKG")

    figure_path = outputs["figures"] / "roof_only_v5_joint_topk_height_map"
    plot_module.plot_simple_height_map(
        vector[["fid", "geometry"]],
        table,
        figure_path,
        title="V5全区建筑高度估计（联合归属与Top-K审计后）",
    )
    outputs["picall_svg"].write_bytes(figure_path.with_suffix(".svg").read_bytes())

    finite = table[table["height_est_m"].notna()].copy()
    strict_common = finite.dropna(subset=["strict_joint_height_m"])
    strict_abs = (strict_common["height_est_m"] - strict_common["strict_joint_height_m"]).abs()
    prior_delta = finite["height_est_m"] - finite["height_prior_m"]
    summary = {
        "method": "v5_strict_roof_overlap_joint_attribution_topk_sar_peak_audit",
        "buildings": int(len(table)),
        "v4_finite_heights": int(table["height_v4_input_m"].notna().sum()),
        "finite_heights": int(len(finite)),
        "joint_conflicts": int(len(conflict_audit)),
        "corrected_joint_topk_sar_peak": int(
            (table["v5_action"] == "corrected_joint_topk_sar_peak").sum()
        ),
        "rejected_joint_overlap_group_outlier": int(
            (table["v5_action"] == "rejected_joint_overlap_group_outlier").sum()
        ),
        "kept_v4_unchanged": int((table["v5_action"] == "kept_v4").sum()),
        "prior_or_neighbor_fill_used": False,
        "height_mean_m": float(finite["height_est_m"].mean()),
        "height_median_m": float(finite["height_est_m"].median()),
        "height_minus_prior_mean_m": float(prior_delta.mean()),
        "height_minus_prior_median_m": float(prior_delta.median()),
        "strict_joint_median_absolute_difference_m": float(strict_abs.median()),
        "strict_joint_p90_absolute_difference_m": float(strict_abs.quantile(0.9)),
        "accuracy_validation": False,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    (outputs["tables"] / "roof_only_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config_v5.json")
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
