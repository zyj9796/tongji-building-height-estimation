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


def load_plotter():
    path = ROOT / "code" / "run_roof_only_height_search.py"
    spec = importlib.util.spec_from_file_location("roof_only_height_search", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load plotting module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plot_simple_height_map


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inputs = config["inputs"]
    outputs = {key: resolve(value) for key, value in config["outputs"].items()}
    for key in ("tables", "vectors", "figures"):
        outputs[key].mkdir(parents=True, exist_ok=True)
    outputs["picall_svg"].parent.mkdir(parents=True, exist_ok=True)

    v3 = pd.read_csv(resolve(inputs["v3_table"]))
    curves = pd.read_csv(resolve(inputs["v3_curves"]))
    vector = gpd.read_file(resolve(inputs["v3_vector"]), layer="roof_only_heights")
    if "fid" not in vector.columns:
        vector = vector.reset_index(drop=True)
        vector["fid"] = np.arange(len(vector), dtype=np.int64)
    strict = pd.read_csv(
        resolve(inputs["strict_joint_reference"]), usecols=["fid", "height_est_m"]
    ).rename(columns={"height_est_m": "strict_joint_height_m"})
    if vector.crs is None:
        raise ValueError("V3 vector has no CRS")
    if not bool(vector.geometry.is_valid.all()):
        raise ValueError("V3 vector contains invalid geometry")
    if len(v3) != len(vector) or v3["fid"].nunique() != len(v3):
        raise ValueError("V3 table/vector row mismatch or duplicate fid")

    table = v3.merge(strict, on="fid", how="left", validate="one_to_one")
    table["height_v3_m"] = table["height_est_m"]
    table["v4_action"] = np.where(table["height_est_m"].notna(), "kept_v3", "not_available_v3")
    table["v4_conflict_flag"] = 0
    table["secondary_peak_height_m"] = np.nan
    table["secondary_peak_score"] = np.nan
    table["secondary_peak_score_loss"] = np.nan
    table["roof_to_prior_difference_m"] = (table["height_v3_m"] - table["height_prior_m"]).abs()
    table["roof_to_strict_difference_m"] = (table["height_v3_m"] - table["strict_joint_height_m"]).abs()
    table["prior_to_strict_difference_m"] = (
        table["height_prior_m"] - table["strict_joint_height_m"]
    ).abs()

    conflict = config["conflict_test"]
    conflict_mask = (
        table["height_v3_m"].notna()
        & table["strict_joint_height_m"].notna()
        & (table["roof_to_prior_difference_m"] > float(conflict["minimum_roof_to_prior_difference_m"]))
        & (table["roof_to_strict_difference_m"] > float(conflict["minimum_roof_to_strict_difference_m"]))
        & (table["prior_to_strict_difference_m"] <= float(conflict["maximum_prior_to_strict_difference_m"]))
    )
    table.loc[conflict_mask, "v4_conflict_flag"] = 1

    peak = config["secondary_sar_peak"]
    audit_rows: list[dict] = []
    for index in table.index[conflict_mask]:
        row = table.loc[index]
        fid = int(row["fid"])
        consensus_center = 0.5 * (float(row["height_prior_m"]) + float(row["strict_joint_height_m"]))
        candidate = curves[
            (curves["fid"] == fid)
            & (curves["stage"] == "coarse")
            & np.isfinite(curves["score"])
            & (curves["score"] > -1e8)
            & (curves["roof_coverage_fraction"] >= float(peak["minimum_roof_coverage_fraction"]))
            & (curves["roof_pixels"] >= int(peak["minimum_roof_pixels"]))
            & ((curves["height_m"] - consensus_center).abs() <= float(peak["consensus_half_window_m"]))
        ].copy()
        all_valid = curves[
            (curves["fid"] == fid)
            & (curves["stage"] == "coarse")
            & np.isfinite(curves["score"])
            & (curves["score"] > -1e8)
        ]
        best_score = float(all_valid["score"].max()) if not all_valid.empty else np.nan
        corrected = False
        if not candidate.empty:
            alternative = candidate.loc[candidate["score"].idxmax()]
            score_loss = best_score - float(alternative["score"])
            table.loc[index, "secondary_peak_height_m"] = float(alternative["height_m"])
            table.loc[index, "secondary_peak_score"] = float(alternative["score"])
            table.loc[index, "secondary_peak_score_loss"] = score_loss
            if score_loss <= float(peak["maximum_score_loss"]):
                corrected_height = float(alternative["height_m"])
                table.loc[index, "height_est_m"] = corrected_height
                table.loc[index, "roof_elevation_m"] = float(row["base_elevation_m"]) + corrected_height
                table.loc[index, "height_uncertainty_m"] = max(float(row["height_uncertainty_m"]), 1.0)
                table.loc[index, "quality"] = "medium"
                table.loc[index, "quality_raw"] = "medium"
                table.loc[index, "accepted_solution"] = 1
                table.loc[index, "rejection_reason"] = ""
                table.loc[index, "scene_consensus"] = "secondary_sar_peak_crosscheck"
                table.loc[index, "solution_source"] = "corrected_v4_secondary_sar_peak"
                table.loc[index, "v4_action"] = "corrected_secondary_sar_peak"
                corrected = True
        if not corrected:
            table.loc[index, "height_est_m"] = np.nan
            table.loc[index, "roof_elevation_m"] = np.nan
            table.loc[index, "accepted_solution"] = 0
            table.loc[index, "quality"] = "rejected_cross_method_conflict"
            table.loc[index, "rejection_reason"] = "cross_method_conflict"
            table.loc[index, "solution_source"] = "not_accepted_v4"
            table.loc[index, "v4_action"] = "rejected_cross_method_conflict"
        audit_rows.append(
            {
                "fid": fid,
                "height_prior_m": float(row["height_prior_m"]),
                "strict_joint_height_m": float(row["strict_joint_height_m"]),
                "height_v3_m": float(row["height_v3_m"]),
                "consensus_center_m": consensus_center,
                "secondary_peak_height_m": table.loc[index, "secondary_peak_height_m"],
                "secondary_peak_score_loss": table.loc[index, "secondary_peak_score_loss"],
                "v4_action": table.loc[index, "v4_action"],
                "height_v4_m": table.loc[index, "height_est_m"],
            }
        )

    table_path = outputs["tables"] / "roof_only_building_heights.csv"
    audit_path = outputs["tables"] / "roof_only_v4_changes.csv"
    table.to_csv(table_path, index=False)
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

    geometry = vector[["fid", "geometry"]].merge(table, on="fid", how="left", validate="one_to_one")
    vector_path = outputs["vectors"] / "roof_only_building_heights.gpkg"
    geometry.to_file(vector_path, layer="roof_only_heights", driver="GPKG")

    plotter = load_plotter()
    figure_path = outputs["figures"] / "roof_only_v4_consensus_audited_height_map"
    plotter(
        vector[["fid", "geometry"]],
        table,
        figure_path,
        title="V4全区建筑高度估计（跨方法冲突审计后）",
    )
    outputs["picall_svg"].write_bytes(figure_path.with_suffix(".svg").read_bytes())

    finite = table[table["height_est_m"].notna()].copy()
    internal = finite.dropna(subset=["strict_joint_height_m"])
    strict_abs = (internal["height_est_m"] - internal["strict_joint_height_m"]).abs()
    prior_delta = finite["height_est_m"] - finite["height_prior_m"]
    summary = {
        "method": "v4_cross_method_conflict_audit_with_sar_secondary_peak_recovery",
        "buildings": int(len(table)),
        "finite_heights": int(len(finite)),
        "v3_finite_heights": int(table["height_v3_m"].notna().sum()),
        "conflicts_detected": int(conflict_mask.sum()),
        "corrected_secondary_sar_peak": int((table["v4_action"] == "corrected_secondary_sar_peak").sum()),
        "rejected_cross_method_conflict": int((table["v4_action"] == "rejected_cross_method_conflict").sum()),
        "prior_fill_used": False,
        "height_mean_m": float(finite["height_est_m"].mean()),
        "height_median_m": float(finite["height_est_m"].median()),
        "height_minus_prior_mean_m": float(prior_delta.mean()),
        "height_minus_prior_median_m": float(prior_delta.median()),
        "strict_joint_common": int(len(internal)),
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
    parser.add_argument("--config", type=Path, default=ROOT / "config_v4.json")
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
