from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "iterative_triangle_adjustment_stable_registration"
DEFAULT_CONFIG = ROOT / "config.json"


def signature_jaccard(left: pd.DataFrame, right: pd.DataFrame) -> pd.Series:
    left_sets = left.groupby("fid").apply(
        lambda group: set(zip(group.ps_id, group.surface, strict=True)), include_groups=False
    )
    right_sets = right.groupby("fid").apply(
        lambda group: set(zip(group.ps_id, group.surface, strict=True)), include_groups=False
    )
    fids = set(left_sets.index) | set(right_sets.index)
    return pd.Series(
        {
            fid: len(left_sets.get(fid, set()) & right_sets.get(fid, set()))
            / max(len(left_sets.get(fid, set()) | right_sets.get(fid, set())), 1)
            for fid in fids
        },
        name="surface_signature_jaccard",
    )


def run(config_path: Path, results_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    convergence = pd.read_csv(results_dir / "iteration_convergence.csv")
    final_iteration = int(convergence.iteration.max())
    previous_iteration = final_iteration - 1
    estimates = pd.read_csv(results_dir / "building_heights_iterative.csv").set_index("fid")
    history = pd.read_csv(results_dir / "building_iteration_history.csv")
    recent = history.loc[history.iteration >= max(0, final_iteration - 2)]
    recent_range = (
        recent.pivot(index="fid", columns="iteration", values="height_est_m")
        .apply(lambda row: row.max(skipna=True) - row.min(skipna=True), axis=1)
        .rename("recent_height_range_m")
    )
    final_history = history.loc[history.iteration == final_iteration].set_index("fid")
    projection_difference = (
        estimates.height_est_m - final_history.height_current_m
    ).abs().rename("projection_estimate_difference_m")
    previous_mapping = pd.read_csv(
        results_dir
        / f"iteration_{previous_iteration:02d}"
        / "mapping"
        / "tables"
        / "ps_building_surface_coordinates.csv",
        usecols=["ps_id", "fid", "surface"],
    )
    final_mapping = pd.read_csv(
        results_dir
        / f"iteration_{final_iteration:02d}"
        / "mapping"
        / "tables"
        / "ps_building_surface_coordinates.csv",
        usecols=["ps_id", "fid", "surface"],
    )
    jaccard = signature_jaccard(previous_mapping, final_mapping)
    table = estimates.join([recent_range, projection_difference, jaccard])
    base_quality = (
        table.quality.isin(["high", "medium"])
        & (table.ps_equations_used >= 3)
        & (table.height_uncertainty_m <= 8.0)
        & (table.weighted_residual_rms_m <= 10.0)
    )
    stable = (
        base_quality
        & (table.projection_estimate_difference_m <= 2.0)
        & (table.recent_height_range_m <= 2.0)
        & (table.surface_signature_jaccard >= 0.90)
    )
    table["final_status"] = "no_ps_equation"
    table.loc[np.isfinite(table.height_est_m), "final_status"] = "low_quality"
    table.loc[base_quality, "final_status"] = "unstable_iteration"
    table.loc[stable, "final_status"] = "stable"
    table["height_final_m"] = table.height_est_m.where(stable)
    table = table.reset_index()
    table.to_csv(results_dir / "building_heights_iterative_screened.csv", index=False)

    buildings = gpd.read_file((ROOT / config["inputs"]["buildings"]).resolve()).reset_index(drop=True)
    buildings["building_fid"] = np.arange(len(buildings), dtype=np.int64)
    attributes = table.rename(columns={"fid": "building_fid"}).drop(columns=["clean_id"], errors="ignore")
    vector = buildings.merge(attributes, on="building_fid", how="left")
    vector_path = results_dir / "building_heights_iterative_screened.gpkg"
    vector.to_file(vector_path, layer="screened_iterative_heights", driver="GPKG")
    stable_table = table.loc[stable].copy()
    summary = {
        "final_iteration": final_iteration,
        "screening_definition": {
            "quality": ["high", "medium"],
            "minimum_ps_equations_used": 3,
            "maximum_uncertainty_m": 8.0,
            "maximum_weighted_residual_rms_m": 10.0,
            "maximum_projection_estimate_difference_m": 2.0,
            "maximum_recent_three_iteration_height_range_m": 2.0,
            "minimum_surface_signature_jaccard": 0.90,
        },
        "status_counts": table.final_status.value_counts().to_dict(),
        "stable_height_m": {
            "count": int(len(stable_table)),
            "min": float(stable_table.height_final_m.min()),
            "median": float(stable_table.height_final_m.median()),
            "mean": float(stable_table.height_final_m.mean()),
            "max": float(stable_table.height_final_m.max()),
        },
        "warning": "Screening measures internal iterative stability, not independent external height accuracy.",
        "outputs": {
            "screened_csv": str(results_dir / "building_heights_iterative_screened.csv"),
            "screened_gpkg": str(vector_path),
        },
    }
    (results_dir / "screened_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen iterative height estimates by per-building numerical and PS-membership stability.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    run(args.config.resolve(), args.results_dir.resolve())


if __name__ == "__main__":
    main()
