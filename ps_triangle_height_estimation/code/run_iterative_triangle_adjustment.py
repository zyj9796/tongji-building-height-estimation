from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from estimate_heights_by_triangle_adjustment import run as run_adjustment
from map_ps_to_building_surfaces import run as run_mapping
from recompute_hybrid_rooftop_registration import run as run_local_registration


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"


def surface_stability(previous: pd.DataFrame | None, current: pd.DataFrame) -> dict:
    if previous is None or previous.empty or current.empty:
        return {"common_ps": 0, "same_building_fraction": np.nan, "same_surface_fraction": np.nan}
    left = previous[["ps_id", "fid", "surface"]].rename(columns={"fid": "fid_prev", "surface": "surface_prev"})
    right = current[["ps_id", "fid", "surface"]].rename(columns={"fid": "fid_now", "surface": "surface_now"})
    joined = left.merge(right, on="ps_id", how="inner")
    return {
        "common_ps": int(len(joined)),
        "same_building_fraction": float(np.mean(joined.fid_prev == joined.fid_now)) if len(joined) else np.nan,
        "same_surface_fraction": float(
            np.mean((joined.fid_prev == joined.fid_now) & (joined.surface_prev == joined.surface_now))
        ) if len(joined) else np.nan,
    }


def update_heights(current: pd.DataFrame, estimates: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = current.merge(
        estimates[
            [
                "fid", "height_est_m", "quality", "height_uncertainty_m",
                "ps_equations_used", "weighted_residual_rms_m",
            ]
        ],
        on="fid",
        how="left",
    )
    quality_damping = cfg["quality_damping"]
    damping = joined.quality.map(quality_damping).fillna(0.0).to_numpy(dtype=np.float64)
    raw = pd.to_numeric(joined.height_est_m, errors="coerce").to_numpy(dtype=np.float64)
    old = joined.height_current_m.to_numpy(dtype=np.float64)
    target_delta = np.where(np.isfinite(raw), raw - old, 0.0)
    limited_delta = np.clip(
        damping * target_delta,
        -float(cfg["maximum_height_change_per_iteration_m"]),
        float(cfg["maximum_height_change_per_iteration_m"]),
    )
    new = np.clip(
        old + limited_delta,
        float(cfg["minimum_height_m"]),
        float(cfg["maximum_height_m"]),
    )
    joined["damping"] = damping
    joined["height_next_m"] = new
    joined["height_change_m"] = new - old
    next_table = joined[["fid", "clean_id", "height_next_m"]].rename(columns={"height_next_m": "height_current_m"})
    return next_table, joined


def run(
    config_path: Path,
    max_buildings: int = 0,
    max_iterations: int | None = None,
    output_root: Path | None = None,
    initial_registration_path: Path | None = None,
    initial_heights_path: Path | None = None,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = config["iterative_adjustment"]
    iterations = int(max_iterations or cfg["maximum_iterations"])
    output_root = (
        output_root.resolve()
        if output_root
        else ROOT / "results" / "iterative_triangle_adjustment_stable_registration"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    buildings = gpd.read_file((ROOT / config["inputs"]["buildings"]).resolve()).reset_index(drop=True)
    buildings["fid"] = np.arange(len(buildings), dtype=np.int64)
    if max_buildings > 0:
        buildings = buildings.iloc[:max_buildings].copy()
    current = pd.DataFrame(
        {
            "fid": buildings.fid.astype(np.int64),
            "clean_id": pd.to_numeric(buildings.clean_id, errors="coerce").astype(np.int64),
            "height_current_m": pd.to_numeric(buildings[config["height_field"]], errors="coerce"),
        }
    )
    if initial_heights_path is not None:
        supplied = pd.read_csv(initial_heights_path.resolve())
        if supplied.fid.duplicated().any():
            raise ValueError("Initial height fid values must be unique")
        supplied_column = (
            "height_current_m"
            if "height_current_m" in supplied
            else "height_projection_m"
        )
        supplied = supplied[["fid", supplied_column]].rename(
            columns={supplied_column: "height_current_m_supplied"}
        )
        current = current.merge(supplied, on="fid", how="left", validate="one_to_one")
        current["height_current_m"] = current.height_current_m_supplied.fillna(
            current.height_current_m
        )
        current = current.drop(columns="height_current_m_supplied")
    previous_mapping: pd.DataFrame | None = None
    iteration_summaries: list[dict] = []
    histories: list[pd.DataFrame] = []
    final_iteration_dir: Path | None = None
    for iteration_index in range(iterations):
        iteration_dir = output_root / f"iteration_{iteration_index:02d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        height_path = iteration_dir / "projection_heights.csv"
        current.to_csv(height_path, index=False)
        if iteration_index == 0 or cfg["registration_update_mode"] == "dynamic":
            registration_path = iteration_dir / "local_registration.csv"
            if iteration_index == 0 and initial_registration_path is not None:
                shutil.copy2(initial_registration_path.resolve(), registration_path)
            else:
                run_local_registration(config_path, height_path, registration_path, max_buildings)
        else:
            registration_path = output_root / "iteration_00" / "local_registration.csv"
        mapping_root = iteration_dir / "mapping"
        run_mapping(
            config_path,
            max_buildings,
            height_path,
            registration_path,
            mapping_root,
        )
        mapped_path = mapping_root / "tables" / "ps_building_surface_coordinates.csv"
        adjustment_dir = iteration_dir / "adjustment"
        run_adjustment(config_path, mapped_path, adjustment_dir)
        mapped = pd.read_csv(mapped_path)
        estimates = pd.read_csv(adjustment_dir / "building_heights_triangle_adjustment.csv")
        if max_buildings > 0:
            estimates = estimates.loc[estimates.fid < max_buildings].copy()
        next_current, history = update_heights(current, estimates, cfg)
        history.insert(0, "iteration", iteration_index)
        histories.append(history)
        stability = surface_stability(previous_mapping, mapped)
        finite_change = np.abs(history.loc[np.isfinite(history.height_est_m), "height_change_m"])
        summary = {
            "iteration": iteration_index,
            "projected_buildings": int(len(current)),
            "mapped_ps": int(len(mapped)),
            "buildings_with_ps": int(mapped.fid.nunique()) if len(mapped) else 0,
            "estimated_buildings": int(np.isfinite(history.height_est_m).sum()),
            "registration_update_mode": str(cfg["registration_update_mode"]),
            "registration_source": str(registration_path),
            "median_abs_height_change_m": float(finite_change.median()) if len(finite_change) else None,
            "p90_abs_height_change_m": float(finite_change.quantile(0.9)) if len(finite_change) else None,
            **stability,
        }
        iteration_summaries.append(summary)
        (iteration_dir / "iteration_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        final_iteration_dir = iteration_dir
        stable = (
            iteration_index + 1 >= int(cfg["minimum_iterations"])
            and summary["median_abs_height_change_m"] is not None
            and summary["median_abs_height_change_m"] <= float(cfg["median_height_change_tolerance_m"])
            and summary["p90_abs_height_change_m"] <= float(cfg["p90_height_change_tolerance_m"])
            and np.isfinite(summary["same_surface_fraction"])
            and summary["same_surface_fraction"] >= float(cfg["surface_stability_fraction"])
        )
        previous_mapping = mapped
        if stable:
            break
        current = next_current

    history_table = pd.concat(histories, ignore_index=True)
    history_table.to_csv(output_root / "building_iteration_history.csv", index=False)
    pd.DataFrame(iteration_summaries).to_csv(output_root / "iteration_convergence.csv", index=False)
    final_estimates = pd.read_csv(
        final_iteration_dir / "adjustment" / "building_heights_triangle_adjustment.csv"
    )
    if max_buildings > 0:
        final_estimates = final_estimates.loc[final_estimates.fid < max_buildings].copy()
    final_estimates.to_csv(output_root / "building_heights_iterative.csv", index=False)
    initial_registration = pd.read_csv(output_root / "iteration_00" / "local_registration.csv")
    registration_feature_mode = (
        str(initial_registration.registration_feature_mode.dropna().iloc[0])
        if "registration_feature_mode" in initial_registration
        and initial_registration.registration_feature_mode.notna().any()
        else "legacy_amplitude_inside_ring"
    )
    summary = {
        "method": "iterative_height_projection_local_registration_ps_surface_classification_and_robust_adjustment",
        "registration_update_mode": str(cfg["registration_update_mode"]),
        "registration_feature_mode": registration_feature_mode,
        "initial_registration_accepted_buildings": int(initial_registration.accepted.sum()),
        "initial_registration_boundary_limited_buildings": (
            int(initial_registration.boundary_limited.sum())
            if "boundary_limited" in initial_registration
            else None
        ),
        "initial_registration_scene_consensus_shift_px": (
            {
                "row": int(initial_registration.scene_consensus_row_shift.iloc[0]),
                "col": int(initial_registration.scene_consensus_col_shift.iloc[0]),
            }
            if "scene_consensus_row_shift" in initial_registration
            and "scene_consensus_col_shift" in initial_registration
            else None
        ),
        "initial_registration_local_refinements": (
            int(initial_registration.local_refinement_accepted.sum())
            if "local_refinement_accepted" in initial_registration
            else None
        ),
        "initial_registration_applied_at_search_boundary": int(
            np.sum(
                (initial_registration.accepted == 1)
                & (
                    (
                        np.abs(initial_registration.applied_row_shift)
                        == int(cfg["local_registration_max_shift_px"])
                    )
                    |
                    (
                        np.abs(initial_registration.applied_col_shift)
                        == int(cfg["local_registration_max_shift_px"])
                    )
                )
            )
        ),
        "registration_identifiability_note": (
            "Scene-consensus and building-level rooftop registration are estimated once from the "
            "SHP-height roof projection and then frozen, so later shifts cannot compensate for "
            "height changes."
        ),
        "iterations_completed": int(len(iteration_summaries)),
        "converged": bool(stable),
        "stopping_rules": {
            "median_abs_height_change_m": float(cfg["median_height_change_tolerance_m"]),
            "p90_abs_height_change_m": float(cfg["p90_height_change_tolerance_m"]),
            "same_surface_fraction": float(cfg["surface_stability_fraction"]),
        },
        "final_iteration": str(final_iteration_dir),
        "final_estimated_buildings": int(np.isfinite(final_estimates.height_est_m).sum()),
        "ps_height_provenance_warning": (
            "PS height_m provenance must be read from the configured normalized "
            "PS table. For the 20260724 coordinate package it equals the StaMPS "
            "input DSM height plus the scene-mean-referenced SCLA-equivalent "
            "height residual."
        ),
        "outputs": {
            "final_estimates_csv": str(output_root / "building_heights_iterative.csv"),
            "history_csv": str(output_root / "building_iteration_history.csv"),
            "convergence_csv": str(output_root / "iteration_convergence.csv"),
        },
    }
    (output_root / "iterative_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterate triangle projection, local registration, PS classification, and height adjustment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-buildings", type=int, default=0)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--initial-registration", type=Path)
    parser.add_argument("--initial-heights", type=Path)
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.max_buildings,
        args.max_iterations,
        args.output_root,
        args.initial_registration,
        args.initial_heights,
    )


if __name__ == "__main__":
    main()
