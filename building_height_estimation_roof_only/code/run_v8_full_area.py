from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
SCENES = ["20200708", "20200730", "20200821"]


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


def center_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = max(1.4826 * mad, float(np.std(values)) * 0.25, 1e-6)
    return center, scale


def fixed_z(values: np.ndarray, stats: tuple[float, float]) -> np.ndarray:
    return np.clip((np.asarray(values, dtype=float) - stats[0]) / stats[1], -4.0, 4.0)


def prepare_fixed_scoring(coarse: pd.DataFrame, weights: dict, prominence_scale_m: float) -> dict:
    coarse = coarse.sort_values("height_m").reset_index(drop=True)
    feature_stats: dict[str, tuple[float, float]] = {"ps": center_scale(coarse["ps"].to_numpy())}
    for date in SCENES:
        for feature in ("bright", "edge", "contrast"):
            key = f"{feature}_{date}"
            feature_stats[key] = center_scale(coarse[key].to_numpy())
    ps_term = fixed_z(coarse["ps"], feature_stats["ps"])
    scene_scores: list[np.ndarray] = []
    for date in SCENES:
        scene = (
            float(weights["roof_bright_scatter"]) * fixed_z(coarse[f"bright_{date}"], feature_stats[f"bright_{date}"])
            + float(weights["roof_boundary_edge"]) * fixed_z(coarse[f"edge_{date}"], feature_stats[f"edge_{date}"])
            + float(weights["roof_inside_outside_contrast"]) * fixed_z(coarse[f"contrast_{date}"], feature_stats[f"contrast_{date}"])
            + float(weights["roof_ps_support"]) * ps_term
        )
        scene_scores.append(gaussian_filter1d(scene, sigma=0.8, mode="nearest"))
    stacked = np.stack(scene_scores)
    fused = np.median(stacked, axis=0)
    fused_stats = center_scale(fused)
    step = float(np.median(np.diff(coarse["height_m"])))
    sigma = max(prominence_scale_m / step, 0.8)
    backgrounds = np.stack([gaussian_filter1d(scene, sigma=sigma, mode="nearest") for scene in scene_scores])
    return {
        "coarse": coarse,
        "feature_stats": feature_stats,
        "scene_scores": stacked,
        "backgrounds": backgrounds,
        "fused_stats": fused_stats,
    }


def objective_for_candidates(
    candidates: pd.DataFrame,
    fixed: dict,
    weights: dict,
    kde,
    population_stats: tuple[float, float],
    search: dict,
) -> pd.DataFrame:
    candidates = candidates.sort_values("height_m").drop_duplicates("height_m").reset_index(drop=True).copy()
    ps_term = fixed_z(candidates["ps"], fixed["feature_stats"]["ps"])
    scene_scores: list[np.ndarray] = []
    for date in SCENES:
        score = (
            float(weights["roof_bright_scatter"])
            * fixed_z(candidates[f"bright_{date}"], fixed["feature_stats"][f"bright_{date}"])
            + float(weights["roof_boundary_edge"])
            * fixed_z(candidates[f"edge_{date}"], fixed["feature_stats"][f"edge_{date}"])
            + float(weights["roof_inside_outside_contrast"])
            * fixed_z(candidates[f"contrast_{date}"], fixed["feature_stats"][f"contrast_{date}"])
            + float(weights["roof_ps_support"]) * ps_term
        )
        scene_scores.append(gaussian_filter1d(score, sigma=0.8, mode="nearest"))
    stacked = np.stack(scene_scores)
    heights = candidates["height_m"].to_numpy(dtype=float)
    coarse_heights = fixed["coarse"]["height_m"].to_numpy(dtype=float)
    prominence: list[np.ndarray] = []
    for index, scene in enumerate(stacked):
        background = np.interp(heights, coarse_heights, fixed["backgrounds"][index])
        prominence.append(scene - background)
    candidates["two_scene_prominence"] = np.sort(np.stack(prominence), axis=0)[-2]
    fused = np.median(stacked, axis=0)
    candidates["absolute_support"] = fixed_z(fused, fixed["fused_stats"])
    candidates["population_log_density_z"] = fixed_z(np.log(kde(heights) + 1e-12), population_stats)
    candidates["v8_objective"] = (
        candidates["two_scene_prominence"]
        + float(search["absolute_support_weight"]) * candidates["absolute_support"]
        + float(search["population_regularization_weight"]) * candidates["population_log_density_z"]
    )
    return candidates


def fine_metrics(fid: int, center: float, context: dict, search: dict, module) -> pd.DataFrame:
    config = json.loads(json.dumps(context["config"]))
    half = float(search["fine_half_window_m"])
    config["height_search"].update(
        {
            "minimum_m": max(3.0, center - half),
            "maximum_m": min(150.0, center + half),
            "symmetric_half_window_minimum_m": 200.0,
            "symmetric_half_window_prior_factor": 0.0,
            "prior_penalty_weight": 0.0,
            "coarse_step_m": float(search["fine_step_m"]),
            "fine_step_m": float(search["fine_step_m"]),
            "fine_half_window_m": 0.5,
        }
    )
    ring = context["clean_ring_lonlat"](
        np.asarray(context["lonlat"].iloc[fid].geometry.exterior.coords)
    )
    _, rows = module.score_building(
        fid,
        ring,
        float(context["buildings"].iloc[fid]["height"]),
        context["projector"],
        context["evidence"],
        context["median_amplitude"],
        context["median_edge"],
        context["ps"],
        config,
    )
    frame = pd.DataFrame(rows)
    return frame[
        (frame["stage"] == "coarse")
        & (frame["roof_pixels"] >= 3)
        & (frame["roof_coverage_fraction"] >= 0.98)
    ].copy()


def plot_audit(audit: pd.DataFrame, baseline: pd.DataFrame, result: pd.DataFrame, output: Path, plotter) -> None:
    common = baseline[["fid", "height_est_m"]].rename(columns={"height_est_m": "baseline_m"}).merge(
        result[["fid", "height_est_m", "v8_action"]].rename(columns={"height_est_m": "v8_m"}), on="fid"
    )
    finite = common.dropna(subset=["baseline_m", "v8_m"]).copy()
    fig, axes = plotter.plt.subplots(1, 3, figsize=(11.5, 3.8))
    axes[0].scatter(finite["baseline_m"], finite["v8_m"], s=9, alpha=0.55, color="#4E79A7", edgecolors="none")
    limit = max(50.0, float(np.nanmax(np.r_[finite["baseline_m"], finite["v8_m"]])) + 3.0)
    axes[0].plot([0, limit], [0, limit], color="#888888", ls="--", lw=1)
    axes[0].set_xlim(0, limit)
    axes[0].set_ylim(0, limit)
    axes[0].set_xlabel("V7高度 / m")
    axes[0].set_ylabel("V8高度 / m")
    axes[0].set_title("a  全区有限高度对比", loc="left", fontweight="bold")
    changed = finite[(finite["baseline_m"] - finite["v8_m"]).abs() > 0.05]
    axes[1].hist(changed["v8_m"] - changed["baseline_m"], bins=np.arange(-90, 31, 3), color="#59A14F", edgecolor="white")
    axes[1].axvline(0, color="#555555", ls="--", lw=1)
    axes[1].set_xlabel("V8 − V7 / m")
    axes[1].set_ylabel("建筑数量")
    axes[1].set_title("b  被更新建筑的高度变化", loc="left", fontweight="bold")
    order = ["kept_v7", "updated_v8", "recovered_v8", "rejected_v8_false_peak", "not_available"]
    counts = result["v8_action"].value_counts().reindex(order, fill_value=0)
    axes[2].barh(range(len(order)), counts.to_numpy(), color=["#BAB0AC", "#4E79A7", "#59A14F", "#E15759", "#D8D8D8"])
    axes[2].set_yticks(range(len(order)), ["保留V7", "V8更新", "V8恢复", "假峰拒绝", "无观测/无值"])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("建筑数量")
    axes[2].set_title("c  全区处理结果", loc="left", fontweight="bold")
    for index, value in enumerate(counts):
        axes[2].text(value + 3, index, str(int(value)), va="center", fontsize=7)
    fig.suptitle("V8全区域固定归一化、多景局部峰与假峰抑制审计", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plotter.plt.close(fig)


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    outputs = {key: resolve(value) for key, value in config["outputs"].items()}
    for key in ("tables", "vectors", "figures"):
        outputs[key].mkdir(parents=True, exist_ok=True)
    outputs["picall_map_svg"].parent.mkdir(parents=True, exist_ok=True)
    module = load_module("roof_only_search_v8", "run_roof_only_height_search.py")
    base_config = json.loads((ROOT / "config_v3.json").read_text(encoding="utf-8"))
    buildings = gpd.read_file(module.resolve(base_config["inputs"]["buildings"]), engine="pyogrio").reset_index(drop=True)
    baseline = pd.read_csv(resolve(config["inputs"]["baseline_table"]))
    result = baseline.copy()
    vector = gpd.read_file(resolve(config["inputs"]["baseline_vector"]), layer="roof_only_heights").reset_index(drop=True)
    vector["fid"] = np.arange(len(vector), dtype=np.int64)
    id_field = str(config["identifier_field"])
    result[id_field] = buildings[id_field].to_numpy()
    vector[id_field] = buildings[id_field].to_numpy()
    result["height_v7_input_m"] = result["height_est_m"]
    result["v8_action"] = np.where(result["height_est_m"].notna(), "kept_v7", "not_available")
    curves = pd.read_csv(resolve(config["inputs"]["coarse_curves"]))
    curves = curves[
        (curves["stage"] == "coarse")
        & (curves["roof_pixels"] >= 3)
        & (curves["roof_coverage_fraction"] >= 0.98)
    ].copy()
    population = baseline["height_est_m"].dropna().to_numpy(dtype=float)
    kde = gaussian_kde(population, bw_method=float(config["search"]["population_kde_bandwidth"]))
    population_grid = np.arange(3.0, 151.0, 0.1)
    population_stats = center_scale(np.log(kde(population_grid) + 1e-12))
    projector_class, clean_ring_lonlat = module.load_shared_projection(base_config)
    evidence, median_amplitude, median_edge = module.load_evidence(base_config)
    context = {
        "config": base_config,
        "buildings": buildings,
        "lonlat": buildings.to_crs("EPSG:4326"),
        "clean_ring_lonlat": clean_ring_lonlat,
        "projector": projector_class(
            module.resolve(base_config["inputs"]["rslc_dir"]) / f"{base_config['master_scene']}.rslc.par"
        ),
        "evidence": evidence,
        "median_amplitude": median_amplitude,
        "median_edge": median_edge,
        "ps": module.load_ps(module.resolve(base_config["inputs"]["ps_points"])),
    }
    audit_rows: list[dict] = []
    changes: list[dict] = []
    groups = list(curves.groupby("fid"))
    for number, (fid_value, coarse_raw) in enumerate(groups, start=1):
        fid = int(fid_value)
        if len(coarse_raw) < 5:
            continue
        fixed = prepare_fixed_scoring(
            coarse_raw,
            base_config["score_weights"],
            float(config["search"]["prominence_scale_m"]),
        )
        coarse = objective_for_candidates(
            fixed["coarse"], fixed, base_config["score_weights"], kde, population_stats, config["search"]
        )
        best = coarse.loc[coarse["v8_objective"].idxmax()]
        coarse_height = float(best["height_m"])
        alternatives = np.abs(coarse["height_m"].to_numpy(dtype=float) - coarse_height) >= 2.0
        margin = float(best["v8_objective"] - coarse.loc[alternatives, "v8_objective"].max()) if np.any(alternatives) else 0.0
        old_height = result.loc[fid, "height_est_m"]
        threshold = (
            float(config["acceptance"]["existing_minimum_coarse_margin"])
            if pd.notna(old_height)
            else float(config["acceptance"]["recovery_minimum_coarse_margin"])
        )
        boundary = (
            coarse_height <= float(coarse["height_m"].min()) + float(config["acceptance"]["boundary_guard_m"])
            or coarse_height >= float(coarse["height_m"].max()) - float(config["acceptance"]["boundary_guard_m"])
        )
        coarse_accepted = bool(
            not boundary
            and float(best["two_scene_prominence"]) > float(config["acceptance"]["minimum_two_scene_prominence"])
            and margin >= threshold
        )
        final_height = np.nan
        action = result.loc[fid, "v8_action"]
        fine_peak = np.nan
        if coarse_accepted:
            fine_raw = fine_metrics(fid, coarse_height, context, config["search"], module)
            fine = objective_for_candidates(
                fine_raw, fixed, base_config["score_weights"], kde, population_stats, config["search"]
            )
            fine_best = fine.loc[fine["v8_objective"].idxmax()]
            fine_peak = float(fine_best["height_m"])
            fine_boundary = (
                fine_peak <= float(fine["height_m"].min()) + 0.4
                or fine_peak >= float(fine["height_m"].max()) - 0.4
            )
            if not fine_boundary and float(fine_best["two_scene_prominence"]) > 0:
                final_height = fine_peak
                action = "updated_v8" if pd.notna(old_height) else "recovered_v8"
        if not np.isfinite(final_height) and pd.notna(old_height):
            suspected = bool(
                float(old_height) >= float(config["acceptance"]["suspect_old_height_minimum_m"])
                and abs(float(old_height) - coarse_height)
                >= float(config["acceptance"]["suspect_old_to_candidate_difference_m"])
            )
            if suspected:
                result.loc[fid, "height_est_m"] = np.nan
                result.loc[fid, "roof_elevation_m"] = np.nan
                result.loc[fid, "accepted_solution"] = 0
                result.loc[fid, "quality"] = "rejected_v8_false_peak"
                result.loc[fid, "rejection_reason"] = "v8_global_false_peak_suspected"
                result.loc[fid, "solution_source"] = "not_accepted_v8_global_false_peak"
                action = "rejected_v8_false_peak"
        elif np.isfinite(final_height):
            result.loc[fid, "height_est_m"] = final_height
            result.loc[fid, "height_raw_m"] = final_height
            result.loc[fid, "roof_elevation_m"] = float(result.loc[fid, "base_elevation_m"]) + final_height
            result.loc[fid, "roof_elevation_raw_m"] = float(result.loc[fid, "base_elevation_m"]) + final_height
            result.loc[fid, "height_uncertainty_m"] = max(1.0, float(config["search"]["fine_step_m"]))
            result.loc[fid, "accepted_solution"] = 1
            result.loc[fid, "quality"] = "medium"
            result.loc[fid, "quality_raw"] = "medium"
            result.loc[fid, "rejection_reason"] = ""
            result.loc[fid, "scene_consensus"] = "v8_fixed_scale_two_scene_local_peak"
            result.loc[fid, "solution_source"] = "updated_v8_full_area" if pd.notna(old_height) else "recovered_v8_full_area"
        result.loc[fid, "v8_action"] = action
        audit_rows.append(
            {
                "fid": fid,
                id_field: int(result.loc[fid, id_field]),
                "height_v7_m": old_height,
                "v8_coarse_height_m": coarse_height,
                "v8_fine_height_m": fine_peak,
                "v8_final_height_m": result.loc[fid, "height_est_m"],
                "two_scene_prominence": float(best["two_scene_prominence"]),
                "coarse_margin": margin,
                "coarse_boundary": boundary,
                "coarse_accepted": coarse_accepted,
                "v8_action": action,
            }
        )
        if action in {"updated_v8", "recovered_v8", "rejected_v8_false_peak"}:
            changes.append(audit_rows[-1])
        if number % 50 == 0 or number == len(groups):
            print(f"V8 full-area audit {number}/{len(groups)} changes={len(changes)}", flush=True)

    for group in config.get("known_equal_height_groups", []):
        field = next(key for key in group if key in buildings.columns)
        group_height = float(group["height_m"])
        for building_id in group[field]:
            index = result.index[result[field] == int(building_id)]
            if len(index) != 1:
                raise ValueError(f"Missing or duplicate {field}={building_id}")
            index = index[0]
            result.loc[index, "height_est_m"] = group_height
            result.loc[index, "roof_elevation_m"] = float(result.loc[index, "base_elevation_m"]) + group_height
            result.loc[index, "accepted_solution"] = 1
            result.loc[index, "quality"] = "medium"
            result.loc[index, "rejection_reason"] = ""
            result.loc[index, "solution_source"] = str(group["source"])
            result.loc[index, "v8_action"] = "known_equal_height_group_joint_sar"

    audit = pd.DataFrame(audit_rows)
    result.to_csv(outputs["tables"] / "roof_only_building_heights.csv", index=False)
    audit.to_csv(outputs["tables"] / "roof_only_v8_full_area_audit.csv", index=False)
    pd.DataFrame(changes).to_csv(outputs["tables"] / "roof_only_v8_changes.csv", index=False)
    geometry = vector[["fid", id_field, "geometry"]].merge(result, on=["fid", id_field], how="left", validate="one_to_one")
    geometry.to_file(outputs["vectors"] / "roof_only_building_heights.gpkg", layer="roof_only_heights", driver="GPKG")
    map_stem = outputs["figures"] / "roof_only_v8_full_area_height_map"
    module.plot_simple_height_map(
        vector[["fid", "geometry"]], result, map_stem, title="V8全区域建筑高度估计（统一假峰抑制后）"
    )
    outputs["picall_map_svg"].write_bytes(map_stem.with_suffix(".svg").read_bytes())
    audit_figure = outputs["figures"] / "图件_154998064990.svg"
    plot_audit(audit, baseline, result, audit_figure, module)
    outputs["picall_audit_svg"].write_bytes(audit_figure.read_bytes())
    finite = result[result["height_est_m"].notna()]
    summary = {
        "method": "v8_full_area_fixed_normalization_two_scene_local_prominence",
        "buildings": int(len(result)),
        "sar_valid_curve_buildings": int(audit["fid"].nunique()),
        "finite_heights": int(len(finite)),
        "action_counts": {str(k): int(v) for k, v in result["v8_action"].value_counts().items()},
        "height_mean_m": float(finite["height_est_m"].mean()),
        "height_median_m": float(finite["height_est_m"].median()),
        "height_p95_m": float(finite["height_est_m"].quantile(0.95)),
        "height_max_m": float(finite["height_est_m"].max()),
        "building_specific_height_prior_fill_used": False,
        "population_distribution_regularization_used": True,
        "external_reference_height_used_in_score": False,
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
    parser.add_argument("--config", type=Path, default=ROOT / "config_v8.json")
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
