from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
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


def morphology_frame(vector: gpd.GeoDataFrame, settings: dict) -> pd.DataFrame:
    metric = vector.to_crs("EPSG:32651").reset_index(drop=True)
    records = []
    for fid, geometry in enumerate(metric.geometry):
        rectangle = geometry.minimum_rotated_rectangle
        xy = np.asarray(rectangle.exterior.coords, dtype=float)[:-1]
        lengths = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
        records.append(
            {
                "fid": fid,
                "building_area_m2": float(geometry.area),
                "building_perimeter_m": float(geometry.length),
                "building_aspect_ratio": float(lengths.max() / max(lengths.min(), 1e-6)),
            }
        )
    frame = pd.DataFrame(records)
    small_limit = float(frame["building_area_m2"].quantile(float(settings["small_area_quantile"])))
    large_limit = float(frame["building_area_m2"].quantile(float(settings["large_area_quantile"])))
    elongated_limit = float(frame["building_aspect_ratio"].quantile(float(settings["elongated_aspect_quantile"])))
    profile = np.full(len(frame), "standard", dtype=object)
    profile[frame["building_area_m2"].to_numpy() <= small_limit] = "small"
    profile[frame["building_area_m2"].to_numpy() >= large_limit] = "large"
    profile[frame["building_aspect_ratio"].to_numpy() >= elongated_limit] = "elongated"
    frame["scale_profile"] = profile
    frame["small_area_limit_m2"] = small_limit
    frame["large_area_limit_m2"] = large_limit
    frame["elongated_aspect_limit"] = elongated_limit
    return frame


def weights_for(profile: dict) -> dict[str, float]:
    return {
        "roof_bright_scatter": float(profile["bright"]),
        "roof_boundary_edge": float(profile["edge"]),
        "roof_inside_outside_contrast": float(profile["contrast"]),
        "roof_ps_support": float(profile["ps"]),
    }


def observability_metrics(frame: pd.DataFrame, settings: dict) -> pd.DataFrame:
    frame = frame.copy()
    scene_scores = []
    for date in SCENES:
        shadow = frame[f"shadow_fraction_{date}"].to_numpy(dtype=float)
        texture = frame[f"texture_p90_p10_{date}"].to_numpy(dtype=float)
        extreme = frame[f"extreme_bright_fraction_{date}"].to_numpy(dtype=float)
        concentration = frame[f"top5_energy_fraction_{date}"].to_numpy(dtype=float)
        shadow_score = np.clip(1.0 - shadow / float(settings["maximum_shadow_fraction"]), 0.0, 1.0)
        texture_score = np.clip(texture / float(settings["minimum_texture_p90_p10"]), 0.0, 1.0)
        extreme_score = np.clip(1.0 - extreme / float(settings["maximum_extreme_bright_fraction"]), 0.0, 1.0)
        concentration_score = np.clip(
            (float(settings["maximum_top5_energy_fraction"]) - concentration)
            / max(
                float(settings["maximum_top5_energy_fraction"])
                - float(settings["ideal_top5_energy_fraction"]),
                1e-6,
            ),
            0.0,
            1.0,
        )
        score = 0.30 * shadow_score + 0.30 * texture_score + 0.15 * extreme_score + 0.25 * concentration_score
        frame[f"observability_{date}"] = score
        scene_scores.append(score)
    stacked = np.stack(scene_scores)
    frame["observable_scene_count"] = np.sum(
        stacked >= float(settings["scene_score_threshold"]), axis=0
    ).astype(int)
    frame["two_scene_observability"] = np.sort(stacked, axis=0)[-2]
    frame["median_shadow_fraction"] = np.median(
        np.stack([frame[f"shadow_fraction_{date}"] for date in SCENES]), axis=0
    )
    frame["median_texture_p90_p10"] = np.median(
        np.stack([frame[f"texture_p90_p10_{date}"] for date in SCENES]), axis=0
    )
    frame["median_extreme_bright_fraction"] = np.median(
        np.stack([frame[f"extreme_bright_fraction_{date}"] for date in SCENES]), axis=0
    )
    frame["median_top5_energy_fraction"] = np.median(
        np.stack([frame[f"top5_energy_fraction_{date}"] for date in SCENES]), axis=0
    )
    return frame


def risk_label(row: pd.Series, settings: dict) -> str:
    if not np.isfinite(float(row.get("two_scene_observability", np.nan))):
        return "not_evaluated"
    risks = []
    if float(row["median_shadow_fraction"]) > 0.75 * float(settings["maximum_shadow_fraction"]):
        risks.append("shadow_risk")
    if float(row["median_texture_p90_p10"]) < 0.70 * float(settings["minimum_texture_p90_p10"]):
        risks.append("low_texture")
    if (
        float(row["median_extreme_bright_fraction"]) > 0.75 * float(settings["maximum_extreme_bright_fraction"])
        or float(row["median_top5_energy_fraction"]) > 0.75 * float(settings["maximum_top5_energy_fraction"])
    ):
        risks.append("isolated_strong_scatter")
    if len(risks) > 1:
        return "mixed_risk"
    if risks:
        return risks[0]
    if float(row["two_scene_observability"]) < float(settings["minimum_two_scene_score"]):
        return "limited_multiscene"
    return "observable"


def fine_metrics(fid: int, center: float, half_window: float, context: dict, module) -> pd.DataFrame:
    config = json.loads(json.dumps(context["config"]))
    config["height_search"].update(
        {
            "minimum_m": max(3.0, center - half_window),
            "maximum_m": min(150.0, center + half_window),
            "symmetric_half_window_minimum_m": 200.0,
            "symmetric_half_window_prior_factor": 0.0,
            "prior_penalty_weight": 0.0,
            "coarse_step_m": float(context["fine_step_m"]),
            "fine_step_m": float(context["fine_step_m"]),
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
    return frame[frame["stage"] == "coarse"].copy()


def score_candidates(frame, fixed, weights, kde, population_stats, search, obs_settings, v8):
    scored = v8.objective_for_candidates(frame, fixed, weights, kde, population_stats, search)
    if all(f"shadow_fraction_{date}" in scored for date in SCENES):
        scored = observability_metrics(scored, obs_settings)
        scored["v10_objective"] = scored["v8_objective"] + float(search["observability_weight"]) * (
            scored["two_scene_observability"] - 0.5
        )
    else:
        scored["v10_objective"] = scored["v8_objective"]
    return scored


def plot_audit(audit: pd.DataFrame, output: Path, plotter) -> None:
    evaluated = audit[audit["two_scene_observability"].notna()]
    changed = audit.dropna(subset=["height_v9_m", "height_v10_m"])
    changed = changed[(changed["height_v10_m"] - changed["height_v9_m"]).abs() > 0.05]
    fig, axes = plotter.plt.subplots(2, 2, figsize=(10.8, 8.0))
    profile_order = ["small", "standard", "large", "elongated"]
    counts = audit["scale_profile"].value_counts().reindex(profile_order, fill_value=0)
    axes[0, 0].bar(["小型", "标准", "大型", "狭长"], counts, color=["#F28E2B", "#4E79A7", "#59A14F", "#B07AA1"])
    axes[0, 0].set_ylabel("建筑数量")
    axes[0, 0].set_title("a  全区尺度自适应分组", loc="left", fontweight="bold")
    axes[0, 1].hist(evaluated["two_scene_observability"], bins=np.linspace(0, 1, 21), color="#76B7B2", edgecolor="white")
    axes[0, 1].axvline(0.55, color="#E15759", ls="--", lw=1.2, label="接受阈值 0.55")
    axes[0, 1].set_xlabel("两景可观测性（第二高景）")
    axes[0, 1].set_ylabel("建筑数量")
    axes[0, 1].set_title("b  严格投影屋顶可观测性", loc="left", fontweight="bold")
    axes[0, 1].legend(fontsize=7)
    if not changed.empty:
        axes[1, 0].hist(changed["height_v10_m"] - changed["height_v9_m"], bins=np.arange(-60, 61, 2), color="#4E79A7", edgecolor="white")
    axes[1, 0].axvline(0, color="#777777", ls="--", lw=1)
    axes[1, 0].set_xlabel("V10 − V9 / m")
    axes[1, 0].set_ylabel("建筑数量")
    axes[1, 0].set_title("c  被修改建筑的高度变化", loc="left", fontweight="bold")
    action_order = ["updated_v10", "recovered_v10", "verified_v10", "rejected_v10_unobservable", "kept_v9", "not_available"]
    action_counts = audit["v10_action"].value_counts().reindex(action_order, fill_value=0)
    axes[1, 1].barh(range(len(action_order)), action_counts, color=["#4E79A7", "#59A14F", "#E15759", "#BAB0AC", "#E1E1E1"])
    axes[1, 1].set_yticks(range(len(action_order)), ["V10更新", "V10恢复", "V10验证", "不可观测拒绝", "保留V9", "无值"])
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xscale("symlog", linthresh=10)
    axes[1, 1].set_xlabel("建筑数量")
    axes[1, 1].set_title("d  V10处理动作", loc="left", fontweight="bold")
    for index, value in enumerate(action_counts):
        axes[1, 1].text(value + 2, index, str(int(value)), va="center", fontsize=7)
    fig.suptitle("V10建筑尺度自适应评分与SAR可观测性审计", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plotter.plt.close(fig)


def plot_observability_map(vector, table, output, plotter):
    mapped = vector.to_crs("EPSG:32651").merge(
        table[["fid", "height_est_m", "v10_observability_class"]], on="fid", validate="one_to_one"
    )
    palette = {
        "observable": "#2E8B57",
        "shadow_risk": "#4E79A7",
        "low_texture": "#F28E2B",
        "isolated_strong_scatter": "#E15759",
        "mixed_risk": "#B07AA1",
        "limited_multiscene": "#EDC948",
        "not_evaluated": "#D9D9D9",
    }
    labels = {
        "observable": "可观测",
        "shadow_risk": "阴影风险",
        "low_texture": "低纹理",
        "isolated_strong_scatter": "孤立强散射/叠掩代理风险",
        "mixed_risk": "混合风险",
        "limited_multiscene": "多景可观测性不足",
        "not_evaluated": "未评估/无曲线",
    }
    fig, ax = plotter.plt.subplots(figsize=(10.2, 10.0))
    for key, color in palette.items():
        part = mapped[mapped["v10_observability_class"] == key]
        if not part.empty:
            part.plot(ax=ax, color=color, edgecolor="#FFFFFF", linewidth=0.16, label=f"{labels[key]}（{len(part)}）")
    ax.set_title("V10全区域SAR屋顶可观测性", fontsize=13)
    ax.set_xlabel("Easting / m (UTM 51N)")
    ax.set_ylabel("Northing / m (UTM 51N)")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=7, ncol=2, frameon=True)
    ax.text(0.01, 0.99, "叠掩类别为幅度混合与能量集中代理风险，不是外部DSM真值", transform=ax.transAxes,
            va="top", fontsize=7, bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92})
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plotter.plt.close(fig)


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    outputs = {key: resolve(value) for key, value in config["outputs"].items()}
    for key in ("tables", "vectors", "figures"):
        outputs[key].mkdir(parents=True, exist_ok=True)
    outputs["picall_map_svg"].parent.mkdir(parents=True, exist_ok=True)
    module = load_module("roof_only_search_v10", "run_roof_only_height_search.py")
    v8 = load_module("roof_only_v8_helpers", "run_v8_full_area.py")
    base_config = json.loads((ROOT / "config_v3.json").read_text(encoding="utf-8"))
    buildings = gpd.read_file(module.resolve(base_config["inputs"]["buildings"]), engine="pyogrio").reset_index(drop=True)
    baseline = pd.read_csv(resolve(config["inputs"]["baseline_table"]))
    baseline_audit = pd.read_csv(resolve(config["inputs"]["baseline_audit"]))
    result = baseline.copy()
    vector = gpd.read_file(resolve(config["inputs"]["baseline_vector"]), layer="roof_only_heights").reset_index(drop=True)
    vector["fid"] = np.arange(len(vector), dtype=np.int64)
    morphology = morphology_frame(vector[["fid", "geometry"]], config["scale_classes"])
    result = result.merge(morphology, on="fid", validate="one_to_one")
    result["height_v9_input_m"] = result["height_est_m"]
    result["v10_action"] = np.where(result["height_est_m"].notna(), "kept_v9", "not_available")
    result["v10_observability_class"] = "not_evaluated"
    curves = pd.read_csv(resolve(config["inputs"]["coarse_curves"]))
    curves = curves[(curves["stage"] == "coarse") & (curves["roof_coverage_fraction"] >= 0.98)].copy()
    population = baseline["height_est_m"].dropna().to_numpy(dtype=float)
    kde = gaussian_kde(population, bw_method=float(config["search"]["population_kde_bandwidth"]))
    population_grid = np.arange(3.0, 151.0, 0.1)
    population_stats = v8.center_scale(np.log(kde(population_grid) + 1e-12))
    projector_class, clean_ring_lonlat = module.load_shared_projection(base_config)
    evidence, median_amplitude, median_edge = module.load_evidence(base_config)
    context = {
        "config": base_config,
        "buildings": buildings,
        "lonlat": buildings.to_crs("EPSG:4326"),
        "clean_ring_lonlat": clean_ring_lonlat,
        "projector": projector_class(module.resolve(base_config["inputs"]["rslc_dir"]) / f"{base_config['master_scene']}.rslc.par"),
        "evidence": evidence,
        "median_amplitude": median_amplitude,
        "median_edge": median_edge,
        "ps": module.load_ps(module.resolve(base_config["inputs"]["ps_points"])),
        "fine_step_m": float(config["search"]["fine_step_m"]),
    }
    audit_rows = []
    profiles = config["scale_classes"]["profiles"]
    obs_settings = config["observability"]
    acceptance = config["acceptance"]
    strict_supported = baseline_audit.set_index("fid")["strict_supported"].to_dict()
    strict_reliable = baseline_audit.set_index("fid")["strict_reliable"].to_dict()
    strict_height = baseline_audit.set_index("fid")["strict_height_m"].to_dict()
    cross_veto = config["cross_method_veto"]
    groups = list(curves.groupby("fid"))
    for number, (fid_value, coarse_raw) in enumerate(groups, start=1):
        fid = int(fid_value)
        profile_name = str(result.loc[fid, "scale_profile"])
        profile = profiles[profile_name]
        minimum_pixels = int(profile["minimum_roof_pixels"])
        coarse_raw = coarse_raw[coarse_raw["roof_pixels"] >= minimum_pixels].copy()
        if len(coarse_raw) < 5:
            continue
        weights = weights_for(profile)
        fixed = v8.prepare_fixed_scoring(coarse_raw, weights, float(config["search"]["prominence_scale_m"]))
        coarse = score_candidates(coarse_raw, fixed, weights, kde, population_stats, config["search"], obs_settings, v8)
        coarse_best = coarse.loc[coarse["v10_objective"].idxmax()]
        coarse_height = float(coarse_best["height_m"])
        half_window = float(profile["fine_half_window_m"])
        fine_raw = fine_metrics(fid, coarse_height, half_window, context, module)
        fine_raw = fine_raw[(fine_raw["roof_coverage_fraction"] >= 0.98) & (fine_raw["roof_pixels"] >= minimum_pixels)].copy()
        if len(fine_raw) < 5:
            continue
        fine = score_candidates(fine_raw, fixed, weights, kde, population_stats, config["search"], obs_settings, v8)
        best = fine.loc[fine["v10_objective"].idxmax()]
        candidate_height = float(best["height_m"])
        alternatives = np.abs(fine["height_m"].to_numpy(dtype=float) - candidate_height) >= 1.0
        margin = float(best["v10_objective"] - fine.loc[alternatives, "v10_objective"].max()) if np.any(alternatives) else 0.0
        boundary = candidate_height <= float(fine["height_m"].min()) + float(acceptance["fine_boundary_guard_m"]) or candidate_height >= float(fine["height_m"].max()) - float(acceptance["fine_boundary_guard_m"])
        old_height = result.loc[fid, "height_est_m"]
        old_row = None
        if pd.notna(old_height):
            if abs(float(old_height) - coarse_height) <= half_window:
                old_row = fine.iloc[int(np.argmin(np.abs(fine["height_m"].to_numpy(dtype=float) - float(old_height))))]
            else:
                old_raw = fine_metrics(fid, float(old_height), 0.6, context, module)
                old_raw = old_raw[(old_raw["roof_coverage_fraction"] >= 0.98) & (old_raw["roof_pixels"] >= minimum_pixels)].copy()
                if not old_raw.empty:
                    old_scored = score_candidates(old_raw, fixed, weights, kde, population_stats, config["search"], obs_settings, v8)
                    old_row = old_scored.iloc[int(np.argmin(np.abs(old_scored["height_m"].to_numpy(dtype=float) - float(old_height))))]
        threshold = float(acceptance["existing_minimum_fine_margin"] if pd.notna(old_height) else acceptance["recovery_minimum_fine_margin"])
        accepted = bool(
            not boundary
            and float(best["two_scene_prominence"]) > float(acceptance["minimum_two_scene_prominence"])
            and margin >= threshold
            and int(best["observable_scene_count"]) >= int(obs_settings["minimum_observable_scenes"])
            and float(best["two_scene_observability"]) >= float(obs_settings["minimum_two_scene_score"])
        )
        cross_method_veto_passed = True
        reference_height = strict_height.get(fid, np.nan)
        if accepted and bool(strict_reliable.get(fid, False)) and np.isfinite(reference_height):
            candidate_error = abs(candidate_height - float(reference_height))
            if pd.notna(old_height):
                old_error = abs(float(old_height) - float(reference_height))
                cross_method_veto_passed = candidate_error <= old_error + float(
                    cross_veto["maximum_added_error_for_existing_m"]
                )
            else:
                cross_method_veto_passed = candidate_error <= float(cross_veto["maximum_error_for_recovery_m"])
            accepted = accepted and cross_method_veto_passed
        action = str(result.loc[fid, "v10_action"])
        objective_improvement = np.nan
        if old_row is not None:
            objective_improvement = float(best["v10_objective"] - old_row["v10_objective"])
            old_risk = risk_label(old_row, obs_settings)
            result.loc[fid, "v10_observability_class"] = old_risk
        if accepted:
            minor = pd.notna(old_height) and abs(candidate_height - float(old_height)) <= float(acceptance["minor_refinement_maximum_difference_m"])
            improved = old_row is None or objective_improvement >= float(acceptance["minimum_objective_improvement_for_large_change"])
            unchanged = pd.notna(old_height) and abs(candidate_height - float(old_height)) <= 0.05
            if unchanged:
                action = "verified_v10"
                result.loc[fid, "v10_observability_class"] = risk_label(best, obs_settings)
            elif pd.isna(old_height) or minor or improved:
                result.loc[fid, "height_est_m"] = candidate_height
                result.loc[fid, "height_raw_m"] = candidate_height
                result.loc[fid, "roof_elevation_m"] = float(result.loc[fid, "base_elevation_m"]) + candidate_height
                result.loc[fid, "roof_elevation_raw_m"] = float(result.loc[fid, "base_elevation_m"]) + candidate_height
                result.loc[fid, "height_uncertainty_m"] = max(0.5, 0.5 * float(np.ptp(fine.loc[fine["v10_objective"] >= float(best["v10_objective"]) - 0.35, "height_m"])))
                result.loc[fid, "accepted_solution"] = 1
                result.loc[fid, "quality"] = "medium"
                result.loc[fid, "rejection_reason"] = ""
                result.loc[fid, "solution_source"] = "v10_scale_adaptive_observable_roof_sar"
                action = "updated_v10" if pd.notna(old_height) else "recovered_v10"
                result.loc[fid, "v10_observability_class"] = risk_label(best, obs_settings)
        if pd.notna(old_height) and old_row is not None and action == "kept_v9":
            severe = (
                float(old_row["two_scene_observability"]) < float(obs_settings["severe_two_scene_score"])
                and int(old_row["observable_scene_count"]) == 0
                and not bool(strict_supported.get(fid, False))
            )
            if severe:
                result.loc[fid, "height_est_m"] = np.nan
                result.loc[fid, "roof_elevation_m"] = np.nan
                result.loc[fid, "accepted_solution"] = 0
                result.loc[fid, "quality"] = "rejected_v10_unobservable"
                result.loc[fid, "rejection_reason"] = "v10_severe_sar_unobservability_without_strict_support"
                result.loc[fid, "solution_source"] = "not_accepted_v10_unobservable"
                action = "rejected_v10_unobservable"
        result.loc[fid, "v10_action"] = action
        if old_row is None:
            result.loc[fid, "v10_observability_class"] = risk_label(best, obs_settings)
        chosen = best if action in {"updated_v10", "recovered_v10", "verified_v10"} or old_row is None else old_row
        record = {
            "fid": fid,
            "clean_id": int(result.loc[fid, "clean_id"]),
            "scale_profile": profile_name,
            "building_area_m2": float(result.loc[fid, "building_area_m2"]),
            "building_perimeter_m": float(result.loc[fid, "building_perimeter_m"]),
            "building_aspect_ratio": float(result.loc[fid, "building_aspect_ratio"]),
            "minimum_roof_pixels": minimum_pixels,
            "adaptive_edge_weight": float(weights["roof_boundary_edge"]),
            "adaptive_fine_half_window_m": half_window,
            "height_v9_m": old_height,
            "candidate_height_m": candidate_height,
            "height_v10_m": result.loc[fid, "height_est_m"],
            "candidate_margin": margin,
            "candidate_objective_improvement": objective_improvement,
            "cross_method_veto_passed": cross_method_veto_passed,
            "v10_action": action,
            "v10_observability_class": result.loc[fid, "v10_observability_class"],
        }
        for key in ("two_scene_observability", "observable_scene_count", "median_shadow_fraction", "median_texture_p90_p10", "median_extreme_bright_fraction", "median_top5_energy_fraction"):
            record[key] = float(chosen[key]) if chosen is not None and key in chosen and pd.notna(chosen[key]) else np.nan
        audit_rows.append(record)
        if number % 25 == 0 or number == len(groups):
            print(f"V10 scale/observability {number}/{len(groups)}", flush=True)
    audit = pd.DataFrame(audit_rows)
    missing_fids = sorted(set(result["fid"]) - set(audit["fid"]))
    if missing_fids:
        extra = result.loc[missing_fids, ["fid", "clean_id", "scale_profile", "building_area_m2", "building_perimeter_m", "building_aspect_ratio", "height_v9_input_m", "height_est_m", "v10_action", "v10_observability_class"]].rename(columns={"height_v9_input_m": "height_v9_m", "height_est_m": "height_v10_m"})
        audit = pd.concat([audit, extra], ignore_index=True, sort=False)
    result.to_csv(outputs["tables"] / "roof_only_building_heights.csv", index=False)
    audit.sort_values("fid").to_csv(outputs["tables"] / "roof_only_v10_audit.csv", index=False)
    audit[audit["v10_action"].isin(["updated_v10", "recovered_v10", "rejected_v10_unobservable"])].to_csv(outputs["tables"] / "roof_only_v10_changes.csv", index=False)
    geometry = vector[["fid", "clean_id", "geometry"]].merge(result, on=["fid", "clean_id"], validate="one_to_one")
    geometry.to_file(outputs["vectors"] / "roof_only_building_heights.gpkg", layer="roof_only_heights", driver="GPKG")
    map_stem = outputs["figures"] / "roof_only_v10_height_map"
    module.plot_simple_height_map(vector[["fid", "geometry"]], result, map_stem, title="V10全区域建筑高度估计（尺度自适应与SAR可观测性筛选）")
    outputs["picall_map_svg"].write_bytes(map_stem.with_suffix(".svg").read_bytes())
    audit_figure = outputs["figures"] / "图件_101239916810.svg"
    plot_audit(audit, audit_figure, module)
    outputs["picall_audit_svg"].write_bytes(audit_figure.read_bytes())
    obs_figure = outputs["figures"] / "图件_565217711133.svg"
    plot_observability_map(vector[["fid", "geometry"]], result, obs_figure, module)
    outputs["picall_observability_map_svg"].write_bytes(obs_figure.read_bytes())
    finite = result[result["height_est_m"].notna()]
    summary = {
        "method": "v10_scale_adaptive_scoring_and_multiscene_sar_observability_mask",
        "buildings": int(len(result)),
        "sar_evaluated_buildings": int(audit["two_scene_observability"].notna().sum()),
        "finite_heights": int(len(finite)),
        "action_counts": {str(k): int(v) for k, v in result["v10_action"].value_counts().items()},
        "scale_profile_counts": {str(k): int(v) for k, v in result["scale_profile"].value_counts().items()},
        "observability_counts": {str(k): int(v) for k, v in result["v10_observability_class"].value_counts().items()},
        "height_mean_m": float(finite["height_est_m"].mean()),
        "height_median_m": float(finite["height_est_m"].median()),
        "height_p95_m": float(finite["height_est_m"].quantile(0.95)),
        "height_max_m": float(finite["height_est_m"].max()),
        "external_reference_height_used_in_score": False,
        "neighbor_or_cross_method_height_fill_used": False,
        "layover_mask_interpretation": "amplitude_mixing_and_energy_concentration_proxy_not_external_truth",
        "accuracy_validation": False,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    (outputs["tables"] / "roof_only_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config_v10.json")
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
