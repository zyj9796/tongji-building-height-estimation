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


def score_candidates(
    ids: list[int],
    id_field: str,
    minimum_m: float,
    maximum_m: float,
    step_m: float,
    module,
    context: dict,
) -> pd.DataFrame:
    config = json.loads(json.dumps(context["config"]))
    config["height_search"].update(
        {
            "minimum_m": minimum_m,
            "maximum_m": maximum_m,
            "symmetric_half_window_minimum_m": 100.0,
            "symmetric_half_window_prior_factor": 0.0,
            "prior_penalty_weight": 0.0,
            "coarse_step_m": step_m,
            "fine_step_m": step_m,
            "fine_half_window_m": max(step_m, 0.5),
        }
    )
    frames: list[pd.DataFrame] = []
    for building_id in ids:
        fid = context["id_to_fid"][building_id]
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
        frame = frame[(frame["stage"] == "coarse") & (frame["score"] > -1e8)].copy()
        keep = ["height_m", "score", *[f"score_{date}" for date in config["scenes"]]]
        rename = {"score": f"fused_score_{building_id}"}
        rename.update({f"score_{date}": f"scene_score_{building_id}_{date}" for date in config["scenes"]})
        frames.append(frame[keep].rename(columns=rename))
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="height_m", how="inner", validate="one_to_one")
    return merged.sort_values("height_m").reset_index(drop=True)


def add_v7_objective(
    frame: pd.DataFrame,
    ids: list[int],
    scenes: list[str],
    population_heights: np.ndarray,
    settings: dict,
    module,
) -> pd.DataFrame:
    result = frame.copy()
    heights = result["height_m"].to_numpy(dtype=float)
    step = float(np.median(np.diff(heights)))
    sigma = max(float(settings["prominence_scale_m"]) / step, 0.8)
    building_consensus: list[np.ndarray] = []
    fused_support: list[np.ndarray] = []
    for building_id in ids:
        scene_prominences: list[np.ndarray] = []
        for date in scenes:
            values = module.robust_z(result[f"scene_score_{building_id}_{date}"].to_numpy(dtype=float))
            prominence = values - gaussian_filter1d(values, sigma=sigma, mode="nearest")
            result[f"local_prominence_{building_id}_{date}"] = prominence
            scene_prominences.append(prominence)
        stacked = np.stack(scene_prominences)
        two_of_three = np.sort(stacked, axis=0)[-2]
        result[f"two_scene_prominence_{building_id}"] = two_of_three
        building_consensus.append(two_of_three)
        support = module.robust_z(result[f"fused_score_{building_id}"].to_numpy(dtype=float))
        result[f"absolute_support_{building_id}"] = support
        fused_support.append(support)
    result["pair_prominence_floor"] = np.min(np.stack(building_consensus), axis=0)
    result["pair_absolute_support_floor"] = np.min(np.stack(fused_support), axis=0)
    kde = gaussian_kde(population_heights, bw_method=float(settings["population_kde_bandwidth"]))
    log_density = np.log(kde(heights) + 1e-12)
    population_term = module.robust_z(log_density)
    result["population_log_density_z"] = population_term
    result["v6_mean_joint_score"] = np.mean(np.stack(fused_support), axis=0)
    result["v7_objective"] = (
        result["pair_prominence_floor"]
        + float(settings["absolute_support_weight"]) * result["pair_absolute_support_floor"]
        + float(settings["population_regularization_weight"]) * result["population_log_density_z"]
    )
    return result


def plot_diagnostic(
    audit: pd.DataFrame,
    selected_height: float,
    raw_v6_height: float,
    reference_height: float,
    output: Path,
    plotter,
) -> None:
    fig, axes = plotter.plt.subplots(1, 3, figsize=(12.0, 3.8))
    coarse = audit[audit["stage"] == "coarse"].sort_values("height_m")
    fine = audit[audit["stage"] == "fine"].sort_values("height_m")
    axes[0].plot(coarse["height_m"], coarse["v6_mean_joint_score"], color="#9C755F", lw=1.7)
    axes[0].axvline(raw_v6_height, color="#E15759", ls="--", lw=1.2, label=f"V6假峰 {raw_v6_height:.1f} m")
    axes[0].axvline(reference_height, color="#59A14F", ls=":", lw=1.2, label="约18 m外部核验")
    axes[0].set_title("a  V6全域绝对得分", loc="left", fontweight="bold")
    axes[0].set_ylabel("标准化联合得分")
    axes[0].legend(fontsize=7)
    axes[1].plot(coarse["height_m"], coarse["v7_objective"], color="#4E79A7", lw=1.8)
    axes[1].axvline(selected_height, color="#E15759", ls="--", lw=1.2)
    axes[1].set_title("b  V7假峰抑制后的粗搜索", loc="left", fontweight="bold")
    axes[1].set_ylabel("V7目标函数")
    axes[2].plot(fine["height_m"], fine["v7_objective"], color="#4E79A7", lw=1.8)
    axes[2].axvline(selected_height, color="#E15759", ls="--", lw=1.2, label=f"V7估计 {selected_height:.1f} m")
    if float(fine["height_m"].min()) - 0.05 <= reference_height <= float(fine["height_m"].max()) + 0.05:
        axes[2].axvline(reference_height, color="#59A14F", ls=":", lw=1.2, label="约18 m外部核验")
    axes[2].set_title("c  0.1 m细搜索与外部核验", loc="left", fontweight="bold")
    axes[2].set_ylabel("V7目标函数")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel("共同候选高度 / m")
    fig.suptitle("clean_id=776与788：V7多景局部峰和短板联合抑制远距离假峰", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plotter.plt.close(fig)


def run(config_path: Path) -> dict:
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    outputs = {key: resolve(value) for key, value in settings["outputs"].items()}
    for key in ("tables", "vectors", "figures"):
        outputs[key].mkdir(parents=True, exist_ok=True)
    outputs["picall_map_svg"].parent.mkdir(parents=True, exist_ok=True)
    module = load_module("roof_only_search_v7", "run_roof_only_height_search.py")
    base_config = json.loads((ROOT / "config_v3.json").read_text(encoding="utf-8"))
    projector_class, clean_ring_lonlat = module.load_shared_projection(base_config)
    buildings = gpd.read_file(module.resolve(base_config["inputs"]["buildings"]), engine="pyogrio").reset_index(drop=True)
    id_field = str(settings["id_field"])
    ids = [int(value) for value in settings["ids"]]
    if id_field not in buildings or buildings[id_field].duplicated().any():
        raise ValueError(f"Missing or non-unique identifier field: {id_field}")
    id_to_fid = {int(value): int(fid) for fid, value in enumerate(buildings[id_field])}
    if any(value not in id_to_fid for value in ids):
        raise ValueError(f"Unknown {id_field} in {ids}")
    evidence, median_amplitude, median_edge = module.load_evidence(base_config)
    context = {
        "config": base_config,
        "buildings": buildings,
        "lonlat": buildings.to_crs("EPSG:4326"),
        "id_to_fid": id_to_fid,
        "clean_ring_lonlat": clean_ring_lonlat,
        "projector": projector_class(
            module.resolve(base_config["inputs"]["rslc_dir"]) / f"{base_config['master_scene']}.rslc.par"
        ),
        "evidence": evidence,
        "median_amplitude": median_amplitude,
        "median_edge": median_edge,
        "ps": module.load_ps(module.resolve(base_config["inputs"]["ps_points"])),
    }
    table = pd.read_csv(resolve(settings["inputs"]["v5_table"]))
    vector = gpd.read_file(resolve(settings["inputs"]["v5_vector"]), layer="roof_only_heights").reset_index(drop=True)
    vector["fid"] = np.arange(len(vector), dtype=np.int64)
    table[id_field] = buildings[id_field].to_numpy()
    vector[id_field] = buildings[id_field].to_numpy()
    population = table["height_est_m"].dropna().to_numpy(dtype=float)
    search = settings["search"]
    coarse_raw = score_candidates(
        ids,
        id_field,
        float(search["minimum_m"]),
        float(search["maximum_m"]),
        float(search["coarse_step_m"]),
        module,
        context,
    )
    coarse = add_v7_objective(coarse_raw, ids, base_config["scenes"], population, search, module)
    coarse_best = float(coarse.loc[coarse["v7_objective"].idxmax(), "height_m"])
    half = float(search["fine_half_window_m"])
    fine_center = coarse_best
    fine = pd.DataFrame()
    selected_height = float("nan")
    for _ in range(4):
        fine_low = max(float(search["minimum_m"]), fine_center - half)
        fine_high = min(float(search["maximum_m"]), fine_center + half)
        fine_raw = score_candidates(
            ids,
            id_field,
            fine_low,
            fine_high,
            float(search["fine_step_m"]),
            module,
            context,
        )
        fine = add_v7_objective(fine_raw, ids, base_config["scenes"], population, search, module)
        selected = fine.loc[fine["v7_objective"].idxmax()]
        selected_height = float(selected["height_m"])
        boundary = (
            selected_height <= float(fine["height_m"].min()) + 0.4
            or selected_height >= float(fine["height_m"].max()) - 0.4
        )
        if not boundary:
            break
        fine_center = selected_height
    else:
        raise ValueError(
            f"V7 fine optimum remains close to boundary: {selected_height}; "
            f"range={fine['height_m'].min()}..{fine['height_m'].max()}"
        )
    coarse["stage"] = "coarse"
    fine["stage"] = "fine"
    audit = pd.concat([coarse, fine], ignore_index=True, sort=False)
    audit.to_csv(outputs["tables"] / "clean_id_776_788_v7_score_audit.csv", index=False)

    reference = float(settings["external_reference_height_m"])
    reference_difference = abs(selected_height - reference)
    accepted = reference_difference <= float(settings["external_reference_max_abs_difference_m"])
    changes: list[dict] = []
    table["height_v5_input_m"] = table["height_est_m"]
    table["v7_action"] = np.where(table["height_est_m"].notna(), "kept_v5", "not_available_v5")
    for building_id in ids:
        index = table.index[table[id_field] == building_id]
        if len(index) != 1:
            raise ValueError(f"Missing or duplicate {id_field}={building_id}")
        index = index[0]
        fid = int(table.loc[index, "fid"])
        old = table.loc[index, "height_est_m"]
        table.loc[index, "height_raw_m"] = selected_height
        table.loc[index, "roof_elevation_raw_m"] = float(table.loc[index, "base_elevation_m"]) + selected_height
        if accepted:
            table.loc[index, "height_est_m"] = selected_height
            table.loc[index, "roof_elevation_m"] = float(table.loc[index, "base_elevation_m"]) + selected_height
            table.loc[index, "height_uncertainty_m"] = max(1.0, 0.5 * float(search["fine_step_m"]))
            table.loc[index, "quality"] = "medium"
            table.loc[index, "quality_raw"] = "medium"
            table.loc[index, "accepted_solution"] = 1
            table.loc[index, "rejection_reason"] = ""
            table.loc[index, "scene_consensus"] = "v7_two_scene_local_peak_pair_floor"
            table.loc[index, "solution_source"] = "corrected_v7_false_peak_suppression"
            table.loc[index, "v7_action"] = "v7_pair_false_peak_suppressed_update"
        else:
            table.loc[index, "height_est_m"] = np.nan
            table.loc[index, "roof_elevation_m"] = np.nan
            table.loc[index, "accepted_solution"] = 0
            table.loc[index, "quality"] = "rejected_external_reference_conflict"
            table.loc[index, "rejection_reason"] = "external_reference_conflict"
            table.loc[index, "solution_source"] = "not_accepted_v7_external_reference_conflict"
            table.loc[index, "v7_action"] = "external_reference_veto"
        changes.append(
            {
                "fid": fid,
                id_field: building_id,
                "height_v5_m": old,
                "height_v7_m": selected_height if accepted else np.nan,
                "height_v7_raw_m": selected_height,
                "external_reference_height_m": reference,
                "external_reference_difference_m": reference_difference,
                "external_reference_used_in_score": False,
                "accepted": accepted,
                "prior_or_neighbor_fill_used": False,
            }
        )
    table.to_csv(outputs["tables"] / "roof_only_building_heights.csv", index=False)
    pd.DataFrame(changes).to_csv(outputs["tables"] / "roof_only_v7_changes.csv", index=False)
    geometry = vector[["fid", id_field, "geometry"]].merge(table, on=["fid", id_field], how="left", validate="one_to_one")
    geometry.to_file(outputs["vectors"] / "roof_only_building_heights.gpkg", layer="roof_only_heights", driver="GPKG")
    map_stem = outputs["figures"] / "roof_only_v7_false_peak_suppressed_map"
    module.plot_simple_height_map(
        vector[["fid", "geometry"]],
        table,
        map_stem,
        title="V7全区建筑高度估计（远距离假峰抑制后）",
    )
    outputs["picall_map_svg"].write_bytes(map_stem.with_suffix(".svg").read_bytes())
    raw_v6_height = 41.7
    diagnostic = outputs["figures"] / "图件_846111917778.svg"
    plot_diagnostic(audit, selected_height, raw_v6_height, reference, diagnostic, module)
    outputs["picall_diagnostic_svg"].write_bytes(diagnostic.read_bytes())
    finite = table[table["height_est_m"].notna()]
    summary = {
        "method": "v7_two_scene_local_prominence_pair_floor_population_regularization",
        "id_field": id_field,
        "ids": ids,
        "internal_fids": [id_to_fid[value] for value in ids],
        "v6_raw_false_peak_m": raw_v6_height,
        "v7_selected_height_m": selected_height,
        "external_reference_height_m": reference,
        "external_reference_difference_m": reference_difference,
        "external_reference_used_in_score": False,
        "accepted": accepted,
        "finite_heights": int(len(finite)),
        "prior_or_neighbor_fill_used": False,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    (outputs["tables"] / "roof_only_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config_v7.json")
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
