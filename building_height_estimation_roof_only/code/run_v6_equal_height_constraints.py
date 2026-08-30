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


def plot_pair_diagnostic(
    curve: pd.DataFrame,
    ids: list[int],
    id_field: str,
    selected_height: float,
    output: Path,
    plot_module,
    external_reference_height_m: float | None = None,
) -> None:
    fig, axes = plot_module.plt.subplots(1, 2, figsize=(10.0, 4.1))
    colors = ["#4C78A8", "#F28E2B"]
    for ax, stage, label in zip(axes, ["coarse", "fine"], ["a  扩大范围共同搜索", "b  0.1 m共同细搜索"]):
        part = curve[curve["stage"] == stage].sort_values("height_m")
        for building_id, color in zip(ids, colors):
            ax.plot(
                part["height_m"],
                part[f"normalized_score_{id_field}_{building_id}"],
                color=color,
                lw=1.4,
                label=f"{id_field}={building_id}",
            )
        ax.plot(part["height_m"], part["equal_height_joint_score"], color="#222222", lw=2.0, label="等高联合得分")
        ax.axvline(selected_height, color="#E45756", ls="--", lw=1.2)
        if (
            external_reference_height_m is not None
            and float(part["height_m"].min()) <= external_reference_height_m <= float(part["height_m"].max())
        ):
            ax.axvline(external_reference_height_m, color="#59A14F", ls=":", lw=1.2, label="约18 m真实高度参照")
        elif external_reference_height_m is not None:
            ax.text(
                0.03,
                0.05,
                f"真实高度参照约{external_reference_height_m:.0f} m\n位于本细搜索窗口之外",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                color="#3A7D44",
                fontsize=7,
            )
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_xlabel("共同候选高度 / m")
        ax.set_ylabel("标准化SAR得分")
        ax.legend(fontsize=7)
    axes[1].text(
        0.97,
        0.95,
        f"共同最优高度\n{selected_height:.1f} m",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        color="#E45756",
    )
    pair_label = "与".join(str(building_id) for building_id in ids)
    fig.suptitle(f"{id_field}建筑{pair_label}：等高约束下的联合SAR顶面搜索", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plot_module.plt.close(fig)


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    outputs = {key: resolve(value) for key, value in config["outputs"].items()}
    for key in ("tables", "vectors", "figures"):
        outputs[key].mkdir(parents=True, exist_ok=True)
    outputs["picall_svg"].parent.mkdir(parents=True, exist_ok=True)

    analyzer = load_module("equal_height_pair_analysis", "analyze_v6_equal_height_pair.py")
    plotter = load_module("roof_only_height_search", "run_roof_only_height_search.py")
    table = pd.read_csv(resolve(config["inputs"]["v5_table"]))
    vector = gpd.read_file(resolve(config["inputs"]["v5_vector"]), layer="roof_only_heights").reset_index(drop=True)
    vector["fid"] = np.arange(len(vector), dtype=np.int64)
    if vector.crs is None or not bool(vector.geometry.is_valid.all()):
        raise ValueError("Invalid or missing V5 vector CRS/geometry")
    if len(table) != len(vector) or table["fid"].nunique() != len(table):
        raise ValueError("V5 table/vector mismatch")
    source_config = json.loads((ROOT / "config_v3.json").read_text(encoding="utf-8"))
    source_buildings = gpd.read_file(
        plotter.resolve(source_config["inputs"]["buildings"]), engine="pyogrio"
    ).reset_index(drop=True)
    if len(source_buildings) != len(table):
        raise ValueError("Source building vector and V5 results have different row counts")
    identifier_fields = {str(item["id_field"]) for item in config["equal_height_constraints"]}
    for id_field in identifier_fields:
        if id_field not in source_buildings.columns:
            raise ValueError(f"Missing identifier field in source building vector: {id_field}")
        if source_buildings[id_field].duplicated().any():
            raise ValueError(f"Identifier field is not unique: {id_field}")
        table[id_field] = source_buildings[id_field].to_numpy()
        vector[id_field] = source_buildings[id_field].to_numpy()

    table["height_v5_input_m"] = table["height_est_m"]
    table["v6_action"] = np.where(table["height_est_m"].notna(), "kept_v5", "not_available_v5")
    table["v6_constraint_source"] = ""
    table["v6_equal_height_joint_score"] = np.nan
    changes: list[dict] = []
    constraint_details: list[dict] = []
    last_curve = pd.DataFrame()
    last_ids: list[int] = []
    last_id_field = ""
    last_height = np.nan
    last_external_reference_height_m: float | None = None
    for constraint in config["equal_height_constraints"]:
        id_field = str(constraint["id_field"])
        ids = [int(value) for value in constraint["ids"]]
        curve_path = resolve(config["inputs"]["equal_height_joint_curve"])
        curve = analyzer.run(ids, curve_path, id_field)
        fine = curve[curve["stage"] == "fine"].copy()
        if fine.empty:
            raise ValueError(f"No fine equal-height candidates for {id_field}={ids}")
        selected = fine.loc[fine["equal_height_joint_score"].idxmax()]
        common_height = float(selected["height_m"])
        if common_height <= float(fine["height_m"].min()) + 0.5 or common_height >= float(fine["height_m"].max()) - 0.5:
            raise ValueError(f"Equal-height optimum is too close to search boundary: {common_height}")
        external_reference = (
            float(constraint["external_reference_height_m"])
            if "external_reference_height_m" in constraint
            else None
        )
        maximum_reference_difference = float(
            constraint.get("external_reference_max_abs_difference_m", float("inf"))
        )
        reference_difference = (
            abs(common_height - external_reference) if external_reference is not None else None
        )
        validation_conflict = bool(
            reference_difference is not None and reference_difference > maximum_reference_difference
        )
        for building_id in ids:
            index = table.index[table[id_field] == building_id]
            if len(index) != 1:
                raise ValueError(f"Missing or duplicate {id_field}={building_id}")
            index = index[0]
            fid = int(table.loc[index, "fid"])
            old_height = table.loc[index, "height_est_m"]
            table.loc[index, "height_raw_m"] = common_height
            table.loc[index, "roof_elevation_raw_m"] = float(table.loc[index, "base_elevation_m"]) + common_height
            if validation_conflict:
                table.loc[index, "height_est_m"] = np.nan
                table.loc[index, "roof_elevation_m"] = np.nan
                table.loc[index, "quality"] = "rejected_external_reference_conflict"
                table.loc[index, "accepted_solution"] = 0
                table.loc[index, "rejection_reason"] = "external_reference_conflict"
                table.loc[index, "scene_consensus"] = "equal_height_joint_sar_peak_but_external_conflict"
                table.loc[index, "solution_source"] = "not_accepted_v6_external_reference_conflict"
                table.loc[index, "v6_action"] = "external_reference_veto"
            else:
                table.loc[index, "height_est_m"] = common_height
                table.loc[index, "roof_elevation_m"] = float(table.loc[index, "base_elevation_m"]) + common_height
                table.loc[index, "height_uncertainty_m"] = max(
                    float(table.loc[index, "height_uncertainty_m"])
                    if pd.notna(table.loc[index, "height_uncertainty_m"])
                    else 1.0,
                    1.0,
                )
                table.loc[index, "quality"] = "medium"
                table.loc[index, "quality_raw"] = "medium"
                table.loc[index, "accepted_solution"] = 1
                table.loc[index, "rejection_reason"] = ""
                table.loc[index, "scene_consensus"] = "equal_height_joint_sar_peak"
                table.loc[index, "solution_source"] = "corrected_v6_equal_height_joint_sar"
                table.loc[index, "v6_action"] = "equal_height_joint_sar_update"
            table.loc[index, "v6_constraint_source"] = str(constraint["source"])
            table.loc[index, "v6_equal_height_joint_score"] = float(selected["equal_height_joint_score"])
            changes.append(
                {
                    "fid": fid,
                    id_field: building_id,
                    "height_v5_m": old_height,
                    "height_v6_m": np.nan if validation_conflict else common_height,
                    "height_v6_raw_joint_sar_m": common_height,
                    "height_prior_m": float(table.loc[index, "height_prior_m"]),
                    "constraint_source": str(constraint["source"]),
                    "selection": str(constraint["selection"]),
                    "equal_height_joint_score": float(selected["equal_height_joint_score"]),
                    "coarse_or_fine_stage": "fine",
                    "prior_fill_used": False,
                    "external_reference_height_m": external_reference,
                    "external_reference_difference_m": reference_difference,
                    "external_reference_used_in_score": False,
                    "external_reference_veto": validation_conflict,
                }
            )
        constraint_details.append(
            {
                "id_field": id_field,
                "ids": ids,
                "raw_common_height_m": common_height,
                "accepted_common_height_m": None if validation_conflict else common_height,
                "source": str(constraint["source"]),
                "external_reference_height_m": external_reference,
                "external_reference_max_abs_difference_m": maximum_reference_difference,
                "external_reference_difference_m": reference_difference,
                "external_reference_used_in_score": False,
                "external_reference_veto": validation_conflict,
            }
        )
        last_curve = curve
        last_ids = ids
        last_id_field = id_field
        last_height = common_height
        last_external_reference_height_m = external_reference

    table.to_csv(outputs["tables"] / "roof_only_building_heights.csv", index=False)
    pd.DataFrame(changes).to_csv(outputs["tables"] / "roof_only_v6_changes.csv", index=False)
    geometry = vector[["fid", "geometry"]].merge(table, on="fid", how="left", validate="one_to_one")
    geometry.to_file(outputs["vectors"] / "roof_only_building_heights.gpkg", layer="roof_only_heights", driver="GPKG")

    figure_path = outputs["figures"] / "roof_only_v6_equal_height_constrained_map"
    plotter.plot_simple_height_map(
        vector[["fid", "geometry"]],
        table,
        figure_path,
        title="V6全区建筑高度估计（等高建筑对外部核验后）",
    )
    outputs["picall_svg"].write_bytes(figure_path.with_suffix(".svg").read_bytes())
    if not last_curve.empty:
        pair_slug = "_".join(str(building_id) for building_id in last_ids)
        diagnostic_path = outputs["figures"] / f"buildings_{last_id_field}_{pair_slug}_equal_height_joint_curve.svg"
        plot_pair_diagnostic(
            last_curve,
            last_ids,
            last_id_field,
            float(last_height),
            diagnostic_path,
            plotter,
            last_external_reference_height_m,
        )
        outputs["pair_diagnostic_svg"].write_bytes(diagnostic_path.read_bytes())

    finite = table[table["height_est_m"].notna()].copy()
    prior_delta = finite["height_est_m"] - finite["height_prior_m"]
    summary = {
        "method": "v6_clean_id_equal_height_joint_sar_with_external_validation_veto",
        "buildings": int(len(table)),
        "v5_finite_heights": int(table["height_v5_input_m"].notna().sum()),
        "finite_heights": int(len(finite)),
        "equal_height_pairs": int(len(config["equal_height_constraints"])),
        "buildings_updated": int(len(changes)),
        "equal_height_constraints_detail": constraint_details,
        "prior_or_neighbor_copy_fill_used": False,
        "height_mean_m": float(finite["height_est_m"].mean()),
        "height_median_m": float(finite["height_est_m"].median()),
        "height_minus_prior_mean_m": float(prior_delta.mean()),
        "height_minus_prior_median_m": float(prior_delta.median()),
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
    parser.add_argument("--config", type=Path, default=ROOT / "config_v6.json")
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
