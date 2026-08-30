from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def resolve(text: str) -> Path:
    return (ROOT / text).resolve()


def load_plotter():
    path = ROOT / "code" / "run_roof_only_height_search.py"
    spec = importlib.util.spec_from_file_location("roof_only_plotter_v9", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def morphology(geometry) -> tuple[float, float, float]:
    rectangle = geometry.minimum_rotated_rectangle
    xy = np.asarray(rectangle.exterior.coords, dtype=float)[:-1]
    vectors = np.roll(xy, -1, axis=0) - xy
    lengths = np.linalg.norm(vectors, axis=1)
    index = int(np.argmax(lengths))
    aspect = float(lengths.max() / max(lengths.min(), 1e-6))
    orientation = float(math.degrees(math.atan2(vectors[index, 1], vectors[index, 0])) % 180.0)
    return float(geometry.area), aspect, orientation


def spatial_neighbors(vector: gpd.GeoDataFrame, table: pd.DataFrame, settings: dict) -> pd.DataFrame:
    metric = vector.to_crs("EPSG:32651").reset_index(drop=True)
    centroids = np.asarray([(point.x, point.y) for point in metric.geometry.centroid], dtype=float)
    features = np.asarray([morphology(geometry) for geometry in metric.geometry], dtype=float)
    areas, aspects, orientations = features.T
    heights = table["height_est_m"].to_numpy(dtype=float)
    finite = np.isfinite(heights)
    records: list[dict] = []
    for fid in range(len(table)):
        if not finite[fid]:
            records.append({"fid": fid, "comparable_neighbors": 0, "neighbor_median_m": np.nan, "neighbor_difference_m": np.nan})
            continue
        distance = np.linalg.norm(centroids - centroids[fid], axis=1)
        area_ratio = areas / max(areas[fid], 1e-6)
        aspect_ratio = aspects / max(aspects[fid], 1e-6)
        orientation_difference = np.abs(orientations - orientations[fid])
        orientation_difference = np.minimum(orientation_difference, 180.0 - orientation_difference)
        selector = (
            finite
            & (distance > 0)
            & (distance <= float(settings["maximum_centroid_distance_m"]))
            & (area_ratio >= float(settings["minimum_area_ratio"]))
            & (area_ratio <= float(settings["maximum_area_ratio"]))
            & (aspect_ratio >= float(settings["minimum_aspect_ratio_ratio"]))
            & (aspect_ratio <= float(settings["maximum_aspect_ratio_ratio"]))
            & (orientation_difference <= float(settings["maximum_orientation_difference_deg"]))
        )
        values = heights[selector]
        median = float(np.median(values)) if len(values) >= int(settings["minimum_comparable_neighbors"]) else np.nan
        records.append(
            {
                "fid": fid,
                "comparable_neighbors": int(len(values)),
                "neighbor_median_m": median,
                "neighbor_difference_m": float(heights[fid] - median) if np.isfinite(median) else np.nan,
            }
        )
    return pd.DataFrame(records)


def plot_reliability_map(vector: gpd.GeoDataFrame, table: pd.DataFrame, output: Path, plotter) -> None:
    mapped = vector.to_crs("EPSG:32651").merge(table[["fid", "v9_reliability"]], on="fid", validate="one_to_one")
    palette = {
        "supported_both": "#2E8B57",
        "supported_cross_method": "#4E79A7",
        "supported_spatial": "#76B7B2",
        "limited_evidence": "#BAB0AC",
        "review_cross_method_conflict": "#F28E2B",
        "review_spatial_outlier": "#EDC948",
        "rejected_multisource_conflict": "#E15759",
        "not_available": "#E1E1E1",
    }
    labels = {
        "supported_both": "双重支持",
        "supported_cross_method": "严格几何支持",
        "supported_spatial": "同型邻楼支持",
        "limited_evidence": "证据有限",
        "review_cross_method_conflict": "跨方法冲突",
        "review_spatial_outlier": "空间异常",
        "rejected_multisource_conflict": "多源冲突拒绝",
        "not_available": "无最终高度",
    }
    fig, ax = plotter.plt.subplots(figsize=(10.2, 10.0))
    for key in palette:
        part = mapped[mapped["v9_reliability"] == key]
        if not part.empty:
            part.plot(ax=ax, color=palette[key], edgecolor="#FFFFFF", linewidth=0.16, label=f"{labels[key]}（{len(part)}）")
    ax.set_title("V9全区域高度可靠性审计", fontsize=13)
    ax.set_xlabel("Easting / m (UTM 51N)")
    ax.set_ylabel("Northing / m (UTM 51N)")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=7, ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plotter.plt.close(fig)


def plot_audit(audit: pd.DataFrame, result: pd.DataFrame, output: Path, plotter) -> None:
    common = audit.dropna(subset=["height_v9_m", "strict_height_m"])
    spatial = audit.dropna(subset=["height_v9_m", "neighbor_median_m"])
    fig, axes = plotter.plt.subplots(1, 3, figsize=(11.7, 3.8))
    colors = np.where(common["strict_reliable"], "#4E79A7", "#BAB0AC")
    axes[0].scatter(common["strict_height_m"], common["height_v9_m"], c=colors, s=9, alpha=0.6, edgecolors="none")
    limit = max(50.0, float(np.nanmax(np.r_[common["strict_height_m"], common["height_v9_m"]])) + 3.0)
    axes[0].plot([0, limit], [0, limit], color="#777777", ls="--", lw=1)
    axes[0].set_xlim(0, limit)
    axes[0].set_ylim(0, limit)
    axes[0].set_xlabel("严格几何高度 / m")
    axes[0].set_ylabel("V9高度 / m")
    axes[0].set_title("a  跨方法一致性", loc="left", fontweight="bold")
    axes[1].hist(spatial["neighbor_difference_m"], bins=np.arange(-55, 56, 3), color="#76B7B2", edgecolor="white")
    axes[1].axvline(0, color="#666666", ls="--", lw=1)
    axes[1].set_xlabel("V9 − 同型邻楼中位数 / m")
    axes[1].set_ylabel("建筑数量")
    axes[1].set_title("b  空间同型建筑一致性", loc="left", fontweight="bold")
    order = [
        "restored_v7_cross_method_support",
        "reverted_to_v7_cross_method_support",
        "rejected_multisource_conflict",
        "kept_v8",
    ]
    counts = result["v9_action"].value_counts().reindex(order, fill_value=0)
    axes[2].barh(range(len(order)), counts, color=["#59A14F", "#4E79A7", "#E15759", "#BAB0AC"])
    axes[2].set_yticks(range(len(order)), ["恢复V7拒绝值", "撤销V8降高", "多源冲突拒绝", "保留V8"])
    axes[2].invert_yaxis()
    axes[2].set_xscale("symlog", linthresh=10)
    axes[2].set_xlabel("建筑数量")
    axes[2].set_title("c  V9审计动作", loc="left", fontweight="bold")
    for index, value in enumerate(counts):
        axes[2].text(value + 3, index, str(int(value)), va="center", fontsize=7)
    fig.suptitle("V9全区域跨方法与空间同型建筑联合可靠性审计", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plotter.plt.close(fig)


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    outputs = {key: resolve(value) for key, value in config["outputs"].items()}
    for key in ("tables", "vectors", "figures"):
        outputs[key].mkdir(parents=True, exist_ok=True)
    outputs["picall_map_svg"].parent.mkdir(parents=True, exist_ok=True)
    plotter = load_plotter()
    v7 = pd.read_csv(resolve(config["inputs"]["v7_table"]))
    v8 = pd.read_csv(resolve(config["inputs"]["v8_table"]))
    result = v8.copy()
    vector = gpd.read_file(resolve(config["inputs"]["v8_vector"]), layer="roof_only_heights").reset_index(drop=True)
    vector["fid"] = np.arange(len(vector), dtype=np.int64)
    strict = pd.read_csv(resolve(config["inputs"]["strict_joint_table"]))
    spatial = spatial_neighbors(vector[["fid", "geometry"]], v8, config["spatial_audit"])
    audit = v8[["fid", "clean_id", "height_est_m", "v8_action"]].rename(columns={"height_est_m": "height_v8_m"})
    audit = audit.merge(v7[["fid", "height_est_m"]].rename(columns={"height_est_m": "height_v7_m"}), on="fid")
    audit = audit.merge(
        strict[["fid", "height_est_m", "quality", "height_scene_range_m", "score_margin"]].rename(
            columns={"height_est_m": "strict_height_m", "quality": "strict_quality"}
        ),
        on="fid",
    ).merge(spatial, on="fid")
    cross = config["cross_method_audit"]
    audit["strict_reliable"] = (
        audit["strict_quality"].isin(cross["reliable_qualities"])
        & audit["strict_height_m"].notna()
        & (audit["height_scene_range_m"] <= float(cross["maximum_scene_range_m"]))
    )
    audit["strict_v7_difference_m"] = (audit["strict_height_m"] - audit["height_v7_m"]).abs()
    audit["strict_v8_difference_m"] = (audit["strict_height_m"] - audit["height_v8_m"]).abs()
    restore = (
        (audit["v8_action"] == "rejected_v8_false_peak")
        & audit["strict_reliable"]
        & (audit["strict_v7_difference_m"] <= float(cross["restore_rejected_v7_difference_m"]))
    )
    revert = (
        audit["v8_action"].isin(["updated_v8", "recovered_v8"])
        & audit["height_v7_m"].notna()
        & audit["strict_reliable"]
        & (audit["strict_v7_difference_m"] <= float(cross["revert_update_v7_difference_m"]))
        & (audit["strict_v8_difference_m"] >= float(cross["minimum_v8_conflict_difference_m"]))
    )
    result["v9_action"] = "kept_v8"
    result.loc[restore, "height_est_m"] = audit.loc[restore, "height_v7_m"].to_numpy()
    result.loc[restore, "v9_action"] = "restored_v7_cross_method_support"
    result.loc[revert, "height_est_m"] = audit.loc[revert, "height_v7_m"].to_numpy()
    result.loc[revert, "v9_action"] = "reverted_to_v7_cross_method_support"
    changed_to_v7 = restore | revert
    result.loc[changed_to_v7, "roof_elevation_m"] = (
        result.loc[changed_to_v7, "base_elevation_m"] + result.loc[changed_to_v7, "height_est_m"]
    )
    result.loc[changed_to_v7, "accepted_solution"] = 1
    result.loc[changed_to_v7, "quality"] = "medium"
    result.loc[changed_to_v7, "rejection_reason"] = ""
    result.loc[changed_to_v7, "solution_source"] = "v9_restored_cross_method_supported_v7"

    updated_spatial = spatial_neighbors(vector[["fid", "geometry"]], result, config["spatial_audit"])
    audit = audit.drop(columns=["comparable_neighbors", "neighbor_median_m", "neighbor_difference_m"]).merge(
        updated_spatial, on="fid"
    )
    audit["height_v9_m"] = result["height_est_m"].to_numpy()
    audit["strict_v9_difference_m"] = (audit["strict_height_m"] - audit["height_v9_m"]).abs()
    audit["strict_supported"] = audit["strict_reliable"] & (
        audit["strict_v9_difference_m"] <= float(cross["support_difference_m"])
    )
    spatial_settings = config["spatial_audit"]
    audit["spatial_supported"] = (
        (audit["comparable_neighbors"] >= int(spatial_settings["minimum_comparable_neighbors"]))
        & (audit["neighbor_difference_m"].abs() <= float(spatial_settings["support_difference_m"]))
    )
    audit["spatial_conflict"] = (
        (audit["comparable_neighbors"] >= int(spatial_settings["minimum_comparable_neighbors"]))
        & (audit["neighbor_difference_m"].abs() > float(spatial_settings["conflict_difference_m"]))
    )
    audit["strict_conflict"] = audit["strict_reliable"] & (
        audit["strict_v9_difference_m"] > float(cross["minimum_v8_conflict_difference_m"])
    )
    multisource_conflict = audit["height_v9_m"].notna() & audit["strict_conflict"] & audit["spatial_conflict"]
    unresolved_high = (
        audit["height_v9_m"].notna()
        & (audit["height_v9_m"] > 40.0)
        & audit["spatial_conflict"]
        & ~audit["strict_supported"]
    )
    reject = multisource_conflict | unresolved_high
    result.loc[reject, "height_est_m"] = np.nan
    result.loc[reject, "roof_elevation_m"] = np.nan
    result.loc[reject, "accepted_solution"] = 0
    result.loc[reject, "quality"] = "rejected_v9_multisource_conflict"
    result.loc[reject, "rejection_reason"] = "v9_cross_method_and_spatial_conflict"
    result.loc[reject, "solution_source"] = "not_accepted_v9_multisource_conflict"
    result.loc[reject, "v9_action"] = "rejected_multisource_conflict"
    audit["height_v9_m"] = result["height_est_m"].to_numpy()

    reliability = np.full(len(result), "limited_evidence", dtype=object)
    unavailable = result["height_est_m"].isna().to_numpy()
    reliability[unavailable] = "not_available"
    both = audit["strict_supported"].to_numpy() & audit["spatial_supported"].to_numpy() & ~unavailable
    reliability[both] = "supported_both"
    reliability[audit["strict_supported"].to_numpy() & ~both & ~unavailable] = "supported_cross_method"
    reliability[audit["spatial_supported"].to_numpy() & ~both & ~unavailable] = "supported_spatial"
    reliability[audit["strict_conflict"].to_numpy() & ~unavailable] = "review_cross_method_conflict"
    reliability[audit["spatial_conflict"].to_numpy() & ~unavailable] = "review_spatial_outlier"
    reliability[reject.to_numpy()] = "rejected_multisource_conflict"
    result["v9_reliability"] = reliability
    audit["v9_action"] = result["v9_action"].to_numpy()
    audit["v9_reliability"] = reliability
    result.to_csv(outputs["tables"] / "roof_only_building_heights.csv", index=False)
    audit.to_csv(outputs["tables"] / "roof_only_v9_audit.csv", index=False)
    result[result["v9_action"] != "kept_v8"].to_csv(outputs["tables"] / "roof_only_v9_changes.csv", index=False)
    geometry = vector[["fid", "clean_id", "geometry"]].merge(result, on=["fid", "clean_id"], validate="one_to_one")
    geometry.to_file(outputs["vectors"] / "roof_only_building_heights.gpkg", layer="roof_only_heights", driver="GPKG")
    map_stem = outputs["figures"] / "roof_only_v9_audited_height_map"
    plotter.plot_simple_height_map(vector[["fid", "geometry"]], result, map_stem, title="V9全区域建筑高度估计（跨方法与空间审计后）")
    outputs["picall_map_svg"].write_bytes(map_stem.with_suffix(".svg").read_bytes())
    audit_figure = outputs["figures"] / "图件_783851324222.svg"
    plot_audit(audit, result, audit_figure, plotter)
    outputs["picall_audit_svg"].write_bytes(audit_figure.read_bytes())
    risk_map = outputs["figures"] / "图件_604913619463.svg"
    plot_reliability_map(vector[["fid", "geometry"]], result, risk_map, plotter)
    outputs["picall_reliability_map_svg"].write_bytes(risk_map.read_bytes())
    finite = result[result["height_est_m"].notna()]
    summary = {
        "method": "v9_cross_method_and_spatial_morphology_audit",
        "buildings": int(len(result)),
        "finite_heights": int(len(finite)),
        "action_counts": {str(k): int(v) for k, v in result["v9_action"].value_counts().items()},
        "reliability_counts": {str(k): int(v) for k, v in result["v9_reliability"].value_counts().items()},
        "height_mean_m": float(finite["height_est_m"].mean()),
        "height_median_m": float(finite["height_est_m"].median()),
        "height_p95_m": float(finite["height_est_m"].quantile(0.95)),
        "height_max_m": float(finite["height_est_m"].max()),
        "cross_method_or_neighbor_height_fill_used": False,
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
    parser.add_argument("--config", type=Path, default=ROOT / "config_v9.json")
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
