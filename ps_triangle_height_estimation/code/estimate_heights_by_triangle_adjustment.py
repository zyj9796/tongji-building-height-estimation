from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"


def robust_adjustment(
    fraction: np.ndarray,
    observed_above_base: np.ndarray,
    base_weight: np.ndarray,
    iterations: int = 20,
    huber_c: float = 1.5,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Robust weighted least squares for y_i = f_i * H + error_i."""
    f = np.asarray(fraction, dtype=np.float64)
    y = np.asarray(observed_above_base, dtype=np.float64)
    w0 = np.asarray(base_weight, dtype=np.float64)
    denominator = float(np.sum(w0 * f * f))
    if denominator <= 1e-12:
        raise ValueError("height is not observable from the selected PS equations")
    height = float(np.sum(w0 * f * y) / denominator)
    robust_weight = np.ones_like(f)
    scale = float("nan")
    for _ in range(iterations):
        residual = y - f * height
        centered = residual - np.median(residual)
        scale = max(1.4826 * float(np.median(np.abs(centered))), 0.5)
        normalized = np.abs(residual) / (huber_c * scale)
        robust_weight = np.where(normalized <= 1.0, 1.0, 1.0 / np.maximum(normalized, 1e-12))
        weight = w0 * robust_weight
        denominator = float(np.sum(weight * f * f))
        if denominator <= 1e-12:
            raise ValueError("robust adjustment lost all geometric leverage")
        updated = float(np.sum(weight * f * y) / denominator)
        if abs(updated - height) < 1e-7:
            height = updated
            break
        height = updated
    residual = y - f * height
    return height, residual, w0 * robust_weight, scale


def robust_adjustment_with_wall_bias(
    fraction: np.ndarray,
    observed_above_base: np.ndarray,
    base_weight: np.ndarray,
    is_wall: np.ndarray,
    estimate_wall_bias: bool,
    iterations: int = 30,
    huber_c: float = 1.5,
    wall_bias_prior_sigma_m: float = 3.0,
) -> tuple[float, float, np.ndarray, np.ndarray, float]:
    """Heteroscedastic IRLS for y = f*H + I_wall*b_wall + error.

    The weak zero-centred ridge on ``b_wall`` absorbs systematic layover or
    side-facet assignment error without allowing wall observations to move the
    roof-supported height arbitrarily.
    """
    f = np.asarray(fraction, dtype=np.float64)
    y = np.asarray(observed_above_base, dtype=np.float64)
    w0 = np.asarray(base_weight, dtype=np.float64)
    wall = np.asarray(is_wall, dtype=np.float64)
    if estimate_wall_bias:
        design = np.column_stack([f, wall])
        ridge = np.diag([0.0, 1.0 / wall_bias_prior_sigma_m**2])
    else:
        design = f[:, None]
        ridge = np.zeros((1, 1), dtype=np.float64)

    def solve(weight: np.ndarray) -> np.ndarray:
        normal = design.T @ (weight[:, None] * design) + ridge
        right = design.T @ (weight * y)
        if np.linalg.cond(normal) > 1e10:
            raise ValueError("height/wall-bias normal matrix is ill-conditioned")
        return np.linalg.solve(normal, right)

    parameters = solve(w0)
    robust_weight = np.ones_like(f)
    scale = float("nan")
    for _ in range(iterations):
        residual = y - design @ parameters
        scale = max(
            1.4826 * float(np.median(np.abs(residual - np.median(residual)))),
            0.5,
        )
        normalized = np.abs(residual) / (huber_c * scale)
        robust_weight = np.where(
            normalized <= 1.0,
            1.0,
            1.0 / np.maximum(normalized, 1e-12),
        )
        updated = solve(w0 * robust_weight)
        if float(np.max(np.abs(updated - parameters))) < 1e-7:
            parameters = updated
            break
        parameters = updated
    residual = y - design @ parameters
    wall_bias = float(parameters[1]) if estimate_wall_bias else 0.0
    return (
        float(parameters[0]),
        wall_bias,
        residual,
        w0 * robust_weight,
        scale,
    )


def add_ps_quality_metrics(observations: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-fit PS reliability components used by the adjustment."""
    result = observations.copy()
    coherence = pd.to_numeric(result.coherence, errors="coerce").to_numpy(float)
    fraction = pd.to_numeric(
        result.vertical_fraction_of_prior, errors="coerce"
    ).to_numpy(float)
    building_candidates = np.maximum(
        pd.to_numeric(
            result.overlapping_building_candidates, errors="coerce"
        ).fillna(1.0),
        1.0,
    ).to_numpy(float)
    barycentric_min = (
        result[["bary_alpha", "bary_beta", "bary_gamma"]]
        .apply(pd.to_numeric, errors="coerce")
        .min(axis=1)
        .fillna(0.0)
        .to_numpy(float)
    )
    registration = (
        pd.to_numeric(
            result.get(
                "registration_reliability",
                pd.Series(0.35, index=result.index),
            ),
            errors="coerce",
        )
        .fillna(0.35)
        .clip(0.20, 1.0)
        .to_numpy(float)
    )
    is_roof = result.surface.to_numpy() == "roof"
    result["ps_coherence_score"] = np.clip(
        (coherence - 0.55) / (0.95 - 0.55), 0.05, 1.0
    )
    result["ps_ambiguity_score"] = 1.0 / np.sqrt(building_candidates)
    result["ps_triangle_interior_score"] = 0.35 + 0.65 * np.clip(
        barycentric_min / 0.12, 0.0, 1.0
    )
    result["ps_vertical_leverage_score"] = np.where(
        is_roof,
        1.0,
        np.clip((fraction - 0.10) / 0.55, 0.08, 1.0),
    )
    result["ps_registration_score"] = registration
    result["ps_surface_score"] = np.where(is_roof, 1.0, 0.82)
    components = result[
        [
            "ps_coherence_score",
            "ps_ambiguity_score",
            "ps_triangle_interior_score",
            "ps_vertical_leverage_score",
            "ps_registration_score",
            "ps_surface_score",
        ]
    ].to_numpy(float)
    result["ps_quality_pre_fit"] = np.exp(
        np.mean(np.log(np.clip(components, 1e-6, 1.0)), axis=1)
    )
    result["observation_sigma_m"] = (
        0.75
        + 3.25 * (1.0 - result.ps_coherence_score)
        + np.where(is_roof, 0.0, 0.75)
    )
    return result


def estimate_one_building(group: pd.DataFrame, base_elevation_m: float, minimum_fraction: float) -> tuple[dict, pd.DataFrame]:
    observations = add_ps_quality_metrics(group)
    observations["height_fraction"] = pd.to_numeric(observations["vertical_fraction_of_prior"], errors="coerce")
    observations["ps_elevation_m"] = pd.to_numeric(observations["height_m"], errors="coerce")
    observations["observed_above_base_m"] = observations.ps_elevation_m - base_elevation_m
    observations["individual_height_m"] = observations.observed_above_base_m / observations.height_fraction
    observations["equation_valid"] = (
        np.isfinite(observations.height_fraction)
        & np.isfinite(observations.ps_elevation_m)
        & (observations.height_fraction >= minimum_fraction)
        & (observations.observed_above_base_m > 0.0)
        & (observations.individual_height_m >= 3.0)
        & (observations.individual_height_m <= 150.0)
        & (observations.ps_quality_pre_fit >= 0.15)
    )
    valid = observations.loc[observations.equation_valid].copy()
    if valid.empty:
        raise ValueError("no valid roof/wall PS height equations")
    base_weight = (
        valid.ps_quality_pre_fit.to_numpy(dtype=np.float64) ** 2
        / valid.observation_sigma_m.to_numpy(dtype=np.float64) ** 2
    )
    is_wall = valid.surface.to_numpy() == "wall"
    estimate_wall_bias = bool(
        np.sum(~is_wall) >= 2
        and np.sum(is_wall) >= 3
        and np.nanstd(valid.loc[is_wall, "height_fraction"]) >= 0.08
    )
    height, wall_bias, residual, final_weight, robust_scale = (
        robust_adjustment_with_wall_bias(
            valid.height_fraction.to_numpy(),
            valid.observed_above_base_m.to_numpy(),
            base_weight,
            is_wall,
            estimate_wall_bias,
        )
    )
    inlier_threshold = max(2.5 * robust_scale, 1.5)
    inlier = np.abs(residual) <= inlier_threshold
    minimum_inliers = 3 if estimate_wall_bias else 2
    if np.sum(inlier) >= minimum_inliers and np.any(~inlier):
        retained_wall = is_wall[inlier]
        retained_bias_model = bool(
            estimate_wall_bias
            and np.sum(~retained_wall) >= 2
            and np.sum(retained_wall) >= 3
            and np.nanstd(valid.loc[inlier & is_wall, "height_fraction"]) >= 0.08
        )
        (
            height,
            wall_bias,
            residual_inlier,
            final_weight_inlier,
            robust_scale,
        ) = robust_adjustment_with_wall_bias(
            valid.loc[inlier, "height_fraction"].to_numpy(),
            valid.loc[inlier, "observed_above_base_m"].to_numpy(),
            base_weight[inlier],
            retained_wall,
            retained_bias_model,
        )
        estimate_wall_bias = retained_bias_model
        full_prediction = (
            valid.height_fraction.to_numpy() * height
            + is_wall.astype(float) * wall_bias
        )
        residual = valid.observed_above_base_m.to_numpy() - full_prediction
        final_weight = np.zeros(len(valid), dtype=np.float64)
        final_weight[inlier] = final_weight_inlier
    else:
        inlier = np.ones(len(valid), dtype=bool)

    valid["adjustment_residual_m"] = residual
    valid["adjustment_weight"] = final_weight
    valid["adjustment_inlier"] = inlier.astype(np.int8)
    valid["ps_residual_score"] = np.exp(
        -np.abs(residual) / max(2.5 * robust_scale, 1.5)
    )
    valid["ps_quality_score"] = (
        valid.ps_quality_pre_fit * valid.ps_residual_score
    ).clip(0.0, 1.0)
    valid["ps_quality_class"] = np.select(
        [
            (~inlier),
            valid.ps_quality_score >= 0.75,
            valid.ps_quality_score >= 0.50,
            valid.ps_quality_score >= 0.30,
        ],
        ["rejected", "A", "B", "C"],
        default="D",
    )
    quality_columns = [
        "ps_id",
        "adjustment_residual_m",
        "adjustment_weight",
        "adjustment_inlier",
        "ps_residual_score",
        "ps_quality_score",
        "ps_quality_class",
    ]
    observations = observations.merge(
        valid[quality_columns],
        on="ps_id",
        how="left",
    )
    observations["adjustment_inlier"] = (
        observations.adjustment_inlier.fillna(0).astype(np.int8)
    )
    observations["ps_quality_class"] = observations.ps_quality_class.fillna(
        "invalid"
    )

    used = valid.loc[inlier].copy()
    used_weight = final_weight[inlier]
    used_fraction = used.height_fraction.to_numpy(dtype=np.float64)
    used_residual = residual[inlier]
    n = len(used)
    effective_n = float(
        np.sum(used_weight) ** 2
        / max(float(np.sum(used_weight**2)), 1e-12)
    )
    used_wall = used.surface.to_numpy() == "wall"
    design = (
        np.column_stack([used_fraction, used_wall.astype(float)])
        if estimate_wall_bias
        else used_fraction[:, None]
    )
    normal = design.T @ (used_weight[:, None] * design)
    if estimate_wall_bias:
        normal[1, 1] += 1.0 / 9.0
    if n > design.shape[1] and np.linalg.cond(normal) < 1e10:
        variance = float(
            np.sum(used_weight * used_residual**2)
            / max(effective_n - design.shape[1], 1.0)
        )
        formal_se = float(
            np.sqrt(max(variance * np.linalg.inv(normal)[0, 0], 0.0))
        )
        corrected_individual = (
            used.observed_above_base_m.to_numpy(dtype=np.float64)
            - used_wall.astype(float) * wall_bias
        ) / used_fraction
        robust_sem = (
            1.4826
            * float(
                np.median(
                    np.abs(corrected_individual - np.median(corrected_individual))
                )
            )
            / np.sqrt(max(effective_n, 1.0))
        )
        uncertainty = max(0.5, formal_se, robust_sem)
    else:
        formal_se = float("nan")
        uncertainty = 10.0
    inlier_ratio = float(n / len(valid))
    plausible = bool(3.0 <= height <= 150.0)
    residual_rms = (
        float(
            np.sqrt(
                np.average(
                    used_residual**2,
                    weights=np.maximum(used_weight, 1e-12),
                )
            )
        )
        if n
        else np.nan
    )
    roof_implied = used.loc[~used_wall, "individual_height_m"]
    wall_implied = (
        (
            used.loc[used_wall, "observed_above_base_m"] - wall_bias
        )
        / used.loc[used_wall, "height_fraction"]
    )
    cross_surface_difference = (
        float(abs(roof_implied.median() - wall_implied.median()))
        if len(roof_implied) and len(wall_implied)
        else np.nan
    )
    quality_counts = used.ps_quality_class.value_counts()
    strong_ps = int(quality_counts.get("A", 0) + quality_counts.get("B", 0))
    if (
        n >= 6
        and effective_n >= 4.0
        and strong_ps >= 4
        and int(np.sum(~used_wall)) >= 1
        and uncertainty <= 3.0
        and residual_rms <= 5.0
        and inlier_ratio >= 0.75
        and abs(wall_bias) <= 8.0
        and plausible
    ):
        quality = "high"
    elif (
        n >= 3
        and effective_n >= 2.2
        and strong_ps >= 2
        and uncertainty <= 8.0
        and residual_rms <= 10.0
        and abs(wall_bias) <= 12.0
        and plausible
    ):
        quality = "medium"
    else:
        quality = "low"
    row = {
        "fid": int(group.fid.iloc[0]),
        "clean_id": int(group.clean_id.iloc[0]),
        "height_prior_m": float(group.height_prior_m.iloc[0]),
        "height_est_m": float(height) if plausible else np.nan,
        "roof_elevation_est_m": float(base_elevation_m + height) if plausible else np.nan,
        "height_uncertainty_m": float(uncertainty),
        "formal_standard_error_m": formal_se,
        "quality": quality,
        "ps_mapped": int(len(group)),
        "ps_equations_valid": int(len(valid)),
        "ps_equations_used": int(n),
        "roof_ps_used": int(np.sum(used.surface == "roof")),
        "wall_ps_used": int(np.sum(used.surface == "wall")),
        "effective_ps_count": effective_n,
        "strong_ps_count": strong_ps,
        "ps_quality_a": int(quality_counts.get("A", 0)),
        "ps_quality_b": int(quality_counts.get("B", 0)),
        "ps_quality_c": int(quality_counts.get("C", 0)),
        "wall_bias_est_m": float(wall_bias),
        "wall_bias_model_used": int(estimate_wall_bias),
        "roof_wall_height_difference_m": cross_surface_difference,
        "inlier_ratio": inlier_ratio,
        "weighted_residual_rms_m": residual_rms,
        "minimum_height_fraction": float(minimum_fraction),
        "base_elevation_m": float(base_elevation_m),
    }
    return row, observations


def run(
    config_path: Path,
    mapped_csv: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_elevation_m = float(config["base_elevation_m"])
    output_dir = output_dir.resolve() if output_dir else ROOT / "results" / "triangle_adjustment_height_estimation"
    output_dir.mkdir(parents=True, exist_ok=True)
    mapped_path = mapped_csv.resolve() if mapped_csv else ROOT / "results" / "tables" / "ps_building_surface_coordinates.csv"
    mapped = pd.read_csv(mapped_path)
    ps_height_path = (ROOT / config["inputs"]["ps_points"]).resolve()
    ps_height_columns = pd.read_csv(ps_height_path, nrows=0).columns
    ps_height = pd.read_csv(
        ps_height_path,
        usecols=["ps_id", "height_m", "z_dsm_m"],
    )
    uses_wusong_psi_elevation = (
        "psi_scatterer_elevation_wusong_m" in ps_height_columns
    )
    height_equals_z_dsm = bool(
        np.allclose(
            pd.to_numeric(ps_height.height_m, errors="coerce"),
            pd.to_numeric(ps_height.z_dsm_m, errors="coerce"),
            rtol=0.0,
            atol=1e-9,
            equal_nan=False,
        )
    )
    equations = mapped.merge(ps_height, on="ps_id", how="left", validate="one_to_one")
    minimum_fraction = 0.10
    estimates: list[dict] = []
    equation_tables: list[pd.DataFrame] = []
    failures: list[dict] = []
    for fid, group in equations.groupby("fid", sort=True):
        try:
            estimate, equation_table = estimate_one_building(group, base_elevation_m, minimum_fraction)
            estimates.append(estimate)
            equation_tables.append(equation_table)
        except Exception as exc:
            failures.append({"fid": int(fid), "reason": str(exc)})
            assessed = add_ps_quality_metrics(group)
            assessed["height_fraction"] = pd.to_numeric(
                assessed["vertical_fraction_of_prior"], errors="coerce"
            )
            assessed["ps_elevation_m"] = pd.to_numeric(
                assessed["height_m"], errors="coerce"
            )
            assessed["observed_above_base_m"] = (
                assessed.ps_elevation_m - base_elevation_m
            )
            assessed["individual_height_m"] = (
                assessed.observed_above_base_m / assessed.height_fraction
            )
            assessed["equation_valid"] = False
            assessed["adjustment_residual_m"] = np.nan
            assessed["adjustment_weight"] = 0.0
            assessed["adjustment_inlier"] = np.int8(0)
            assessed["ps_residual_score"] = np.nan
            assessed["ps_quality_score"] = 0.0
            assessed["ps_quality_class"] = "invalid"
            equation_tables.append(assessed)

    buildings = gpd.read_file((ROOT / config["inputs"]["buildings"]).resolve()).reset_index(drop=True)
    buildings["building_fid"] = np.arange(len(buildings), dtype=np.int64)
    buildings = buildings.rename(columns={config["height_field"]: "height_shp_m"})
    estimates_table = pd.DataFrame(estimates)
    all_fids = pd.DataFrame({"fid": np.arange(len(buildings), dtype=np.int64)})
    estimates_table = all_fids.merge(estimates_table, on="fid", how="left")
    estimates_table["quality"] = estimates_table.quality.fillna("no_ps_equation")
    estimates_csv = output_dir / "building_heights_triangle_adjustment.csv"
    estimates_table.to_csv(estimates_csv, index=False)
    equation_table = pd.concat(equation_tables, ignore_index=True) if equation_tables else pd.DataFrame()
    equations_csv = output_dir / "ps_triangle_height_equations.csv"
    equation_table.to_csv(equations_csv, index=False)
    failures_csv = output_dir / "triangle_adjustment_failures.csv"
    pd.DataFrame(failures, columns=["fid", "reason"]).to_csv(failures_csv, index=False)

    attributes = estimates_table.drop(columns=["clean_id"], errors="ignore").rename(columns={"fid": "building_fid"})
    vector = buildings.merge(attributes, on="building_fid", how="left")
    vector_path = output_dir / "building_heights_triangle_adjustment.gpkg"
    vector.to_file(vector_path, layer="triangle_adjustment_heights", driver="GPKG")

    finite = estimates_table.loc[np.isfinite(estimates_table.height_est_m)].copy()
    summary = {
        "method": "ps_quality_weighted_heteroscedastic_wall_bias_robust_triangle_adjustment",
        "observation_equation": (
            "ps_elevation_m - base_elevation_m = height_fraction * "
            "building_height_m + I_wall * wall_bias_m + error"
        ),
        "roof_ps_height_fraction": 1.0,
        "wall_ps_height_fraction": "sum of barycentric weights belonging to top vertices",
        "base_elevation_m": base_elevation_m,
        "ps_height_field_used": "height_m",
        "ps_height_provenance_warning": (
            "height_m is normalized from the internally ground-referenced PSI "
            "scatterer elevation in the project Wusong datum; it has no independent "
            "LiDAR, GNSS, or leveling validation."
            if uses_wusong_psi_elevation
            else (
                "height_m equals z_dsm_m in the current PS file; results are "
                "DSM-assisted and cannot be independently validated against the same DSM."
                if height_equals_z_dsm
                else (
                    "height_m differs from z_dsm_m; consult the normalized PS input "
                    "and route configuration for the explicit height construction."
                )
            )
        ),
        "input_buildings": int(len(buildings)),
        "mapped_ps": int(len(mapped)),
        "mapped_ps_source": str(mapped_path),
        "buildings_with_mapped_ps": int(mapped.fid.nunique()),
        "estimated_buildings": int(len(finite)),
        "failed_building_adjustments": int(len(failures)),
        "quality_counts": estimates_table.quality.value_counts().to_dict(),
        "ps_quality_counts": (
            equation_table.ps_quality_class.value_counts().to_dict()
            if len(equation_table)
            else {}
        ),
        "weight_components": [
            "coherence",
            "building_overlap_ambiguity",
            "triangle_interior_position",
            "vertical_height_leverage",
            "roof_registration_reliability",
            "surface_type",
            "robust_residual",
        ],
        "height_m": {
            "min": float(finite.height_est_m.min()) if len(finite) else None,
            "median": float(finite.height_est_m.median()) if len(finite) else None,
            "mean": float(finite.height_est_m.mean()) if len(finite) else None,
            "max": float(finite.height_est_m.max()) if len(finite) else None,
        },
        "outputs": {
            "building_estimates_csv": str(estimates_csv),
            "ps_equations_csv": str(equations_csv),
            "building_vector_gpkg": str(vector_path),
            "failures_csv": str(failures_csv),
        },
    }
    summary_path = output_dir / "triangle_adjustment_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate building heights from known PS elevations and registered triangle barycentric geometry.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mapped-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.config.resolve(), args.mapped_csv, args.output_dir)


if __name__ == "__main__":
    main()
