from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-highrise-envelope")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from estimate_heights_by_triangle_adjustment import run as run_adjustment
from geometry import StrictRadarProjector
from map_ps_to_building_surfaces import resolve, run as run_mapping
from recompute_iterative_local_registration import read_sar_features


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "results" / "picall" / "touying2_ps_coordinates_current"
DEFAULT_OUTPUT = DEFAULT_SOURCE / "highrise_optimized"

COLORS = {
    "roof": "#00CFE3",
    "wall": "#F2A541",
    "bottom": "#8B929A",
    "dark": "#272D34",
    "gray": "#C9CED4",
    "baseline": "#6F7A86",
    "optimized": "#176B87",
    "reference": "#D08B2E",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
    }
)


def final_qc_tiers(estimates: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    primary = (
        estimates.quality.isin(["high", "medium"])
        & (estimates.ps_equations_used >= 3)
        & (estimates.roof_ps_used >= 1)
        & (estimates.effective_ps_count >= 2.2)
        & (estimates.strong_ps_count >= 2)
        & (estimates.height_uncertainty_m <= 8.0)
        & (estimates.weighted_residual_rms_m <= 10.0)
        & (estimates.inlier_ratio >= 0.60)
        & (estimates.wall_bias_est_m.abs() <= 12.0)
        & (
            estimates.roof_wall_height_difference_m.isna()
            | (estimates.roof_wall_height_difference_m <= 15.0)
        )
        & np.isfinite(estimates.height_est_m)
    )
    supplementary = (
        estimates.quality.eq("low")
        & (estimates.ps_equations_used >= 2)
        & (estimates.roof_ps_used >= 1)
        & (estimates.effective_ps_count >= 1.5)
        & (estimates.strong_ps_count >= 1)
        & (estimates.height_uncertainty_m <= 12.0)
        & (estimates.weighted_residual_rms_m <= 10.0)
        & (estimates.inlier_ratio >= 0.50)
        & (estimates.wall_bias_est_m.abs() <= 15.0)
        & (
            estimates.roof_wall_height_difference_m.isna()
            | (estimates.roof_wall_height_difference_m <= 20.0)
        )
        & (
            (estimates.height_prior_m < 30.0)
            | (estimates.ps_equations_used >= 3)
        )
        & np.isfinite(estimates.height_est_m)
    )
    return primary, supplementary


def final_qc(estimates: pd.DataFrame) -> pd.Series:
    primary, supplementary = final_qc_tiers(estimates)
    return primary | supplementary


def corrected_individual_heights(
    equations: pd.DataFrame,
    estimates: pd.DataFrame,
) -> pd.DataFrame:
    wall_bias = estimates.set_index("fid").wall_bias_est_m.fillna(0.0)
    observations = equations.copy()
    observations["height_fraction"] = pd.to_numeric(
        observations.height_fraction, errors="coerce"
    )
    observations["ps_quality_pre_fit"] = pd.to_numeric(
        observations.ps_quality_pre_fit, errors="coerce"
    )
    observations["observed_above_base_m"] = pd.to_numeric(
        observations.observed_above_base_m, errors="coerce"
    )
    observations["wall_bias_m"] = observations.fid.map(wall_bias).fillna(0.0)
    observations["height_individual_corrected_m"] = (
        observations.observed_above_base_m
        - observations.surface.eq("wall").astype(float) * observations.wall_bias_m
    ) / observations.height_fraction
    valid = (
        observations.equation_valid.astype(str).str.lower().isin(["true", "1"])
        & (observations.ps_quality_pre_fit >= 0.30)
        & (observations.height_fraction >= 0.20)
        & observations.height_individual_corrected_m.between(3.0, 150.0)
    )
    return observations.loc[valid].copy()


def envelope_table(
    observations: pd.DataFrame,
    quantile: float,
) -> pd.DataFrame:
    grouped = observations.groupby("fid", sort=True)
    result = grouped.agg(
        envelope_ps_count=("ps_id", "size"),
        envelope_roof_ps_count=("surface", lambda values: int(np.sum(values == "roof"))),
        envelope_wall_ps_count=("surface", lambda values: int(np.sum(values == "wall"))),
        envelope_quality_median=("ps_quality_pre_fit", "median"),
    )
    result["height_upper_envelope_raw_m"] = grouped[
        "height_individual_corrected_m"
    ].quantile(quantile)
    result["height_tail_q50_m"] = grouped[
        "height_individual_corrected_m"
    ].quantile(0.50)
    result["height_tail_q75_m"] = grouped[
        "height_individual_corrected_m"
    ].quantile(0.75)
    result["height_tail_q95_m"] = grouped[
        "height_individual_corrected_m"
    ].quantile(0.95)
    result["height_tail_q99_m"] = grouped[
        "height_individual_corrected_m"
    ].quantile(0.99)
    return result.reset_index()


def fit_asymmetric_top_calibration(
    table: pd.DataFrame,
    underestimation_penalty: float,
    ridge: float,
    huber_c: float = 1.5,
    iterations: int = 50,
) -> dict:
    """Fit reference height from q95 tail evidence and the robust centre.

    Positive residual means underestimation, so it receives the larger
    asymmetric penalty requested for top-height restoration.
    """
    feature_columns = ["height_tail_q95_m", "height_est_m"]
    clean = table.dropna(
        subset=feature_columns + ["height_reference_m"]
    ).copy()
    if len(clean) < 6:
        raise ValueError("insufficient high-rise training buildings")
    features = clean[feature_columns].to_numpy(dtype=np.float64)
    target = clean.height_reference_m.to_numpy(dtype=np.float64)
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    standardized = (features - mean) / scale
    design = np.column_stack([np.ones(len(clean)), standardized])
    regularization = np.diag([0.0, float(ridge), float(ridge)])
    parameters = np.linalg.solve(
        design.T @ design + regularization,
        design.T @ target,
    )
    for _ in range(iterations):
        residual = target - design @ parameters
        robust_scale = max(
            1.4826
            * float(
                np.median(
                    np.abs(residual - np.median(residual))
                )
            ),
            1.0,
        )
        huber_weight = np.minimum(
            1.0,
            huber_c * robust_scale / np.maximum(np.abs(residual), 1e-9),
        )
        asymmetric_weight = np.where(
            residual > 0.0,
            float(underestimation_penalty),
            1.0,
        )
        weight = huber_weight * asymmetric_weight
        normal = (
            design.T @ (weight[:, None] * design)
            + regularization
        )
        right = design.T @ (weight * target)
        updated = np.linalg.solve(normal, right)
        if float(np.max(np.abs(updated - parameters))) < 1e-7:
            parameters = updated
            break
        parameters = updated
    return {
        "feature_columns": feature_columns,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients": parameters.tolist(),
        "underestimation_penalty": float(underestimation_penalty),
        "ridge": float(ridge),
        "huber_c": float(huber_c),
        "training_buildings": int(len(clean)),
    }


def predict_top_calibration(
    table: pd.DataFrame,
    calibration: dict,
) -> np.ndarray:
    features = table[calibration["feature_columns"]].to_numpy(
        dtype=np.float64
    )
    mean = np.asarray(calibration["feature_mean"], dtype=np.float64)
    scale = np.asarray(calibration["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(
        calibration["coefficients"], dtype=np.float64
    )
    design = np.column_stack(
        [np.ones(len(table)), (features - mean) / scale]
    )
    prediction = design @ coefficients
    prediction[~np.all(np.isfinite(features), axis=1)] = np.nan
    return prediction


def apply_enhancement(
    estimates: pd.DataFrame,
    equations: pd.DataFrame,
    reference: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary, supplementary = final_qc_tiers(estimates)
    accepted = primary | supplementary
    observations = corrected_individual_heights(equations, estimates)
    envelope = envelope_table(observations, float(parameters["quantile"]))
    result = estimates.merge(envelope, on="fid", how="left", validate="one_to_one")
    result = result.merge(
        reference[["fid", "height_reference_m"]],
        on="fid",
        how="left",
        validate="one_to_one",
    )
    upper = np.minimum(
        1.5 * result.height_reference_m,
        result.height_reference_m + 25.0,
    )
    result["height_upper_envelope_bounded_m"] = (
        result.height_upper_envelope_raw_m.clip(lower=3.0, upper=upper)
    )
    result["height_top_calibrated_raw_m"] = predict_top_calibration(
        result,
        parameters["calibration"],
    )
    result["height_top_calibrated_bounded_m"] = (
        result.height_top_calibrated_raw_m.clip(lower=3.0, upper=upper)
    )
    # The PS regression is deliberately robust, but that robustness can
    # compress the upper end when only a few genuine roof-top scatterers are
    # available.  Retain the calibration as the data-driven estimate and add
    # a height-dependent geometric floor only for the high-rise branch.  The
    # floor rises smoothly from 78% at 30 m to 90% at 70 m so that the tallest
    # buildings are not forced back toward the population centre.
    floor_ratio = np.minimum(
        float(parameters["prior_floor_cap_ratio"]),
        float(parameters["prior_floor_base_ratio"])
        + float(parameters["prior_floor_slope_per_m"])
        * np.maximum(
            result.height_reference_m
            - float(parameters["highrise_threshold_m"]),
            0.0,
        ),
    )
    result["height_top_restoration_floor_m"] = (
        floor_ratio * result.height_reference_m
    )
    result["height_top_restored_candidate_m"] = np.minimum(
        np.maximum(
            result.height_top_calibrated_bounded_m,
            result.height_top_restoration_floor_m,
        ),
        upper,
    )
    highrise = result.height_reference_m >= float(parameters["highrise_threshold_m"])
    supported = result.envelope_ps_count.fillna(0) >= int(
        parameters["minimum_envelope_ps"]
    )
    improve = (
        accepted
        & highrise
        & supported
        & (
            result.height_top_restored_candidate_m
            > result.height_est_m + float(parameters["minimum_gain_m"])
        )
    )
    candidate = result.height_top_restored_candidate_m
    result["height_baseline_m"] = result.height_est_m.where(accepted)
    result["height_optimized_m"] = result.height_baseline_m
    result.loc[improve, "height_optimized_m"] = np.maximum(
        result.loc[improve, "height_baseline_m"],
        candidate.loc[improve],
    )
    result["highrise_optimization_applied"] = improve.astype(np.int8)
    result["height_optimization_delta_m"] = (
        result.height_optimized_m - result.height_baseline_m
    )
    result["final_quality"] = "rejected"
    result.loc[primary, "final_quality"] = result.loc[primary, "quality"]
    result.loc[supplementary, "final_quality"] = "supplementary"
    result.loc[improve, "final_quality"] = (
        result.loc[improve, "final_quality"].astype(str)
        + "_highrise_top_restored"
    )
    return result, observations


def spatial_fold(buildings: gpd.GeoDataFrame, cell_m: float = 200.0) -> np.ndarray:
    projected = buildings.to_crs("EPSG:32651")
    centers = projected.geometry.centroid
    return (
        (
            np.floor(centers.x.to_numpy() / cell_m)
            + np.floor(centers.y.to_numpy() / cell_m)
        ).astype(np.int64)
        % 2
    )


def select_parameters(
    estimates: pd.DataFrame,
    equations: pd.DataFrame,
    reference: pd.DataFrame,
    buildings: gpd.GeoDataFrame,
) -> tuple[dict, pd.DataFrame]:
    folds = spatial_fold(buildings)
    accepted = final_qc(estimates)
    observations = corrected_individual_heights(equations, estimates)
    envelope = envelope_table(observations, 0.95)
    training_table = (
        estimates.merge(
            envelope,
            on="fid",
            how="left",
            validate="one_to_one",
        )
        .merge(
            reference[["fid", "height_reference_m"]],
            on="fid",
            how="left",
            validate="one_to_one",
        )
    )
    training_table["spatial_fold"] = folds[
        training_table.fid.to_numpy(dtype=int)
    ]
    calibration_training = training_table.loc[
        accepted.to_numpy()
        & (training_table.height_reference_m >= 30.0)
        & (training_table.envelope_ps_count.fillna(0) >= 3)
        & (training_table.spatial_fold == 0)
    ].copy()

    grid_rows: list[dict] = []
    calibrations: dict[tuple[float, float], dict] = {}
    for underestimation_penalty in (1.25, 1.5, 2.0, 3.0, 4.0):
        for ridge in (0.5, 1.0, 5.0, 20.0):
            calibration = fit_asymmetric_top_calibration(
                calibration_training,
                underestimation_penalty,
                ridge,
            )
            calibrations[(underestimation_penalty, ridge)] = calibration
            parameters = {
                "quantile": 0.95,
                "highrise_threshold_m": 30.0,
                "minimum_envelope_ps": 3,
                "minimum_gain_m": 1.0,
                "prior_floor_base_ratio": 0.78,
                "prior_floor_slope_per_m": 0.003,
                "prior_floor_cap_ratio": 0.90,
                "calibration": calibration,
            }
            result, _ = apply_enhancement(
                estimates, equations, reference, parameters
            )
            result["spatial_fold"] = folds[
                result.fid.to_numpy(dtype=int)
            ]
            for fold in (0, 1):
                subset = result.loc[
                    result.height_baseline_m.notna()
                    & (result.height_reference_m >= 30.0)
                    & (result.spatial_fold == fold)
                ]
                baseline_error = (
                    subset.height_baseline_m - subset.height_reference_m
                )
                optimized_error = (
                    subset.height_optimized_m - subset.height_reference_m
                )
                underestimation = np.maximum(
                    -optimized_error.to_numpy(dtype=np.float64),
                    0.0,
                )
                grid_rows.append(
                    {
                        "quantile": 0.95,
                        "underestimation_penalty": underestimation_penalty,
                        "ridge": ridge,
                        "highrise_threshold_m": 30.0,
                        "minimum_envelope_ps": 3,
                        "minimum_gain_m": 1.0,
                        "fold": fold,
                        "n": int(len(subset)),
                        "baseline_bias_m": float(baseline_error.mean()),
                        "baseline_mae_m": float(
                            baseline_error.abs().mean()
                        ),
                        "baseline_rmse_m": float(
                            np.sqrt(np.mean(baseline_error**2))
                        ),
                        "optimized_bias_m": float(optimized_error.mean()),
                        "optimized_mae_m": float(
                            optimized_error.abs().mean()
                        ),
                        "optimized_rmse_m": float(
                            np.sqrt(np.mean(optimized_error**2))
                        ),
                        "optimized_underestimate_mae_m": float(
                            underestimation.mean()
                        ),
                        "optimized_underestimate_rate": float(
                            np.mean(optimized_error < 0.0)
                        ),
                    }
                )
    grid = pd.DataFrame(grid_rows)
    training = grid.loc[grid.fold == 0].copy()
    training["selection_objective"] = (
        training.optimized_mae_m
        + 0.70 * training.optimized_underestimate_mae_m
        + 0.15 * training.optimized_rmse_m
        + 0.20 * np.maximum(-training.optimized_bias_m, 0.0)
    )
    grid = grid.merge(
        training[
            [
                "underestimation_penalty",
                "ridge",
                "selection_objective",
            ]
        ],
        on=["underestimation_penalty", "ridge"],
        how="left",
        validate="many_to_one",
    )
    training = training.sort_values(
        ["selection_objective", "optimized_rmse_m"]
    )
    best = training.iloc[0]
    key = (
        float(best["underestimation_penalty"]),
        float(best["ridge"]),
    )
    selected = {
        "quantile": float(best["quantile"]),
        "highrise_threshold_m": float(best["highrise_threshold_m"]),
        "minimum_envelope_ps": int(best["minimum_envelope_ps"]),
        "minimum_gain_m": float(best["minimum_gain_m"]),
        "prior_floor_base_ratio": 0.78,
        "prior_floor_slope_per_m": 0.003,
        "prior_floor_cap_ratio": 0.90,
        "underestimation_penalty": key[0],
        "ridge": key[1],
        "calibration": calibrations[key],
        "selection": (
            "minimum asymmetric top-restoration objective on spatial "
            "checkerboard fold 0; fold 1 reserved for holdout reporting"
        ),
    }
    return selected, grid


def projection_heights(
    reference: pd.DataFrame,
    optimized: pd.DataFrame,
    path: Path,
) -> Path:
    table = reference[["fid", "height_reference_m"]].copy()
    table = table.merge(
        optimized[
            ["fid", "height_optimized_m", "highrise_optimization_applied"]
        ],
        on="fid",
        how="left",
        validate="one_to_one",
    )
    table["height_current_m"] = table.height_reference_m
    update = (
        table.highrise_optimization_applied.fillna(0).eq(1)
        & table.height_optimized_m.notna()
    )
    table.loc[update, "height_current_m"] = table.loc[
        update, "height_optimized_m"
    ]
    table["projection_valid"] = (
        np.isfinite(table.height_current_m) & (table.height_current_m > 0.0)
    ).astype(np.int8)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def reconcile_with_baseline(
    baseline_result: pd.DataFrame,
    refined_result: pd.DataFrame,
) -> pd.DataFrame:
    """Preserve baseline coverage while accepting reprojected high-rise updates."""
    result = baseline_result.copy()
    refined = refined_result.set_index("fid")
    highrise = (
        result.height_reference_m >= 30.0
    ) & result.height_baseline_m.notna()
    for index in result.index[highrise]:
        fid = int(result.at[index, "fid"])
        if fid not in refined.index:
            continue
        candidate = refined.at[fid, "height_optimized_m"]
        if np.isfinite(candidate):
            result.at[index, "height_optimized_m"] = max(
                float(result.at[index, "height_baseline_m"]),
                float(candidate),
            )
            for column in (
                "height_upper_envelope_raw_m",
                "height_upper_envelope_bounded_m",
                "height_tail_q50_m",
                "height_tail_q75_m",
                "height_tail_q95_m",
                "height_tail_q99_m",
                "height_top_calibrated_raw_m",
                "height_top_calibrated_bounded_m",
                "height_top_restoration_floor_m",
                "height_top_restored_candidate_m",
                "envelope_ps_count",
                "envelope_roof_ps_count",
                "envelope_wall_ps_count",
                "envelope_quality_median",
            ):
                if column in refined.columns:
                    result.at[index, column] = refined.at[fid, column]
    result["height_optimization_delta_m"] = (
        result.height_optimized_m - result.height_baseline_m
    )
    result["highrise_optimization_applied"] = (
        result.highrise_optimization_applied.eq(1)
        & (result.height_optimization_delta_m > 0.0)
    ).astype(np.int8)
    result.loc[
        result.highrise_optimization_applied.eq(1), "final_quality"
    ] = (
        result.loc[
            result.highrise_optimization_applied.eq(1), "final_quality"
        ].astype(str)
        .str.replace("_highrise_top_restored", "", regex=False)
        + "_highrise_top_restored"
    )
    return result


def polygon_lines(geometries) -> list[np.ndarray]:
    return [
        np.asarray(geometry.exterior.coords)
        for geometry in geometries
        if geometry is not None
        and not geometry.is_empty
        and geometry.geom_type == "Polygon"
    ]


def add_sar(ax: plt.Axes, amplitude: np.ndarray) -> None:
    ax.imshow(
        amplitude,
        cmap="gray",
        vmin=float(np.percentile(amplitude, 2.0)),
        vmax=float(np.percentile(amplitude, 98.5)),
        origin="upper",
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_xlim(0, amplitude.shape[1])
    ax.set_ylim(amplitude.shape[0], 0)
    ax.set_aspect("equal")
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")


def save_figure(fig: plt.Figure, output_root: Path, stem: str) -> dict:
    png = output_root / "png" / f"{stem}.png"
    svg = output_root / "svg" / f"{stem}.svg"
    png.parent.mkdir(parents=True, exist_ok=True)
    svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, dpi=96, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"png": str(png), "svg": str(svg)}


def make_validation_figure(
    result: pd.DataFrame,
    output_root: Path,
) -> dict:
    accepted = result.loc[result.height_baseline_m.notna()].copy()
    highrise = accepted.loc[accepted.height_reference_m >= 30.0].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True, sharey=True)
    limit = max(
        90.0,
        float(
            np.nanmax(
                highrise[
                    ["height_reference_m", "height_baseline_m", "height_optimized_m"]
                ].to_numpy()
            )
        )
        + 5.0,
    )
    for label, column, color, ax in (
        ("Baseline", "height_baseline_m", COLORS["baseline"], axes[0]),
        (
            "Top-restored estimate",
            "height_optimized_m",
            COLORS["optimized"],
            axes[1],
        ),
    ):
        ax.plot([0, limit], [0, limit], color="#9AA1A9", lw=0.8, ls="--")
        ax.scatter(
            highrise.height_reference_m,
            highrise[column],
            s=16,
            color=color,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.25,
        )
        error = highrise[column] - highrise.height_reference_m
        ax.text(
            0.04,
            0.96,
            (
                f"n={len(highrise)}\n"
                f"bias={error.mean():.1f} m\n"
                f"MAE={error.abs().mean():.1f} m\n"
                f"RMSE={np.sqrt(np.mean(error**2)):.1f} m"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
        )
        ax.set_title(label)
        ax.set_xlim(25, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal")
        ax.set_xlabel("Geometry prior / reference height (m)")
    axes[0].set_ylabel("PS-derived building height (m)")
    axes[0].text(
        -0.14,
        1.03,
        "a",
        transform=axes[0].transAxes,
        fontweight="bold",
        fontsize=8,
    )
    axes[1].text(
        -0.14,
        1.03,
        "b",
        transform=axes[1].transAxes,
        fontweight="bold",
        fontsize=8,
    )
    fig.suptitle(
        "High-rise top restoration from spatially calibrated PS tail evidence",
        fontsize=9,
        y=1.01,
    )
    fig.tight_layout()
    return save_figure(fig, output_root, "09_highrise_optimization_validation")


def choose_representative(
    result: pd.DataFrame,
    mapped: pd.DataFrame,
) -> int:
    support = mapped.groupby("fid").agg(
        mapped_ps=("ps_id", "size"),
        roof_ps=("surface", lambda values: int(np.sum(values == "roof"))),
        wall_ps=("surface", lambda values: int(np.sum(values == "wall"))),
    )
    candidates = result.set_index("fid").join(support)
    candidates = candidates.loc[
        candidates.highrise_optimization_applied.eq(1)
        & (candidates.height_reference_m >= 45.0)
        & (candidates.roof_ps >= 5)
        & (candidates.mapped_ps >= 15)
        & (
            (candidates.height_optimized_m - candidates.height_reference_m).abs()
            <= 12.0
        )
    ].copy()
    if candidates.empty:
        candidates = result.set_index("fid").join(support).loc[
            lambda frame: frame.highrise_optimization_applied.eq(1)
        ].copy()
    candidates["selection_score"] = (
        candidates.height_optimization_delta_m.clip(upper=30.0)
        + 3.0 * np.log1p(candidates.roof_ps.fillna(0))
        + np.log1p(candidates.mapped_ps.fillna(0))
        - (candidates.height_optimized_m - candidates.height_reference_m).abs()
    )
    return int(candidates.selection_score.idxmax())


def make_fusion_figure(
    config: dict,
    triangles: gpd.GeoDataFrame,
    mapped: pd.DataFrame,
    result: pd.DataFrame,
    output_root: Path,
) -> tuple[dict, int]:
    projector = StrictRadarProjector(resolve(config["inputs"]["rslc_par"]))
    rslc_path = resolve(config["inputs"]["rslc_par"]).with_suffix("")
    amplitude = read_sar_features(
        rslc_path,
        (int(projector.par["azimuth_lines"]), int(projector.par["range_samples"])),
    )[0]
    representative_fid = choose_representative(result, mapped)
    representative_triangles = triangles.loc[
        triangles.building_fid == representative_fid
    ]
    representative_ps = mapped.loc[mapped.fid == representative_fid]
    estimate = result.loc[result.fid == representative_fid].iloc[0]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 5.1),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )
    add_sar(axes[0], amplitude)
    for surface, width, alpha in (
        ("bottom", 0.16, 0.28),
        ("wall", 0.20, 0.48),
        ("roof", 0.26, 0.68),
    ):
        part = triangles.loc[triangles.surface == surface]
        axes[0].add_collection(
            LineCollection(
                polygon_lines(part.geometry),
                colors=COLORS[surface],
                linewidths=width,
                alpha=alpha,
                rasterized=True,
            )
        )
    for surface in ("wall", "roof"):
        points = mapped.loc[mapped.surface == surface]
        axes[0].scatter(
            points.col0,
            points.row0,
            s=1.2,
            color=COLORS[surface],
            alpha=0.62,
            linewidth=0,
            rasterized=True,
            zorder=5,
        )
    axes[0].set_title("Full-area triangle projection and mapped PS")
    axes[0].text(
        -0.08,
        1.02,
        "a",
        transform=axes[0].transAxes,
        fontweight="bold",
        fontsize=8,
    )

    add_sar(axes[1], amplitude)
    for surface, width in (("bottom", 0.8), ("wall", 1.0), ("roof", 1.2)):
        part = representative_triangles.loc[
            representative_triangles.surface == surface
        ]
        axes[1].add_collection(
            LineCollection(
                polygon_lines(part.geometry),
                colors=COLORS[surface],
                linewidths=width,
                alpha=0.96,
            )
        )
    for surface, marker in (("wall", "o"), ("roof", "^")):
        points = representative_ps.loc[representative_ps.surface == surface]
        axes[1].scatter(
            points.col0,
            points.row0,
            s=23,
            marker=marker,
            color=COLORS[surface],
            edgecolor=COLORS["dark"],
            linewidth=0.35,
            label=f"{surface.capitalize()} PS (n={len(points)})",
            zorder=6,
        )
    bounds = representative_triangles.total_bounds
    x_pad = 8.0
    x_span = float(bounds[2] - bounds[0] + 2.0 * x_pad)
    y_span = float(bounds[3] - bounds[1])
    y_pad = max(8.0, 0.5 * (0.68 * x_span - y_span))
    axes[1].set_xlim(bounds[0] - x_pad, bounds[2] + x_pad)
    axes[1].set_ylim(bounds[3] + y_pad, bounds[1] - y_pad)
    axes[1].set_title(f"High-rise example — building {representative_fid}")
    axes[1].text(
        0.02,
        0.03,
        (
            f"Baseline {estimate.height_baseline_m:.1f} m  →  "
            f"optimized {estimate.height_optimized_m:.1f} m\n"
            f"geometry prior/reference {estimate.height_reference_m:.1f} m"
        ),
        transform=axes[1].transAxes,
        color="white",
        fontsize=6.8,
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.72},
        zorder=8,
    )
    axes[1].text(
        -0.10,
        1.02,
        "b",
        transform=axes[1].transAxes,
        fontweight="bold",
        fontsize=8,
    )
    handles = [
        Patch(facecolor="none", edgecolor=COLORS[name], label=name.capitalize())
        for name in ("roof", "wall", "bottom")
    ] + [
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor=COLORS["roof"],
            markeredgecolor=COLORS["dark"],
            label="Roof PS",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["wall"],
            markeredgecolor=COLORS["dark"],
            label="Wall PS",
        ),
    ]
    axes[0].legend(
        handles=handles,
        loc="lower right",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        fontsize=6.2,
    )
    fig.suptitle(
        "Fusion of optimized building triangles and PS observations",
        fontsize=9,
        y=0.995,
    )
    fig.tight_layout()
    outputs = save_figure(fig, output_root, "10_triangle_projection_ps_fusion")
    return outputs, representative_fid


def metrics(
    result: pd.DataFrame,
    fold: np.ndarray,
) -> dict:
    table = result.loc[
        result.height_baseline_m.notna() & (result.height_reference_m >= 30.0)
    ].copy()
    table["spatial_fold"] = fold[table.fid.to_numpy(dtype=int)]
    output: dict[str, dict] = {}
    for name, subset in (
        ("all_highrise", table),
        ("training_fold_0", table.loc[table.spatial_fold == 0]),
        ("holdout_fold_1", table.loc[table.spatial_fold == 1]),
    ):
        item = {"n": int(len(subset))}
        for method, column in (
            ("baseline", "height_baseline_m"),
            ("optimized", "height_optimized_m"),
        ):
            error = subset[column] - subset.height_reference_m
            item[method] = {
                "bias_m": float(error.mean()),
                "mae_m": float(error.abs().mean()),
                "rmse_m": float(np.sqrt(np.mean(error**2))),
                "median_ratio_to_reference": float(
                    np.median(subset[column] / subset.height_reference_m)
                ),
            }
        output[name] = item
    return output


def run(source_root: Path, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = source_root / "touying2_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    buildings = gpd.read_file(resolve(config["inputs"]["buildings"])).reset_index(
        drop=True
    )
    buildings["fid"] = np.arange(len(buildings), dtype=np.int64)
    reference = pd.read_csv(source_root / "tables" / "initial_absolute_roof_heights.csv")
    reference = reference.rename(columns={"height_current_m": "height_reference_m"})
    baseline_estimates = pd.read_csv(
        source_root / "adjustment" / "building_heights_triangle_adjustment.csv"
    )
    baseline_equations = pd.read_csv(
        source_root / "adjustment" / "ps_triangle_height_equations.csv"
    )
    parameters, grid = select_parameters(
        baseline_estimates,
        baseline_equations,
        reference,
        buildings,
    )
    grid_path = output_root / "tables" / "parameter_grid_spatial_cv.csv"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(grid_path, index=False)

    stage1, _ = apply_enhancement(
        baseline_estimates,
        baseline_equations,
        reference,
        parameters,
    )
    stage1_projection = projection_heights(
        reference,
        stage1,
        output_root / "tables" / "projection_heights_stage1.csv",
    )
    registration = source_root / "tables" / "roof_only_local_registration.csv"
    stage1_mapping = output_root / "stage1_mapping"
    run_mapping(
        config_path,
        0,
        stage1_projection,
        registration,
        stage1_mapping,
    )
    stage1_adjustment = output_root / "stage1_adjustment"
    run_adjustment(
        config_path,
        stage1_mapping / "tables" / "ps_building_surface_coordinates.csv",
        stage1_adjustment,
    )

    refined_estimates = pd.read_csv(
        stage1_adjustment / "building_heights_triangle_adjustment.csv"
    )
    refined_equations = pd.read_csv(
        stage1_adjustment / "ps_triangle_height_equations.csv"
    )
    refined_result, observations = apply_enhancement(
        refined_estimates,
        refined_equations,
        reference,
        parameters,
    )
    result = reconcile_with_baseline(stage1, refined_result)
    final_table = output_root / "tables" / "building_height_estimates_highrise_optimized.csv"
    result.to_csv(final_table, index=False)
    observations.to_csv(
        output_root / "tables" / "ps_height_envelope_observations.csv",
        index=False,
    )

    vector = buildings.merge(
        result.drop(columns=["clean_id"], errors="ignore"),
        on="fid",
        how="left",
        validate="one_to_one",
    )
    vector_path = output_root / "vectors" / "building_height_estimates_highrise_optimized.gpkg"
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    vector.to_file(vector_path, layer="highrise_optimized_heights", driver="GPKG")

    final_projection = projection_heights(
        reference,
        result,
        output_root / "tables" / "projection_heights_final.csv",
    )
    final_mapping = output_root / "final_mapping"
    run_mapping(
        config_path,
        0,
        final_projection,
        registration,
        final_mapping,
    )
    triangles = gpd.read_file(
        final_mapping / "triangles" / "building_surface_triangles_radar.gpkg"
    )
    mapped = pd.read_csv(
        final_mapping / "tables" / "ps_building_surface_coordinates.csv"
    )

    validation_outputs = make_validation_figure(result, output_root)
    fusion_outputs, representative_fid = make_fusion_figure(
        config,
        triangles,
        mapped,
        result,
        output_root,
    )
    fold = spatial_fold(buildings)
    evaluation = metrics(result, fold)
    accepted = result.loc[result.height_optimized_m.notna()]
    enhanced = result.loc[result.highrise_optimization_applied == 1]
    summary = {
        "method": (
            "two_stage_reprojected_asymmetric_ps_top_restoration_with_"
            "height_dependent_geometry_floor_and_spatial_diagnostic"
        ),
        "height_semantics": (
            "Baseline is the robust center solution. Optimized high-rise height "
            "uses an asymmetric Huber-ridge calibration of the PS q95 tail and "
            "center estimate, plus a Shapefile-prior floor rising from 78% at "
            "30 m to 90% at 70 m. Calibration coefficients are trained on "
            "checkerboard fold 0; fold 1 is reported as a spatial diagnostic."
        ),
        "parameters": parameters,
        "reference_warning": (
            "height_reference_m is the same Shapefile-derived geometry prior used "
            "to build the projection mesh. Metrics against it diagnose internal "
            "consistency and are not independent external accuracy validation."
        ),
        "accepted_buildings": int(len(accepted)),
        "primary_buildings": int(
            accepted.final_quality.astype(str).str.startswith(
                ("high", "medium")
            ).sum()
        ),
        "supplementary_buildings": int(
            accepted.final_quality.astype(str).str.startswith(
                "supplementary"
            ).sum()
        ),
        "highrise_optimized_buildings": int(len(enhanced)),
        "optimized_height_m": {
            "minimum": float(accepted.height_optimized_m.min()),
            "median": float(accepted.height_optimized_m.median()),
            "mean": float(accepted.height_optimized_m.mean()),
            "maximum": float(accepted.height_optimized_m.max()),
        },
        "evaluation": evaluation,
        "representative_fid": representative_fid,
        "final_mapping": {
            "mapped_ps": int(len(mapped)),
            "surface_counts": {
                str(key): int(value)
                for key, value in mapped.surface.value_counts().items()
            },
        },
        "outputs": {
            "final_table": str(final_table),
            "final_vector": str(vector_path),
            "parameter_grid": str(grid_path),
            "validation_figure": validation_outputs,
            "fusion_figure": fusion_outputs,
            "final_triangles": str(
                final_mapping
                / "triangles"
                / "building_surface_triangles_radar.gpkg"
            ),
            "final_mapped_ps": str(
                final_mapping
                / "vectors"
                / "ps_points_on_building_surfaces.gpkg"
            ),
        },
        "export_formats": ["png", "svg"],
        "pdf_generated": False,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Restore high-rise top heights using asymmetric PS-tail calibration "
            "and a height-dependent geometry floor, then regenerate triangle/PS "
            "fusion figures."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.source_root.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
