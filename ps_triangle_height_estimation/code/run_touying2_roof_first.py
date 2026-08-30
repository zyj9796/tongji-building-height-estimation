from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-touying2")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
from shapely.geometry import Polygon

from estimate_heights_by_triangle_adjustment import run as run_adjustment
from geometry import StrictRadarProjector
from map_ps_to_building_surfaces import (
    rasterize_projected_triangles,
    refine_projected_mask,
    resolve,
    run as run_mapping,
)
from recompute_hybrid_rooftop_registration import merge_reference_registration
from recompute_iterative_local_registration import read_sar_features


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"
DEFAULT_OUTPUT = ROOT / "results" / "picall" / "touying2"

COLORS = {
    "roof": "#24C4D8",
    "wall": "#F2A541",
    "bottom": "#8A8F98",
    "ps_roof": "#00E5FF",
    "ps_wall": "#FFB547",
    "dark": "#2F343B",
    "gray": "#D9DDE2",
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
        "pdf.fonttype": 42,
    }
)


def save_figure(fig: plt.Figure, output_root: Path, stem: str) -> None:
    png_dir = output_root / "png"
    svg_dir = output_root / "svg"
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_dir / f"{stem}.svg", dpi=96, bbox_inches="tight", facecolor="white")
    fig.savefig(png_dir / f"{stem}.png", dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def normalize_ps_input(
    ps_points: Path,
    output_root: Path,
    reference_dsm: Path | None = None,
) -> tuple[Path, dict]:
    """Validate an external PS table and construct the height_m observation."""
    source = ps_points.resolve()
    table = pd.read_csv(source)
    package_aliases = {
        "longitude_epsg4326_deg": "longitude",
        "latitude_epsg4326_deg": "latitude",
        "x_utm51n_epsg32651_m": "x_utm51n_m",
        "y_utm51n_epsg32651_m": "y_utm51n_m",
        "sar_azimuth_pixel_1based": "azimuth_pixel",
        "sar_range_pixel_1based": "range_pixel",
    }
    applied_aliases = {}
    for source_column, normalized_column in package_aliases.items():
        if normalized_column not in table and source_column in table:
            table[normalized_column] = table[source_column]
            applied_aliases[source_column] = normalized_column
    required = {
        "ps_id",
        "azimuth_pixel",
        "range_pixel",
        "coherence",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(
            f"PS input {source} is missing required columns: "
            + ", ".join(missing)
        )
    if table.ps_id.duplicated().any():
        raise ValueError(f"PS input {source} contains duplicate ps_id values")
    if table[["azimuth_pixel", "range_pixel"]].duplicated().any():
        raise ValueError(f"PS input {source} contains duplicate SAR pixels")

    if reference_dsm is not None:
        required_dsm_columns = {"x_utm51n_m", "y_utm51n_m", "z_dsm_m"}
        missing_dsm_columns = sorted(required_dsm_columns.difference(table.columns))
        if missing_dsm_columns:
            raise ValueError(
                "DSM-referenced PS normalization requires columns: "
                + ", ".join(missing_dsm_columns)
            )
        dsm_path = reference_dsm.resolve()
        if not dsm_path.is_file():
            raise FileNotFoundError(f"Reference DSM does not exist: {dsm_path}")
        x = pd.to_numeric(table["x_utm51n_m"], errors="coerce")
        y = pd.to_numeric(table["y_utm51n_m"], errors="coerce")
        relative = pd.to_numeric(table["z_dsm_m"], errors="coerce")
        if x.isna().any() or y.isna().any() or relative.isna().any():
            raise ValueError(
                "x_utm51n_m, y_utm51n_m, and z_dsm_m must be finite numeric values"
            )
        with rasterio.open(dsm_path) as dataset:
            if dataset.crs is None:
                raise ValueError(f"Reference DSM has no CRS: {dsm_path}")
            if dataset.crs.to_epsg() != 32651:
                raise ValueError(
                    "PS x_utm51n_m/y_utm51n_m require an EPSG:32651 DSM; "
                    f"got {dataset.crs}"
                )
            dsm_prior = np.asarray(
                [
                    sample[0]
                    for sample in dataset.sample(
                        zip(x.to_numpy(), y.to_numpy(), strict=True)
                    )
                ],
                dtype=np.float64,
            )
            nodata = dataset.nodata
        invalid_dsm = ~np.isfinite(dsm_prior)
        if nodata is not None:
            invalid_dsm |= np.isclose(dsm_prior, float(nodata))
        if invalid_dsm.any():
            raise ValueError(
                f"Reference DSM sampling failed for {int(invalid_dsm.sum())} PS points"
            )
        table["ps_relative_to_dsm_m"] = relative
        table["dsm_planar_prior_m"] = dsm_prior
        table["height_m"] = relative.to_numpy(dtype=np.float64) + dsm_prior
        elevation_source = "z_dsm_m + sampled_reference_dsm"
        height_formula = (
            "height_m = z_dsm_m + DSM(x_utm51n_m, y_utm51n_m)"
        )
        interpretation_warning = (
            "Height combines the supplied PS offset and a newly sampled DSM prior."
        )
    elif "psi_scatterer_elevation_wusong_m" in table:
        corrected = pd.to_numeric(
            table["psi_scatterer_elevation_wusong_m"], errors="coerce"
        )
        if corrected.isna().any():
            raise ValueError(
                "psi_scatterer_elevation_wusong_m must contain finite numeric values"
            )
        if "psi_height_above_4m_ground_m" in table:
            above_ground = pd.to_numeric(
                table["psi_height_above_4m_ground_m"], errors="coerce"
            )
            if above_ground.isna().any():
                raise ValueError(
                    "psi_height_above_4m_ground_m must contain finite numeric values"
                )
            if not np.allclose(
                corrected,
                above_ground + 4.0,
                rtol=0.0,
                atol=1e-7,
            ):
                raise ValueError(
                    "psi_scatterer_elevation_wusong_m does not equal "
                    "psi_height_above_4m_ground_m + 4.000 m"
                )
        table["height_m"] = corrected
        table["z_dsm_m"] = corrected
        elevation_source = "psi_scatterer_elevation_wusong_m"
        height_formula = (
            "height_m = psi_scatterer_elevation_wusong_m = "
            "psi_height_above_4m_ground_m + 4.000 m"
            if "psi_height_above_4m_ground_m" in table
            else "height_m = psi_scatterer_elevation_wusong_m"
        )
        interpretation_warning = (
            "Height uses the internally ground-referenced PSI scatterer elevation "
            "in the project Wusong datum. It has no independent LiDAR, GNSS, or "
            "leveling validation."
        )
    elif "psi_corrected_elevation_m" in table:
        if "stamps_input_dsm_hgt_m" not in table:
            raise ValueError(
                "psi_corrected_elevation_m requires stamps_input_dsm_hgt_m "
                "for provenance checking"
            )
        input_dsm = pd.to_numeric(
            table["stamps_input_dsm_hgt_m"], errors="coerce"
        )
        residual = pd.to_numeric(
            table["psi_dem_height_residual_m"], errors="coerce"
        )
        corrected = pd.to_numeric(
            table["psi_corrected_elevation_m"], errors="coerce"
        )
        if input_dsm.isna().any() or residual.isna().any() or corrected.isna().any():
            raise ValueError(
                "PSI input DSM height, raw residual, and corrected elevation "
                "must be finite numeric values"
            )
        if not np.allclose(
            corrected,
            input_dsm + residual,
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(
                "psi_corrected_elevation_m does not equal "
                "stamps_input_dsm_hgt_m + psi_dem_height_residual_m"
            )
        table["height_m"] = corrected
        table["z_dsm_m"] = input_dsm
        elevation_source = "psi_corrected_elevation_m"
        height_formula = (
            "height_m = stamps_input_dsm_hgt_m + "
            "psi_dem_height_residual_m"
        )
        interpretation_warning = (
            "The SCLA-equivalent residual is scene-mean referenced and may also "
            "contain orbit, baseline, unwrapping, or atmospheric errors; its "
            "vertical offset and sign require external control."
        )
    elif "height_m" in table:
        elevation_source = "height_m"
        height_formula = "height_m = input height_m"
        interpretation_warning = "Height uses the input height_m field."
    elif "z_dsm_m" in table:
        elevation_source = "z_dsm_m"
        table["height_m"] = pd.to_numeric(table["z_dsm_m"], errors="coerce")
        height_formula = "height_m = z_dsm_m"
        interpretation_warning = "Height directly uses z_dsm_m."
    elif "terrain_height_m" in table:
        elevation_source = "terrain_height_m"
        table["height_m"] = pd.to_numeric(
            table["terrain_height_m"], errors="coerce"
        )
        height_formula = "height_m = terrain_height_m"
        interpretation_warning = "Height directly uses terrain_height_m."
    else:
        raise ValueError(
            f"PS input {source} has no height_m, z_dsm_m, or terrain_height_m"
        )
    if "z_dsm_m" not in table:
        table["z_dsm_m"] = table["height_m"]

    numeric = [
        "azimuth_pixel",
        "range_pixel",
        "coherence",
        "height_m",
        "z_dsm_m",
    ]
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if table[numeric].isna().any().any():
        bad = table[numeric].isna().sum()
        raise ValueError(
            "PS input contains non-numeric or missing required values: "
            + ", ".join(f"{key}={value}" for key, value in bad.items() if value)
        )

    normalized = output_root / "tables" / "ps_points_input_normalized.csv"
    normalized.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(normalized, index=False)
    metadata = {
        "source": str(source),
        "normalized": str(normalized),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": int(len(table)),
        "height_m_normalized_from": elevation_source,
        "height_formula": height_formula,
        "interpretation_warning": interpretation_warning,
        "applied_column_aliases": applied_aliases,
        "reference_dsm": str(reference_dsm.resolve())
        if reference_dsm is not None
        else None,
        "height_equals_z_dsm": bool(
            np.allclose(
                table.height_m,
                table.z_dsm_m,
                rtol=0.0,
                atol=1e-9,
            )
        ),
        "minimum_coherence": float(table.coherence.min()),
        "maximum_coherence": float(table.coherence.max()),
        "height_m_statistics": {
            "minimum": float(table.height_m.min()),
            "median": float(table.height_m.median()),
            "mean": float(table.height_m.mean()),
            "maximum": float(table.height_m.max()),
        },
        "sar_pixel_bounds_one_based": {
            "azimuth": [
                int(table.azimuth_pixel.min()),
                int(table.azimuth_pixel.max()),
            ],
            "range": [
                int(table.range_pixel.min()),
                int(table.range_pixel.max()),
            ],
        },
    }
    return normalized, metadata


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


def polygon_lines(geometries) -> list[np.ndarray]:
    return [
        np.asarray(geometry.exterior.coords)
        for geometry in geometries
        if geometry is not None and not geometry.is_empty and geometry.geom_type == "Polygon"
    ]


def prepare_inputs(config: dict, output_root: Path) -> tuple[Path, Path, pd.DataFrame]:
    buildings = gpd.read_file(resolve(config["inputs"]["buildings"])).reset_index(drop=True)
    buildings["fid"] = np.arange(len(buildings), dtype=np.int64)
    base = float(config["base_elevation_m"])
    roof_absolute = pd.to_numeric(buildings[config["height_field"]], errors="coerce")
    above_ground = roof_absolute - base
    heights = pd.DataFrame(
        {
            "fid": buildings.fid,
            "clean_id": pd.to_numeric(buildings.clean_id, errors="coerce"),
            "roof_absolute_height_m": roof_absolute,
            "height_current_m": above_ground.where(above_ground > 0.0),
            "projection_valid": (above_ground > 0.0).astype(int),
        }
    )
    heights_path = output_root / "tables" / "initial_absolute_roof_heights.csv"
    heights_path.parent.mkdir(parents=True, exist_ok=True)
    heights.to_csv(heights_path, index=False)

    reference_path = resolve(
        config["iterative_adjustment"]["registration_reference_v4"]["registration_table"]
    )
    reference = pd.read_csv(reference_path)
    aligned = reference.set_index("fid").reindex(buildings.fid).reset_index()
    if config.get("figure2_registration_mode") == "all_reference_accepted":
        accepted = aligned.registration_accepted.fillna(0).eq(1)
        registration = aligned.copy()
        registration["applied_row_shift"] = aligned.dy.where(accepted, 0.0)
        registration["applied_col_shift"] = aligned.dx.where(accepted, 0.0)
        registration["accepted"] = accepted.astype(int)
        registration["local_refinement_accepted"] = accepted.astype(int)
        registration["reference_multiscene_override"] = accepted.astype(int)
    else:
        base_registration = pd.DataFrame(
            {
                "fid": buildings.fid,
                "applied_row_shift": 0.0,
                "applied_col_shift": 0.0,
                "accepted": 0,
                "local_refinement_accepted": 0,
                "registration_feature_mode": "global_shift_only",
            }
        )
        registration = merge_reference_registration(
            base_registration,
            reference,
            config,
        )
        accepted = registration.reference_multiscene_override.eq(1)
    margin_score = np.clip(
        (
            pd.to_numeric(aligned.score_margin, errors="coerce").fillna(0.0)
            - float(
                config["iterative_adjustment"]["registration_reference_v4"][
                    "minimum_score_margin"
                ]
            )
        )
        / 0.5,
        0.0,
        1.0,
    )
    edge_score = np.clip(
        pd.to_numeric(aligned.gain_oriented_edge, errors="coerce").fillna(0.0)
        / 0.30,
        0.0,
        1.0,
    )
    continuity_score = np.clip(
        pd.to_numeric(aligned.gain_continuity, errors="coerce").fillna(0.0)
        / 0.45,
        0.0,
        1.0,
    )
    scene_score = np.exp(
        -pd.to_numeric(aligned.pair_distance_px, errors="coerce").fillna(9.0)
        / 3.0
        -pd.to_numeric(aligned.fused_to_pair_px, errors="coerce").fillna(9.0)
        / 2.0
    )
    registration["registration_reliability"] = 0.35
    registration.loc[accepted, "registration_reliability"] = np.clip(
        0.55
        + 0.15 * margin_score[accepted]
        + 0.10 * edge_score[accepted]
        + 0.10 * continuity_score[accepted]
        + 0.10 * scene_score[accepted],
        0.55,
        1.0,
    )
    registration["apply_local_shift_to_roof_only"] = 1
    registration["registration_feature_mode"] = (
        "figure2_all_accepted_height_as_absolute_roof_multiscene_registration"
        if config.get("figure2_registration_mode") == "all_reference_accepted"
        else "quality_gated_height_as_absolute_roof_multiscene_registration_touying2"
    )
    registration_path = output_root / "tables" / "roof_only_local_registration.csv"
    registration.to_csv(registration_path, index=False)
    return heights_path, registration_path, buildings


def initial_roofs(config: dict, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    geographic = buildings.to_crs("EPSG:4326")
    projector = StrictRadarProjector(resolve(config["inputs"]["rslc_par"]))
    global_row = float(config["registration"]["global_row_shift_px"])
    global_col = float(config["registration"]["global_col_shift_px"])
    rows = []
    for building in geographic.itertuples():
        roof_absolute = float(getattr(building, config["height_field"]))
        ring = np.asarray(building.geometry.exterior.coords)
        ecef_ring = projector.build_mesh(
            ring,
            roof_absolute,
            0.0,
            global_row,
            global_col,
        )
        vertex_count = len(ecef_ring.projected_xy) // 2
        geometry = Polygon(ecef_ring.projected_xy[vertex_count:]).buffer(0)
        rows.append(
            {
                "fid": int(building.fid),
                "clean_id": int(building.clean_id),
                "roof_absolute_height_m": roof_absolute,
                "geometry": geometry,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=None)


def make_figures(
    config: dict,
    output_root: Path,
    buildings: gpd.GeoDataFrame,
    roofs_before: gpd.GeoDataFrame,
    triangles: gpd.GeoDataFrame,
    mapped: pd.DataFrame,
    equations: pd.DataFrame,
    final: pd.DataFrame,
) -> int:
    projector = StrictRadarProjector(resolve(config["inputs"]["rslc_par"]))
    rslc_path = resolve(config["inputs"]["rslc_par"]).with_suffix("")
    amplitude = read_sar_features(
        rslc_path,
        (int(projector.par["azimuth_lines"]), int(projector.par["range_samples"])),
    )[0]

    roof_after = triangles.loc[triangles.surface == "roof"]
    roof_after_outlines = roof_after.dissolve(by="building_fid")
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    add_sar(ax, amplitude)
    ax.add_collection(
        LineCollection(
            polygon_lines(roofs_before.geometry),
            colors=COLORS["roof"],
            linewidths=0.34,
            alpha=0.76,
        )
    )
    ax.set_title("Roof projection using Shapefile height as absolute elevation")
    ax.text(
        0.01,
        0.99,
        f"Absolute roof elevation = height; projected roofs: {len(roofs_before)}",
        transform=ax.transAxes,
        va="top",
        color="white",
        bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
    )
    save_figure(fig, output_root, "01_absolute_height_roof_projection")

    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    add_sar(ax, amplitude)
    ax.add_collection(
        LineCollection(
            polygon_lines(roof_after_outlines.geometry),
            colors=COLORS["roof"],
            linewidths=0.38,
            alpha=0.82,
        )
    )
    figure2_basis = (
        config.get("figure2_registration_mode") == "all_reference_accepted"
    )
    ax.set_title(
        "Roofs after three-scene subpixel feature registration"
        if figure2_basis
        else "Roofs after quality-gated three-scene feature registration"
    )
    ax.text(
        0.01,
        0.99,
        (
            "Only roof vertices receive the building-level image registration"
            if figure2_basis
            else "Only roof vertices receive local shifts passing all multiscene gates"
        ),
        transform=ax.transAxes,
        va="top",
        color="white",
        bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
    )
    save_figure(fig, output_root, "02_registered_roof_projection")

    support = (
        mapped.groupby("fid")
        .agg(
            roof_ps=("surface", lambda values: int(np.sum(values == "roof"))),
            wall_ps=("surface", lambda values: int(np.sum(values == "wall"))),
        )
        .assign(total=lambda frame: frame.roof_ps + frame.wall_ps)
    )
    triangle_bounds = (
        triangles.groupby("building_fid")
        .geometry.apply(lambda values: values.total_bounds)
        .apply(pd.Series)
    )
    triangle_bounds.columns = ["minx", "miny", "maxx", "maxy"]
    triangle_bounds["width"] = triangle_bounds.maxx - triangle_bounds.minx
    triangle_bounds["height"] = triangle_bounds.maxy - triangle_bounds.miny
    candidates = support.join(triangle_bounds).join(
        final.set_index("fid")[["height_final_m"]]
    )
    candidates["balanced_support"] = candidates[["roof_ps", "wall_ps"]].min(axis=1)
    candidates["aspect"] = np.maximum(
        candidates.width / np.maximum(candidates.height, 1e-6),
        candidates.height / np.maximum(candidates.width, 1e-6),
    )
    candidates = candidates.loc[
        (candidates.roof_ps >= 3)
        & (candidates.wall_ps >= 3)
        & (candidates.total <= 150)
        & (candidates.minx >= 5)
        & (candidates.miny >= 5)
        & (candidates.maxx <= amplitude.shape[1] - 5)
        & (candidates.maxy <= amplitude.shape[0] - 5)
        & (candidates.width <= 80)
        & (candidates.height <= 60)
        & (candidates.aspect <= 2.0)
        & np.isfinite(candidates.height_final_m)
    ]
    representative_fid = int(
        candidates.sort_values(
            ["balanced_support", "total"], ascending=False
        ).index[0]
        if len(candidates)
        else support.sort_values("total", ascending=False).index[0]
    )
    representative_triangles = triangles.loc[
        triangles.building_fid == representative_fid
    ]
    bounds = representative_triangles.total_bounds
    pad = 9.0

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    add_sar(ax, amplitude)
    for surface in ("wall", "roof", "bottom"):
        part = representative_triangles.loc[representative_triangles.surface == surface]
        ax.add_collection(
            LineCollection(
                polygon_lines(part.geometry),
                colors=COLORS[surface],
                linewidths=1.0 if surface == "roof" else 0.72,
                alpha=0.95,
                label=surface.capitalize(),
            )
        )
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[3] + pad, bounds[1] - pad)
    handles = [
        Patch(facecolor="none", edgecolor=COLORS[name], label=name.capitalize())
        for name in ("roof", "wall", "bottom")
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.82,
    )
    ax.set_title(
        f"Roof-anchored triangular projection — building {representative_fid}"
    )
    save_figure(fig, output_root, "03_roof_anchored_side_triangle_projection")

    representative_ps = mapped.loc[mapped.fid == representative_fid]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    add_sar(ax, amplitude)
    for surface in ("wall", "roof"):
        part = representative_triangles.loc[representative_triangles.surface == surface]
        ax.add_collection(
            LineCollection(
                polygon_lines(part.geometry),
                colors=COLORS[surface],
                linewidths=0.8,
                alpha=0.9,
            )
        )
        points = representative_ps.loc[representative_ps.surface == surface]
        ax.scatter(
            points.col0,
            points.row0,
            s=18,
            color=COLORS[f"ps_{surface}"],
            edgecolor=COLORS["dark"],
            linewidth=0.35,
            label=f"{surface.capitalize()} PS (n={len(points)})",
            zorder=5,
        )
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[3] + pad, bounds[1] - pad)
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.82,
    )
    estimate = final.loc[final.fid == representative_fid]
    label = (
        f"Estimated height = {float(estimate.height_final_m.iloc[0]):.1f} m"
        if len(estimate) and np.isfinite(estimate.height_final_m.iloc[0])
        else "Height estimate rejected by quality control"
    )
    ax.set_title(f"PS roof/wall assignment — building {representative_fid}")
    ax.text(
        0.02,
        0.03,
        label,
        transform=ax.transAxes,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.68, "edgecolor": "none"},
    )
    save_figure(fig, output_root, "04_ps_roof_wall_assignment")

    refinement = config.get("mask_refinement", {})
    if refinement.get("enabled", False):
        surface_triangles = {
            str(surface): [
                np.asarray(geometry.exterior.coords)[:3]
                for geometry in part.geometry
                if geometry is not None and not geometry.is_empty
            ]
            for surface, part in representative_triangles.groupby("surface")
        }
        refined_mask, mask_stats = refine_projected_mask(
            amplitude,
            surface_triangles,
            refinement,
        )
        initial_mask = rasterize_projected_triangles(
            [
                triangle
                for surface in refinement.get("surfaces", ["roof", "wall"])
                for triangle in surface_triangles.get(surface, [])
            ],
            amplitude.shape,
        )
        threshold_mask = initial_mask & (
            amplitude > float(mask_stats["threshold"])
        )
        audit = pd.read_csv(
            output_root
            / "mapping"
            / "tables"
            / "ps_mask_refinement_audit.csv",
            dtype={
                "candidate_buildings_before": str,
                "candidate_buildings_after": str,
            },
        )
        token = str(representative_fid)
        before = audit.loc[
            audit.candidate_buildings_before.fillna("").str.split(";").apply(
                lambda values: token in values
            )
        ]
        fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.35), sharex=True, sharey=True)
        for ax in axes:
            add_sar(ax, amplitude)
            ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
            ax.set_ylim(bounds[3] + pad, bounds[1] - pad)
        axes[0].contour(
            initial_mask,
            levels=[0.5],
            colors=[COLORS["roof"]],
            linewidths=1.0,
        )
        axes[0].set_title(f"(a) Initial projected mask $M_0$\n{mask_stats['initial_pixels']} pixels")
        axes[1].imshow(
            np.ma.masked_where(~threshold_mask, threshold_mask),
            cmap=mpl.colors.ListedColormap(["#FFD166"]),
            alpha=0.68,
            origin="upper",
            interpolation="nearest",
        )
        axes[1].set_title(
            "(b) Local amplitude selection\n"
            f"$A>\\tau$, $\\tau$={mask_stats['threshold']:.3f}"
        )
        axes[2].imshow(
            np.ma.masked_where(~refined_mask, refined_mask),
            cmap=mpl.colors.ListedColormap(["#32D6C5"]),
            alpha=0.70,
            origin="upper",
            interpolation="nearest",
        )
        axes[2].scatter(
            before.col0,
            before.row0,
            s=7,
            color="#D9DDE2",
            edgecolor="none",
            label=f"Before (n={len(before)})",
            zorder=4,
        )
        for surface in ("wall", "roof"):
            points = representative_ps.loc[representative_ps.surface == surface]
            axes[2].scatter(
                points.col0,
                points.row0,
                s=17,
                color=COLORS[f"ps_{surface}"],
                edgecolor=COLORS["dark"],
                linewidth=0.3,
                label=f"{surface.capitalize()} kept (n={len(points)})",
                zorder=5,
            )
        axes[2].set_title(
            "(c) Geometry-constrained refined mask\n"
            f"{mask_stats['refined_pixels']} pixels retained"
        )
        axes[2].legend(
            loc="upper right",
            frameon=True,
            facecolor="white",
            framealpha=0.84,
            fontsize=6,
        )
        for ax in axes[1:]:
            ax.set_ylabel("")
        fig.suptitle(
            f"Local SAR-amplitude mask refinement — building {representative_fid}",
            y=0.995,
        )
        fig.tight_layout()
        save_figure(fig, output_root, "10_local_amplitude_mask_refinement")

        refinement_surfaces = tuple(
            refinement.get("surfaces", ["roof", "wall"])
        )
        initial_union = rasterize_projected_triangles(
            [
                np.asarray(geometry.exterior.coords)[:3]
                for surface in refinement_surfaces
                for geometry in triangles.loc[
                    triangles.surface == surface, "geometry"
                ]
                if geometry is not None and not geometry.is_empty
            ],
            amplitude.shape,
        )
        refined_union = np.zeros_like(initial_union)
        for _, building_triangles in triangles.groupby(
            "building_fid", sort=True
        ):
            building_surfaces = {
                str(surface): [
                    np.asarray(geometry.exterior.coords)[:3]
                    for geometry in part.geometry
                    if geometry is not None and not geometry.is_empty
                ]
                for surface, part in building_triangles.groupby("surface")
            }
            building_refined, _ = refine_projected_mask(
                amplitude,
                building_surfaces,
                refinement,
            )
            refined_union |= building_refined
        removed_union = initial_union & ~refined_union
        mask_table = pd.read_csv(
            output_root
            / "mapping"
            / "tables"
            / "mask_refinement_summary.csv"
        )
        initial_pixels = int(mask_table.initial_pixels.sum())
        refined_pixels = int(mask_table.refined_pixels.sum())
        candidate_before = int(len(audit))
        candidate_after = int(len(mapped))

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(11.8, 4.55),
            sharex=True,
            sharey=True,
        )
        for ax in axes:
            add_sar(ax, amplitude)
        axes[0].imshow(
            np.ma.masked_where(~initial_union, initial_union),
            cmap=mpl.colors.ListedColormap([COLORS["roof"]]),
            alpha=0.52,
            origin="upper",
            interpolation="nearest",
        )
        axes[0].set_title(
            "(a) Full-area initial projected mask $M_0$"
        )
        axes[1].imshow(
            np.ma.masked_where(~removed_union, removed_union),
            cmap=mpl.colors.ListedColormap(["#D7A13B"]),
            alpha=0.48,
            origin="upper",
            interpolation="nearest",
        )
        axes[1].imshow(
            np.ma.masked_where(~refined_union, refined_union),
            cmap=mpl.colors.ListedColormap(["#20B8AA"]),
            alpha=0.68,
            origin="upper",
            interpolation="nearest",
        )
        locator_pad = 4.0
        axes[1].add_patch(
            Rectangle(
                (
                    bounds[0] - locator_pad,
                    bounds[1] - locator_pad,
                ),
                bounds[2] - bounds[0] + 2 * locator_pad,
                bounds[3] - bounds[1] + 2 * locator_pad,
                fill=False,
                edgecolor="#D64B4B",
                linewidth=1.15,
                zorder=7,
            )
        )
        axes[1].text(
            bounds[2] + locator_pad + 2,
            bounds[1] - locator_pad,
            f"FID {representative_fid}\nlocal detail: Fig. 10",
            color="#D64B4B",
            fontsize=6.2,
            va="bottom",
            ha="left",
            bbox={
                "facecolor": "white",
                "alpha": 0.82,
                "edgecolor": "none",
                "pad": 1.5,
            },
            zorder=8,
        )
        axes[1].set_title(
            "(b) Full-area geometry-constrained refined mask $M_f$"
        )
        legend_handles = [
            Patch(
                facecolor="#20B8AA",
                edgecolor="none",
                label="Retained by local amplitude + connectivity",
            ),
            Patch(
                facecolor="#D7A13B",
                edgecolor="none",
                label="Removed weak/disconnected model pixels",
            ),
            Patch(
                facecolor="none",
                edgecolor="#D64B4B",
                label="Local enlargement locator",
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.015),
            ncol=3,
            frameon=False,
            fontsize=6.2,
        )
        axes[0].text(
            0.015,
            0.018,
            (
                f"Projected buildings: {len(mask_table):,}\n"
                f"Per-building mask pixels: {initial_pixels:,}"
            ),
            transform=axes[0].transAxes,
            color="white",
            fontsize=6.4,
            va="bottom",
            bbox={
                "facecolor": "black",
                "alpha": 0.66,
                "edgecolor": "none",
            },
        )
        axes[1].text(
            0.015,
            0.018,
            (
                f"Per-building refined pixels: {refined_pixels:,} "
                f"({refined_pixels / initial_pixels:.1%})\n"
                f"PS candidates retained: {candidate_after:,}/"
                f"{candidate_before:,} "
                f"({candidate_after / candidate_before:.1%})"
            ),
            transform=axes[1].transAxes,
            color="white",
            fontsize=6.4,
            va="bottom",
            bbox={
                "facecolor": "black",
                "alpha": 0.66,
                "edgecolor": "none",
            },
        )
        axes[1].set_ylabel("")
        fig.suptitle(
            "Full-area SAR building-mask refinement",
            y=0.995,
        )
        fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.98))
        save_figure(fig, output_root, "11_full_area_mask_refinement_overview")

    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    add_sar(ax, amplitude)
    for surface, linewidth, alpha in (
        ("bottom", 0.18, 0.32),
        ("wall", 0.22, 0.56),
        ("roof", 0.30, 0.82),
    ):
        part = triangles.loc[triangles.surface == surface]
        collection = LineCollection(
            polygon_lines(part.geometry),
            colors=COLORS[surface],
            linewidths=linewidth,
            alpha=alpha,
            rasterized=True,
        )
        ax.add_collection(collection)
    handles = [
        Patch(facecolor="none", edgecolor=COLORS[name], label=name.capitalize())
        for name in ("roof", "wall", "bottom")
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.88,
    )
    ax.set_title("Full-area roof-anchored triangular projection")
    ax.text(
        0.01,
        0.99,
        (
            f"{triangles.building_fid.nunique()} projected buildings; "
            f"{len(triangles)} roof/wall/bottom triangles"
        ),
        transform=ax.transAxes,
        va="top",
        color="white",
        bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
    )
    save_figure(fig, output_root, "06_full_area_triangle_projection")

    quality_order = ["invalid", "rejected", "D", "C", "B", "A"]
    quality_colors = {
        "invalid": "#B7BCC2",
        "rejected": "#777D84",
        "D": "#C65D4B",
        "C": "#E2A33A",
        "B": "#4FA7A0",
        "A": "#00D5E8",
    }
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    add_sar(ax, amplitude)
    quality_values = equations.ps_quality_class.fillna("invalid")
    for quality in quality_order:
        points = equations.loc[quality_values == quality]
        if points.empty:
            continue
        ax.scatter(
            points.col0,
            points.row0,
            s=2.0 if quality in {"A", "B"} else 1.2,
            color=quality_colors[quality],
            alpha=0.78 if quality in {"A", "B"} else 0.48,
            linewidth=0,
            rasterized=True,
            label=f"{quality} (n={len(points)})",
        )
    ax.legend(
        loc="lower right",
        ncol=2,
        frameon=True,
        facecolor="white",
        framealpha=0.90,
        markerscale=2.4,
    )
    ax.set_title("Full-area PS quality assessment")
    ax.text(
        0.01,
        0.99,
        "A/B: strong observations; C/D: down-weighted; rejected/invalid: excluded",
        transform=ax.transAxes,
        va="top",
        color="white",
        bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
    )
    save_figure(fig, output_root, "07_full_area_ps_quality_assessment")
    return representative_fid


def run(
    config_path: Path,
    output_root: Path,
    ps_points: Path | None = None,
    ps_reference_dsm: Path | None = None,
    base_elevation_m: float | None = None,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    route_config = json.loads(json.dumps(config))
    route_config["surface_priority"] = ["roof", "wall", "bottom"]
    route_config["figure2_registration_mode"] = "all_reference_accepted"
    if base_elevation_m is not None:
        route_config["base_elevation_m"] = float(base_elevation_m)
    if ps_points is not None:
        normalized_ps, ps_input = normalize_ps_input(
            ps_points,
            output_root,
            reference_dsm=ps_reference_dsm,
        )
        route_config["inputs"]["ps_points"] = str(normalized_ps)
    else:
        configured_ps = resolve(route_config["inputs"]["ps_points"])
        ps_input = {
            "source": str(configured_ps),
            "normalized": None,
            "rows": int(len(pd.read_csv(configured_ps))),
            "height_m_normalized_from": "height_m",
        }
    route_config_path = output_root / "touying2_config.json"
    route_config_path.write_text(
        json.dumps(route_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    heights_path, registration_path, buildings = prepare_inputs(
        route_config, output_root
    )
    roofs_before = initial_roofs(route_config, buildings)
    roof_vector = output_root / "vectors" / "initial_absolute_height_roofs.gpkg"
    roof_vector.parent.mkdir(parents=True, exist_ok=True)
    roofs_before.to_file(roof_vector, layer="absolute_height_roofs", driver="GPKG")

    mapping_root = output_root / "mapping"
    mapping_summary = run_mapping(
        route_config_path,
        0,
        heights_path,
        registration_path,
        mapping_root,
    )
    mapped_path = mapping_root / "tables" / "ps_building_surface_coordinates.csv"
    adjustment_root = output_root / "adjustment"
    adjustment_summary = run_adjustment(
        route_config_path,
        mapped_path,
        adjustment_root,
    )
    estimates = pd.read_csv(
        adjustment_root / "building_heights_triangle_adjustment.csv"
    )
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
    accepted = primary | supplementary
    estimates["final_quality"] = "rejected"
    estimates.loc[primary, "final_quality"] = estimates.loc[primary, "quality"]
    estimates.loc[supplementary, "final_quality"] = "supplementary"
    estimates["height_final_m"] = estimates.height_est_m.where(accepted)
    final_table = output_root / "tables" / "building_height_estimates.csv"
    estimates.to_csv(final_table, index=False)

    vector_buildings = gpd.read_file(resolve(route_config["inputs"]["buildings"])).reset_index(drop=True)
    vector_buildings["fid"] = np.arange(len(vector_buildings), dtype=np.int64)
    final_vector = vector_buildings.merge(
        estimates.drop(columns=["clean_id"], errors="ignore"),
        on="fid",
        how="left",
        validate="one_to_one",
    )
    final_vector_path = output_root / "vectors" / "building_height_estimates.gpkg"
    final_vector.to_file(
        final_vector_path,
        layer="roof_first_height_estimates",
        driver="GPKG",
    )

    triangles = gpd.read_file(
        mapping_root / "triangles" / "building_surface_triangles_radar.gpkg"
    )
    mapped = pd.read_csv(mapped_path)
    equations = pd.read_csv(
        adjustment_root / "ps_triangle_height_equations.csv"
    )
    ps_quality_table = output_root / "tables" / "ps_quality_assessment.csv"
    equations.to_csv(ps_quality_table, index=False)
    representative_fid = make_figures(
        route_config,
        output_root,
        buildings,
        roofs_before,
        triangles,
        mapped,
        equations,
        estimates,
    )

    finite = estimates.loc[np.isfinite(estimates.height_final_m)]
    registration = pd.read_csv(registration_path)
    summary = {
        "method": (
            "figure2_registered_absolute_roof_projection_full_area_triangles_"
            "ps_quality_scoring_heteroscedastic_wall_bias_robust_adjustment"
        ),
        "core_conclusion": (
            "Registering the most observable roof first provides the anchor for "
            "subsequent wall-triangle projection and PS height equations."
        ),
        "base_elevation_m": float(route_config["base_elevation_m"]),
        "roof_absolute_elevation_source": "Shapefile height",
        "roof_registration_basis": (
            "Figure 02: all reference rows with registration_accepted=1"
        ),
        "roof_registration_accepted_buildings": int(registration.accepted.sum()),
        "roof_registration_rejected_or_global_only_buildings": int(
            len(registration) - registration.accepted.sum()
        ),
        "invalid_roof_not_above_base": int(
            (pd.read_csv(heights_path).projection_valid == 0).sum()
        ),
        "mapped_ps": int(mapping_summary["mapped_ps"]),
        "candidate_ps_before_mask_refinement": int(
            mapping_summary["candidate_ps_before_mask_refinement"]
        ),
        "mask_refinement": mapping_summary["mask_refinement"],
        "surface_counts": mapping_summary["surface_counts"],
        "ps_quality_counts": {
            str(key): int(value)
            for key, value in equations.ps_quality_class.value_counts().items()
        },
        "estimated_buildings_before_final_qc": int(
            adjustment_summary["estimated_buildings"]
        ),
        "final_accepted_buildings": int(len(finite)),
        "final_primary_buildings": int(primary.sum()),
        "final_supplementary_buildings": int(supplementary.sum()),
        "height_m": {
            "min": float(finite.height_final_m.min()) if len(finite) else None,
            "median": float(finite.height_final_m.median()) if len(finite) else None,
            "mean": float(finite.height_final_m.mean()) if len(finite) else None,
            "max": float(finite.height_final_m.max()) if len(finite) else None,
        },
        "representative_fid": representative_fid,
        "ps_height_definition": ps_input.get(
            "height_formula", "height_m = input height_m"
        ),
        "ps_height_warning": ps_input.get(
            "interpretation_warning",
            "Consult the normalized PS input for height provenance.",
        ),
        "ps_input": ps_input,
        "figure_contract": {
            "archetype": "standalone image plates plus final quantitative map",
            "one_plot_per_file": True,
            "backend": "Python/matplotlib",
            "base_png_count": 8,
            "base_svg_count": 8,
            "final_height_map": "09_highrise_optimized_building_height_estimation",
        },
        "adjustment_model": {
            "equation": (
                "z_ps - z_base = height_fraction * H + "
                "I_wall * wall_bias + error"
            ),
            "base_weight_components": [
                "coherence",
                "building_overlap_ambiguity",
                "triangle_interior_position",
                "vertical_height_leverage",
                "roof_registration_reliability",
                "surface_type",
            ],
            "robust_loss": "Huber IRLS followed by explicit outlier removal",
            "wall_bias_constraint": "zero-centred ridge, sigma=3 m",
        },
        "outputs": {
            "root": str(output_root),
            "final_table": str(final_table),
            "final_vector": str(final_vector_path),
            "full_area_triangle_vector": str(
                mapping_root
                / "triangles"
                / "building_surface_triangles_radar.gpkg"
            ),
            "ps_quality_table": str(ps_quality_table),
            "png": str(output_root / "png"),
            "svg": str(output_root / "svg"),
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Project roofs at absolute Shapefile height, register roofs first, "
            "then form side triangles and estimate building heights."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--ps-points",
        type=Path,
        help=(
            "Override the configured PS table. PSI coordinate-package columns "
            "(including psi_scatterer_elevation_wusong_m) are normalized "
            "automatically; otherwise height_m, z_dsm_m, or terrain_height_m "
            "is used."
        ),
    )
    parser.add_argument(
        "--ps-reference-dsm",
        type=Path,
        help=(
            "Sample this EPSG:32651 DSM at each PS x/y coordinate and construct "
            "height_m as z_dsm_m plus the sampled DSM prior."
        ),
    )
    parser.add_argument(
        "--base-elevation-m",
        type=float,
        help="Override the configured ground/base elevation for this run.",
    )
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.output_root.resolve(),
        ps_points=args.ps_points,
        ps_reference_dsm=args.ps_reference_dsm,
        base_elevation_m=args.base_elevation_m,
    )


if __name__ == "__main__":
    main()
