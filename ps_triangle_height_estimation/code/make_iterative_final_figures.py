from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ps-triangle-iterative-figures")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import binary_dilation
from shapely.affinity import translate
from shapely.ops import unary_union

from geometry import parse_gamma_par
from recompute_iterative_local_registration import rasterize_triangles
from recompute_rooftop_registration import read_rooftop_features


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "iterative_triangle_adjustment_stable_registration"
CONFIG = ROOT / "config.json"
PROJECTION_RESULTS: Path | None = None
PICALL = ROOT / "results" / "picall"
PNG_DIR = PICALL / "png"
SVG_DIR = PICALL / "svg"
REPRESENTATIVE_FID = 243
ROOFTOP_REGISTRATION_FID = 926

COLORS = {
    "blue": "#277DA1",
    "teal": "#38A3A5",
    "gold": "#F2B134",
    "magenta": "#C84C9A",
    "red": "#D6604D",
    "dark": "#30343B",
    "mid": "#6B7280",
    "light": "#D8DEE4",
    "pale": "#EEF1F3",
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
        "path.simplify": False,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    """Write one scientific plot per file, separated by output format."""
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        SVG_DIR / f"{stem}.svg",
        dpi=144,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "Python/matplotlib; editable vector overlays"},
    )
    fig.savefig(PNG_DIR / f"{stem}.png", dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def final_iteration() -> int:
    return int(pd.read_csv(RESULTS / "iteration_convergence.csv").iteration.max())


def final_iteration_dir() -> Path:
    return RESULTS / f"iteration_{final_iteration():02d}"


def final_projection_dir() -> Path:
    """Return an optional height-refined mapping, otherwise the final iteration."""
    if PROJECTION_RESULTS is not None:
        return PROJECTION_RESULTS
    return final_iteration_dir() / "mapping"


def read_rslc_display(path: Path, par_path: Path) -> np.ndarray:
    par = parse_gamma_par(par_path)
    rows, cols = int(par["azimuth_lines"]), int(par["range_samples"])
    raw = np.fromfile(path, dtype=">i2")
    if raw.size != rows * cols * 2:
        raise ValueError(f"Unexpected RSLC sample count: {raw.size}")
    iq = raw.reshape(rows, cols, 2).astype(np.float32)
    amplitude = np.hypot(iq[:, :, 0], iq[:, :, 1])
    valid = amplitude[amplitude > 0]
    low, high = np.percentile(valid, [2, 98])
    return np.clip((amplitude - low) / max(float(high - low), 1e-6), 0, 1) ** 0.55


def load_amplitude() -> np.ndarray:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    par_path = (ROOT / config["inputs"]["rslc_par"]).resolve()
    return read_rslc_display(par_path.with_suffix(""), par_path)


def geometry_segments(frame: gpd.GeoDataFrame) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for geometry in frame.geometry:
        if geometry is None or geometry.is_empty:
            continue
        if geometry.geom_type == "Polygon":
            parts = [geometry]
        elif geometry.geom_type == "MultiPolygon":
            parts = list(geometry.geoms)
        else:
            continue
        segments.extend(np.asarray(part.exterior.coords) for part in parts)
    return segments


def figure_01_final_projection() -> None:
    """Full-scene bottom/wall/roof triangle projections on the SAR image."""
    amplitude = load_amplitude()
    triangles = gpd.read_file(
        final_projection_dir() / "triangles" / "building_surface_triangles_radar.gpkg"
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    ax.imshow(
        np.clip(amplitude * 0.78, 0.0, 1.0),
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        rasterized=True,
    )
    surface_styles = {
        "bottom": ("#45D4D4", 0.16, 0.24),
        "wall": (COLORS["magenta"], 0.19, 0.30),
        "roof": (COLORS["gold"], 0.28, 0.64),
    }
    for surface, (color, width, alpha) in surface_styles.items():
        selected = triangles.loc[triangles.surface == surface]
        collection = LineCollection(
            geometry_segments(selected),
            colors=color,
            linewidths=width,
            alpha=alpha,
            rasterized=False,
            zorder=3,
        )
        collection.set_gid(f"projection-{surface}-triangles")
        ax.add_collection(collection)
    roof_outlines = triangles.loc[triangles.surface == "roof"].dissolve(
        by="building_fid"
    )
    roof_outline_collection = LineCollection(
        geometry_segments(roof_outlines),
        colors="#FFD166",
        linewidths=0.42,
        alpha=0.78,
        rasterized=False,
        zorder=4,
    )
    roof_outline_collection.set_gid("projection-roof-outlines")
    ax.add_collection(roof_outline_collection)
    ax.set_xlim(0, amplitude.shape[1])
    ax.set_ylim(amplitude.shape[0], 0)
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    ax.set_title(
        "Height-refined bottom, wall and roof projections on the SAR image",
        fontsize=9,
    )
    ax.legend(
        handles=[
            Patch(facecolor="none", edgecolor=surface_styles["bottom"][0], label="Bottom triangles"),
            Patch(facecolor="none", edgecolor=surface_styles["wall"][0], label="Wall triangles"),
            Patch(facecolor="none", edgecolor=surface_styles["roof"][0], label="Roof triangles"),
        ],
        loc="upper right",
        fontsize=6.5,
        frameon=True,
        framealpha=0.86,
        facecolor="white",
        edgecolor="#D4D8DC",
    )
    surface_counts = triangles.surface.value_counts()
    stable_count = int(
        pd.read_csv(RESULTS / "building_heights_iterative_screened.csv")
        .final_status.eq("stable")
        .sum()
    )
    ax.text(
        0.015,
        0.02,
        f"Iteration {final_iteration()}  |  {triangles.building_fid.nunique():,} buildings  |  "
        f"{len(triangles):,} projected triangles\n"
        f"bottom={surface_counts.get('bottom', 0):,}, wall={surface_counts.get('wall', 0):,}, "
        f"roof={surface_counts.get('roof', 0):,}  |  final stable heights inserted={stable_count:,}",
        transform=ax.transAxes,
        color="white",
        fontsize=6.5,
        bbox={"facecolor": "black", "alpha": 0.58, "edgecolor": "none", "pad": 3},
    )
    fig.tight_layout()
    save(fig, "01_final_triangle_projections_on_sar")


def figure_02_ps_surface_classification() -> None:
    """One representative SAR crop showing roof/wall triangle and PS assignments."""
    amplitude = load_amplitude()
    triangles = gpd.read_file(
        final_projection_dir() / "triangles" / "building_surface_triangles_radar.gpkg"
    )
    mapped = pd.read_csv(
        final_projection_dir() / "tables" / "ps_building_surface_coordinates.csv"
    )
    local_triangles = triangles.loc[triangles.building_fid == REPRESENTATIVE_FID]
    local_ps = mapped.loc[mapped.fid == REPRESENTATIVE_FID]
    bounds = local_triangles.total_bounds
    pad = 7.0
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    ax.imshow(
        np.clip(amplitude * 0.78, 0.0, 1.0),
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        rasterized=True,
    )
    surface_styles = {
        "bottom": ("#45D4D4", 0.75),
        "wall": (COLORS["magenta"], 1.05),
        "roof": (COLORS["gold"], 1.20),
    }
    for surface, (color, width) in surface_styles.items():
        selected = local_triangles.loc[local_triangles.surface == surface]
        collection = LineCollection(
            geometry_segments(selected),
            colors=color,
            linewidths=width,
            alpha=0.95,
            rasterized=False,
            zorder=3,
        )
        collection.set_gid(f"classification-{surface}-triangles")
        ax.add_collection(collection)
    for surface, color, marker in [
        ("roof", COLORS["teal"], "o"),
        ("wall", COLORS["magenta"], "^"),
    ]:
        selected = local_ps.loc[local_ps.surface == surface]
        ax.scatter(
            selected.col0,
            selected.row0,
            s=28,
            c=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.55,
            label=f"{surface.capitalize()} PS (n={len(selected)})",
            zorder=5,
        )
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[3] + pad, bounds[1] - pad)
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    ax.set_title(
        f"PS surface classification after final projection — building {REPRESENTATIVE_FID}",
        fontsize=9,
    )
    ax.legend(
        loc="upper right",
        fontsize=6.5,
        frameon=True,
        facecolor="white",
        edgecolor="#D4D8DC",
        framealpha=0.88,
    )
    fig.tight_layout()
    save(fig, "02_ps_roof_wall_classification_on_sar")


def figure_03_initial_registration() -> None:
    """Two-dimensional distribution of integer and subpixel rooftop refinements."""
    table = pd.read_csv(RESULTS / "iteration_00" / "local_registration.csv")
    accepted = table.loc[table.local_refinement_accepted == 1].copy()
    accepted["reference_multiscene_override"] = (
        accepted.get("reference_multiscene_override", 0).fillna(0).astype(int)
    )
    limit = int(
        np.ceil(
        max(
            accepted.applied_row_shift.abs().max(),
            accepted.applied_col_shift.abs().max(),
            2,
        )
        )
    )
    fig, ax = plt.subplots(figsize=(4.7, 4.0))
    styles = [
        (0, COLORS["blue"], "Single-scene rooftop refinement"),
        (1, COLORS["gold"], "Multi-scene subpixel override"),
    ]
    for flag, color, label in styles:
        subset = accepted.loc[accepted.reference_multiscene_override == flag]
        grouped = (
            subset.groupby(["applied_row_shift", "applied_col_shift"])
            .size()
            .rename("count")
            .reset_index()
        )
        if grouped.empty:
            continue
        ax.scatter(
            grouped.applied_col_shift,
            grouped.applied_row_shift,
            s=16 + 12 * np.sqrt(grouped["count"]),
            color=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.82,
            label=f"{label} (n={len(subset)})",
        )
    ax.axhline(0, color=COLORS["mid"], lw=0.6, alpha=0.7)
    ax.axvline(0, color=COLORS["mid"], lw=0.6, alpha=0.7)
    ticks = np.arange(-limit, limit + 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(-limit - 0.5, limit + 0.5)
    ax.set_ylim(-limit - 0.5, limit + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Range refinement (pixels)")
    ax.set_ylabel("Azimuth refinement (pixels)")
    ax.set_title("High-confidence rooftop refinements retained after quality gating", fontsize=9)
    ax.legend(loc="upper right", fontsize=6.2)
    fig.text(
        0.47,
        0.015,
        f"Retained: {len(accepted)}/{len(table)} buildings  |  "
        "scene consensus: (row 0, col 0)",
        ha="center",
        va="bottom",
        color=COLORS["dark"],
        fontsize=6.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "03_initial_local_registration_shifts")


def figure_04_height_equations() -> None:
    """Representative building equations y=fH with inlier/outlier visibility."""
    iteration_dir = final_iteration_dir()
    equations = pd.read_csv(
        iteration_dir / "adjustment" / "ps_triangle_height_equations.csv"
    )
    screened = pd.read_csv(RESULTS / "building_heights_iterative_screened.csv")
    group = equations.loc[
        (equations.fid == REPRESENTATIVE_FID) & equations.equation_valid
    ].copy()
    result = screened.loc[screened.fid == REPRESENTATIVE_FID].iloc[0]
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    base_elevation = float(config["base_elevation_m"])
    inlier = group.adjustment_inlier == 1
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    outliers = group.loc[~inlier]
    if len(outliers):
        ax.scatter(
            outliers.height_fraction,
            outliers.observed_above_base_m,
            s=24,
            facecolors="none",
            edgecolors="#A7ADB4",
            linewidths=0.8,
            label=f"Down-weighted / rejected (n={len(outliers)})",
            zorder=2,
        )
    for surface, color, marker in [
        ("roof", COLORS["teal"], "o"),
        ("wall", COLORS["magenta"], "^"),
    ]:
        selected = group.loc[inlier & (group.surface == surface)]
        ax.scatter(
            selected.height_fraction,
            selected.observed_above_base_m,
            s=30,
            c=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.88,
            label=f"{surface.capitalize()} inliers (n={len(selected)})",
            zorder=4,
        )
    x = np.linspace(0.0, 1.02, 120)
    height = float(result.height_final_m)
    ax.plot(x, height * x, color=COLORS["dark"], lw=1.5, label=rf"Robust fit: $H={height:.2f}$ m")
    ax.set_xlim(0, 1.04)
    ymax = max(float(group.observed_above_base_m.max()), height) * 1.08
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"Vertical fraction from barycentric geometry, $f_i$")
    ax.set_ylabel(
        rf"PS elevation above the {base_elevation:g} m base, "
        rf"$z_i-{base_elevation:g}$ (m)"
    )
    ax.set_title(
        f"Robust PS height equations for representative building {REPRESENTATIVE_FID}",
        fontsize=9,
    )
    ax.grid(color="#E1E5E9", lw=0.5)
    ax.legend(loc="upper left", fontsize=6.2)
    ax.text(
        0.98,
        0.05,
        f"Used equations: {int(result.ps_equations_used)}\n"
        f"Weighted residual RMS: {float(result.weighted_residual_rms_m):.2f} m",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=COLORS["dark"],
    )
    fig.tight_layout()
    save(fig, "04_representative_building_ps_height_equations")


def figure_05_height_convergence() -> None:
    """One plot dedicated to the height-update convergence evidence."""
    table = pd.read_csv(RESULTS / "iteration_convergence.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(
        table.iteration,
        table.median_abs_height_change_m,
        "o-",
        color=COLORS["blue"],
        lw=1.7,
        ms=4,
        label="Median absolute update",
    )
    ax.plot(
        table.iteration,
        table.p90_abs_height_change_m,
        "o-",
        color=COLORS["red"],
        lw=1.3,
        ms=3.5,
        label="90th percentile",
    )
    ax.axhline(0.25, color=COLORS["blue"], lw=0.8, ls="--", alpha=0.75)
    ax.axhline(1.0, color=COLORS["red"], lw=0.8, ls="--", alpha=0.75)
    ax.set_yscale("log")
    ax.set_xticks(table.iteration)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Absolute damped height update (m)")
    ax.set_title("Height updates decrease, but the scene-wide P90 criterion is not met", fontsize=9)
    ax.grid(axis="y", color="#D8DDE3", lw=0.5)
    ax.legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    save(fig, "05_iterative_height_update_convergence")


def figure_06_assignment_stability() -> None:
    """One plot dedicated to PS building/surface assignment stability."""
    table = pd.read_csv(RESULTS / "iteration_convergence.csv")
    valid = table.loc[table.iteration > 0]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(
        valid.iteration,
        100 * valid.same_building_fraction,
        "o-",
        color=COLORS["teal"],
        lw=1.4,
        ms=4,
        label="Same building",
    )
    ax.plot(
        valid.iteration,
        100 * valid.same_surface_fraction,
        "o-",
        color=COLORS["blue"],
        lw=1.7,
        ms=4,
        label="Same building and surface",
    )
    ax.axhline(98, color=COLORS["mid"], lw=0.8, ls="--", label="98% stopping threshold")
    ax.set_ylim(90, 100.2)
    ax.set_xticks(table.iteration)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("PS assignment stability (%)")
    ax.set_title("PS building and roof/wall assignments stabilize across iterations", fontsize=9)
    ax.grid(axis="y", color="#D8DDE3", lw=0.5)
    ax.legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    save(fig, "06_ps_assignment_stability")


def figure_07_screened_height_map() -> None:
    """Final spatial result; only stable buildings receive a height color."""
    vector = gpd.read_file(RESULTS / "building_heights_iterative_screened.gpkg")
    bounds = [3, 10, 20, 30, 45, 65, 90, 150]
    colors = ["#4575B4", "#74ADD1", "#66C2A5", "#2CA25F", "#F6C85F", "#F08A5D", "#C74440"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    stable = vector.loc[vector.final_status == "stable"].copy()
    rejected = vector.loc[vector.final_status.isin(["unstable_iteration", "low_quality"])].copy()
    no_solution = vector.loc[
        ~vector.index.isin(stable.index) & ~vector.index.isin(rejected.index)
    ].copy()
    fig, ax = plt.subplots(figsize=(5.0, 6.6))
    no_solution.plot(
        ax=ax,
        color=COLORS["pale"],
        edgecolor="#D5D9DD",
        linewidth=0.18,
        rasterized=False,
    )
    rejected.plot(
        ax=ax,
        color="#B9C0C7",
        edgecolor="#F8F9FA",
        linewidth=0.20,
        rasterized=False,
    )
    stable.plot(
        ax=ax,
        column="height_final_m",
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.28,
        rasterized=False,
    )
    for row in stable.itertuples():
        point = row.geometry.representative_point()
        height = float(row.height_final_m)
        rgba = cmap(norm(height))
        luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        text_color = "#202020" if luminance > 0.60 else "white"
        ax.text(
            point.x,
            point.y,
            f"{height:.0f}",
            ha="center",
            va="center",
            fontsize=2.25,
            fontweight="bold",
            color=text_color,
            clip_on=True,
            zorder=5,
        )
    ax.set_axis_off()
    ax.set_title(
        "Building heights retained after per-building convergence screening — values in metres",
        fontsize=9,
        pad=8,
    )
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cbar = fig.colorbar(
        scalar,
        ax=ax,
        orientation="horizontal",
        fraction=0.045,
        pad=0.025,
        ticks=bounds,
        spacing="proportional",
    )
    cbar.set_label("Stable estimated building height (m)")
    cbar.ax.tick_params(length=2)
    ax.legend(
        handles=[
            Patch(facecolor="#B9C0C7", edgecolor="none", label="Estimated but unstable / low quality"),
            Patch(facecolor=COLORS["pale"], edgecolor="#D5D9DD", label="No valid PS height equation"),
        ],
        loc="lower left",
        fontsize=6.5,
    )
    ax.text(
        0.99,
        0.015,
        f"Stable estimates: n={len(stable)} of {len(vector)} buildings",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color=COLORS["dark"],
    )
    fig.tight_layout()
    save(fig, "07_screened_iterative_building_height_map")


def figure_rooftop_registration(after: bool) -> None:
    """Standalone before/after roof-outline plate using the same SAR crop and features."""
    amplitude = load_amplitude()
    triangles = gpd.read_file(
        final_projection_dir() / "triangles" / "building_surface_triangles_radar.gpkg"
    )
    roof_triangles = triangles.loc[
        (triangles.building_fid == ROOFTOP_REGISTRATION_FID)
        & (triangles.surface == "roof")
    ].copy()
    registration = pd.read_csv(RESULTS / "iteration_00" / "local_registration.csv")
    row = registration.loc[registration.fid == ROOFTOP_REGISTRATION_FID].iloc[0]
    dr, dc = float(row.applied_row_shift), float(row.applied_col_shift)
    registered_roof = unary_union(list(roof_triangles.geometry))
    displayed_roof = registered_roof if after else translate(registered_roof, xoff=-dc, yoff=-dr)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    par_path = (ROOT / config["inputs"]["rslc_par"]).resolve()
    par = parse_gamma_par(par_path)
    features = read_rooftop_features(
        par_path.with_suffix(""),
        (int(par["azimuth_lines"]), int(par["range_samples"])),
    )
    registered_mask = rasterize_triangles(
        [np.asarray(geometry.exterior.coords) for geometry in roof_triangles.geometry],
        amplitude.shape,
    )
    neighborhood = binary_dilation(registered_mask, iterations=6)
    support = (
        features["roof_likelihood"] >= float(row.support_threshold)
    ) & neighborhood

    all_bounds = np.asarray([registered_roof.bounds, displayed_roof.bounds])
    pad = 10
    x0 = max(0, int(np.floor(all_bounds[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(all_bounds[:, 1].min())) - pad)
    x1 = min(amplitude.shape[1], int(np.ceil(all_bounds[:, 2].max())) + pad)
    y1 = min(amplitude.shape[0], int(np.ceil(all_bounds[:, 3].max())) + pad)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.imshow(
        np.clip(amplitude[y0:y1, x0:x1] * 0.78, 0.0, 1.0),
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=(x0, x1, y1, y0),
        interpolation="nearest",
    )
    local_support = support[y0:y1, x0:x1]
    yy, xx = np.nonzero(local_support)
    ax.scatter(
        xx + x0,
        yy + y0,
        s=13,
        marker="s",
        c=COLORS["gold"],
        alpha=0.30,
        linewidths=0,
        label="Adaptive SAR rooftop-feature support",
        zorder=2,
    )
    outline = np.asarray(displayed_roof.exterior.coords)
    ax.plot(
        outline[:, 0],
        outline[:, 1],
        color="#22D3EE",
        lw=2.0,
        label="Projected vector roof outline",
        zorder=4,
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    state = "after" if after else "before"
    ax.set_title(
        f"Projected roof outline {state} rooftop-feature registration — building "
        f"{ROOFTOP_REGISTRATION_FID}",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=6.5, frameon=True, facecolor="white")
    if after:
        annotation = (
        f"Accepted refinement: row {dr:+.2f}, col {dc:+.2f} px\n"
            f"score gain={float(row.score_gain):.2f}, peak margin={float(row.peak_margin):.2f}"
        )
    else:
        annotation = "Unrefined position: scene-consensus projection"
    ax.text(
        0.02,
        0.04,
        annotation,
        transform=ax.transAxes,
        color="white",
        fontsize=6.5,
        bbox={"facecolor": "black", "alpha": 0.58, "edgecolor": "none", "pad": 3},
    )
    fig.tight_layout()
    save(
        fig,
        "09_rooftop_vector_after_feature_registration"
        if after
        else "08_rooftop_vector_before_feature_registration",
    )


def figure_08_rooftop_before() -> None:
    figure_rooftop_registration(after=False)


def figure_09_rooftop_after() -> None:
    figure_rooftop_registration(after=True)


FIGURES = {
    "01": figure_01_final_projection,
    "02": figure_02_ps_surface_classification,
    "03": figure_03_initial_registration,
    "04": figure_04_height_equations,
    "05": figure_05_height_convergence,
    "06": figure_06_assignment_stability,
    "07": figure_07_screened_height_map,
    "08": figure_08_rooftop_before,
    "09": figure_09_rooftop_after,
}


def main() -> None:
    global RESULTS, CONFIG, PROJECTION_RESULTS, PNG_DIR, SVG_DIR
    global REPRESENTATIVE_FID, ROOFTOP_REGISTRATION_FID
    parser = argparse.ArgumentParser(
        description="Create standalone figures for the current stable-registration height method."
    )
    parser.add_argument("--figure", choices=[*FIGURES, "all"], default="all")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--projection-dir",
        type=Path,
        help=(
            "Optional optimized mapping directory containing triangles/, "
            "tables/, and vectors/."
        ),
    )
    parser.add_argument("--png-dir", type=Path, default=PNG_DIR)
    parser.add_argument("--svg-dir", type=Path, default=SVG_DIR)
    parser.add_argument("--representative-fid", type=int, default=REPRESENTATIVE_FID)
    parser.add_argument(
        "--rooftop-registration-fid",
        type=int,
        default=ROOFTOP_REGISTRATION_FID,
    )
    args = parser.parse_args()
    RESULTS = args.results_dir.resolve()
    CONFIG = args.config.resolve()
    PROJECTION_RESULTS = (
        args.projection_dir.resolve() if args.projection_dir is not None else None
    )
    PNG_DIR = args.png_dir.resolve()
    SVG_DIR = args.svg_dir.resolve()
    REPRESENTATIVE_FID = int(args.representative_fid)
    ROOFTOP_REGISTRATION_FID = int(args.rooftop_registration_fid)
    selected = FIGURES.values() if args.figure == "all" else [FIGURES[args.figure]]
    for function in selected:
        function()
    print(
        json.dumps(
            {
                "method": "stable_registration_iterative_triangle_ps_height_adjustment",
                "png_dir": str(PNG_DIR),
                "svg_dir": str(SVG_DIR),
                "figures": list(FIGURES),
                "one_plot_per_file": True,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
