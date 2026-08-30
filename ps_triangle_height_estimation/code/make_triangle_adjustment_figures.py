from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ps-triangle-adjustment")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from geometry import parse_gamma_par
from optimize_svg import optimize_svg_file


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
PICALL = ROOT / "results" / "picall"
ADJUSTMENT = ROOT / "results" / "triangle_adjustment_height_estimation"
TRIANGLES = ROOT / "results" / "triangles"

COLORS = {
    "roof": "#E8A317",
    "wall": "#C6519C",
    "ps": "#20B9C7",
    "blue": "#19558D",
    "teal": "#3C8D93",
    "red": "#B64342",
    "dark": "#2A2A2A",
    "mid": "#777777",
    "pale": "#E6E8EA",
}
QUALITY_COLORS = {
    "high": COLORS["blue"],
    "medium": COLORS["teal"],
    "low": "#A5A5A5",
    "no_ps_equation": "#E1E1E1",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def panel_label(ax, label: str, inside: bool = False, color: str = "black") -> None:
    ax.text(
        0.015 if inside else -0.10,
        0.985 if inside else 1.02,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top" if inside else "bottom",
        fontsize=9,
        fontweight="bold",
        color=color,
        zorder=30,
    )


def save_figure(fig, stem: str, tiff: bool = False, dpi: int = 350) -> None:
    PICALL.mkdir(parents=True, exist_ok=True)
    # Hybrid SVG: dense geographic layers are rasterized at high resolution,
    # while labels, axes, legends, and annotations stay editable vectors.
    svg_path = PICALL / f"{stem}.svg"
    fig.savefig(svg_path, dpi=200, bbox_inches="tight", facecolor="white")
    optimize_svg_file(svg_path)
    fig.savefig(PICALL / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    if tiff:
        fig.savefig(
            PICALL / f"{stem}.tiff",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    plt.close(fig)


def read_rslc_amplitude(path: Path, par_path: Path) -> np.ndarray:
    par = parse_gamma_par(par_path)
    rows, cols = int(par["azimuth_lines"]), int(par["range_samples"])
    raw = np.fromfile(path, dtype=">i2")
    if raw.size != rows * cols * 2:
        raise ValueError(f"Unexpected RSLC sample count: {raw.size}")
    iq = raw.reshape(rows, cols, 2).astype(np.float32)
    return np.hypot(iq[:, :, 0], iq[:, :, 1])


def display_stretch(amplitude: np.ndarray) -> np.ndarray:
    valid = amplitude[np.isfinite(amplitude) & (amplitude > 0)]
    low, high = np.percentile(valid, [2, 98])
    return np.clip((amplitude - low) / max(high - low, 1e-6), 0, 1) ** 0.55


def polygon_segments(frame: gpd.GeoDataFrame) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for geometry in frame.geometry:
        if geometry is None or geometry.is_empty:
            continue
        if geometry.geom_type == "Polygon":
            segments.append(np.asarray(geometry.exterior.coords))
        elif geometry.geom_type == "MultiPolygon":
            segments.extend(np.asarray(part.exterior.coords) for part in geometry.geoms)
    return segments


def add_scalebar(ax, length_m: float = 500.0) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + 0.055 * (xmax - xmin)
    y0 = ymin + 0.055 * (ymax - ymin)
    ax.plot([x0, x0 + length_m], [y0, y0], color=COLORS["dark"], lw=2)
    ax.text(x0 + length_m / 2, y0 + 0.018 * (ymax - ymin), f"{int(length_m)} m", ha="center", fontsize=6)


def add_north_arrow(ax) -> None:
    ax.annotate(
        "N",
        xy=(0.94, 0.92),
        xytext=(0.94, 0.82),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": COLORS["dark"], "lw": 1.1},
    )


def figure_22_adjustment_geometry(
    amplitude: np.ndarray,
    triangles: gpd.GeoDataFrame,
    equations: pd.DataFrame,
    estimates: pd.DataFrame,
) -> None:
    # A balanced roof/wall case with low residuals, selected for method display.
    fid = 185
    local_triangles = triangles.loc[triangles.building_fid == fid]
    local = equations.loc[(equations.fid == fid) & equations.equation_valid].copy()
    result = estimates.loc[estimates.fid == fid].iloc[0]
    bounds = local_triangles.total_bounds
    display = np.clip(display_stretch(amplitude) * 0.72, 0, 1)

    fig = plt.figure(figsize=(7.2, 3.25), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.08, 1.15, 0.85], wspace=0.16)
    ax_a, ax_b, ax_c = [fig.add_subplot(grid[0, index]) for index in range(3)]

    ax_a.imshow(display, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for surface in ["wall", "roof"]:
        selected = local_triangles.loc[local_triangles.surface == surface]
        ax_a.add_collection(
            LineCollection(
                polygon_segments(selected),
                colors=COLORS[surface],
                linewidths=0.9 if surface == "wall" else 1.15,
                alpha=0.9,
            )
        )
    for surface in ["wall", "roof"]:
        selected = local.loc[local.surface == surface]
        ax_a.scatter(
            selected.col0,
            selected.row0,
            s=22,
            c=COLORS[surface],
            marker="o" if surface == "roof" else "^",
            edgecolors="white",
            linewidths=0.45,
            zorder=8,
            label=f"{surface.capitalize()} PS",
        )
    pad = 7
    ax_a.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax_a.set_ylim(bounds[3] + pad, bounds[1] - pad)
    ax_a.set_xlabel("Range column")
    ax_a.set_ylabel("Azimuth row")
    ax_a.set_title("Locally registered PS classification")
    ax_a.legend(loc="upper right", fontsize=5.5)
    panel_label(ax_a, "a", inside=True, color="white")

    inlier = local.adjustment_inlier == 1
    for surface in ["wall", "roof"]:
        selected = local.loc[inlier & (local.surface == surface)]
        ax_b.scatter(
            selected.height_fraction,
            selected.observed_above_base_m,
            s=22 if surface == "roof" else 17,
            c=COLORS[surface],
            marker="o" if surface == "roof" else "^",
            edgecolors="white",
            linewidths=0.35,
            alpha=0.85,
            label=f"{surface.capitalize()} PS",
        )
    rejected = local.loc[~inlier]
    if len(rejected):
        ax_b.scatter(
            rejected.height_fraction,
            rejected.observed_above_base_m,
            s=17,
            facecolors="none",
            edgecolors=COLORS["mid"],
            linewidths=0.7,
            label="Robustly rejected",
        )
    x = np.linspace(0, 1.03, 120)
    ax_b.plot(x, result.height_est_m * x, color=COLORS["blue"], lw=1.6, label="Adjusted $fH$")
    ax_b.set_xlim(0, 1.04)
    ax_b.set_ylim(bottom=0)
    ax_b.set_xlabel("Triangle height fraction, $f_i$")
    ax_b.set_ylabel("Known PS elevation above 4 m, $z_i-4$")
    ax_b.set_title("All PS form one adjustment line")
    ax_b.text(
        0.04,
        0.96,
        f"$z_i-4=f_iH+\\epsilon_i$\n$\\hat H$ = {result.height_est_m:.1f} m\n"
        f"roof/wall used = {int(result.roof_ps_used)}/{int(result.wall_ps_used)}",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
    )
    ax_b.legend(loc="lower right", fontsize=5.2)
    panel_label(ax_b, "b")

    used = local.loc[inlier]
    groups = [used.loc[used.surface == name, "adjustment_residual_m"] for name in ["roof", "wall"]]
    box = ax_c.boxplot(groups, tick_labels=["Roof", "Wall"], patch_artist=True, widths=0.55, showfliers=False)
    for patch, color in zip(box["boxes"], [COLORS["roof"], COLORS["wall"]], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    for median in box["medians"]:
        median.set_color(COLORS["dark"])
        median.set_linewidth(1.2)
    ax_c.axhline(0, color=COLORS["dark"], lw=0.8, ls="--")
    ax_c.set_ylabel("Adjustment residual (m)")
    ax_c.set_title("Roof and wall consistency")
    ax_c.text(
        0.5,
        0.04,
        f"weighted RMS = {result.weighted_residual_rms_m:.1f} m\n"
        f"uncertainty = {result.height_uncertainty_m:.1f} m",
        transform=ax_c.transAxes,
        ha="center",
        va="bottom",
        fontsize=6,
    )
    panel_label(ax_c, "c")
    fig.suptitle("Known PS elevations constrain one building height through triangle geometry", fontsize=9)
    save_figure(fig, "fig_22_triangle_ps_height_adjustment")


def figure_23_adjustment_diagnostics(estimates: pd.DataFrame, equations: pd.DataFrame) -> None:
    finite = estimates.loc[np.isfinite(estimates.height_est_m)].copy()
    used = equations.loc[equations.adjustment_inlier == 1].copy()
    order = ["high", "medium", "low", "no_ps_equation"]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    counts = estimates.quality.value_counts().reindex(order, fill_value=0)
    bars = ax_a.bar(
        np.arange(4),
        counts,
        width=0.7,
        color=[QUALITY_COLORS[value] for value in order],
    )
    ax_a.set_xticks(np.arange(4), ["High", "Medium", "Low", "No equation"])
    ax_a.set_ylabel("Buildings")
    ax_a.set_title("Adjustment quality")
    for bar, value in zip(bars, counts, strict=True):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 7, str(int(value)), ha="center", fontsize=6)

    for quality in ["high", "medium", "low"]:
        values = finite.loc[finite.quality == quality, "height_uncertainty_m"].clip(upper=15)
        ax_b.hist(
            values,
            bins=np.arange(0, 15.6, 0.75),
            histtype="step",
            lw=1.4,
            color=QUALITY_COLORS[quality],
            label=f"{quality} (n={len(values)})",
        )
    ax_b.set_xlabel("Estimated height uncertainty (m)")
    ax_b.set_ylabel("Buildings")
    ax_b.set_title("Internal uncertainty")
    ax_b.legend(fontsize=5.5)

    sample = used.sample(min(7000, len(used)), random_state=7)
    for surface in ["wall", "roof"]:
        selected = sample.loc[sample.surface == surface]
        ax_c.scatter(
            selected.height_fraction,
            selected.adjustment_residual_m,
            s=5,
            c=COLORS[surface],
            alpha=0.22,
            linewidths=0,
            label=f"{surface.capitalize()} PS",
            rasterized=True,
        )
    ax_c.axhline(0, color=COLORS["dark"], lw=0.8, ls="--")
    ax_c.set_xlim(0.08, 1.03)
    ax_c.set_xlabel("Triangle height fraction, $f_i$")
    ax_c.set_ylabel("Adjustment residual (m)")
    ax_c.set_title("Residuals across roof and wall geometry")
    ax_c.legend(fontsize=5.5)

    for quality in ["low", "medium", "high"]:
        selected = finite.loc[finite.quality == quality]
        ax_d.scatter(
            selected.height_prior_m,
            selected.height_est_m,
            s=9,
            c=QUALITY_COLORS[quality],
            alpha=0.6,
            linewidths=0,
            label=quality,
        )
    limit = 150
    ax_d.plot([0, limit], [0, limit], color=COLORS["mid"], lw=0.9, ls="--")
    ax_d.set_xlim(0, limit)
    ax_d.set_ylim(0, limit)
    ax_d.set_xlabel("SHP height used for projection (m)")
    ax_d.set_ylabel("PS triangle-adjusted height (m)")
    ax_d.set_title("Consistency check — not validation")
    ax_d.legend(fontsize=5.3, loc="upper left")

    for label, ax in zip("abcd", axes.ravel(), strict=True):
        panel_label(ax, label)
    fig.suptitle("Internal diagnostics of multi-PS triangle height adjustment", fontsize=9, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, "fig_23_triangle_adjustment_diagnostics")


def plot_height_map(ax, frame: gpd.GeoDataFrame, norm: Normalize, cmap) -> None:
    no_solution = frame.loc[~np.isfinite(frame.height_est_m)]
    finite = frame.loc[np.isfinite(frame.height_est_m)]
    if len(no_solution):
        no_solution.plot(
            ax=ax,
            facecolor=QUALITY_COLORS["no_ps_equation"],
            edgecolor="white",
            linewidth=0.08,
            rasterized=True,
        )
    finite.plot(
        ax=ax,
        column="height_est_m",
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.10,
        rasterized=True,
    )
    reliable = finite.loc[finite.quality.isin(["high", "medium"])]
    if len(reliable):
        reliable.boundary.plot(
            ax=ax,
            color=COLORS["dark"],
            linewidth=0.22,
            alpha=0.72,
            rasterized=True,
        )


def figure_24_final_map(buildings: gpd.GeoDataFrame, estimates: pd.DataFrame) -> None:
    frame = buildings.to_crs("EPSG:32651")
    finite = estimates.loc[np.isfinite(estimates.height_est_m)]
    cmap = mpl.colormaps["cividis"]
    norm = Normalize(vmin=3, vmax=130)
    fig = plt.figure(figsize=(7.2, 5.8))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.35, 0.9], hspace=0.31, wspace=0.28)
    ax_map = fig.add_subplot(grid[:, :2])
    ax_hist = fig.add_subplot(grid[0, 2])
    ax_quality = fig.add_subplot(grid[1, 2])

    plot_height_map(ax_map, frame, norm, cmap)
    ax_map.set_aspect("equal")
    ax_map.set_axis_off()
    ax_map.set_title("Multi-PS triangle-adjusted building height", fontsize=9)
    add_scalebar(ax_map)
    add_north_arrow(ax_map)
    panel_label(ax_map, "a")
    color_ax = inset_axes(ax_map, width="42%", height="2.8%", loc="lower right", borderpad=1.4)
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_ax, orientation="horizontal")
    colorbar.set_label("Estimated building height (m)", fontsize=6)
    colorbar.ax.tick_params(labelsize=5, length=2)
    ax_map.legend(
        handles=[
            Patch(facecolor=QUALITY_COLORS["no_ps_equation"], edgecolor="none", label="No valid PS equation"),
            Patch(facecolor="none", edgecolor=COLORS["dark"], label="High / medium outline"),
        ],
        loc="upper left",
        fontsize=5.4,
    )

    ax_hist.hist(finite.height_est_m, bins=np.arange(0, 151, 5), color=COLORS["blue"], edgecolor="white", linewidth=0.3)
    median = finite.height_est_m.median()
    ax_hist.axvline(median, color=COLORS["red"], lw=1.0, ls="--")
    ax_hist.text(0.96, 0.94, f"n = {len(finite)}\nmedian = {median:.1f} m", transform=ax_hist.transAxes, ha="right", va="top", fontsize=6)
    ax_hist.set_xlabel("Adjusted height (m)")
    ax_hist.set_ylabel("Buildings")
    ax_hist.set_title("Height distribution")
    panel_label(ax_hist, "b")

    order = ["high", "medium", "low", "no_ps_equation"]
    counts = estimates.quality.value_counts().reindex(order, fill_value=0)
    ax_quality.barh(np.arange(4), counts, color=[QUALITY_COLORS[value] for value in order], height=0.65)
    ax_quality.set_yticks(np.arange(4), ["High", "Medium", "Low", "No valid eq."])
    ax_quality.tick_params(axis="y", labelsize=5.8, pad=2)
    ax_quality.invert_yaxis()
    ax_quality.set_xlabel("Buildings")
    ax_quality.set_title("Solution quality")
    for y, value in enumerate(counts):
        ax_quality.text(value + 7, y, str(int(value)), va="center", fontsize=6)
    ax_quality.set_xlim(0, counts.max() * 1.20)
    panel_label(ax_quality, "c")

    fig.suptitle("Building-height estimates from registered roof and wall PS", fontsize=9, y=0.985)
    fig.text(
        0.5,
        0.012,
        "PS height is currently DSM-derived; quality is internal and is not independent accuracy validation.",
        ha="center",
        fontsize=6,
        color=COLORS["mid"],
    )
    save_figure(fig, "fig_24_triangle_adjusted_building_height_map", tiff=True)


def figure_25_annotated(buildings: gpd.GeoDataFrame) -> None:
    frame = buildings.to_crs("EPSG:32651")
    boundaries = np.array([3, 10, 20, 30, 45, 65, 90, 150], dtype=float)
    palette = [
        "#2C6BAA",
        "#3E91B8",
        "#54B6AE",
        "#89C98F",
        "#D5D56A",
        "#ECA75A",
        "#D76355",
    ]
    cmap = ListedColormap(palette, name="height_classes")
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    fig, ax = plt.subplots(figsize=(16, 14))
    no_solution = frame.loc[~np.isfinite(frame.height_est_m)]
    finite = frame.loc[np.isfinite(frame.height_est_m)]
    if len(no_solution):
        no_solution.plot(
            ax=ax,
            facecolor="#F0F1F2",
            edgecolor="white",
            linewidth=0.12,
            rasterized=True,
        )
    finite.plot(
        ax=ax,
        column="height_est_m",
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.12,
        rasterized=True,
    )
    for row in frame.itertuples():
        point = row.geometry.representative_point()
        label = "N/A" if not np.isfinite(row.height_est_m) else f"{row.height_est_m:.0f}"
        if label == "N/A":
            color = "#9A9A9A"
            weight = "normal"
        else:
            rgba = cmap(norm(float(row.height_est_m)))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "#202020" if luminance > 0.60 else "white"
            weight = "bold"
        ax.text(
            point.x,
            point.y,
            label,
            ha="center",
            va="center",
            fontsize=2.35,
            color=color,
            fontweight=weight,
            clip_on=True,
        )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Triangle-adjusted building height", fontsize=14, pad=8)
    add_scalebar(ax)
    add_north_arrow(ax)
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        orientation="horizontal",
        fraction=0.026,
        pad=0.010,
        aspect=48,
        boundaries=boundaries,
        ticks=boundaries,
        spacing="proportional",
    )
    colorbar.set_label("Estimated building height (m)", fontsize=10)
    colorbar.ax.tick_params(labelsize=8, length=3)
    ax.legend(
        handles=[Patch(facecolor="#F0F1F2", edgecolor="#D5D5D5", label="No valid PS equation")],
        loc="upper left",
        fontsize=7,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#DDDDDD",
    )
    save_figure(fig, "fig_25_triangle_adjusted_building_height_map_annotated", dpi=400)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate formal triangle-adjustment figures.")
    parser.add_argument("--figure", type=int, choices=[22, 23, 24, 25], help="Generate only one figure.")
    args = parser.parse_args()
    apply_style()
    estimates = pd.read_csv(ADJUSTMENT / "building_heights_triangle_adjustment.csv")
    equations = pd.read_csv(ADJUSTMENT / "ps_triangle_height_equations.csv")
    triangles = gpd.read_file(TRIANGLES / "building_surface_triangles_radar.gpkg")
    buildings = gpd.read_file(
        ADJUSTMENT / "building_heights_triangle_adjustment.gpkg",
        layer="triangle_adjustment_heights",
    )
    if args.figure in (None, 22):
        amplitude = read_rslc_amplitude(
            PROJECT / "data" / "RE_SLAVES" / "20200708.rslc",
            PROJECT / "data" / "RE_SLAVES" / "20200708.rslc.par",
        )
        figure_22_adjustment_geometry(amplitude, triangles, equations, estimates)
    if args.figure in (None, 23):
        figure_23_adjustment_diagnostics(estimates, equations)
    if args.figure in (None, 24):
        figure_24_final_map(buildings, estimates)
    if args.figure in (None, 25):
        figure_25_annotated(buildings)
    generated = [args.figure] if args.figure is not None else list(range(22, 26))
    print(
        json.dumps(
            {
                "output_dir": str(PICALL),
                "formal_triangle_adjustment_figures": generated,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
