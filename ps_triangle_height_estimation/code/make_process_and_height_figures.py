from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ps-triangle-figures")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from geometry import parse_gamma_par
from optimize_svg import optimize_svg_file


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
PIC_ALL = ROOT / "results" / "picall"
ESTIMATION = ROOT / "results" / "height_estimation"
TABLES = ROOT / "results" / "tables"
VECTORS = ROOT / "results" / "vectors"
TRIANGLES = ROOT / "results" / "triangles"

COLORS = {
    "blue": "#0F4D92",
    "blue2": "#3775BA",
    "teal": "#42949E",
    "violet": "#7C6CCF",
    "red": "#B64342",
    "gold": "#E8A317",
    "wall": "#D948A1",
    "roof": "#E8A317",
    "ps": "#13B9C8",
    "dark": "#272727",
    "mid": "#767676",
    "light": "#D8D8D8",
    "pale": "#F1F3F5",
}
QUALITY_COLORS = {"high": COLORS["blue"], "medium": COLORS["teal"], "low": "#A8A8A8", "no_ps": "#E1E1E1"}


def apply_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False


def panel_label(ax, label: str, color: str = "black", inside: bool = False) -> None:
    ax.text(
        0.015 if inside else -0.08,
        0.985 if inside else 1.02,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top" if inside else "bottom",
        fontsize=9,
        fontweight="bold",
        color=color,
        zorder=20,
    )


def save_figure(fig, stem: str, tiff: bool = False, png_dpi: int = 300) -> None:
    PIC_ALL.mkdir(parents=True, exist_ok=True)
    # Keep text and small annotations as vectors, but render explicitly
    # rasterized map layers at publication resolution.  This avoids SVGs with
    # thousands of polygon paths, which are slow to open in browsers/editors.
    svg_path = PIC_ALL / f"{stem}.svg"
    fig.savefig(svg_path, dpi=200, bbox_inches="tight", facecolor="white")
    optimize_svg_file(svg_path)
    fig.savefig(PIC_ALL / f"{stem}.png", dpi=png_dpi, bbox_inches="tight", facecolor="white")
    if tiff:
        fig.savefig(PIC_ALL / f"{stem}.tiff", dpi=600, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
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
    segments = []
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
    ax.plot([x0, x0 + length_m], [y0, y0], color=COLORS["dark"], lw=2.0, solid_capstyle="butt")
    ax.text(x0 + length_m / 2, y0 + 0.018 * (ymax - ymin), f"{int(length_m)} m", ha="center", va="bottom", fontsize=6)


def add_north_arrow(ax) -> None:
    ax.annotate(
        "N",
        xy=(0.94, 0.92),
        xytext=(0.94, 0.82),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": COLORS["dark"], "lw": 1.1},
    )


def figure_16_process(amplitude: np.ndarray, triangles: gpd.GeoDataFrame, mapped: pd.DataFrame, buildings: gpd.GeoDataFrame) -> None:
    representative_fid = 177
    display = display_stretch(amplitude)
    ps_all = pd.read_csv(PROJECT / "data" / "ps_points_all.csv")
    ps_all = ps_all.loc[ps_all.coherence >= 0.55].copy()
    ps_all["row0"] = ps_all.azimuth_pixel - 1.0
    ps_all["col0"] = ps_all.range_pixel - 1.0

    fig = plt.figure(figsize=(7.2, 5.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], width_ratios=[1.15, 1.0, 1.0], hspace=0.48, wspace=0.25)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[:, 1:])
    ax_c = fig.add_subplot(grid[1, 0])

    ax_a.imshow(display, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    roof = triangles.loc[triangles.surface == "roof"]
    ax_a.add_collection(
        LineCollection(
            polygon_segments(roof),
            colors=COLORS["gold"],
            linewidths=0.12,
            alpha=0.28,
            rasterized=True,
        )
    )
    ax_a.scatter(ps_all.col0, ps_all.row0, s=0.08, c=COLORS["ps"], alpha=0.25, linewidths=0, rasterized=True)
    ax_a.set_xlim(0, amplitude.shape[1])
    ax_a.set_ylim(amplitude.shape[0], 0)
    ax_a.set_xlabel("Range column")
    ax_a.set_ylabel("Azimuth row")
    ax_a.set_title("Scene-wide roof projections + PS", fontsize=7.5)
    panel_label(ax_a, "a", color="white", inside=True)

    local_triangles = triangles.loc[triangles.building_fid == representative_fid]
    bounds = local_triangles.total_bounds
    pad = 8.0
    ax_b.imshow(display, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for surface, color, width in [("wall", COLORS["wall"], 0.9), ("roof", COLORS["roof"], 1.1)]:
        selected = local_triangles.loc[local_triangles.surface == surface]
        ax_b.add_collection(LineCollection(polygon_segments(selected), colors=color, linewidths=width, alpha=0.9))
    local_ps = mapped.loc[mapped.fid == representative_fid]
    ax_b.scatter(local_ps.col0, local_ps.row0, s=16, c=COLORS["ps"], edgecolors="white", linewidths=0.35, zorder=5)
    ax_b.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax_b.set_ylim(bounds[3] + pad, bounds[1] - pad)
    ax_b.set_xlabel("Range column")
    ax_b.set_ylabel("Azimuth row")
    ax_b.set_title("Local triangle-to-PS assignment", fontsize=8)
    ax_b.legend(
        handles=[Patch(facecolor="none", edgecolor=COLORS["wall"], label="Wall triangles"), Patch(facecolor="none", edgecolor=COLORS["roof"], label="Roof triangles"), mpl.lines.Line2D([], [], marker="o", color="none", markerfacecolor=COLORS["ps"], markeredgecolor="white", label="Mapped PS")],
        loc="upper right",
        fontsize=6,
    )
    panel_label(ax_b, "b", color="white", inside=True)

    footprint = buildings.loc[buildings.building_fid == representative_fid].to_crs("EPSG:32651")
    ps_geo = gpd.GeoDataFrame(local_ps.copy(), geometry=gpd.points_from_xy(local_ps.surface_lon, local_ps.surface_lat), crs="EPSG:4326").to_crs("EPSG:32651")
    x_origin = 10.0 * np.floor(min(footprint.total_bounds[0], ps_geo.total_bounds[0]) / 10.0)
    y_origin = 10.0 * np.floor(min(footprint.total_bounds[1], ps_geo.total_bounds[1]) / 10.0)
    footprint_local = footprint.copy()
    footprint_local.geometry = footprint_local.geometry.translate(xoff=-x_origin, yoff=-y_origin)
    footprint_local.plot(ax=ax_c, facecolor="#ECEFF4", edgecolor=COLORS["dark"], linewidth=1.0)
    scatter = ax_c.scatter(ps_geo.geometry.x - x_origin, ps_geo.geometry.y - y_origin, c=ps_geo.surface_elevation_m, cmap="cividis", s=25, edgecolors="white", linewidths=0.35)
    colorbar = fig.colorbar(scatter, ax=ax_c, fraction=0.035, pad=0.02)
    colorbar.set_label("Mapped surface elevation (m)", fontsize=6)
    ax_c.set_aspect("equal")
    ax_c.set_xlabel(f"Easting offset from {x_origin:.0f} m")
    ax_c.set_ylabel(f"Northing offset from {y_origin:.0f} m")
    ax_c.set_title("PS mapped to 3-D building surfaces", fontsize=8)
    ax_c.ticklabel_format(style="plain", useOffset=False)
    panel_label(ax_c, "c", inside=True)
    fig.suptitle("PS-constrained triangle projection and 3-D surface mapping", fontsize=9, y=0.995)
    save_figure(fig, "fig_16_ps_triangle_projection_and_mapping")


def figure_17_curves(estimates: pd.DataFrame, curves: pd.DataFrame) -> None:
    examples = [(177, "High identifiability"), (978, "Medium identifiability"), (97, "Low / boundary solution")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), sharey=True)
    for ax, (fid, title) in zip(axes, examples, strict=True):
        curve = curves.loc[curves.fid == fid].sort_values("height_m")
        result = estimates.loc[estimates.fid == fid].iloc[0]
        ax.plot(curve.height_m, curve.score, color=COLORS["blue"], lw=1.6, label="Combined PS score")
        ax.plot(curve.height_m, 0.45 * curve.roof_density_z, color=COLORS["teal"], lw=0.9, alpha=0.85, label="Roof density")
        ax.plot(curve.height_m, 0.35 * curve.roof_edge_z, color=COLORS["gold"], lw=0.9, alpha=0.85, label="Roof edge")
        ax.plot(curve.height_m, 0.20 * curve.wall_density_z, color=COLORS["wall"], lw=0.9, alpha=0.85, label="Wall density")
        ax.axvline(result.height_est_m, color=COLORS["red"], lw=1.1, ls="--")
        if np.isfinite(result.height_uncertainty_m):
            ax.axvspan(result.height_est_m - result.height_uncertainty_m, result.height_est_m + result.height_uncertainty_m, color=COLORS["red"], alpha=0.08, lw=0)
        ax.text(0.97, 0.95, f"h = {result.height_est_m:.1f} m\nPS = {int(result.candidate_ps)}\nquality: {result.quality}", transform=ax.transAxes, ha="right", va="top", fontsize=6)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel("Candidate building height (m)")
        ax.axhline(0, color="#CCCCCC", lw=0.6)
    axes[0].set_ylabel("Standardized PS geometry score")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    panel_label(axes[2], "c")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=4, fontsize=6)
    fig.suptitle("Candidate-height PS score curves reveal sharp and ambiguous solutions", fontsize=9, y=1.12)
    fig.tight_layout()
    save_figure(fig, "fig_17_ps_height_candidate_score_curves")


def figure_18_diagnostics(estimates: pd.DataFrame) -> None:
    finite = estimates.loc[np.isfinite(estimates.height_est_m)].copy()
    order = ["high", "medium", "low", "no_ps"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    counts = estimates.quality.value_counts().reindex(order, fill_value=0)
    bars = ax_a.bar(np.arange(len(order)), counts.values, color=[QUALITY_COLORS[value] for value in order], width=0.7)
    ax_a.set_xticks(np.arange(len(order)), ["High", "Medium", "Low", "No PS"])
    ax_a.set_ylabel("Buildings")
    ax_a.set_title("Internal quality classes")
    for bar, value in zip(bars, counts.values, strict=True):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8, str(int(value)), ha="center", va="bottom", fontsize=6)

    for quality in ["high", "medium", "low"]:
        values = finite.loc[finite.quality == quality, "height_uncertainty_m"]
        if len(values):
            ax_b.hist(values, bins=np.arange(0, 31, 1.5), histtype="step", lw=1.4, color=QUALITY_COLORS[quality], label=f"{quality} (n={len(values)})")
    ax_b.axvline(finite.height_uncertainty_m.median(), color=COLORS["dark"], ls="--", lw=1.0, label=f"median = {finite.height_uncertainty_m.median():.1f} m")
    ax_b.set_xlim(0, min(30, max(10, finite.height_uncertainty_m.quantile(0.98))))
    ax_b.set_xlabel("Peak-width uncertainty (m)")
    ax_b.set_ylabel("Buildings")
    ax_b.set_title("Height-peak uncertainty")
    ax_b.legend(fontsize=5.5)

    for quality in ["low", "medium", "high"]:
        selected = finite.loc[finite.quality == quality]
        ax_c.scatter(selected.candidate_ps, selected.height_uncertainty_m, s=8 if quality == "low" else 13, c=QUALITY_COLORS[quality], alpha=0.55 if quality == "low" else 0.8, linewidths=0, label=quality)
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Candidate-corridor PS count")
    ax_c.set_ylabel("Peak-width uncertainty (m)")
    ax_c.set_title("More PS does not always resolve ambiguity")
    ax_c.legend(fontsize=5.5)

    ax_d.scatter(finite.height_prior_m, finite.height_est_m, c=[QUALITY_COLORS[q] for q in finite.quality], s=8, alpha=0.55, linewidths=0)
    limit = max(float(finite.height_prior_m.max()), float(finite.height_est_m.max()))
    ax_d.plot([0, limit], [0, limit], color=COLORS["mid"], lw=0.9, ls="--")
    ax_d.set_xlim(0, limit * 1.02)
    ax_d.set_ylim(0, limit * 1.02)
    ax_d.set_xlabel("Input SHP height (m; search bounds only)")
    ax_d.set_ylabel("PS-estimated height (m)")
    ax_d.set_title("Sensitivity comparison — not validation")
    ax_d.text(0.03, 0.96, f"median |difference| = {np.median(np.abs(finite.height_est_m - finite.height_prior_m)):.1f} m", transform=ax_d.transAxes, ha="left", va="top", fontsize=6)

    for label, ax in zip("abcd", axes.ravel(), strict=True):
        panel_label(ax, label)
    fig.suptitle("Internal diagnostics of PS-only building-height inversion", fontsize=9, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, "fig_18_ps_height_estimation_diagnostics")


def plot_height_map(ax, frame: gpd.GeoDataFrame, norm: Normalize, cmap) -> None:
    no_ps = frame.loc[~np.isfinite(frame.height_est_m)]
    finite = frame.loc[np.isfinite(frame.height_est_m)]
    if len(no_ps):
        no_ps.plot(
            ax=ax,
            facecolor=QUALITY_COLORS["no_ps"],
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
            alpha=0.7,
            rasterized=True,
        )


def figure_19_final_map(buildings: gpd.GeoDataFrame, estimates: pd.DataFrame) -> None:
    frame = buildings.to_crs("EPSG:32651").copy()
    cmap = mpl.colormaps["cividis"]
    norm = Normalize(vmin=3, vmax=130)
    finite = estimates.loc[np.isfinite(estimates.height_est_m)].copy()

    fig = plt.figure(figsize=(7.2, 5.8))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.35, 0.9], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.30)
    ax_map = fig.add_subplot(grid[:, :2])
    ax_hist = fig.add_subplot(grid[0, 2])
    ax_quality = fig.add_subplot(grid[1, 2])

    plot_height_map(ax_map, frame, norm, cmap)
    ax_map.set_aspect("equal")
    ax_map.set_axis_off()
    ax_map.set_title("PS-estimated building height", fontsize=9, pad=4)
    add_scalebar(ax_map, 500)
    add_north_arrow(ax_map)
    panel_label(ax_map, "a")
    color_ax = inset_axes(ax_map, width="42%", height="2.8%", loc="lower right", borderpad=1.4)
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_ax, orientation="horizontal")
    colorbar.set_label("Estimated height (m)", fontsize=6)
    colorbar.ax.tick_params(labelsize=5, length=2)
    ax_map.legend(
        handles=[Patch(facecolor=QUALITY_COLORS["no_ps"], edgecolor="none", label="No PS solution"), Patch(facecolor="none", edgecolor=COLORS["dark"], label="High / medium outline")],
        loc="upper left",
        fontsize=5.5,
    )

    ax_hist.hist(finite.height_est_m, bins=np.arange(0, 136, 5), color=COLORS["blue2"], edgecolor="white", linewidth=0.35)
    ax_hist.axvline(finite.height_est_m.median(), color=COLORS["red"], lw=1.1, ls="--")
    ax_hist.text(0.97, 0.93, f"n = {len(finite)}\nmedian = {finite.height_est_m.median():.1f} m", transform=ax_hist.transAxes, ha="right", va="top", fontsize=6)
    ax_hist.set_xlabel("Estimated height (m)")
    ax_hist.set_ylabel("Buildings")
    ax_hist.set_title("Height distribution", fontsize=7.5)
    panel_label(ax_hist, "b")

    order = ["high", "medium", "low", "no_ps"]
    counts = estimates.quality.value_counts().reindex(order, fill_value=0)
    ax_quality.barh(np.arange(len(order)), counts.values, color=[QUALITY_COLORS[q] for q in order], height=0.65)
    ax_quality.set_yticks(np.arange(len(order)), ["High", "Medium", "Low", "No PS"])
    ax_quality.invert_yaxis()
    ax_quality.set_xlabel("Buildings")
    ax_quality.set_title("Solution quality", fontsize=7.5)
    for y, value in enumerate(counts.values):
        ax_quality.text(value + 7, y, str(int(value)), va="center", fontsize=6)
    ax_quality.set_xlim(0, counts.max() * 1.18)
    panel_label(ax_quality, "c")
    fig.suptitle("Building-height estimates from strict triangle geometry and coherent PS", fontsize=9, y=0.985)
    fig.text(0.51, 0.012, "Quality denotes internal peak identifiability; no independent height truth is used.", ha="center", fontsize=6, color=COLORS["mid"])
    save_figure(fig, "fig_19_tongji_ps_building_height_estimates", tiff=True)


def figure_20_annotated(buildings: gpd.GeoDataFrame) -> None:
    frame = buildings.to_crs("EPSG:32651").copy()
    cmap = mpl.colormaps["cividis"]
    norm = Normalize(vmin=3, vmax=130)
    fig, ax = plt.subplots(figsize=(16, 14))
    plot_height_map(ax, frame, norm, cmap)
    for row in frame.itertuples():
        point = row.geometry.representative_point()
        label = "N/A" if not np.isfinite(row.height_est_m) else f"{row.height_est_m:.0f}"
        if label == "N/A":
            color = "#777777"
        else:
            rgba = cmap(norm(float(row.height_est_m)))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "#202020" if luminance > 0.60 else "white"
        ax.text(
            point.x,
            point.y,
            label,
            ha="center",
            va="center",
            fontsize=2.25,
            color=color,
            fontweight="bold",
            clip_on=True,
        )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("PS-estimated building height (m); N/A = no PS solution", fontsize=14, pad=10)
    add_scalebar(ax, 500)
    add_north_arrow(ax)
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.026, pad=0.015)
    colorbar.set_label("Estimated height (m)", fontsize=10)
    save_figure(fig, "fig_20_tongji_ps_building_height_estimates_annotated", png_dpi=400)


def figure_21_projection(amplitude: np.ndarray, triangles: gpd.GeoDataFrame, mapped: pd.DataFrame) -> None:
    """Dedicated strict bottom-wall-roof projection figure on the real SAR grid."""
    display = np.clip(display_stretch(amplitude) * 0.72, 0.0, 1.0)
    representative_fid = 177
    local = triangles.loc[triangles.building_fid == representative_fid]
    local_mapped = mapped.loc[mapped.fid == representative_fid]
    bounds = local.total_bounds

    fig = plt.figure(figsize=(7.2, 3.55), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.10)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    surface_styles = {
        "bottom": ("#22D7E6", 0.15, 0.28),
        "wall": ("#FF2AD4", 0.13, 0.26),
        "roof": ("#E8A317", 0.18, 0.42),
    }

    ax = axes[0]
    ax.imshow(display, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    envelopes = triangles[["building_fid", "geometry"]].dissolve(by="building_fid")
    ax.add_collection(
        LineCollection(
            polygon_segments(envelopes),
            colors="#F2B134",
            linewidths=0.28,
            alpha=0.60,
            rasterized=True,
        )
    )
    visible_mapped = mapped.loc[
        mapped.col0.between(0, amplitude.shape[1] - 1)
        & mapped.row0.between(0, amplitude.shape[0] - 1)
    ]
    ax.scatter(visible_mapped.col0, visible_mapped.row0, s=0.16, c="#25D7E5", alpha=0.40, linewidths=0, rasterized=True)
    pad = 8.0
    roi = Rectangle(
        (bounds[0] - pad, bounds[1] - pad),
        (bounds[2] - bounds[0]) + 2 * pad,
        (bounds[3] - bounds[1]) + 2 * pad,
        fill=False,
        edgecolor="#E44B4B",
        linewidth=1.0,
        zorder=8,
    )
    ax.add_patch(roi)
    ax.text(bounds[0] - pad + 2, bounds[1] - pad - 3, "zoom b", color="#E44B4B", fontsize=5.5, fontweight="bold")
    ax.set_xlim(0, amplitude.shape[1])
    ax.set_ylim(amplitude.shape[0], 0)
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    ax.set_title("Building projection envelopes and mapped PS", fontsize=8)
    ax.text(
        0.02,
        0.03,
        f"1,028 buildings\n{len(triangles):,} surface triangles\n{len(visible_mapped):,} mapped PS",
        transform=ax.transAxes,
        color="white",
        fontsize=5.8,
        va="bottom",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 2.5},
    )
    panel_label(ax, "a", color="white", inside=True)

    ax = axes[1]
    ax.imshow(display, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for surface, (color, _, _) in surface_styles.items():
        selected = local.loc[local.surface == surface]
        ax.add_collection(LineCollection(polygon_segments(selected), colors=color, linewidths=1.15 if surface == "roof" else 0.85, alpha=0.95))
    ax.scatter(local_mapped.col0, local_mapped.row0, s=22, c="#25D7E5", edgecolors="white", linewidths=0.45, zorder=6)
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[3] + pad, bounds[1] - pad)
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    ax.set_title("Target building: bottom-wall-roof triangles", fontsize=8)
    ax.legend(
        handles=[
            Patch(facecolor="none", edgecolor=surface_styles["bottom"][0], label="Bottom"),
            Patch(facecolor="none", edgecolor=surface_styles["wall"][0], label="Wall"),
            Patch(facecolor="none", edgecolor=surface_styles["roof"][0], label="Roof"),
            mpl.lines.Line2D([], [], marker="o", color="none", markerfacecolor="#25D7E5", markeredgecolor="white", label="Mapped PS"),
        ],
        loc="upper right",
        fontsize=5.8,
        ncol=2,
    )
    panel_label(ax, "b", color="white", inside=True)
    fig.suptitle("Strict building-surface projection on the SAR image", fontsize=9)
    save_figure(fig, "fig_21_strict_building_triangle_projection_on_sar")


def main() -> None:
    apply_style()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    estimates = pd.read_csv(ESTIMATION / "building_heights_from_ps.csv")
    curves = pd.read_csv(ESTIMATION / "building_height_ps_score_curves.csv")
    mapped = pd.read_csv(TABLES / "ps_building_surface_coordinates.csv")
    triangles = gpd.read_file(TRIANGLES / "building_surface_triangles_radar.gpkg")
    buildings = gpd.read_file(ESTIMATION / "building_heights_from_ps.gpkg")
    amplitude = read_rslc_amplitude(PROJECT / "data" / "RE_SLAVES" / "20200708.rslc", PROJECT / "data" / "RE_SLAVES" / "20200708.rslc.par")

    figure_16_process(amplitude, triangles, mapped, buildings)
    figure_17_curves(estimates, curves)
    figure_18_diagnostics(estimates)
    figure_19_final_map(buildings, estimates)
    figure_20_annotated(buildings)
    figure_21_projection(amplitude, triangles, mapped)
    print(json.dumps({"output_dir": str(PIC_ALL), "figures": list(range(16, 22))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
