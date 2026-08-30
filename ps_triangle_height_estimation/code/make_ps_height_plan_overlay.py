from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ps-height-plan")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILDINGS = (
    ROOT.parents[1]
    / "geocoding"
    / "data"
    / "shp"
    / "tongji_clip_rslc_extent_equal_height_clean.shp"
)
DEFAULT_PS = ROOT.parent / "ps_coordinates_current" / "ps_positive_coordinates.gpkg"
DEFAULT_MAPPED = (
    ROOT
    / "results"
    / "picall"
    / "touying2_ps_coordinates_current"
    / "mapping"
    / "vectors"
    / "ps_points_on_building_surfaces.gpkg"
)
DEFAULT_ESTIMATES = (
    ROOT
    / "results"
    / "picall"
    / "touying2_ps_coordinates_current"
    / "tables"
    / "building_height_estimates.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "picall"
    / "touying2_ps_coordinates_current"
)
STEM = "08_ps_height_plan_overlay_buildings"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
    }
)


def add_scale_bar(ax: plt.Axes, length_m: float = 250.0) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + 0.055 * (xmax - xmin)
    y0 = ymin + 0.045 * (ymax - ymin)
    ax.plot([x0, x0 + length_m], [y0, y0], color="#252A31", lw=2.0, zorder=9)
    ax.plot([x0, x0], [y0 - 6, y0 + 6], color="#252A31", lw=0.8, zorder=9)
    ax.plot(
        [x0 + length_m, x0 + length_m],
        [y0 - 6, y0 + 6],
        color="#252A31",
        lw=0.8,
        zorder=9,
    )
    ax.text(
        x0 + 0.5 * length_m,
        y0 + 11,
        f"{length_m:g} m",
        ha="center",
        va="bottom",
        color="#252A31",
        fontsize=6.5,
        zorder=9,
    )


def add_north_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "N",
        xy=(0.965, 0.955),
        xytext=(0.965, 0.885),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#252A31",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#252A31",
            "lw": 1.1,
            "mutation_scale": 10,
        },
        zorder=9,
    )


def run(
    buildings_path: Path,
    ps_path: Path,
    mapped_path: Path,
    estimates_path: Path,
    output_root: Path,
) -> dict:
    buildings = gpd.read_file(buildings_path).to_crs("EPSG:32651").reset_index(drop=True)
    buildings["fid"] = np.arange(len(buildings), dtype=np.int64)
    ps = gpd.read_file(ps_path, layer="ps_positive_3d_wusong").to_crs("EPSG:32651")
    mapped = gpd.read_file(mapped_path)
    estimates = pd.read_csv(estimates_path)

    required_ps = {"ps_id", "h_agl_4m", "geometry"}
    missing = sorted(required_ps.difference(ps.columns))
    if missing:
        raise ValueError("PS input is missing columns: " + ", ".join(missing))
    if buildings.crs != ps.crs:
        raise ValueError(f"CRS mismatch: buildings={buildings.crs}, PS={ps.crs}")

    mapped_ids = set(mapped["ps_id"].astype(np.int64))
    ps["mapped_to_surface"] = ps["ps_id"].astype(np.int64).isin(mapped_ids)
    mapped_ps = ps.loc[ps.mapped_to_surface].copy()
    unmapped_ps = ps.loc[~ps.mapped_to_surface].copy()

    accepted_fids = set(
        estimates.loc[estimates["height_final_m"].notna(), "fid"].astype(np.int64)
    )
    buildings["final_height_accepted"] = buildings["fid"].isin(accepted_fids)
    accepted_buildings = buildings.loc[buildings.final_height_accepted]

    height = mapped_ps["h_agl_4m"].astype(float)
    vmax = float(np.ceil(np.quantile(height, 0.99) / 5.0) * 5.0)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax, clip=True)
    cmap = mpl.colormaps["cividis"]

    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    buildings.plot(
        ax=ax,
        facecolor="#F0F2F4",
        edgecolor="#C4CAD1",
        linewidth=0.28,
        zorder=1,
    )
    accepted_buildings.plot(
        ax=ax,
        facecolor="#E2E8ED",
        edgecolor="#596675",
        linewidth=0.48,
        zorder=2,
    )
    unmapped_ps.plot(
        ax=ax,
        color="#AEB5BD",
        markersize=1.1,
        alpha=0.34,
        linewidth=0,
        rasterized=True,
        zorder=3,
    )
    mapped_ps.plot(
        ax=ax,
        column="h_agl_4m",
        cmap=cmap,
        norm=norm,
        markersize=2.8,
        alpha=0.90,
        linewidth=0,
        rasterized=True,
        zorder=4,
    )

    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(
        scalar,
        ax=ax,
        orientation="vertical",
        fraction=0.036,
        pad=0.015,
        extend="max",
    )
    colorbar.set_label("PSI height above the 4 m model ground (m)")
    colorbar.ax.tick_params(length=2.2)

    ax.legend(
        handles=[
            Patch(
                facecolor="#E2E8ED",
                edgecolor="#596675",
                label=f"Buildings with accepted height (n={len(accepted_buildings)})",
            ),
            Patch(
                facecolor="#F0F2F4",
                edgecolor="#C4CAD1",
                label=f"Other building footprints (n={len(buildings) - len(accepted_buildings)})",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="#6A737D",
                markeredgecolor="none",
                markersize=3.2,
                label=f"PS not mapped to a building surface (n={len(unmapped_ps):,})",
            ),
        ],
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#D3D7DC",
        framealpha=0.94,
        fontsize=6.2,
    )
    ax.set_title("Plan view of PSI heights over building footprints", fontsize=9, pad=8)
    ax.set_xlabel("Easting (m, UTM zone 51N)")
    ax.set_ylabel("Northing (m, UTM zone 51N)")
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax.text(
        0.01,
        0.012,
        (
            f"Mapped PS colored by height: n={len(mapped_ps):,}; "
            f"display clipped above the 99th percentile ({vmax:g} m)"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.3,
        color="#252A31",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2.2},
        zorder=8,
    )
    add_scale_bar(ax)
    add_north_arrow(ax)
    fig.tight_layout()

    png_dir = output_root / "png"
    svg_dir = output_root / "svg"
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)
    png_path = png_dir / f"{STEM}.png"
    svg_path = svg_dir / f"{STEM}.svg"
    fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, dpi=96, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "figure": STEM,
        "core_conclusion": (
            "Mapped PSI height observations can be inspected in plan view against "
            "the building footprints used by the touying2 inversion."
        ),
        "crs": "EPSG:32651",
        "height_field": "h_agl_4m",
        "height_definition": (
            "psi_height_above_4m_ground_m = "
            "psi_scatterer_elevation_wusong_m - 4.000 m"
        ),
        "buildings": int(len(buildings)),
        "buildings_with_accepted_height": int(len(accepted_buildings)),
        "ps_total": int(len(ps)),
        "ps_mapped_to_building_surface": int(len(mapped_ps)),
        "ps_not_mapped_to_building_surface": int(len(unmapped_ps)),
        "display_height_max_m": vmax,
        "display_clipping": "99th percentile rounded up to 5 m",
        "outputs": {"png": str(png_path), "svg": str(svg_path)},
        "pdf_generated": False,
    }
    summary_path = output_root / "tables" / f"{STEM}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot current PSI heights in plan view over building footprints."
    )
    parser.add_argument("--buildings", type=Path, default=DEFAULT_BUILDINGS)
    parser.add_argument("--ps", type=Path, default=DEFAULT_PS)
    parser.add_argument("--mapped", type=Path, default=DEFAULT_MAPPED)
    parser.add_argument("--estimates", type=Path, default=DEFAULT_ESTIMATES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(
        args.buildings.resolve(),
        args.ps.resolve(),
        args.mapped.resolve(),
        args.estimates.resolve(),
        args.output_root.resolve(),
    )


if __name__ == "__main__":
    main()
