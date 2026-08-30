from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-highrise-height-map")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = (
    ROOT
    / "results"
    / "picall"
    / "touying2_ps_coordinates_current"
    / "highrise_optimized"
)
STEM = "11_highrise_optimized_building_height_map"

COLORS = {
    "rejected": "#ECEFF2",
    "rejected_edge": "#CDD2D8",
    "accepted_edge": "#3E4650",
    "enhanced": "#E47720",
    "dark": "#252B32",
}

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
    y0 = ymin + 0.040 * (ymax - ymin)
    ax.plot([x0, x0 + length_m], [y0, y0], color=COLORS["dark"], lw=2.0, zorder=9)
    ax.plot([x0, x0], [y0 - 6, y0 + 6], color=COLORS["dark"], lw=0.8, zorder=9)
    ax.plot(
        [x0 + length_m, x0 + length_m],
        [y0 - 6, y0 + 6],
        color=COLORS["dark"],
        lw=0.8,
        zorder=9,
    )
    ax.text(
        x0 + 0.5 * length_m,
        y0 + 11,
        f"{length_m:g} m",
        ha="center",
        va="bottom",
        fontsize=6.3,
        color=COLORS["dark"],
        zorder=9,
    )


def add_north_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "N",
        xy=(0.965, 0.955),
        xytext=(0.965, 0.890),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=COLORS["dark"],
        arrowprops={
            "arrowstyle": "-|>",
            "color": COLORS["dark"],
            "lw": 1.1,
            "mutation_scale": 10,
        },
        zorder=9,
    )


def run(results_root: Path) -> dict:
    vector_path = (
        results_root
        / "vectors"
        / "building_height_estimates_highrise_optimized.gpkg"
    )
    buildings = gpd.read_file(
        vector_path, layer="highrise_optimized_heights"
    ).to_crs("EPSG:32651")
    required = {
        "height_optimized_m",
        "highrise_optimization_applied",
        "geometry",
    }
    missing = sorted(required.difference(buildings.columns))
    if missing:
        raise ValueError("Optimized vector is missing: " + ", ".join(missing))
    if not buildings.geometry.is_valid.all():
        raise ValueError("Optimized building vector contains invalid geometries")

    accepted = buildings.loc[buildings.height_optimized_m.notna()].copy()
    rejected = buildings.loc[buildings.height_optimized_m.isna()].copy()
    enhanced = accepted.loc[
        accepted.highrise_optimization_applied.fillna(0).astype(int).eq(1)
    ].copy()

    cmap = mpl.colormaps["cividis"]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=80.0, clip=True)
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    rejected.plot(
        ax=ax,
        facecolor=COLORS["rejected"],
        edgecolor=COLORS["rejected_edge"],
        linewidth=0.20,
        zorder=1,
    )
    accepted.plot(
        ax=ax,
        column="height_optimized_m",
        cmap=cmap,
        norm=norm,
        edgecolor=COLORS["accepted_edge"],
        linewidth=0.24,
        zorder=2,
    )
    enhanced.plot(
        ax=ax,
        facecolor="none",
        edgecolor=COLORS["enhanced"],
        linewidth=0.82,
        zorder=4,
    )

    for row in accepted.itertuples():
        area = float(row.geometry.area)
        is_enhanced = bool(row.highrise_optimization_applied)
        if not is_enhanced and area < 105.0:
            continue
        point = row.geometry.representative_point()
        height = float(row.height_optimized_m)
        rgba = cmap(norm(height))
        luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        ax.text(
            point.x,
            point.y,
            f"{height:.0f}",
            ha="center",
            va="center",
            fontsize=2.35 if is_enhanced else 1.75,
            fontweight="bold" if is_enhanced else "normal",
            color=(
                COLORS["enhanced"]
                if is_enhanced
                else ("#202020" if luminance > 0.58 else "white")
            ),
            zorder=6,
            clip_on=True,
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
    colorbar.set_label("Optimized building height above the 4 m ground base (m)")
    colorbar.ax.tick_params(length=2.2)

    ax.legend(
        handles=[
            Patch(
                facecolor="none",
                edgecolor=COLORS["enhanced"],
                linewidth=1.2,
                label=f"High-rise top restoration applied (n={len(enhanced)})",
            ),
            Patch(
                facecolor=COLORS["rejected"],
                edgecolor=COLORS["rejected_edge"],
                label=f"Rejected or unsupported (n={len(rejected)})",
            ),
        ],
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#D1D6DB",
        framealpha=0.94,
        fontsize=6.3,
    )
    ax.set_title(
        "High-rise-enhanced PS-InSAR building-height estimates",
        fontsize=9,
        pad=8,
    )
    ax.set_xlabel("Easting (m, UTM zone 51N)")
    ax.set_ylabel("Northing (m, UTM zone 51N)")
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax.text(
        0.01,
        0.012,
        (
            f"Accepted estimates: {len(accepted)}/{len(buildings)}; "
            f"median={accepted.height_optimized_m.median():.1f} m; "
            "orange outlines and labels identify high-rise enhancement"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=COLORS["dark"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.2},
        zorder=8,
    )
    add_scale_bar(ax)
    add_north_arrow(ax)
    fig.tight_layout()

    png_path = results_root / "png" / f"{STEM}.png"
    svg_path = results_root / "svg" / f"{STEM}.svg"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, dpi=96, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "figure": STEM,
        "core_conclusion": (
            f"The final plan-view map preserves {len(accepted)} accepted estimates "
            f"while making the {len(enhanced)} high-rise top-restored updates "
            "spatially explicit."
        ),
        "crs": "EPSG:32651",
        "height_field": "height_optimized_m",
        "total_buildings": int(len(buildings)),
        "accepted_buildings": int(len(accepted)),
        "highrise_enhanced_buildings": int(len(enhanced)),
        "rejected_or_unsupported_buildings": int(len(rejected)),
        "height_m": {
            "minimum": float(accepted.height_optimized_m.min()),
            "median": float(accepted.height_optimized_m.median()),
            "mean": float(accepted.height_optimized_m.mean()),
            "maximum": float(accepted.height_optimized_m.max()),
        },
        "color_scale_m": [0.0, 80.0],
        "labels": (
            "All high-rise-enhanced buildings plus accepted buildings with "
            "footprint area >= 105 square metres"
        ),
        "outputs": {"png": str(png_path), "svg": str(svg_path)},
        "pdf_generated": False,
    }
    summary_path = results_root / "tables" / f"{STEM}_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the final high-rise-enhanced building-height map."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    run(args.results_root.resolve())


if __name__ == "__main__":
    main()
