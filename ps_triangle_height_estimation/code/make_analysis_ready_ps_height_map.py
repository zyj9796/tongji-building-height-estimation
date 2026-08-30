from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-analysis-ready-ps-height-map")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "height_estimation_analysis_ready_ps"
DEFAULT_STEM = "analysis_ready_ps_building_height_map"

COLORS = {
    "dark": "#30343B",
    "mid": "#68717D",
    "low_quality": "#B8BEC6",
    "no_ps": "#EEF1F3",
    "outline": "#D2D7DC",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def add_scale_bar(ax: plt.Axes, length_m: float = 250.0) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + 0.72 * (xmax - xmin)
    y0 = ymin + 0.035 * (ymax - ymin)
    ax.plot(
        [x0, x0 + length_m],
        [y0, y0],
        color=COLORS["dark"],
        lw=1.8,
        solid_capstyle="butt",
        zorder=8,
    )
    ax.plot(
        [x0, x0],
        [y0 - 7, y0 + 7],
        color=COLORS["dark"],
        lw=0.8,
        zorder=8,
    )
    ax.plot(
        [x0 + length_m, x0 + length_m],
        [y0 - 7, y0 + 7],
        color=COLORS["dark"],
        lw=0.8,
        zorder=8,
    )
    ax.text(
        x0 + 0.5 * length_m,
        y0 + 13,
        f"{length_m:g} m",
        ha="center",
        va="bottom",
        fontsize=6,
        color=COLORS["dark"],
        zorder=8,
    )


def add_north_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "N",
        xy=(0.965, 0.955),
        xytext=(0.965, 0.895),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        color=COLORS["dark"],
        arrowprops={
            "arrowstyle": "-|>",
            "color": COLORS["dark"],
            "lw": 1.0,
            "mutation_scale": 9,
        },
        zorder=8,
    )


def build_figure(vector_path: Path) -> tuple[plt.Figure, dict]:
    vector = gpd.read_file(vector_path, layer="ps_height_estimates")
    if vector.crs is None:
        raise ValueError("Building-height vector has no CRS")
    required = {"height_recommended_m", "quality", "geometry"}
    missing = sorted(required.difference(vector.columns))
    if missing:
        raise ValueError("Missing required map columns: " + ", ".join(missing))

    vector = vector.to_crs("EPSG:32651")
    recommended = vector.loc[vector.height_recommended_m.notna()].copy()
    low_quality = vector.loc[vector.quality == "low"].copy()
    no_ps = vector.loc[vector.quality == "no_ps"].copy()

    bounds = [3, 10, 20, 30, 45, 65, 90, 120]
    base_cmap = mpl.colormaps["cividis"]
    colors = [base_cmap(value) for value in np.linspace(0.12, 0.92, len(bounds) - 1)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N, clip=True)

    fig, ax = plt.subplots(figsize=(6.3, 6.7))
    no_ps.plot(
        ax=ax,
        color=COLORS["no_ps"],
        edgecolor=COLORS["outline"],
        linewidth=0.18,
        zorder=1,
    )
    low_quality.plot(
        ax=ax,
        color=COLORS["low_quality"],
        edgecolor="white",
        linewidth=0.20,
        zorder=2,
    )
    recommended.plot(
        ax=ax,
        column="height_recommended_m",
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.28,
        zorder=3,
    )

    for row in recommended.itertuples():
        point = row.geometry.representative_point()
        height = float(row.height_recommended_m)
        rgba = cmap(norm(height))
        luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        ax.text(
            point.x,
            point.y,
            f"{height:.0f}",
            ha="center",
            va="center",
            fontsize=2.25,
            fontweight="bold",
            color="#202020" if luminance > 0.58 else "white",
            clip_on=True,
            zorder=5,
        )

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(
        "PS-InSAR building-height estimates retained after quality screening",
        fontsize=9,
        pad=8,
    )

    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cbar = fig.colorbar(
        scalar,
        ax=ax,
        orientation="horizontal",
        fraction=0.040,
        pad=0.018,
        ticks=bounds,
        spacing="proportional",
    )
    cbar.set_label("Recommended building height (m)")
    cbar.ax.tick_params(length=2)

    ax.legend(
        handles=[
            Patch(
                facecolor=COLORS["low_quality"],
                edgecolor="none",
                label=f"Low-quality raw solution (n={len(low_quality)})",
            ),
            Patch(
                facecolor=COLORS["no_ps"],
                edgecolor=COLORS["outline"],
                label=f"No PS support (n={len(no_ps)})",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.005, 0.995),
        fontsize=6.2,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.92,
        borderpad=0.35,
    )
    ax.text(
        0.01,
        0.012,
        f"Recommended estimates: n={len(recommended)} of {len(vector)} buildings; "
        "labels show rounded metres",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color=COLORS["dark"],
    )
    add_scale_bar(ax)
    add_north_arrow(ax)
    fig.tight_layout()

    summary = {
        "source_vector": str(vector_path),
        "crs": str(vector.crs),
        "total_buildings": int(len(vector)),
        "recommended_buildings": int(len(recommended)),
        "low_quality_buildings": int(len(low_quality)),
        "no_ps_buildings": int(len(no_ps)),
        "recommended_height_m": {
            "minimum": float(recommended.height_recommended_m.min()),
            "median": float(recommended.height_recommended_m.median()),
            "maximum": float(recommended.height_recommended_m.max()),
        },
        "height_field": "height_recommended_m",
        "low_quality_values_are_colored": False,
        "labels": "rounded recommended height in metres",
        "map_projection": "EPSG:32651",
    }
    return fig, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the screened building-height hero map from analysis-ready PS points."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    vector_path = results_dir / "building_heights_from_ps.gpkg"
    output_dir = results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, summary = build_figure(vector_path)
    outputs = {
        "svg": output_dir / f"{args.stem}.svg",
        "pdf": output_dir / f"{args.stem}.pdf",
        "png": output_dir / f"{args.stem}.png",
    }
    fig.savefig(outputs["svg"], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["pdf"], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["png"], dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary["outputs"] = {key: str(path) for key, path in outputs.items()}
    summary_path = output_dir / f"{args.stem}_qa.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
