from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-touying2-optimized-map")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "results"
    / "picall"
    / "touying2_ps_coordinates_current"
    / "highrise_optimized"
    / "vectors"
    / "building_height_estimates_highrise_optimized.gpkg"
)
DEFAULT_OUTPUT = (
    ROOT / "results" / "picall" / "touying2_ps_coordinates_current"
)
STEM = "09_highrise_optimized_building_height_estimation"

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


def run(source_vector: Path, output_root: Path) -> dict:
    buildings = gpd.read_file(
        source_vector, layer="highrise_optimized_heights"
    ).to_crs("EPSG:32651")
    required = {"height_optimized_m", "geometry"}
    missing = sorted(required.difference(buildings.columns))
    if missing:
        raise ValueError("Optimized building vector is missing: " + ", ".join(missing))

    accepted = buildings.loc[np.isfinite(buildings.height_optimized_m)].copy()
    rejected = buildings.loc[~np.isfinite(buildings.height_optimized_m)].copy()
    vmax = (
        max(40.0, float(accepted.height_optimized_m.quantile(0.98)))
        if len(accepted)
        else 40.0
    )
    norm = Normalize(0.0, vmax)

    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    rejected.plot(
        ax=ax,
        color="#D9DDE2",
        edgecolor="#BFC5CC",
        linewidth=0.18,
    )
    accepted.plot(
        ax=ax,
        column="height_optimized_m",
        cmap="viridis",
        norm=norm,
        edgecolor="#40454B",
        linewidth=0.22,
    )
    for row in accepted.itertuples():
        point = row.geometry.representative_point()
        ax.text(
            point.x,
            point.y,
            f"{row.height_optimized_m:.0f}",
            ha="center",
            va="center",
            fontsize=1.8,
            color="white" if norm(row.height_optimized_m) < 0.58 else "#111111",
        )

    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap="viridis"),
        ax=ax,
        fraction=0.032,
        pad=0.015,
        extend="max",
    )
    colorbar.set_label("Estimated building height above 4 m ground base / m")
    ax.set_title("High-rise-optimized roof-first building height estimates")
    ax.set_xlabel("Easting / m (UTM 51N)")
    ax.set_ylabel("Northing / m (UTM 51N)")
    ax.set_aspect("equal")
    ax.text(
        0.01,
        0.99,
        (
            f"Accepted PS-supported estimates: "
            f"{len(accepted)}/{len(buildings)}; "
            f"{int(buildings.highrise_optimization_applied.fillna(0).astype(int).sum())} "
            "high-rise estimates enhanced"
        ),
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#AAB0B6", "alpha": 0.92},
    )
    fig.tight_layout()

    svg_path = output_root / "svg" / f"{STEM}.svg"
    png_path = output_root / "png" / f"{STEM}.png"
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path, dpi=96, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "figure": STEM,
        "style_reference": "touying2 figures 01-08",
        "height_field": "height_optimized_m",
        "crs": "EPSG:32651",
        "total_buildings": int(len(buildings)),
        "accepted_buildings": int(len(accepted)),
        "rejected_buildings": int(len(rejected)),
        "highrise_enhanced_buildings": int(
            buildings.highrise_optimization_applied.fillna(0).astype(int).sum()
        ),
        "height_m": {
            "minimum": float(accepted.height_optimized_m.min()),
            "median": float(accepted.height_optimized_m.median()),
            "mean": float(accepted.height_optimized_m.mean()),
            "maximum": float(accepted.height_optimized_m.max()),
        },
        "display_vmax_m": vmax,
        "outputs": {"svg": str(svg_path), "png": str(png_path)},
        "pdf_generated": False,
    }
    summary_path = output_root / "tables" / f"{STEM}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render optimized heights using the existing touying2 map style."
    )
    parser.add_argument("--source-vector", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.source_vector.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
