from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from map_ps_to_building_surfaces import resolve
from recompute_rooftop_registration import run as run_rooftop_v3


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"


def merge_reference_registration(
    base: pd.DataFrame,
    reference: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Overlay only independently high-confidence multiscene subpixel corrections."""
    required = {
        "fid",
        "dx",
        "dy",
        "registration_accepted",
        "score_margin",
        "gain_oriented_edge",
        "gain_continuity",
        "pair_distance_px",
        "fused_to_pair_px",
        "shape_strategy",
        "registration_quality",
    }
    missing = required - set(reference.columns)
    if missing:
        raise ValueError(f"Reference registration is missing columns: {sorted(missing)}")
    if base.fid.duplicated().any() or reference.fid.duplicated().any():
        raise ValueError("Registration fid values must be unique")
    aligned = reference.set_index("fid").reindex(base.fid).reset_index(drop=True)
    if aligned.dx.isna().all():
        raise ValueError("Reference registration fid values do not match the current buildings")

    cfg = config["iterative_adjustment"]["registration_reference_v4"]
    magnitude = np.hypot(aligned.dx, aligned.dy)
    use_reference = (
        aligned.registration_accepted.eq(1)
        & (magnitude <= float(cfg["maximum_reference_shift_px"]))
        & (aligned.score_margin >= float(cfg["minimum_score_margin"]))
        & (aligned.gain_oriented_edge > float(cfg["minimum_oriented_edge_gain"]))
        & (aligned.gain_continuity > float(cfg["minimum_continuity_gain"]))
        & (aligned.pair_distance_px <= float(cfg["maximum_scene_pair_distance_px"]))
        & (aligned.fused_to_pair_px <= float(cfg["maximum_fused_to_pair_distance_px"]))
    )

    result = base.copy()
    result["applied_row_shift"] = result.applied_row_shift.astype(float)
    result["applied_col_shift"] = result.applied_col_shift.astype(float)
    result.loc[use_reference, "applied_row_shift"] = aligned.loc[use_reference, "dy"].to_numpy()
    result.loc[use_reference, "applied_col_shift"] = aligned.loc[use_reference, "dx"].to_numpy()
    result.loc[use_reference, "accepted"] = 1
    result.loc[use_reference, "local_refinement_accepted"] = 1
    result["registration_feature_mode"] = "hybrid_multiscene_shape_adaptive_subpixel_v4"
    result["reference_multiscene_override"] = use_reference.astype(int)
    result["reference_row_shift"] = aligned.dy
    result["reference_col_shift"] = aligned.dx
    result["reference_score_margin"] = aligned.score_margin
    result["reference_pair_distance_px"] = aligned.pair_distance_px
    result["reference_fused_to_pair_px"] = aligned.fused_to_pair_px
    result["reference_oriented_edge_gain"] = aligned.gain_oriented_edge
    result["reference_continuity_gain"] = aligned.gain_continuity
    result["reference_shape_strategy"] = aligned.shape_strategy
    result["reference_registration_quality"] = aligned.registration_quality
    return result


def run(
    config_path: Path,
    height_overrides_path: Path,
    output_csv: Path,
    max_buildings: int = 0,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_path = output_csv.with_name(f"{output_csv.stem}.feature_v3_base.csv")
    run_rooftop_v3(config_path, height_overrides_path, base_path, max_buildings)
    try:
        base = pd.read_csv(base_path)
        reference_path = resolve(
            config["iterative_adjustment"]["registration_reference_v4"]["registration_table"]
        )
        reference = pd.read_csv(reference_path)
        if max_buildings > 0:
            reference = reference.loc[reference.fid < max_buildings].copy()
        merged = merge_reference_registration(base, reference, config)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_csv, index=False)
    finally:
        base_path.unlink(missing_ok=True)
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine conservative rooftop registration with high-confidence multiscene subpixel corrections."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--height-overrides", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--max-buildings", type=int, default=0)
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.height_overrides.resolve(),
        args.output_csv.resolve(),
        args.max_buildings,
    )


if __name__ == "__main__":
    main()
