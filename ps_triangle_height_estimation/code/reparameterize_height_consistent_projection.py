from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from geometry import StrictRadarProjector
from map_ps_to_building_surfaces import resolve


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"


def decompose_shift(
    shift_col_row: np.ndarray,
    height_sensitivity_col_row: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Split a registration shift into height-direction and perpendicular parts."""
    shift = np.asarray(shift_col_row, dtype=float)
    sensitivity = np.asarray(height_sensitivity_col_row, dtype=float)
    norm_squared = float(np.dot(sensitivity, sensitivity))
    if norm_squared <= 1e-18 or not np.isfinite(norm_squared):
        return 0.0, shift.copy()
    height_change = float(np.dot(shift, sensitivity) / norm_squared)
    return height_change, shift - height_change * sensitivity


def roof_centroid(
    projector: StrictRadarProjector,
    ring: np.ndarray,
    base_elevation_m: float,
    building_height_m: float,
    global_row_shift_px: float,
    global_col_shift_px: float,
) -> np.ndarray:
    mesh = projector.build_mesh(
        ring,
        base_elevation_m,
        building_height_m,
        global_row_shift_px,
        global_col_shift_px,
    )
    vertex_count = len(mesh.projected_xy) // 2
    return np.mean(mesh.projected_xy[vertex_count:], axis=0)


def run(
    config_path: Path,
    heights_path: Path,
    registration_path: Path,
    output_heights_path: Path,
    output_registration_path: Path,
) -> tuple[Path, Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = config["iterative_adjustment"]["height_consistent_projection_v6"]
    buildings = (
        gpd.read_file(resolve(config["inputs"]["buildings"]))
        .reset_index(drop=True)
        .to_crs("EPSG:4326")
    )
    buildings["fid"] = np.arange(len(buildings), dtype=np.int64)
    heights = pd.read_csv(heights_path)
    registration = pd.read_csv(registration_path)
    if heights.fid.duplicated().any() or registration.fid.duplicated().any():
        raise ValueError("Height and registration fid values must be unique")
    height_column = (
        "height_current_m"
        if "height_current_m" in heights
        else "height_projection_m"
    )
    height_by_fid = heights.set_index("fid")[height_column]
    registration = registration.set_index("fid").reindex(buildings.fid).reset_index()

    projector = StrictRadarProjector(resolve(config["inputs"]["rslc_par"]))
    base_elevation = float(config["base_elevation_m"])
    global_row = float(config["registration"]["global_row_shift_px"])
    global_col = float(config["registration"]["global_col_shift_px"])
    derivative_half_step = float(cfg["derivative_half_step_m"])
    minimum_height = float(config["iterative_adjustment"]["minimum_height_m"])
    maximum_height = float(config["iterative_adjustment"]["maximum_height_m"])
    maximum_height_change = float(cfg["maximum_initial_height_change_m"])
    maximum_perpendicular = float(cfg["maximum_perpendicular_residual_px"])
    selection_column = str(cfg.get("selection_column", "")).strip()

    output_height_rows: list[dict] = []
    audit_rows: list[dict] = []
    for building in buildings.itertuples():
        fid = int(building.fid)
        height = float(height_by_fid.loc[fid])
        row = registration.loc[registration.fid == fid].iloc[0]
        dc = float(row.applied_col_shift)
        dr = float(row.applied_row_shift)
        selected = bool(row.get(selection_column, 0)) if selection_column else True
        ring = np.asarray(building.geometry.exterior.coords)
        lower = max(minimum_height, height - derivative_half_step)
        upper = min(maximum_height, height + derivative_half_step)
        if upper <= lower:
            dxy = np.asarray([np.nan, np.nan])
        else:
            lower_centroid = roof_centroid(
                projector, ring, base_elevation, lower, global_row, global_col
            )
            upper_centroid = roof_centroid(
                projector, ring, base_elevation, upper, global_row, global_col
            )
            dxy = (upper_centroid - lower_centroid) / (upper - lower)
        pixels_per_m = float(np.hypot(*dxy))
        shift = np.asarray([dc, dr], dtype=float)
        if not selected:
            height_change = 0.0
            perpendicular = shift
            accepted = False
            reason = "not_selected"
        elif pixels_per_m <= 1e-9 or not np.isfinite(pixels_per_m):
            height_change = 0.0
            perpendicular = shift
            accepted = False
            reason = "invalid_height_sensitivity"
        else:
            height_change, perpendicular = decompose_shift(shift, dxy)
            perpendicular_magnitude = float(np.hypot(*perpendicular))
            accepted = bool(
                abs(height_change) <= maximum_height_change
                and perpendicular_magnitude <= maximum_perpendicular
            )
            reason = "accepted"
            if abs(height_change) > maximum_height_change:
                reason = "height_component_too_large"
            elif perpendicular_magnitude > maximum_perpendicular:
                reason = "perpendicular_component_too_large"
        if accepted:
            adjusted_height = float(
                np.clip(height + height_change, minimum_height, maximum_height)
            )
            applied_dc, applied_dr = map(float, perpendicular)
        elif not selected:
            adjusted_height = height
            applied_dc, applied_dr = dc, dr
        else:
            adjusted_height = height
            applied_dc = 0.0 if bool(row.accepted) else dc
            applied_dr = 0.0 if bool(row.accepted) else dr
            height_change = 0.0

        output_height_rows.append(
            {
                "fid": fid,
                "clean_id": int(building.clean_id),
                "height_current_m": adjusted_height,
            }
        )
        audit_rows.append(
            {
                "fid": fid,
                "height_sensitivity_col_px_per_m": float(dxy[0]),
                "height_sensitivity_row_px_per_m": float(dxy[1]),
                "height_sensitivity_px_per_m": pixels_per_m,
                "original_row_shift_px": dr,
                "original_col_shift_px": dc,
                "projection_height_change_m": height_change,
                "perpendicular_row_shift_px": applied_dr,
                "perpendicular_col_shift_px": applied_dc,
                "height_consistent_reparameterization_accepted": int(accepted),
                "height_consistent_reparameterization_selected": int(selected),
                "height_consistent_reparameterization_reason": reason,
            }
        )

    output_heights_path.parent.mkdir(parents=True, exist_ok=True)
    output_registration_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_height_rows).to_csv(output_heights_path, index=False)
    audit = pd.DataFrame(audit_rows)
    output_registration = registration.merge(audit, on="fid", validate="one_to_one")
    output_registration["applied_row_shift"] = output_registration[
        "perpendicular_row_shift_px"
    ]
    output_registration["applied_col_shift"] = output_registration[
        "perpendicular_col_shift_px"
    ]
    output_registration["accepted"] = (
        np.hypot(
            output_registration.applied_row_shift,
            output_registration.applied_col_shift,
        )
        > 1e-9
    ).astype(int)
    output_registration["local_refinement_accepted"] = output_registration["accepted"]
    output_registration["registration_feature_mode"] = (
        "height_direction_strict_reprojection_plus_perpendicular_registration_v6"
    )
    output_registration.to_csv(output_registration_path, index=False)
    return output_heights_path, output_registration_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the height-direction component of rooftop registration into "
            "a strict reprojection height update and retain only perpendicular residuals."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--heights", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-heights", type=Path, required=True)
    parser.add_argument("--output-registration", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.heights.resolve(),
        args.registration.resolve(),
        args.output_heights.resolve(),
        args.output_registration.resolve(),
    )


if __name__ == "__main__":
    main()
