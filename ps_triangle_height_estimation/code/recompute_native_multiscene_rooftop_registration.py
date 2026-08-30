from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from geometry import StrictRadarProjector
from map_ps_to_building_surfaces import load_height_overrides, resolve
from recompute_rooftop_registration import run as run_rooftop_v3


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"


def load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_reference_algorithms(config: dict):
    """Load the reviewed pixel-offset feature code without reusing its projection."""
    cfg = config["iterative_adjustment"]["registration_native_v5"]
    code_dir = resolve(cfg["reference_code_dir"])
    code_text = str(code_dir)
    if code_text not in sys.path:
        sys.path.insert(0, code_text)
    image_registration = importlib.import_module("run_image_feature_only_registration")
    shape_adaptive = importlib.import_module("run_shape_adaptive_enhanced_sar_correction")
    roof_evidence = load_file_module(
        "native_multiscene_roof_evidence",
        resolve(cfg["roof_evidence_code"]),
    )
    return image_registration, shape_adaptive, roof_evidence


def quality_gate(result: dict, config: dict) -> bool:
    cfg = config["iterative_adjustment"]["registration_native_v5"]
    return bool(
        int(result["registration_accepted"]) == 1
        and np.hypot(float(result["dx"]), float(result["dy"]))
        <= float(cfg["maximum_shift_px"])
        and float(result["score_margin"]) >= float(cfg["minimum_score_margin"])
        and float(result["gain_oriented_edge"]) > float(cfg["minimum_oriented_edge_gain"])
        and float(result["gain_continuity"]) > float(cfg["minimum_continuity_gain"])
        and float(result["pair_distance_px"]) <= float(cfg["maximum_scene_pair_distance_px"])
        and float(result["fused_to_pair_px"]) <= float(cfg["maximum_fused_to_pair_distance_px"])
    )


def run(
    config_path: Path,
    height_overrides_path: Path,
    output_csv: Path,
    max_buildings: int = 0,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = config["iterative_adjustment"]["registration_native_v5"]
    base_path = output_csv.with_name(f"{output_csv.stem}.feature_v3_base.csv")
    run_rooftop_v3(config_path, height_overrides_path, base_path, max_buildings)
    try:
        base = pd.read_csv(base_path)
        image_registration, shape_adaptive, roof_evidence = load_reference_algorithms(config)
        evidence_config = {
            "scenes": list(cfg["scenes"]),
            "inputs": {"rslc_dir": str(resolve(cfg["rslc_dir"]))},
        }
        raw_evidence, _, _ = roof_evidence.load_evidence(evidence_config)
        evidence, _ = image_registration.prepare_evidence(raw_evidence)

        buildings = (
            gpd.read_file(resolve(config["inputs"]["buildings"]))
            .reset_index(drop=True)
            .to_crs("EPSG:4326")
        )
        buildings["fid"] = np.arange(len(buildings), dtype=np.int64)
        buildings_metric = buildings.to_crs("EPSG:32651")
        if max_buildings > 0:
            buildings = buildings.iloc[:max_buildings].copy()
            buildings_metric = buildings_metric.iloc[:max_buildings].copy()

        heights = load_height_overrides(height_overrides_path)
        projector = StrictRadarProjector(resolve(config["inputs"]["rslc_par"]))
        global_row = float(config["registration"]["global_row_shift_px"])
        global_col = float(config["registration"]["global_col_shift_px"])
        native_rows: list[dict] = []
        for position, building in enumerate(buildings.itertuples(), start=1):
            fid = int(building.fid)
            record: dict = {"fid": fid}
            try:
                height = heights.get(fid, float(getattr(building, config["height_field"])))
                mesh = projector.build_mesh(
                    np.asarray(building.geometry.exterior.coords),
                    float(config["base_elevation_m"]),
                    float(height),
                    global_row,
                    global_col,
                )
                vertex_count = len(mesh.projected_xy) // 2
                roof = Polygon(mesh.projected_xy[vertex_count:]).buffer(0)
                descriptor = shape_adaptive.shape_descriptor(
                    buildings_metric.iloc[position - 1].geometry
                )
                strategy = str(descriptor["shape_strategy"])
                result = image_registration.best_registration(
                    roof,
                    evidence,
                    strategy,
                )
                if result is None:
                    record.update(
                        {
                            "native_candidate_available": 0,
                            "native_quality_gate": 0,
                            "native_shape_strategy": strategy,
                            "native_failure_reason": "insufficient_roof_samples",
                        }
                    )
                else:
                    accepted = quality_gate(result, config)
                    record.update(
                        {
                            "native_candidate_available": 1,
                            "native_quality_gate": int(accepted),
                            "native_shape_strategy": strategy,
                            **{f"native_{key}": value for key, value in result.items()},
                        }
                    )
            except Exception as exc:
                record.update(
                    {
                        "native_candidate_available": 0,
                        "native_quality_gate": 0,
                        "native_failure_reason": str(exc),
                    }
                )
            native_rows.append(record)
            if position % 50 == 0 or position == len(buildings):
                accepted_count = sum(row.get("native_quality_gate", 0) for row in native_rows)
                print(
                    f"native-multiscene {position}/{len(buildings)} accepted={accepted_count}",
                    flush=True,
                )

        native = pd.DataFrame(native_rows)
        result = base.merge(native, on="fid", how="left", validate="one_to_one")
        use_native = result.native_quality_gate.fillna(0).eq(1)
        for column in ("native_dx", "native_dy"):
            if column not in result:
                result[column] = np.nan
        result["applied_row_shift"] = result.applied_row_shift.astype(float)
        result["applied_col_shift"] = result.applied_col_shift.astype(float)
        result.loc[use_native, "applied_row_shift"] = result.loc[
            use_native, "native_dy"
        ].to_numpy()
        result.loc[use_native, "applied_col_shift"] = result.loc[
            use_native, "native_dx"
        ].to_numpy()
        result.loc[use_native, "accepted"] = 1
        result.loc[use_native, "local_refinement_accepted"] = 1
        result["registration_feature_mode"] = (
            "native_current_geometry_multiscene_shape_adaptive_subpixel_v5"
        )
        result["native_multiscene_override"] = use_native.astype(int)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
    finally:
        base_path.unlink(missing_ok=True)
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reviewed multiscene feature registration directly on roofs "
            "projected with the current base-elevation and height convention."
        )
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
