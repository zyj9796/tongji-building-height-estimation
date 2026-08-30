"""Run the parent project's complete 16-figure chain in this isolated root."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
CODE = PARENT / "code"
CONFIG = ROOT / "config.json"


def patched_module(name: str):
    module = importlib.import_module(name)
    if hasattr(module, "ROOT"):
        module.ROOT = ROOT
    return module


def main() -> None:
    if not (ROOT / "inputs/RE_SLAVES/20260624.rslc").exists():
        raise FileNotFoundError("run code/prepare_siwei_inputs.py first")
    for directory in ("results/tables", "results/vectors", "results/picall/过程图件", "results/picall/正式图件", "results/processed_sar"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(CODE))
    pixel = patched_module("run_pixel_offset_height")
    pixel.run(CONFIG)
    patched_module("export_building_base_4m").main()
    patched_module("export_all_buildings_shp_height_projection").main()
    patched_module("run_shp_height_local_sar_correction").run(CONFIG)
    patched_module("run_shape_adaptive_enhanced_sar_correction").main()
    patched_module("build_hybrid_shape_adaptive_correction").main()
    patched_module("export_hybrid_pixel_offset_height_map").main()
    patched_module("run_image_feature_only_registration").main()
    patched_module("run_joint_quantity_quality_optimization").main()
    patched_module("finalize_outputs").main()
    patched_module("build_clean_delivery").main()


if __name__ == "__main__":
    main()
