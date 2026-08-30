"""Rerun every stage affected by the corrected Wusong 4 m base definition."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT_CODE = ROOT.parent / "code"


def module(name: str):
    item = importlib.import_module(name)
    if hasattr(item, "ROOT"):
        item.ROOT = ROOT
    return item


def main() -> None:
    sys.path.insert(0, str(PARENT_CODE))
    module("run_pixel_offset_height")
    module("export_building_base_4m").main()
    module("export_all_buildings_shp_height_projection").main()
    module("run_shp_height_local_sar_correction").run(ROOT / "config.json")
    module("run_shape_adaptive_enhanced_sar_correction").main()
    module("build_hybrid_shape_adaptive_correction").main()
    module("export_hybrid_pixel_offset_height_map").main()
    module("run_image_feature_only_registration").main()
    module("run_joint_quantity_quality_optimization").main()
    module("finalize_outputs").main()


if __name__ == "__main__":
    main()
