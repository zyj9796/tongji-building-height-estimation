"""Resume the isolated reproduction after figures 01--10 already exist."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT.parent / "code"


def patched(name: str):
    module = importlib.import_module(name)
    if hasattr(module, "ROOT"): module.ROOT = ROOT
    return module


def main():
    sys.path.insert(0, str(CODE))
    pixel = patched("run_pixel_offset_height")
    patched("run_shape_adaptive_enhanced_sar_correction")
    patched("export_hybrid_pixel_offset_height_map").main()
    patched("run_image_feature_only_registration").main()
    patched("run_joint_quantity_quality_optimization").main()
    patched("finalize_outputs").main()
    patched("build_clean_delivery").main()


if __name__ == "__main__": main()
