"""Prepare two same-geometry Siwei SLCs on a common 900 x 630 radar grid."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy.ndimage import map_coordinates


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
GEOCODING = REPO / "geocoding"
DATA = GEOCODING / "data/四维高景"
INPUT = ROOT / "inputs/RE_SLAVES"
BUILDINGS = GEOCODING / "data/shp/tongji_clip_rslc_extent_equal_height_clean.shp"
GAMMA_CORE = GEOCODING / "siwei_image1_volume_geocode/code/gamma_projection_core.py"
DIFF_PAR = GEOCODING / "siwei_image1_volume_geocode/work/gamma_base_refinement/base_refinement.diff_par"
GAMMA_COORD_LIST = Path("/usr/local/GAMMA/DIFF/bin/coord_to_sarpix_list")
GAMMA_LIB = GEOCODING / "results/outputs/work/gamma_dsm_geocode/lib"
SOURCE_WINDOW = {"xoff": 22351, "yoff": 0, "width": 6726, "height": 4703}
OUTPUT_SHAPE = {"width": 900, "height": 630}


SCENES = {
    "20260624": DATA / "SVN2-03/6062500184900001",
    "20260616": DATA / "SVN2-05/6062500184900003",
}


def one(directory: Path, suffix: str) -> Path:
    matches = list(directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one *{suffix} in {directory}, found {len(matches)}")
    return matches[0]


def load_gamma_core():
    spec = importlib.util.spec_from_file_location("siwei_gamma_core_for_prepare", GAMMA_CORE)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def gamma_project_list(par: Path, longitude: np.ndarray, latitude: np.ndarray, ellipsoid: np.ndarray) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="siwei_prepare_project_") as temporary:
        temporary = Path(temporary); map_path = temporary / "map.txt"; sar_path = temporary / "sar.txt"
        np.savetxt(map_path, np.column_stack([latitude, longitude, ellipsoid]), fmt="%.12f %.12f %.6f")
        environment = os.environ.copy(); environment["LD_LIBRARY_PATH"] = str(GAMMA_LIB)
        subprocess.run([str(GAMMA_COORD_LIST), str(par), "-", "-", str(map_path), str(sar_path)],
                       check=True, env=environment, capture_output=True, text=True)
        result = np.loadtxt(sar_path)
    return result[None, :] if result.ndim == 1 else result


def building_control_points(gamma_core) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    buildings = gpd.read_file(BUILDINGS, engine="pyogrio")
    if buildings.crs is None:
        raise ValueError("building vector has no CRS")
    source_crs = str(buildings.crs)
    buildings = buildings.to_crs(4326)
    if not buildings.geometry.is_valid.all():
        buildings.geometry = buildings.geometry.make_valid()
    coordinates = []
    for geometry in buildings.geometry:
        polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
        for polygon in polygons:
            coordinates.extend(np.asarray(polygon.exterior.coords)[::max(1, len(polygon.exterior.coords)//8), :2])
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if len(coordinates) > 1200:
        coordinates = coordinates[np.linspace(0, len(coordinates)-1, 1200).astype(int)]
    longitude, latitude = coordinates[:, 0], coordinates[:, 1]
    ellipsoid = 4.0 + gamma_core.egm96_undulation(latitude, longitude)
    audit = {"source_crs": source_crs, "working_crs": str(buildings.crs), "features": len(buildings),
             "valid_geometries": int(buildings.geometry.is_valid.sum()), "control_points": len(coordinates)}
    return longitude, latitude, ellipsoid, audit


def write_crop_par(path: Path, scene: str) -> None:
    path.write_text(
        "Gamma Interferometric SAR Processor (ISP) - Cropped/Resampled Parameter File\n\n"
        f"title: Siwei {scene} coregistered crop\n"
        "image_format: SCOMPLEX\nimage_geometry: SLANT_RANGE\n"
        f"range_samples: {OUTPUT_SHAPE['width']}\nazimuth_lines: {OUTPUT_SHAPE['height']}\n",
        encoding="utf-8",
    )


def write_rslc(path: Path, array: np.ndarray) -> None:
    array = np.clip(np.rint(array), -32768, 32767).astype(">i2")
    array.transpose(1, 2, 0).tofile(path)


def main() -> None:
    INPUT.mkdir(parents=True, exist_ok=True)
    gamma_core = load_gamma_core()
    source = {}
    for scene, directory in SCENES.items():
        source[scene] = {"tiff": one(directory, ".tiff"), "meta": one(directory, ".meta.xml")}
        gamma_core.write_gamma_par(source[scene]["meta"], INPUT / f"{scene}.source.slc.par")

    longitude, latitude, ellipsoid, vector_audit = building_control_points(gamma_core)
    master_full = gamma_project_list(INPUT / "20260624.source.slc.par", longitude, latitude, ellipsoid)[:, :2]
    slave_full = gamma_project_list(INPUT / "20260616.source.slc.par", longitude, latitude, ellipsoid)[:, :2]
    design = np.column_stack([master_full[:, 0], master_full[:, 1], np.ones(len(master_full))])
    col_coefficients = np.linalg.lstsq(design, slave_full[:, 0], rcond=None)[0]
    row_coefficients = np.linalg.lstsq(design, slave_full[:, 1], rcond=None)[0]
    fitted = np.column_stack([design @ col_coefficients, design @ row_coefficients])
    residual = slave_full - fitted

    with rasterio.open(source["20260624"]["tiff"]) as dataset:
        master = dataset.read(
            [1, 2], window=Window(SOURCE_WINDOW["xoff"], SOURCE_WINDOW["yoff"], SOURCE_WINDOW["width"], SOURCE_WINDOW["height"]),
            out_shape=(2, OUTPUT_SHAPE["height"], OUTPUT_SHAPE["width"]),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
    write_rslc(INPUT / "20260624.rslc", master)

    out_cols = np.arange(OUTPUT_SHAPE["width"], dtype=np.float64)
    out_rows = np.arange(OUTPUT_SHAPE["height"], dtype=np.float64)
    col_grid, row_grid = np.meshgrid(out_cols, out_rows)
    master_col = SOURCE_WINDOW["xoff"] + (col_grid + 0.5) * SOURCE_WINDOW["width"] / OUTPUT_SHAPE["width"] - 0.5
    master_row = SOURCE_WINDOW["yoff"] + (row_grid + 0.5) * SOURCE_WINDOW["height"] / OUTPUT_SHAPE["height"] - 0.5
    flat_design = np.column_stack([master_col.ravel(), master_row.ravel(), np.ones(master_col.size)])
    slave_col = (flat_design @ col_coefficients).reshape(master_col.shape)
    slave_row = (flat_design @ row_coefficients).reshape(master_row.shape)
    x0 = int(np.floor(slave_col.min())) - 3; y0 = int(np.floor(slave_row.min())) - 3
    x1 = int(np.ceil(slave_col.max())) + 4; y1 = int(np.ceil(slave_row.max())) + 4
    with rasterio.open(source["20260616"]["tiff"]) as dataset:
        window = Window(x0, y0, x1-x0, y1-y0)
        crop = dataset.read([1, 2], window=window, boundless=True, fill_value=0).astype(np.float32)
    coordinates = [slave_row - y0, slave_col - x0]
    slave = np.stack([map_coordinates(crop[band], coordinates, order=1, mode="constant", cval=0.0)
                      for band in range(2)])
    write_rslc(INPUT / "20260616.rslc", slave)

    for scene in SCENES:
        par = INPUT / f"{scene}.rslc.par"; write_crop_par(par, scene)
        projection = {
            "source_gamma_par": str(INPUT / "20260624.source.slc.par"),
            "diff_par": str(DIFF_PAR), "source_window": SOURCE_WINDOW, "output_shape": OUTPUT_SHAPE,
            "height_input": "Wusong elevation metres; converted with provisional GAMMA EGM96 proxy",
        }
        par.with_suffix(".projection.json").write_text(json.dumps(projection, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "included_scenes": [
            {"scene": "20260624", "sensor": "SVN2-03", "beam": "SP_527", "orbit": "descending", "role": "master"},
            {"scene": "20260616", "sensor": "SVN2-05", "beam": "SP_527", "orbit": "descending", "role": "coregistered evidence"},
        ],
        "excluded_scenes": [
            {"scene": "20260412", "reason": "ascending orbit; incompatible layover direction"},
            {"scene": "20260328", "reason": "SP_506 and 23.50 degree incidence; incompatible pixel geometry"},
        ],
        "building_vector_audit": vector_audit,
        "master_source_window": SOURCE_WINDOW, "output_shape": OUTPUT_SHAPE,
        "slave_to_master_ground_control_affine": {
            "slave_col_from_master_col_row_1": col_coefficients.tolist(),
            "slave_row_from_master_col_row_1": row_coefficients.tolist(),
            "residual_full_resolution_px": {
                "median": float(np.median(np.linalg.norm(residual, axis=1))),
                "p95": float(np.percentile(np.linalg.norm(residual, axis=1), 95)),
                "max": float(np.max(np.linalg.norm(residual, axis=1))),
            },
        },
        "vertical_datum": "PROVISIONAL: Wusong + GAMMA EGM96 proxy",
    }
    # The parent workflow treats this table only as a cross-method diagnostic.
    # Adapt the available historical strict-audit columns without using them in
    # the Siwei candidate score or as a missing-height fill.
    historical = pd.read_csv(
        REPO / "height_estimation/building_height_estimation_roof_only/results/tables/roof_only_v9_cross_method_spatial_audit/roof_only_v9_audit.csv"
    )[["fid", "strict_height_m", "strict_quality", "height_scene_range_m"]]
    historical = historical.rename(columns={"strict_height_m": "height_est_m", "strict_quality": "quality"})
    historical.to_csv(ROOT / "inputs/strict_audit_adapter.csv", index=False)
    (ROOT / "inputs/input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
