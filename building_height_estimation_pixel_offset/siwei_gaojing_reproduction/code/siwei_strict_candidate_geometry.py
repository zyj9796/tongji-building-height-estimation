"""Strict GAMMA LLH-to-SAR projector for the Siwei cropped/multilooked grid."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


GAMMA_COORD_LIST = Path("/usr/local/GAMMA/DIFF/bin/coord_to_sarpix_list")
GAMMA_EGM96 = Path("/usr/local/GAMMA/DIFF/scripts/egm96.dem")
GAMMA_LIB = Path("/home/u/geocoding/geo_bc/a_geo_tongji/geocoding/results/outputs/work/gamma_dsm_geocode/lib")
_EGM96_GRID = None


def clean_ring_lonlat(ring: np.ndarray) -> np.ndarray:
    ring = np.asarray(ring, dtype=np.float64)[:, :2]
    if len(ring) > 2 and np.allclose(ring[0], ring[-1], rtol=0.0, atol=1e-12):
        ring = ring[:-1]
    keep = np.r_[True, np.any(np.diff(ring, axis=0) != 0, axis=1)]
    ring = ring[keep]
    if len(ring) < 3:
        raise ValueError("building exterior has fewer than three unique vertices")
    return ring


def egm96_undulation(latitude, longitude):
    global _EGM96_GRID
    if _EGM96_GRID is None:
        values = np.fromfile(GAMMA_EGM96, dtype=">f4")
        if values.size != 720 * 1440:
            raise ValueError("unexpected GAMMA EGM96 grid size")
        _EGM96_GRID = values.reshape(720, 1440).astype(np.float64)
    latitude, longitude = np.broadcast_arrays(np.asarray(latitude, float), np.asarray(longitude, float))
    row = (latitude - 89.875) / -0.25; col = (longitude + 179.875) / 0.25
    r0 = np.clip(np.floor(row).astype(int), 0, 718); c0 = np.clip(np.floor(col).astype(int), 0, 1438)
    dr, dc = row - r0, col - c0
    return (_EGM96_GRID[r0, c0] * (1-dr) * (1-dc) + _EGM96_GRID[r0+1, c0] * dr * (1-dc)
            + _EGM96_GRID[r0, c0+1] * (1-dr) * dc + _EGM96_GRID[r0+1, c0+1] * dr * dc)


class StrictRadarProjector:
    def __init__(self, par_path: Path):
        self.par_path = Path(par_path)
        metadata_path = self.par_path.with_suffix(".projection.json")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.source_par = Path(self.metadata["source_gamma_par"])
        self.diff_par = Path(self.metadata["diff_par"]) if self.metadata.get("diff_par") else None
        self.x0 = float(self.metadata["source_window"]["xoff"])
        self.y0 = float(self.metadata["source_window"]["yoff"])
        self.source_width = float(self.metadata["source_window"]["width"])
        self.source_height = float(self.metadata["source_window"]["height"])
        self.width = int(self.metadata["output_shape"]["width"])
        self.height = int(self.metadata["output_shape"]["height"])
        self._cache = {}

    def _project(self, latitude: np.ndarray, longitude: np.ndarray, wusong_height: np.ndarray) -> np.ndarray:
        keys = [(round(float(lat), 11), round(float(lon), 11), round(float(h), 4))
                for lat, lon, h in zip(latitude, longitude, wusong_height)]
        missing_indices = [i for i, key in enumerate(keys) if key not in self._cache]
        if missing_indices:
            lat = latitude[missing_indices]; lon = longitude[missing_indices]; h = wusong_height[missing_indices]
            ellipsoid = h + egm96_undulation(lat, lon)
            with tempfile.TemporaryDirectory(prefix="siwei_gamma_project_") as temporary:
                temporary = Path(temporary); map_path = temporary / "map.txt"; sar_path = temporary / "sar.txt"
                np.savetxt(map_path, np.column_stack([lat, lon, ellipsoid]), fmt="%.12f %.12f %.6f")
                command = [str(GAMMA_COORD_LIST), str(self.source_par), "-", "-", str(map_path), str(sar_path)]
                if self.diff_par is not None:
                    command.append(str(self.diff_par))
                environment = os.environ.copy(); environment["LD_LIBRARY_PATH"] = str(GAMMA_LIB)
                subprocess.run(command, check=True, env=environment, capture_output=True, text=True)
                projected = np.loadtxt(sar_path)
                if projected.ndim == 1: projected = projected[None, :]
            for index, xy in zip(missing_indices, projected):
                full_col, full_row = float(xy[0]), float(xy[1])
                col = (full_col - self.x0 + 0.5) * self.width / self.source_width - 0.5
                row = (full_row - self.y0 + 0.5) * self.height / self.source_height - 0.5
                self._cache[keys[index]] = (row, col)
        return np.asarray([self._cache[key] for key in keys], dtype=np.float64)

    def project_height_grid(self, ring: np.ndarray, absolute_heights: np.ndarray):
        ring = clean_ring_lonlat(ring)
        heights = np.asarray(absolute_heights, dtype=np.float64)
        longitude = np.tile(ring[:, 0], len(heights)); latitude = np.tile(ring[:, 1], len(heights))
        flat_heights = np.repeat(heights, len(ring))
        projected = self._project(latitude, longitude, flat_heights).reshape(len(heights), len(ring), 2)
        return projected[:, :, 0], projected[:, :, 1]
