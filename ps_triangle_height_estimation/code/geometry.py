from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline


SPEED_OF_LIGHT = 299792458.0
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def parse_gamma_par(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, object] = {"state_vectors": []}
    scalar_keys = {
        "start_time": float,
        "azimuth_line_time": float,
        "range_samples": int,
        "azimuth_lines": int,
        "range_pixel_spacing": float,
        "near_range_slc": float,
        "center_range_slc": float,
        "radar_frequency": float,
        "time_of_first_state_vector": float,
        "state_vector_interval": float,
    }
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        key, rest = key.strip(), rest.strip()
        if key in scalar_keys:
            out[key] = scalar_keys[key](float(rest.split()[0]))
            continue
        if key == "doppler_polynomial":
            out[key] = [float(value) for value in rest.split()[:4]]
            continue
        match_position = re.match(r"state_vector_position_(\d+)", key)
        match_velocity = re.match(r"state_vector_velocity_(\d+)", key)
        if match_position or match_velocity:
            index = int((match_position or match_velocity).group(1)) - 1
            values = [float(value) for value in rest.split()[:3]]
            vectors = out["state_vectors"]
            while len(vectors) <= index:
                vectors.append({})
            vectors[index]["pos" if match_position else "vel"] = values
    out.setdefault("doppler_polynomial", [0.0, 0.0, 0.0, 0.0])
    required = [*scalar_keys, "state_vectors"]
    missing = [key for key in required if key not in out or not out[key]]
    if missing:
        raise ValueError(f"Missing GAMMA parameters in {path}: {missing}")
    return out


def clean_ring(coordinates: np.ndarray) -> np.ndarray:
    ring = np.asarray(coordinates, dtype=np.float64)[:, :2]
    if ring.shape[0] > 1 and np.allclose(ring[0], ring[-1], atol=1e-12, rtol=0):
        ring = ring[:-1]
    keep = [0]
    for index in range(1, ring.shape[0]):
        if not np.allclose(ring[index], ring[keep[-1]], atol=1e-12, rtol=0):
            keep.append(index)
    ring = ring[keep]
    if ring.shape[0] < 3:
        raise ValueError("Footprint has fewer than three unique vertices")
    return ring


def signed_area(xy: np.ndarray) -> float:
    return 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))


def cross2(left: np.ndarray, right: np.ndarray) -> float:
    """Scalar 2-D cross product, compatible with NumPy 2.x."""
    return float(left[0] * right[1] - left[1] * right[0])


def point_in_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float) -> bool:
    cross_ab = cross2(b - a, point - a)
    cross_bc = cross2(c - b, point - b)
    cross_ca = cross2(a - c, point - c)
    return cross_ab >= -eps and cross_bc >= -eps and cross_ca >= -eps


def ear_clip_triangulation(ring_lonlat: np.ndarray) -> np.ndarray:
    ring = clean_ring(ring_lonlat)
    latitude = float(np.mean(ring[:, 1]))
    xy = np.column_stack([ring[:, 0] * math.cos(math.radians(latitude)), ring[:, 1]])
    xy -= np.mean(xy, axis=0)
    order = list(range(ring.shape[0]))
    if signed_area(xy) < 0:
        order.reverse()
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(order) > 3:
        found = False
        scale = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-12)
        eps = scale * scale * 1e-10
        for position, current in enumerate(order):
            previous = order[(position - 1) % len(order)]
            following = order[(position + 1) % len(order)]
            a, b, c = xy[previous], xy[current], xy[following]
            if cross2(b - a, c - b) <= eps:
                continue
            if any(
                point_in_triangle(xy[index], a, b, c, eps)
                for index in order
                if index not in (previous, current, following)
            ):
                continue
            triangles.append((previous, current, following))
            del order[position]
            found = True
            break
        guard += 1
        if not found or guard > ring.shape[0] ** 2:
            raise ValueError("Constrained footprint triangulation failed")
    triangles.append(tuple(order))
    return np.asarray(triangles, dtype=np.int32)


def llh_to_ecef(lon_deg: np.ndarray, lat_deg: np.ndarray, height_m: np.ndarray) -> np.ndarray:
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    height = np.asarray(height_m, dtype=np.float64)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    prime_vertical = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (prime_vertical + height) * cos_lat * np.cos(lon)
    y = (prime_vertical + height) * cos_lat * np.sin(lon)
    z = (prime_vertical * (1.0 - WGS84_E2) + height) * sin_lat
    return np.column_stack([x, y, z])


def ecef_to_llh(points_ecef: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(np.asarray(points_ecef, dtype=np.float64))
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        height = p / np.maximum(np.cos(lat), 1e-15) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / np.maximum(n + height, 1.0)))
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    height = p / np.maximum(np.cos(lat), 1e-15) - n
    return np.column_stack([np.degrees(lon), np.degrees(lat), height])


@dataclass(frozen=True)
class BuildingMesh:
    vertices_ecef: np.ndarray
    vertices_lonlat: np.ndarray
    vertex_is_top: np.ndarray
    projected_xy: np.ndarray
    triangles: np.ndarray
    surfaces: np.ndarray


class StrictRadarProjector:
    def __init__(self, par_path: Path, newton_iterations: int = 8) -> None:
        self.par_path = Path(par_path)
        self.par = parse_gamma_par(self.par_path)
        vectors = self.par["state_vectors"]
        first_time = float(self.par["time_of_first_state_vector"])
        interval = float(self.par["state_vector_interval"])
        times = np.asarray([first_time + i * interval for i in range(len(vectors))])
        self.position_spline = CubicSpline(times, np.asarray([v["pos"] for v in vectors]), axis=0)
        self.velocity_spline = CubicSpline(times, np.asarray([v["vel"] for v in vectors]), axis=0)
        self.newton_iterations = int(newton_iterations)

    def _residual(self, times: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        satellite = np.asarray(self.position_spline(times), dtype=np.float64)
        velocity = np.asarray(self.velocity_spline(times), dtype=np.float64)
        los = points - satellite
        slant_range = np.linalg.norm(los, axis=-1)
        wavelength = SPEED_OF_LIGHT / float(self.par["radar_frequency"])
        observed = -2.0 * np.sum(los * velocity, axis=-1) / (wavelength * slant_range)
        coefficients = np.asarray(self.par["doppler_polynomial"], dtype=np.float64)
        offset = slant_range - float(self.par["center_range_slc"])
        model = coefficients[0] + coefficients[1] * offset + coefficients[2] * offset**2 + coefficients[3] * offset**3
        return observed - model, slant_range

    def project_ecef(self, points_ecef: np.ndarray) -> np.ndarray:
        points = np.asarray(points_ecef, dtype=np.float64)
        center = float(self.par["start_time"]) + 0.5 * float(self.par["azimuth_lines"]) * float(self.par["azimuth_line_time"])
        minimum = float(self.par["start_time"]) - 2.0
        maximum = float(self.par["start_time"]) + float(self.par["azimuth_lines"]) * float(self.par["azimuth_line_time"]) + 2.0
        times = np.full(points.shape[0], center, dtype=np.float64)
        dt = 1e-3
        for _ in range(self.newton_iterations):
            residual, _ = self._residual(times, points)
            plus, _ = self._residual(times + dt, points)
            minus, _ = self._residual(times - dt, points)
            derivative = (plus - minus) / (2.0 * dt)
            step = np.nan_to_num(residual / np.where(np.abs(derivative) > 1e-8, derivative, np.nan))
            times = np.clip(times - np.clip(step, -0.5, 0.5), minimum, maximum)
        residual, slant_range = self._residual(times, points)
        if float(np.max(np.abs(residual))) > 1e-3:
            raise RuntimeError(f"Zero-Doppler solution failed: {np.max(np.abs(residual)):.6g} Hz")
        rows = (times - float(self.par["start_time"])) / float(self.par["azimuth_line_time"])
        cols = (slant_range - float(self.par["near_range_slc"])) / float(self.par["range_pixel_spacing"])
        return np.column_stack([cols, rows])

    def build_mesh(
        self,
        ring_lonlat: np.ndarray,
        base_elevation_m: float,
        building_height_m: float,
        row_shift_px: float,
        col_shift_px: float,
        top_row_shift_px: float | None = None,
        top_col_shift_px: float | None = None,
    ) -> BuildingMesh:
        ring = clean_ring(ring_lonlat)
        n = ring.shape[0]
        bottom = llh_to_ecef(ring[:, 0], ring[:, 1], np.full(n, base_elevation_m))
        top = llh_to_ecef(ring[:, 0], ring[:, 1], np.full(n, base_elevation_m + building_height_m))
        vertices = np.vstack([bottom, top])
        projected = self.project_ecef(vertices)
        projected[:n] += np.asarray([col_shift_px, row_shift_px])
        projected[n:] += np.asarray(
            [
                col_shift_px if top_col_shift_px is None else top_col_shift_px,
                row_shift_px if top_row_shift_px is None else top_row_shift_px,
            ]
        )
        footprint = ear_clip_triangulation(ring)
        triangles: list[tuple[int, int, int]] = []
        surfaces: list[str] = []
        for i in range(n):
            j = (i + 1) % n
            triangles.extend([(i, j, n + j), (i, n + j, n + i)])
            surfaces.extend(["wall", "wall"])
        for a, b, c in footprint:
            triangles.append((n + int(a), n + int(b), n + int(c)))
            surfaces.append("roof")
            triangles.append((int(c), int(b), int(a)))
            surfaces.append("bottom")
        return BuildingMesh(
            vertices_ecef=vertices,
            vertices_lonlat=np.vstack([ring, ring]),
            vertex_is_top=np.r_[np.zeros(n), np.ones(n)],
            projected_xy=projected,
            triangles=np.asarray(triangles, dtype=np.int32),
            surfaces=np.asarray(surfaces),
        )


def barycentric_weights(point_xy: np.ndarray, triangle_xy: np.ndarray) -> tuple[np.ndarray, float]:
    point = np.asarray(point_xy, dtype=np.float64)
    triangle = np.asarray(triangle_xy, dtype=np.float64)
    matrix = np.column_stack([triangle[0] - triangle[2], triangle[1] - triangle[2]])
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-10:
        return np.full(3, np.nan), float("inf")
    first, second = np.linalg.solve(matrix, point - triangle[2])
    weights = np.asarray([first, second, 1.0 - first - second])
    violation = float(max(0.0, -np.min(weights), np.max(weights) - 1.0))
    return weights, violation
