from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from geometry import StrictRadarProjector, barycentric_weights, ecef_to_llh, llh_to_ecef  # noqa: E402
from estimate_heights_from_ps import search_bounds  # noqa: E402
from estimate_heights_by_triangle_adjustment import (  # noqa: E402
    robust_adjustment,
    robust_adjustment_with_wall_bias,
)
from recompute_iterative_local_registration import optimize_mask  # noqa: E402
from recompute_rooftop_registration import (  # noqa: E402
    evaluate_rooftop_grid,
    finalize_grid,
    read_rooftop_features,
    select_scene_consensus,
)
from recompute_hybrid_rooftop_registration import merge_reference_registration  # noqa: E402
from run_iterative_triangle_adjustment import surface_stability, update_heights  # noqa: E402
from map_ps_to_building_surfaces import (  # noqa: E402
    rasterize_projected_triangles,
    refine_projected_mask,
)
from run_highrise_envelope_optimization import (  # noqa: E402
    fit_asymmetric_top_calibration,
    predict_top_calibration,
)


class GeometryTests(unittest.TestCase):
    def test_asymmetric_top_calibration_penalizes_underestimation(self) -> None:
        table = pd.DataFrame(
            {
                "height_tail_q95_m": [20, 25, 30, 35, 40, 45, 50, 55],
                "height_est_m": [10, 12, 15, 18, 20, 23, 25, 28],
                "height_reference_m": [35, 38, 42, 47, 50, 55, 62, 75],
            }
        )
        symmetric = fit_asymmetric_top_calibration(
            table, underestimation_penalty=1.0, ridge=1.0
        )
        top_weighted = fit_asymmetric_top_calibration(
            table, underestimation_penalty=4.0, ridge=1.0
        )
        symmetric_prediction = predict_top_calibration(table, symmetric)
        top_prediction = predict_top_calibration(table, top_weighted)
        self.assertTrue(np.all(np.isfinite(top_prediction)))
        self.assertGreaterEqual(
            float(np.mean(top_prediction)),
            float(np.mean(symmetric_prediction)),
        )
        self.assertLess(
            float(
                np.maximum(
                    table.height_reference_m.to_numpy() - top_prediction,
                    0.0,
                ).mean()
            ),
            float(
                np.maximum(
                    table.height_reference_m.to_numpy()
                    - symmetric_prediction,
                    0.0,
                ).mean()
            ),
        )

    def test_amplitude_refinement_stays_inside_projected_model(self) -> None:
        triangles = [
            np.asarray([[3.0, 3.0], [12.0, 3.0], [12.0, 12.0]]),
            np.asarray([[3.0, 3.0], [12.0, 12.0], [3.0, 12.0]]),
        ]
        amplitude = np.full((16, 16), 0.1, dtype=np.float32)
        amplitude[5:10, 6:10] = 0.9
        refined, stats = refine_projected_mask(
            amplitude,
            {"roof": triangles},
            {
                "surfaces": ["roof"],
                "background_gap_px": 1,
                "background_buffer_px": 2,
                "threshold_sigma": 0.0,
                "closing_iterations": 0,
                "minimum_component_pixels": 2,
                "minimum_component_fraction_of_largest": 0.0,
                "maximum_components_per_surface": 1,
                "minimum_fallback_pixels": 0,
            },
        )
        initial = rasterize_projected_triangles(triangles, amplitude.shape)
        self.assertTrue(np.any(refined))
        self.assertFalse(np.any(refined & ~initial))
        self.assertTrue(np.all(amplitude[refined] > stats["threshold"]))
        self.assertLess(stats["refined_pixels"], stats["initial_pixels"])

    def test_barycentric_round_trip(self) -> None:
        triangle = np.asarray([[1.0, 2.0], [7.0, 3.0], [2.5, 9.0]])
        expected = np.asarray([0.2, 0.35, 0.45])
        point = expected @ triangle
        actual, violation = barycentric_weights(point, triangle)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertLessEqual(violation, 1e-12)
        self.assertAlmostEqual(float(actual.sum()), 1.0)

    def test_llh_ecef_round_trip(self) -> None:
        llh = np.asarray(
            [
                [121.4907, 31.2780, 4.0],
                [121.5050, 31.2850, 106.0],
            ]
        )
        ecef = llh_to_ecef(llh[:, 0], llh[:, 1], llh[:, 2])
        recovered = ecef_to_llh(ecef)
        np.testing.assert_allclose(recovered, llh, atol=1e-6)

    def test_mesh_can_shift_roof_without_shifting_bottom(self) -> None:
        class DummyProjector(StrictRadarProjector):
            def project_ecef(self, points_ecef: np.ndarray) -> np.ndarray:
                return points_ecef[:, :2] * 1e-6

        projector = object.__new__(DummyProjector)
        ring = np.asarray(
            [
                [121.4900, 31.2800],
                [121.4901, 31.2800],
                [121.4901, 31.2801],
                [121.4900, 31.2801],
            ]
        )
        common = projector.build_mesh(ring, 4.0, 20.0, 34.0, -1.0)
        roof_only = projector.build_mesh(
            ring,
            4.0,
            20.0,
            34.0,
            -1.0,
            top_row_shift_px=36.0,
            top_col_shift_px=2.0,
        )
        n = len(common.projected_xy) // 2
        np.testing.assert_allclose(roof_only.projected_xy[:n], common.projected_xy[:n])
        np.testing.assert_allclose(
            roof_only.projected_xy[n:] - common.projected_xy[n:],
            np.tile(np.asarray([3.0, 2.0]), (n, 1)),
        )

    def test_prior_only_defines_broad_search_bounds(self) -> None:
        config = {
            "minimum_height_m": 3.0,
            "maximum_height_m": 150.0,
            "prior_lower_factor": 0.35,
            "prior_upper_factor": 2.0,
            "prior_upper_margin_m": 12.0,
            "prior_minimum_span_above_m": 25.0,
        }
        self.assertEqual(search_bounds(24.0, config), (8, 60))
        self.assertEqual(search_bounds(102.0, config), (35, 150.0))

    def test_triangle_height_adjustment_recovers_known_height_with_outlier(self) -> None:
        fraction = np.asarray([1.0, 0.8, 0.5, 0.25, 0.6])
        true_height = 30.0
        observed = fraction * true_height + np.asarray([0.2, -0.1, 0.1, 0.0, 20.0])
        estimate, _, weights, _ = robust_adjustment(fraction, observed, np.ones_like(fraction))
        self.assertAlmostEqual(estimate, true_height, delta=1.0)
        self.assertLess(weights[-1], weights[0])

    def test_quality_adjustment_separates_wall_bias_from_height(self) -> None:
        fraction = np.asarray([1.0, 1.0, 0.80, 0.55, 0.30, 0.65])
        is_wall = np.asarray([False, False, True, True, True, True])
        true_height = 28.0
        true_wall_bias = 2.5
        observed = (
            fraction * true_height
            + is_wall.astype(float) * true_wall_bias
            + np.asarray([0.1, -0.1, 0.2, -0.2, 0.1, 18.0])
        )
        height, wall_bias, _, weights, _ = robust_adjustment_with_wall_bias(
            fraction,
            observed,
            np.ones_like(fraction),
            is_wall,
            estimate_wall_bias=True,
        )
        self.assertAlmostEqual(height, true_height, delta=1.0)
        self.assertAlmostEqual(wall_bias, true_wall_bias, delta=1.5)
        self.assertLess(weights[-1], weights[2])

    def test_iterative_height_update_is_damped_and_limited(self) -> None:
        import pandas as pd

        current = pd.DataFrame({"fid": [0, 1], "clean_id": [10, 11], "height_current_m": [20.0, 20.0]})
        estimates = pd.DataFrame(
            {
                "fid": [0, 1],
                "height_est_m": [30.0, 100.0],
                "quality": ["high", "low"],
                "height_uncertainty_m": [1.0, 8.0],
                "ps_equations_used": [8, 2],
                "weighted_residual_rms_m": [2.0, 9.0],
            }
        )
        cfg = {
            "quality_damping": {"high": 0.7, "medium": 0.5, "low": 0.25},
            "maximum_height_change_per_iteration_m": 8.0,
            "minimum_height_m": 3.0,
            "maximum_height_m": 150.0,
        }
        updated, history = update_heights(current, estimates, cfg)
        np.testing.assert_allclose(updated.height_current_m, [27.0, 28.0])
        np.testing.assert_allclose(history.height_change_m, [7.0, 8.0])

    def test_surface_stability_requires_same_building_and_surface(self) -> None:
        import pandas as pd

        previous = pd.DataFrame({"ps_id": [1, 2], "fid": [5, 5], "surface": ["roof", "wall"]})
        current = pd.DataFrame({"ps_id": [1, 2], "fid": [5, 6], "surface": ["roof", "wall"]})
        result = surface_stability(previous, current)
        self.assertEqual(result["common_ps"], 2)
        self.assertAlmostEqual(result["same_building_fraction"], 0.5)
        self.assertAlmostEqual(result["same_surface_fraction"], 0.5)

    def test_feature_registration_recovers_known_row_shift(self) -> None:
        from scipy.ndimage import gaussian_filter, sobel

        mask = np.zeros((64, 64), dtype=bool)
        mask[22:38, 20:44] = True
        amplitude = np.zeros(mask.shape, dtype=np.float32)
        amplitude[25:41, 20:44] = 1.0
        smoothed = gaussian_filter(amplitude, sigma=0.8)
        gradient_x = sobel(smoothed, axis=1, mode="reflect") / 8.0
        gradient_y = sobel(smoothed, axis=0, mode="reflect") / 8.0
        edges = np.hypot(gradient_x, gradient_y)
        scale = max(float(np.percentile(edges[edges > 0], 98)), 1e-6)
        edges = np.clip(edges / scale, 0.0, 1.0)
        gradient_x /= scale
        gradient_y /= scale
        result = optimize_mask(
            mask,
            amplitude,
            edges,
            gradient_x,
            gradient_y,
            max_shift=5,
            coarse_step=2,
            min_score_gain=1.0,
            search_col=False,
            weights={
                "amplitude_contrast": 80.0,
                "boundary_edge_contrast": 45.0,
                "normal_edge_response": 35.0,
                "bright_fraction_contrast": 25.0,
                "shift_penalty": 0.02,
                "minimum_improved_feature_count": 2,
                "reject_search_boundary": True,
            },
        )
        self.assertEqual(result["candidate_row_shift"], 3)
        self.assertEqual(result["applied_row_shift"], 3)
        self.assertEqual(result["accepted"], 1)

    def test_rooftop_registration_recovers_known_two_dimensional_shift(self) -> None:
        from scipy.ndimage import gaussian_filter, maximum_filter, sobel

        mask = np.zeros((72, 72), dtype=bool)
        mask[24:42, 20:48] = True
        shifted = np.zeros_like(mask, dtype=np.float32)
        shifted[27:45, 22:50] = 1.0
        amplitude = gaussian_filter(shifted, sigma=0.7)
        gx = sobel(amplitude, axis=1, mode="reflect") / 8.0
        gy = sobel(amplitude, axis=0, mode="reflect") / 8.0
        edges = np.hypot(gx, gy)
        features = {
            "amplitude": amplitude,
            "edges": edges,
            "gradient_x": gx,
            "gradient_y": gy,
            "roof_likelihood": amplitude,
            "texture": edges,
            "edge_proximity": maximum_filter(edges, size=5),
        }
        weights = {
            "top_region_contrast_weight": 0.30,
            "top_support_contrast_weight": 0.25,
            "roof_boundary_support_weight": 0.20,
            "normal_edge_response_weight": 0.15,
            "top_texture_contrast_weight": 0.10,
            "shift_penalty": 0.05,
            "support_quantile": 0.62,
            "minimum_roof_pixels": 24,
            "local_refinement_radius_px": 2,
            "minimum_local_score_gain": 0.35,
            "minimum_total_score_gain": 0.50,
            "minimum_peak_margin": 0.10,
            "minimum_improved_feature_count": 3,
        }
        grid = evaluate_rooftop_grid(mask, features, max_shift=6, weights=weights)
        scene_dr, scene_dc, scene_score = select_scene_consensus([grid], max_shift=6)
        result = finalize_grid(grid, (scene_dr, scene_dc), scene_score, weights)
        self.assertEqual((scene_dr, scene_dc), (3, 2))
        self.assertEqual(
            (result["applied_row_shift"], result["applied_col_shift"]),
            (3, 2),
        )

    def test_multiscene_reference_gate_rejects_large_shift(self) -> None:
        import pandas as pd

        base = pd.DataFrame(
            {
                "fid": [0, 1],
                "applied_row_shift": [0, 0],
                "applied_col_shift": [0, 0],
                "accepted": [0, 0],
                "local_refinement_accepted": [0, 0],
                "registration_feature_mode": ["v3", "v3"],
            }
        )
        reference = pd.DataFrame(
            {
                "fid": [0, 1],
                "dx": [1.25, 8.0],
                "dy": [-0.5, 0.0],
                "registration_accepted": [1, 1],
                "score_margin": [0.4, 0.4],
                "gain_oriented_edge": [0.2, 0.2],
                "gain_continuity": [0.2, 0.2],
                "pair_distance_px": [1.0, 1.0],
                "fused_to_pair_px": [1.0, 1.0],
                "shape_strategy": ["regular", "regular"],
                "registration_quality": ["accepted", "accepted"],
            }
        )
        config = {
            "iterative_adjustment": {
                "registration_reference_v4": {
                    "maximum_reference_shift_px": 6.0,
                    "minimum_score_margin": 0.2,
                    "minimum_oriented_edge_gain": 0.0,
                    "minimum_continuity_gain": 0.0,
                    "maximum_scene_pair_distance_px": 3.0,
                    "maximum_fused_to_pair_distance_px": 2.0,
                }
            }
        }
        result = merge_reference_registration(base, reference, config)
        self.assertEqual(int(result.reference_multiscene_override.sum()), 1)
        self.assertAlmostEqual(float(result.loc[0, "applied_row_shift"]), -0.5)
        self.assertAlmostEqual(float(result.loc[0, "applied_col_shift"]), 1.25)
        self.assertEqual(int(result.loc[1, "accepted"]), 0)

    def test_height_consistent_shift_decomposition(self) -> None:
        from reparameterize_height_consistent_projection import decompose_shift

        height_change, perpendicular = decompose_shift(
            np.asarray([4.0, 3.0]),
            np.asarray([2.0, 0.0]),
        )
        self.assertAlmostEqual(height_change, 2.0)
        np.testing.assert_allclose(perpendicular, np.asarray([0.0, 3.0]))
        self.assertAlmostEqual(float(np.dot(perpendicular, [2.0, 0.0])), 0.0)


if __name__ == "__main__":
    unittest.main()
