import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import cv2

from main import (
    draw_landmarks,
    eye_aspect_ratio,
    extract_iris_centers,
    fit_calibration,
    calibration_acceptance_passed,
    calibration_config_status,
    calibration_sample_rejection_reason,
    countdown_number,
    FACE_LANDMARK_INDICES,
    FACE_MODEL_POINTS,
    estimate_head_pose,
    bad_calibration_dots,
    gaze_is_steady,
    gaze_standard_deviation,
    mean_2d_error,
    next_dot_retry_count,
    median_head_pose,
    normalized_gaze,
    per_dot_residuals,
    predict_screen_position,
    relative_head_pose,
    save_calibration,
    verification_mean_error,
    verification_passed,
    verification_points,
)


class DrawLandmarksTests(unittest.TestCase):
    def test_draws_all_478_landmark_points(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        points = [(0.5, 0.5)] * 478

        with patch("main.cv2.circle") as draw_circle:
            draw_landmarks(frame, points)

        self.assertEqual(draw_circle.call_count, 478)

    def test_clamps_partial_out_of_frame_landmarks(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        with patch("main.cv2.circle") as draw_circle:
            draw_landmarks(frame, [(-0.1, -0.1), (1.1, 1.1)])

        coordinates = [call.args[1] for call in draw_circle.call_args_list]
        self.assertEqual(coordinates, [(0, 0), (199, 99)])


class IrisGazeTests(unittest.TestCase):
    @staticmethod
    def make_points() -> list[tuple[float, float]]:
        points = [(0.0, 0.0)] * 478
        points[33] = (0.30, 0.40)
        points[133] = (0.40, 0.40)
        points[159] = (0.35, 0.39)
        points[145] = (0.35, 0.41)
        points[468] = (0.35, 0.40)
        points[362] = (0.60, 0.40)
        points[263] = (0.70, 0.40)
        points[386] = (0.65, 0.39)
        points[374] = (0.65, 0.41)
        points[473] = (0.65, 0.40)
        return points

    def test_extracts_refined_iris_centers(self) -> None:
        points = self.make_points()

        self.assertEqual(
            extract_iris_centers(points),
            ((0.35, 0.40), (0.65, 0.40)),
        )

    def test_returns_centered_normalized_gaze(self) -> None:
        gaze = normalized_gaze(self.make_points(), (640, 480))

        self.assertIsNotNone(gaze)
        self.assertAlmostEqual(gaze[0], 0.5)
        self.assertAlmostEqual(gaze[1], 0.5)

    def test_gaze_changes_consistently_with_iris_direction(self) -> None:
        center_points = self.make_points()
        left_points = self.make_points()
        left_points[468] = (0.33, 0.40)
        left_points[473] = (0.63, 0.40)
        up_points = self.make_points()
        up_points[468] = (0.35, 0.395)
        up_points[473] = (0.65, 0.395)

        center_gaze = normalized_gaze(center_points, (640, 480))
        left_gaze = normalized_gaze(left_points, (640, 480))
        up_gaze = normalized_gaze(up_points, (640, 480))

        self.assertIsNotNone(center_gaze)
        self.assertIsNotNone(left_gaze)
        self.assertIsNotNone(up_gaze)
        self.assertLess(left_gaze[0], center_gaze[0])
        self.assertLess(up_gaze[1], center_gaze[1])

    def test_uses_open_eye_when_other_eye_is_closed(self) -> None:
        points = self.make_points()
        points[159] = (0.35, 0.40)
        points[145] = (0.35, 0.40)

        gaze = normalized_gaze(points, (640, 480))

        self.assertIsNotNone(gaze)
        self.assertAlmostEqual(gaze[0], 0.5)
        self.assertAlmostEqual(gaze[1], 0.5)

    def test_returns_no_gaze_when_iris_landmarks_are_missing(self) -> None:
        self.assertIsNone(normalized_gaze([(0.0, 0.0)] * 468, (640, 480)))


class CalibrationTests(unittest.TestCase):
    def test_sample_rejection_explains_ear_and_head_pose_thresholds(self) -> None:
        reason = calibration_sample_rejection_reason(
            (0.5, 0.5),
            0.22,
            (17.0, -16.0),
            (0.0, 0.0),
        )

        self.assertEqual(
            reason,
            "EAR 0.22 <= 0.25; yaw +17.0 >= 15 deg; pitch -16.0 <= -15 deg",
        )

    def test_sample_is_accepted_when_all_locked_filters_pass(self) -> None:
        self.assertIsNone(
            calibration_sample_rejection_reason(
                (0.5, 0.5), 0.30, (2.0, -3.0), (0.0, 0.0)
            )
        )

    def test_head_pose_filter_uses_neutral_baseline(self) -> None:
        baseline = (24.0, -18.0)
        pose = (26.0, -15.0)

        self.assertEqual(relative_head_pose(pose, baseline), (2.0, 3.0))
        self.assertIsNone(
            calibration_sample_rejection_reason(
                (0.5, 0.5), 0.30, pose, baseline
            )
        )

    def test_neutral_pose_uses_median_and_requires_ten_readings(self) -> None:
        samples = [(float(index), float(index + 2)) for index in range(10)]

        self.assertIsNone(median_head_pose(samples[:9]))
        self.assertEqual(median_head_pose(samples), (4.5, 6.5))

    def test_eye_aspect_ratio_uses_both_eyes(self) -> None:
        points = [(0.0, 0.0)] * 478
        for corner_a, corner_b, upper_a, lower_a, upper_b, lower_b in (
            (33, 133, 160, 144, 158, 153),
            (362, 263, 385, 380, 387, 373),
        ):
            points[corner_a] = (0.3, 0.4)
            points[corner_b] = (0.4, 0.4)
            points[upper_a] = (0.35, 0.39)
            points[lower_a] = (0.35, 0.41)
            points[upper_b] = (0.35, 0.39)
            points[lower_b] = (0.35, 0.41)

        self.assertAlmostEqual(eye_aspect_ratio(points, (100, 100)), 0.2)

    def test_head_pose_reports_neutral_and_rotated_angles(self) -> None:
        width, height = 640, 480
        camera_matrix = np.array(
            [
                [width, 0, width / 2],
                [0, width, height / 2],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        translation = np.array([[0.0], [0.0], [1000.0]])
        points = [(0.5, 0.5)] * 478

        projected, _ = cv2.projectPoints(
            FACE_MODEL_POINTS,
            np.zeros((3, 1)),
            translation,
            camera_matrix,
            np.zeros((4, 1)),
        )
        for index, (x, y) in zip(
            FACE_LANDMARK_INDICES, projected.reshape(-1, 2)
        ):
            points[index] = (float(x / width), float(y / height))
        neutral_pose = estimate_head_pose(points, (width, height))

        rotation, _ = cv2.Rodrigues(
            np.deg2rad(np.array([[8.0], [10.0], [0.0]]))
        )
        rotated_vector, _ = cv2.Rodrigues(rotation)
        projected, _ = cv2.projectPoints(
            FACE_MODEL_POINTS,
            rotated_vector,
            translation,
            camera_matrix,
            np.zeros((4, 1)),
        )
        for index, (x, y) in zip(
            FACE_LANDMARK_INDICES, projected.reshape(-1, 2)
        ):
            points[index] = (float(x / width), float(y / height))
        rotated_pose = estimate_head_pose(points, (width, height))

        self.assertIsNotNone(neutral_pose)
        self.assertAlmostEqual(neutral_pose[0], 0.0, delta=0.1)
        self.assertAlmostEqual(neutral_pose[1], 0.0, delta=0.1)
        self.assertIsNotNone(rotated_pose)
        self.assertAlmostEqual(rotated_pose[0], 10.0, delta=0.2)
        self.assertAlmostEqual(rotated_pose[1], 8.0, delta=0.2)

    def test_least_squares_recovers_quadratic_calibration(self) -> None:
        gaze_samples = [
            (gx, gy)
            for gy in (0.2, 0.5, 0.8)
            for gx in (0.2, 0.5, 0.8)
        ]
        x_expected = [100, 800, 0, 0, 0, 0]
        y_expected = [200, 0, 700, 0, 0, 0]
        screen_samples = [
            (
                x_expected[0] + x_expected[1] * gx,
                y_expected[0] + y_expected[2] * gy,
            )
            for gx, gy in gaze_samples
        ]

        x_actual, y_actual = fit_calibration(gaze_samples, screen_samples)

        np.testing.assert_allclose(x_actual, x_expected, atol=1e-9)
        np.testing.assert_allclose(y_actual, y_expected, atol=1e-9)

    def test_rejects_rank_deficient_calibration_samples(self) -> None:
        gaze_samples = [(0.5, 0.5)] * 9
        screen_samples = [(100.0, 100.0)] * 9

        with self.assertRaisesRegex(ValueError, "do not span the screen"):
            fit_calibration(gaze_samples, screen_samples)

    def test_accuracy_uses_mean_2d_error_normalized_by_screen_diagonal(self) -> None:
        gaze_samples = [
            (gx, gy)
            for gy in (0.2, 0.5, 0.8)
            for gx in (0.2, 0.5, 0.8)
        ]
        x_coefficients, y_coefficients = fit_calibration(
            gaze_samples,
            [(100 + 800 * gx, 200 + 700 * gy) for gx, gy in gaze_samples],
        )
        expected_error = 10 / np.hypot(640, 480)
        screen_samples = [
            (100 + 800 * gx + 10, 200 + 700 * gy)
            for gx, gy in gaze_samples
        ]

        error = mean_2d_error(
            gaze_samples,
            screen_samples,
            x_coefficients,
            y_coefficients,
            (640, 480),
        )

        self.assertAlmostEqual(error, expected_error)

    def test_saves_both_six_coefficient_arrays_to_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            with patch("main.CALIBRATION_CONFIG", config_path):
                save_calibration([1.0] * 6, [2.0] * 6, verified=True)

            self.assertEqual(
                json.loads(config_path.read_text(encoding="utf-8")),
                {
                    "screen_x_coefficients": [1.0] * 6,
                    "screen_y_coefficients": [2.0] * 6,
                    "verified": True,
                },
            )

    def test_config_status_distinguishes_missing_valid_and_corrupt_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            with patch("main.CALIBRATION_CONFIG", config_path):
                self.assertEqual(calibration_config_status(), "missing")
                save_calibration([1.0] * 6, [2.0] * 6, verified=True)
                self.assertEqual(calibration_config_status(), "valid")
                save_calibration([1.0] * 6, [2.0] * 6, verified=False)
                self.assertEqual(calibration_config_status(), "corrupt")
                config_path.write_text("{broken", encoding="utf-8")
                self.assertEqual(calibration_config_status(), "corrupt")

    def test_per_dot_residuals_identify_only_outlying_and_empty_dots(self) -> None:
        gaze_samples = [
            [(0.0, 0.0)],
            [(1.0, 0.0)],
            [(0.0, 1.0)],
            [],
        ]
        targets = [(0, 0), (100, 0), (200, 100), (100, 100)]
        x_coefficients = [0.0, 100.0, 0.0, 0.0, 0.0, 0.0]
        y_coefficients = [0.0, 0.0, 100.0, 0.0, 0.0, 0.0]

        residuals = per_dot_residuals(
            gaze_samples,
            targets,
            x_coefficients,
            y_coefficients,
        )

        self.assertEqual(residuals, [0.0, 0.0, 200.0, None])
        self.assertEqual(bad_calibration_dots(residuals, (640, 480)), [2, 3])

    def test_gaze_steadiness_checks_both_axis_standard_deviations(self) -> None:
        steady_samples = [(0.50, 0.50), (0.51, 0.49), (0.49, 0.51)]
        threshold_samples = [(0.0, 0.0), (0.04, 0.0)]
        unstable_x_samples = [(0.47, 0.50), (0.50, 0.50), (0.53, 0.50)]
        unstable_y_samples = [(0.50, 0.47), (0.50, 0.50), (0.50, 0.53)]

        deviations = gaze_standard_deviation(steady_samples)

        self.assertIsNotNone(deviations)
        self.assertLessEqual(deviations[0], 0.02)
        self.assertLessEqual(deviations[1], 0.02)
        self.assertTrue(gaze_is_steady(steady_samples))
        self.assertTrue(gaze_is_steady(threshold_samples))
        self.assertFalse(gaze_is_steady(unstable_x_samples))
        self.assertFalse(gaze_is_steady(unstable_y_samples))
        self.assertIsNone(gaze_standard_deviation([]))
        self.assertFalse(gaze_is_steady([]))

    def test_dot_retry_count_allows_three_retries_then_stops(self) -> None:
        self.assertEqual(next_dot_retry_count(0), 1)
        self.assertEqual(next_dot_retry_count(1), 2)
        self.assertEqual(next_dot_retry_count(2), 3)
        self.assertIsNone(next_dot_retry_count(3))

    def test_countdown_number_changes_once_per_second(self) -> None:
        self.assertEqual(countdown_number(0.0), 3)
        self.assertEqual(countdown_number(0.99), 3)
        self.assertEqual(countdown_number(1.0), 2)
        self.assertEqual(countdown_number(1.99), 2)
        self.assertEqual(countdown_number(2.0), 1)

    def test_prediction_applies_fitted_quadratic_coefficients(self) -> None:
        position = predict_screen_position(
            (0.5, 0.25),
            [10, 100, 20, 0, 0, 0],
            [5, 0, 200, 0, 0, 0],
        )

        self.assertEqual(position, (65.0, 55.0))

    def test_verification_error_pass_threshold_and_missing_data(self) -> None:
        self.assertEqual(verification_mean_error([3.0, 6.0, 9.0]), 6.0)
        self.assertIsNone(verification_mean_error([]))
        self.assertTrue(verification_passed(50.0, (640, 480)))
        self.assertFalse(verification_passed(100.0, (640, 480)))
        self.assertFalse(verification_passed(None, (640, 480)))
        self.assertTrue(
            calibration_acceptance_passed([True] * 9, 50.0, (640, 480))
        )
        self.assertFalse(
            calibration_acceptance_passed(
                [True] * 8 + [False], 50.0, (640, 480)
            )
        )

    def test_verification_points_are_inset_from_screen_edges(self) -> None:
        with patch(
            "main.random.uniform",
            side_effect=lambda lower, upper: (lower + upper) / 2,
        ):
            points = verification_points((640, 480))

        self.assertEqual(len(points), 5)
        self.assertTrue(all(64 <= x <= 576 for x, _ in points))
        self.assertTrue(all(48 <= y <= 432 for _, y in points))


if __name__ == "__main__":
    unittest.main()
