import unittest
from unittest.mock import patch

import numpy as np

from main import draw_landmarks


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


if __name__ == "__main__":
    unittest.main()
