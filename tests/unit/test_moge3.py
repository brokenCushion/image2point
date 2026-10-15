import unittest

import numpy as np

from image2point.moge3 import camera_points


class MoGe3Tests(unittest.TestCase):
    def test_camera_points_use_z_depth(self):
        k = np.array([[2.0, 0, 0.5], [0, 2.0, 0.5], [0, 0, 1.0]])
        points = camera_points(np.full((2, 3), 4.0), k)
        np.testing.assert_allclose(points[1, 2], [4, 2, 4])

    def test_invalid_depth_does_not_warn(self):
        k = np.array([[2.0, 0, 0.5], [0, 2.0, 0.5], [0, 0, 1.0]])
        with np.errstate(all="raise"):
            # Pixel (0,0) has a zero ray component: 0 * inf is NaN, not an error.
            points = camera_points(np.full((2, 3), np.inf), k)
        self.assertTrue(np.isnan(points[0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
