import unittest

import numpy as np
from scipy.signal import convolve2d

from p2.filters import convolve_four_loops, convolve_two_loops


class ConvolutionTests(unittest.TestCase):
    functions = (convolve_four_loops, convolve_two_loops)

    def test_matches_scipy(self):
        image = np.arange(20).reshape(4, 5)
        kernel = np.array([[1, 2, 0], [-1, 3, 1], [0, 2, -2]])
        expected = convolve2d(image, kernel, mode="same", boundary="fill")
        for fn in self.functions:
            np.testing.assert_allclose(fn(image, kernel), expected)

    def test_derivative_direction_and_padding(self):
        image = np.array([[1, 3, 6], [2, 5, 9]], dtype=np.uint8)
        dx = np.array([[1, -1]])
        dy = dx.T
        for fn in self.functions:
            np.testing.assert_array_equal(fn(image, dx), [[1, 2, 3], [2, 3, 4]])
            np.testing.assert_array_equal(fn(image, dy), [[1, 3, 6], [1, 2, 3]])

if __name__ == "__main__":
    unittest.main()
