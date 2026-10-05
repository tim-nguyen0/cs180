import unittest
import numpy as np
from stacks import gaussian_stack, laplacian_stack


class StackTests(unittest.TestCase):
    def test_reconstruction(self):
        image = np.random.default_rng(180).random((12, 16, 3))
        gaussian = gaussian_stack(image, num_bands=3)
        laplacian = laplacian_stack(gaussian)
        self.assertTrue(all(level.shape == image.shape for level in gaussian + laplacian))
        np.testing.assert_allclose(np.sum(laplacian, axis=0), image, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
