import unittest
import numpy as np
from stacks import gaussian_stack, laplacian_stack
from blend import multiresolution_blend


class StackTests(unittest.TestCase):
    def test_reconstruction(self):
        image = np.random.default_rng(180).random((12, 16, 3))
        gaussian = gaussian_stack(image, num_bands=3)
        laplacian = laplacian_stack(gaussian)
        self.assertTrue(all(level.shape == image.shape for level in gaussian + laplacian))
        np.testing.assert_allclose(np.sum(laplacian, axis=0), image, atol=1e-14)

    def test_blend_same_image(self):
        image = np.random.default_rng(180).random((12, 16, 3))
        mask = np.zeros(image.shape[:2])
        mask[:, :8] = 1
        result, _, _, _ = multiresolution_blend(image, image, mask, num_bands=3)
        np.testing.assert_allclose(result, image, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
