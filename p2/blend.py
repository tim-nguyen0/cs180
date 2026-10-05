import numpy as np
from stacks import gaussian_stack, laplacian_stack


def multiresolution_blend(image_a, image_b, mask, num_bands=5, sigma=2.0):
    if image_a.shape != image_b.shape or mask.shape != image_a.shape[:2]:
        raise ValueError("Images and mask need matching sizes")
    la = laplacian_stack(gaussian_stack(image_a, num_bands, sigma))
    lb = laplacian_stack(gaussian_stack(image_b, num_bands, sigma))
    masks = gaussian_stack(mask, num_bands, sigma)
    parts_a, parts_b = [], []
    for a, b, m in zip(la, lb, masks):
        if a.ndim == 3:
            m = m[..., None]
        parts_a.append(m * a)
        parts_b.append((1 - m) * b)
    result = np.sum(parts_a, axis=0) + np.sum(parts_b, axis=0)
    return result, parts_a, parts_b, masks
