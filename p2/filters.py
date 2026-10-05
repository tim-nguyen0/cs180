"""Filters adapted from p1/pyramid.py. Inputs should be normalized floats."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def gaussian_blur_5(image: np.ndarray, *, boundary: str = "symm") -> np.ndarray:
    """5x5 binomial blur for grayscale or RGB. Padding: 'symm' or zero 'fill'."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError("Expected a grayscale (H, W) or RGB (H, W, 3) image")
    if min(image.shape[:2]) == 0:
        raise ValueError("Image dimensions must be nonempty")
    if boundary not in ("fill", "symm"):
        raise ValueError("boundary must be 'fill' or 'symm'")

    binomial = np.array([1, 4, 6, 4, 1], dtype=np.float64)
    kernel = np.outer(binomial, binomial) / 256.0
    padding = ((2, 2), (2, 2))
    if image.ndim == 3:
        padding += ((0, 0),)
    padded = np.pad(
        image, padding, mode="constant" if boundary == "fill" else "symmetric"
    )
    windows = sliding_window_view(padded, (5, 5), axis=(0, 1))
    return np.einsum("...ij,ij->...", windows, kernel)


def high_pass(image: np.ndarray, *, boundary: str = "symm") -> np.ndarray:
    """Subtract the blurred image from the original."""
    image = np.asarray(image, dtype=np.float64)
    return image - gaussian_blur_5(image, boundary=boundary)


def band_pass(
    image: np.ndarray, low: int = 2, high: int = 5, *, boundary: str = "symm"
) -> np.ndarray:
    """Difference of two blurs; low and high are blur counts, not sigmas."""
    if not isinstance(low, (int, np.integer)) or not isinstance(high, (int, np.integer)):
        raise ValueError("low and high must be integer blur counts")
    if not 0 <= low < high:
        raise ValueError("Blur counts must satisfy 0 <= low < high")

    lower = np.asarray(image, dtype=np.float64)
    for _ in range(low):
        lower = gaussian_blur_5(lower, boundary=boundary)
    upper = lower
    for _ in range(high - low):
        upper = gaussian_blur_5(upper, boundary=boundary)
    return lower - upper
