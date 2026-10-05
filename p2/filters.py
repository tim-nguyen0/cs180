"""filters from p1, expects normalized floats"""
import numpy as np
import cv2
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import convolve2d

def _prepare_convolution(image: np.ndarray, kernel: np.ndarray):
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("Image and kernel must be 2-D")
    if min(image.shape) == 0 or min(kernel.shape) == 0:
        raise ValueError("Image and kernel must be nonempty")

    kh, kw = kernel.shape
    # extra pad on top/left for even kernels
    padding = ((kh // 2, (kh - 1) // 2), (kw // 2, (kw - 1) // 2))
    padded = np.pad(image, padding, mode="constant")
    return padded, np.flip(kernel, axis=(0, 1)), np.zeros_like(image)


def convolve_four_loops(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    padded, kernel, out = _prepare_convolution(image, kernel)
    kh, kw = kernel.shape
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            for m in range(kh):
                for n in range(kw):
                    out[i, j] += padded[i + m, j + n] * kernel[m, n]
    return out


def convolve_two_loops(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    padded, kernel, out = _prepare_convolution(image, kernel)
    kh, kw = kernel.shape
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(padded[i:i + kh, j:j + kw] * kernel)
    return out


def gaussian_kernel(sigma=2.0, size=None):
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if size is None:
        size = 2 * int(np.ceil(3 * sigma)) + 1
    if size < 1 or size % 2 == 0:
        raise ValueError("size must be positive and odd")
    g = cv2.getGaussianKernel(size, sigma)
    return g @ g.T


def filter_image(image, kernel, boundary="symm"):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        return convolve2d(image, kernel, mode="same", boundary=boundary)
    return np.stack([
        convolve2d(image[..., c], kernel, mode="same", boundary=boundary)
        for c in range(image.shape[2])
    ], axis=-1)


def gaussian_blur(image, sigma=2.0, size=None, boundary="symm"):
    g = gaussian_kernel(sigma, size).sum(axis=0)
    return filter_image(filter_image(image, g[None, :], boundary), g[:, None], boundary)


def gaussian_blur_5(image: np.ndarray, *, boundary: str = "symm") -> np.ndarray:
    """5x5 blur, works on grayscale and RGB"""
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
    image = np.asarray(image, dtype=np.float64)
    return image - gaussian_blur_5(image, boundary=boundary)


def band_pass(
    image: np.ndarray, low: int = 2, high: int = 5, *, boundary: str = "symm"
) -> np.ndarray:
    """low/high are number of blurs, not sigma"""
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
