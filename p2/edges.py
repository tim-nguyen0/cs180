import numpy as np
from scipy.signal import convolve2d

from filters import gaussian_kernel


def finite_difference(image, threshold=0.2, boundary="symm"):
    image = np.asarray(image, dtype=np.float64)
    dx = np.array([[1, -1]])
    dy = dx.T
    ix = convolve2d(image, dx, mode="same", boundary=boundary)
    iy = convolve2d(image, dy, mode="same", boundary=boundary)
    magnitude = np.sqrt(ix**2 + iy**2)
    return ix, iy, magnitude, magnitude > threshold


def dog_filters(sigma=2.0, size=None):
    kernel = gaussian_kernel(sigma, size)
    dx = np.array([[1, -1]])
    # full keeps the whole combined filter
    return convolve2d(kernel, dx, mode="full"), convolve2d(kernel, dx.T, mode="full")


def derivative_of_gaussian(image, sigma=2.0, size=None, threshold=0.04, boundary="symm"):
    image = np.asarray(image, dtype=np.float64)
    kx, ky = dog_filters(sigma, size)
    ix = convolve2d(image, kx, mode="same", boundary=boundary)
    iy = convolve2d(image, ky, mode="same", boundary=boundary)
    magnitude = np.sqrt(ix**2 + iy**2)
    return ix, iy, magnitude, magnitude > threshold
