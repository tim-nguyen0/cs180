import numpy as np
from scipy.signal import convolve2d


def finite_difference(image, threshold=0.2, boundary="symm"):
    image = np.asarray(image, dtype=np.float64)
    dx = np.array([[1, -1]])
    dy = dx.T
    ix = convolve2d(image, dx, mode="same", boundary=boundary)
    iy = convolve2d(image, dy, mode="same", boundary=boundary)
    magnitude = np.sqrt(ix**2 + iy**2)
    return ix, iy, magnitude, magnitude > threshold
