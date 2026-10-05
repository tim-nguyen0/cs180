import numpy as np

from filters import gaussian_blur, gaussian_kernel


def unsharp_mask(image, sigma=2.0, size=None, alpha=1.0, boundary="symm"):
    image = np.asarray(image, dtype=np.float64)
    low = gaussian_blur(image, sigma, size, boundary)
    high = image - low
    return low, high, image + alpha * high


def unsharp_kernel(sigma=2.0, size=None, alpha=1.0):
    gaussian = gaussian_kernel(sigma, size)
    impulse = np.zeros_like(gaussian)
    impulse[gaussian.shape[0] // 2, gaussian.shape[1] // 2] = 1
    return (1 + alpha) * impulse - alpha * gaussian


def hybrid_image(image_a, image_b, sigma_low=8.0, sigma_high=4.0):
    if image_a.shape != image_b.shape:
        raise ValueError("Align and crop the images first")
    low = gaussian_blur(image_a, sigma_low)
    high = image_b - gaussian_blur(image_b, sigma_high)
    return low, high, low + high


def fourier_magnitude(image):
    if image.ndim == 3:
        image = image @ np.array([0.2126, 0.7152, 0.0722])
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(image))) + 1e-8)
