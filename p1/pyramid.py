import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as sliding_window

def mirror_pad_2(image: np.array):
    rows, cols = image.shape

    padded = np.zeros((rows+4, cols+4),  dtype=image.dtype)
    padded[2:-2,2:-2] = image

    padded[0:2,2:-2] = image[2:0:-1, :] # top pad
    padded[-2: , 2:-2] = image[-2:-4:-1, :] # bottom pad
    padded[:, 0:2] = padded[:, 4:2:-1] # left padding (using already padded)
    padded[:, -2: ] = padded[:, -4:-6:-1] # right padding

    return padded

def gaussian_blur_5(image:np.array) -> np.array:

    binom_vec_5 = np.array([1, 4, 6, 4, 1])
    gaussian_kernel_5 = 1/256*np.outer(binom_vec_5,binom_vec_5)

    padded = mirror_pad_2(image)
    windows = sliding_window(padded, (5,5))

    return np.einsum('ijhw,hw->ij', windows, gaussian_kernel_5)


def downsample_half(image: np.array):
    return gaussian_blur_5(image)[::2, ::2]

def image_pyramid(image: np.array, depth: int, filter=None) -> list:
    """Return an image pyramid (list) of depth (depth) from (0) coarsest to (depth) finest"""

    pyramid = [image]
    
    for i in range(depth-1):
        pyramid.insert(0,downsample_half(pyramid[0]))

    if not filter:
        return pyramid

    return [filter(x) for x in pyramid]

def auto_pyramid(image: np.array, target_max_dim: int=800, filter=None)->list:
    """Returns a pyramid with coarsest image less target_max_dim"""
    depth = int(np.ceil(np.log2(max(image.shape)/target_max_dim)))
    return image_pyramid(image, max(depth, 0) + 1, filter)
        
def high_pass(image: np.array):
    return image - gaussian_blur_5((image))

def band_pass(image, low=2, high=5):
    a = image
    for i in range(low): a = gaussian_blur_5(a)
    b = a
    for i in range(high-low): b= gaussian_blur_5(b)
    return a-b







