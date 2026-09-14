import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as sliding_window

def mirror_pad_2(image: np.array):
    rows, cols = image.shape

    padded = np.zeros((rows+4, cols+4))
    padded[2:-2,2:-2] = image

    padded[0:2,2:-2] = image[2:0:-1, :] # top pad
    padded[-2: , 2:-2] = image[-2:-4:-1, :] # bottom pad
    padded[:, 0:2] = padded[:, 4:2:-1] # left padding (using already padded)
    padded[:, -2: ] = padded[:, -4:-6:-1] # right padding

    return padded

def downsample_half(image: np.array):

    padded = mirror_pad_2(image)
    windows = sliding_window(padded, (5,5))

    halved_windows = windows[::2,::2]

    binom_vec_5 = np.array([1, 4, 6, 4, 1])

    gaussian_kernel_5 = 1/256*np.outer(binom_vec_5,binom_vec_5)

    downsampled = np.sum(halved_windows * gaussian_kernel_5, axis=(2, 3))

    return downsampled

def image_pyramid(image: np.array, depth: int) -> list:
    """Return an image pyramid (list) of depth (depth) from (0) coarsest to (depth) finest"""

    pyramid = []
    pyramid.append(image)
    
    for i in range(depth-1):
        pyramid.insert(0,downsample_half(pyramid[0]))
    return pyramid

def auto_pyramid(image: np.array, target_max_dim: int=500)->list:
    """Returns a pyramid with coarsest image less target_max_dim"""
    depth = np.ceil(np.log2(max(image.shape)/target_max_dim))
    return image_pyramid(image, depth)
        






