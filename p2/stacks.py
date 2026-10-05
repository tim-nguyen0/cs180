import numpy as np
from filters import gaussian_blur


def gaussian_stack(image, num_bands=5, sigma=2.0):
    stack = [np.asarray(image, dtype=np.float64)]
    for i in range(num_bands):
        stack.append(gaussian_blur(stack[-1], sigma * 2**i))
    return stack


def laplacian_stack(gaussian):
    stack = [gaussian[i] - gaussian[i + 1] for i in range(len(gaussian) - 1)]
    stack.append(gaussian[-1])
    return stack
