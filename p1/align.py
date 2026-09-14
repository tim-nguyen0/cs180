import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as sliding_window
import pyramid as pyramid


def interior(img: np.array, frac: float = 0.1) -> np.array:
    """helper function to crop with"""
    h, w = img.shape
    return img[int(h*frac):int(h*(1-frac)), int(w*frac):int(w*(1-frac))]

def calculate_offset_pyramid(ref: np.array, child: np.array, initial_max_offset: int=50, step_max_offset: int=5, crop_frac: float=0.1) -> tuple:
    """Calculates offset for a larger image using image pyramid. Smallest image will be normalized to ~500 px on largest axis"""

    ref = interior(ref.astype(np.float32, copy=False), crop_frac)
    child = interior(child.astype(np.float32, copy=False), crop_frac)

    child_pyramid = pyramid.auto_pyramid(child)
    pdepth = len(child_pyramid)
    ref_pyramid= pyramid.image_pyramid(ref, pdepth)

    offset = vectorized_calculate_offset_ncc(ref_pyramid[0], child_pyramid[0], max_offset=initial_max_offset)

    for i in range(1, pdepth):
        offset = (offset[0]*2, offset[1]*2)
        offset = vectorized_calculate_offset_ncc(ref_pyramid[i], child_pyramid[i], max_offset=step_max_offset, known_offset=offset)

    return offset



def calculate_offset_ncc(ref: np.array, child: np.array, max_offset: int=100) -> tuple:

    if max_offset>=min(ref.shape):
        raise ValueError("Offset cannot be larger than image")

    rows, cols = ref.shape
    max_ncc = -np.inf
    offset = (0, 0)

    for i in range(-max_offset, max_offset):
        for j in range(-max_offset, max_offset):
            
            ref_top = max(0, i)
            ref_bottom = min(rows, rows + i)
            ref_left = max(0, j)
            ref_right = min(cols, cols + j)
            
            child_top = max(0, -i)
            child_bottom = min(rows, rows - i)
            child_left = max(0, -j)
            child_right = min(cols, cols - j)

            ref_patch = ref[ref_top:ref_bottom, ref_left:ref_right]
            child_patch = child[child_top:child_bottom, child_left:child_right]

            if ref_patch.shape[0] < (rows * 0.7) or ref_patch.shape[1] < (cols * 0.7):
                continue

            normalized_ref = normalize_matrix(ref_patch)
            normalized_child = normalize_matrix(child_patch)
            current_ncc = np.mean(normalized_ref * normalized_child)

            if current_ncc>max_ncc:
                max_ncc = current_ncc
                offset = (i,j)

    return offset

def vectorized_calculate_offset_ncc(ref_in: np.array, child: np.array, max_offset: int=5, known_offset: tuple=(0,0)) -> tuple:

    ref_top= max(0, known_offset[0])
    ref_bottom = ref_in.shape[0]+min(0, known_offset[0])
    ref_left = max(0, known_offset[1])
    ref_right = ref_in.shape[1]+min(0,known_offset[1])

    ref = ref_in[ref_top:ref_bottom, ref_left:ref_right]

    if max_offset>=2*min(ref.shape):
            raise ValueError("Offset cannot be larger than image")

    rows, cols = ref.shape
    template_size = (rows-2*max_offset, cols-2*max_offset)
    n = template_size[0]*template_size[1]

    child_patch = child[max_offset:max_offset + template_size[0], max_offset:max_offset + template_size[1]]
    child_patch_normed = normalize_matrix(child_patch)

    ref_windows = sliding_window(ref, template_size)

    means = np.mean(ref_windows, axis=(-2,-1))/n

    sum_sq = np.einsum('ijhw,ijhw->ij', ref_windows, ref_windows)
    variances = sum_sq / n - means**2
    sigmas = np.sqrt(np.maximum(variances, 1e-12))

    ncc_map = np.einsum('ijhw,hw->ij', ref_windows, child_patch_normed) / (n * sigmas)   # CHANGED
    max_y, max_x = np.unravel_index(np.argmax(ncc_map), ncc_map.shape)

    offset = (max_y-max_offset, max_x-max_offset)

    offset = (offset[0]+known_offset[0], offset[1]+known_offset[1])
    return offset


def align_and_crop(b: np.array, g: np.array, g_offset: tuple, r: np.array, r_offset: tuple) -> tuple:
    rows, cols = b.shape

    (gi, gj), (ri, rj) = g_offset, r_offset

    top, bottom, left, right = max(g_offset[0], r_offset[0], 0), min(rows+g_offset[0], rows+r_offset[0], rows), max(g_offset[1], r_offset[1], 0), min(cols+g_offset[1], cols+r_offset[1], cols)

    acb = b[top:bottom, left:right]
    acg = g[top-gi:bottom-gi, left-gj:right-gj]
    acr = r[top-ri:bottom-ri, left-rj:right-rj]

    return (acr, acg, acb)

def normalize_matrix(mat: np.array) -> np.array:
    std, mean = np.std(mat), np.mean(mat)
    if std == 0:
        return mat - mean
    return (mat - mean)/std