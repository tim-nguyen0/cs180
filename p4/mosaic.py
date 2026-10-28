import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt, zoom

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "p2"))
from stacks import laplacian_stack
from filters import gaussian_blur
from utils import load_image

from homography import computeH, warpImageBilinear, warpImageNearestNeighbor, _output_bbox


def scale_H(H, s):
    S = np.diag([s, s, 1.0])
    return S @ H @ np.linalg.inv(S)


def chain_homographies(pair_Hs, ref):
    n = len(pair_Hs) + 1
    Hs = [None] * n
    Hs[ref] = np.eye(3)
    for i in range(ref - 1, -1, -1):
        Hs[i] = Hs[i + 1] @ pair_Hs[i]
    for i in range(ref + 1, n):
        Hs[i] = Hs[i - 1] @ np.linalg.inv(pair_Hs[i - 1])
    return Hs


def feather_weight(h, w):
    d = distance_transform_edt(np.pad(np.ones((h, w)), 1))[1:-1, 1:-1]
    return d / d.max()


def mosaic_canvas(images, Hs, limit=3.0):
    h, w = images[0].shape[:2]
    boxes = np.array([_output_bbox(im, H) for im, H in zip(images, Hs)])
    xmin, ymin = boxes[:, :2].min(0); xmax, ymax = boxes[:, 2:].max(0)
    return (max(xmin, -limit * w), max(ymin, -limit * h), min(xmax, (limit + 1) * w), min(ymax, (limit + 1) * h))


def warp_all(images, Hs, bbox, warp=warpImageBilinear):
    ims, alphas, weights = [], [], []
    for im, H in zip(images, Hs):
        w, a, _ = warp(np.dstack([im, feather_weight(*im.shape[:2])]), H, bbox)
        ims.append(w[..., :3]); alphas.append(a); weights.append(w[..., 3] * a)
    return np.array(ims), np.array(alphas), np.array(weights)


def lowpass(x, sigma):
    # blur a downsampled copy for large sigma, then upsample
    f = max(1, int(sigma / 4))
    if f == 1:
        return gaussian_blur(x, sigma)
    h, w = x.shape[:2]
    small = x[:h // f * f, :w // f * f].reshape(h // f, f, w // f, f, *x.shape[2:]).mean(axis=(1, 3))
    small = gaussian_blur(small, sigma / f)
    up = zoom(small, (f, f) + (1,) * (x.ndim - 2), order=1)
    out = np.zeros_like(x, dtype=float)
    out[:up.shape[0], :up.shape[1]] = up
    out[up.shape[0]:] = out[up.shape[0] - 1:up.shape[0]]; out[:, up.shape[1]:] = out[:, up.shape[1] - 1:up.shape[1]]
    return out


def blend_average(ims, weights):
    wsum = weights.sum(0)
    return np.einsum("nhwc,nhw->hwc", ims, weights) / np.maximum(wsum, 1e-8)[..., None]


def blend_two_band(ims, alphas, weights, sigma):
    # high band uses hard argmax masks, low band uses the blurred masks
    masks = (weights == weights.max(0)) & (weights > 0)
    acc = np.zeros(ims.shape[1:]); norm = np.zeros(ims.shape[1:3])
    for im, a, m in zip(ims, alphas, masks):
        low = lowpass(im * a[..., None], sigma) / np.maximum(lowpass(a, sigma), 1e-8)[..., None]
        high, low = laplacian_stack([im, low])
        m_high = m.astype(float)
        m_low = lowpass(m_high, sigma) * a
        acc += m_high[..., None] * high + m_low[..., None] * low
        norm += m_low
    return acc / np.maximum(norm, 1e-8)[..., None]


def load_scaled(path, scale):
    im = load_image(path)
    if scale == 1:
        return im
    from PIL import Image
    pil = Image.fromarray((im * 255).astype(np.uint8))
    return np.asarray(pil.resize((round(im.shape[1] * scale), round(im.shape[0] * scale)), Image.LANCZOS)) / 255.0


def build_mosaic(names, corr_dir, data_dir, scale=0.5, ref=None, sigma=None, limit=3.0, warp=warpImageBilinear):
    names = list(names)
    ref = len(names) // 2 if ref is None else ref
    pair_Hs = []
    for a, b in zip(names, names[1:]):
        d = np.load(Path(corr_dir) / f"{Path(a).stem}_{Path(b).stem}.npz")
        pair_Hs.append(scale_H(computeH(d["im1_pts"], d["im2_pts"]), scale))
    images = [load_scaled(Path(data_dir) / n, scale) for n in names]
    Hs = chain_homographies(pair_Hs, ref)
    bbox = mosaic_canvas(images, Hs, limit)
    ims, alphas, weights = warp_all(images, Hs, bbox, warp)
    sigma = sigma or 0.03 * images[ref].shape[1]
    blend = blend_two_band(ims, alphas, weights, sigma)
    average = blend_average(ims, weights)
    return dict(blend=np.clip(blend, 0, 1), average=np.clip(average, 0, 1), ims=ims, alphas=alphas,
                weights=weights, Hs=Hs, images=images, sigma=sigma, bbox=bbox)
