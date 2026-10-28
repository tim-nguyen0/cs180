import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch


# Part A.2

def build_system(im1_pts, im2_pts):
    """Return (A, b) with A (2n x 8) and b (2n,) such that A h = b for the 8 unknown entries of H."""
    im1_pts = np.asarray(im1_pts, dtype=float)
    im2_pts = np.asarray(im2_pts, dtype=float)
    n = im1_pts.shape[0]
    assert n >= 4, "need at least 4 correspondences"
    x, y = im1_pts[:, 0], im1_pts[:, 1]
    xp, yp = im2_pts[:, 0], im2_pts[:, 1]
    zero, one = np.zeros(n), np.ones(n)
    A = np.empty((2 * n, 8))
    A[0::2] = np.stack([x, y, one, zero, zero, zero, -x * xp, -y * xp], axis=1)
    A[1::2] = np.stack([zero, zero, zero, x, y, one, -x * yp, -y * yp], axis=1)
    b = np.empty(2 * n)
    b[0::2] = xp
    b[1::2] = yp
    return A, b


def computeH(im1_pts, im2_pts):
    """Recover H (3x3) mapping im1 points to im2 points: p2 ~ H p1.

    im1_pts, im2_pts: n-by-2 arrays of (x, y) correspondences, n >= 4.
    With n > 4 the overdetermined system is solved by least squares.
    """
    A, b = build_system(im1_pts, im2_pts)
    h, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.append(h, 1.0).reshape(3, 3)


def show_correspondences(im1, im2, im1_pts, im2_pts, save=None):
    """Draw numbered correspondences side by side with matching colors and connecting lines."""
    im1_pts = np.asarray(im1_pts, float)
    im2_pts = np.asarray(im2_pts, float)
    n = len(im1_pts)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n, 2)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(im1); ax2.imshow(im2)
    for ax, pts in ((ax1, im1_pts), (ax2, im2_pts)):
        ax.scatter(pts[:, 0], pts[:, 1], c=colors[:n], s=60, edgecolors="white", linewidths=1.2, zorder=3)
        for i, (x, y) in enumerate(pts):
            ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords="offset points", color="white",
                        fontsize=10, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc=colors[i], ec="none", alpha=0.85))
        ax.set_axis_off()
    for i in range(n):
        fig.add_artist(ConnectionPatch(xyA=im1_pts[i], coordsA=ax1.transData,
                                       xyB=im2_pts[i], coordsB=ax2.transData,
                                       color=colors[i], lw=0.8, alpha=0.6))
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig



# Part A.3

def _output_bbox(im, H):
    """Map the four image corners through H; return integer (xmin, ymin, xmax, ymax) of the result."""
    h, w = im.shape[:2]
    corners = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]], float)
    q = corners @ H.T
    q = q[:, :2] / q[:, 2:3]
    xmin, ymin = np.floor(q.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(q.max(axis=0)).astype(int)
    return xmin, ymin, xmax, ymax


def _sample_nearest(im, xs, ys):
    """Round source coords to the nearest pixel. Returns (values, valid mask)."""
    h, w = im.shape[:2]
    xi, yi = np.rint(xs).astype(int), np.rint(ys).astype(int)
    valid = (xi >= 0) & (xi <= w - 1) & (yi >= 0) & (yi <= h - 1)
    out = np.zeros(xs.shape + im.shape[2:], dtype=float)
    out[valid] = im[yi[valid], xi[valid]]
    return out, valid


def _sample_bilinear(im, xs, ys):
    """Weighted average of the four pixels surrounding each source coord. Returns (values, valid mask)."""
    h, w = im.shape[:2]
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    valid = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)  # on the last row/col fx or fy is 0
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y1, 0, h - 1)
    fx, fy = xs - x0, ys - y0
    if im.ndim == 3:  # broadcast weights over channels
        fx, fy = fx[..., None], fy[..., None]
    out = ((1 - fx) * (1 - fy) * im[y0c, x0c] + fx * (1 - fy) * im[y0c, x1c]
           + (1 - fx) * fy * im[y1c, x0c] + fx * fy * im[y1c, x1c])
    out[~valid] = 0
    return out, valid


def warpImage(im, H, sampler, bbox=None):
    """Inverse-warp im by H. Returns (warped, alpha, (xmin, ymin)).

    warped: float image covering the bounding box of H(im) (or the given
    bbox = (xmin, ymin, xmax, ymax) in output coordinates, e.g. a shared mosaic
    canvas); alpha: 1 where a source pixel exists, else 0; (xmin, ymin):
    location of warped[0, 0] in the output coordinate frame.
    """
    im = np.asarray(im, dtype=float)
    xmin, ymin, xmax, ymax = _output_bbox(im, H) if bbox is None else bbox
    xs, ys = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
    # inverse map every output pixel center back into the source image
    P = np.stack([xs, ys, np.ones_like(xs)], axis=-1).reshape(-1, 3) @ np.linalg.inv(H).T
    src = P[:, :2] / P[:, 2:3]
    sx, sy = src[:, 0].reshape(xs.shape), src[:, 1].reshape(xs.shape)
    warped, valid = sampler(im, sx, sy)
    return warped, valid.astype(float), (xmin, ymin)


def warpImageNearestNeighbor(im, H, bbox=None):
    return warpImage(im, H, _sample_nearest, bbox)


def warpImageBilinear(im, H, bbox=None):
    return warpImage(im, H, _sample_bilinear, bbox)
