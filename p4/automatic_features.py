import sys
from pathlib import Path
import numpy as np
from skimage.feature import corner_harris, peak_local_max

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "p2"))
from filters import gaussian_blur
import homography as hom



def get_harris_corners(im, edge_discard=20):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).

    Courtesy of Starter Code Provided in CS180 Project 4 Webpage
    """

    assert edge_discard >= 20

    # find harris corners
    h = corner_harris(im, method='eps', sigma=1)
    coords = peak_local_max(h, min_distance=1)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask].T
    return h, coords


def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """

    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, 'Data dimension does not match dimension of centers'

    return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
            np.ones((   ndata, 1)) * np.sum((c**2).T, axis=0)    - \
            2 * np.inner(x, c)

def anms(h, coords, n_ip=500, c=0.9):
    """
    h: Harris response (H, W). coords: 2 x n (ys, xs) from get_harris_corners.
    Returns 2 x n_ip coords, sorted by suppression radius desc.
    """
    ys, xs = coords
    f = h[ys, xs]                          
    pts = coords.T.astype(float)           

    d2 = dist2(pts, pts)                   
    suppress = f[:, None] < c * f[None, :] 
    d2[~suppress] = np.inf
    r2 = d2.min(axis=1)                

    keep = np.argsort(-r2)[:n_ip]
    return coords[:, keep]

def get_descriptors(im_gs, pts, sigma=2.0):
    """
    im_gs: grayscale image. 
    pts: 2 x n as (ys, xs) >=20 px from the border.
    Blurs and normalizes grayscaled 40x40 patch
    Returns n x 64 (flattened 8x8 descriptor).
    """
    blurred = gaussian_blur(im_gs, sigma)
    descs = []
    for y, x in pts.T:
        patch = blurred[y-20:y+20, x-20:x+20].reshape(8,5,8,5).mean(axis=(1,3))
        patch = (patch - patch.mean()) / (patch.std() + 1e-8)
        descs.append(patch.ravel())
    return np.array(descs)

def lowes_threshold(descs1, descs2, threshold=0.3):
    """
    descs1, descs2 : N x 64 and M x 64 sized feature matrices
    threshold: Lowe's ratio test threshold
    """

    sq_dists = np.sum(descs1**2, axis=1, keepdims=True) + np.sum(descs2**2, axis=1)[None, :] - 2 * descs1 @ descs2.T
    dists = np.sqrt(np.maximum(sq_dists,0.0)) # for fp errors

    sorted_i = np.argsort(dists,axis=1)
    sorted_dists = np.sort(dists, axis=1)

    nn1_i = sorted_i[:, 0]
    nn1_dist = sorted_dists[:,0]
    nn2_dist = sorted_dists[:, 1]

    ratios = nn1_dist/np.maximum(nn2_dist, 1e-12)
    keep = ratios < threshold

    i_1 = np.where(keep)[0]
    i_2 = nn1_i[keep]
    return np.column_stack((i_1, i_2))

def ransac(pts1, pts2, corr, n=500, r=5.0):
    num_corrs = corr.shape[0]
    rng = np.random.default_rng()

    all_pts1 = pts1[corr[:, 0]]  
    all_pts2 = pts2[corr[:, 1]]  

    pts1_hom = np.column_stack((all_pts1, np.ones(num_corrs)))

    max_inliers = 0
    best_inliers = None
    best_H = None
    for i in range(n):
        idx = rng.choice(num_corrs, size=4, replace=False)
        im1_pts = pts1[corr[idx, 0]]
        im2_pts = pts2[corr[idx, 1]]
        H = hom.computeH(im1_pts, im2_pts)

        proj_hom = (H @ pts1_hom.T).T
        proj_pts = proj_hom[:, :2] / proj_hom[:, 2:3]

        sq_dists = np.diag(dist2(proj_pts, all_pts2))
        inliers = sq_dists < (r ** 2)
        num_inliers = np.sum(inliers)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers = inliers
            best_H = H

    # Run least squares on inliers for more robustness like mentioned in lecture
    if best_inliers is not None and max_inliers >= 4:
        best_H = hom.computeH(all_pts1[best_inliers], all_pts2[best_inliers])

    return best_H, best_inliers


def extract_features(im_gs, n_ip=500, c=0.9, pre=5000, sigma=2.0):
    h, coords = get_harris_corners(im_gs)
    strongest = np.argsort(-h[coords[0], coords[1]])[:pre]
    kept = anms(h, coords[:, strongest], n_ip, c)
    return kept, get_descriptors(im_gs, kept, sigma)

