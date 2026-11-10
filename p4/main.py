import argparse
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from homography import (computeH, build_system, show_correspondences, warpImageNearestNeighbor,
                        warpImageBilinear, _output_bbox)
import mosaic
from automatic_features import get_harris_corners, anms, get_descriptors, lowes_threshold, ransac, extract_features

ROOT = Path(__file__).resolve().parent
DATA, CORR, OUT = ROOT / "data", ROOT / "corr", ROOT / "out"
SETS = {1: ["%d.JPG" % i for i in range(1, 8)], 2: ["2-%d.JPG" % i for i in range(1, 6)]}


def save(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(img, 0, 1))


def grid(images, titles, path, ncols=None, size=4):
    n = len(images); ncols = ncols or n; nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows * 0.65), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, im, t in zip(axes.flat, images, titles):
        ax.imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1); ax.set_title(t)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- A.2
def run_corr():
    out = OUT / "a2"; out.mkdir(parents=True, exist_ok=True)
    for f in sorted(CORR.glob("*_*.npz")):
        if f.stem.startswith("rect_"):
            continue
        a, b = f.stem.split("_")
        d = np.load(f); p1, p2 = d["im1_pts"], d["im2_pts"]
        H = computeH(p1, p2); A, bvec = build_system(p1, p2)
        q = np.hstack([p1, np.ones((len(p1), 1))]) @ H.T; q = q[:, :2] / q[:, 2:]
        err = np.linalg.norm(q - p2, axis=1)
        pa, pb = next(DATA.glob(a + ".*")), next(DATA.glob(b + ".*"))
        s = 0.25 if mosaic.load_image(pa).shape[1] > 2000 else 1.0
        im1, im2 = mosaic.load_scaled(pa, s), mosaic.load_scaled(pb, s)
        fig = show_correspondences(im1, im2, p1 * s, p2 * s, save=out / f"corr_{f.stem}.png"); plt.close(fig)
        with np.printoptions(precision=2, suppress=True, linewidth=250):
            txt = [f"pair {a} -> {b}: {len(p1)} correspondences", "",
                   "im1_pts (x, y)            im2_pts (x', y')", str(np.hstack([p1, p2])), "",
                   f"A ({A.shape[0]} x 8), h = (a b c d e f g h)^T, b:", str(np.c_[A, bvec]), "",
                   "least-squares H (H[2,2] = 1):", np.array2string(H, precision=6), "",
                   f"reprojection error px: mean {err.mean():.2f}, max {err.max():.2f}"]
        (out / f"system_{f.stem}.txt").write_text("\n".join(txt))
        print(f"{f.stem}: n={len(p1)} err mean {err.mean():.2f} max {err.max():.2f}")


# ---------------------------------------------------------------- A.3
def order_corners(c):
    """Sort 4 points clockwise starting at the top-left, whatever order they were clicked in."""
    c = np.asarray(c, float)
    ang = np.arctan2(*(c - c.mean(0))[:, ::-1].T)          # angle about the centroid (y down => clockwise)
    c = c[np.argsort(ang)]
    tl = np.argmin(c.sum(1))                               # smallest x+y is the top-left corner
    return np.roll(c, -tl, axis=0)


def rectify_H(corners, pixels=None):
    """H sending the 4 clicked corners of a rectangle to an axis-aligned rectangle whose aspect ratio
    is taken from the average measured side lengths (or given as (w, h) pixels)."""
    c = order_corners(corners)                             # TL, TR, BR, BL
    if pixels is None:
        w = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2
        h = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2
    else:
        w, h = pixels
    target = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    return computeH(c, target), (w, h)


def run_rectify(names, scale=0.5):
    out = OUT / "a3"; out.mkdir(parents=True, exist_ok=True)
    for name in names:
        stem = Path(name).stem
        corners = order_corners(np.load(CORR / f"rect_{stem}.npz")["corners"] * scale)
        im = mosaic.load_scaled(DATA / name, scale)
        H, (w, h) = rectify_H(corners)
        # whole projected image; only if a corner maps behind the camera (w <= 0, infinite extent)
        # fall back to a window around the rectangle
        Hh, W = im.shape[:2]
        wc = (np.array([[0, 0, 1], [W - 1, 0, 1], [0, Hh - 1, 1], [W - 1, Hh - 1, 1]]) @ H.T)[:, 2]
        if wc.min() > 0:
            bbox = _output_bbox(im, H)
        else:
            k = 2 * W
            bbox = (-k, -k, w + k, h + k)
        # pixels past the plane's horizon map to source points behind the camera: mask them out
        G = np.linalg.inv(H)
        xs, ys = np.meshgrid(np.arange(bbox[0], bbox[2] + 1), np.arange(bbox[1], bbox[3] + 1))
        front = (G[2, 0] * xs + G[2, 1] * ys + G[2, 2]) > 0
        results, times = {}, {}
        for tag, fn in (("nearest", warpImageNearestNeighbor), ("bilinear", warpImageBilinear)):
            t = time.perf_counter(); warped, alpha, _ = fn(im, H, bbox); times[tag] = time.perf_counter() - t
            alpha = alpha * front
            results[tag] = np.dstack([warped * alpha[..., None], alpha])
        ys, xs = np.where(results["bilinear"][..., 3] > 0)      # drop empty canvas around the warp
        trim = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        for tag in results:
            results[tag] = results[tag][trim]
            save(results[tag], out / f"{stem}_{tag}.png")
        bbox = (bbox[0] + xs.min(), bbox[1] + ys.min(), bbox[0] + xs.max(), bbox[1] + ys.max())
        # zoomed crop on the rectified object for the quality comparison
        ox, oy = -bbox[0], -bbox[1]
        crop = np.s_[int(oy):int(oy + h), int(ox):int(ox + w)]
        cw = max(1, int(min(w, h) * 0.15))
        # zoom on the most detailed patch inside the rectangle (highest local gradient energy)
        g = results["bilinear"][crop][..., :3].mean(-1)
        g = np.abs(np.diff(g, axis=0))[:, :-1] + np.abs(np.diff(g, axis=1))[:-1]
        e = np.cumsum(np.cumsum(np.pad(g, ((1, 0), (1, 0))), 0), 1)
        n = 2 * cw
        if g.shape[0] > n and g.shape[1] > n:
            box = e[n:, n:] - e[:-n, n:] - e[n:, :-n] + e[:-n, :-n]
            zy, zx = np.unravel_index(box.argmax(), box.shape)
        else:
            zy, zx = 0, 0
        zoom = np.s_[int(oy) + zy:int(oy) + zy + n, int(ox) + zx:int(ox) + zx + n]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes[0, 0].imshow(im); axes[0, 0].plot(*np.vstack([corners, corners[:1]]).T, "y-o", ms=4)
        axes[0, 0].set_title(f"{name} (clicked rectangle)")
        axes[0, 1].imshow(results["nearest"][crop]); axes[0, 1].set_title(f"nearest neighbour ({times['nearest']:.2f}s)")
        axes[0, 2].imshow(results["bilinear"][crop]); axes[0, 2].set_title(f"bilinear ({times['bilinear']:.2f}s)")
        axes[1, 0].imshow(results["nearest"]); axes[1, 0].set_title("full warp (nearest)")
        axes[1, 1].imshow(results["nearest"][zoom], interpolation="nearest"); axes[1, 1].set_title("zoom: nearest")
        axes[1, 2].imshow(results["bilinear"][zoom], interpolation="nearest"); axes[1, 2].set_title("zoom: bilinear")
        for ax in axes.flat:
            ax.axis("off")
        fig.tight_layout(); fig.savefig(out / f"{stem}_comparison.png", dpi=120, bbox_inches="tight"); plt.close(fig)
        with np.printoptions(precision=6, suppress=True):
            (out / f"{stem}_H.txt").write_text(
                f"corners (scale {scale}):\n{corners}\n\ntarget rectangle {w:.1f} x {h:.1f}\n\nH:\n{H}\n\n"
                f"nearest {times['nearest']:.3f}s   bilinear {times['bilinear']:.3f}s\n")
        print(f"{name}: nearest {times['nearest']:.2f}s, bilinear {times['bilinear']:.2f}s -> {out}")


# ---------------------------------------------------------------- A.4
def save_mosaic(r, names, out, scale):
    save(r["blend"], out / "mosaic_two_band.jpg")
    save(r["average"], out / "mosaic_feather_average.jpg")
    grid(r["images"], [Path(n).stem for n in names], out / "sources.png", ncols=min(4, len(names)))
    grid(list(r["ims"]), [f"warped {Path(n).stem}" for n in names], out / "warped.png", ncols=min(4, len(names)), size=5)
    grid(list(r["weights"]), [f"weight {Path(n).stem}" for n in names], out / "weights.png", ncols=min(4, len(names)), size=5)
    label = np.where(r["weights"].max(0) > 0, r["weights"].argmax(0) + 1, 0)   # which image owns each pixel
    plt.imsave(out / "seams.png", label, cmap="tab10", vmin=0, vmax=9)
    # overlap-only comparison: hard overwrite vs feather average vs two-band blend
    overwrite = np.zeros_like(r["blend"])
    for im, a in zip(r["ims"], r["alphas"]):
        overwrite[a > 0] = im[a > 0]
    save(overwrite, out / "mosaic_overwrite.jpg")
    grid([overwrite, r["average"], r["blend"]], ["overwrite", "feathered average", "2-band blend"],
         out / "blend_comparison.png", ncols=1, size=14)
    with np.printoptions(precision=6, suppress=True):
        (out / "homographies.txt").write_text("\n\n".join(
            f"H {Path(n).stem} -> reference {Path(names[len(names)//2]).stem} (scale {scale}):\n{H}" for n, H in zip(names, r["Hs"])))


def run_mosaic(names, tag, scale=0.5, limit=3.0, sigma=None, sigma_high=0):
    out = OUT / "a4" / tag; out.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    r = mosaic.build_mosaic(names, CORR, DATA, scale=scale, limit=limit, sigma=sigma, sigma_high=sigma_high)
    print(f"{tag}: canvas {r['blend'].shape[1]}x{r['blend'].shape[0]} at scale {scale}, "
          f"sigma {r['sigma']:.1f}, {time.perf_counter() - t:.1f}s")
    save_mosaic(r, names, out, scale)


# ---------------------------------------------------------------- B.1
def run_corners(names, scale=0.5, n_ip=500, c=0.9, pre=5000):
    out = OUT / "b1"; out.mkdir(parents=True, exist_ok=True)
    for name in names:
        stem = Path(name).stem
        im = mosaic.load_scaled(DATA / name, scale)
        gray = im.mean(-1)
        t = time.perf_counter(); h, coords = get_harris_corners(gray); t_h = time.perf_counter() - t
        strongest = np.argsort(-h[coords[0], coords[1]])[:pre]
        t = time.perf_counter(); kept = anms(h, coords[:, strongest], n_ip, c); t_a = time.perf_counter() - t
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(h, cmap="magma", vmin=0, vmax=np.percentile(h, 99.9)); axes[0].set_title("Harris response")
        axes[1].imshow(im); axes[1].scatter(coords[1], coords[0], s=2, c="yellow", lw=0)
        axes[1].set_title(f"all Harris corners ({coords.shape[1]})")
        axes[2].imshow(im); axes[2].scatter(kept[1], kept[0], s=8, c="cyan", edgecolors="black", lw=0.3)
        axes[2].set_title(f"ANMS, {kept.shape[1]} strongest (c = {c})")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout(); fig.savefig(out / f"{stem}_corners.png", dpi=120, bbox_inches="tight"); plt.close(fig)
        np.savez(out / f"{stem}_corners.npz", harris=coords, anms=kept)
        print(f"{name}: {coords.shape[1]} corners in {t_h:.2f}s, ANMS -> {kept.shape[1]} in {t_a:.2f}s")


# ---------------------------------------------------------------- B.2
def run_descriptors(names, scale=0.5, n_ip=500, c=0.9, pre=5000, show=8, sigma=2.0):
    out = OUT / "b2"; out.mkdir(parents=True, exist_ok=True)
    for name in names:
        stem = Path(name).stem
        im = mosaic.load_scaled(DATA / name, scale)
        gray = im.mean(-1)
        kept, descs = extract_features(gray, n_ip, c, pre, sigma)
        np.savez(out / f"{stem}_descriptors.npz", coords=kept, descriptors=descs)
        # show the first `show` corners: location, raw 40x40 window, normalized 8x8 descriptor
        idx = np.linspace(0, kept.shape[1] - 1, show).astype(int)
        fig, axes = plt.subplots(3, show, figsize=(2.2 * show, 7))
        for j, i in enumerate(idx):
            y, x = kept[:, i]
            axes[0, j].imshow(im); axes[0, j].set_xlim(x - 60, x + 60); axes[0, j].set_ylim(y + 60, y - 60)
            axes[0, j].add_patch(plt.Rectangle((x - 20, y - 20), 40, 40, fill=False, ec="cyan", lw=1.5))
            axes[1, j].imshow(gray[y - 20:y + 20, x - 20:x + 20], cmap="gray", vmin=0, vmax=1)
            axes[2, j].imshow(descs[i].reshape(8, 8), cmap="gray", interpolation="nearest")
            axes[0, j].set_title(f"({x}, {y})", fontsize=9)
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        for ax, lab in zip(axes[:, 0], ["corner", "40x40 window", "8x8 descriptor"]):
            ax.set_ylabel(lab)
        fig.tight_layout(); fig.savefig(out / f"{stem}_descriptors.png", dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"{name}: {descs.shape[0]} descriptors of length {descs.shape[1]}, "
              f"mean |{np.abs(descs.mean(1)).max():.1e}|, std {descs.std(1).min():.3f}-{descs.std(1).max():.3f}")


def draw_matches(im1, im2, pts1, pts2, corners1, corners2, title, path, inliers=None):
    H = max(im1.shape[0], im2.shape[0]); W1 = im1.shape[1]
    canvas = np.ones((H, W1 + im2.shape[1], 3))
    canvas[:im1.shape[0], :W1] = im1; canvas[:im2.shape[0], W1:] = im2
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(canvas)
    ax.scatter(corners1[1], corners1[0], s=4, c="yellow", alpha=0.4)
    ax.scatter(corners2[1] + W1, corners2[0], s=4, c="yellow", alpha=0.4)
    ok = np.ones(len(pts1), bool) if inliers is None else inliers
    for (x1, y1), (x2, y2), good in zip(pts1, pts2, ok):
        ax.plot([x1, x2 + W1], [y1, y2], "-", c="lime" if good else "red", lw=0.6 if good else 1.0, alpha=0.8)
    for p, dx in ((pts1, 0), (pts2, W1)):
        ax.scatter(p[ok, 0] + dx, p[ok, 1], s=10, c="lime", edgecolors="black", linewidths=0.3)
        ax.scatter(p[~ok, 0] + dx, p[~ok, 1], s=14, c="red", marker="x")
    ax.set_axis_off(); ax.set_title(title)
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def run_lowes_threshold(names, scale=0.5, n_ip=500, c=0.9, pre=5000, sigma=2.0, threshold=0.3):
    out = OUT / "b3"; out.mkdir(parents=True, exist_ok=True)
    ims = [mosaic.load_scaled(DATA / n, scale) for n in names]
    feats = [extract_features(im.mean(-1), n_ip, c, pre, sigma) for im in ims]
    for (na, (pa, da)), (nb, (pb, db)) in zip(zip(names, feats), zip(names[1:], feats[1:])):
        tag = f"{Path(na).stem}_{Path(nb).stem}"
        m = lowes_threshold(da, db, threshold)
        pts1 = pa[::-1, m[:, 0]].T; pts2 = pb[::-1, m[:, 1]].T
        np.savez(out / f"{tag}_matches.npz", im1_pts=pts1, im2_pts=pts2, matches=m)
        im1, im2 = ims[names.index(na)], ims[names.index(nb)]
        draw_matches(im1, im2, pts1, pts2, pa, pb, f"{tag}: {len(m)} matches (ratio < {threshold})",
                     out / f"{tag}_matches.png")
        print(f"{tag}: {len(m)} matches of {len(da)} x {len(db)} features at threshold {threshold}")


def run_ransac_compute_hom(names, scale=0.5, n_ip=500, c=0.9, pre=5000, sigma=2.0, threshold=0.3,
                           iters=500, r=3.0):
    out = OUT / "b4"; out.mkdir(parents=True, exist_ok=True)
    ims = [mosaic.load_scaled(DATA / n, scale) for n in names]
    feats = [extract_features(im.mean(-1), n_ip, c, pre, sigma) for im in ims]
    for k, (na, nb) in enumerate(zip(names, names[1:])):
        tag = f"{Path(na).stem}_{Path(nb).stem}"
        (pa, da), (pb, db) = feats[k], feats[k + 1]
        m = lowes_threshold(da, db, threshold)
        pts1, pts2 = pa[::-1].T.astype(float), pb[::-1].T.astype(float)
        H, inliers = ransac(pts1, pts2, m, iters, r)
        mp1, mp2 = pts1[m[:, 0]], pts2[m[:, 1]]
        np.savez(out / f"{tag}_ransac.npz", H=H, im1_pts=mp1[inliers], im2_pts=mp2[inliers])
        np.savetxt(out / f"{tag}_H.txt", H, fmt="%12.6f")
        draw_matches(ims[k], ims[k + 1], mp1, mp2, pa, pb,
                     f"{tag}: {inliers.sum()} inliers / {len(m)} matches ({iters} iters, r = {r} px)",
                     out / f"{tag}_ransac.png", inliers)
        line = f"{tag}: {inliers.sum()}/{len(m)} inliers"
        manual = CORR / f"{tag}.npz"
        if manual.exists():
            d = np.load(manual)
            q1, q2 = d["im1_pts"] * scale, d["im2_pts"] * scale
            proj = (H @ np.column_stack((q1, np.ones(len(q1)))).T).T
            err = np.linalg.norm(proj[:, :2] / proj[:, 2:] - q2, axis=1)
            line += f", manual clicks reproject with mean {err.mean():.2f} px (max {err.max():.2f})"
        print(line)


def run_auto_mosaic(names, tag, scale=0.5, limit=3.0, sigma=None, sigma_high=0, n_ip=500, c=0.9, pre=5000,
                    desc_sigma=2.0, threshold=0.3, iters=500, r=3.0):
    out = OUT / "b5" / tag; out.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    res = mosaic.build_auto_mosaic(names, DATA, scale=scale, limit=limit, sigma=sigma, sigma_high=sigma_high,
                                   n_ip=n_ip, c=c, pre=pre, desc_sigma=desc_sigma, threshold=threshold, iters=iters, r=r)
    print(f"{tag}: canvas {res['blend'].shape[1]}x{res['blend'].shape[0]} at scale {scale}, "
          f"sigma {res['sigma']:.1f}, {time.perf_counter() - t:.1f}s")
    save_mosaic(res, names, out, scale)
    for k, (na, nb) in enumerate(zip(names, names[1:])):
        pr = res["pairs"][k]; pt = f"{Path(na).stem}_{Path(nb).stem}"
        draw_matches(res["images"][k], res["images"][k + 1], pr["pts1"], pr["pts2"], res["feats"][k][0], res["feats"][k + 1][0],
                     f"{pt}: {pr['inliers'].sum()} inliers / {len(pr['pts1'])} matches", out / f"{pt}_matches.png", pr["inliers"])
        print(f"  {pt}: {pr['inliers'].sum()}/{len(pr['pts1'])} inliers")
    manual = OUT / "a4" / tag / "mosaic_two_band.jpg"
    if manual.exists():
        grid([plt.imread(manual) / 255.0, res["blend"]], ["manual correspondences", "automatic (Harris + MOPS + RANSAC)"],
             out / "manual_vs_auto.png", ncols=1, size=14)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("corr")
    p = sub.add_parser("rectify"); p.add_argument("images", nargs="+"); p.add_argument("--scale", type=float, default=0.5)
    p = sub.add_parser("mosaic")
    p.add_argument("--set", type=int, choices=SETS); p.add_argument("--names", nargs="+"); p.add_argument("--tag")
    p.add_argument("--scale", type=float, default=0.5); p.add_argument("--limit", type=float, default=3.0)
    p.add_argument("--sigma", type=float); p.add_argument("--sigma-high", type=float, default=0)
    p = sub.add_parser("corners"); p.add_argument("images", nargs="+"); p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--n", type=int, default=500); p.add_argument("--c", type=float, default=0.9)
    p.add_argument("--pre", type=int, default=5000, help="strongest corners kept before ANMS")
    p = sub.add_parser("descriptors"); p.add_argument("images", nargs="+"); p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--n", type=int, default=500); p.add_argument("--c", type=float, default=0.9)
    p.add_argument("--pre", type=int, default=5000); p.add_argument("--show", type=int, default=8)
    p.add_argument("--sigma", type=float, default=2.0)
    p = sub.add_parser("match"); p.add_argument("images", nargs="+"); p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--n", type=int, default=500); p.add_argument("--c", type=float, default=0.9)
    p.add_argument("--pre", type=int, default=5000); p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--threshold", type=float, default=0.3)
    p = sub.add_parser("ransac"); p.add_argument("images", nargs="+"); p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--n", type=int, default=500); p.add_argument("--c", type=float, default=0.9)
    p.add_argument("--pre", type=int, default=5000); p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--threshold", type=float, default=0.3); p.add_argument("--iters", type=int, default=500)
    p.add_argument("--r", type=float, default=3.0, help="inlier reprojection radius in pixels")
    p = sub.add_parser("automosaic")
    p.add_argument("--set", type=int, choices=SETS); p.add_argument("--names", nargs="+"); p.add_argument("--tag")
    p.add_argument("--scale", type=float, default=0.5); p.add_argument("--limit", type=float, default=3.0)
    p.add_argument("--sigma", type=float); p.add_argument("--sigma-high", type=float, default=0)
    p.add_argument("--n", type=int, default=500); p.add_argument("--c", type=float, default=0.9)
    p.add_argument("--pre", type=int, default=5000); p.add_argument("--desc-sigma", type=float, default=2.0)
    p.add_argument("--threshold", type=float, default=0.3); p.add_argument("--iters", type=int, default=500)
    p.add_argument("--r", type=float, default=3.0)
    a = ap.parse_args()
    if a.cmd == "corr":
        run_corr()
    elif a.cmd == "rectify":
        run_rectify(a.images, a.scale)
    elif a.cmd == "corners":
        run_corners(a.images, a.scale, a.n, a.c, a.pre)
    elif a.cmd == "descriptors":
        run_descriptors(a.images, a.scale, a.n, a.c, a.pre, a.show, a.sigma)
    elif a.cmd == "match":
        run_lowes_threshold(a.images, a.scale, a.n, a.c, a.pre, a.sigma, a.threshold)
    elif a.cmd == "ransac":
        run_ransac_compute_hom(a.images, a.scale, a.n, a.c, a.pre, a.sigma, a.threshold, a.iters, a.r)
    elif a.cmd == "automosaic":
        names = SETS[a.set] if a.set else a.names
        run_auto_mosaic(names, a.tag or (f"set{a.set}" if a.set else "custom"), a.scale, a.limit, a.sigma, a.sigma_high,
                        a.n, a.c, a.pre, a.desc_sigma, a.threshold, a.iters, a.r)
    else:
        names = SETS[a.set] if a.set else a.names
        run_mosaic(names, a.tag or (f"set{a.set}" if a.set else "custom"), a.scale, a.limit, a.sigma, a.sigma_high)


if __name__ == "__main__":
    main()
