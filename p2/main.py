import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

from edges import finite_difference, dog_filters, derivative_of_gaussian
from filters import gaussian_blur, gaussian_kernel, filter_image
from frequencies import unsharp_mask, unsharp_kernel, hybrid_image, fourier_magnitude
from alignment import align_images
from stacks import gaussian_stack, laplacian_stack
from blend import multiresolution_blend
from utils import load_image, save_grid


ROOT = Path(__file__).resolve().parent


def run_edges(image_path, threshold=0.2):
    image = load_image(image_path, grayscale=True)
    ix, iy, magnitude, edges = finite_difference(image, threshold)
    out = ROOT / "out" / "p1_2"
    out.mkdir(parents=True, exist_ok=True)

    limit = max(np.max(np.abs(ix)), np.max(np.abs(iy)), 1e-8)
    plt.imsave(out / "dx.png", ix, cmap="RdBu_r", vmin=-limit, vmax=limit)
    plt.imsave(out / "dy.png", iy, cmap="RdBu_r", vmin=-limit, vmax=limit)
    plt.imsave(out / "magnitude.png", magnitude, cmap="gray", vmin=0, vmax=1)
    plt.imsave(out / "edges.png", edges, cmap="gray", vmin=0, vmax=1)

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    panels = [image, ix, iy, magnitude, edges]
    titles = ["original", "Dx", "Dy", "magnitude", f"edges, threshold={threshold:g}"]
    for i, (ax, panel, title) in enumerate(zip(axes, panels, titles)):
        if i in (1, 2):
            ax.imshow(panel, cmap="RdBu_r", vmin=-limit, vmax=limit)
        else:
            ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    thresholds = sorted({0.05, 0.1, 0.2, threshold})
    fig, axes = plt.subplots(1, len(thresholds), figsize=(4 * len(thresholds), 4))
    for ax, value in zip(axes, thresholds):
        ax.imshow(magnitude > value, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"threshold={value:g}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "thresholds.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    (out / "parameters.txt").write_text(
        f"image: {image_path}\nthreshold: {threshold:g}\nboundary: symm\n"
        "Dx: [1, -1]\nDy: [[1], [-1]]\n"
    )
    print(f"saved edge results to {out}")


def run_dog(image_path, sigma=2.0, size=None, threshold=0.04, raw_threshold=0.2):
    image = load_image(image_path, grayscale=True)
    kernel = gaussian_kernel(sigma, size)
    blur = gaussian_blur(image, sigma, size)
    raw = finite_difference(image, raw_threshold)
    smooth = finite_difference(blur, threshold)
    dog = derivative_of_gaussian(image, sigma, size, threshold)
    kx, ky = dog_filters(sigma, size)
    out = ROOT / "out" / "p1_3"
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [raw[2], smooth[2], dog[2]]
    titles = ["finite difference", "blur then difference", "DoG"]
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "magnitude_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, result, title, value in zip(
        axes, [raw, smooth, dog],
        ["finite difference", "blur then difference", "DoG"],
        [raw_threshold, threshold, threshold]
    ):
        ax.imshow(result[3], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{title}, threshold={value:g}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "edge_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    limit = max(np.max(np.abs(kx)), np.max(np.abs(ky)))
    for ax, panel, title in zip(axes, [kernel, kx, ky], ["Gaussian", "DoG x", "DoG y"]):
        if title == "Gaussian":
            ax.imshow(panel, cmap="gray", vmin=0, vmax=kernel.max())
        else:
            ax.imshow(panel, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "filters.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    plt.imsave(out / "blurred.png", blur, cmap="gray", vmin=0, vmax=1)
    plt.imsave(out / "magnitude.png", dog[2], cmap="gray", vmin=0, vmax=1)
    plt.imsave(out / "edges.png", dog[3], cmap="gray", vmin=0, vmax=1)

    # skip the border when comparing the two methods
    margin = kernel.shape[0] // 2 + 1
    interior = (slice(margin, -margin), slice(margin, -margin))
    error = max(np.max(np.abs(smooth[i][interior] - dog[i][interior])) for i in (0, 1))
    border_error = max(np.max(np.abs(smooth[i] - dog[i])) for i in (0, 1))
    (out / "parameters.txt").write_text(
        f"image: {image_path}\nsigma: {sigma:g}\nsize: {kernel.shape[0]}\n"
        f"threshold: {threshold:g}\nraw threshold: {raw_threshold:g}\nboundary: symm\n"
        f"comparison margin: {margin}\ninterior max error: {error:.3e}\n"
        f"whole image max error: {border_error:.3e}\n"
    )
    print(f"saved DoG results to {out}; interior error {error:.3e}")


def run_sharpen(image_path, sigma=2.0, size=None, alpha=1.0):
    image = load_image(image_path)
    low, high, sharp = unsharp_mask(image, sigma, size, alpha)
    kernel = unsharp_kernel(sigma, size, alpha)
    single = filter_image(image, kernel)
    error = np.max(np.abs(single - sharp))
    out = ROOT / "out" / "p2_1" / image_path.stem
    out.mkdir(parents=True, exist_ok=True)

    # scale signed detail for display
    limit = max(np.max(np.abs(high)), 1e-8)
    detail = 0.5 + high / (2 * limit)
    save_grid(
        [image, low, detail, sharp],
        ["original", "blurred", "high frequencies (scaled)", f"sharpened, alpha={alpha:g}"],
        out / "overview.png"
    )
    alphas = sorted({0.5, 1.0, 2.0, alpha})
    save_grid(
        [image] + [image + value * high for value in alphas],
        ["original"] + [f"alpha={value:g}" for value in alphas],
        out / "alpha_comparison.png"
    )

    _, _, restored = unsharp_mask(low, sigma, size, alpha)
    save_grid(
        [image, low, restored], ["original", "blurred", "blurred then sharpened"],
        out / "blur_then_sharpen.png"
    )
    for name, panel in [("blurred", low), ("high", detail), ("sharpened", sharp)]:
        plt.imsave(out / f"{name}.png", np.clip(panel, 0, 1))
    limit = np.max(np.abs(kernel))
    plt.imsave(out / "kernel.png", kernel, cmap="RdBu_r", vmin=-limit, vmax=limit)

    blur_mse = np.mean((low - image)**2)
    restored_mse = np.mean((np.clip(restored, 0, 1) - image)**2)
    (out / "parameters.txt").write_text(
        f"image: {image_path}\nsigma: {sigma:g}\nsize: {kernel.shape[0]}\n"
        f"alpha: {alpha:g}\nboundary: symm\n"
        "single filter: (1 + alpha)*delta - alpha*G\n"
        f"single filter max error: {error:.3e}\n"
        f"blur MSE: {blur_mse:.6f}\nsharpened blur MSE (clipped): {restored_mse:.6f}\n"
    )
    print(f"saved sharpening results to {out}; single filter error {error:.3e}")


def run_hybrid(path_a, path_b, points, crop, sigma_low=8.0, sigma_high=4.0,
               color_low=False, color_high=False, mask_a=None, mask_b=None):
    original_a = load_image(path_a)
    original_b = load_image(path_b)
    points = np.array(points).reshape(4, 2)
    image_b, image_a = align_images(original_b, original_a, tuple(points[2:]) + tuple(points[:2]))
    if crop is not None:
        top, bottom, left, right = crop
        image_a, image_b = image_a[top:bottom, left:right], image_b[top:bottom, left:right]
    low_mask = None
    if mask_a:
        low_mask = load_image(mask_a, grayscale=True)
        mask = low_mask[..., None]
        image_a = mask * image_a + (1 - mask) * image_b
    if mask_b:
        mask = load_image(mask_b, grayscale=True)[..., None]
        image_b = mask * image_b + (1 - mask) * 0.9
    if not color_low:
        image_a = rgb2gray(image_a)
    if not color_high:
        image_b = rgb2gray(image_b)
    if color_low or color_high:
        if image_a.ndim == 2:
            image_a = np.repeat(image_a[..., None], 3, axis=2)
        if image_b.ndim == 2:
            image_b = np.repeat(image_b[..., None], 3, axis=2)
    low, high, hybrid = hybrid_image(image_a, image_b, sigma_low, sigma_high)
    if low_mask is not None:
        mask = low_mask[..., None] if hybrid.ndim == 3 else low_mask
        hybrid = mask * hybrid + (1 - mask) * image_b
    out = ROOT / "out" / "p2_2" / f"{path_a.stem}_{path_b.stem}"
    out.mkdir(parents=True, exist_ok=True)

    save_grid([original_a, original_b], ["original A", "original B"], out / "originals.png")
    save_grid([image_a, image_b], ["aligned A", "aligned B"], out / "aligned.png")
    limit = max(np.max(np.abs(high)), 1e-8)
    save_grid(
        [low, 0.5 + high / (2 * limit), hybrid],
        [f"low, sigma={sigma_low:g}", f"high (scaled), sigma={sigma_high:g}", "hybrid"],
        out / "process.png"
    )
    plt.imsave(out / "hybrid.png", np.clip(hybrid, 0, 1), cmap="gray", vmin=0, vmax=1)

    spectra = [fourier_magnitude(im) for im in [image_a, image_b, low, high, hybrid]]
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    vmin, vmax = min(im.min() for im in spectra), max(im.max() for im in spectra)
    for ax, spectrum, title in zip(axes, spectra, ["input A", "input B", "low", "high", "hybrid"]):
        ax.imshow(spectrum, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "fourier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    (out / "parameters.txt").write_text(
        f"low image: {path_a}\nhigh image: {path_b}\n"
        f"points (x, y): {points.tolist()}\ncrop (top, bottom, left, right): {crop}\n"
        f"sigma low: {sigma_low:g}\nsigma high: {sigma_high:g}\nboundary: symm\n"
        f"low image color: {color_low}\nhigh image color: {color_high}\n"
        f"low mask: {mask_a}\nhigh mask: {mask_b}\n"
        f"preserve high image outside low mask: {mask_a is not None}\n"
    )
    print(f"saved hybrid results to {out}")


def run_stacks(image_path, num_bands=5, sigma=2.0):
    image = load_image(image_path)
    gaussian = gaussian_stack(image, num_bands, sigma)
    laplacian = laplacian_stack(gaussian)
    out = ROOT / "out" / "p2_3" / image_path.stem
    out.mkdir(parents=True, exist_ok=True)

    save_grid(gaussian, [f"G{i}" for i in range(len(gaussian))], out / "gaussian.png")
    # scale bands for display, keep the actual values signed
    bands = [0.5 + band / (2 * max(np.max(np.abs(band)), 1e-8)) for band in laplacian[:-1]]
    save_grid(bands + [laplacian[-1]],
              [f"L{i} (scaled)" for i in range(num_bands)] + ["residual"],
              out / "laplacian.png")
    reconstructed = np.sum(laplacian, axis=0)
    error = np.max(np.abs(reconstructed - image))
    save_grid([image, reconstructed], ["original", "reconstructed"], out / "reconstruction.png")
    (out / "parameters.txt").write_text(
        f"image: {image_path}\nnum bands: {num_bands}\n"
        f"blur sigmas: {[sigma * 2**i for i in range(num_bands)]}\nboundary: symm\n"
        f"reconstruction max error: {error:.3e}\n"
    )
    print(f"saved stacks to {out}; reconstruction error {error:.3e}")


def run_blend(path_a, path_b, mask_path=None, num_bands=5, sigma=2.0):
    image_a, image_b = load_image(path_a), load_image(path_b)
    if mask_path:
        mask = load_image(mask_path, grayscale=True)
    else:
        mask = np.zeros(image_a.shape[:2])
        mask[:, :mask.shape[1] // 2] = 1
    result, parts_a, parts_b, masks = multiresolution_blend(image_a, image_b, mask, num_bands, sigma)
    out = ROOT / "out" / "p2_4" / f"{path_a.stem}_{path_b.stem}"
    out.mkdir(parents=True, exist_ok=True)

    weight = mask[..., None]
    save_grid([image_a, image_b, mask], ["input A", "input B", "mask"], out / "inputs.png")
    save_grid([weight * image_a, (1 - weight) * image_b],
              ["masked A", "masked B"], out / "masked_inputs.png")
    save_grid([weight * image_a + (1 - weight) * image_b, result],
              ["direct mask blend", "multiresolution blend"], out / "comparison.png")
    save_grid(masks, [f"mask G{i}" for i in range(len(masks))], out / "mask_stack.png")
    plt.imsave(out / "blend.png", np.clip(result, 0, 1))

    fig, axes = plt.subplots(4, 3, figsize=(9, 12))
    for row, level in enumerate([0, num_bands // 2, num_bands - 1]):
        panels = [parts_a[level], parts_b[level], parts_a[level] + parts_b[level]]
        limit = max(max(np.max(np.abs(panel)) for panel in panels), 1e-8)
        for col, (ax, panel, title) in enumerate(zip(axes[row], panels, ["A", "B", "sum"])):
            # same display scale across each row
            ax.imshow(np.clip(0.5 + panel / (2 * limit), 0, 1))
            ax.set_title(f"({chr(97 + row * 3 + col)}) {title}, L{level}")
            ax.axis("off")
    panels = [np.sum(parts_a, axis=0), np.sum(parts_b, axis=0), result]
    for ax, panel, title in zip(axes[3], panels, ["(j) A contribution", "(k) B contribution", "(l) blend"]):
        ax.imshow(np.clip(panel, 0, 1))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "figure_3_42.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    (out / "parameters.txt").write_text(
        f"image A: {path_a}\nimage B: {path_b}\nmask: {mask_path or 'left half selects A'}\n"
        f"num bands: {num_bands}\nblur sigmas: {[sigma * 2**i for i in range(num_bands)]}\n"
        "boundary: symm\n"
    )
    print(f"saved blend results to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    parser.add_argument("--extra-image", type=Path, default=ROOT / "data" / "cameraman.png")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--part", choices=["edges", "dog", "sharpen", "hybrid", "stacks", "blend", "all"], default="all")
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--size", type=int)
    parser.add_argument("--dog-threshold", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--low-image", type=Path)
    parser.add_argument("--high-image", type=Path)
    parser.add_argument("--points", type=float, nargs=8, metavar="XY")
    parser.add_argument("--crop", type=int, nargs=4, metavar="BOUND")
    parser.add_argument("--sigma-low", type=float, default=8.0)
    parser.add_argument("--sigma-high", type=float, default=4.0)
    parser.add_argument("--color-low", action="store_true")
    parser.add_argument("--color-high", action="store_true")
    parser.add_argument("--low-mask", type=Path)
    parser.add_argument("--high-mask", type=Path)
    parser.add_argument("--num-bands", type=int, default=5)
    parser.add_argument("--blend-a", type=Path, default=ROOT / "starter" / "apple.jpeg")
    parser.add_argument("--blend-b", type=Path, default=ROOT / "starter" / "orange.jpeg")
    parser.add_argument("--mask", type=Path)
    args = parser.parse_args()
    if args.part in ("hybrid", "all") and (args.low_image or args.high_image):
        if not (args.low_image and args.high_image and args.points):
            parser.error("custom hybrids need --low-image, --high-image, and --points")
    if args.part in ("edges", "all"):
        run_edges(args.image or ROOT / "data" / "cameraman.png", args.threshold)
    if args.part in ("dog", "all"):
        run_dog(args.image or ROOT / "data" / "cameraman.png", args.sigma, args.size,
                args.dog_threshold, args.threshold)
    if args.part in ("sharpen", "all"):
        run_sharpen(args.image or ROOT / "data" / "taj.jpg", args.sigma, args.size, args.alpha)
        run_sharpen(args.extra_image, args.sigma, args.size, args.alpha)
    if args.part in ("hybrid", "all"):
        points = args.points or [606, 289, 752, 363, 299, 345, 440, 331]
        crop = args.crop if args.low_image else args.crop or [480, 1010, 150, 600]
        run_hybrid(args.low_image or ROOT / "starter" / "nutmeg.jpg",
                   args.high_image or ROOT / "starter" / "DerekPicture.jpg",
                   points, crop, args.sigma_low, args.sigma_high, args.color_low, args.color_high,
                   args.low_mask, args.high_mask)
    if args.part in ("stacks", "all"):
        if args.image:
            run_stacks(args.image, args.num_bands, args.sigma)
        else:
            paths = [ROOT / "starter" / "apple.jpeg", ROOT / "starter" / "orange.jpeg"]
            if all(path.exists() for path in paths):
                for path in paths:
                    run_stacks(path, args.num_bands, args.sigma)
            else:
                run_stacks(ROOT / "data" / "taj.jpg", args.num_bands, args.sigma)
                print("apple/orange inputs still needed for the course figures")
    if args.part in ("blend", "all"):
        run_blend(args.blend_a, args.blend_b, args.mask, args.num_bands, args.sigma)
