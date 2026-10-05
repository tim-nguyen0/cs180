import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .edges import finite_difference, dog_filters, derivative_of_gaussian
    from .filters import gaussian_blur, gaussian_kernel
    from .utils import load_image
else:
    from edges import finite_difference, dog_filters, derivative_of_gaussian
    from filters import gaussian_blur, gaussian_kernel
    from utils import load_image


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=ROOT / "data" / "cameraman.png")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--part", choices=["edges", "dog", "all"], default="all")
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--size", type=int)
    parser.add_argument("--dog-threshold", type=float, default=0.04)
    args = parser.parse_args()
    if args.part in ("edges", "all"):
        run_edges(args.image, args.threshold)
    if args.part in ("dog", "all"):
        run_dog(args.image, args.sigma, args.size, args.dog_threshold, args.threshold)
