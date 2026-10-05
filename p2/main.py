import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .edges import finite_difference
    from .utils import load_image
else:
    from edges import finite_difference
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=ROOT / "data" / "cameraman.png")
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()
    run_edges(args.image, args.threshold)
