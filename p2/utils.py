import numpy as np
from PIL import Image


def load_image(path, grayscale=False):
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba)
        image = image.convert("L" if grayscale else "RGB")
        return np.asarray(image, dtype=np.float64) / 255


def save_grid(images, titles, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4), squeeze=False)
    for ax, image, title in zip(axes[0], images, titles):
        ax.imshow(np.clip(image, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
