import numpy as np
from PIL import Image


def load_image(path, grayscale=False):
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba)
        image = image.convert("L" if grayscale else "RGB")
        return np.asarray(image, dtype=np.float64) / 255
