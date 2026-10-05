import cv2
import numpy as np
import matplotlib.pyplot as plt

from main import ROOT, run_blend
from utils import load_image, save_grid


def run_eye_moon():
    eye = load_image(ROOT / "data" / "eye.png")
    moon = load_image(ROOT / "data" / "moon.jpeg")
    out = ROOT / "out" / "p2_4" / "moon_eye"
    prepared = out / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    save_grid([eye, moon], ["original eye", "original moon"], out / "originals.png")

    scale = 640 / eye.shape[1]
    eye = cv2.resize(eye, (640, round(eye.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    h, w = eye.shape[:2]
    center = np.array([566, 738]) * scale
    radius = 125 * scale
    moon_scale = radius / 270
    shift = center - np.array([436, 623]) * moon_scale
    matrix = np.array([[moon_scale, 0, shift[0]], [0, moon_scale, shift[1]]])
    moon = cv2.warpAffine(moon, matrix, (w, h))

    y, x = np.ogrid[:h, :w]
    mask = ((x - center[0])**2 + (y - center[1])**2 <= radius**2)
    # keep the moon below the upper lid
    outline = np.array([[278, 819], [347, 739], [432, 667], [518, 642],
                        [601, 644], [697, 649], [798, 681], [884, 704],
                        [795, 770], [705, 820], [614, 847], [515, 853],
                        [401, 845], [308, 837]])
    opening = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(opening, [np.round(outline * scale).astype(np.int32)], 1)
    mask = (mask * opening).astype(np.uint8)

    # match the iris color
    gray = moon @ np.array([0.299, 0.587, 0.114])
    gamma = 1.8
    gray = np.clip(gray, 0, 1)**gamma
    region = mask.astype(bool)
    iris_color = eye[region].mean(axis=0)
    texture = (gray - gray[region].mean()) / (gray[region].std() + 1e-8)
    shading = cv2.GaussianBlur(eye, (0, 0), 4)
    mapped = iris_color + 0.07 + 0.055 * texture[..., None] + 0.5 * (shading - iris_color)
    moon = np.clip(mapped, 0, 1)

    # keep the pupil and reflection
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    mask = np.clip(distance / 8, 0, 1)
    pupil_center = np.array([557, 746]) * scale
    pupil_distance = np.hypot(x - pupil_center[0], y - pupil_center[1])
    mask *= np.clip((pupil_distance - 32 * scale) / (18 * scale), 0, 1)
    reflection = (eye.mean(axis=2) > 0.4).astype(float)
    reflection = cv2.GaussianBlur(reflection, (0, 0), 1.5)
    mask *= 1 - reflection
    moon = np.where(region[..., None], moon, eye)

    plt.imsave(prepared / "eye.png", eye)
    plt.imsave(prepared / "moon.png", moon)
    plt.imsave(prepared / "mask.png", mask, cmap="gray", vmin=0, vmax=1)
    run_blend(prepared / "moon.png", prepared / "eye.png", prepared / "mask.png", sigma=2.0)
    (out / "alignment.txt").write_text(
        "sources: p2/data/eye.png, p2/data/moon.jpeg\n"
        "eye width: 640\n"
        "iris center (original x, y): (566, 738)\niris radius: 125\n"
        "moon center (original x, y): (436, 623)\nmoon radius: 270\n"
        "mask: circle intersected with the eye opening\n"
        "gamma: 1.8, grayscale mapped to iris color, brightness: +0.07, texture: 0.055\n"
        "mask feather: 8 pixels, pupil and reflection preserved\n"
    )


if __name__ == "__main__":
    run_eye_moon()
