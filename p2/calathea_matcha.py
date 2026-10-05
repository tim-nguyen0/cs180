import cv2
import numpy as np
from PIL import Image, ImageOps

from main import ROOT, run_blend
import matplotlib.pyplot as plt
from utils import load_image, save_grid


def run_calathea_matcha():
    plant = load_image(ROOT / "data" / "calathea.jpeg")
    with Image.open(ROOT / "data" / "matcha2.jpeg") as image:
        drink = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=float) / 255
    out = ROOT / "out" / "p2_4" / "calathea_matcha"
    prepared = out / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    save_grid([plant, drink], ["original calathea", "original matcha"], out / "originals.png")

    scale = 640 / plant.shape[1]
    plant = cv2.resize(plant, (640, round(plant.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    drink = cv2.resize(drink, (640, round(drink.shape[0] * 640 / drink.shape[1])),
                       interpolation=cv2.INTER_AREA)
    h, w = plant.shape[:2]
    matrix = np.array([[1.2, 0, 320 - 300 * 1.2],
                       [0, 1.2, 665 - 235 * 1.2]])
    background = cv2.warpAffine(drink, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    top = cv2.resize(drink[:140], (w, 443))
    top = cv2.GaussianBlur(top, (0, 0), 18)
    fade = np.clip((443 - np.arange(443)) / 60, 0, 1)[:, None, None]
    background[:443] = fade * top + (1 - fade) * background[:443]
    background = np.clip(background, 0, 1)

    # keep the leaves and stems
    hsv = cv2.cvtColor(plant.astype(np.float32), cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 0] > 50) & (hsv[..., 0] < 105) & (hsv[..., 1] > 0.3)).astype(np.uint8)
    outline = np.array([[421, 119], [487, 125], [476, 207], [526, 210],
                        [542, 298], [585, 300], [618, 367], [570, 401],
                        [525, 370], [517, 449], [571, 489], [631, 523],
                        [628, 607], [596, 643], [653, 631], [670, 592],
                        [739, 597], [744, 690], [710, 718], [670, 702],
                        [634, 742], [594, 832], [574, 866], [488, 900],
                        [405, 910], [320, 916], [228, 887], [217, 798],
                        [176, 754], [169, 694], [192, 650], [234, 628],
                        [265, 628], [275, 576], [237, 565], [246, 539],
                        [296, 530], [287, 509], [266, 463], [273, 406],
                        [315, 360], [306, 337], [337, 317], [383, 315],
                        [396, 269], [386, 241], [391, 175]])
    area = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(area, [np.round(outline * scale).astype(np.int32)], 1)
    mask *= area
    mask[665:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] > 50
    mask = keep[labels].astype(np.uint8)
    mask = cv2.GaussianBlur(mask.astype(float), (0, 0), 0.7)
    y = np.arange(h)[:, None]
    mask *= np.clip((665 - y) / 15, 0, 1)
    plant = np.clip(plant, 0, 1)**1.15
    plant = np.where((mask > 0.01)[..., None], plant, background)

    plt.imsave(prepared / "calathea.png", plant)
    plt.imsave(prepared / "matcha.png", background)
    plt.imsave(prepared / "mask.png", mask, cmap="gray", vmin=0, vmax=1)
    run_blend(prepared / "calathea.png", prepared / "matcha.png",
              prepared / "mask.png", sigma=1.0)
    (out / "alignment.txt").write_text(
        "sources: p2/data/calathea.jpeg, p2/data/matcha2.jpeg\n"
        "plant width: 640; matcha width: 640 before scaling by 1.2\n"
        "matcha orientation corrected; upper background extended and blurred\n"
        "matcha lid center mapped from (300, 235) to (320, 665)\n"
        "mask: green leaves and stems, ending at the lid, feathered 0.7 pixels\n"
        "plant gamma: 1.15\n"
    )


if __name__ == "__main__":
    run_calathea_matcha()
