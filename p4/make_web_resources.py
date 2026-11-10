"""Convert p4/out figures into web-sized JPEGs for docs/4/resources/."""
from pathlib import Path
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).parent
OUT, DATA = HERE / "out", HERE / "data"
RES = HERE.parent / "docs" / "4" / "resources"
QUALITY = 85


def save(src, dest, long_edge):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode == "RGBA":   # transparent canvas -> white
            bg = Image.new("RGBA", im.size, "white"); im = Image.alpha_composite(bg, im)
        im = im.convert("RGB")
        s = long_edge / max(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        im.save(dest, "JPEG", quality=QUALITY, progressive=True, optimize=True)
    print(dest.relative_to(RES))


def main():
    for n in ["1", "2", "3", "4", "5", "6", "7", "2-1", "2-2", "2-3", "2-4", "2-5"]:
        save(DATA / f"{n}.JPG", RES / "a1" / f"{n}.jpg", 600)
    save(DATA / "tissue.jpeg", RES / "a1" / "tissue.jpg", 600)
    for f in sorted((OUT / "a2").glob("corr_*.png")):
        save(f, RES / "a2" / f"{f.stem}.jpg", 1600)
    for stem in ["3", "2-5", "tissue"]:
        save(OUT / "a3" / f"{stem}_comparison.png", RES / "a3" / f"{stem}_comparison.jpg", 1700)
        save(OUT / "a3" / f"{stem}_bilinear.png", RES / "a3" / f"{stem}_bilinear.jpg", 1600)
    for s in ["set1", "set2", "flower"]:
        d = OUT / "a4" / s
        save(d / "sources.png", RES / "a4" / f"{s}_sources.jpg", 1600)
        save(d / "warped.png", RES / "a4" / f"{s}_warped.jpg", 1600)
        save(d / "weights.png", RES / "a4" / f"{s}_weights.jpg", 1600)
        save(d / "seams.png", RES / "a4" / f"{s}_seams.jpg", 1200)
        save(d / "mosaic_overwrite.jpg", RES / "a4" / f"{s}_overwrite.jpg", 2000)
        save(d / "mosaic_feather_average.jpg", RES / "a4" / f"{s}_average.jpg", 2000)
        save(d / "mosaic_two_band.jpg", RES / "a4" / f"{s}_two_band.jpg", 2400)
    for stem in ["2", "2-3", "flower-2"]:
        save(OUT / "b1" / f"{stem}_corners.png", RES / "b1" / f"{stem}_corners.jpg", 1800)
    for stem in ["2", "flower-2"]:
        save(OUT / "b2" / f"{stem}_descriptors.png", RES / "b2" / f"{stem}_descriptors.jpg", 1800)
    for f in sorted((OUT / "b3").glob("*_matches.png")):
        save(f, RES / "b3" / f"{f.stem}.jpg", 1800)
    for f in sorted((OUT / "b4").glob("*_ransac.png")):
        save(f, RES / "b4" / f"{f.stem}.jpg", 1800)
    for s in ["set1", "set2", "flower"]:
        d = OUT / "b5" / s
        if not d.exists():
            continue
        save(d / "mosaic_two_band.jpg", RES / "b5" / f"{s}_two_band.jpg", 2400)
        save(d / "manual_vs_auto.png", RES / "b5" / f"{s}_manual_vs_auto.jpg", 2000)
        save(d / "seams.png", RES / "b5" / f"{s}_seams.jpg", 1200)
        for f in sorted(d.glob("*_matches.png")):
            save(f, RES / "b5" / f"{s}_{f.stem}.jpg", 1600)


if __name__ == "__main__":
    main()
