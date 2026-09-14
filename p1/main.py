from pathlib import Path
import align as align
from datetime import datetime



images_path = "data"
extensions = {".jpg", ".tif"}

paths = sorted(str(p) for p in Path(images_path).iterdir() if p.suffix.lower() in extensions)

#out_path = "out/multiple_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_path = "out_numpy"
Path(out_path).mkdir(parents=True, exist_ok=True)

align.align_and_save_multiple(paths, out_path, max_offset_initial=200, max_offset_step=10, crop_frac=0.4)