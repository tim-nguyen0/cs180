from pathlib import Path
import align as align
from datetime import datetime


# align.USE_CUDA=False # modify here if align import is align-cuda

images_path = "data"
extensions = {".jpg", ".tif"}

paths = sorted(str(p) for p in Path(images_path).iterdir() if p.suffix.lower() in extensions)

#out_path = "out/multiple_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_path = "out_numpy_0.35cf"
Path(out_path).mkdir(parents=True, exist_ok=True)

align.align_and_save_multiple(paths, out_path, max_offset_initial=250, max_offset_step=10, crop_frac=0.33)
