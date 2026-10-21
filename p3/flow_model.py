"""Provided PixNerd adapter: float32 states, text encoding and model evaluation."""
import sys
import os
import hashlib
import json
from pathlib import Path
import torch
from PIL import Image, ImageOps
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_ID = "MCG-NJU/PixNerd-XXL-P16-T2I"
MODEL_REVISION = "9fa8836b553976b88a3b27649ed1628499e02bf9"
TEXT_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
RESOLUTION = 512


def preprocess(image):
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    image = ImageOps.fit(image.convert("RGB"), (RESOLUTION, RESOLUTION), method=Image.Resampling.LANCZOS)
    return torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1


def to_pil(x):
    if x.ndim == 4:
        x = x[0]
    a = ((x.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte()
    return Image.fromarray(a.permute(1, 2, 0).numpy())


class FlowModel:
    def __init__(self, device="cuda:0"):
        from huggingface_hub import hf_hub_download
        sys.path.insert(0, str(ROOT / "vendor" / "PixNerd"))
        from src.models.transformer.pixnerd_t2i import PixNerDiT
        from transformers import Qwen3Model, Qwen2Tokenizer
        self.device = torch.device(device)
        self.net = PixNerDiT(in_channels=3, patch_size=16, num_groups=24,
            hidden_size=1536, txt_embed_dim=2048, txt_max_length=128,
            num_text_blocks=4, decoder_hidden_size=64, num_encoder_blocks=16, num_decoder_blocks=2)
        path = hf_hub_download(MODEL_ID, "model.ckpt", revision=MODEL_REVISION)
        # Released Lightning checkpoint contains metadata as well as tensor weights.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = {k.removeprefix("ema_denoiser."): v for k, v in checkpoint["state_dict"].items()
                 if k.startswith("ema_denoiser.")}
        self.net.load_state_dict(state, strict=True)
        del checkpoint, state
        self.net.eval().requires_grad_(False).to(device=self.device, dtype=torch.float32)
        self.tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=TEXT_REVISION, padding_side="right")
        self.encoder = Qwen3Model.from_pretrained("Qwen/Qwen3-1.7B", revision=TEXT_REVISION,
            torch_dtype=torch.float32).eval().requires_grad_(False).to(self.device)
        self.cache = {}
        self.prompt_cache = Path(os.environ.get('CS180_PROMPT_CACHE',str(ROOT/'.cache/prompt_embeddings')))
        self.prompt_cache.mkdir(parents=True,exist_ok=True)
        self.encode("")

    @torch.inference_mode()
    def encode(self, prompt):
        if prompt not in self.cache:
            key = hashlib.sha256(json.dumps(dict(prompt=prompt,revision=TEXT_REVISION,
                tokenizer='Qwen2Tokenizer',length=128,padding_side='right',dtype='float32',
                transformers=__import__('transformers').__version__,torch=str(torch.__version__)),
                sort_keys=True).encode()).hexdigest()
            path = self.prompt_cache / (key+'.pt')
            if path.is_file():
                embedding = torch.load(path,map_location='cpu',weights_only=True)
                if embedding.shape == (1,128,2048) and embedding.dtype == torch.float32 and torch.isfinite(embedding).all():
                    self.cache[prompt] = embedding.to(self.device)
                    return self.cache[prompt]
            tokens = self.tokenizer([prompt], truncation=True, max_length=128,
                padding="max_length", return_tensors="pt").to(self.device)
            self.cache[prompt] = self.encoder(**tokens).last_hidden_state.detach()
            temp=path.with_suffix('.tmp')
            torch.save(self.cache[prompt].cpu(),temp)
            temp.replace(path)
        return self.cache[prompt]

    @torch.inference_mode()
    def predict_velocity(self, x, t, condition):
        """Native model is already dx/dt, t=0 noise, t=1 clean; no sign flip."""
        if x.device != self.device or x.ndim != 4 or x.shape[1:] != (3, 512, 512):
            raise ValueError("Expected RGB state (B,3,512,512) on model.device")
        ts = torch.as_tensor(t, device=self.device, dtype=torch.float32).reshape(-1)
        ts = ts.expand(x.shape[0])
        if not bool(((ts >= 0) & (ts <= 1)).all()):
            raise ValueError("Flow time must be in [0,1]")
        if isinstance(condition, str):
            condition = self.encode(condition)
        condition = condition.expand(x.shape[0], -1, -1)
        with torch.autocast("cuda", enabled=False):
            v = self.net(x.float(), ts, condition.float())
        return v.to(x.dtype)
