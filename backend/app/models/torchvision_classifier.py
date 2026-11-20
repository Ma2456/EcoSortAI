"""Loads an existing trained PyTorch image classifier from backend/model/.

Supported checkpoint formats (set "format" in model_config.json):
  state_dict   — weights for a torchvision architecture ("architecture": "resnet50", …)
  full         — a pickled nn.Module saved with torch.save(model) (trusted files only)
  torchscript  — torch.jit.save output (Grad-CAM not available for scripted models)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ..categories import CATEGORIES, UNKNOWN, resolve_category
from .base import Classifier

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ModelConfigError(RuntimeError):
    pass


def pick_device(pref: str) -> torch.device:
    if pref and pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchvisionClassifier(Classifier):
    kind = "custom-checkpoint"

    def __init__(self, config_path: Path, device: str = "auto"):
        if not config_path.exists():
            raise ModelConfigError(f"Model config not found: {config_path}")
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        base = config_path.parent
        self.cfg = cfg
        self.name = cfg.get("name", "custom-classifier")
        self.class_names = list(cfg["class_names"])
        mapping = cfg.get("class_to_category", {})
        self.class_to_category = {c: resolve_category(c, mapping) for c in self.class_names}
        bad = [c for c, cat in self.class_to_category.items() if cat not in CATEGORIES and cat != UNKNOWN]
        if bad:
            raise ModelConfigError(f"class_to_category maps to unknown categories: {bad}")
        self.input_size = int(cfg.get("input_size", 224))
        self.mean = cfg.get("mean", IMAGENET_MEAN)
        self.std = cfg.get("std", IMAGENET_STD)
        self.uncertainty_threshold = float(cfg.get("uncertainty_threshold", 0.55))
        self.resize_mode = cfg.get("resize", "center_crop")  # or "stretch"
        self.device = pick_device(device)

        ckpt = Path(cfg["checkpoint"])
        ckpt = ckpt if ckpt.is_absolute() else base / ckpt
        if not ckpt.exists():
            raise ModelConfigError(f"Checkpoint not found: {ckpt}")
        fmt = cfg.get("format", "state_dict")
        self.model = self._load(ckpt, fmt, cfg).to(self.device).eval()
        for m in self.model.modules():  # in-place activations break Grad-CAM hooks
            if hasattr(m, "inplace") and isinstance(getattr(m, "inplace"), bool):
                m.inplace = False

        self.supports_gradcam = fmt != "torchscript"
        self.target_layer = None
        if self.supports_gradcam:
            self.target_layer = self._find_target_layer(cfg.get("gradcam_target_layer"))
            self.supports_gradcam = self.target_layer is not None
            self.target_layer_name = next((n for n, m in self.model.named_modules() if m is self.target_layer), None)

        with torch.no_grad():  # sanity check: output size matches class list
            out = self.model(torch.zeros(1, 3, self.input_size, self.input_size, device=self.device))
            out = out[0] if isinstance(out, (tuple, list)) else out
            if out.shape[-1] != len(self.class_names):
                raise ModelConfigError(
                    f"Model outputs {out.shape[-1]} classes but class_names lists {len(self.class_names)}"
                )

    # ------------------------------------------------------------------ load
    def _load(self, ckpt: Path, fmt: str, cfg: dict) -> nn.Module:
        if fmt == "torchscript":
            return torch.jit.load(str(ckpt), map_location="cpu")
        if fmt == "full":
            return torch.load(str(ckpt), map_location="cpu", weights_only=False)
        if fmt != "state_dict":
            raise ModelConfigError(f"Unknown checkpoint format '{fmt}'")

        import torchvision.models as tvm

        arch = cfg.get("architecture")
        if not arch or not hasattr(tvm, arch):
            raise ModelConfigError(f"Unknown torchvision architecture '{arch}'")
        model = getattr(tvm, arch)(weights=None, num_classes=len(self.class_names))
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        key = cfg.get("state_dict_key")
        if key:
            state = state[key]
        elif isinstance(state, dict) and not any(torch.is_tensor(v) for v in state.values()):
            for k in ("model_state_dict", "state_dict", "model"):
                if k in state:
                    state = state[k]
                    break
        prefix = cfg.get("strip_prefix", "module.")
        if prefix:
            state = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ModelConfigError(
                "Checkpoint does not match the architecture. "
                f"Missing keys: {missing[:6]}{'…' if len(missing) > 6 else ''}; "
                f"unexpected keys: {unexpected[:6]}{'…' if len(unexpected) > 6 else ''}. "
                "If you changed the classifier head when training, save the full model (format 'full') or TorchScript."
            )
        return model

    def _find_target_layer(self, name: str | None) -> nn.Module | None:
        if name:
            mod = self.model
            for part in name.split("."):
                mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
            return mod
        last = None
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        return last

    # ------------------------------------------------------------ inference
    def _prep(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        s = self.input_size
        if img.size == (s, s):
            pass  # already the model view (LIME/SHAP perturbations)
        elif self.resize_mode == "stretch":
            img = img.resize((s, s), Image.BILINEAR)
        else:  # resize shorter side to s/0.875 then centre-crop s (standard eval transform)
            short = int(round(s / 0.875))
            w, h = img.size
            scale = short / min(w, h)
            img = img.resize((max(s, round(w * scale)), max(s, round(h * scale))), Image.BILINEAR)
            w, h = img.size
            l, t = (w - s) // 2, (h - s) // 2
            img = img.crop((l, t, l + s, t + s))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
        return torch.from_numpy(arr.transpose(2, 0, 1))

    def model_view(self, img: Image.Image) -> Image.Image:
        """The exact RGB crop the network sees (used for overlays)."""
        t = self._prep(img).numpy().transpose(1, 2, 0)
        t = t * np.array(self.std) + np.array(self.mean)
        return Image.fromarray(np.clip(t * 255, 0, 255).astype(np.uint8))

    @torch.no_grad()
    def predict_proba(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self._prep(i) for i in images]).to(self.device)
        out = self.model(batch)
        out = out[0] if isinstance(out, (tuple, list)) else out
        return F.softmax(out.float(), dim=1).cpu().numpy()

    def gradcam(self, image: Image.Image, class_index: int) -> np.ndarray:
        acts: dict[str, torch.Tensor] = {}

        def fwd_hook(_m, _i, o):
            acts["a"] = o
            o.register_hook(lambda g: acts.__setitem__("g", g))

        h = self.target_layer.register_forward_hook(fwd_hook)
        try:
            x = self._prep(image).unsqueeze(0).to(self.device).requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                out = self.model(x)
                out = out[0] if isinstance(out, (tuple, list)) else out
                out[0, class_index].backward()
        finally:
            h.remove()
        a, g = acts["a"].detach()[0], acts["g"].detach()[0]
        weights = g.mean(dim=(1, 2))
        cam = F.relu((weights[:, None, None] * a).sum(0))
        cam = F.interpolate(cam[None, None], size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam
