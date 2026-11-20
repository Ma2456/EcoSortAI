"""Binary waste/non-waste gate matching EcoSortAI Hybrid V2."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

from .torchvision_classifier import pick_device


class BinaryWasteFilter:
    def __init__(self, checkpoint: Path, device: str = "auto", threshold: float = 0.35):
        self.device = pick_device(device)
        self.threshold = float(threshold)
        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        self.class_names = list(ckpt.get("class_names", ["non_waste", "waste"]))
        if "waste" not in self.class_names:
            raise ValueError(f"Binary checkpoint has no 'waste' class: {self.class_names}")

        self.model = models.mobilenet_v3_small(weights=None)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, len(self.class_names))
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

        # Same ImageNet normalization and 224px input used by the project classifier.
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.waste_index = self.class_names.index("waste")

    def _tensor(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize((224, 224), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr.transpose(2, 0, 1))
        return (x - self.mean[0]) / self.std[0]

    @torch.inference_mode()
    def predict_waste_probability(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self._tensor(im) for im in images]).to(self.device)
        logits = self.model(batch)
        return F.softmax(logits.float(), dim=1)[:, self.waste_index].cpu().numpy()
