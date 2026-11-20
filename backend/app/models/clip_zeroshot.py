"""OPTIONAL zero-shot baseline using OpenAI CLIP via Hugging Face transformers.

This is NOT a trained waste classifier. It is offered only so the pipeline can
run end-to-end before a trained checkpoint exists, and every response is
labelled kind="zero-shot-baseline". Enable with ECOSORT_BACKEND=clip and
`pip install -r requirements-baseline.txt` (downloads ~600 MB on first run).
Grad-CAM is not available (ViT backbone); LIME and SHAP are model-agnostic and work.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .base import Classifier
from .torchvision_classifier import pick_device

PROMPTS: dict[str, list[str]] = {
    "plastic": ["a photo of plastic waste", "a photo of a used plastic bottle", "a photo of plastic wrappers and bags"],
    "paper": ["a photo of paper waste", "a photo of old newspapers", "a photo of cardboard boxes for recycling"],
    "metal": ["a photo of metal waste", "a photo of empty aluminium cans", "a photo of tin cans"],
    "organic": ["a photo of food waste", "a photo of fruit and vegetable peels", "a photo of compostable kitchen scraps"],
    "ewaste": ["a photo of electronic waste", "a photo of old phones, chargers and cables", "a photo of circuit boards"],
    "glass": ["a photo of glass waste", "a photo of empty glass bottles", "a photo of glass jars"],
    "hazardous": ["a photo of hazardous household waste", "a photo of paint and chemical containers with warning labels", "a photo of expired medicines and syringes"],
    "unknown": ["a photo of old clothes", "a photo of a person", "a photo of an empty room", "a photo of a landscape", "a photo of an animal"],
}


class ClipZeroShotClassifier(Classifier):
    kind = "zero-shot-baseline"
    supports_gradcam = False

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "auto"):
        from transformers import CLIPModel, CLIPProcessor  # optional dependency

        self.name = f"{model_id.split('/')[-1]} (zero-shot)"
        self.device = pick_device(device)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.class_names = list(PROMPTS.keys())
        self.class_to_category = {c: c for c in self.class_names}
        self.input_size = 224
        self.uncertainty_threshold = 0.5
        prompts = [p for c in self.class_names for p in PROMPTS[c]]
        self._owner = np.array([i for i, c in enumerate(self.class_names) for _ in PROMPTS[c]])
        with torch.no_grad():
            t = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
            emb = self.model.get_text_features(**t)
            self.text = emb / emb.norm(dim=-1, keepdim=True)

    def model_view(self, img: Image.Image) -> Image.Image:
        s = self.input_size
        w, h = img.size
        scale = s / min(w, h)
        img = img.convert("RGB").resize((max(s, round(w * scale)), max(s, round(h * scale))), Image.BICUBIC)
        w, h = img.size
        l, t = (w - s) // 2, (h - s) // 2
        return img.crop((l, t, l + s, t + s))

    @torch.no_grad()
    def predict_proba(self, images: list[Image.Image]) -> np.ndarray:
        inp = self.processor(images=[i.convert("RGB") for i in images], return_tensors="pt").to(self.device)
        emb = self.model.get_image_features(**inp)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        logits = (self.model.logit_scale.exp() * emb @ self.text.T).cpu().numpy()  # (N, P)
        # Average prompt logits per class, then softmax over classes.
        cls = np.stack([logits[:, self._owner == i].mean(1) for i in range(len(self.class_names))], 1)
        cls = cls - cls.max(1, keepdims=True)
        p = np.exp(cls)
        return p / p.sum(1, keepdims=True)
