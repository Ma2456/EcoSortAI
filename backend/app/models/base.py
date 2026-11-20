"""Classifier interface shared by every model backend."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class Classifier(ABC):
    name: str = "classifier"
    kind: str = "custom-checkpoint"  # or "zero-shot-baseline"
    class_names: list[str] = []
    class_to_category: dict[str, str] = {}
    input_size: int = 224
    uncertainty_threshold: float = 0.55
    supports_gradcam: bool = False

    @abstractmethod
    def predict_proba(self, images: list[Image.Image]) -> np.ndarray:
        """Return an (N, C) array of class probabilities for N RGB images."""

    def predict_proba_arrays(self, arrays: np.ndarray) -> np.ndarray:
        """(N, H, W, 3) uint8/float arrays → (N, C) probabilities. Used by LIME/SHAP."""
        imgs = [Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)) for a in arrays]
        out = []
        for i in range(0, len(imgs), 32):
            out.append(self.predict_proba(imgs[i : i + 32]))
        return np.concatenate(out, axis=0)

    def gradcam(self, image: Image.Image, class_index: int) -> np.ndarray:  # pragma: no cover - optional
        """Return an (H, W) heatmap in [0, 1] aligned to the model input crop."""
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "classes": self.class_names,
            "class_to_category": self.class_to_category,
            "input_size": self.input_size,
            "uncertainty_threshold": self.uncertainty_threshold,
            "gradcam": self.supports_gradcam,
        }
