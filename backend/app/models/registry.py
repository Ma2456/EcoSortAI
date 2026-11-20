"""Loads the configured classifier/detector once at startup and records why if it can't."""
from __future__ import annotations

import logging

from ..settings import settings
from .base import Classifier

log = logging.getLogger("ecosort.models")


class Registry:
    def __init__(self) -> None:
        self.classifier: Classifier | None = None
        self.detector = None
        self.binary_filter = None
        self.binary_error: str | None = None
        self.model_error: str | None = None
        self.detector_error: str | None = None

    def load(self) -> None:
        mode = settings.backend
        try:
            if mode in ("custom", "auto"):
                if settings.model_config.exists():
                    from .torchvision_classifier import TorchvisionClassifier

                    self.classifier = TorchvisionClassifier(settings.model_config, settings.device)
                elif mode == "custom":
                    raise FileNotFoundError(f"{settings.model_config} not found")
                else:
                    self.model_error = (
                        "No trained model configured. Add your checkpoint and model_config.json to backend/model/ "
                        "(see backend/model/README.md)."
                    )
            elif mode == "clip":
                from .clip_zeroshot import ClipZeroShotClassifier

                self.classifier = ClipZeroShotClassifier(device=settings.device)
            elif mode == "none":
                self.model_error = "Model loading disabled (ECOSORT_BACKEND=none)."
            else:
                self.model_error = f"Unknown ECOSORT_BACKEND '{mode}'"
        except Exception as exc:  # report, don't crash the API
            log.exception("Model failed to load")
            self.classifier = None
            self.model_error = f"Model failed to load: {exc}"

        if self.classifier is not None:
            log.info("Loaded classifier %s (%s)", self.classifier.name, self.classifier.kind)

        if settings.binary_checkpoint.exists() and self.classifier is not None:
            try:
                from .binary_filter import BinaryWasteFilter
                self.binary_filter = BinaryWasteFilter(
                    settings.binary_checkpoint,
                    settings.device,
                    settings.binary_threshold,
                )
                log.info("Loaded binary waste filter (%s)", self.binary_filter.class_names)
            except Exception as exc:
                log.exception("Binary waste filter failed to load")
                self.binary_error = f"Binary waste filter failed to load: {exc}"

        if settings.detector_weights:
            try:
                from .detector import YoloDetector

                self.detector = YoloDetector(settings.detector_weights, settings.detector_conf, settings.detector_iou, settings.max_objects)
            except Exception as exc:
                log.exception("Detector failed to load")
                self.detector_error = f"Detector failed to load: {exc}"


registry = Registry()
