"""OPTIONAL object detector (Ultralytics YOLO). Enabled when ECOSORT_DETECTOR_WEIGHTS
points to a .pt file (e.g. your own waste-trained YOLO weights). Each detection is
cropped and passed to the classifier, so every object gets its own result."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    label: str
    confidence: float


class YoloDetector:
    def __init__(self, weights: str, conf: float = 0.05, iou: float = 0.45, max_det: int = 30):
        from ultralytics import YOLO  # optional dependency

        if not Path(weights).exists():
            raise FileNotFoundError(f"Detector weights not found: {weights}")
        self.model = YOLO(weights)
        self.name = Path(weights).stem
        self.conf = conf
        self.iou = iou
        self.max_det = max_det

    def detect(self, image: Image.Image) -> list[Detection]:
        res = self.model.predict(image, conf=self.conf, iou=self.iou, agnostic_nms=True, max_det=self.max_det, verbose=False)[0]
        names = res.names
        out = []
        for box in res.boxes:
            x0, y0, x1, y1 = (int(round(v)) for v in box.xyxy[0].tolist())
            out.append(Detection((x0, y0, x1, y1), names[int(box.cls)], float(box.conf)))
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out[: self.max_det]
