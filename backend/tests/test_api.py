"""API tests. They use an UNTRAINED resnet18 so they check plumbing and the
response contract only — never prediction quality."""
import importlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CLASSES = ["plastic", "paper", "metal", "organic", "ewaste", "glass", "hazardous"]


def _png(size=(320, 240), color=(40, 120, 200)) -> bytes:
    arr = np.zeros((size[1], size[0], 3), np.uint8)
    arr[:] = color
    arr[60:180, 100:220] = (230, 230, 230)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def _client(monkeypatch, tmp_path, with_model: bool):
    from fastapi.testclient import TestClient

    if with_model:
        torch.manual_seed(0)
        m = torchvision.models.resnet18(weights=None, num_classes=7)
        torch.save(m.state_dict(), tmp_path / "w.pth")
        (tmp_path / "model_config.json").write_text(
            json.dumps({"name": "test", "checkpoint": "w.pth", "architecture": "resnet18", "class_names": CLASSES, "uncertainty_threshold": 0.0})
        )
    monkeypatch.setenv("ECOSORT_MODEL_CONFIG", str(tmp_path / "model_config.json"))
    monkeypatch.setenv("ECOSORT_BACKEND", "auto")
    # Plumbing tests use the untrained test classifier only: disable the bundled
    # YOLO detector and binary gate that backend/.env enables for real use.
    monkeypatch.setenv("ECOSORT_DETECTOR_WEIGHTS", "")
    monkeypatch.setenv("ECOSORT_BINARY_CHECKPOINT", str(tmp_path / "no-binary.pth"))
    monkeypatch.setenv("ECOSORT_EVAL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ECOSORT_LIME_SAMPLES", "40")
    monkeypatch.setenv("ECOSORT_SHAP_SAMPLES", "60")
    for mod in [m for m in list(sys.modules) if m.startswith("app")]:
        del sys.modules[mod]
    main = importlib.import_module("app.main")
    return TestClient(main.app)


def test_no_model_is_reported_honestly(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, with_model=False) as c:
        h = c.get("/api/health").json()
        assert h["model_loaded"] is False and "No trained model" in h["model_error"]
        r = c.post("/api/analyze", files={"file": ("a.png", _png(), "image/png")})
        assert r.status_code == 503 and r.json()["detail"]["code"] == "model_not_loaded"


def test_validation(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, with_model=True) as c:
        assert c.post("/api/analyze", files={"file": ("a.gif", b"GIF89a", "image/gif")}).status_code == 415
        assert c.post("/api/analyze", files={"file": ("a.png", b"", "image/png")}).status_code == 400
        assert c.post("/api/analyze", files={"file": ("a.png", b"not an image", "image/png")}).status_code == 400
        r = c.post("/api/analyze", files={"file": ("a.png", _png(), "image/png")}, data={"explain": "magic"})
        assert r.status_code == 422


def test_analyze_contract_and_xai(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, with_model=True) as c:
        assert c.get("/api/health").json()["model_loaded"] is True
        r = c.post("/api/analyze", files={"file": ("a.png", _png(), "image/png")}, data={"explain": "gradcam,lime,shap"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["summary"]["object_count"] == 1
        o = body["objects"][0]
        p = o["prediction"]
        assert 0 <= p["confidence"] <= 1 and len(p["top_k"]) == 3
        assert abs(sum(k["confidence"] for k in p["top_k"])) <= 1.0001
        for k in ("gradcam", "lime", "shap"):
            assert o["xai"][k]["available"], o["xai"][k]
            assert o["xai"][k]["image"].startswith("data:image/png;base64,")
        assert o["explanation"]["summary"] and o["recommendation"]["destination"]


def test_multiple_detections_are_classified_separately(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, with_model=True) as c:
        from app.models.detector import Detection
        from app.models.registry import registry

        class StubDetector:
            name = "stub-detector"

            def detect(self, _img):
                return [Detection((10, 10, 150, 200), "bottle", 0.9), Detection((170, 40, 310, 230), "can", 0.8)]

        registry.detector = StubDetector()
        r = c.post("/api/analyze", files={"file": ("a.png", _png(), "image/png")}, data={"explain": "gradcam"})
        body = r.json()
        assert body["summary"]["object_count"] == 2 and body["model"]["detector"]["name"] == "stub-detector"
        assert [o["bbox"] for o in body["objects"]] == [[10, 10, 150, 200], [170, 40, 310, 230]]
        assert all(o["source"] == "detector" for o in body["objects"])
        out = os.environ.get("ECOSORT_DUMP_MULTI")
        if out:
            Path(out).write_text(json.dumps(body))
        registry.detector = None


def test_uncertain_when_below_threshold(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, with_model=True) as c:
        from app.models.registry import registry

        registry.classifier.uncertainty_threshold = 1.01  # nothing can pass
        r = c.post("/api/analyze", files={"file": ("a.png", _png(), "image/png")}, data={"explain": ""})
        o = r.json()["objects"][0]
        assert o["prediction"]["uncertain"] is True and o["prediction"]["category"] == "unknown"
        assert r.json()["summary"]["uncertain"] is True
