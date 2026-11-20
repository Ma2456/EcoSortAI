"""Detection → classification → explanation → recommendation for one image."""
from __future__ import annotations

import time
import uuid

import numpy as np
from PIL import Image

from .categories import CATEGORIES, UNKNOWN, category_label, recommendation
from .models.base import Classifier
from .settings import settings
from .xai.explainers import run_gradcam, run_lime, run_shap


def _category_scores(clf: Classifier, probs: np.ndarray) -> list[dict]:
    """Sum raw-class probabilities into categories; remember the strongest raw class."""
    agg: dict[str, dict] = {}
    for i, p in enumerate(probs):
        raw = clf.class_names[i]
        cat = clf.class_to_category.get(raw, UNKNOWN)
        e = agg.setdefault(cat, {"category": cat, "confidence": 0.0, "raw_label": raw, "raw_index": i, "raw_p": -1.0})
        e["confidence"] += float(p)
        if p > e["raw_p"]:
            e.update(raw_label=raw, raw_index=i, raw_p=float(p))
    out = sorted(agg.values(), key=lambda e: -e["confidence"])
    for e in out:
        e["category_label"] = category_label(e["category"])
    return out


def _explanation(pred: dict, scores: list[dict], xai: dict, image_size: tuple[int, int], clf: Classifier) -> dict:
    top = scores[0]
    second = scores[1] if len(scores) > 1 else None
    conf = top["confidence"]
    if pred["uncertain"]:
        summary = (
            f"The model's highest score is {conf:.0%} for {top['category_label']}, below the {clf.uncertainty_threshold:.0%} "
            "confidence threshold, so EcoSortAI does not assign a category."
        )
    else:
        summary = f"The model assigns {conf:.0%} probability to {top['category_label']}"
        summary += f"; the runner-up is {second['category_label']} at {second['confidence']:.0%}." if second else "."
    evidence: list[str] = []
    g = xai.get("gradcam", {})
    if g.get("available"):
        evidence.append(
            f"Grad-CAM: activation is concentrated in the {g['focus_region']} of the crop "
            f"({g['focus_share']:.0%} of the heat lies in the hottest 20% of pixels)."
        )
    l = xai.get("lime", {})
    if l.get("available"):
        s = f"LIME: {l['n_supporting']} of the 8 most influential superpixels support the prediction"
        if l["supporting_regions"]:
            s += f" (mainly {', '.join(l['supporting_regions'])})"
        if l["n_opposing"]:
            s += f"; {l['n_opposing']} oppose it" + (f" ({', '.join(l['opposing_regions'])})" if l["opposing_regions"] else "")
        evidence.append(s + ".")
    sh = xai.get("shap", {})
    if sh.get("available") and sh.get("contributions"):
        best = max(sh["contributions"], key=lambda c: c["value"])
        worst = min(sh["contributions"], key=lambda c: c["value"])
        s = f"SHAP: the largest positive contribution comes from the {best['region'].split(' #')[0]} ({best['value']:+.3f})"
        if worst["value"] < 0:
            s += f"; the largest negative from the {worst['region'].split(' #')[0]} ({worst['value']:+.3f})"
        evidence.append(s + ".")

    limitations = ["Explanations show where the model looked, not whether that evidence is correct."]
    if second and conf - second["confidence"] < 0.2:
        limitations.append(f"Small margin between {top['category_label']} and {second['category_label']} — treat this result with caution.")
    if min(image_size) < clf.input_size:
        limitations.append("The object region is smaller than the model's input size, so fine detail is limited.")
    if top["category"] in CATEGORIES:
        limitations.append(CATEGORIES[top["category"]]["limitation"])
    if clf.kind == "zero-shot-baseline":
        limitations.append("Zero-shot baseline: this model was not trained on waste images.")
    return {"summary": summary, "evidence": evidence, "limitations": limitations}


def _crop(img: Image.Image, bbox: tuple[int, int, int, int], pad: float = 0.06) -> Image.Image:
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    return img.crop(
        (
            max(0, int(x0 - w * pad)),
            max(0, int(y0 - h * pad)),
            min(img.width, int(x1 + w * pad)),
            min(img.height, int(y1 + h * pad)),
        )
    )


def analyze(image: Image.Image, clf: Classifier, detector=None, explain: tuple[str, ...] = (), binary_filter=None) -> dict:
    t0 = time.perf_counter()
    timing: dict[str, float] = {}
    warnings: list[str] = []

    regions: list[dict] = []
    if detector is not None:
        td = time.perf_counter()
        dets = detector.detect(image)
        timing["detection"] = round((time.perf_counter() - td) * 1000)
        for d in dets:
            regions.append({"bbox": list(d.bbox), "source": "detector", "detector": {"label": d.label, "confidence": round(d.confidence, 4)}})
        if not regions:
            warnings.append("YOLO found no objects at the configured confidence threshold.")
    else:
        regions.append({"bbox": None, "source": "whole-image", "detector": None})

    # Match the original Hybrid V2 gate: YOLO crop -> binary waste probability
    # -> retain only crops with P(waste) >= 0.35 -> seven-class CNN.
    rejected_count = 0
    crops = [_crop(image, tuple(r["bbox"])) if r["bbox"] else image for r in regions]
    waste_probabilities: list[float | None] = [None] * len(crops)
    if binary_filter is not None and crops:
        tb = time.perf_counter()
        waste_probabilities = binary_filter.predict_waste_probability(crops).tolist()
        keep = [
            i for i, prob in enumerate(waste_probabilities)
            if prob >= binary_filter.threshold
        ]
        rejected_count = len(crops) - len(keep)
        regions = [regions[i] for i in keep]
        crops = [crops[i] for i in keep]
        waste_probabilities = [waste_probabilities[i] for i in keep]
        timing["binary_filter"] = round((time.perf_counter() - tb) * 1000)
    elif binary_filter is None:
        warnings.append("Binary waste filter is not loaded; binary validation was skipped.")

    if detector is not None and not regions:
        timing["classification"] = 0
        timing["explanations"] = 0
        timing["total"] = round((time.perf_counter() - t0) * 1000)
        return {
            "request_id": uuid.uuid4().hex[:12],
            "demo": False,
            "model": {
                "name": clf.name,
                "kind": clf.kind,
                "uncertainty_threshold": clf.uncertainty_threshold,
                "detector": {"name": detector.name} if detector is not None else None,
                "binary_filter": {"name": "MobileNetV3 binary waste filter", "threshold": binary_filter.threshold} if binary_filter else None,
            },
            "image": {"width": image.width, "height": image.height},
            "summary": {
                "primary_category": UNKNOWN,
                "primary_label": "No waste detected",
                "confidence": 0.0,
                "uncertain": True,
                "object_count": 0,
                "candidate_count": len(regions) + rejected_count,
                "rejected_count": rejected_count,
            },
            "objects": [],
            "warnings": warnings,
            "timing_ms": timing,
        }

    tc = time.perf_counter()
    probs = clf.predict_proba(crops)
    timing["classification"] = round((time.perf_counter() - tc) * 1000)

    objects = []
    tx = time.perf_counter()
    for i, (r, crop, p) in enumerate(zip(regions, crops, probs)):
        scores = _category_scores(clf, p)
        top = scores[0]
        uncertain = top["category"] == UNKNOWN or top["confidence"] < clf.uncertainty_threshold
        pred = {
            "category": UNKNOWN if uncertain else top["category"],
            "category_label": "Uncertain" if uncertain else top["category_label"],
            "raw_label": top["raw_label"],
            "confidence": round(top["confidence"], 4),
            "uncertain": uncertain,
            "top_k": [
                {"category": s["category"], "category_label": s["category_label"], "raw_label": s["raw_label"], "confidence": round(s["confidence"], 4)}
                for s in scores[:3]
            ],
        }
        xai: dict = {}
        if i < settings.xai_max_objects and explain:
            view = clf.model_view(crop) if hasattr(clf, "model_view") else crop.resize((clf.input_size, clf.input_size))
            idx = top["raw_index"]
            if "gradcam" in explain:
                xai["gradcam"] = run_gradcam(clf, crop, view, idx)
            if "lime" in explain:
                xai["lime"] = run_lime(clf, view, idx, settings.lime_samples)
            if "shap" in explain:
                xai["shap"] = run_shap(clf, view, idx, settings.shap_samples)
        for k in ("gradcam", "lime", "shap"):
            if k not in xai:
                note = "Not requested." if k not in explain else f"Skipped: explanations are computed for the first {settings.xai_max_objects} objects only."
                xai[k] = {"available": False, "note": note}
        objects.append(
            {
                "id": f"obj-{i + 1}",
                "bbox": r["bbox"],
                "source": r["source"],
                "detector": r["detector"],
                "binary_waste_probability": round(float(waste_probabilities[i]), 4) if waste_probabilities[i] is not None else None,
                "prediction": pred,
                "explanation": _explanation(pred, scores, xai, crop.size, clf),
                "xai": xai,
                "recommendation": recommendation(pred["category"]),
            }
        )
    timing["explanations"] = round((time.perf_counter() - tx) * 1000)

    primary = max(objects, key=lambda o: (not o["prediction"]["uncertain"], o["prediction"]["confidence"]))
    timing["total"] = round((time.perf_counter() - t0) * 1000)
    return {
        "request_id": uuid.uuid4().hex[:12],
        "demo": False,
        "model": {
            "name": clf.name,
            "kind": clf.kind,
            "uncertainty_threshold": clf.uncertainty_threshold,
            "detector": {"name": detector.name, "confidence_threshold": settings.detector_conf, "iou_threshold": settings.detector_iou} if detector is not None else None,
            "binary_filter": {"name": "MobileNetV3 binary waste filter", "threshold": binary_filter.threshold} if binary_filter else None,
        },
        "image": {"width": image.width, "height": image.height},
        "summary": {
            "primary_category": primary["prediction"]["category"],
            "primary_label": primary["prediction"]["category_label"],
            "confidence": primary["prediction"]["confidence"],
            "uncertain": all(o["prediction"]["uncertain"] for o in objects),
            "object_count": len(objects),
            "candidate_count": len(regions) + rejected_count,
            "rejected_count": rejected_count,
        },
        "objects": objects,
        "warnings": warnings,
        "timing_ms": timing,
    }
