"""Evaluation runners.

``timed_pipeline`` executes the SAME stages, components and thresholds as
``app.pipeline.analyze`` (YOLO → crop → binary gate → seven-class CNN →
Grad-CAM → recommendation) using the already-loaded model instances, but times
each stage separately with ``time.perf_counter`` (and CUDA synchronisation when
running on a GPU). ``tests/test_evaluation.py`` asserts that its predictions are
identical to ``pipeline.analyze`` for the same image.

The task runners below never invent labels: metrics are only computed for
samples that have ground truth.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ..categories import CATEGORIES, UNKNOWN, category_label, recommendation
from ..pipeline import _category_scores, _crop, _explanation
from ..runtime import model_lock
from ..settings import settings
from ..xai.explainers import run_gradcam
from . import dataset as ds
from .metrics import DetectionAccumulator, binary_report, classification_report, timing_stats, unsupervised_acceptance

log = logging.getLogger("ecosort.eval")

CATEGORY_ORDER = ["ewaste", "glass", "hazardous", "metal", "organic", "paper", "plastic"]
CATEGORY_NAMES = {"ewaste": "E-Waste", "glass": "Glass", "hazardous": "Hazardous", "metal": "Metal",
                  "organic": "Organic", "paper": "Paper & Cardboard", "plastic": "Plastic"}
STAGES = ["preprocessing", "detection", "binary_filter", "classification", "gradcam", "recommendation"]
STAGE_LABELS = {
    "preprocessing": "Image preprocessing (decode, EXIF orientation, RGB)",
    "detection": "YOLOv8 detection",
    "binary_filter": "Binary waste filter (crop + MobileNetV3)",
    "classification": "Seven-class MobileNetV3 classification",
    "gradcam": "Grad-CAM generation",
    "recommendation": "Disposal recommendation + explanation text",
}


class Cancelled(Exception):
    pass


Progress = Callable[[int, int, str], None]


def _sync(device) -> None:
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class Models:
    clf: object
    detector: object | None
    binary: object | None

    @property
    def device(self):
        return getattr(self.clf, "device", None)


def _detect(detector, image, conf: float, iou: float, max_det: int):
    """Mirror of YoloDetector.detect with evaluation-selected thresholds. When the
    thresholds equal the configured ones this is exactly ``detector.detect``."""
    if conf == getattr(detector, "conf", None) and iou == getattr(detector, "iou", None) and max_det == getattr(detector, "max_det", None):
        return detector.detect(image)
    from ..models.detector import Detection

    res = detector.model.predict(image, conf=conf, iou=iou, agnostic_nms=True, max_det=max_det, verbose=False)[0]
    names = res.names
    out = []
    for box in res.boxes:
        x0, y0, x1, y1 = (int(round(v)) for v in box.xyxy[0].tolist())
        out.append(Detection((x0, y0, x1, y1), names[int(box.cls)], float(box.conf)))
    out.sort(key=lambda d: d.confidence, reverse=True)
    return out[:max_det]


def timed_pipeline(path: Path, m: Models, *, conf: float, iou: float, max_det: int, gradcam: bool, use_detector: bool = True) -> dict:
    t: dict[str, float | None] = {k: None for k in STAGES}
    dev = m.device
    t0 = time.perf_counter()
    img = ds.load_image(path, settings.max_pixels)
    t["preprocessing"] = (time.perf_counter() - t0) * 1000

    detector = m.detector if use_detector else None
    regions = []
    if detector is not None:
        a = time.perf_counter()
        dets = _detect(detector, img, conf, iou, max_det)
        _sync(dev)
        t["detection"] = (time.perf_counter() - a) * 1000
        regions = [{"bbox": list(d.bbox), "det_label": d.label, "det_conf": d.confidence} for d in dets]
    else:
        regions = [{"bbox": None, "det_label": None, "det_conf": None}]

    crops = [_crop(img, tuple(r["bbox"])) if r["bbox"] else img for r in regions]
    probs_w: list[float | None] = [None] * len(crops)
    n_candidates = len(crops)
    if m.binary is not None and crops:
        a = time.perf_counter()
        probs_w = m.binary.predict_waste_probability(crops).tolist()
        keep = [i for i, p in enumerate(probs_w) if p >= m.binary.threshold]
        regions = [regions[i] for i in keep]
        crops = [crops[i] for i in keep]
        probs_w = [probs_w[i] for i in keep]
        _sync(dev)
        t["binary_filter"] = (time.perf_counter() - a) * 1000

    objects = []
    if regions:
        a = time.perf_counter()
        probs = m.clf.predict_proba(crops)
        _sync(dev)
        t["classification"] = (time.perf_counter() - a) * 1000
        scored = [(_category_scores(m.clf, p), p) for p in probs]

        if gradcam:
            a = time.perf_counter()
            gc = []
            for i, (crop, (scores, _)) in enumerate(zip(crops, scored)):
                if i >= settings.xai_max_objects:
                    gc.append(None)
                    continue
                view = m.clf.model_view(crop) if hasattr(m.clf, "model_view") else crop.resize((m.clf.input_size, m.clf.input_size))
                gc.append(run_gradcam(m.clf, crop, view, scores[0]["raw_index"]))
            _sync(dev)
            t["gradcam"] = (time.perf_counter() - a) * 1000
        else:
            gc = [None] * len(crops)

        a = time.perf_counter()
        for i, (r, crop, (scores, p)) in enumerate(zip(regions, crops, scored)):
            top = scores[0]
            uncertain = top["category"] == UNKNOWN or top["confidence"] < m.clf.uncertainty_threshold
            pred = {"category": UNKNOWN if uncertain else top["category"], "uncertain": uncertain}
            xai = {"gradcam": gc[i]} if gc[i] else {}
            _explanation(pred, scores, xai, crop.size, m.clf)
            rec = recommendation(pred["category"])
            objects.append(
                {
                    "bbox": r["bbox"],
                    "detector_label": r["det_label"],
                    "detector_confidence": None if r["det_conf"] is None else round(r["det_conf"], 4),
                    "binary_waste_probability": None if probs_w[i] is None else round(float(probs_w[i]), 4),
                    "top_category": top["category"],
                    "confidence": round(top["confidence"], 4),
                    "uncertain": uncertain,
                    "gradcam": bool(gc[i] and gc[i].get("available")),
                    "recommendation_bin": rec.get("bin"),
                }
            )
        t["recommendation"] = (time.perf_counter() - a) * 1000

    total = (time.perf_counter() - t0) * 1000
    return {
        "image_size": [img.width, img.height],
        "n_candidates": n_candidates,
        "n_rejected": n_candidates - len(objects) if (m.binary is not None) else 0,
        "objects": objects,
        "timing_ms": {k: (None if v is None else round(v, 3)) for k, v in t.items()},
        "total_ms": round(total, 3),
    }


def _timing_summary(records: list[dict], includes_note: dict, device) -> dict:
    stages = {}
    for k in STAGES:
        stats = timing_stats(r["timing_ms"].get(k) for r in records)
        if stats:
            stages[k] = {"label": STAGE_LABELS[k], **stats}
    total = timing_stats(r["total_ms"] for r in records)
    fps = None if not total or total["mean"] <= 0 else round(1000.0 / total["mean"], 2)
    return {
        "n_images_timed": len(records),
        "device": str(device),
        "batch_size": 1,
        "stages": stages,
        "end_to_end": total,
        "fps": fps,
        "fps_definition": "1000 / mean end-to-end latency per image (sequential, batch size 1, after warm-up).",
        "includes": includes_note,
    }


def _includes(m: Models, use_detector: bool, gradcam: bool) -> dict:
    return {
        "preprocessing": True,
        "detection": bool(use_detector and m.detector is not None),
        "binary_filter": m.binary is not None,
        "classification": True,
        "gradcam": gradcam,
        "recommendation": True,
        "lime_shap": False,
        "http_upload_and_json": False,
    }


def _warmup(paths: list[Path], m: Models, n: int, **kw) -> None:
    for p in paths[: max(0, n)]:
        try:
            with model_lock:
                timed_pipeline(p, m, **kw)
        except ds.DatasetError:
            continue


def _check(cancel: Callable[[], bool]) -> None:
    if cancel():
        raise Cancelled()


# --------------------------------------------------------------------------- tasks
def run_classification(root: Path, m: Models, *, mode: str, conf: float, iou: float, max_det: int, warmup: int,
                       progress: Progress, cancel: Callable[[], bool]) -> dict:
    """mode "classifier": the seven-class CNN alone on each full image (no YOLO, no
    binary gate) — the usual crop-level test-set evaluation. mode "pipeline": the full
    hybrid pipeline; the image prediction is the most confident retained object."""
    disc = ds.discover_labeled_images(root, "classification")
    if not disc.samples:
        raise ds.DatasetError("no_images", "No images (.jpg, .jpeg, .png, .webp, .bmp) were found in the dataset.")
    pipeline_mode = mode == "pipeline"
    use_det = pipeline_mode
    gradcam = False
    kw = dict(conf=conf, iou=iou, max_det=max_det, gradcam=gradcam, use_detector=use_det)
    clf_only = Models(m.clf, None, None)
    _warmup([s.path for s in disc.samples], m if pipeline_mode else clf_only, warmup, **kw)

    y_true, y_pred, per_image, timings, invalid = [], [], [], [], list(disc.invalid_files)
    n_uncertain = 0
    total_objects = 0
    n = len(disc.samples)
    for i, s in enumerate(disc.samples):
        _check(cancel)
        progress(i, n, s.rel)
        try:
            with model_lock:
                r = timed_pipeline(s.path, m if pipeline_mode else clf_only, **kw)
        except ds.DatasetError as e:
            invalid.append({"file": s.rel, "reason": e.message})
            continue
        timings.append(r)
        objs = r["objects"]
        total_objects += len(objs)
        if objs:
            best = max(objs, key=lambda o: o["confidence"])
            pred_cat = best["top_category"] if best["top_category"] in CATEGORIES else None
            unc = best["uncertain"]
            conf_v = best["confidence"]
        else:
            pred_cat, unc, conf_v = None, None, None
        if unc:
            n_uncertain += 1
        per_image.append({"file": s.rel, "label": s.label, "label_raw": s.label_raw, "prediction": pred_cat,
                          "confidence": conf_v, "uncertain": unc, "objects": len(objs),
                          "rejected": r["n_rejected"], "total_ms": r["total_ms"]})
        if s.label is not None:
            y_true.append(CATEGORY_ORDER.index(s.label))
            y_pred.append(CATEGORY_ORDER.index(pred_cat) if pred_cat else None)
    progress(n, n, "")

    result = {
        "dataset_stats": _ds_stats(disc, len(timings), invalid, "classification"),
        "per_image": per_image[:5000],
        "inference": _timing_summary(timings, _includes(m if pipeline_mode else clf_only, use_det, gradcam), m.device),
        "objects_detected": total_objects if pipeline_mode else None,
        "warnings": list(disc.warnings),
    }
    if y_true:
        rep = classification_report(y_true, y_pred, CATEGORY_ORDER, [CATEGORY_NAMES[c] for c in CATEGORY_ORDER])
        rep["mode"] = mode
        rep["n_below_uncertainty_threshold"] = sum(
            1 for p in per_image if p["label"] is not None and p["uncertain"]
        )
        rep["uncertainty_threshold"] = m.clf.uncertainty_threshold
        rep["prediction_rule"] = (
            "Arg-max category of the seven-class CNN on the full image."
            if not pipeline_mode
            else "Arg-max category of the most confident object retained by YOLO + binary gate; images with no retained object count as errors (no prediction)."
        )
        result["classification"] = rep
        result["supervised"] = True
    else:
        result["supervised"] = False
        result["warnings"].append(
            "No ground-truth labels were found, so accuracy, precision, recall and F1 are NOT reported. "
            "Arrange images in one folder per class or add a labels.csv (filename,label). Inference statistics are shown instead."
        )
    return result


def run_binary(root: Path, m: Models, *, warmup: int, progress: Progress, cancel: Callable[[], bool]) -> dict:
    if m.binary is None:
        raise ds.DatasetError("binary_not_loaded", "The binary waste filter is not loaded, so it cannot be evaluated.")
    disc = ds.discover_labeled_images(root, "binary")
    if not disc.samples:
        raise ds.DatasetError("no_images", "No images were found in the dataset.")
    import time as _t

    probs, labels, per_image, invalid, times = [], [], [], list(disc.invalid_files), []
    n = len(disc.samples)
    for s in disc.samples[:warmup]:
        try:
            with model_lock:
                m.binary.predict_waste_probability([ds.load_image(s.path, settings.max_pixels)])
        except ds.DatasetError:
            pass
    for i, s in enumerate(disc.samples):
        _check(cancel)
        progress(i, n, s.rel)
        try:
            a = _t.perf_counter()
            img = ds.load_image(s.path, settings.max_pixels)
            pre = (_t.perf_counter() - a) * 1000
            with model_lock:
                b = _t.perf_counter()
                p = float(m.binary.predict_waste_probability([img])[0])
                _sync(m.device)
                inf = (_t.perf_counter() - b) * 1000
        except ds.DatasetError as e:
            invalid.append({"file": s.rel, "reason": e.message})
            continue
        times.append({"timing_ms": {"preprocessing": pre, "binary_filter": inf}, "total_ms": pre + inf})
        per_image.append({"file": s.rel, "label": s.label, "label_raw": s.label_raw, "waste_probability": round(p, 4),
                          "accepted": p >= m.binary.threshold, "total_ms": round(pre + inf, 3)})
        probs.append(p)
        labels.append(s.label)
    progress(n, n, "")
    lab_idx = [i for i, l in enumerate(labels) if l is not None]
    inc = {k: False for k in STAGES}
    inc.update(preprocessing=True, binary_filter=True, lime_shap=False, http_upload_and_json=False)
    result = {
        "dataset_stats": _ds_stats(disc, len(per_image), invalid, "binary"),
        "per_image": per_image[:5000],
        "inference": _timing_summary(times, inc, m.device),
        "warnings": list(disc.warnings),
        "binary_input": "Whole image resized to 224×224 (the gate normally sees YOLO crops).",
    }
    if lab_idx:
        result["binary"] = binary_report([labels[i] for i in lab_idx], [probs[i] for i in lab_idx], m.binary.threshold)
        result["supervised"] = True
        if result["binary"]["n_non_waste"] == 0:
            result["warnings"].append("The dataset contains no non-waste images, so precision, specificity and the non-waste rejection rate cannot be measured meaningfully.")
        if result["binary"]["n_waste"] == 0:
            result["warnings"].append("The dataset contains no waste images, so recall and the waste acceptance rate are undefined.")
    else:
        result["supervised"] = False
        result["binary_unsupervised"] = unsupervised_acceptance(probs, m.binary.threshold)
        result["warnings"].append(
            "No waste / non_waste labels were found, so binary accuracy, precision, recall and F1 are NOT reported. "
            "Use folders named waste/ and non_waste/ (or a labels.csv)."
        )
    return result


def run_detection(root: Path, m: Models, *, conf: float, iou: float, max_det: int, warmup: int,
                  progress: Progress, cancel: Callable[[], bool]) -> dict:
    if m.detector is None:
        raise ds.DatasetError("detector_not_loaded", "The YOLO detector is not loaded, so detection cannot be evaluated.")
    y = ds.discover_yolo(root)
    if not y.images:
        raise ds.DatasetError("no_images", "No images were found in the dataset.")
    model_names = [y_ for _, y_ in sorted(detector_names(m.detector).items())]
    warnings: list[str] = []
    names = y.names
    if names is None:
        names = model_names
        if y.has_labels_dir:
            warnings.append("No data.yaml / classes.txt found: label class ids are assumed to use the detector's own class order "
                            f"({', '.join(model_names)}).")
    # map dataset class id → detector class id through EcoSortAI categories
    model_cat = {i: ds.resolve_eval_category(n) for i, n in enumerate(model_names)}
    cat_to_model = {c: i for i, c in model_cat.items() if c in CATEGORIES}
    ds_to_model: dict[int, int | None] = {}
    unmapped = []
    for i, n in enumerate(names):
        c = ds.resolve_eval_category(n)
        ds_to_model[i] = cat_to_model.get(c)
        if ds_to_model[i] is None:
            unmapped.append(n)
    if unmapped:
        warnings.append(f"Dataset classes not matching any detector class: {', '.join(unmapped)}. "
                        "Their boxes count as misses in the class-aware metrics (they still count in the class-agnostic metrics).")

    acc = DetectionAccumulator(model_names + ["(unmatched dataset class)"])
    unk = len(model_names)
    per_image, invalid, label_problems, timings = [], [], [], []
    n_missing_label = 0
    n = len(y.images)

    if not y.has_labels_dir:
        warnings.append("No labels/ folder with YOLO .txt annotations was found, so detection precision, recall and mAP are NOT reported. "
                        "Inference statistics are shown instead.")
    kw = dict(conf=conf, iou=iou, max_det=max_det)
    for p in y.images[:warmup]:
        try:
            with model_lock:
                _detect(m.detector, ds.load_image(p, settings.max_pixels), **kw)
        except ds.DatasetError:
            pass
    for i, p in enumerate(y.images):
        _check(cancel)
        rel = p.relative_to(root).as_posix()
        progress(i, n, rel)
        try:
            a = time.perf_counter()
            img = ds.load_image(p, settings.max_pixels)
            pre = (time.perf_counter() - a) * 1000
            with model_lock:
                b = time.perf_counter()
                dets = _detect(m.detector, img, **kw)
                _sync(m.device)
                dt = (time.perf_counter() - b) * 1000
        except ds.DatasetError as e:
            invalid.append({"file": rel, "reason": e.message})
            continue
        timings.append({"timing_ms": {"preprocessing": pre, "detection": dt}, "total_ms": pre + dt})
        inv = {v: k for k, v in detector_names(m.detector).items()}
        pb = [d.bbox for d in dets]
        pc = [d.confidence for d in dets]
        pk = [inv.get(d.label, -1) for d in dets]
        rec = {"file": rel, "predictions": len(dets), "total_ms": round(pre + dt, 3)}
        if y.has_labels_dir:
            lp = ds.yolo_label_path(p)
            if not lp.exists():
                n_missing_label += 1
            gb, gk, probs = ds.read_yolo_labels(lp, img.width, img.height, len(names))
            if probs:
                label_problems.append({"file": lp.relative_to(root).as_posix() if lp.is_relative_to(root) else lp.name, "problems": probs[:5]})
            gk_m = [ds_to_model.get(k) if ds_to_model.get(k) is not None else unk for k in gk]
            acc.add(pb, pc, pk, gb, gk_m)
            rec["ground_truth"] = len(gb)
        per_image.append(rec)
    progress(n, n, "")
    if n_missing_label:
        warnings.append(f"{n_missing_label} image(s) had no label file and were treated as background images (no objects), as in YOLO.")
    if label_problems:
        warnings.append(f"{len(label_problems)} label file(s) contained invalid lines that were skipped.")

    inc = {k: False for k in STAGES}
    inc.update(preprocessing=True, detection=True, lime_shap=False, http_upload_and_json=False)
    result = {
        "dataset_stats": {
            "n_images_found": n, "n_images_evaluated": len(per_image), "invalid_files": invalid[:200],
            "n_invalid": len(invalid), "label_format": "yolo-txt" if y.has_labels_dir else "none",
            "class_names": names, "class_names_source": y.names_source,
            "label_problems": label_problems[:100],
        },
        "per_image": per_image[:5000],
        "inference": _timing_summary(timings, inc, m.device),
        "objects_detected": int(sum(r["predictions"] for r in per_image)),
        "warnings": warnings,
    }
    if y.has_labels_dir and acc.n_images:
        rep = acc.report()
        rep.update(
            confidence_threshold=conf, nms_iou_threshold=iou, max_det=max_det, agnostic_nms=True,
            detector_classes=model_names, class_mapping={names[k]: (model_names[v] if v is not None else None) for k, v in ds_to_model.items()},
            methodology=("YOLO (Ultralytics) evaluation: predictions matched one-to-one to ground-truth boxes of the same class "
                         "at each IoU threshold 0.50…0.95 (visited in descending confidence), AP = 101-point interpolated area under the "
                         "precision–recall curve, mAP = mean over classes with ground truth, P and R reported at the confidence that maximises mean F1. "
                         "Predictions come from YOLO.predict() on each image (letterboxed, batch 1), so numbers can differ "
                         "slightly from `yolo val`, which uses rectangular batches."),
        )
        result["detection"] = rep
        result["supervised"] = rep["n_ground_truth_boxes"] > 0
        if rep["n_ground_truth_boxes"] == 0:
            result["warnings"].append("The label files contain no boxes, so mAP cannot be computed.")
    else:
        result["supervised"] = False
    return result


def run_inference(root: Path, m: Models, *, conf: float, iou: float, max_det: int, warmup: int, gradcam: bool,
                  progress: Progress, cancel: Callable[[], bool]) -> dict:
    images, ignored = ds.list_images(root)
    if not images:
        raise ds.DatasetError("no_images", "No images were found in the upload.")
    kw = dict(conf=conf, iou=iou, max_det=max_det, gradcam=gradcam, use_detector=True)
    _warmup(images, m, warmup, **kw)
    timings, per_image, invalid = [], [], []
    n = len(images)
    for i, p in enumerate(images):
        _check(cancel)
        rel = p.relative_to(root).as_posix()
        progress(i, n, rel)
        try:
            with model_lock:
                r = timed_pipeline(p, m, **kw)
        except ds.DatasetError as e:
            invalid.append({"file": rel, "reason": e.message})
            continue
        timings.append(r)
        per_image.append({"file": rel, "candidates": r["n_candidates"], "rejected": r["n_rejected"], "objects": len(r["objects"]),
                          "categories": [o["top_category"] for o in r["objects"]], "total_ms": r["total_ms"], "timing_ms": r["timing_ms"]})
    progress(n, n, "")
    cands = sum(r["n_candidates"] for r in timings)
    rej = sum(r["n_rejected"] for r in timings)
    return {
        "dataset_stats": {"n_images_found": n, "n_images_evaluated": len(timings), "invalid_files": invalid[:200], "n_invalid": len(invalid),
                          "label_format": "not used", "ignored_files": ignored},
        "per_image": per_image[:5000],
        "inference": _timing_summary(timings, _includes(m, True, gradcam), m.device),
        "objects_detected": cands if m.detector is not None else None,
        "pipeline_counts": {"candidates": cands, "rejected_by_binary_gate": rej, "classified": cands - rej},
        "supervised": False,
        "warnings": ["Inference benchmark: no ground truth is used, so no accuracy, precision, recall, F1 or mAP is reported."],
    }


def detector_names(detector) -> dict[int, str]:
    names = getattr(getattr(detector, "model", None), "names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return dict(enumerate(map(str, names)))
    return {}


def _ds_stats(disc: ds.Discovery, n_eval: int, invalid: list[dict], kind: str) -> dict:
    dist: dict[str, int] = {}
    for s in disc.samples:
        if s.label is not None:
            key = (CATEGORY_NAMES.get(s.label, s.label) if kind == "classification" else ("waste" if s.label == 1 else "non_waste"))
            dist[key] = dist.get(key, 0) + 1
    return {
        "n_images_found": len(disc.samples),
        "n_images_evaluated": n_eval,
        "n_labeled": len(disc.labeled),
        "n_unlabeled": len(disc.samples) - len(disc.labeled),
        "label_format": disc.label_format,
        "class_distribution": dist,
        "unmapped_labels": disc.unmapped_labels,
        "invalid_files": invalid[:200],
        "n_invalid": len(invalid),
        "ignored_files": disc.ignored_files,
    }
