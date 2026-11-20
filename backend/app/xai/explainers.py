"""Grad-CAM, LIME and SHAP for a single object crop.

All three explain the *same* classifier on the *same* model-view crop and the
same target class. Results include overlay images (data URLs) plus a few
numbers used to write the plain-language explanation.
"""
from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from ..models.base import Classifier
from .render import heat_overlay, region_name, segment_overlay, to_data_url

log = logging.getLogger("ecosort.xai")


def _segments(view: np.ndarray, n: int = 40) -> np.ndarray:
    from skimage.segmentation import slic

    return slic(view, n_segments=n, compactness=12, start_label=0)


def _seg_center(segments: np.ndarray, seg: int) -> tuple[float, float]:
    ys, xs = np.nonzero(segments == seg)
    h, w = segments.shape
    return float(xs.mean() / w), float(ys.mean() / h)


def run_gradcam(clf: Classifier, image: Image.Image, view: Image.Image, class_index: int) -> dict:
    if not clf.supports_gradcam:
        return {"available": False, "note": "Grad-CAM needs a CNN with accessible convolutional layers; not supported by this model."}
    try:
        cam = clf.gradcam(image, class_index)
    except Exception as exc:
        log.exception("Grad-CAM failed")
        return {"available": False, "note": f"Grad-CAM failed: {exc}"}
    h, w = cam.shape
    total = cam.sum() or 1.0
    ys, xs = np.mgrid[0:h, 0:w]
    cx, cy = float((cam * xs).sum() / total / w), float((cam * ys).sum() / total / h)
    hot = cam >= np.quantile(cam, 0.8)
    share = float(cam[hot].sum() / total)
    return {
        "available": True,
        "image": to_data_url(heat_overlay(view, cam)),
        "note": "Warmer colours mark regions that most increased the score for the predicted class.",
        "target_layer": getattr(clf, "target_layer_name", None),
        "focus_region": region_name(cx, cy),
        "focus_share": round(share, 3),
    }


def run_lime(clf: Classifier, view: Image.Image, class_index: int, num_samples: int) -> dict:
    try:
        from lime import lime_image
    except ImportError:
        return {"available": False, "note": "LIME is not installed on the server (pip install lime)."}
    try:
        arr = np.asarray(view.convert("RGB"))
        explainer = lime_image.LimeImageExplainer(random_state=0)
        exp = explainer.explain_instance(
            arr,
            clf.predict_proba_arrays,
            labels=(class_index,),
            top_labels=None,
            hide_color=0,
            num_samples=num_samples,
            batch_size=32,
            segmentation_fn=lambda x: _segments(x, 40),
            random_seed=0,
        )
        weights = dict(exp.local_exp[class_index])
        ranked = sorted(weights.items(), key=lambda kv: -abs(kv[1]))
        keep = dict(ranked[:8])
        pos = [s for s, w in ranked[:8] if w > 0]
        neg = [s for s, w in ranked[:8] if w < 0]
        return {
            "available": True,
            "image": to_data_url(segment_overlay(view, exp.segments, keep)),
            "note": f"Green superpixels support the prediction, red ones oppose it ({num_samples} perturbed samples).",
            "num_samples": num_samples,
            "supporting_regions": sorted({region_name(*_seg_center(exp.segments, s)) for s in pos[:3]}),
            "opposing_regions": sorted({region_name(*_seg_center(exp.segments, s)) for s in neg[:3]}),
            "n_supporting": len(pos),
            "n_opposing": len(neg),
            "local_fit_r2": round(float(exp.score), 3) if exp.score is not None else None,
        }
    except Exception as exc:
        log.exception("LIME failed")
        return {"available": False, "note": f"LIME failed: {exc}"}


def run_shap(clf: Classifier, view: Image.Image, class_index: int, nsamples: int) -> dict:
    try:
        import shap
    except ImportError:
        return {"available": False, "note": "SHAP is not installed on the server (pip install shap)."}
    try:
        arr = np.asarray(view.convert("RGB")).astype(np.float32)
        segments = _segments(arr.astype(np.uint8), 32)
        n = int(segments.max()) + 1
        # Removed superpixels are replaced by the image's mean colour.
        fill = arr.reshape(-1, 3).mean(0)

        def f(z: np.ndarray) -> np.ndarray:
            batch = np.empty((z.shape[0],) + arr.shape, dtype=np.float32)
            for i, row in enumerate(z):
                img = arr.copy()
                off = np.isin(segments, np.nonzero(row == 0)[0])
                img[off] = fill
                batch[i] = img
            return clf.predict_proba_arrays(batch)[:, class_index]

        explainer = shap.KernelExplainer(f, np.zeros((1, n)))
        values = explainer.shap_values(np.ones((1, n)), nsamples=nsamples, silent=True)
        values = np.asarray(values).reshape(-1)[:n]
        weights = {i: float(v) for i, v in enumerate(values)}
        ranked = sorted(weights.items(), key=lambda kv: -abs(kv[1]))
        contrib = [
            {"region": f"{region_name(*_seg_center(segments, s))} #{s}", "value": round(v, 4)} for s, v in ranked[:6]
        ]
        return {
            "available": True,
            "image": to_data_url(segment_overlay(view, segments, dict(ranked[:10]))),
            "note": f"Kernel SHAP over {n} superpixels ({nsamples} samples). Green adds to the class probability, red subtracts.",
            "contributions": contrib,
            "base_value": round(float(np.asarray(explainer.expected_value).reshape(-1)[0]), 4),
            "sum_of_contributions": round(float(values.sum()), 4),
        }
    except Exception as exc:
        log.exception("SHAP failed")
        return {"available": False, "note": f"SHAP failed: {exc}"}
