"""Metric computations for the Performance Evaluation module.

Everything here is computed from (prediction, ground truth) pairs. Nothing is
estimated, interpolated from defaults or taken from model confidence alone.

* Classification: per-class precision / recall / F1 / support, accuracy,
  macro and weighted averages, confusion matrix. Matches scikit-learn's
  ``classification_report`` (cross-checked in tests).
* Binary waste verification: TP/FP/TN/FN with "waste" as the positive class,
  plus the acceptance / rejection rates, which are reported separately from
  accuracy.
* Detection: uses the installed Ultralytics matching
  (``BaseValidator.match_predictions``) and AP (``utils.metrics.ap_per_class``:
  101-point interpolated AP over IoU 0.50:0.05:0.95; P and R at the confidence
  that maximises mean F1), with a NumPy port as fallback (cross-checked in tests).
* Timing: summary statistics over measured per-image stage durations.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

EPS = 1e-16
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def _safe_div(a: float, b: float) -> float | None:
    return None if b == 0 else float(a) / float(b)


def _r(x: float | None, nd: int = 4) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), nd)


# --------------------------------------------------------------------------- classification
def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int | None],
    labels: Sequence[str],
    label_names: Sequence[str],
) -> dict:
    """``y_pred`` may contain ``None`` (the system made no prediction, e.g. the
    hybrid pipeline rejected every crop). Such samples count as errors: they
    lower accuracy and the recall of their true class, but are not attributed to
    any predicted class. They are reported in ``abstained_per_class``.

    Macro / weighted averages are taken over classes that occur in the ground
    truth (support > 0). A class that is never predicted has precision 0 and
    ``precision_defined: false`` (scikit-learn's ``zero_division=0``)."""
    n = len(labels)
    y_true = list(y_true)
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    cm = confusion_matrix([t for t, _ in pairs], [p for _, p in pairs], n)
    abstained = np.zeros(n, dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if p is None:
            abstained[t] += 1

    support = cm.sum(axis=1) + abstained
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)
    total = int(support.sum())

    rows = []
    for i in range(n):
        prec = _safe_div(tp[i], predicted[i])
        rec = _safe_div(tp[i], support[i])
        p0 = 0.0 if prec is None else prec
        r0 = 0.0 if rec is None else rec
        f1 = 0.0 if (p0 + r0) == 0 else 2 * p0 * r0 / (p0 + r0)
        rows.append(
            {
                "category": labels[i],
                "label": label_names[i],
                "precision": _r(p0) if support[i] > 0 or predicted[i] > 0 else None,
                "precision_defined": prec is not None,
                "recall": _r(rec),
                "f1": _r(f1) if support[i] > 0 or predicted[i] > 0 else None,
                "support": int(support[i]),
                "predicted": int(predicted[i]),
                "true_positive": int(tp[i]),
                "abstained": int(abstained[i]),
            }
        )

    present = [i for i in range(n) if support[i] > 0]

    def avg(key: str, weighted: bool) -> float | None:
        if not present:
            return None
        vals = np.array([rows[i][key] or 0.0 for i in present], dtype=float)
        if weighted:
            w = np.array([support[i] for i in present], dtype=float)
            return _r(float((vals * w).sum() / w.sum()))
        return _r(float(vals.mean()))

    correct = int(tp.sum())
    n_abst = int(abstained.sum())
    return {
        "labels": list(labels),
        "label_names": list(label_names),
        "n_samples": total,
        "n_correct": correct,
        "accuracy": _r(_safe_div(correct, total)),
        "n_abstained": n_abst,
        "coverage": _r(_safe_div(total - n_abst, total)),
        "accuracy_on_predicted": _r(_safe_div(correct, total - n_abst)),
        "macro_avg": {k: avg(k, False) for k in ("precision", "recall", "f1")},
        "weighted_avg": {k: avg(k, True) for k in ("precision", "recall", "f1")},
        "averaged_over_classes": [labels[i] for i in present],
        "per_class": rows,
        "confusion_matrix": cm.tolist(),
        "abstained_per_class": abstained.tolist(),
    }


# --------------------------------------------------------------------------- binary
def binary_report(y_true: Sequence[int], waste_prob: Sequence[float], threshold: float) -> dict:
    """``y_true``: 1 = waste, 0 = non-waste. Prediction: waste when P(waste) >= threshold
    (the exact gate used by the inference pipeline)."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(waste_prob, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = len(y)
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    f1 = None if prec is None or rec is None else (0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    bal = None if rec is None or spec is None else (rec + spec) / 2
    auc = _roc_auc(y, p)
    return {
        "threshold": threshold,
        "positive_class": "waste",
        "n_samples": n,
        "n_waste": int((y == 1).sum()),
        "n_non_waste": int((y == 0).sum()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "confusion_labels": ["non_waste", "waste"],
        "accuracy": _r(_safe_div(tp + tn, n)),
        "precision": _r(prec),
        "recall": _r(rec),
        "f1": _r(f1),
        "specificity": _r(spec),
        "balanced_accuracy": _r(bal),
        "roc_auc": _r(auc),
        # Rates conditioned on the true class — NOT the same as accuracy.
        "waste_acceptance_rate": _r(rec),          # P(accepted | truly waste) = TPR
        "non_waste_rejection_rate": _r(spec),      # P(rejected | truly non-waste) = TNR
        # Share of all evaluated images the gate let through, regardless of label.
        "overall_acceptance_rate": _r(_safe_div(tp + fp, n)),
        "probability_histogram": _histogram(p, y),
    }


def _roc_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # Mann–Whitney U with average ranks for ties.
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    u = r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def _histogram(p: np.ndarray, y: np.ndarray, bins: int = 20) -> dict:
    edges = np.linspace(0, 1, bins + 1)
    return {
        "edges": [round(float(e), 3) for e in edges],
        "waste": np.histogram(p[y == 1], edges)[0].tolist(),
        "non_waste": np.histogram(p[y == 0], edges)[0].tolist(),
    }


def unsupervised_acceptance(waste_prob: Sequence[float], threshold: float) -> dict:
    p = np.asarray(waste_prob, dtype=float)
    n = len(p)
    return {
        "threshold": threshold,
        "n_samples": n,
        "accepted": int((p >= threshold).sum()),
        "overall_acceptance_rate": _r(_safe_div(int((p >= threshold).sum()), n)),
        "note": "Share of images the binary gate accepted. Without ground truth this is NOT accuracy.",
    }


# --------------------------------------------------------------------------- detection
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (N,4) and (M,4) xyxy boxes → (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = a[:, None, :]
    b = b[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-7)


def match_predictions(pred_cls: np.ndarray, gt_cls: np.ndarray, iou: np.ndarray) -> np.ndarray:
    """Port of Ultralytics 8.4 ``BaseValidator.match_predictions`` (default, non-SciPy
    branch): predictions are visited in descending-confidence order and each claims
    the best still-unclaimed ground-truth box of the same class, per IoU threshold.
    ``iou`` is (n_gt, n_pred) with predictions already sorted by confidence.
    Returns bool (n_pred, 10)."""
    correct = np.zeros((len(pred_cls), len(IOU_THRESHOLDS)), dtype=bool)
    if len(pred_cls) == 0 or len(gt_cls) == 0:
        return correct
    iou = iou * (gt_cls[:, None] == pred_cls[None, :])
    thr = IOU_THRESHOLDS
    matched = np.zeros((iou.shape[0], len(thr)), dtype=bool)
    for j in np.flatnonzero((iou >= thr.min()).any(0)):
        available = np.where(matched, 0, iou[:, j, None])
        k = available.argmax(0)
        correct[j] = available[k, range(len(thr))] >= thr
        matched[k, range(len(thr))] |= correct[j]
    return correct


def _smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Port of Ultralytics 8.4 ``compute_ap`` (101-point COCO interpolation; precision
    drops to 0 beyond the highest recall reached)."""
    last = recall[-1] if len(recall) else 1.0
    mrec = np.concatenate(([0.0], recall, [last], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2 renamed trapz
    return float(trapz(np.interp(x, mrec, mpre), x))


def ap_per_class(tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, target_cls: np.ndarray) -> dict:
    """Port of ``ultralytics.utils.metrics.ap_per_class`` (no plotting)."""
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    px = np.linspace(0, 1, 1000)
    ap = np.zeros((nc, tp.shape[1]))
    p_curve, r_curve = np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        sel = pred_cls == c
        n_l = nt[ci]
        n_p = sel.sum()
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[sel]).cumsum(0)
        tpc = tp[sel].cumsum(0)
        recall = tpc / (n_l + EPS)
        r_curve[ci] = np.interp(-px, -conf[sel], recall[:, 0], left=0)
        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-px, -conf[sel], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + EPS)
    k = int(_smooth(f1_curve.mean(0), 0.1).argmax()) if nc else 0
    p, r = (p_curve[:, k], r_curve[:, k]) if nc else (np.zeros(0), np.zeros(0))
    return {"classes": unique_classes, "n_targets": nt, "p": p, "r": r, "ap": ap, "best_conf": float(px[k]) if nc else None}


# --- Prefer the installed Ultralytics implementation, so results follow exactly the
# --- methodology of the YOLO version in use; the ports above are the fallback.
def _ultralytics_version() -> str | None:
    try:
        import ultralytics

        return ultralytics.__version__
    except Exception:
        return None


def _match(pred_cls: np.ndarray, gt_cls: np.ndarray, iou: np.ndarray) -> tuple[np.ndarray, str]:
    if len(pred_cls) and len(gt_cls):
        try:
            from types import SimpleNamespace

            import torch
            from ultralytics.engine.validator import BaseValidator

            ns = SimpleNamespace(iouv=torch.tensor(IOU_THRESHOLDS, dtype=torch.float32))
            out = BaseValidator.match_predictions(ns, torch.from_numpy(pred_cls), torch.from_numpy(gt_cls), torch.from_numpy(iou).float())
            return out.cpu().numpy().astype(bool), "ultralytics"
        except Exception:
            pass
    return match_predictions(pred_cls, gt_cls, iou), "port"


def _ap(tp, conf, pred_cls, target_cls) -> tuple[dict, str]:
    try:
        from ultralytics.utils.metrics import ap_per_class as ul_ap

        res = ul_ap(tp.astype(bool), conf, pred_cls, target_cls)
        p, r, ap, classes = res[2], res[3], res[5], res[6]
        _, nt = np.unique(target_cls, return_counts=True)
        x = res[10] if len(res) > 10 else np.linspace(0, 1, 1000)
        f1c = res[9] if len(res) > 9 else None
        best = float(x[int(_smooth(f1c.mean(0), 0.1).argmax())]) if f1c is not None and len(classes) else None
        return {"classes": classes, "n_targets": nt, "p": p, "r": r, "ap": ap, "best_conf": best}, "ultralytics"
    except Exception:
        return ap_per_class(tp, conf, pred_cls, target_cls), "port"


class DetectionAccumulator:
    """Collects per-image predictions/ground truth, then reports YOLO-style metrics."""

    def __init__(self, class_names: Sequence[str]):
        self.class_names = list(class_names)
        self.tp: list[np.ndarray] = []
        self.conf: list[np.ndarray] = []
        self.pred_cls: list[np.ndarray] = []
        self.target_cls: list[np.ndarray] = []
        self.tp_agn: list[np.ndarray] = []
        self.n_images = 0
        self.n_pred = 0
        self.n_gt = 0
        self.impl_match = "port"
        self.impl_ap = "port"

    def add(self, pred_boxes, pred_conf, pred_cls, gt_boxes, gt_cls) -> None:
        pb = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
        pc = np.asarray(pred_conf, dtype=float).reshape(-1)
        pk = np.asarray(pred_cls, dtype=int).reshape(-1)
        gb = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
        gk = np.asarray(gt_cls, dtype=int).reshape(-1)
        self.n_images += 1
        self.n_pred += len(pb)
        self.n_gt += len(gb)
        order = np.argsort(-pc, kind="stable")  # matching visits predictions by confidence
        pb, pc, pk = pb[order], pc[order], pk[order]
        iou = box_iou(gb, pb)
        tp, self.impl_match = _match(pk, gk, iou)
        self.tp.append(tp)
        self.tp_agn.append(_match(np.zeros_like(pk), np.zeros_like(gk), iou)[0])
        self.conf.append(pc)
        self.pred_cls.append(pk)
        self.target_cls.append(gk)

    def _summ(self, tp, conf, pred_cls, target_cls, names) -> dict:
        if len(target_cls) == 0:
            return {"available": False, "note": "No ground-truth boxes in the evaluated images."}
        res, self.impl_ap = _ap(tp, conf, pred_cls, target_cls)
        per_class = []
        for ci, c in enumerate(res["classes"]):
            per_class.append(
                {
                    "class_id": int(c),
                    "name": names[int(c)] if int(c) < len(names) else str(c),
                    "instances": int(res["n_targets"][ci]),
                    "precision": _r(res["p"][ci]),
                    "recall": _r(res["r"][ci]),
                    "map50": _r(res["ap"][ci, 0]),
                    "map50_95": _r(res["ap"][ci].mean()),
                }
            )
        return {
            "available": True,
            "precision": _r(float(res["p"].mean())),
            "recall": _r(float(res["r"].mean())),
            "map50": _r(float(res["ap"][:, 0].mean())),
            "map50_95": _r(float(res["ap"].mean())),
            "operating_confidence": _r(res["best_conf"], 3),
            "per_class": per_class,
        }

    def report(self) -> dict:
        cat = lambda xs, d: np.concatenate(xs, 0) if xs else np.zeros((0,) + d)  # noqa: E731
        tp = cat(self.tp, (10,)).astype(float)
        conf = cat(self.conf, ())
        pc = cat(self.pred_cls, ())
        tc = cat(self.target_cls, ())
        out = {
            "n_images": self.n_images,
            "n_ground_truth_boxes": self.n_gt,
            "n_predicted_boxes": self.n_pred,
            "iou_thresholds": "0.50:0.05:0.95",
            "class_aware": self._summ(tp, conf, pc, tc, self.class_names),
            "class_agnostic": self._summ(cat(self.tp_agn, (10,)).astype(float), conf, np.zeros_like(pc), np.zeros_like(tc), ["object"]),
        }
        v = _ultralytics_version()
        out["implementation"] = {
            "matching": f"ultralytics {v} BaseValidator.match_predictions" if self.impl_match == "ultralytics" else "built-in port (Ultralytics 8.4 logic)",
            "ap": f"ultralytics {v} utils.metrics.ap_per_class" if self.impl_ap == "ultralytics" else "built-in port (Ultralytics 8.4 logic)",
        }
        return out


# --------------------------------------------------------------------------- timing
def timing_stats(values_ms: Iterable[float]) -> dict | None:
    v = np.asarray([x for x in values_ms if x is not None], dtype=float)
    if len(v) == 0:
        return None
    return {
        "n": int(len(v)),
        "mean": round(float(v.mean()), 3),
        "median": round(float(np.median(v)), 3),
        "min": round(float(v.min()), 3),
        "max": round(float(v.max()), 3),
        "p95": round(float(np.percentile(v, 95)), 3),
        "std": round(float(v.std(ddof=0)), 3),
    }
