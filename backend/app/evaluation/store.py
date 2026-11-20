"""Evaluation jobs, run cache, saved history and model identifiers.

* Jobs run one at a time in a background thread so long evaluations do not block
  the API; the browser polls for progress.
* Finished runs are cached in memory (most recent ``RUN_CACHE`` runs). A run is
  written to ``backend/evaluation_runs/<run_id>.json`` only when the user saves it.
  Uploaded datasets are never written there — they live in a temp directory
  that is deleted when the job ends.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__
from ..settings import BACKEND_DIR, settings

log = logging.getLogger("ecosort.eval")

RUNS_DIR = Path(os.environ.get("ECOSORT_EVAL_RUNS_DIR", str(BACKEND_DIR / "evaluation_runs")))
RUN_CACHE = 30
_ID = re.compile(r"^[a-f0-9]{12}$")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- model identifiers
_hash_cache: dict[tuple[str, float, int], str] = {}


def file_identity(path: Path | str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = BACKEND_DIR / p
    if not p.exists():
        return {"file": p.name, "exists": False}
    st = p.stat()
    key = (str(p), st.st_mtime, st.st_size)
    if key not in _hash_cache:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _hash_cache[key] = h.hexdigest()
    return {
        "file": p.name,
        "sha256": _hash_cache[key][:16],
        "size_mb": round(st.st_size / 1048576, 2),
        "modified": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
    }


def model_identifiers(registry) -> dict:
    clf = registry.classifier
    clf_ckpt = None
    if clf is not None and getattr(clf, "cfg", None):
        c = Path(clf.cfg.get("checkpoint", ""))
        clf_ckpt = c if c.is_absolute() else settings.model_config.parent / c
    return {
        "classifier": {
            "name": getattr(clf, "name", None),
            "architecture": (getattr(clf, "cfg", {}) or {}).get("architecture"),
            "classes": getattr(clf, "class_names", None),
            "input_size": getattr(clf, "input_size", None),
            "resize": getattr(clf, "resize_mode", None),
            "uncertainty_threshold": getattr(clf, "uncertainty_threshold", None),
            "checkpoint": file_identity(clf_ckpt),
            "loaded": clf is not None,
        },
        "binary_filter": {
            "architecture": "mobilenet_v3_small" if registry.binary_filter is not None else None,
            "classes": getattr(registry.binary_filter, "class_names", None),
            "threshold": settings.binary_threshold,
            "checkpoint": file_identity(settings.binary_checkpoint),
            "loaded": registry.binary_filter is not None,
        },
        "detector": {
            "name": getattr(registry.detector, "name", None),
            "inference_confidence": settings.detector_conf,
            "inference_iou": settings.detector_iou,
            "agnostic_nms": True,
            "max_objects": settings.max_objects,
            "checkpoint": file_identity(settings.detector_weights) if settings.detector_weights else None,
            "loaded": registry.detector is not None,
        },
        "device": str(getattr(clf, "device", "n/a")),
        "app_version": __version__,
    }


# --------------------------------------------------------------------------- historical metrics
_historical: list[dict] | None = None


def historical_metrics() -> list[dict]:
    """Metrics recorded INSIDE the bundled checkpoints at training time. They were
    not produced by this website; source, split and what is unknown are stated."""
    global _historical
    if _historical is not None:
        return _historical
    out: list[dict] = []
    try:
        import torch

        cfg = json.loads(settings.model_config.read_text(encoding="utf-8")) if settings.model_config.exists() else {}
        clf_path = settings.model_config.parent / cfg.get("checkpoint", "") if cfg.get("checkpoint") else None
        if clf_path and clf_path.exists():
            ck = torch.load(str(clf_path), map_location="cpu", weights_only=False)
            if isinstance(ck, dict) and "best_val_accuracy" in ck:
                out.append({
                    "model": "Seven-class MobileNetV3",
                    "checkpoint": file_identity(clf_path),
                    "source": f"checkpoint key 'best_val_accuracy' (saved at epoch {ck.get('epoch', '?')})",
                    "split": "validation split of the training run (dataset not recorded in the checkpoint)",
                    "sample_count": None,
                    "metrics": {"accuracy": round(float(ck["best_val_accuracy"]), 4)},
                })
        if settings.binary_checkpoint.exists():
            ck = torch.load(str(settings.binary_checkpoint), map_location="cpu", weights_only=False)
            if isinstance(ck, dict) and "best_val_balanced_accuracy" in ck:
                out.append({
                    "model": "Binary MobileNetV3 waste filter",
                    "checkpoint": file_identity(settings.binary_checkpoint),
                    "source": f"checkpoint key 'best_val_balanced_accuracy' (saved at epoch {ck.get('epoch', '?')})",
                    "split": "validation split of the training run (dataset and decision threshold not recorded)",
                    "sample_count": None,
                    "metrics": {"balanced_accuracy": round(float(ck["best_val_balanced_accuracy"]), 4)},
                })
        if settings.detector_weights:
            dp = Path(settings.detector_weights)
            dp = dp if dp.is_absolute() else BACKEND_DIR / dp
            if dp.exists():
                ck = torch.load(str(dp), map_location="cpu", weights_only=False)
                tm = ck.get("train_metrics") if isinstance(ck, dict) else None
                if tm:
                    ta = ck.get("train_args") or {}
                    out.append({
                        "model": "YOLOv8 detector",
                        "checkpoint": file_identity(dp),
                        "source": f"checkpoint 'train_metrics' written by Ultralytics {ck.get('version', '?')} on {ck.get('date', '?')}",
                        "split": f"'{ta.get('split', 'val')}' split of {Path(str(ta.get('data', '?'))).name} "
                                 f"(imgsz {ta.get('imgsz', '?')}, val IoU {ta.get('iou', '?')}); image count not recorded",
                        "sample_count": None,
                        "metrics": {
                            "precision": tm.get("metrics/precision(B)"),
                            "recall": tm.get("metrics/recall(B)"),
                            "map50": tm.get("metrics/mAP50(B)"),
                            "map50_95": tm.get("metrics/mAP50-95(B)"),
                        },
                    })
    except Exception as exc:  # never break the summary
        log.warning("Could not read checkpoint metadata: %s", exc)
    _historical = out
    return out


# --------------------------------------------------------------------------- runs
_runs: "OrderedDict[str, dict]" = OrderedDict()
_runs_lock = threading.Lock()


def put_run(run: dict) -> None:
    with _runs_lock:
        _runs[run["run_id"]] = run
        _runs.move_to_end(run["run_id"])
        while len(_runs) > RUN_CACHE:
            _runs.popitem(last=False)


def get_run(run_id: str) -> dict | None:
    if not _ID.match(run_id or ""):
        return None
    with _runs_lock:
        if run_id in _runs:
            return _runs[run_id]
    p = RUNS_DIR / f"{run_id}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def save_run(run_id: str, name: str | None) -> dict | None:
    run = get_run(run_id)
    if run is None:
        return None
    run["saved"] = True
    run["saved_at"] = now_iso()
    if name:
        run["dataset"]["display_name"] = name.strip()[:120]
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RUNS_DIR / f".{run_id}.tmp"
    tmp.write_text(json.dumps(run), encoding="utf-8")
    tmp.replace(RUNS_DIR / f"{run_id}.json")
    put_run(run)
    return run


def delete_run(run_id: str) -> bool:
    if not _ID.match(run_id or ""):
        return False
    p = RUNS_DIR / f"{run_id}.json"
    existed = p.exists()
    p.unlink(missing_ok=True)
    with _runs_lock:
        if run_id in _runs:
            _runs[run_id]["saved"] = False
    return existed


def run_summary(run: dict) -> dict:
    c = run.get("classification") or {}
    b = run.get("binary") or {}
    d = (run.get("detection") or {}).get("class_aware") or {}
    inf = run.get("inference") or {}
    e2e = inf.get("end_to_end") or {}
    return {
        "run_id": run["run_id"],
        "task": run["task"],
        "task_label": run["task_label"],
        "created_at": run["created_at"],
        "saved": run.get("saved", False),
        "dataset_name": run["dataset"].get("display_name") or run["dataset"]["name"],
        "n_images": run["dataset"].get("n_images_evaluated"),
        "supervised": run.get("supervised", False),
        "models": {k: ((v or {}).get("checkpoint") or {}).get("sha256") for k, v in run["config"]["models"].items() if isinstance(v, dict)},
        "accuracy": c.get("accuracy"),
        "macro_f1": (c.get("macro_avg") or {}).get("f1"),
        "weighted_f1": (c.get("weighted_avg") or {}).get("f1"),
        "binary_accuracy": b.get("accuracy"),
        "binary_f1": b.get("f1"),
        "map50": d.get("map50"),
        "map50_95": d.get("map50_95"),
        "det_precision": d.get("precision"),
        "det_recall": d.get("recall"),
        "avg_inference_ms": e2e.get("mean"),
        "fps": inf.get("fps"),
    }


def list_history() -> list[dict]:
    out = []
    if RUNS_DIR.exists():
        for p in RUNS_DIR.glob("*.json"):
            try:
                out.append(run_summary(json.loads(p.read_text(encoding="utf-8"))))
            except Exception:
                log.warning("Skipping unreadable history file %s", p.name)
    with _runs_lock:
        saved_ids = {r["run_id"] for r in out}
        for r in _runs.values():
            if r["run_id"] not in saved_ids:
                out.append(run_summary(r))
    out.sort(key=lambda r: r["created_at"], reverse=True)
    return out


# --------------------------------------------------------------------------- jobs
class Job:
    def __init__(self, task: str):
        self.id = uuid.uuid4().hex[:12]
        self.task = task
        self.status = "queued"
        self.done = 0
        self.total = 0
        self.current = ""
        self.message = ""
        self.run_id: str | None = None
        self.error: dict | None = None
        self.cancel_requested = False
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None

    def as_dict(self) -> dict:
        return {
            "job_id": self.id, "task": self.task, "status": self.status, "done": self.done, "total": self.total,
            "current": self.current, "message": self.message, "run_id": self.run_id, "error": self.error,
            "elapsed_s": round((self.finished or time.time()) - (self.started or self.created), 1),
        }


_jobs: dict[str, Job] = {}
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ecosort-eval")


def submit(task: str, fn) -> Job:
    job = Job(task)
    _jobs[job.id] = job
    # prune old finished jobs
    for jid in [j.id for j in _jobs.values() if j.finished and time.time() - j.finished > 3600]:
        _jobs.pop(jid, None)

    def wrapper():
        job.status = "running"
        job.started = time.time()
        try:
            run = fn(job)
            put_run(run)
            job.run_id = run["run_id"]
            job.status = "done"
        except Exception as exc:  # noqa: BLE001
            from .dataset import DatasetError
            from .runner import Cancelled

            if isinstance(exc, Cancelled):
                job.status = "cancelled"
            elif isinstance(exc, DatasetError):
                job.status = "error"
                job.error = {"code": exc.code, "message": exc.message}
            else:
                log.error("Evaluation job failed:\n%s", traceback.format_exc())
                job.status = "error"
                job.error = {"code": "evaluation_failed", "message": f"Evaluation failed: {exc}"}
        finally:
            job.finished = time.time()

    _executor.submit(wrapper)
    return job


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def busy() -> bool:
    return any(j.status in ("queued", "running") for j in _jobs.values())
