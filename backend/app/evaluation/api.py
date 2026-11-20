"""/api/evaluation/* — the Performance Evaluation module.

  GET    /api/evaluation/summary                 models, checkpoint ids, thresholds, capabilities, historical metrics
  POST   /api/evaluation/classification          start a classification evaluation job (mode=classifier|pipeline)
  POST   /api/evaluation/binary                  start a binary waste-verification job
  POST   /api/evaluation/detection               start a YOLO detection (mAP) job
  POST   /api/evaluation/inference               start an inference-speed benchmark job
  GET    /api/evaluation/jobs/{job_id}           job progress / status
  POST   /api/evaluation/jobs/{job_id}/cancel
  GET    /api/evaluation/runs/{run_id}           full result of a run
  GET    /api/evaluation/history                 saved runs (+ unsaved runs from this session)
  POST   /api/evaluation/history                 {run_id, name?} save a run
  DELETE /api/evaluation/history/{run_id}
  GET    /api/evaluation/compare?ids=a,b         side-by-side summaries
  GET    /api/evaluation/runs/{run_id}/export?format=pdf|csv&table=all
  POST   /api/evaluation/export                  {run_id, format, table}
"""
from __future__ import annotations

import json
import os
import uuid

from fastapi import APIRouter, Body, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from ..models.registry import registry
from ..settings import settings
from . import dataset as ds
from . import report, runner, store

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

MAX_DATASET_MB = float(os.environ.get("ECOSORT_EVAL_MAX_DATASET_MB", "2048"))
MAX_FILES = int(os.environ.get("ECOSORT_EVAL_MAX_FILES", "20000"))

TASK_LABELS = {
    "classification": "Seven-class classification",
    "pipeline": "End-to-end hybrid pipeline classification",
    "binary": "Binary waste verification",
    "detection": "YOLOv8 object detection",
    "inference": "Inference speed benchmark",
}


def _err(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _models() -> runner.Models:
    if registry.classifier is None:
        raise _err(503, "model_not_loaded", registry.model_error or "No model is loaded.")
    return runner.Models(registry.classifier, registry.detector, registry.binary_filter)


def _defaults() -> dict:
    return {
        "classification": {"mode": "classifier", "conf": settings.detector_conf, "iou": settings.detector_iou, "max_det": settings.max_objects, "warmup": 1},
        "binary": {"warmup": 1, "threshold": settings.binary_threshold},
        "detection": {"conf": 0.001, "iou": settings.detector_iou, "max_det": 300, "warmup": 1},
        "inference": {"conf": settings.detector_conf, "iou": settings.detector_iou, "max_det": settings.max_objects, "warmup": 2, "gradcam": True},
    }


@router.get("/summary")
def summary() -> dict:
    clf, det, binf = registry.classifier, registry.detector, registry.binary_filter
    hist = store.list_history()
    return {
        "models": store.model_identifiers(registry),
        "capabilities": {
            "classification": clf is not None,
            "pipeline": clf is not None and det is not None,
            "binary": binf is not None,
            "detection": det is not None,
            "inference": clf is not None,
        },
        "errors": {"model": registry.model_error, "detector": registry.detector_error, "binary": registry.binary_error},
        "defaults": _defaults(),
        "historical": store.historical_metrics(),
        "history_count": sum(1 for h in hist if h["saved"]),
        "latest": hist[0] if hist else None,
        "busy": store.busy(),
        "limits": {"max_dataset_mb": MAX_DATASET_MB, "max_files": MAX_FILES},
        "dataset_formats": {
            "classification": "Folder per class (<root>/<class>/<image>) or labels.csv (filename,label). Class names: ewaste/e-waste, glass, hazardous, metal, organic, paper/paper_cardboard/cardboard, plastic.",
            "binary": "Folders waste/ and non_waste/ (a folder named after a waste category also counts as waste), or labels.csv.",
            "detection": "Ultralytics YOLO: images/… and labels/… (.txt, 'class cx cy w h' normalised) plus data.yaml or classes.txt.",
            "inference": "Any images; labels are ignored.",
        },
    }


# --------------------------------------------------------------------------- job start
async def _stage(files: list[UploadFile], paths: str | None, dataset_name: str | None) -> ds.StagedDataset:
    if not files:
        raise _err(400, "no_files", "Upload a ZIP file or the images of a folder.")
    rels = None
    if paths:
        try:
            rels = json.loads(paths)
            if not isinstance(rels, list) or len(rels) != len(files):
                rels = None
        except ValueError:
            rels = None
    pairs = [((rels[i] if rels else None) or f.filename or f"file{i}", f.file) for i, f in enumerate(files)]
    try:
        return await run_in_threadpool(ds.stage_upload, pairs, (dataset_name or "").strip() or None, int(MAX_DATASET_MB * 1048576), MAX_FILES)
    except ds.DatasetError as e:
        raise _err(400, e.code, e.message)


def _check_range(name: str, v: float, lo: float, hi: float) -> None:
    if not (lo <= v <= hi):
        raise _err(422, "invalid_parameter", f"{name} must be between {lo} and {hi}.")


def _start(task: str, staged: ds.StagedDataset, params: dict, fn) -> dict:
    models = store.model_identifiers(registry)

    def job_fn(job: store.Job) -> dict:
        def progress(done, total, current):
            job.done, job.total, job.current = done, total, current

        try:
            res = fn(staged.root, progress, lambda: job.cancel_requested)
        finally:
            staged.cleanup()
        dstats = res.pop("dataset_stats")
        run = {
            "run_id": uuid.uuid4().hex[:12],
            "task": task,
            "task_label": TASK_LABELS[task],
            "created_at": store.now_iso(),
            "kind": "measured",
            "dataset": {"name": staged.name, "source": staged.source, **dstats},
            "config": {"models": models, "params": params},
            "saved": False,
            "notes": [
                "Measured in this run from model predictions on the uploaded dataset.",
                "The uploaded dataset was processed in a temporary folder and deleted after the run.",
            ],
            **res,
        }
        if res.get("supervised") is False:
            run["notes"].append("Supervised metrics are not shown because ground-truth labels/annotations were not available.")
        return run

    job = store.submit(task, job_fn)
    return {"job_id": job.id, "status": job.status}


def _busy_guard(staged: ds.StagedDataset) -> None:
    if store.busy():
        staged.cleanup()
        raise _err(409, "busy", "Another evaluation is running. Wait for it to finish or cancel it.")


@router.post("/classification")
async def start_classification(
    files: list[UploadFile] = File(...), paths: str | None = Form(None), dataset_name: str | None = Form(None),
    mode: str = Form("classifier"), conf: float | None = Form(None), iou: float | None = Form(None), max_det: int | None = Form(None), warmup: int = Form(1),
):
    m = _models()
    d = _defaults()["classification"]
    conf = d["conf"] if conf is None else conf
    iou = d["iou"] if iou is None else iou
    max_det = d["max_det"] if max_det is None else max_det
    if mode not in ("classifier", "pipeline"):
        raise _err(422, "invalid_parameter", "mode must be 'classifier' or 'pipeline'.")
    if mode == "pipeline" and m.detector is None:
        raise _err(409, "detector_not_loaded", "The YOLO detector is not loaded, so the end-to-end pipeline cannot be evaluated.")
    _check_range("conf", conf, 0.001, 0.99); _check_range("iou", iou, 0.05, 0.95); _check_range("max_det", max_det, 1, 300); _check_range("warmup", warmup, 0, 10)
    staged = await _stage(files, paths, dataset_name)
    _busy_guard(staged)
    task = "pipeline" if mode == "pipeline" else "classification"
    params = {"mode": mode, "warmup": warmup}
    if mode == "pipeline":
        params.update(conf=conf, iou=iou, max_det=max_det, binary_threshold=settings.binary_threshold)
    fn = lambda root, p, c: runner.run_classification(root, m, mode=mode, conf=conf, iou=iou, max_det=max_det, warmup=warmup, progress=p, cancel=c)  # noqa: E731
    return _start(task, staged, params, fn)


@router.post("/binary")
async def start_binary(files: list[UploadFile] = File(...), paths: str | None = Form(None), dataset_name: str | None = Form(None), warmup: int = Form(1)):
    m = _models()
    if m.binary is None:
        raise _err(409, "binary_not_loaded", registry.binary_error or "The binary waste filter is not loaded.")
    _check_range("warmup", warmup, 0, 10)
    staged = await _stage(files, paths, dataset_name)
    _busy_guard(staged)
    fn = lambda root, p, c: runner.run_binary(root, m, warmup=warmup, progress=p, cancel=c)  # noqa: E731
    return _start("binary", staged, {"threshold": m.binary.threshold, "warmup": warmup}, fn)


@router.post("/detection")
async def start_detection(
    files: list[UploadFile] = File(...), paths: str | None = Form(None), dataset_name: str | None = Form(None),
    conf: float | None = Form(None), iou: float | None = Form(None), max_det: int | None = Form(None), warmup: int = Form(1),
):
    m = _models()
    if m.detector is None:
        raise _err(409, "detector_not_loaded", registry.detector_error or "The YOLO detector is not loaded.")
    d = _defaults()["detection"]
    conf = d["conf"] if conf is None else conf
    iou = d["iou"] if iou is None else iou
    max_det = d["max_det"] if max_det is None else max_det
    _check_range("conf", conf, 0.001, 0.99); _check_range("iou", iou, 0.05, 0.95); _check_range("max_det", max_det, 1, 300); _check_range("warmup", warmup, 0, 10)
    staged = await _stage(files, paths, dataset_name)
    _busy_guard(staged)
    fn = lambda root, p, c: runner.run_detection(root, m, conf=conf, iou=iou, max_det=max_det, warmup=warmup, progress=p, cancel=c)  # noqa: E731
    return _start("detection", staged, {"conf": conf, "nms_iou": iou, "max_det": max_det, "agnostic_nms": True, "warmup": warmup}, fn)


@router.post("/inference")
async def start_inference(
    files: list[UploadFile] = File(...), paths: str | None = Form(None), dataset_name: str | None = Form(None),
    conf: float | None = Form(None), iou: float | None = Form(None), max_det: int | None = Form(None), warmup: int = Form(2), gradcam: bool = Form(True),
):
    m = _models()
    d = _defaults()["inference"]
    conf = d["conf"] if conf is None else conf
    iou = d["iou"] if iou is None else iou
    max_det = d["max_det"] if max_det is None else max_det
    _check_range("conf", conf, 0.001, 0.99); _check_range("iou", iou, 0.05, 0.95); _check_range("max_det", max_det, 1, 300); _check_range("warmup", warmup, 0, 10)
    staged = await _stage(files, paths, dataset_name)
    _busy_guard(staged)
    fn = lambda root, p, c: runner.run_inference(root, m, conf=conf, iou=iou, max_det=max_det, warmup=warmup, gradcam=gradcam, progress=p, cancel=c)  # noqa: E731
    return _start("inference", staged, {"conf": conf, "iou": iou, "max_det": max_det, "warmup": warmup, "gradcam": gradcam,
                                        "binary_threshold": settings.binary_threshold}, fn)


# --------------------------------------------------------------------------- jobs / runs
@router.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = store.get_job(job_id)
    if job is None:
        raise _err(404, "job_not_found", "Unknown evaluation job.")
    return job.as_dict()


@router.post("/jobs/{job_id}/cancel")
def job_cancel(job_id: str) -> dict:
    job = store.get_job(job_id)
    if job is None:
        raise _err(404, "job_not_found", "Unknown evaluation job.")
    job.cancel_requested = True
    return job.as_dict()


def _run_or_404(run_id: str) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise _err(404, "run_not_found", "Evaluation run not found (unsaved runs are lost when the server restarts).")
    return run


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return _run_or_404(run_id)


@router.get("/history")
def history() -> dict:
    return {"runs": store.list_history()}


@router.post("/history")
def save_history(body: dict = Body(...)) -> dict:
    run = store.save_run(str(body.get("run_id", "")), body.get("name"))
    if run is None:
        raise _err(404, "run_not_found", "Evaluation run not found.")
    return store.run_summary(run)


@router.delete("/history/{run_id}")
def delete_history(run_id: str) -> dict:
    if not store.delete_run(run_id):
        raise _err(404, "run_not_found", "No saved run with that id.")
    return {"deleted": run_id}


@router.get("/compare")
def compare(ids: str = Query(..., description="comma-separated run ids")) -> dict:
    out = []
    for rid in [x.strip() for x in ids.split(",") if x.strip()][:8]:
        run = _run_or_404(rid)
        s = store.run_summary(run)
        c = run.get("classification")
        s["per_class_f1"] = {r["label"]: r["f1"] for r in c["per_class"]} if c else None
        out.append(s)
    if len(out) < 2:
        raise _err(422, "need_two_runs", "Select at least two runs to compare.")
    return {"runs": out}


def _export(run_id: str, fmt: str, table: str) -> Response:
    run = _run_or_404(run_id)
    if fmt == "pdf":
        data = report.to_pdf(run)
        return Response(data, media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="ecosortai_evaluation_{run_id}.pdf"'})
    if fmt == "csv":
        try:
            text, name = report.to_csv(run, table)
        except KeyError:
            raise _err(404, "table_not_found", f"This run has no '{table}' table.")
        return Response(text.encode("utf-8-sig"), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})
    raise _err(422, "invalid_format", "format must be 'pdf' or 'csv'.")


@router.get("/runs/{run_id}/export")
def export_get(run_id: str, format: str = "pdf", table: str = "all") -> Response:  # noqa: A002
    return _export(run_id, format, table)


@router.post("/export")
def export_post(body: dict = Body(...)) -> Response:
    return _export(str(body.get("run_id", "")), str(body.get("format", "pdf")), str(body.get("table", "all")))
