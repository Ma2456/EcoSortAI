"""EcoSortAI FastAPI application.

Endpoints
  GET  /api/health      backend + model status
  GET  /api/categories  the seven categories and disposal guidance
  POST /api/analyze     multipart image upload → detection, classification, XAI, recommendation
  /api/evaluation/*     Performance Evaluation module (see app/evaluation/api.py)
"""
from __future__ import annotations

import asyncio
import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps, UnidentifiedImageError

from . import __version__
from .categories import CATEGORIES
from .models.registry import registry
from .evaluation.api import router as evaluation_router
from .pipeline import analyze
from .runtime import model_lock
from .settings import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ecosort")

ACCEPTED = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
ALLOWED_EXPLAINERS = {"gradcam", "lime", "shap"}
_lock = asyncio.Semaphore(1)  # one inference at a time (models/hooks are not thread-safe)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await run_in_threadpool(registry.load)
    yield


app = FastAPI(title="EcoSortAI API", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
app.include_router(evaluation_router)


def _locked_analyze(*args):
    # Shared with evaluation jobs so two forward passes never use the same model at once.
    with model_lock:
        return analyze(*args)


def _err(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


@app.get("/api/health")
def health() -> dict:
    clf = registry.classifier
    xai_ok = {"gradcam": bool(clf and clf.supports_gradcam), "lime": False, "shap": False}
    try:
        import lime  # noqa: F401

        xai_ok["lime"] = clf is not None
    except ImportError:
        pass
    try:
        import shap  # noqa: F401

        xai_ok["shap"] = clf is not None
    except ImportError:
        pass
    return {
        "status": "ok",
        "version": __version__,
        "model_loaded": clf is not None,
        "model": clf.describe() if clf else None,
        "binary_filter_loaded": registry.binary_filter is not None,
        "binary_filter_error": registry.binary_error,
        "binary_threshold": settings.binary_threshold,
        "model_error": registry.model_error,
        "detector": {"name": registry.detector.name} if registry.detector else None,
        "detector_error": registry.detector_error,
        "explainers": xai_ok,
        "limits": {"max_upload_mb": settings.max_upload_mb, "accepted_types": list(ACCEPTED)},
    }


@app.get("/api/categories")
def categories() -> dict:
    return {"categories": [{"id": k, **v} for k, v in CATEGORIES.items()]}


@app.post("/api/analyze")
async def analyze_endpoint(file: UploadFile = File(...), explain: str | None = Form(None)) -> dict:
    clf = registry.classifier
    if clf is None:
        raise _err(503, "model_not_loaded", registry.model_error or "No model is loaded.")

    if file.content_type not in ACCEPTED:
        raise _err(415, "unsupported_type", "Unsupported file type. Upload a JPG, JPEG, PNG or WebP image.")

    limit = int(settings.max_upload_mb * 1024 * 1024)
    data = await file.read(limit + 1)
    if len(data) == 0:
        raise _err(400, "empty_file", "The uploaded file is empty.")
    if len(data) > limit:
        raise _err(413, "file_too_large", f"The image exceeds the {settings.max_upload_mb:g} MB limit.")

    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ACCEPTED.values():
            raise _err(415, "unsupported_type", "The file content is not a JPG, PNG or WebP image.")
        if img.width * img.height > settings.max_pixels:
            raise _err(413, "image_too_large", "The image resolution is too large.")
        img.load()
        img = ImageOps.exif_transpose(img).convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ImportError):
        # ImportError: newer Pillow versions try to load optional format plugins for unknown files.
        raise _err(400, "invalid_image", "The file could not be read as an image. It may be corrupted.")

    if min(img.size) < 16:
        raise _err(400, "image_too_small", "The image is too small to analyse.")

    requested = settings.default_explainers if explain is None else tuple(x.strip().lower() for x in explain.split(",") if x.strip())
    bad = [x for x in requested if x not in ALLOWED_EXPLAINERS]
    if bad:
        raise _err(422, "invalid_explainer", f"Unknown explainer(s): {', '.join(bad)}")

    async with _lock:
        try:
            result = await run_in_threadpool(_locked_analyze, img, clf, registry.detector, requested, registry.binary_filter)
        except Exception as exc:
            log.exception("Analysis failed")
            raise _err(500, "analysis_failed", f"Analysis failed: {exc}")
    if registry.detector_error:
        result["warnings"].append(registry.detector_error)
    return result
