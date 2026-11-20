"""Runtime settings, read from environment variables (optionally a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = BACKEND_DIR / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (BACKEND_DIR / p)


@dataclass(frozen=True)
class Settings:
    # "auto": load model/model_config.json if present, else run without a model.
    # "custom": require the custom checkpoint. "clip": zero-shot baseline. "none": no model.
    backend: str = field(default_factory=lambda: _env("ECOSORT_BACKEND", "auto").lower())
    model_config: Path = field(default_factory=lambda: _path(_env("ECOSORT_MODEL_CONFIG", "model/model_config.json")))
    device: str = field(default_factory=lambda: _env("ECOSORT_DEVICE", "auto"))

    detector_weights: str = field(default_factory=lambda: _env("ECOSORT_DETECTOR_WEIGHTS", ""))
    detector_conf: float = field(default_factory=lambda: float(_env("ECOSORT_DETECTOR_CONF", "0.05")))
    detector_iou: float = field(default_factory=lambda: float(_env("ECOSORT_DETECTOR_IOU", "0.45")))
    max_objects: int = field(default_factory=lambda: int(_env("ECOSORT_MAX_OBJECTS", "30")))

    binary_checkpoint: Path = field(default_factory=lambda: _path(_env("ECOSORT_BINARY_CHECKPOINT", "model/binary_mobilenetv3_best.pth")))
    binary_threshold: float = field(default_factory=lambda: float(_env("ECOSORT_BINARY_THRESHOLD", "0.35")))

    default_explainers: tuple[str, ...] = field(
        default_factory=lambda: tuple(x.strip() for x in _env("ECOSORT_EXPLAINERS", "gradcam,lime,shap").split(",") if x.strip())
    )
    lime_samples: int = field(default_factory=lambda: int(_env("ECOSORT_LIME_SAMPLES", "300")))
    shap_samples: int = field(default_factory=lambda: int(_env("ECOSORT_SHAP_SAMPLES", "300")))
    xai_max_objects: int = field(default_factory=lambda: int(_env("ECOSORT_XAI_MAX_OBJECTS", "30")))

    max_upload_mb: float = field(default_factory=lambda: float(_env("ECOSORT_MAX_UPLOAD_MB", "10")))
    max_pixels: int = field(default_factory=lambda: int(_env("ECOSORT_MAX_PIXELS", str(40_000_000))))
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            x.strip()
            for x in _env(
                "ECOSORT_CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173,null",
            ).split(",")
            if x.strip()
        )
    )


settings = Settings()
