"""Create a RANDOMLY INITIALISED resnet18 checkpoint for testing the plumbing.

Its predictions are meaningless. It exists only so the API, Grad-CAM, LIME and
SHAP code paths can be exercised without a trained model. Never ship it.

    python tools/make_pipeline_test_model.py /tmp/ecosort-test-model
    ECOSORT_MODEL_CONFIG=/tmp/ecosort-test-model/model_config.json uvicorn app.main:app
"""
import json
import sys
from pathlib import Path

import torch
import torchvision

out = Path(sys.argv[1] if len(sys.argv) > 1 else "test-model")
out.mkdir(parents=True, exist_ok=True)
classes = ["plastic", "paper", "metal", "organic", "ewaste", "glass", "hazardous"]
torch.manual_seed(0)
model = torchvision.models.resnet18(weights=None, num_classes=len(classes))
torch.save(model.state_dict(), out / "random_resnet18.pth")
(out / "model_config.json").write_text(
    json.dumps(
        {
            "name": "PIPELINE-TEST random resnet18 (untrained)",
            "checkpoint": "random_resnet18.pth",
            "format": "state_dict",
            "architecture": "resnet18",
            "class_names": classes,
            "input_size": 224,
            "uncertainty_threshold": 0.55,
        },
        indent=2,
    )
)
print(f"Wrote untrained test model to {out}")
