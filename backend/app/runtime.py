"""Process-wide runtime primitives shared by inference and evaluation."""
from __future__ import annotations

import threading

# The loaded models (and Grad-CAM hooks) are not thread-safe. Live analysis and
# evaluation jobs both hold this lock around each model call so they never run
# a forward pass concurrently on the same model instance.
model_lock = threading.Lock()
