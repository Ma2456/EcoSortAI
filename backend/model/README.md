# Put your trained model here

The API looks for `backend/model/model_config.json`. If it is missing, the API still
runs but `/api/analyze` returns **503 `model_not_loaded`** and the Waste Lab shows
"no trained model configured". Nothing is ever faked.

## 1. Copy your checkpoint

```
backend/model/waste_classifier.pth
```

## 2. Create `model_config.json`

Start from `model_config.example.json`:

| key | meaning |
| --- | --- |
| `checkpoint` | file name (relative to this folder) or absolute path |
| `format` | `state_dict` (weights for a torchvision model), `full` (`torch.save(model)`), or `torchscript` |
| `architecture` | torchvision model name for `state_dict`, e.g. `resnet18`, `resnet50`, `efficientnet_b0`, `mobilenet_v3_large`, `densenet121`, `convnext_tiny` |
| `class_names` | output classes **in the exact order your model was trained with** (e.g. `ImageFolder.classes`) |
| `class_to_category` | map your labels to `plastic, paper, metal, organic, ewaste, glass, hazardous` or `unknown`. Common names (`cardboard`, `trash`, `biological`, `battery`, `medical`, `chemical`, `white-glass`…) are mapped automatically |
| `input_size`, `mean`, `std` | the eval transform you trained with |
| `resize` | `center_crop` (resize → centre crop, the usual eval transform) or `stretch` |
| `uncertainty_threshold` | below this category probability the result is reported as **Uncertain** |
| `gradcam_target_layer` | optional module path (e.g. `layer4`, `features.8`); default is the last `Conv2d` |

Example for a TrashNet-style 6-class model:

```json
{
  "name": "trashnet-resnet34",
  "checkpoint": "trashnet_resnet34.pth",
  "format": "state_dict",
  "architecture": "resnet34",
  "class_names": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
  "class_to_category": {"cardboard": "paper", "trash": "unknown"}
}
```

If you replaced the classifier head with a custom `nn.Sequential`, a plain `state_dict`
will not match the stock architecture — save the full model (`torch.save(model, ...)`,
`"format": "full"`) or export TorchScript. Only load `full` checkpoints you trust
(they are unpickled).

## 3. Optional: object detection

Set `ECOSORT_DETECTOR_WEIGHTS=path/to/yolo.pt` (requires `pip install ultralytics`).
Each detected object is cropped and classified separately.

## 4. Restart the API and check

```
curl http://127.0.0.1:8000/api/health
```

`model_loaded: true` and your model name confirm it. Any loading error (wrong key
names, class-count mismatch…) is reported in `model_error`.
