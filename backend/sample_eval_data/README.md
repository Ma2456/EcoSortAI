# Sample evaluation datasets (format examples)

These small folders show the exact layouts the **Performance Evaluation** page accepts.
Zip any folder (right-click → *Send to → Compressed (zipped) folder* on Windows) or pick
it with **Choose folder** on the page.

> **They are not a benchmark.** The images are crops and frames of the project's own
> AI-rendered park video (bins full of waste) plus two Ultralytics sample photos. The
> trained models were not trained on this kind of image, so the scores you get here are
> low and mean nothing about real-world accuracy. Use them to check the workflow, then
> evaluate your real held-out test split.

| Folder | Task | Layout |
|---|---|---|
| `classification/` | Seven-class classification, or end-to-end pipeline | one folder per class: `e-waste/ glass/ hazardous/ metal/ organic/ paper_cardboard/ plastic/` (21 images) |
| `classification_csv/` | same, CSV labels | `images/*.jpg` + `labels.csv` with `filename,label` |
| `binary/` | Binary waste verification | `waste/` (14) and `non_waste/` (8) |
| `detection/` | YOLOv8 detection (mAP) | `images/*.jpg`, `labels/*.txt` (`class cx cy w h`, normalised 0–1), `data.yaml` with `names` |
| `unlabeled/` | Inference benchmark, or to see the "labels required" warning | 3 frames, no labels |

## Accepted class names

`ewaste` / `e-waste` / `e_waste`, `glass`, `hazardous`, `metal`, `organic`,
`paper` / `cardboard` / `paper_cardboard` / `Paper & Cardboard`, `plastic`
(plus common aliases such as `battery → ewaste`, `food → organic`). Folders that do not
map to one of the seven categories (e.g. `textile/`) are reported and excluded.

Binary: `waste` / `non_waste` (also `non-waste`, `nonwaste`, `background`, `other`).
A folder named after a waste category also counts as `waste`.

## Detection labels

```
# labels/plastic.txt   (one line per box)
6 0.22000 0.68500 0.16000 0.41000
```
`data.yaml`:
```yaml
names: [ewaste, glass, hazardous, metal, organic, paper_cardboard, plastic]
```
Dataset class names are matched to the YOLO model's own classes by name, so the order
in your `data.yaml` does not have to match the model. Images without a label file are
treated as background images (no objects), as in Ultralytics.
