"""Dataset ingestion and validation for evaluation runs.

Uploaded datasets (a ZIP or the files of a folder) are written to a private
temporary directory that is deleted when the evaluation finishes. Nothing is
kept on disk afterwards.

Supported layouts
-----------------
Classification / binary (either):
  <root>/<label>/<image>          label = immediate parent folder name
  <root>/labels.csv               columns: filename,label   (paths relative to the CSV)

Object detection (Ultralytics YOLO format):
  <root>/images/<...>/<name>.jpg
  <root>/labels/<...>/<name>.txt  one "class cx cy w h" line per box (normalised 0–1)
  <root>/data.yaml (names: [...] or {0: ...})  or  <root>/classes.txt
"""
from __future__ import annotations

import csv
import io
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps, UnidentifiedImageError

from ..categories import CATEGORIES, DEFAULT_ALIASES, UNKNOWN

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PIL_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "MPO"}
SKIP_PARTS = {"__macosx", ".ds_store", "thumbs.db"}

# Extra spellings commonly used for EcoSortAI's seven categories in folder names.
EVAL_ALIASES = {
    "paper_cardboard": "paper", "paper-cardboard": "paper", "paper & cardboard": "paper",
    "paper and cardboard": "paper", "paper&cardboard": "paper", "paper_and_cardboard": "paper",
    "e waste": "ewaste", "e_waste": "ewaste", "e-waste": "ewaste",
    "hazardous_waste": "hazardous", "organic_waste": "organic", "food_waste": "organic",
}
NON_WASTE_ALIASES = {"non_waste", "non-waste", "nonwaste", "non waste", "not_waste", "not-waste", "notwaste", "background", "negative", "other"}
WASTE_ALIASES = {"waste", "positive"}


class DatasetError(ValueError):
    """A problem with the uploaded dataset that the user must fix."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def resolve_eval_category(raw: str, model_mapping: dict[str, str] | None = None) -> str:
    key = raw.strip().lower()
    if model_mapping:
        for k, v in model_mapping.items():
            if k.lower() == key:
                return v
    if key in EVAL_ALIASES:
        return EVAL_ALIASES[key]
    key2 = key.replace(" ", "_")
    if key2 in EVAL_ALIASES:
        return EVAL_ALIASES[key2]
    if key in CATEGORIES:
        return key
    return DEFAULT_ALIASES.get(key, DEFAULT_ALIASES.get(key2, UNKNOWN))


def resolve_binary_label(raw: str) -> int | None:
    key = raw.strip().lower()
    if key in NON_WASTE_ALIASES or key.replace(" ", "_") in NON_WASTE_ALIASES:
        return 0
    if key in WASTE_ALIASES:
        return 1
    cat = resolve_eval_category(raw)
    if cat in CATEGORIES:
        return 1  # a folder named after a waste category is waste
    return None


# --------------------------------------------------------------------------- ingestion
@dataclass
class StagedDataset:
    root: Path
    name: str
    source: str  # "zip" | "folder" | "files"
    _tmp: Path | None = None

    def cleanup(self) -> None:
        if self._tmp and self._tmp.exists():
            shutil.rmtree(self._tmp, ignore_errors=True)


def _safe_rel(name: str) -> PurePosixPath | None:
    name = name.replace("\\", "/")
    p = PurePosixPath(name)
    parts = [x for x in p.parts if x not in ("", ".")]
    if not parts or any(x == ".." for x in parts) or p.is_absolute() or re.match(r"^[a-zA-Z]:", name):
        return None
    if any(x.lower() in SKIP_PARTS or x.startswith("._") for x in parts):
        return None
    return PurePosixPath(*parts)


def stage_upload(
    files: list[tuple[str, io.BufferedIOBase | bytes]],
    dataset_name: str | None,
    max_extract_bytes: int,
    max_files: int,
) -> StagedDataset:
    """``files``: (relative path, file-like) pairs from the multipart upload."""
    tmp = Path(tempfile.mkdtemp(prefix="ecosort-eval-"))
    root = tmp / "data"
    root.mkdir()
    try:
        if len(files) == 1 and files[0][0].lower().endswith(".zip"):
            name = dataset_name or Path(files[0][0]).stem
            src = files[0][1]
            zpath = tmp / "upload.zip"
            with open(zpath, "wb") as out:
                if isinstance(src, (bytes, bytearray)):
                    out.write(src)
                else:
                    shutil.copyfileobj(src, out, 1024 * 1024)
            _extract_zip(zpath, root, max_extract_bytes, max_files)
            zpath.unlink(missing_ok=True)
            source = "zip"
        else:
            if len(files) > max_files:
                raise DatasetError("too_many_files", f"The dataset has {len(files)} files; the limit is {max_files}.")
            total = 0
            for rel, src in files:
                rp = _safe_rel(rel)
                if rp is None:
                    continue
                dest = root / rp
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as out:
                    if isinstance(src, (bytes, bytearray)):
                        out.write(src)
                        total += len(src)
                    else:
                        shutil.copyfileobj(src, out, 1024 * 1024)
                        total += out.tell()
                if total > max_extract_bytes:
                    raise DatasetError("dataset_too_large", f"The dataset exceeds {max_extract_bytes // 1048576} MB.")
            first = _safe_rel(files[0][0]) if files else None
            name = dataset_name or (first.parts[0] if first and len(first.parts) > 1 else "uploaded-images")
            source = "folder" if any("/" in f[0].replace("\\", "/") for f in files) else "files"
        root = _collapse_single_dir(root)
        return StagedDataset(root=root, name=name, source=source, _tmp=tmp)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def stage_directory(path: Path) -> StagedDataset:
    """Use an existing directory in place (CLI / tests). Never deleted."""
    if not path.is_dir():
        raise DatasetError("not_a_directory", f"{path} is not a directory.")
    return StagedDataset(root=_collapse_single_dir(path), name=path.name, source="directory")


def _extract_zip(zpath: Path, root: Path, max_bytes: int, max_files: int) -> None:
    try:
        zf = zipfile.ZipFile(zpath)
    except zipfile.BadZipFile:
        raise DatasetError("invalid_zip", "The uploaded file is not a valid ZIP archive.")
    with zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) > max_files:
            raise DatasetError("too_many_files", f"The ZIP contains {len(infos)} files; the limit is {max_files}.")
        if sum(i.file_size for i in infos) > max_bytes:
            raise DatasetError("dataset_too_large", f"The extracted dataset would exceed {max_bytes // 1048576} MB.")
        for info in infos:
            rp = _safe_rel(info.filename)
            if rp is None:
                continue
            dest = root / rp
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out, 1024 * 1024)


def _collapse_single_dir(root: Path) -> Path:
    """A ZIP of a folder contains one top-level directory; descend into it."""
    cur = root
    for _ in range(3):
        entries = [e for e in cur.iterdir() if e.name.lower() not in SKIP_PARTS and not e.name.startswith(".")]
        if len(entries) == 1 and entries[0].is_dir() and entries[0].name.lower() not in ("images", "labels"):
            # stop if it's a class folder (contains images directly) – only descend when it holds dirs/files of a dataset
            inner = list(entries[0].iterdir())
            if any(e.is_file() and e.suffix.lower() in IMAGE_EXT for e in inner) and not any(e.is_dir() for e in inner):
                break  # e.g. a single class folder – keep it as a label
            cur = entries[0]
        else:
            break
    return cur


# --------------------------------------------------------------------------- discovery
@dataclass
class Sample:
    path: Path
    rel: str
    label_raw: str | None = None
    label: str | int | None = None  # category id (classification) / 0-1 (binary)


@dataclass
class Discovery:
    samples: list[Sample] = field(default_factory=list)
    label_format: str = "none"
    invalid_files: list[dict] = field(default_factory=list)
    unmapped_labels: dict[str, int] = field(default_factory=dict)
    ignored_files: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def labeled(self) -> list[Sample]:
        return [s for s in self.samples if s.label is not None]


def list_images(root: Path) -> tuple[list[Path], int]:
    imgs, ignored = [], 0
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d.lower() not in SKIP_PARTS and not d.startswith("."))
        for f in sorted(fns):
            if f.startswith(".") or f.lower() in SKIP_PARTS:
                continue
            p = Path(dp) / f
            if p.suffix.lower() in IMAGE_EXT:
                imgs.append(p)
            elif p.suffix.lower() not in {".csv", ".txt", ".yaml", ".yml", ".json", ".md"}:
                ignored += 1
    return imgs, ignored


def _read_label_csv(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise DatasetError("invalid_labels_csv", "labels.csv is empty.")
    cols = {c.strip().lower(): c for c in reader.fieldnames}
    fcol = next((cols[c] for c in ("filename", "file", "image", "path", "image_path") if c in cols), None)
    lcol = next((cols[c] for c in ("label", "class", "category", "target") if c in cols), None)
    if not fcol or not lcol:
        raise DatasetError("invalid_labels_csv", "labels.csv needs a 'filename' column and a 'label' column.")
    out = {}
    for row in reader:
        f, l = (row.get(fcol) or "").strip(), (row.get(lcol) or "").strip()
        if f and l:
            out[f.replace("\\", "/")] = l
    return out


GENERIC_PARENTS = {"images", "image", "imgs", "img", "test", "val", "valid", "validation", "train", "data", "dataset", "samples", "unlabeled", "unlabelled"}


def discover_labeled_images(root: Path, kind: str, model_mapping: dict[str, str] | None = None) -> Discovery:
    """kind: "classification" (labels → category ids) or "binary" (labels → 0/1)."""
    d = Discovery()
    images, d.ignored_files = list_images(root)
    csvs = sorted(root.rglob("labels.csv"), key=lambda p: len(p.parts))
    csv_map: dict[Path, str] = {}
    if csvs:
        base = csvs[0].parent
        for f, l in _read_label_csv(csvs[0]).items():
            csv_map[(base / f).resolve()] = l
        d.label_format = "labels.csv"
    else:
        d.label_format = "folder-per-class"

    for p in images:
        rel = p.relative_to(root).as_posix()
        if csvs:
            raw = csv_map.get(p.resolve())
            if raw is None:  # allow bare file names in the CSV
                raw = next((l for k, l in csv_map.items() if k.name == p.name), None)
        else:
            parent = p.parent.name if p.parent != root else None
            raw = parent if parent and parent.lower() not in GENERIC_PARENTS else None
        s = Sample(path=p, rel=rel, label_raw=raw)
        if raw is not None:
            if kind == "binary":
                s.label = resolve_binary_label(raw)
            else:
                cat = resolve_eval_category(raw, model_mapping)
                s.label = cat if cat in CATEGORIES else None
            if s.label is None:
                d.unmapped_labels[raw] = d.unmapped_labels.get(raw, 0) + 1
        d.samples.append(s)

    if not d.labeled:
        d.label_format = "none"
    if d.unmapped_labels:
        allowed = "waste / non_waste (or a waste category name)" if kind == "binary" else ", ".join(CATEGORIES)
        d.warnings.append(
            f"{sum(d.unmapped_labels.values())} image(s) had labels that could not be mapped "
            f"({', '.join(sorted(d.unmapped_labels))}) and were excluded from supervised metrics. Expected: {allowed}."
        )
    return d


def load_image(path: Path, max_pixels: int) -> Image.Image:
    """Same decoding rules as /api/analyze: EXIF orientation, RGB, size limits."""
    try:
        img = Image.open(path)
        if img.format not in PIL_FORMATS:
            raise DatasetError("unsupported_image", f"Unsupported image format {img.format}.")
        if img.width * img.height > max_pixels:
            raise DatasetError("image_too_large", "Image resolution is too large.")
        img.load()
        img = ImageOps.exif_transpose(img).convert("RGB")
    except DatasetError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError, ValueError, ImportError):
        raise DatasetError("invalid_image", "Could not be read as an image (corrupted or not an image).")
    if min(img.size) < 16:
        raise DatasetError("image_too_small", "Image is smaller than 16 px.")
    return img


# --------------------------------------------------------------------------- YOLO detection labels
@dataclass
class YoloDataset:
    images: list[Path]
    names: list[str] | None
    names_source: str
    has_labels_dir: bool
    warnings: list[str] = field(default_factory=list)


def _read_names(root: Path) -> tuple[list[str] | None, str]:
    for yml in sorted(list(root.rglob("*.yaml")) + list(root.rglob("*.yml")), key=lambda p: len(p.parts)):
        try:
            import yaml

            data = yaml.safe_load(yml.read_text(encoding="utf-8", errors="replace")) or {}
        except Exception:
            continue
        names = data.get("names") if isinstance(data, dict) else None
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x))], yml.name
        if isinstance(names, list):
            return [str(n) for n in names], yml.name
    for txt in ("classes.txt", "obj.names", "names.txt"):
        hits = sorted(root.rglob(txt), key=lambda p: len(p.parts))
        if hits:
            names = [l.strip() for l in hits[0].read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
            if names:
                return names, hits[0].name
    return None, "none"


def discover_yolo(root: Path) -> YoloDataset:
    images, _ = list_images(root)
    has_labels = any(p.is_dir() and p.name.lower() == "labels" for p in root.rglob("labels"))
    names, src = _read_names(root)
    return YoloDataset(images=images, names=names, names_source=src, has_labels_dir=has_labels)


def yolo_label_path(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return img.with_suffix(".txt")  # labels next to the image


def read_yolo_labels(path: Path, width: int, height: int, n_classes: int | None) -> tuple[list[list[float]], list[int], list[str]]:
    """Returns pixel xyxy boxes, class ids, and a list of problems found."""
    boxes, classes, problems = [], [], []
    if not path.exists():
        return boxes, classes, problems
    for ln, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            problems.append(f"line {ln}: non-numeric values")
            continue
        if len(vals) < 5:
            problems.append(f"line {ln}: expected 'class cx cy w h'")
            continue
        c = vals[0]
        if c != int(c) or c < 0 or (n_classes is not None and int(c) >= n_classes):
            problems.append(f"line {ln}: class id {parts[0]} out of range")
            continue
        if len(vals) == 5:
            cx, cy, w, h = vals[1:]
            x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        elif (len(vals) - 1) % 2 == 0 and len(vals) >= 7:  # polygon (segmentation format) → bbox
            xs, ys = vals[1::2], vals[2::2]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        else:
            problems.append(f"line {ln}: unexpected number of values ({len(vals)})")
            continue
        if min(x0, y0) < -0.01 or max(x1, y1) > 1.01 or x1 <= x0 or y1 <= y0:
            problems.append(f"line {ln}: coordinates outside 0–1 or zero-size box")
            continue
        x0, y0, x1, y1 = (min(max(v, 0.0), 1.0) for v in (x0, y0, x1, y1))
        boxes.append([x0 * width, y0 * height, x1 * width, y1 * height])
        classes.append(int(c))
    return boxes, classes, problems
