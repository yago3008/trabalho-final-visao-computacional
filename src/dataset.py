from __future__ import annotations

import csv
import hashlib
import random
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageEntry:
    zip_path: str
    original_split: str
    original_class: str
    label: str
    canonical_id: str


def infer_label(folder_name: str) -> str | None:
    lower = folder_name.lower()
    if lower.startswith("fresh"):
        return "fresh"
    if lower.startswith("rotten"):
        return "rotten"
    return None


def canonical_source_name(filename: str) -> str:
    """Remove common augmentation prefixes so related images can be grouped."""
    name = Path(filename).name
    patterns = [
        r"^rotated_by_\d+_",
        r"^translation_",
        r"^vertical_flip_",
        r"^horizontal_flip_",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            new_name = re.sub(pattern, "", name, flags=re.IGNORECASE)
            if new_name != name:
                name = new_name
                changed = True
    return name


def list_dataset_entries(zip_path: Path) -> list[ImageEntry]:
    entries: list[ImageEntry] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            suffix = Path(info.filename).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                continue

            parts = info.filename.replace("\\", "/").split("/")
            # Expected: dataset/dataset/train/freshapples/file.png
            if len(parts) < 5:
                continue
            original_split = parts[2].lower()
            original_class = parts[3].lower()
            if original_split not in {"train", "test"}:
                continue
            label = infer_label(original_class)
            if label is None:
                continue

            canonical = f"{original_split}/{original_class}/{canonical_source_name(parts[-1])}"
            entries.append(
                ImageEntry(
                    zip_path=info.filename,
                    original_split=original_split,
                    original_class=original_class,
                    label=label,
                    canonical_id=canonical,
                )
            )
    return entries


def choose_group_representatives(entries: list[ImageEntry], seed: int) -> list[ImageEntry]:
    """Keep one entry per canonical source image to reduce augmentation leakage."""
    rng = random.Random(seed)
    grouped: dict[str, list[ImageEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.canonical_id, []).append(entry)
    return [rng.choice(items) for items in grouped.values()]


def balanced_sample(
    entries: list[ImageEntry],
    split: str,
    label: str,
    limit: int,
    seed: int,
) -> list[ImageEntry]:
    candidates = [
        item
        for item in entries
        if item.original_split == split and item.label == label
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[: min(limit, len(candidates))]


def split_train_val(
    train_entries: list[ImageEntry],
    val_ratio: float,
    seed: int,
) -> tuple[list[ImageEntry], list[ImageEntry]]:
    rng = random.Random(seed)
    by_label: dict[str, list[ImageEntry]] = {"fresh": [], "rotten": []}
    for item in train_entries:
        by_label[item.label].append(item)

    train_final: list[ImageEntry] = []
    val_final: list[ImageEntry] = []
    for label_entries in by_label.values():
        rng.shuffle(label_entries)
        val_size = max(1, int(round(len(label_entries) * val_ratio)))
        val_final.extend(label_entries[:val_size])
        train_final.extend(label_entries[val_size:])

    rng.shuffle(train_final)
    rng.shuffle(val_final)
    return train_final, val_final


def safe_output_name(entry: ImageEntry) -> str:
    suffix = Path(entry.zip_path).suffix.lower()
    digest = hashlib.sha1(entry.zip_path.encode("utf-8")).hexdigest()[:12]
    return f"{entry.original_class}_{digest}{suffix}"


def extract_entries(
    zip_path: Path,
    output_dir: Path,
    split_name: str,
    entries: list[ImageEntry],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for entry in entries:
            class_dir = output_dir / split_name / entry.label
            class_dir.mkdir(parents=True, exist_ok=True)
            target = class_dir / safe_output_name(entry)
            with zf.open(entry.zip_path) as source, target.open("wb") as destination:
                destination.write(source.read())
            rows.append(
                {
                    "image_path": str(target.as_posix()),
                    "split": split_name,
                    "label": entry.label,
                    "original_class": entry.original_class,
                    "zip_path": entry.zip_path,
                }
            )
    return rows


def write_manifest(rows: list[dict[str, str]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "split", "label", "original_class", "zip_path"]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

