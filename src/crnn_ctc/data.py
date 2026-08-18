from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .vocab import CTCVocabulary


class ManifestError(ValueError):
    """The dataset manifest does not satisfy the baseline input contract."""


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    split: str
    text: str
    layout: str
    file_name: str
    image_path: Path
    metadata: dict[str, str]


def _normalise_filter(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    selected = {value.strip().lower() for value in values if value.strip()}
    return selected or None


def _label_rows(root: Path) -> tuple[str, list[dict]]:
    """Label rows from manifest.csv, or from the imagefolder metadata files.

    The builder writes both; only images/<split>/metadata.jsonl is distributed,
    since it carries every field read here and loads as a HuggingFace
    imagefolder. manifest.csv additionally holds the per-character boxes.
    """

    manifest = root / "manifest.csv"
    if manifest.is_file():
        with manifest.open(encoding="utf-8", newline="") as handle:
            return str(manifest), list(csv.DictReader(handle))
    rows: list[dict] = []
    for split in ("train", "val", "test"):
        path = root / "images" / split / "metadata.jsonl"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise FileNotFoundError(
            f"no labels found in {root}: expected manifest.csv or "
            "images/<split>/metadata.jsonl"
        )
    return f"{root / 'images'}/<split>/metadata.jsonl", rows


def read_manifest(
    dataset_dir: str | Path,
    split: str | None = None,
    layouts: Iterable[str] | None = None,
    tiers: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[ManifestSample]:
    root = Path(dataset_dir)
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    source, rows = _label_rows(root)

    wanted_split = split.strip() if split else None
    wanted_layouts = _normalise_filter(layouts)
    wanted_tiers = _normalise_filter(tiers)
    required = {"sample_id", "split", "text", "layout", "file_name"}
    samples: list[ManifestSample] = []
    seen_ids: set[str] = set()

    missing = required.difference(rows[0])
    if missing:
        raise ManifestError(
            f"{source} is missing required columns: " + ", ".join(sorted(missing))
        )

    for line_number, row in enumerate(rows, start=2):
        row_split = row["split"].strip()
        layout = row["layout"].strip().lower()
        tier = row.get("tier", "").strip().lower()
        if wanted_split is not None and row_split != wanted_split:
            continue
        if wanted_layouts is not None and layout not in wanted_layouts:
            continue
        if wanted_tiers is not None and tier not in wanted_tiers:
            continue
        if layout not in {"horizontal", "vertical"}:
            raise ManifestError(
                f"unsupported layout {layout!r} at {source}:{line_number}"
            )

        sample_id = row["sample_id"].strip()
        file_name = row["file_name"].strip()
        if not sample_id or not file_name:
            raise ManifestError(
                f"blank sample_id or file_name at {source}:{line_number}"
            )
        if sample_id in seen_ids:
            raise ManifestError(f"duplicate sample_id in manifest: {sample_id}")
        seen_ids.add(sample_id)

        relative_image = Path(file_name)
        image_path = (
            relative_image
            if relative_image.is_absolute()
            else root / "images" / row_split / relative_image
        )
        samples.append(
            ManifestSample(
                sample_id=sample_id,
                split=row_split,
                text=row["text"],
                layout=layout,
                file_name=file_name,
                image_path=image_path,
                metadata=dict(row),
            )
        )
        if limit is not None and len(samples) >= limit:
            break

    return samples


def load_line_image(
    path: str | Path,
    layout: str,
    target_height: int,
) -> torch.Tensor:
    if target_height <= 0:
        raise ValueError("target_height must be positive")
    layout = layout.lower()
    if layout not in {"horizontal", "vertical"}:
        raise ValueError(f"unsupported layout: {layout!r}")

    with Image.open(path) as source:
        image = source.convert("L")
        if layout == "vertical":
            image = image.transpose(Image.Transpose.ROTATE_90)
        if image.height <= 0 or image.width <= 0:
            raise ValueError(f"empty image: {path}")
        target_width = max(1, round(image.width * target_height / image.height))
        if image.size != (target_width, target_height):
            image = image.resize(
                (target_width, target_height), resample=Image.Resampling.LANCZOS
            )
        pixels = np.array(image, dtype=np.float32, copy=True)

    pixels = (255.0 - pixels) / 255.0
    return torch.from_numpy(pixels).unsqueeze(0)


class TangutLineDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[ManifestSample],
        vocabulary: CTCVocabulary | None,
        height: int = 64,
    ) -> None:
        if height <= 0:
            raise ValueError("height must be positive")
        self.samples = list(samples)
        self.vocabulary = vocabulary
        self.height = height

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = load_line_image(sample.image_path, sample.layout, self.height)
        target = (
            self.vocabulary.encode(sample.text)
            if self.vocabulary is not None
            else None
        )
        return {
            "image": image,
            "target": target,
            "sample_id": sample.sample_id,
            "file_name": sample.file_name,
            "text": sample.text,
            "layout": sample.layout,
        }


def collate_line_images(batch: Sequence[dict], pad_to_multiple: int = 4) -> dict:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    if pad_to_multiple <= 0:
        raise ValueError("pad_to_multiple must be positive")

    channels, height = batch[0]["image"].shape[:2]
    widths = torch.tensor([item["image"].shape[-1] for item in batch], dtype=torch.long)
    max_width = int(widths.max().item())
    padded_width = int(math.ceil(max_width / pad_to_multiple) * pad_to_multiple)
    images = torch.zeros(
        len(batch), channels, height, padded_width, dtype=batch[0]["image"].dtype
    )
    for index, item in enumerate(batch):
        image = item["image"]
        if image.shape[0] != channels or image.shape[1] != height:
            raise ValueError("all images in a batch must share channel count and height")
        images[index, :, :, : image.shape[-1]] = image

    raw_targets = [item["target"] for item in batch]
    if all(target is not None for target in raw_targets):
        target_lengths = torch.tensor(
            [int(target.numel()) for target in raw_targets], dtype=torch.long
        )
        targets = (
            torch.cat(raw_targets)
            if int(target_lengths.sum().item())
            else torch.empty(0, dtype=torch.long)
        )
    elif all(target is None for target in raw_targets):
        targets = None
        target_lengths = None
    else:
        raise ValueError("a batch cannot mix encoded and unencoded targets")

    return {
        "images": images,
        "image_widths": widths,
        "targets": targets,
        "target_lengths": target_lengths,
        "sample_ids": [item["sample_id"] for item in batch],
        "file_names": [item["file_name"] for item in batch],
        "texts": [item["text"] for item in batch],
        "layouts": [item["layout"] for item in batch],
    }


def make_collate_fn(pad_to_multiple: int = 4):
    return partial(collate_line_images, pad_to_multiple=pad_to_multiple)
