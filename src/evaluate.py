#!/usr/bin/env python
"""Score Tangut OCR predictions against the dataset labels.

    python src/evaluate.py --predictions eval/crnn_ctc/test_predictions.jsonl \
        --json-out eval/crnn_ctc/test_results.json

Micro-CER is the primary metric: total edit distance over total reference
characters. Exact-line accuracy, punctuation-stripped CER and the
substitution/deletion/insertion split are reported alongside, broken down by
tier, layout, font and document.

Coverage is strict: exactly one prediction per expected sample. A missing or
duplicated prediction is an error rather than a skipped row, because scoring a
subset silently inflates the result. --allow-partial relaxes this for debugging.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Importable when run directly as a script, not only as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dataset.corpus import KEPT_PUNCTUATION

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset" / "synthetic"


class EvaluationInputError(ValueError):
    """A manifest or prediction file violates the evaluator input contract."""


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def edit_counts(reference: str, prediction: str) -> tuple[int, int, int]:
    """Split one alignment into substitutions, deletions and insertions."""

    rows, columns = len(reference) + 1, len(prediction) + 1
    distance = [[0] * columns for _ in range(rows)]
    for i in range(rows):
        distance[i][0] = i
    for j in range(columns):
        distance[0][j] = j
    for i, ref_char in enumerate(reference, start=1):
        for j, pred_char in enumerate(prediction, start=1):
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + (ref_char != pred_char),
            )
    # Backtrack the finished matrix: the distance gives the total, only the path
    # says which operation each edit was.
    substitutions = deletions = insertions = 0
    i, j = len(reference), len(prediction)
    while i or j:
        if (
            i
            and j
            and reference[i - 1] == prediction[j - 1]
            and distance[i][j] == distance[i - 1][j - 1]
        ):
            i -= 1
            j -= 1
        elif i and j and distance[i][j] == distance[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i and distance[i][j] == distance[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return substitutions, deletions, insertions


def strip_punctuation(text: str) -> str:
    return "".join(ch for ch in text if ch not in KEPT_PUNCTUATION)


class Accumulator:
    def __init__(self) -> None:
        self.edits = 0
        self.substitutions = 0
        self.deletions = 0
        self.insertions = 0
        self.ref_chars = 0
        self.no_punctuation_edits = 0
        self.no_punctuation_ref_chars = 0
        self.exact = 0
        self.n = 0

    def add(self, reference: str, prediction: str) -> None:
        substitutions, deletions, insertions = edit_counts(reference, prediction)
        self.substitutions += substitutions
        self.deletions += deletions
        self.insertions += insertions
        self.edits += substitutions + deletions + insertions
        self.ref_chars += len(reference)
        ref_no_p = strip_punctuation(reference)
        pred_no_p = strip_punctuation(prediction)
        self.no_punctuation_edits += levenshtein(ref_no_p, pred_no_p)
        self.no_punctuation_ref_chars += len(ref_no_p)
        self.exact += reference == prediction
        self.n += 1

    @property
    def cer(self) -> float:
        return self.edits / self.ref_chars if self.ref_chars else float("nan")

    @property
    def punctuation_stripped_cer(self) -> float:
        if not self.no_punctuation_ref_chars:
            return float("nan")
        return self.no_punctuation_edits / self.no_punctuation_ref_chars

    @property
    def exact_rate(self) -> float:
        return self.exact / self.n if self.n else float("nan")


def _label_rows(root: Path) -> tuple[str, list[dict]]:
    """Label rows from manifest.csv, or from the imagefolder metadata files.

    The builder writes both; only images/<split>/metadata.jsonl is distributed,
    since it carries every field read here. manifest.csv additionally holds the
    per-character boxes.
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


def load_manifest(dataset_dir: Path, split: str | None) -> dict[str, dict[str, str]]:
    source, records = _label_rows(dataset_dir)
    rows: dict[str, dict[str, str]] = {}
    required = {"sample_id", "split", "text", "tier", "layout", "font", "doc", "file_name"}
    absent = required.difference(records[0])
    if absent:
        raise EvaluationInputError(
            f"{source} is missing required column(s): {', '.join(sorted(absent))}"
        )
    for line_number, row in enumerate(records, start=2):
        if split and row["split"] != split:
            continue
        sid = row["sample_id"]
        if not sid:
            raise EvaluationInputError(f"{source} line {line_number}: empty sample_id")
        if sid in rows:
            raise EvaluationInputError(
                f"{source} line {line_number}: duplicate sample_id {sid!r}"
            )
        rows[sid] = row
    return rows


@dataclass(frozen=True)
class LoadedPredictions:
    predictions: dict[str, str]
    unmatched: tuple[str, ...]


def _normalise_file_name(name: str) -> str:
    return name.replace("\\", "/")


def _file_name_index(manifest: dict[str, dict]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for sid, row in manifest.items():
        name = _normalise_file_name(row["file_name"])
        index[name].add(sid)
        parts = [part for part in name.split("/") if part]
        if len(parts) >= 2:
            index["/".join(parts[-2:])].add(sid)
    return index


def _resolve_file_name(name: str, index: dict[str, set[str]]) -> str | None:
    normalised = _normalise_file_name(name)
    candidates = index.get(normalised)
    if not candidates:
        parts = [part for part in normalised.split("/") if part]
        if len(parts) >= 2:
            candidates = index.get("/".join(parts[-2:]))
    if candidates and len(candidates) == 1:
        return next(iter(candidates))
    return None


def read_predictions(path: Path, manifest: dict[str, dict]) -> LoadedPredictions:
    by_file = _file_name_index(manifest)
    predictions: dict[str, str] = {}
    prediction_lines: dict[str, int] = {}
    unmatched: list[str] = []
    errors: list[str] = []

    with path.open(encoding="utf-8") as fh:
        for line_number, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(
                    f"line {line_number}: invalid JSON ({exc.msg} at column {exc.colno})"
                )
                continue
            if not isinstance(obj, dict):
                errors.append(f"line {line_number}: each record must be a JSON object")
                continue
            if "prediction" not in obj:
                errors.append(f"line {line_number}: missing required field 'prediction'")
                continue
            prediction = obj["prediction"]
            if not isinstance(prediction, str):
                errors.append(f"line {line_number}: 'prediction' must be a string")
                continue

            has_sid = "sample_id" in obj
            has_file = "file_name" in obj
            if not has_sid and not has_file:
                errors.append(
                    f"line {line_number}: missing identifier ('sample_id' or 'file_name')"
                )
                continue
            sid_value = obj.get("sample_id")
            file_value = obj.get("file_name")
            if has_sid and (not isinstance(sid_value, str) or not sid_value):
                errors.append(f"line {line_number}: 'sample_id' must be a non-empty string")
                continue
            if has_file and (not isinstance(file_value, str) or not file_value):
                errors.append(f"line {line_number}: 'file_name' must be a non-empty string")
                continue

            sid_from_id = sid_value if has_sid and sid_value in manifest else None
            sid_from_file = _resolve_file_name(file_value, by_file) if has_file else None
            if has_sid and has_file:
                if sid_from_id is None and sid_from_file is None:
                    unmatched.append(
                        f"line {line_number} (sample_id={sid_value!r}, file_name={file_value!r})"
                    )
                    continue
                if sid_from_id != sid_from_file:
                    errors.append(
                        f"line {line_number}: 'sample_id' and 'file_name' do not identify "
                        "the same manifest row"
                    )
                    continue
                sid = sid_from_id
            elif has_sid:
                sid = sid_from_id
                if sid is None:
                    unmatched.append(f"line {line_number} (sample_id={sid_value!r})")
                    continue
            else:
                sid = sid_from_file
                if sid is None:
                    unmatched.append(f"line {line_number} (file_name={file_value!r})")
                    continue

            assert sid is not None
            if sid in predictions:
                errors.append(
                    f"line {line_number}: duplicate prediction for sample_id {sid!r} "
                    f"(first seen on line {prediction_lines[sid]})"
                )
                continue
            predictions[sid] = prediction
            prediction_lines[sid] = line_number

    if errors:
        preview = errors[:10]
        if len(errors) > 10:
            preview.append(f"... and {len(errors) - 10} more error(s)")
        raise EvaluationInputError("invalid predictions file:\n  " + "\n  ".join(preview))
    return LoadedPredictions(predictions, tuple(unmatched))


def report(
    manifest: dict[str, dict],
    preds: dict[str, str],
    field_order=("tier", "layout", "font", "doc"),
):
    overall = Accumulator()
    groups: dict[str, dict[str, Accumulator]] = {
        field: defaultdict(Accumulator) for field in field_order
    }
    for sid, prediction in preds.items():
        row = manifest[sid]
        reference = row["text"]
        overall.add(reference, prediction)
        for field in field_order:
            groups[field][row[field]].add(reference, prediction)

    lines = [f"scored {overall.n:,} of {len(manifest):,} samples\n"]
    lines.append(
        f"  CER {overall.cer:7.4f}   CER-no-p {overall.punctuation_stripped_cer:7.4f}   "
        f"exact {overall.exact_rate:7.4f}\n"
    )
    lines.append(
        f"  edits: S={overall.substitutions:,} D={overall.deletions:,} "
        f"I={overall.insertions:,} / N={overall.ref_chars:,}\n"
    )
    tier_order = ["clean", "light", "medium", "heavy"]
    for field in field_order:
        table = groups[field]
        if len(table) < 2:
            continue
        keys = (
            [tier for tier in tier_order if tier in table]
            if field == "tier"
            else sorted(table, key=lambda key: -table[key].cer)
        )
        lines.append(f"\nby {field}:")
        lines.append(
            f"  {'':28s} {'n':>7s} {'CER':>8s} {'CER-no-p':>9s} {'exact':>8s}"
        )
        for key in keys:
            acc = table[key]
            lines.append(
                f"  {key[:28]:28s} {acc.n:7d} {acc.cer:8.4f} "
                f"{acc.punctuation_stripped_cer:9.4f} {acc.exact_rate:8.4f}"
            )
    return "\n".join(lines), overall, groups


def _metric_payload(acc: Accumulator) -> dict[str, int | float]:
    return {
        "n": acc.n,
        "cer": acc.cer,
        "substitutions": acc.substitutions,
        "deletions": acc.deletions,
        "insertions": acc.insertions,
        "reference_characters": acc.ref_chars,
        "punctuation_stripped_cer": acc.punctuation_stripped_cer,
        "exact": acc.exact_rate,
    }


def _preview(items: list[str] | tuple[str, ...], limit: int = 10) -> str:
    shown = list(items[:limit])
    suffix = f", ... (+{len(items) - limit} more)" if len(items) > limit else ""
    return ", ".join(shown) + suffix


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET),
                        help="dataset directory")
    parser.add_argument("--predictions", required=True, help="predictions .jsonl")
    parser.add_argument("--split", default="test", help="restrict to a split ('' for all)")
    parser.add_argument("--json-out", default=None, help="also write results as JSON")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="debug only: score matched records even when coverage is incomplete",
    )
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(Path(args.dataset), args.split or None)
    except (OSError, csv.Error, EvaluationInputError) as exc:
        print(f"failed to load manifest: {exc}", file=sys.stderr)
        return 2
    if not manifest:
        print(f"no manifest rows for split {args.split!r}", file=sys.stderr)
        return 2
    try:
        loaded = read_predictions(Path(args.predictions), manifest)
    except (OSError, EvaluationInputError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    predictions = loaded.predictions
    missing = sorted(set(manifest).difference(predictions))
    if loaded.unmatched:
        level = "warning" if args.allow_partial else "error"
        print(
            f"{level}: {len(loaded.unmatched)} unmatched/extra prediction(s): "
            f"{_preview(loaded.unmatched)}",
            file=sys.stderr,
        )
    if missing:
        level = "warning" if args.allow_partial else "error"
        print(
            f"{level}: missing predictions for {len(missing)} of {len(manifest)} "
            f"expected sample(s): {_preview(missing)}",
            file=sys.stderr,
        )
    if not args.allow_partial and (loaded.unmatched or missing):
        print(
            "prediction coverage must exactly match the selected manifest; "
            "use --allow-partial only for debugging",
            file=sys.stderr,
        )
        return 2
    if not predictions:
        print("no predictions matched the manifest", file=sys.stderr)
        return 2

    report_text, overall, groups = report(manifest, predictions)
    print(report_text)
    if args.json_out:
        payload = {
            "input": {
                "expected": len(manifest),
                "matched": len(predictions),
                "missing": len(missing),
                "unmatched": len(loaded.unmatched),
                "allow_partial": args.allow_partial,
            },
            "overall": _metric_payload(overall),
            "by": {
                field: {key: _metric_payload(acc) for key, acc in table.items()}
                for field, table in groups.items()
            },
        }
        try:
            Path(args.json_out).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            print(f"failed to write JSON report: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
