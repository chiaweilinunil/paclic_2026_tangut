#!/usr/bin/env python
"""Extract character-level confusions from scored predictions.

    python src/error_analysis.py --predictions eval/crnn_ctc/test_predictions.jsonl

Writes <out-dir>/<name>_confusions.tsv and <out-dir>/<name>.json, defaulting to
error_analysis/ with <name> taken from the predictions directory. The TSV holds
the top pairs; the JSON holds every pair plus example crops, so a frequent
confusion can be inspected against the images that produced it.

Substitutions are Tangut-to-Tangut by default. Counts here are lower than
evaluate.py's totals by exactly the substitutions involving a non-Tangut
character; pass --include-non-tangut to fold those in.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

# Importable when run directly as a script, not only as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset" / "synthetic"
DEFAULT_OUT_DIR = ROOT / "error_analysis"


@dataclass(frozen=True)
class EditOperation:
    kind: str
    reference: str | None
    prediction: str | None
    reference_index: int
    prediction_index: int


def align_edits(reference: str, prediction: str) -> list[EditOperation]:
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

    operations: list[EditOperation] = []
    i, j = len(reference), len(prediction)
    while i or j:
        if (
            i
            and j
            and reference[i - 1] == prediction[j - 1]
            and distance[i][j] == distance[i - 1][j - 1]
        ):
            operations.append(EditOperation("match", reference[i - 1], prediction[j - 1], i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and j and distance[i][j] == distance[i - 1][j - 1] + 1:
            operations.append(
                EditOperation("substitution", reference[i - 1], prediction[j - 1], i - 1, j - 1)
            )
            i -= 1
            j -= 1
        elif i and distance[i][j] == distance[i - 1][j] + 1:
            operations.append(EditOperation("deletion", reference[i - 1], None, i - 1, j))
            i -= 1
        else:
            if not j or distance[i][j] != distance[i][j - 1] + 1:
                raise RuntimeError("internal error while backtracking edit alignment")
            operations.append(EditOperation("insertion", None, prediction[j - 1], i, j - 1))
            j -= 1
    operations.reverse()
    return operations


def is_tangut_character(character: str) -> bool:
    if len(character) != 1:
        return False
    codepoint = ord(character)
    return (
        codepoint == 0x16FE0
        or 0x17000 <= codepoint <= 0x187FF
        or 0x18D00 <= codepoint <= 0x18D1E
    )


def _codepoint(character: str | None) -> str | None:
    return None if character is None else f"U+{ord(character):04X}"


def analyse(
    manifest: dict[str, dict[str, str]],
    predictions: dict[str, str],
    *,
    tangut_only: bool = True,
    examples_per_pair: int = 5,
) -> dict:
    substitutions: Counter[tuple[str, str]] = Counter()
    deletions: Counter[str] = Counter()
    insertions: Counter[str] = Counter()
    examples: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    totals: Counter[str] = Counter()

    for sample_id, prediction in predictions.items():
        row = manifest[sample_id]
        for operation in align_edits(row["text"], prediction):
            totals[operation.kind] += 1
            if operation.kind == "substitution":
                assert operation.reference is not None and operation.prediction is not None
                if tangut_only and not (
                    is_tangut_character(operation.reference)
                    and is_tangut_character(operation.prediction)
                ):
                    continue
                pair = (operation.reference, operation.prediction)
                substitutions[pair] += 1
                if len(examples[pair]) < examples_per_pair:
                    examples[pair].append(
                        {
                            "sample_id": sample_id,
                            "file_name": row.get("file_name", ""),
                            "layout": row.get("layout", ""),
                            "font": row.get("font", ""),
                            "tier": row.get("tier", ""),
                        }
                    )
            elif operation.kind == "deletion":
                assert operation.reference is not None
                if not tangut_only or is_tangut_character(operation.reference):
                    deletions[operation.reference] += 1
            elif operation.kind == "insertion":
                assert operation.prediction is not None
                if not tangut_only or is_tangut_character(operation.prediction):
                    insertions[operation.prediction] += 1

    substitution_total = sum(substitutions.values())
    pair_rows = []
    for (gold, predicted), count in sorted(
        substitutions.items(), key=lambda item: (-item[1], ord(item[0][0]), ord(item[0][1]))
    ):
        pair_rows.append(
            {
                "gold": gold,
                "gold_codepoint": _codepoint(gold),
                "predicted": predicted,
                "predicted_codepoint": _codepoint(predicted),
                "count": count,
                "share_of_substitutions": count / substitution_total if substitution_total else 0.0,
                "examples": examples[(gold, predicted)],
            }
        )
    return {
        "scope": "tangut_only" if tangut_only else "all_unicode_characters",
        "samples": len(predictions),
        "alignment_totals": dict(sorted(totals.items())),
        "reported_substitutions": substitution_total,
        "substitutions": pair_rows,
        "deletions": [
            {"character": char, "codepoint": _codepoint(char), "count": count}
            for char, count in deletions.most_common()
        ],
        "insertions": [
            {"character": char, "codepoint": _codepoint(char), "count": count}
            for char, count in insertions.most_common()
        ],
    }


def _write_table(
    path: Path, rows: list[dict], top: int | None, delimiter: str = ","
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = rows if top is None else rows[:top]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter=delimiter,
            fieldnames=(
                "gold",
                "gold_codepoint",
                "predicted",
                "predicted_codepoint",
                "count",
                "share_of_substitutions",
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(selected)


def _print_table(rows: list[dict], top: int) -> None:
    print("Gold\tPredicted\tCount\tShare")
    for row in rows[:top]:
        print(
            f"{row['gold']} ({row['gold_codepoint']})\t"
            f"{row['predicted']} ({row['predicted_codepoint']})\t"
            f"{row['count']}\t{row['share_of_substitutions']:.2%}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="destination directory for both outputs")
    parser.add_argument("--name", default=None,
                        help="output file stem; default: the predictions directory name")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--examples-per-pair", type=int, default=5)
    parser.add_argument(
        "--include-non-tangut",
        action="store_true",
        help="include punctuation and any other Unicode substitutions",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="debug only: analyse matched rows when prediction coverage is incomplete",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.top <= 0:
        parser.error("--top must be positive")
    if args.examples_per_pair < 0:
        parser.error("--examples-per-pair must be non-negative")
    try:
        manifest = evaluate.load_manifest(Path(args.dataset), args.split)
        loaded = evaluate.read_predictions(Path(args.predictions), manifest)
    except (OSError, evaluate.EvaluationInputError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    missing = sorted(set(manifest).difference(loaded.predictions))
    if not args.allow_partial and (missing or loaded.unmatched):
        details = []
        if missing:
            details.append(f"missing predictions for {len(missing)} of {len(manifest)} samples")
        if loaded.unmatched:
            details.append(f"{len(loaded.unmatched)} unmatched/extra prediction(s)")
        print("error: prediction coverage must exactly match the selected manifest: " + "; ".join(details), file=sys.stderr)
        return 2
    if args.allow_partial:
        if missing:
            print(f"warning: missing predictions for {len(missing)} sample(s)", file=sys.stderr)
        if loaded.unmatched:
            print(f"warning: {len(loaded.unmatched)} unmatched/extra prediction(s)", file=sys.stderr)

    report = analyse(
        manifest,
        loaded.predictions,
        tangut_only=not args.include_non_tangut,
        examples_per_pair=args.examples_per_pair,
    )
    name = args.name or Path(args.predictions).parent.name or "confusions"
    out_dir = Path(args.out_dir)
    tsv_path = out_dir / f"{name}_confusions.tsv"
    json_path = out_dir / f"{name}.json"
    _write_table(tsv_path, report["substitutions"], args.top, "\t")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _print_table(report["substitutions"], args.top)
    print(
        json.dumps(
            {
                "samples": report["samples"],
                "reported_substitutions": report["reported_substitutions"],
                "tsv_out": str(tsv_path),
                "json_out": str(json_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
