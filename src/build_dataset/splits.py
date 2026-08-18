from __future__ import annotations

import random
from collections import defaultdict

from .corpus import TextLine
from .fonts import FontInfo

SPLITS = ("train", "val", "test")

NEAR_DUPLICATE_FONT_GROUPS = (("tangut-n4694-v3-10", "tangut_n4694_v17"),)


def split_lines(
    lines: list[TextLine],
    mode: str = "random",
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 20260805,
) -> dict[str, list[TextLine]]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {ratios}")
    if mode == "random":
        return _split_stratified(lines, ratios, seed)
    if mode == "document":
        return _split_by_document(lines, ratios, seed)
    raise ValueError(f"unknown split mode {mode!r}; expected 'random' or 'document'")


def _split_stratified(
    lines: list[TextLine], ratios: tuple[float, float, float], seed: int
) -> dict[str, list[TextLine]]:
    by_doc: dict[str, dict[int, list[TextLine]]] = defaultdict(
        lambda: defaultdict(list)
    )
    # Group on (doc, source_line), never on the line itself: one source line
    # may be chunked into several, and splitting those apart puts near-identical
    # text on both sides of the split.
    for line in lines:
        by_doc[line.doc][line.source_line].append(line)

    out: dict[str, list[TextLine]] = {s: [] for s in SPLITS}
    rng = random.Random(seed)
    for doc in sorted(by_doc):
        groups = list(by_doc[doc].values())
        rng.shuffle(groups)
        n = len(groups)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        if n >= 3:
            n_train = min(n_train, n - 2)
            n_val = min(max(n_val, 1), n - n_train - 1)
        assignments = {
            "train": groups[:n_train],
            "val": groups[n_train : n_train + n_val],
            "test": groups[n_train + n_val :],
        }
        for split, split_groups in assignments.items():
            for source_group in split_groups:
                out[split].extend(source_group)
    return out


def _split_by_document(
    lines: list[TextLine], ratios: tuple[float, float, float], seed: int
) -> dict[str, list[TextLine]]:
    by_doc: dict[str, list[TextLine]] = defaultdict(list)
    for line in lines:
        by_doc[line.doc].append(line)

    total = len(lines)
    targets = dict(zip(SPLITS, (r * total for r in ratios)))
    out: dict[str, list[TextLine]] = {s: [] for s in SPLITS}

    docs = sorted(by_doc, key=lambda d: (-len(by_doc[d]), d))
    rng = random.Random(seed)
    rng.shuffle(docs := docs[:])
    docs.sort(key=lambda d: -len(by_doc[d]))

    for doc in docs:
        deficits = {s: targets[s] - len(out[s]) for s in SPLITS}
        chosen = max(SPLITS, key=lambda s: deficits[s])
        out[chosen] += by_doc[doc]
    return out


def assign_fonts(
    fonts: list[FontInfo], holdout: list[str] | None = None
) -> dict[str, list[FontInfo]]:
    holdout = holdout or []
    unknown = set(holdout) - {f.name for f in fonts}
    if unknown:
        raise ValueError(f"holdout fonts not found: {sorted(unknown)}")

    if not holdout:
        return {s: list(fonts) for s in SPLITS}

    seen = [f for f in fonts if f.name not in holdout]
    if not seen:
        raise ValueError("cannot hold out every font; training would have none")
    held = [f for f in fonts if f.name in holdout]
    return {"train": seen, "val": seen, "test": held}


def check_font_holdout(holdout: list[str] | None) -> list[str]:
    if not holdout:
        return []
    warnings = []
    for group in NEAR_DUPLICATE_FONT_GROUPS:
        held = [n for n in group if n in holdout]
        kept = [n for n in group if n not in holdout]
        if held and kept:
            warnings.append(
                f"fonts {held} and {kept} are revisions of the same design; "
                "holding one out does not test typeface generalisation"
            )
    return warnings
