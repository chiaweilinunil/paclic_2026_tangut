from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("Tangut Iteration Mark", 0x16FE0, 0x16FE0),
    ("Tangut", 0x17000, 0x187FF),
    ("Tangut Components", 0x18800, 0x18AFF),
    ("Tangut Supplement", 0x18D00, 0x18D1E),
    ("Tangut Components Supplement", 0x18D80, 0x18DF2),
)

OCR_RANGE_NAMES = frozenset(
    {"Tangut Iteration Mark", "Tangut", "Tangut Supplement"}
)
COMPONENT_RANGE_NAMES = frozenset(
    {"Tangut Components", "Tangut Components Supplement"}
)

DEFAULT_SOURCES_FILE = Path(__file__).resolve().parents[2] / "data" / "TangutSources.txt"

_LFW_PREFIX = "L2008-"


@dataclass(frozen=True)
class CharInfo:
    char: str
    codepoint: int
    block: str
    lfw: str = ""
    source: str = ""
    radical_stroke: str = ""

    @property
    def codepoint_str(self) -> str:
        return f"U+{self.codepoint:04X}"


def block_of(codepoint: int) -> str | None:
    for name, first, last in BLOCKS:
        if first <= codepoint <= last:
            return name
    return None


def all_codepoints() -> list[int]:
    return [cp for _, first, last in BLOCKS for cp in range(first, last + 1)]


def _parse_sources(path: Path) -> dict[int, dict[str, str]]:
    data: dict[int, dict[str, str]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 3:
            continue
        cp_str, tag, value = parts
        try:
            cp = int(cp_str[2:], 16)
        except ValueError:
            continue
        data.setdefault(cp, {})[tag] = value
    return data


def _lfw_from_source(value: str) -> str:
    if not value.startswith(_LFW_PREFIX):
        return ""
    parts = []
    for token in value[len(_LFW_PREFIX):].split("-"):
        digits = re.match(r"^(\d+)([A-Za-z]*)$", token)
        if digits:
            parts.append(f"{int(digits.group(1))}{digits.group(2)}")
        elif token:
            parts.append(token)
    return "+".join(parts)


def load_inventory(sources_file: str | Path | None = None) -> dict[int, CharInfo]:
    path = Path(sources_file) if sources_file else DEFAULT_SOURCES_FILE
    if not path.is_file():
        raise FileNotFoundError(f"TangutSources file not found: {path}")
    sources = _parse_sources(path)

    inventory: dict[int, CharInfo] = {}
    for cp in all_codepoints():
        tags = sources.get(cp, {})
        merged = tags.get("kTGT_MergedSrc", "")
        inventory[cp] = CharInfo(
            char=chr(cp),
            codepoint=cp,
            block=block_of(cp) or "",
            lfw=_lfw_from_source(merged),
            source=merged,
            radical_stroke=tags.get("kTGT_RSUnicode", ""),
        )
    return inventory


def is_ideograph(codepoint: int) -> bool:
    return block_of(codepoint) in ("Tangut", "Tangut Supplement")


def is_iteration_mark(codepoint: int) -> bool:
    return codepoint == 0x16FE0


def is_ocr_character(codepoint: int) -> bool:
    return block_of(codepoint) in OCR_RANGE_NAMES


def is_component(codepoint: int) -> bool:
    return block_of(codepoint) in COMPONENT_RANGE_NAMES


def summarise(inventory: dict[int, CharInfo]) -> dict[str, int]:
    per_block = {name: 0 for name, _, _ in BLOCKS}
    for info in inventory.values():
        if info.block in per_block:
            per_block[info.block] += 1

    ideographs = [c for cp, c in inventory.items() if is_ideograph(cp)]
    ocr_characters = [c for cp, c in inventory.items() if is_ocr_character(cp)]
    components = [c for cp, c in inventory.items() if is_component(cp)]
    with_lfw = sum(1 for c in ideographs if c.lfw)
    return {
        "total": len(inventory),
        "ideographs": len(ideographs),
        "iteration_marks": sum(1 for cp in inventory if is_iteration_mark(cp)),
        "ocr_characters": len(ocr_characters),
        "components": len(components),
        "with_lfw": with_lfw,
        "ideographs_without_lfw": len(ideographs) - with_lfw,
        **{f"block:{k}": v for k, v in per_block.items()},
    }
