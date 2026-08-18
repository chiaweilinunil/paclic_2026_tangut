from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path

TANGUT_IDEOGRAPHS = (0x17000, 0x187FF)
TANGUT_COMPONENTS = (0x18800, 0x18AFF)
TANGUT_SUPPLEMENT = (0x18D00, 0x18D1E)
TANGUT_COMPONENTS_SUPPLEMENT = (0x18D80, 0x18DF2)
TANGUT_ITERATION_MARK = 0x16FE0

KEPT_PUNCTUATION = "。，、；：？！"

_BRACKETED = re.compile(r"\[[^\]]*\]|（[^）]*）|\([^)]*\)")


def _in(cp: int, rng: tuple[int, int]) -> bool:
    return rng[0] <= cp <= rng[1]


def is_ocr_character(ch: str) -> bool:
    cp = ord(ch)
    return (
        _in(cp, TANGUT_IDEOGRAPHS)
        or _in(cp, TANGUT_SUPPLEMENT)
        or cp == TANGUT_ITERATION_MARK
    )


def count_tangut(text: str) -> int:
    return sum(1 for ch in text if is_ocr_character(ch))


def normalise_line(raw: str, keep_punctuation: bool = True) -> str:
    text = unicodedata.normalize("NFC", raw)
    text = _BRACKETED.sub("", text)
    allowed = KEPT_PUNCTUATION if keep_punctuation else ""
    out = [ch for ch in text if is_ocr_character(ch) or ch in allowed]
    return "".join(out).strip(KEPT_PUNCTUATION)


def chunk_line(text: str, max_chars: int) -> list[str]:
    if count_tangut(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    n = 0
    last_break = -1
    for ch in text:
        current.append(ch)
        if is_ocr_character(ch):
            n += 1
        if ch in KEPT_PUNCTUATION:
            last_break = len(current)
        if n >= max_chars:
            if last_break > 0 and last_break >= len(current) * 0.5:
                head, current = current[:last_break], current[last_break:]
            else:
                head, current = current, []
            chunks.append("".join(head).strip(KEPT_PUNCTUATION))
            n = count_tangut("".join(current))
            last_break = -1
    if current:
        chunks.append("".join(current).strip(KEPT_PUNCTUATION))
    return [c for c in chunks if c]


@dataclass
class TextLine:
    line_id: str
    doc: str
    source_line: int
    text: str
    n_chars: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CorpusStats:
    files: int = 0
    raw_lines: int = 0
    kept_lines: int = 0
    dropped_too_short: int = 0
    dropped_duplicate: int = 0
    chars: int = 0
    per_doc: dict[str, int] = field(default_factory=dict)


def _line_id(doc: str, text: str, idx: int) -> str:
    digest = hashlib.sha1(f"{doc}\x00{text}".encode("utf-8")).hexdigest()[:10]
    return f"{doc}-{idx:05d}-{digest}"


def load_corpus(
    corpus_dir: str | Path,
    min_chars: int = 4,
    max_chars: int = 18,
    keep_punctuation: bool = True,
    deduplicate: bool = True,
    exclude: tuple[str, ...] = ("README.md",),
) -> tuple[list[TextLine], CorpusStats]:
    corpus_dir = Path(corpus_dir)
    files = sorted(p for p in corpus_dir.glob("*.txt") if p.name not in exclude)
    if not files:
        raise FileNotFoundError(f"no .txt files found in {corpus_dir}")

    stats = CorpusStats(files=len(files))
    seen: set[str] = set()
    lines: list[TextLine] = []

    for path in files:
        doc = path.stem
        kept_for_doc = 0
        raw_text = path.read_text(encoding="utf-8")
        for src_no, raw in enumerate(raw_text.splitlines(), start=1):
            stats.raw_lines += 1
            normalised = normalise_line(raw, keep_punctuation=keep_punctuation)
            if not normalised:
                continue
            for chunk in chunk_line(normalised, max_chars):
                n = count_tangut(chunk)
                if n < min_chars:
                    stats.dropped_too_short += 1
                    continue
                if deduplicate:
                    if chunk in seen:
                        stats.dropped_duplicate += 1
                        continue
                    seen.add(chunk)
                lines.append(
                    TextLine(
                        line_id=_line_id(doc, chunk, len(lines)),
                        doc=doc,
                        source_line=src_no,
                        text=chunk,
                        n_chars=n,
                    )
                )
                stats.chars += n
                kept_for_doc += 1
        stats.per_doc[doc] = kept_for_doc

    stats.kept_lines = len(lines)
    return lines, stats


def charset(lines: list[TextLine]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in lines:
        for ch in line.text:
            counts[ch] = counts.get(ch, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
