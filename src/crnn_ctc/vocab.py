from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


@dataclass(frozen=True)
class CTCVocabulary:
    chars: tuple[str, ...]
    blank_index: int = 0
    unk_index: int = 1
    unk_symbol: str = "\ufffd"

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "CTCVocabulary":
        return cls(tuple(sorted(set("".join(texts)), key=ord)))

    @property
    def size(self) -> int:
        return len(self.chars) + 2

    @property
    def char_to_index(self) -> dict[str, int]:
        return {char: index + 2 for index, char in enumerate(self.chars)}

    def encode(self, text: str) -> torch.Tensor:
        mapping = self.char_to_index
        return torch.tensor(
            [mapping.get(char, self.unk_index) for char in text], dtype=torch.long
        )

    def decode_indices(self, indices: Iterable[int], collapse_ctc: bool = True) -> str:
        index_to_char = {index + 2: char for index, char in enumerate(self.chars)}
        output: list[str] = []
        previous: int | None = None
        for raw_index in indices:
            index = int(raw_index)
            if collapse_ctc and index == previous:
                continue
            previous = index
            if index == self.blank_index:
                continue
            output.append(
                self.unk_symbol
                if index == self.unk_index
                else index_to_char.get(index, self.unk_symbol)
            )
        return "".join(output)

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "unit": "unicode_scalar_value",
            "blank_index": self.blank_index,
            "unk_index": self.unk_index,
            "unk_symbol": self.unk_symbol,
            "chars": list(self.chars),
            "codepoints": [f"U+{ord(char):04X}" for char in self.chars],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CTCVocabulary":
        if payload.get("blank_index", 0) != 0 or payload.get("unk_index", 1) != 1:
            raise ValueError("unsupported vocabulary indices; expected blank=0 and unk=1")
        return cls(tuple(payload["chars"]), unk_symbol=payload.get("unk_symbol", "\ufffd"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CTCVocabulary":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
