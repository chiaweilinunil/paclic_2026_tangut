from __future__ import annotations

from dataclasses import dataclass

LFW_UNKNOWN = "[?]"

@dataclass
class LabelEncoder:
    char_to_lfw: dict[str, str]

    @classmethod
    def from_inventory(cls, inventory) -> "LabelEncoder":
        return cls(char_to_lfw={i.char: i.lfw for i in inventory.values() if i.lfw})

    def lfw(self, text: str) -> str:
        out = []
        for ch in text:
            num = self.char_to_lfw.get(ch)
            if num is not None:
                out.append(num)
            elif ch.strip():
                out.append(ch if not _is_tangut(ch) else LFW_UNKNOWN)
        return " ".join(out)

    def coverage(self, chars) -> tuple[int, list[str]]:
        ideographs = {c for c in chars if _is_tangut_ideograph(c)}
        missing = sorted(c for c in ideographs if c not in self.char_to_lfw)
        covered = len(ideographs) - len(missing)
        return covered, missing

def codepoints(text: str) -> str:
    return " ".join(f"U+{ord(ch):04X}" for ch in text)

def _is_tangut(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x17000 <= cp <= 0x187FF
        or 0x18800 <= cp <= 0x18AFF
        or 0x18D00 <= cp <= 0x18D1E
        or 0x18D80 <= cp <= 0x18DF2
        or cp == 0x16FE0
    )

def _is_tangut_ideograph(ch: str) -> bool:
    cp = ord(ch)
    return 0x17000 <= cp <= 0x187FF or 0x18D00 <= cp <= 0x18D1E
