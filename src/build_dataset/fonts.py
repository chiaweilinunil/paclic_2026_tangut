from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import ImageFont

DEFAULT_FONT_DIRS = [
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
    Path.home() / "AppData/Local/Microsoft/Windows/Fonts",
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".fonts",
    Path.home() / "Library/Fonts",
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
]

_NAME_HINTS = ("tangut", "xixia", "西夏", "n4694")


CACHE_DIR = Path(os.environ.get("TANGUT_FONT_CACHE", Path.home() / ".cache/tangut-ocr/fonts"))

_SMOKE_CHARS = "𗀀𗲻𘟞𗳒𗜍"


@dataclass(frozen=True)
class FontInfo:
    name: str
    path: str
    n_ideographs: int
    n_components: int
    source_path: str = ""
    repaired: bool = False

    @property
    def key(self) -> str:
        return self.name


def _slug(path: Path) -> str:
    stem = path.stem
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in stem]
    return "-".join(filter(None, "".join(keep).split("-"))).lower()


def _renders(path: str) -> bool:
    try:
        font = ImageFont.truetype(path, 48)
        for ch in _SMOKE_CHARS:
            font.getlength(ch)
        return True
    except Exception:
        return False


def _repair(path: Path) -> Path | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{_slug(path)}.dehinted.ttf"
    if out.exists() and _renders(str(out)):
        return out
    try:
        font = TTFont(str(path), fontNumber=0)
        for table in ("fpgm", "prep", "cvt ", "gasp"):
            if table in font:
                del font[table]
        font.save(str(out))
    except Exception:
        return None
    return out if _renders(str(out)) else None


def inspect_font(path: Path, repair: bool = True) -> FontInfo | None:
    try:
        font = TTFont(str(path), fontNumber=0, lazy=True)
        cmap = font.getBestCmap()
    except Exception:
        return None
    ideographs = sum(1 for cp in cmap if 0x17000 <= cp <= 0x187FF or 0x18D00 <= cp <= 0x18D1E)
    components = sum(1 for cp in cmap if 0x18800 <= cp <= 0x18AFF)
    if ideographs < 1000:
        return None

    render_path, repaired = str(path), False
    if not _renders(render_path):
        fixed = _repair(path) if repair else None
        if fixed is None:
            return None
        render_path, repaired = str(fixed), True

    return FontInfo(
        name=_slug(path),
        path=render_path,
        n_ideographs=ideographs,
        n_components=components,
        source_path=str(path),
        repaired=repaired,
    )


def discover_fonts(
    extra_dirs: list[str | Path] | None = None,
    explicit: list[str | Path] | None = None,
    repair: bool = True,
) -> list[FontInfo]:
    if explicit:
        found = [inspect_font(Path(p), repair=repair) for p in explicit]
        missing = [p for p, f in zip(explicit, found) if f is None]
        if missing:
            raise ValueError(f"not usable Tangut fonts: {missing}")
        return _dedupe([f for f in found if f])

    dirs = [Path(d) for d in (extra_dirs or [])] + DEFAULT_FONT_DIRS
    candidates: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            continue
        for pattern in ("*.ttf", "*.otf", "*.ttc"):
            for p in d.rglob(pattern):
                if any(h in p.name.lower() for h in _NAME_HINTS):
                    candidates.append(p)

    fonts = [f for f in (inspect_font(p, repair=repair) for p in sorted(set(candidates))) if f]
    return _dedupe(fonts)


def _dedupe(fonts: list[FontInfo]) -> list[FontInfo]:
    seen: dict[tuple[int, int, int], FontInfo] = {}
    for f in fonts:
        try:
            size = Path(f.source_path or f.path).stat().st_size
        except OSError:
            size = -1
        key = (f.n_ideographs, f.n_components, size)
        if key not in seen or len(f.path) < len(seen[key].path):
            seen[key] = f
    return sorted(seen.values(), key=lambda f: f.name)


def uncovered_chars(font: FontInfo, chars) -> list[str]:
    cmap = TTFont(font.path, fontNumber=0, lazy=True).getBestCmap()
    return sorted(ch for ch in set(chars) if ord(ch) not in cmap)


def renderable_codepoints(font: FontInfo) -> set[int]:
    cmap = TTFont(font.path, fontNumber=0, lazy=True).getBestCmap()
    return {
        cp for cp in cmap
        if 0x17000 <= cp <= 0x187FF or 0x18800 <= cp <= 0x18AFF or 0x18D00 <= cp <= 0x18D1E
    }
