from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict

from PIL import Image, ImageDraw, ImageFont

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
LAYOUTS = (HORIZONTAL, VERTICAL)


@dataclass
class RenderSpec:
    layout: str = HORIZONTAL
    line_px: int = 64
    margin_ratio: float = 0.12
    cell_ratio: float = 1.0
    jitter_ratio: float = 0.0
    size_jitter_ratio: float = 0.0
    ink: int = 0
    paper: int = 255
    supersample: int = 3
    pad_to: tuple[int, int] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RenderedLine:
    image: Image.Image
    char_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size


_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]


def _fit_font(path: str, glyph_px: int) -> tuple[ImageFont.FreeTypeFont, int, int]:
    probe = _font(path, glyph_px)
    ascent, descent = probe.getmetrics()
    extent = ascent + descent
    if extent <= 0:
        return probe, ascent, descent
    size = max(4, round(glyph_px * glyph_px / extent))
    fitted = _font(path, size)
    ascent, descent = fitted.getmetrics()
    return fitted, ascent, descent


def render_line(
    text: str,
    font_path: str,
    spec: RenderSpec,
    rng: random.Random | None = None,
) -> RenderedLine:
    if spec.layout not in LAYOUTS:
        raise ValueError(f"unknown layout {spec.layout!r}")
    if not text:
        raise ValueError("cannot render an empty line")
    rng = rng or random.Random(0)

    ss = max(1, spec.supersample)
    line_px = spec.line_px * ss
    margin = round(line_px * spec.margin_ratio)
    glyph_px = line_px - 2 * margin
    cell = max(1, round(glyph_px * spec.cell_ratio))
    jitter = glyph_px * spec.jitter_ratio

    font, ascent, descent = _fit_font(font_path, glyph_px)

    n = len(text)
    if spec.layout == HORIZONTAL:
        width, height = 2 * margin + n * cell, line_px
    else:
        width, height = line_px, 2 * margin + n * cell

    image = Image.new("L", (width, height), spec.paper)
    draw = ImageDraw.Draw(image)
    boxes: list[tuple[int, int, int, int]] = []

    band_offset = margin + (glyph_px - (ascent + descent)) / 2

    for i, ch in enumerate(text):
        char_font = font
        if spec.size_jitter_ratio:
            scale = 1.0 + rng.uniform(-spec.size_jitter_ratio, spec.size_jitter_ratio)
            char_font = _font(font_path, max(4, round(font.size * scale)))
        advance = char_font.getlength(ch)

        dx = rng.uniform(-jitter, jitter) if jitter else 0.0
        dy = rng.uniform(-jitter, jitter) if jitter else 0.0

        if spec.layout == HORIZONTAL:
            x = margin + i * cell + (cell - advance) / 2 + dx
            baseline = band_offset + ascent + dy
        else:
            x = margin + (cell - advance) / 2 + dx
            baseline = margin + i * cell + (cell - (ascent + descent)) / 2 + ascent + dy

        draw.text((x, baseline), ch, font=char_font, fill=spec.ink, anchor="ls")

        left, top, right, bottom = char_font.getbbox(ch, anchor="ls")
        boxes.append((round(x + left), round(baseline + top),
                      round(x + right), round(baseline + bottom)))

    if ss > 1:
        image = image.resize((width // ss, height // ss), Image.Resampling.LANCZOS)
        boxes = [tuple(round(v / ss) for v in box) for box in boxes]

    rendered = RenderedLine(image=image, char_boxes=boxes)
    if spec.pad_to:
        rendered = letterbox(rendered, spec.pad_to, spec.paper)
    return rendered


def letterbox(
    rendered: RenderedLine, target: tuple[int, int], paper: int = 255
) -> RenderedLine:
    tw, th = target
    w, h = rendered.image.size
    scale = min(tw / w, th / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = rendered.image.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new(rendered.image.mode, (tw, th), paper)
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    canvas.paste(resized, (ox, oy))
    boxes = [
        (round(x0 * scale) + ox, round(y0 * scale) + oy,
         round(x1 * scale) + ox, round(y1 * scale) + oy)
        for x0, y0, x1, y1 in rendered.char_boxes
    ]
    return RenderedLine(image=canvas, char_boxes=boxes)
