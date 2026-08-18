from __future__ import annotations

import random
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from PIL import Image

Box = tuple[int, int, int, int]

TIERS = ("clean", "light", "medium", "heavy")


@dataclass
class DegradationParams:
    """Intensity ranges for one tier; each value is sampled per image."""

    rotate_deg: float = 0.0
    blur_sigma: tuple[float, float] = (0.0, 0.0)
    gaussian_noise: tuple[float, float] = (0.0, 0.0)
    ink_weight: tuple[float, float] = (0.0, 0.0)
    paper_texture: float = 0.0
    show_through: float = 0.0
    speckle: float = 0.0
    contrast: tuple[float, float] = (1.0, 1.0)
    brightness: tuple[float, float] = (0.0, 0.0)
    jpeg_quality: tuple[int, int] = (100, 100)
    downscale: tuple[float, float] = (1.0, 1.0)

    def to_dict(self) -> dict:
        return asdict(self)


# Heuristic severity levels, not calibrated against any scan-quality scale.
TIER_PARAMS: dict[str, DegradationParams] = {
    "clean": DegradationParams(),
    "light": DegradationParams(
        rotate_deg=0.8,
        blur_sigma=(0.3, 0.7),
        gaussian_noise=(2.0, 6.0),
        ink_weight=(-0.3, 0.3),
        paper_texture=6.0,
        contrast=(0.92, 1.05),
        brightness=(-8.0, 8.0),
        jpeg_quality=(80, 95),
        downscale=(1.0, 1.15),
    ),
    "medium": DegradationParams(
        rotate_deg=2.0,
        blur_sigma=(0.6, 1.3),
        gaussian_noise=(5.0, 13.0),
        ink_weight=(-0.8, 0.8),
        paper_texture=14.0,
        show_through=0.0,
        speckle=0.002,
        contrast=(0.80, 1.10),
        brightness=(-18.0, 14.0),
        jpeg_quality=(55, 80),
        downscale=(1.1, 1.5),
    ),
    "heavy": DegradationParams(
        rotate_deg=4.0,
        blur_sigma=(1.1, 2.2),
        gaussian_noise=(12.0, 26.0),
        ink_weight=(-1.6, 1.4),
        paper_texture=26.0,
        show_through=0.0,
        speckle=0.008,
        contrast=(0.62, 1.15),
        brightness=(-32.0, 20.0),
        jpeg_quality=(30, 60),
        downscale=(1.4, 2.2),
    ),
}


def _u(rng: random.Random, span: tuple[float, float]) -> float:
    lo, hi = span
    return lo if lo == hi else rng.uniform(lo, hi)


def _paper(shape: tuple[int, int], amplitude: float, rng: random.Random) -> np.ndarray:
    h, w = shape
    seed = np.random.default_rng(rng.getrandbits(32))
    small = seed.normal(0.0, 1.0, (max(2, h // 16), max(2, w // 16))).astype(np.float32)
    field = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return field * amplitude


def _show_through(img: np.ndarray, opacity: float, rng: random.Random) -> np.ndarray:
    ghost = cv2.flip(img, 1)
    ghost = cv2.GaussianBlur(ghost, (0, 0), 1.6)
    shift_x, shift_y = rng.randint(-3, 3), rng.randint(-3, 3)
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    ghost = cv2.warpAffine(
        ghost, matrix, (img.shape[1], img.shape[0]), borderValue=255.0
    )
    return img * (1 - opacity) + np.minimum(img, ghost) * opacity


def _ink_weight(img: np.ndarray, weight: float) -> np.ndarray:
    if abs(weight) < 1e-3:
        return img
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    op = cv2.erode if weight < 0 else cv2.dilate
    strength = min(abs(weight), 3.0)
    iterations = max(1, int(np.ceil(strength)))
    morphed = op(img, kernel, iterations=iterations)
    alpha = strength / iterations
    return img * (1 - alpha) + morphed * alpha


def _rotate(img: np.ndarray, degrees: float) -> np.ndarray:
    if abs(degrees) < 1e-3:
        return img
    h, w = img.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(
        img, matrix, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255.0,
    )


def _rotate_boxes(
    boxes: list[Box], degrees: float, width: int, height: int
) -> list[Box]:
    if abs(degrees) < 1e-3:
        return list(boxes)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    transformed: list[Box] = []
    for x0, y0, x1, y1 in boxes:
        corners = np.asarray(
            [[x0, y0, 1.0], [x1, y0, 1.0], [x1, y1, 1.0], [x0, y1, 1.0]],
            dtype=np.float64,
        )
        points = corners @ matrix.T
        left = max(0, min(width, int(np.floor(points[:, 0].min()))))
        top = max(0, min(height, int(np.floor(points[:, 1].min()))))
        right = max(0, min(width, int(np.ceil(points[:, 0].max()))))
        bottom = max(0, min(height, int(np.ceil(points[:, 1].max()))))
        transformed.append((left, top, right, bottom))
    return transformed


def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    if quality >= 100:
        return img
    ok, buf = cv2.imencode(".jpg", img.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32)


def degrade_with_boxes(
    image: Image.Image, boxes: list[Box], tier: str, rng: random.Random
) -> tuple[Image.Image, list[Box]]:
    return _degrade(image, boxes, tier, rng)


def _degrade(
    image: Image.Image, boxes: list[Box], tier: str, rng: random.Random
) -> tuple[Image.Image, list[Box]]:
    if tier == "clean":
        return image, list(boxes)
    if tier not in TIER_PARAMS:
        raise ValueError(f"unknown degradation tier {tier!r}; expected one of {TIERS}")

    params = TIER_PARAMS[tier]
    img = np.asarray(image.convert("L"), dtype=np.float32)
    h, w = img.shape

    img = _ink_weight(img, _u(rng, params.ink_weight))

    if params.show_through:
        img = _show_through(img, params.show_through, rng)

    if params.paper_texture:
        img = img + _paper((h, w), params.paper_texture, rng)

    if params.rotate_deg:
        angle = rng.uniform(-params.rotate_deg, params.rotate_deg)
        img = _rotate(img, angle)
        boxes = _rotate_boxes(boxes, angle, w, h)

    factor = _u(rng, params.downscale)
    if factor > 1.01:
        small = cv2.resize(
            img, (max(8, int(w / factor)), max(8, int(h / factor))), interpolation=cv2.INTER_AREA
        )
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    sigma = _u(rng, params.blur_sigma)
    if sigma > 0.01:
        img = cv2.GaussianBlur(img, (0, 0), sigma)

    contrast, brightness = _u(rng, params.contrast), _u(rng, params.brightness)
    img = (img - 128.0) * contrast + 128.0 + brightness

    noise_sigma = _u(rng, params.gaussian_noise)
    if noise_sigma > 0.01:
        gen = np.random.default_rng(rng.getrandbits(32))
        img = img + gen.normal(0.0, noise_sigma, img.shape)

    if params.speckle:
        gen = np.random.default_rng(rng.getrandbits(32))
        mask = gen.random(img.shape)
        img = np.where(mask < params.speckle / 2, 0.0, img)
        img = np.where(mask > 1 - params.speckle / 2, 255.0, img)

    img = np.clip(img, 0, 255)
    img = _jpeg(img, int(round(_u(rng, params.jpeg_quality))))

    result = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    return result, boxes
