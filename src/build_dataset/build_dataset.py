#!/usr/bin/env python
"""Build the synthetic Tangut line-image OCR dataset.

    python src/build_dataset/build_dataset.py

Reads one *.txt transcription per work from text/ and writes an
imagefolder-compatible dataset to dataset/synthetic/, together with
build_config.json, which records every parameter and input digest needed to
reproduce the build.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Importable when run directly as a script, not only as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from build_dataset.corpus import TextLine, charset, load_corpus
from build_dataset.degrade import TIERS, degrade_with_boxes
from build_dataset.fonts import FontInfo, discover_fonts, renderable_codepoints, uncovered_chars
from build_dataset.inventory import (
    BLOCKS,
    DEFAULT_SOURCES_FILE,
    is_ocr_character,
    load_inventory,
    summarise,
)
from build_dataset.labels import LabelEncoder, codepoints
from build_dataset.render import LAYOUTS, RenderSpec, render_line
from build_dataset.splits import SPLITS, assign_fonts, check_font_holdout, split_lines

PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parents[1]

DEFAULT_CORPUS = ROOT / "text"
DEFAULT_OUT = ROOT / "dataset" / "synthetic"

DEFAULT_TIER_WEIGHTS = {"clean": 0.40, "light": 0.25, "medium": 0.20, "heavy": 0.15}

GENERATED_DIRS = ("images", "preview")
GENERATED_FILES = (
    "manifest.csv",
    "lines.jsonl",
    "charset.tsv",
    "build_config.json",
    "dataset_card.md",
)


@dataclass
class Sample:
    sample_id: str
    split: str
    line_id: str
    doc: str
    source_line: int
    text: str
    lfw: str
    codepoints: str
    n_chars: int
    layout: str
    font: str
    tier: str
    file_name: str
    width: int
    height: int
    char_boxes: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    io = p.add_argument_group("input / output")
    io.add_argument("--corpus", default=str(DEFAULT_CORPUS),
                    help="directory of *.txt transcriptions; default: text/")
    io.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output directory; default: dataset/synthetic/")
    io.add_argument(
        "--overwrite",
        action="store_true",
        help="replace known generated artifacts in an existing output directory",
    )
    io.add_argument("--tangut-sources", default=None,
                    help="Unicode TangutSources.txt (default: data/TangutSources.txt). "
                         "Refresh from https://www.unicode.org/Public/UCD/latest/ucd/TangutSources.txt")

    text = p.add_argument_group("text extraction")
    text.add_argument("--min-chars", type=int, default=4, help="drop lines shorter than this")
    text.add_argument("--max-chars", type=int, default=18, help="chunk lines longer than this")
    text.add_argument("--no-punctuation", action="store_true",
                      help="strip CJK punctuation from labels and images")
    text.add_argument("--keep-duplicates", action="store_true",
                      help="do not drop repeated lines (risks train/test leakage)")

    render = p.add_argument_group("rendering")
    render.add_argument("--layouts", default="horizontal,vertical",
                        help=f"comma-separated subset of {','.join(LAYOUTS)}")
    render.add_argument("--line-px", type=int, default=64,
                        help="strip height (horizontal) / column width (vertical)")
    render.add_argument("--cell-ratio", type=float, default=1.0, help="advance per character")
    render.add_argument("--jitter", type=float, default=0.02, help="per-character position jitter")
    render.add_argument("--size-jitter", type=float, default=0.02, help="per-character size jitter")
    render.add_argument("--supersample", type=int, default=3, help="antialiasing factor")
    render.add_argument("--pad-to", default=None, metavar="WxH",
                        help="letterbox every image to a fixed size, e.g. 384x384 for TrOCR")

    fonts = p.add_argument_group("fonts")
    fonts.add_argument("--font-dir", action="append", default=[],
                       help="extra directory to scan (repeatable)")
    fonts.add_argument("--font", action="append", default=[],
                       help="use exactly this font file (repeatable; disables discovery)")
    fonts.add_argument("--holdout-fonts", default="",
                       help="comma-separated font names reserved for the test split")
    fonts.add_argument("--fonts-per-line", type=int, default=1,
                       help="how many fonts to sample per train/validation line")
    fonts.add_argument(
        "--test-font-policy",
        choices=("paired", "sampled"),
        default="paired",
        help=(
            "paired renders every eligible test line in every test font "
            "(default); sampled applies --fonts-per-line to test as well"
        ),
    )

    split = p.add_argument_group("splits and degradation")
    split.add_argument("--split-by", choices=("random", "document"), default="random")
    split.add_argument("--ratios", default="0.8,0.1,0.1")
    split.add_argument("--tiers", default=",".join(TIERS),
                       help=f"degradation tiers to build, subset of {','.join(TIERS)}")
    split.add_argument("--tier-weights", default="",
                       help="train/val sampling weights, e.g. clean=0.4,light=0.25,...")
    split.add_argument(
        "--test-tier-policy",
        choices=("balanced", "all"),
        default="balanced",
        help=(
            "balanced assigns one uniformly balanced tier per test line and "
            "reuses it across fonts/layouts (default); all renders every test "
            "line at every selected tier"
        ),
    )
    split.add_argument("--seed", type=int, default=20260805)

    misc = p.add_argument_group("misc")
    misc.add_argument("--limit", type=int, default=0, help="cap the number of lines (smoke test)")
    misc.add_argument("--preview", type=int, default=12, help="preview images per split")
    misc.add_argument("--list-fonts", action="store_true", help="list fonts found, then exit")
    return p.parse_args(argv)


def _parse_pad(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    w, _, h = value.lower().partition("x")
    return int(w), int(h)


def _parse_weights(value: str, tiers: list[str]) -> dict[str, float]:
    weights = dict(DEFAULT_TIER_WEIGHTS)
    if value:
        weights = {}
        for part in value.split(","):
            k, _, v = part.partition("=")
            weights[k.strip()] = float(v)
    weights = {t: weights.get(t, 0.0) for t in tiers}
    total = sum(weights.values())
    if total <= 0:
        return {t: 1.0 / len(tiers) for t in tiers}
    return {t: w / total for t, w in weights.items()}


def _assign_balanced_test_tiers(
    lines: list[TextLine], tiers: list[str], seed: int
) -> dict[str, str]:
    if not tiers:
        raise ValueError("at least one degradation tier is required")
    if len(set(tiers)) != len(tiers):
        raise ValueError(f"degradation tiers must be unique, got {tiers}")

    strata: dict[tuple[str, int], list[TextLine]] = {}
    seen_ids: set[str] = set()
    for line in lines:
        if line.line_id in seen_ids:
            raise ValueError(f"duplicate line_id in test tier assignment: {line.line_id}")
        seen_ids.add(line.line_id)
        strata.setdefault((line.doc, line.n_chars), []).append(line)

    tier_cycle = list(tiers)
    random.Random(f"{seed}:test-tier-cycle").shuffle(tier_cycle)

    assignments: dict[str, str] = {}
    cursor = 0
    for doc, n_chars in sorted(strata):
        group = sorted(strata[(doc, n_chars)], key=lambda line: line.line_id)
        random.Random(f"{seed}:test-tier:{doc}:{n_chars}").shuffle(group)
        for line in group:
            assignments[line.line_id] = tier_cycle[cursor % len(tier_cycle)]
            cursor += 1
    return assignments


def _renderable_fonts(
    fonts: list[FontInfo], all_chars
) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    gaps = {f.name: set(uncovered_chars(f, all_chars)) for f in fonts}
    blocked: dict[str, list[str]] = {}
    for name, chars in gaps.items():
        for ch in chars:
            blocked.setdefault(ch, []).append(name)
    return gaps, blocked


def _prepare_output(out_dir: Path, overwrite: bool) -> None:
    resolved = out_dir.resolve()
    if resolved == PKG_DIR.resolve():
        raise ValueError("refusing to use the package directory itself as --out")

    generated = [resolved / name for name in (*GENERATED_DIRS, *GENERATED_FILES)]
    present = [path for path in generated if path.exists()]
    if present and not overwrite:
        names = ", ".join(path.name for path in present)
        raise FileExistsError(
            f"output already contains generated artifacts ({names}); "
            "choose a new --out directory or pass --overwrite"
        )
    if overwrite:
        for path in generated:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    resolved.mkdir(parents=True, exist_ok=True)


def _relative(path: str | Path) -> str:
    """Path relative to the repository root, so configs stay machine-neutral."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    fonts = discover_fonts(extra_dirs=args.font_dir, explicit=args.font or None)
    if not fonts:
        print("No Tangut fonts found. Install e.g. Noto Serif Tangut, or pass --font.",
              file=sys.stderr)
        return 2
    if args.list_fonts:
        for f in fonts:
            flag = " (de-hinted copy)" if f.repaired else ""
            print(f"{f.name:28s} {f.n_ideographs:5d} ideographs  {f.path}{flag}")
        return 0

    layouts = [layout.strip() for layout in args.layouts.split(",") if layout.strip()]
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    for name, values, allowed in (("layout", layouts, LAYOUTS), ("tier", tiers, TIERS)):
        if not values:
            print(f"at least one {name} is required", file=sys.stderr)
            return 2
        bad = set(values) - set(allowed)
        if bad:
            print(f"unknown {name}(s): {sorted(bad)}", file=sys.stderr)
            return 2
        if len(set(values)) != len(values):
            print(f"duplicate {name}(s) are not allowed: {values}", file=sys.stderr)
            return 2

    holdout = [f.strip() for f in args.holdout_fonts.split(",") if f.strip()]
    for warning in check_font_holdout(holdout):
        print(f"warning: {warning}", file=sys.stderr)

    corpus_dir = Path(args.corpus)
    if not corpus_dir.is_dir() or not any(corpus_dir.glob("*.txt")):
        print(f"no *.txt transcriptions in {corpus_dir}\n"
              "The transcriptions are not distributed with this repository; see "
              "text/README.md. Pass --corpus to read them from elsewhere.",
              file=sys.stderr)
        return 2

    print(f"Reading corpus from {corpus_dir} ...")
    lines, stats = load_corpus(
        corpus_dir,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        keep_punctuation=not args.no_punctuation,
        deduplicate=not args.keep_duplicates,
    )
    if args.limit:
        rng.shuffle(lines)
        lines = lines[: args.limit]
    print(
        f"  {stats.files} documents -> {len(lines)} lines, "
        f"{sum(line.n_chars for line in lines)} characters"
    )

    sources_path = Path(args.tangut_sources) if args.tangut_sources else DEFAULT_SOURCES_FILE
    if not sources_path.is_file():
        print(f"Unicode data not found: {sources_path}\n"
              "Download TangutSources.txt from "
              "https://www.unicode.org/Public/UCD/latest/ucd/TangutSources.txt",
              file=sys.stderr)
        return 2
    inventory = load_inventory(args.tangut_sources)
    counts = charset(lines)
    encoder = LabelEncoder.from_inventory(inventory)
    covered, missing_lfw = encoder.coverage(counts)
    print(f"  {len(counts)} distinct characters; Li Fanwen numbers for {covered} of {covered + len(missing_lfw)}")

    summary = summarise(inventory)
    print(
        f"  Unicode 17 Script=Tangut inventory: {summary['total']} characters "
        f"({summary['ideographs']} ideographs + "
        f"{summary['iteration_marks']} iteration mark + "
        f"{summary['components']} analytic components); "
        f"default OCR repertoire {summary['ocr_characters']}; "
        f"{summary['with_lfw']} ideographs have a Li Fanwen number and "
        f"{summary['ideographs_without_lfw']} do not"
    )
    gaps, blocked = _renderable_fonts(fonts, counts)
    for f in fonts:
        note = f", {len(gaps[f.name])} corpus chars unsupported" if gaps[f.name] else ""
        print(f"  font {f.name:26s} {f.n_ideographs} ideographs{note}")

    ratios = tuple(float(x) for x in args.ratios.split(","))
    parts = split_lines(lines, mode=args.split_by, ratios=ratios, seed=args.seed)
    font_pool = assign_fonts(fonts, holdout)
    weights = _parse_weights(args.tier_weights, tiers)

    test_tier_by_line: dict[str, str] = {}
    if args.test_tier_policy == "balanced":
        test_pool = font_pool["test"]
        eligible_test_lines = []
        for line in parts["test"]:
            usable = [f for f in test_pool if not (gaps[f.name] & set(line.text))]
            complete_pair = len(usable) == len(test_pool)
            if usable and (args.test_font_policy != "paired" or complete_pair):
                eligible_test_lines.append(line)
        test_tier_by_line = _assign_balanced_test_tiers(
            eligible_test_lines, tiers, args.seed
        )

    spec_base = dict(
        line_px=args.line_px,
        cell_ratio=args.cell_ratio,
        jitter_ratio=args.jitter,
        size_jitter_ratio=args.size_jitter,
        supersample=args.supersample,
        pad_to=_parse_pad(args.pad_to),
    )

    out_dir = Path(args.out)
    try:
        _prepare_output(out_dir, args.overwrite)
    except (FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    (out_dir / "preview").mkdir(parents=True, exist_ok=True)

    samples: list[Sample] = []
    skipped_no_font = 0
    skipped_incomplete_font_pair = 0

    for split in SPLITS:
        split_lines_ = parts[split]
        pool = font_pool[split]
        if not pool:
            continue
        for layout in layouts:
            (out_dir / "images" / split / layout).mkdir(parents=True, exist_ok=True)

        desc = f"{split:5s} ({len(split_lines_)} lines)"
        for line in tqdm(split_lines_, desc=desc, unit="line"):
            line_rng = random.Random(f"{args.seed}:{line.line_id}")
            usable = [f for f in pool if not (gaps[f.name] & set(line.text))]
            if not usable:
                skipped_no_font += 1
                continue
            # A paired font comparison needs identical text under every font, so
            # a line no test font can fully draw is dropped rather than rendered
            # in the subset that works, which would confound font with content.
            if split == "test" and args.test_font_policy == "paired":
                if len(usable) != len(pool):
                    skipped_incomplete_font_pair += 1
                    continue
                chosen = usable
            else:
                k = min(args.fonts_per_line, len(usable))
                chosen = line_rng.sample(usable, k)
            lfw = encoder.lfw(line.text)
            cps = codepoints(line.text)

            for font in chosen:
                for layout in layouts:
                    spec = RenderSpec(layout=layout, **spec_base)
                    # Geometry is keyed per (line, font, layout), so tier variants
                    # of one image differ only in degradation.
                    geom_rng = random.Random(f"{args.seed}:{line.line_id}:{font.name}:{layout}")
                    rendered = render_line(line.text, font.path, spec, geom_rng)

                    if split == "test":
                        chosen_tiers = (
                            tiers
                            if args.test_tier_policy == "all"
                            else [test_tier_by_line[line.line_id]]
                        )
                    else:
                        chosen_tiers = [
                            line_rng.choices(tiers, weights=[weights[t] for t in tiers])[0]
                        ]

                    for tier in chosen_tiers:
                        tier_rng = random.Random(
                            f"{args.seed}:{line.line_id}:{font.name}:{layout}:{tier}"
                        )
                        image, char_boxes = degrade_with_boxes(
                            rendered.image, rendered.char_boxes, tier, tier_rng
                        )
                        idx = len(samples)
                        rel = f"{layout}/{idx:07d}_{font.name}_{tier}.png"
                        image.save(out_dir / "images" / split / rel, "PNG", optimize=True)
                        samples.append(
                            Sample(
                                sample_id=f"{split}-{idx:07d}",
                                split=split,
                                line_id=line.line_id,
                                doc=line.doc,
                                source_line=line.source_line,
                                text=line.text,
                                lfw=lfw,
                                codepoints=cps,
                                n_chars=line.n_chars,
                                layout=layout,
                                font=font.name,
                                tier=tier,
                                file_name=rel,
                                width=image.width,
                                height=image.height,
                                char_boxes=json.dumps(char_boxes),
                            )
                        )

    if skipped_no_font:
        print(f"warning: skipped {skipped_no_font} lines that no available font could render",
              file=sys.stderr)
    if skipped_incomplete_font_pair:
        print(
            "warning: skipped "
            f"{skipped_incomplete_font_pair} test lines not supported by every test font "
            "to preserve a paired font comparison",
            file=sys.stderr,
        )

    if not samples:
        print("error: no samples were rendered", file=sys.stderr)
        return 2

    rendered_line_ids = {sample.line_id for sample in samples}
    rendered_lines = [line for line in lines if line.line_id in rendered_line_ids]
    rendered_counts = charset(rendered_lines)
    _, rendered_missing_lfw = encoder.coverage(rendered_counts)

    config = {
        "seed": args.seed,
        "corpus": _relative(corpus_dir),
        "corpus_files_sha256": {
            path.name: _sha256_file(path) for path in sorted(corpus_dir.glob("*.txt"))
        },
        "tangut_sources": {
            "path": _relative(sources_path),
            "sha256": _sha256_file(sources_path),
        },
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "keep_punctuation": not args.no_punctuation,
        "deduplicate": not args.keep_duplicates,
        "layouts": layouts,
        "tiers": tiers,
        "tier_weights": weights,
        "test_tier_policy": args.test_tier_policy,
        "split_by": args.split_by,
        "ratios": list(ratios),
        "holdout_fonts": holdout,
        "fonts_per_line": args.fonts_per_line,
        "test_font_policy": args.test_font_policy,
        "candidate_lines": len(lines),
        "rendered_lines": len(rendered_lines),
        "skipped_no_font": skipped_no_font,
        "skipped_incomplete_font_pair": skipped_incomplete_font_pair,
        "render": spec_base,
        "fonts": {
            s: [
                {
                    "name": f.name,
                    "path": Path(f.path).name,
                    "sha256": _sha256_file(f.path),
                    "repaired": f.repaired,
                }
                for f in font_pool[s]
            ]
            for s in SPLITS
        },
        "code_sha256": {
            path.relative_to(PKG_DIR).as_posix(): _sha256_file(path)
            for path in sorted(PKG_DIR.glob("*.py"))
        },
        "line_limit": args.limit or None,
    }
    write_outputs(
        out_dir,
        samples,
        rendered_lines,
        rendered_counts,
        inventory,
        fonts,
        rendered_missing_lfw,
        config,
        args.preview,
        rng,
    )

    print(f"\nWrote {len(samples)} images to {out_dir.resolve()}")
    for split in SPLITS:
        n = sum(1 for s in samples if s.split == split)
        n_lines = len({s.line_id for s in samples if s.split == split})
        print(f"  {split:5s} {n:7d} images  {n_lines:6d} lines")
    return 0


def write_outputs(out_dir, samples, lines, counts, inventory, fonts, missing_lfw, config,
                  n_preview, rng):
    from PIL import Image

    for split in SPLITS:
        rows = [s for s in samples if s.split == split]
        if not rows:
            continue
        path = out_dir / "images" / split / "metadata.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for s in rows:
                fh.write(json.dumps(
                    {
                        "file_name": s.file_name,
                        "sample_id": s.sample_id,
                        "split": s.split,
                        "text": s.text,
                        "lfw": s.lfw,
                        "codepoints": s.codepoints,
                        "doc": s.doc,
                        "source_line": s.source_line,
                        "layout": s.layout,
                        "font": s.font,
                        "tier": s.tier,
                        "n_chars": s.n_chars,
                        "line_id": s.line_id,
                    },
                    ensure_ascii=False,
                ) + "\n")

    fields = list(Sample.__annotations__)
    with (out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for s in samples:
            writer.writerow(s.__dict__)

    with (out_dir / "lines.jsonl").open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line.to_dict(), ensure_ascii=False) + "\n")

    write_charset(out_dir / "charset.tsv", inventory, counts, fonts)

    (out_dir / "build_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if n_preview > 0:
        for split in SPLITS:
            for layout in sorted({s.layout for s in samples if s.split == split}):
                rows = [s for s in samples if s.split == split and s.layout == layout]
                if not rows:
                    continue
                picks = rng.sample(rows, min(n_preview, len(rows)))
                imgs = [
                    Image.open(out_dir / "images" / split / s.file_name).convert("L")
                    for s in picks
                ]
                _contact_sheet(imgs, stack=(layout == "horizontal")).save(
                    out_dir / "preview" / f"{split}_{layout}.png"
                )

    (out_dir / "dataset_card.md").write_text(
        _dataset_card(samples, lines, counts, inventory, missing_lfw, config), encoding="utf-8"
    )


def write_charset(path, inventory, counts, fonts):
    coverage = {}
    for font in fonts:
        for cp in renderable_codepoints(font):
            coverage[cp] = coverage.get(cp, 0) + 1

    with Path(path).open("w", encoding="utf-8") as fh:
        fh.write(
            "char\tcodepoint\trange\tocr_default\tlfw\tsource\t"
            "radical_stroke\tcorpus_count\tn_fonts\n"
        )
        for cp in sorted(inventory):
            info = inventory[cp]
            fh.write(
                f"{info.char}\t{info.codepoint_str}\t{info.block}\t"
                f"{int(is_ocr_character(cp))}\t{info.lfw}\t"
                f"{info.source}\t{info.radical_stroke}\t{counts.get(info.char, 0)}\t"
                f"{coverage.get(cp, 0)}\n"
            )


def _contact_sheet(imgs, stack: bool, pad: int = 6, background: int = 210):
    from PIL import Image

    if stack:
        width = max(i.width for i in imgs) + 2 * pad
        height = sum(i.height + pad for i in imgs) + pad
    else:
        width = sum(i.width + pad for i in imgs) + pad
        height = max(i.height for i in imgs) + 2 * pad

    sheet = Image.new("L", (width, height), background)
    offset = pad
    for im in imgs:
        sheet.paste(im, (pad, offset) if stack else (offset, pad))
        offset += (im.height if stack else im.width) + pad
    return sheet


def _dataset_card(samples, lines, counts, inventory, missing_lfw, config) -> str:
    by_split = Counter(s.split for s in samples)
    lines_by_split = {s: len({x.line_id for x in samples if x.split == s}) for s in SPLITS}
    by_tier = Counter(s.tier for s in samples)
    by_layout = Counter(s.layout for s in samples)
    by_font = Counter(s.font for s in samples)
    docs = Counter(line.doc for line in lines)
    n_chars = sum(line.n_chars for line in lines)

    def table(counter, header):
        rows = "\n".join(f"| {k} | {v:,} |" for k, v in sorted(counter.items(), key=lambda kv: -kv[1]))
        return f"| {header} | images |\n|---|---:|\n{rows}\n"

    leakage = "yes" if config["deduplicate"] else "NOT CHECKED (--keep-duplicates was set)"
    holdout = ", ".join(config["holdout_fonts"]) or "none (all fonts in all splits)"
    weights = ", ".join(f"{t} {w:.2f}" for t, w in config["tier_weights"].items())
    if config.get("test_tier_policy", "all") == "balanced":
        test_tier_design = (
            "Test lines receive **one** uniformly balanced tier per `line_id`, "
            "reused across all selected fonts and layouts; this preserves paired "
            "font/layout comparisons without duplicating each line across tiers."
        )
    else:
        test_tier_design = (
            "Test lines are rendered at **every** selected tier with font and "
            "geometry held fixed, forming a fully crossed degradation panel."
        )
    inv = summarise(inventory)
    attested = sum(1 for info in inventory.values() if counts.get(info.char, 0))
    block_rows = "\n".join(
        f"| {name} | U+{first:04X}..U+{last:04X} | {inv[f'block:{name}']:,} |"
        for name, first, last in BLOCKS
    )

    return f"""# Synthetic Tangut OCR line dataset

Line images rendered from a corpus of machine-readable Tangut texts, paired with
Unicode ground truth and two derived label serializations. Built with
`build_dataset.py`; every parameter plus input,
font, and code SHA-256 is recorded in `build_config.json`, and the build is
deterministic given `seed = {config["seed"]}`.

## Contents

- **{len(samples):,} images** over **{len(lines):,} unique text lines**
  ({n_chars:,} Tangut characters, {len(counts):,} distinct characters) drawn from
  **{len(docs)} works**.
- Labels per image: `text` (Tangut Unicode, the OCR ground truth), `lfw`
  (Li Fanwen numbers), `codepoints` (`U+17xxx`, ASCII-safe).
- Per-character bounding boxes in `manifest.csv`, usable for a detection or
  segmentation baseline without re-rendering.
- `charset.tsv` lists all **{inv["total"]:,} Unicode 17 `Script=Tangut`**
  characters across five ranges and marks the **{inv["ocr_characters"]:,}**
  ordinary-text characters with `ocr_default = 1`. Analytic components remain
  documented but are not put in the default line-transcription decoder.

| split | images | unique lines |
|---|---:|---:|
""" + "".join(
        f"| {s} | {by_split[s]:,} | {lines_by_split[s]:,} |\n" for s in SPLITS
    ) + f"""
## Design

**Splits.** Partitioned by `{config["split_by"]}` at ratios
{config["ratios"]}. `random` shuffles lines within each document, so train and
test share vocabulary and register. `document` holds out whole works, which is
the harder and more informative setting: 論語 and 華嚴經 share little vocabulary,
so it measures generalisation rather than memorisation of the corpus.

**Held-out fonts.** {holdout}. Test font policy:
`{config.get("test_font_policy", "sampled")}`. Under `paired`, every eligible
test string is rendered in every test font; lines unsupported by any one test
font are excluded, so per-font differences are paired by `line_id`. Note that
`tangut-n4694-v3-10` and
`tangut_n4694_v17` are successive revisions of one reference design and look
nearly identical; splitting across that pair does not test typeface
generalisation.

**Degradation.** Train and validation images get one tier sampled at
{weights}. {test_tier_design}

**Font counts are not uniform** in the table below, and that is expected: a font
is only used for lines it can fully render, and the fonts differ slightly in
coverage of rare characters. Any line no available font can draw is dropped
rather than rendered with `.notdef` boxes.

**Duplicate lines removed:** {leakage}. Several editions repeat formulaic
Buddhist phrasing verbatim; leaving those in would put identical strings on both
sides of the split.

## Character inventory

`charset.tsv` covers every `Script=Tangut` character as of **Unicode 17.0** and
separates the ordinary OCR repertoire from analytic component characters:

| block | range | characters |
|---|---|---:|
{block_rows}
| **total** | | **{inv["total"]:,}** |

Of these, **{attested:,} Tangut characters are attested in this corpus** and the rest carry
`corpus_count = 0`. The `n_fonts` column gives how many of the available
typefaces can draw each character, which bounds what could be rendered at all.

The default OCR repertoire contains **{inv["ocr_characters"]:,} characters**:
{inv["ideographs"]:,} ideographs plus the Tangut iteration mark. The
{inv["components"]:,} components and component-supplement characters are
analytic symbols and are excluded from an ordinary line recognizer vocabulary.

Of the {inv["ideographs"]:,} ideographs, **{inv["with_lfw"]:,} have a Li Fanwen
number and {inv["ideographs_without_lfw"]} do not** — a fact about the script's
documentation rather than a gap in this dataset. Those {inv["ideographs_without_lfw"]}
entered Unicode from Sofronov 1968, Nishida 1966, Hán Xiǎománg 2004/2021, other
Lǐ Fànwén works or later encoding proposals, and were never given an entry in the
2008 夏漢字典; for them, `source` records where the character does come from. The
{inv["components"]} components have no number either, but for a different reason:
they are structural strokes rather than word-writing units, so no dictionary
lists them.

Li Fanwen numbers are taken from Unicode's `TangutSources.txt`
(`kTGT_MergedSrc` field, `L2008-` references).

## Distributions

{table(by_layout, "layout")}
{table(by_tier, "tier")}
{table(by_font, "font")}

## Known limitations

- The images are **synthetic**. They model typeface, layout, ink weight, skew,
  blur, paper texture and sensor/codec noise, but not the failure modes that
  dominate real Khara-Khoto material: torn and missing paper, worn blocks,
  bleed-through of *specific* verso text, marginalia and seals. Numbers here are
  an upper bound on performance against photographed folios.
- All five available typefaces are modern digital fonts. None reproduces a
  specific historical block-cutter's hand, and typeface diversity for Tangut is
  objectively narrow compared with Latin or Han script.
- {len(missing_lfw)} of the corpus characters have no Li Fanwen number
  ({" ".join(missing_lfw) if missing_lfw else "-"}); their `lfw` label carries a
  `[?]` marker at that position. `text` and `codepoints` are always complete.
- Lacunae (`□`), editorial brackets and illegibility marks present in the source
  transcriptions are stripped before rendering, so a line may not be contiguous
  in the original. Image and label always agree with each other.

## Source

Each text file derives from a separate published edition; cite the underlying
editions, not just this dataset.
"""


if __name__ == "__main__":
    raise SystemExit(main())
