# Synthetic Tangut OCR line dataset

Line images rendered from the **LCDTT** corpus (Lausanne-Vienna-Ghent Corpus of
Digital Tangut Texts), paired with Unicode ground truth and two derived label
serializations. Built with `build_dataset.py`; every parameter plus input,
font, and code SHA-256 is recorded in `build_config.json`, and the build is
deterministic given `seed = 20260805`.

## Contents

- **23,464 images** over **8,612 unique text lines**
  (111,463 Tangut characters, 2,816 distinct characters) drawn from
  **29 works**.
- Labels per image: `text` (Tangut Unicode, the OCR ground truth), `lfw`
  (Li Fanwen numbers), `codepoints` (`U+17xxx`, ASCII-safe).
- Per-character bounding boxes in `manifest.csv`, usable for a detection or
  segmentation baseline without re-rendering.
- `charset.tsv` lists all **7,059 Unicode 17 `Script=Tangut`**
  characters across five ranges and marks the **6,176**
  ordinary-text characters with `ocr_default = 1`. Analytic components remain
  documented but are not put in the default line-transcription decoder.

| split | images | unique lines |
|---|---:|---:|
| train | 13,946 | 6,973 |
| val | 1,718 | 859 |
| test | 7,800 | 780 |

## Design

**Splits.** Partitioned by `random` at ratios
[0.8, 0.1, 0.1]. `random` shuffles lines within each document, so train and
test share vocabulary and register. `document` holds out whole works, which is
the harder and more informative setting: 論語 and 華嚴經 share little vocabulary,
so it measures generalisation rather than memorisation of the corpus.

**Held-out fonts.** none (all fonts in all splits). Test font policy:
`paired`. Under `paired`, every eligible
test string is rendered in every test font; lines unsupported by any one test
font are excluded, so per-font differences are paired by `line_id`. Note that
`tangut-n4694-v3-10` and
`tangut_n4694_v17` are successive revisions of one reference design and look
nearly identical; splitting across that pair does not test typeface
generalisation.

**Degradation.** Train and validation images get one tier sampled at
clean 0.40, light 0.25, medium 0.20, heavy 0.15. Test lines receive **one** uniformly balanced tier per `line_id`, reused across all selected fonts and layouts; this preserves paired font/layout comparisons without duplicating each line across tiers.

**Font counts are not uniform** in the table below, and that is expected: a font
is only used for lines it can fully render, and the fonts differ slightly in
coverage of rare characters. Any line no available font can draw is dropped
rather than rendered with `.notdef` boxes.

**Duplicate lines removed:** yes. Several editions repeat formulaic
Buddhist phrasing verbatim; leaving those in would put identical strings on both
sides of the split.

## Character inventory

`charset.tsv` covers every `Script=Tangut` character as of **Unicode 17.0** and
separates the ordinary OCR repertoire from analytic component characters:

| block | range | characters |
|---|---|---:|
| Tangut Iteration Mark | U+16FE0..U+16FE0 | 1 |
| Tangut | U+17000..U+187FF | 6,144 |
| Tangut Components | U+18800..U+18AFF | 768 |
| Tangut Supplement | U+18D00..U+18D1E | 31 |
| Tangut Components Supplement | U+18D80..U+18DF2 | 115 |
| **total** | | **7,059** |

Of these, **2,809 Tangut characters are attested in this corpus** and the rest carry
`corpus_count = 0`. The `n_fonts` column gives how many of the available
typefaces can draw each character, which bounds what could be rendered at all.

The default OCR repertoire contains **6,176 characters**:
6,175 ideographs plus the Tangut iteration mark. The
883 components and component-supplement characters are
analytic symbols and are excluded from an ordinary line recognizer vocabulary.

Of the 6,175 ideographs, **6,068 have a Li Fanwen
number and 107 do not** — a fact about the script's
documentation rather than a gap in this dataset. Those 107
entered Unicode from Sofronov 1968, Nishida 1966, Hán Xiǎománg 2004/2021, other
Lǐ Fànwén works or later encoding proposals, and were never given an entry in the
2008 夏漢字典; for them, `source` records where the character does come from. The
883 components have no number either, but for a different reason:
they are structural strokes rather than word-writing units, so no dictionary
lists them.

Li Fanwen numbers are taken from Unicode's `TangutSources.txt` (`kTGT_MergedSrc`
field, `L2008-` references) and agree with LCDTT's own `tangut_lfw.json` on all
6,068 characters the two share.

## Distributions

| layout | images |
|---|---:|
| horizontal | 11,732 |
| vertical | 11,732 |

| tier | images |
|---|---:|
| clean | 8,214 |
| light | 5,949 |
| medium | 5,006 |
| heavy | 4,295 |

| font | images |
|---|---:|
| notoseriftangut-regular | 4,874 |
| tangut-n4694-v3-10 | 4,864 |
| new-tangut-std-v2-008 | 4,804 |
| mojikyo-tangut-1-1 | 4,690 |
| tangut-twu-regular-v1-007 | 4,232 |


## Known limitations

- The images are **synthetic**. They model typeface, layout, ink weight, skew,
  blur, paper texture and sensor/codec noise, but not the failure modes that
  dominate real Khara-Khoto material: torn and missing paper, worn blocks,
  bleed-through of *specific* verso text, marginalia and seals. Numbers here are
  an upper bound on performance against photographed folios.
- All five available typefaces are modern digital fonts. None reproduces a
  specific historical block-cutter's hand, and typeface diversity for Tangut is
  objectively narrow compared with Latin or Han script.
- 1 of the corpus characters have no Li Fanwen number
  (𘟱); their `lfw` label carries a
  `[?]` marker at that position. `text` and `codepoints` are always complete.
- Lacunae (`□`), editorial brackets and illegibility marks present in LCDTT are
  stripped before rendering, so a line may not be contiguous in the original
  source. Image and label always agree with each other.

## Source

LCDTT is compiled by Chia-Wei Lin (Université de Lausanne), Longyu Zhang
(Universiteit Gent) and Yue Ji (Universität Wien). Each text file derives from a
separate published edition; see the LCDTT `README.md` for per-work attribution,
and cite the underlying editions, not just this dataset.
