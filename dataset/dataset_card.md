# TangutLines

Tangut line images paired with Unicode ground truth: a synthetic set rendered
from a digital corpus, and a small real set cropped from manuscript photographs.

```
dataset/
  build_config.json                every build parameter, plus SHA-256 of each
                                   corpus file, font, and source module
  synthetic/
    lines.jsonl                    the 8,612 source lines
    images/{train,val,test}/
      metadata.jsonl               one record per image: the labels
      {horizontal,vertical}/*.png
  real/
    {妙法蓮華經,維摩詰經}/
      lines.jsonl                  transcriptions
      images/*.png                 one crop per manuscript column
```

## Synthetic

23,464 images from 8,612 lines across 29 works: 111,463 Tangut characters over
2,809 distinct Tangut characters, plus 4,407 punctuation marks over 7 types.
Lines are 4–18 characters long, mean 12.9.

| split | images | lines | renders per line |
|---|---:|---:|---:|
| train | 13,946 | 6,973 | 2 |
| val | 1,718 | 859 | 2 |
| test | 7,800 | 780 | 10 |

Test lines are rendered in all five typefaces and both layouts, so per-typeface and per-layout comparisons are paired by `line_id`. Train and validation lines
get one typeface and one degradation tier sampled per line. Layouts are exactly balanced, 11,732 images each; tiers and typefaces are not, because tiers are sampled by weight and a typeface is only used for lines it can fully render.

| tier | images | | typeface | images |
|---|---:|---|---|---:|
| clean | 8,214 | | Noto Serif Tangut | 4,874 |
| light | 5,949 | | Tangut N4694 | 4,864 |
| medium | 5,006 | | new Tangut Std | 4,804 |
| heavy | 4,295 | | Tangut Mojikyo | 4,690 |
| | | | Tangut TWU | 4,232 |

**Splits are random within documents**, so train and test share vocabulary and
register. Results on this test split measure recognition, not generalisation to
an unseen work. Duplicate lines are removed before splitting: several editions
repeat formulaic Buddhist phrasing verbatim, which would otherwise put identical
strings on both sides of the split.

`metadata.jsonl` fields: `file_name`, `sample_id`, `split`, `line_id`, `doc`,
`source_line`, `text`, `lfw`, `codepoints`, `layout`, `font`, `tier`,
`n_chars`. `text` is the OCR target; `lfw` gives Li Fanwen catalogue numbers
and `codepoints` an ASCII-safe form of the same string.

The build is deterministic given `seed = 20260805`; rebuild with
`src/build_dataset/`. 100 of 8,712 candidate lines were dropped because not
every test typeface could render them.

## Real

1,132 crops from two manuscripts, of which 607 carry a transcription. Every crop
is a vertical column read top to bottom.

| document | images | transcribed | characters |
|---|---:|---:|---:|
| 妙法蓮華經 | 575 | 50 | 761 |
| 維摩詰經 | 557 | 557 | 9,149 |

`lines.jsonl` fields: `file_name`, `line_id`, `doc`, `source_line`, `text`,
`n_chars`. `source_line` numbers columns within each manuscript's own
transcription and does not index the synthetic corpus.
