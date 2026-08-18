# SVTR–CTC baseline

Same manifest, vocabulary, CTC objective, greedy decoder and evaluation contract
as `crnn_ctc`, so the comparison changes the visual encoder and nothing else.

## Architecture

1. two overlapping stride-two convolutions embed image patches;
2. three resolution stages separated by vertical patch-merging layers;
3. early stages mix locally (bounded depthwise), the last stage globally
   (multi-head attention over the recognition axis);
4. height pooling and a linear CTC head.

The stem reduces width by four and the mergers reduce only height, so for an
unpadded width `w` the CTC length is exactly `ceil(w / 4)`. Per-sample widths are
carried through the network: right-padding is zeroed before the stem, excluded
from global attention, masked after every residual block, and excluded from CTC
loss and decoding. Logits for a line are therefore independent of which lines
share its batch.

Preprocessing is imported from `crnn_ctc.data`: greyscale, ink `1.0`, background
`0.0`, aspect-preserving fixed-height resize, and a 90° counter-clockwise
rotation for vertical lines so historical top-to-bottom order becomes
left-to-right model time.

## Usage

Defaults reproduce the paper; see the repository README for the full pipeline.

    python src/svtr_ctc/train.py
    python src/svtr_ctc/predict.py --checkpoint models/svtr_ctc/best.pt \
        --output eval/svtr_ctc/test_predictions.jsonl
