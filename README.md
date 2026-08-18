# TangutLines

Dataset builder, baselines and evaluation for line-level Tangut OCR.

## Layout

    text/                 corpus transcriptions, one *.txt per work 
    data/                 TangutSources.txt, the Unicode input for labels
    dataset/synthetic/    generated images and labels (imagefolder metadata)
    dataset/real/         transcribed manuscript line crops
    models/               checkpoint vocabularies (weights: release asset)
    eval/                 released predictions and scores
    src/build_dataset/    renderer and corpus pipeline
    src/crnn_ctc/         CRNN–CTC baseline
    src/svtr_ctc/         SVTR–CTC baseline
    src/evaluate.py       scorer
    src/error_analysis.py character confusions -> error_analysis/
    error_analysis/       confusion tables (TSV) and full pairs (JSON)

## Requirements

    pip install torch torchvision pillow numpy opencv-python fonttools tqdm

Rendering also needs the five Tangut fonts installed locally; they are
third-party and are not redistributed here. `build_dataset.py --list-fonts`
reports what it can find, and `build_config.json` records a SHA-256 digest of
each font actually used.

## End to end

Every command is run from this directory and defaults to the paths above.

    # 1. render the dataset (needs text/ populated; see text/README.md)
    python src/build_dataset/build_dataset.py

    # 2. train — defaults are the settings the paper reports
    python src/crnn_ctc/train.py
    python src/svtr_ctc/train.py

    # 3. decode the test split
    python src/crnn_ctc/predict.py \
        --checkpoint models/crnn_ctc/best.pt \
        --output eval/crnn_ctc/test_predictions.jsonl

    # 4. score
    python src/evaluate.py \
        --predictions eval/crnn_ctc/test_predictions.jsonl \
        --json-out eval/crnn_ctc/test_results.json

    # 5. character confusions, written to error_analysis/
    python src/error_analysis.py --predictions eval/crnn_ctc/test_predictions.jsonl

To score the released checkpoints without retraining, download them from the
release assets into `models/<baseline>/`, then skip step 2 and pass
`--checkpoint models/crnn_ctc/crnn_ctc.pt` in step 3. The weights are 189 MB
(CRNN--CTC) and 31 MB (SVTR--CTC), so they are not stored in git; each
`vocab.json` is, since it is what maps class indices back to characters.

Training writes `best.pt` and `last.pt` to `models/<baseline>/`; `--resume`
continues from `last.pt`, so an interrupted run can be re-issued unchanged.

## Labels

Labels ship as `dataset/synthetic/images/<split>/metadata.jsonl`, one JSON object
per image, which is the HuggingFace `imagefolder` layout:

    from datasets import load_dataset
    load_dataset("imagefolder", data_dir="dataset/synthetic/images")


## Reproducibility

`dataset/build_config.json` records the seed, every build parameter, and SHA-256
digests of each corpus file, each font and each generator source file. The build
is deterministic given the seed.
