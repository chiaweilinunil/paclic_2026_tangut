# TangutLines

Dataset, baselines, and evaluation tools for line-level Tangut OCR, released with our PACLIC 2026 paper.

## Dataset

TangutLines contains:

- **Synthetic:** 23,464 images rendered from 8,612 lines across 29 works, five typefaces, and horizontal and vertical layouts.
- **Real:** 1,132 manuscript line crops from *妙法蓮華經* and *維摩詰經*, including 607 transcribed lines.

See [`dataset/dataset_card.md`](dataset/dataset_card.md) for the dataset structure, statistics, and limitations.

## Repository

```text
dataset/         labels and real manuscript crops
src/             dataset builder, OCR baselines, and evaluation tools
models/          model vocabularies (checkpoints are release assets)
eval/            predictions and evaluation results
error_analysis/  character-confusion tables
data/            Unicode Tangut character data
```

The synthetic images and model checkpoints are distributed through this repository's GitHub Releases because of their size. Corpus transcriptions and third-party Tangut fonts are not redistributed.

## Setup

```bash
pip install torch torchvision pillow numpy opencv-python fonttools tqdm
```

Dataset generation requires the source corpus as UTF-8 `.txt` files in `text/` and locally installed Tangut fonts. To inspect detected fonts:

```bash
python src/build_dataset/build_dataset.py --list-fonts
```

## Usage

Run all commands from the repository root.

```bash
# Build the synthetic dataset
python src/build_dataset/build_dataset.py

# Train the baselines
python src/crnn_ctc/train.py
python src/svtr_ctc/train.py

# Predict with released checkpoints
python src/crnn_ctc/predict.py --checkpoint models/crnn_ctc/crnn_ctc.pt --output eval/crnn_ctc/test_predictions.jsonl
python src/svtr_ctc/predict.py --checkpoint models/svtr_ctc/svtr_ctc.pt --output eval/svtr_ctc/test_predictions.jsonl

# Evaluate and analyze errors
python src/evaluate.py --predictions eval/crnn_ctc/test_predictions.jsonl --json-out eval/crnn_ctc/test_results.json
python src/error_analysis.py --predictions eval/crnn_ctc/test_predictions.jsonl
```

The dataset build is deterministic with the default seed. Build parameters are recorded in `dataset/synthetic/build_config.json`.

## Results

Results on the 7,800-image synthetic test split:

| Model | CER | Exact-line accuracy |
| --- | ---: | ---: |
| CRNN-CTC | 12.40% | 48.62% |
| SVTR-CTC | 10.88% | 48.69% |

Detailed results and released predictions are available in [`eval/`](eval/).
