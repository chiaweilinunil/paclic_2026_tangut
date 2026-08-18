from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Importable when run directly as a script, not only as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crnn_ctc.data import TangutLineDataset, make_collate_fn, read_manifest

from svtr_ctc.common import (
    greedy_decode,
    load_checkpoint,
    resolve_device,
    seed_data_loader_worker,
    seed_everything,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "dataset" / "synthetic"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET),
                        help="directory containing manifest.csv")
    parser.add_argument("--checkpoint", required=True, help="trusted best.pt or last.pt")
    parser.add_argument("--output", required=True, help="destination predictions.jsonl")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--layouts", nargs="+", choices=("horizontal", "vertical"), default=None
    )
    parser.add_argument("--tiers", nargs="+", default=None, help="optional tier filter")
    parser.add_argument("--height", type=int, default=None, help="override checkpoint height")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or cuda:<index>"
    )
    parser.add_argument("--limit", type=int, default=None, help="smoke-test subset")
    return parser


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.height is not None and args.height <= 0:
        parser.error("--height must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    model, vocabulary, checkpoint = load_checkpoint(args.checkpoint)
    preprocessing = checkpoint.get("preprocessing", {})
    height = args.height or preprocessing.get("height")
    if not isinstance(height, int) or height <= 0:
        parser.error("checkpoint has no valid preprocessing height; pass --height")
    pad_to_multiple = preprocessing.get("pad_to_multiple", 4)
    if not isinstance(pad_to_multiple, int) or pad_to_multiple <= 0:
        parser.error("checkpoint has an invalid pad_to_multiple value")

    samples = read_manifest(
        args.dataset,
        split=args.split,
        layouts=args.layouts,
        tiers=args.tiers,
        limit=args.limit,
    )
    if not samples:
        parser.error("the selected prediction set is empty")
    dataset = TangutLineDataset(samples, vocabulary=None, height=height)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate_fn(pad_to_multiple),
        worker_init_fn=seed_data_loader_worker,
        generator=generator,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model.to(device).eval()

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    written = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for batch in loader:
                images = batch["images"].to(device, non_blocking=True)
                image_widths = batch["image_widths"]
                log_probabilities = model(images, image_widths)
                input_lengths = model.output_lengths(image_widths)
                predictions = greedy_decode(
                    log_probabilities, input_lengths, vocabulary
                )
                for sample_id, file_name, prediction in zip(
                    batch["sample_ids"], batch["file_names"], predictions
                ):
                    handle.write(
                        json.dumps(
                            {
                                "sample_id": sample_id,
                                "file_name": file_name,
                                "prediction": prediction,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "device": str(device),
                "height": height,
                "samples": written,
                "output": str(destination),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
