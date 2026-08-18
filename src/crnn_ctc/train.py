from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

# Importable when run directly as a script, not only as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crnn_ctc.common import (
    build_checkpoint,
    greedy_decode,
    recognition_metrics,
    resolve_device,
    save_checkpoint,
    seed_data_loader_worker,
    seed_everything,
    validate_ctc_lengths,
)
from crnn_ctc.data import TangutLineDataset, make_collate_fn, read_manifest
from crnn_ctc.model import CRNN
from crnn_ctc.vocab import CTCVocabulary

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "dataset" / "synthetic"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET),
                        help="directory containing manifest.csv")
    parser.add_argument("--output-dir", default=str(ROOT / "models" / "crnn_ctc"),
                        help="checkpoint directory")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument(
        "--layouts", nargs="+", choices=("horizontal", "vertical"), default=None
    )
    parser.add_argument("--tiers", nargs="+", default=None, help="optional tier filter")
    parser.add_argument("--height", type=int, default=64, help="resized line height")
    parser.add_argument("--pad-to-multiple", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--backbone",
        choices=("resnet18", "compact"),
        default="resnet18",
        help="ResNet18 paper baseline (default) or legacy compact smoke model",
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or cuda:<index>"
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        metavar="CHECKPOINT",
        help=(
            "resume training. Bare --resume continues from <output-dir>/last.pt "
            "and starts a fresh run if it does not exist yet, so the same "
            "command can be re-issued after a disconnect. An explicit path must "
            "exist."
        ),
    )
    return parser


def _loader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _capture_rng_state(train_generator: torch.Generator) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "train_generator": train_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict, train_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    train_generator.set_state(state["train_generator"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        if len(cuda_state) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_state)
        else:
            print(
                json.dumps({"warning": "cuda rng state skipped: device count differs"}),
                flush=True,
            )


def _loader(
    dataset: TangutLineDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pad_to_multiple: int,
    generator: torch.Generator,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collate_fn(pad_to_multiple),
        worker_init_fn=seed_data_loader_worker,
        generator=generator,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def _forward_loss(
    model: CRNN,
    batch: dict,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = batch["targets"]
    target_lengths = batch["target_lengths"]
    if targets is None or target_lengths is None:
        raise ValueError("training and validation batches require encoded targets")
    image_widths = batch["image_widths"]
    input_lengths = model.output_lengths(image_widths)
    validate_ctc_lengths(
        targets, target_lengths, input_lengths, batch["sample_ids"]
    )
    images = batch["images"].to(device, non_blocking=True)
    log_probabilities = model(images)
    if int(input_lengths.max().item()) > log_probabilities.shape[0]:
        raise ValueError("computed CTC length exceeds the model time dimension")
    loss = criterion(
        log_probabilities,
        targets.to(device, non_blocking=True),
        input_lengths,
        target_lengths,
    )
    return loss, log_probabilities, input_lengths


def train_one_epoch(
    model: CRNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    device: torch.device,
    gradient_clip: float,
) -> float:
    model.train()
    weighted_loss = 0.0
    n_samples = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = _forward_loss(model, batch, criterion, device)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss: {loss.item()}")
        loss.backward()
        if gradient_clip > 0:
            clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        batch_size = len(batch["sample_ids"])
        weighted_loss += float(loss.detach().item()) * batch_size
        n_samples += batch_size
    return weighted_loss / n_samples


@torch.inference_mode()
def validate(
    model: CRNN,
    loader: DataLoader,
    vocabulary: CTCVocabulary,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> tuple[float, dict]:
    model.eval()
    weighted_loss = 0.0
    n_samples = 0
    references: list[str] = []
    predictions: list[str] = []
    for batch in loader:
        loss, log_probabilities, input_lengths = _forward_loss(
            model, batch, criterion, device
        )
        batch_size = len(batch["sample_ids"])
        weighted_loss += float(loss.item()) * batch_size
        n_samples += batch_size
        references.extend(batch["texts"])
        predictions.extend(greedy_decode(log_probabilities, input_lengths, vocabulary))
    metrics = recognition_metrics(references, predictions).to_dict()
    return weighted_loss / n_samples, metrics


def _restore_run(
    spec: str,
    output_dir: Path,
    *,
    model: CRNN,
    optimizer: torch.optim.Optimizer,
    vocabulary: CTCVocabulary,
    train_generator: torch.Generator,
    parser: argparse.ArgumentParser,
) -> tuple[int, float]:
    automatic = spec == "auto"
    path = output_dir / "last.pt" if automatic else Path(spec)
    if not path.exists():
        if automatic:
            print(
                json.dumps({"resume": "no checkpoint found, starting at epoch 1"}),
                flush=True,
            )
            return 1, float("inf")
        parser.error(f"--resume checkpoint not found: {path}")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if payload.get("format_version") not in {1, 2}:
        parser.error(
            f"{path} has unsupported checkpoint format "
            f"{payload.get('format_version')!r}"
        )
    if payload.get("model", {}).get("architecture") != "CRNN":
        parser.error(f"{path} is not a CRNN checkpoint")

    restored_vocabulary = CTCVocabulary.from_dict(payload["vocabulary"])
    if restored_vocabulary.chars != vocabulary.chars:
        parser.error(
            f"{path} was trained on a different vocabulary "
            f"({restored_vocabulary.size} vs {vocabulary.size} classes); "
            "the dataset or sample limits changed, so this run cannot resume it"
        )
    if int(payload["model"]["hidden_size"]) != model.hidden_size:
        parser.error(
            f"{path} has hidden_size {payload['model']['hidden_size']}, "
            f"but --hidden-size is {model.hidden_size}"
        )
    checkpoint_backbone = str(payload["model"].get("backbone", "compact"))
    if checkpoint_backbone != model.backbone_name:
        parser.error(
            f"{path} has backbone {checkpoint_backbone!r}, "
            f"but --backbone is {model.backbone_name!r}"
        )

    model.load_state_dict(payload["model_state_dict"])
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer_state is None:
        print(
            json.dumps({"warning": "checkpoint has no optimizer state; AdamW restarts"}),
            flush=True,
        )
    else:
        optimizer.load_state_dict(optimizer_state)

    rng_state = payload.get("rng_state")
    if rng_state is None:
        print(
            json.dumps({"warning": "checkpoint predates RNG capture; order differs"}),
            flush=True,
        )
    else:
        _restore_rng_state(rng_state, train_generator)

    completed = int(payload["training"]["epoch"])
    best_cer = payload.get("best_cer")
    if best_cer is None:
        best_path = output_dir / "best.pt"
        if best_path.exists():
            try:
                best_payload = torch.load(
                    best_path, map_location="cpu", weights_only=False
                )
            except TypeError:
                best_payload = torch.load(best_path, map_location="cpu")
            best_cer = best_payload.get("metrics", {}).get("val_cer")
        if best_cer is None:
            best_cer = payload.get("metrics", {}).get("val_cer", float("inf"))
    return completed + 1, float(best_cer)


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive = {
        "height": args.height,
        "pad-to-multiple": args.pad_to_multiple,
        "batch-size": args.batch_size,
        "epochs": args.epochs,
        "hidden-size": args.hidden_size,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"--{name} must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    for name in ("max_train_samples", "max_val_samples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    train_samples = read_manifest(
        args.dataset,
        split=args.train_split,
        layouts=args.layouts,
        tiers=args.tiers,
        limit=args.max_train_samples,
    )
    val_samples = read_manifest(
        args.dataset,
        split=args.val_split,
        layouts=args.layouts,
        tiers=args.tiers,
        limit=args.max_val_samples,
    )
    if not train_samples:
        parser.error("the selected training set is empty")
    if not val_samples:
        parser.error("the selected validation set is empty")

    vocabulary = CTCVocabulary.from_texts(sample.text for sample in train_samples)
    train_dataset = TangutLineDataset(train_samples, vocabulary, height=args.height)
    val_dataset = TangutLineDataset(val_samples, vocabulary, height=args.height)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pad_to_multiple": args.pad_to_multiple,
        "pin_memory": device.type == "cuda",
    }
    train_generator = _loader_generator(args.seed)
    train_loader = _loader(
        train_dataset, shuffle=True, generator=train_generator, **loader_options
    )
    val_loader = _loader(
        val_dataset,
        shuffle=False,
        generator=_loader_generator(args.seed + 1),
        **loader_options,
    )

    model = CRNN(
        vocabulary.size,
        hidden_size=args.hidden_size,
        backbone=args.backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CTCLoss(blank=vocabulary.blank_index, zero_infinity=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocabulary.save(output_dir / "vocab.json")

    preprocessing = {
        "height": args.height,
        "pad_to_multiple": args.pad_to_multiple,
        "grayscale": True,
        "pixel_encoding": "ink_1_background_0",
        "vertical_rotation": "90_degrees_counter_clockwise",
        "resize": "fixed_height_aspect_preserving",
    }
    training_config = vars(args).copy()
    training_config["effective_device"] = str(device)
    print(
        json.dumps(
            {
                "device": str(device),
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
                "vocabulary_size_including_blank_unk": vocabulary.size,
                "backbone": model.backbone_name,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    start_epoch, best_cer = 1, float("inf")
    if args.resume is not None:
        start_epoch, best_cer = _restore_run(
            args.resume,
            output_dir,
            model=model,
            optimizer=optimizer,
            vocabulary=vocabulary,
            train_generator=train_generator,
            parser=parser,
        )
        if start_epoch > 1:
            print(
                json.dumps({"resumed_from_epoch": start_epoch - 1, "best_cer": best_cer}),
                flush=True,
            )
    if start_epoch > args.epochs:
        print(
            json.dumps(
                {"status": "already complete", "epochs": args.epochs, "best_cer": best_cer}
            ),
            flush=True,
        )
        return 0

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.gradient_clip
        )
        val_loss, val_metrics = validate(
            model, val_loader, vocabulary, criterion, device
        )
        epoch_metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_cer": val_metrics["cer"],
            "val_exact": val_metrics["exact"],
            "val_edits": val_metrics["edits"],
            "val_reference_characters": val_metrics["reference_characters"],
            "val_samples": val_metrics["n"],
        }
        is_best = val_metrics["cer"] < best_cer
        if is_best:
            best_cer = val_metrics["cer"]
        checkpoint = build_checkpoint(
            model,
            vocabulary,
            epoch=epoch,
            seed=args.seed,
            preprocessing=preprocessing,
            training_config=training_config,
            metrics=epoch_metrics,
            optimizer=optimizer,
        )
        checkpoint["best_cer"] = best_cer
        checkpoint["rng_state"] = _capture_rng_state(train_generator)
        save_checkpoint(checkpoint, output_dir / "last.pt")
        if is_best:
            save_checkpoint(checkpoint, output_dir / "best.pt")
        print(
            json.dumps(
                {"epoch": epoch, "best": is_best, **epoch_metrics},
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
