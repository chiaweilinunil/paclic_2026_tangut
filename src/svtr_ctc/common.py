from __future__ import annotations

import platform
from pathlib import Path

import torch

from crnn_ctc.common import (
    greedy_decode,
    recognition_metrics,
    resolve_device,
    save_checkpoint,
    seed_data_loader_worker,
    seed_everything,
    validate_ctc_lengths,
)
from crnn_ctc.vocab import CTCVocabulary

from .model import SVTRCTC


CHECKPOINT_FORMAT_VERSION = 1
ARCHITECTURE = "SVTR-CTC"


def torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    return payload


def build_checkpoint(
    model: SVTRCTC,
    vocabulary: CTCVocabulary,
    *,
    epoch: int,
    seed: int,
    preprocessing: dict,
    training_config: dict,
    metrics: dict,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    if model.num_classes != vocabulary.size:
        raise ValueError("model class count and vocabulary size differ")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": {
            "architecture": ARCHITECTURE,
            "config": model.to_config(),
            "width_reduction": model.width_reduction,
        },
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "vocabulary": vocabulary.to_dict(),
        "preprocessing": dict(preprocessing),
        "training": {
            "epoch": int(epoch),
            "seed": int(seed),
            "config": dict(training_config),
        },
        "metrics": dict(metrics),
        "software": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def validate_checkpoint(payload: dict) -> tuple[dict, CTCVocabulary]:
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format: {payload.get('format_version')!r}"
        )
    model_spec = payload.get("model")
    if not isinstance(model_spec, dict) or model_spec.get("architecture") != ARCHITECTURE:
        raise ValueError("checkpoint does not contain an SVTR-CTC model")
    config = model_spec.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no SVTR model configuration")
    vocabulary = CTCVocabulary.from_dict(payload["vocabulary"])
    if int(config.get("num_classes", -1)) != vocabulary.size:
        raise ValueError("checkpoint model/vocabulary sizes do not agree")
    return config, vocabulary


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[SVTRCTC, CTCVocabulary, dict]:
    payload = torch_load(path, map_location=map_location)
    config, vocabulary = validate_checkpoint(payload)
    model = SVTRCTC.from_config(config)
    model.load_state_dict(payload["model_state_dict"])
    return model, vocabulary, payload


__all__ = [
    "ARCHITECTURE",
    "CHECKPOINT_FORMAT_VERSION",
    "build_checkpoint",
    "greedy_decode",
    "load_checkpoint",
    "recognition_metrics",
    "resolve_device",
    "save_checkpoint",
    "seed_data_loader_worker",
    "seed_everything",
    "torch_load",
    "validate_checkpoint",
    "validate_ctc_lengths",
]
