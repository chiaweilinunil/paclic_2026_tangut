from __future__ import annotations

import os
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from .model import CRNN
from .vocab import CTCVocabulary


CHECKPOINT_FORMAT_VERSION = 2


@dataclass(frozen=True)
class RecognitionMetrics:
    n: int
    edits: int
    reference_characters: int
    cer: float
    exact: float

    def to_dict(self) -> dict:
        return asdict(self)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def seed_data_loader_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(requested: str) -> torch.device:
    value = requested.strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda" or value.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        device = torch.device(value)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {device.index} does not exist; "
                f"found {torch.cuda.device_count()} device(s)"
            )
        return device
    raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")


def levenshtein(reference: str, prediction: str) -> int:
    if reference == prediction:
        return 0
    if not reference:
        return len(prediction)
    if not prediction:
        return len(reference)
    previous = list(range(len(prediction) + 1))
    for row, ref_char in enumerate(reference, start=1):
        current = [row]
        for column, pred_char in enumerate(prediction, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (ref_char != pred_char),
                )
            )
        previous = current
    return previous[-1]


def recognition_metrics(
    references: Sequence[str], predictions: Sequence[str]
) -> RecognitionMetrics:
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")
    if not references:
        raise ValueError("at least one reference is required")
    edits = sum(levenshtein(ref, pred) for ref, pred in zip(references, predictions))
    reference_characters = sum(len(ref) for ref in references)
    if reference_characters == 0:
        raise ValueError("CER is undefined when all references are empty")
    exact_count = sum(ref == pred for ref, pred in zip(references, predictions))
    return RecognitionMetrics(
        n=len(references),
        edits=edits,
        reference_characters=reference_characters,
        cer=edits / reference_characters,
        exact=exact_count / len(references),
    )


def greedy_decode(
    log_probabilities: torch.Tensor,
    input_lengths: Iterable[int] | torch.Tensor,
    vocabulary: CTCVocabulary,
) -> list[str]:
    if log_probabilities.ndim != 3:
        raise ValueError("log_probabilities must have shape [time, batch, classes]")
    if log_probabilities.shape[2] != vocabulary.size:
        raise ValueError("model output size does not match vocabulary")
    lengths = [int(value) for value in input_lengths]
    if len(lengths) != log_probabilities.shape[1]:
        raise ValueError("one input length is required per batch item")
    best = log_probabilities.argmax(dim=2).transpose(0, 1).detach().cpu()
    output: list[str] = []
    for indices, length in zip(best, lengths):
        if length < 0 or length > log_probabilities.shape[0]:
            raise ValueError(f"invalid CTC input length: {length}")
        output.append(vocabulary.decode_indices(indices[:length].tolist()))
    return output


def ctc_required_timesteps(indices: torch.Tensor) -> int:
    if indices.ndim != 1:
        raise ValueError("CTC target must be one-dimensional")
    if indices.numel() == 0:
        return 0
    repeats = int((indices[1:] == indices[:-1]).sum().item())
    return int(indices.numel()) + repeats


def validate_ctc_lengths(
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    input_lengths: torch.Tensor,
    sample_ids: Sequence[str],
) -> None:
    if len(target_lengths) != len(input_lengths) or len(sample_ids) != len(input_lengths):
        raise ValueError("batch length metadata is inconsistent")
    offset = 0
    failures: list[str] = []
    for sample_id, target_length, input_length in zip(
        sample_ids, target_lengths.tolist(), input_lengths.tolist()
    ):
        target = targets[offset : offset + target_length]
        required = ctc_required_timesteps(target)
        if input_length < required:
            failures.append(f"{sample_id} ({input_length} < {required})")
        offset += target_length
    if offset != targets.numel():
        raise ValueError("target_lengths do not sum to concatenated target size")
    if failures:
        raise ValueError(
            "CTC time axis is too short; increase --height. Invalid samples: "
            + ", ".join(failures[:5])
        )


def build_checkpoint(
    model: CRNN,
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
            "architecture": "CRNN",
            "backbone": model.backbone_name,
            "num_classes": model.num_classes,
            "hidden_size": model.hidden_size,
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


def save_checkpoint(payload: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[CRNN, CTCVocabulary, dict]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if payload.get("format_version") not in {1, CHECKPOINT_FORMAT_VERSION}:
        raise ValueError(
            f"unsupported checkpoint format: {payload.get('format_version')!r}"
        )
    model_spec = payload.get("model", {})
    if model_spec.get("architecture") != "CRNN":
        raise ValueError("checkpoint does not contain a CRNN model")
    vocabulary = CTCVocabulary.from_dict(payload["vocabulary"])
    if model_spec.get("num_classes") != vocabulary.size:
        raise ValueError("checkpoint model/vocabulary sizes do not agree")
    model = CRNN(
        num_classes=vocabulary.size,
        hidden_size=int(model_spec["hidden_size"]),
        backbone=str(model_spec.get("backbone", "compact")),
    )
    model.load_state_dict(payload["model_state_dict"])
    return model, vocabulary, payload
