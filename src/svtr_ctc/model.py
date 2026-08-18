from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _as_three_ints(name: str, values: Sequence[int]) -> tuple[int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    if any(value <= 0 for value in result):
        raise ValueError(f"all {name} values must be positive")
    return result


def _valid_width_mask(
    lengths: torch.Tensor, height: int, width: int, device: torch.device
) -> torch.Tensor:
    columns = torch.arange(width, device=device).unsqueeze(0)
    valid_columns = columns < lengths.to(device=device).unsqueeze(1)
    return valid_columns.unsqueeze(1).expand(-1, height, -1)


def _apply_mask(features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    return features.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


class PatchEmbedding(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        hidden = max(4, out_channels // 2)
        self.projection = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images)


class PatchMerging(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(2, 1),
            padding=1,
        )
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(self.projection(features))


class LocalMixer(nn.Module):
    def __init__(self, dimension: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be a positive odd integer")
        self.depthwise = nn.Conv2d(
            dimension,
            dimension,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dimension,
        )
        self.pointwise = nn.Conv2d(dimension, dimension, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        spatial = features.permute(0, 3, 1, 2)
        spatial = spatial.masked_fill(~valid_mask.unsqueeze(1), 0.0)
        mixed = self.pointwise(self.activation(self.depthwise(spatial)))
        mixed = self.dropout(mixed).permute(0, 2, 3, 1)
        return _apply_mask(mixed, valid_mask)


class GlobalMixer(nn.Module):
    def __init__(self, dimension: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if dimension % num_heads:
            raise ValueError("stage dimension must be divisible by its head count")
        self.attention = nn.MultiheadAttention(
            dimension,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = features.shape
        sequences = features.reshape(batch * height, width, channels)
        padding_mask = ~valid_mask.reshape(batch * height, width)
        mixed, _ = self.attention(
            sequences,
            sequences,
            sequences,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        mixed = self.dropout(mixed).reshape(batch, height, width, channels)
        return _apply_mask(mixed, valid_mask)


class FeedForward(nn.Module):
    def __init__(self, dimension: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = max(1, int(round(dimension * mlp_ratio)))
        self.layers = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class SVTRBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        num_heads: int,
        mixer: str,
        mlp_ratio: float,
        dropout: float,
        local_kernel_size: int,
    ) -> None:
        super().__init__()
        self.mixer_kind = mixer
        self.mixer_norm = nn.LayerNorm(dimension)
        if mixer == "local":
            self.mixer = LocalMixer(dimension, local_kernel_size, dropout)
        elif mixer == "global":
            self.mixer = GlobalMixer(dimension, num_heads, dropout)
        else:
            raise ValueError(f"unknown mixer: {mixer!r}")
        self.mlp_norm = nn.LayerNorm(dimension)
        self.mlp = FeedForward(dimension, mlp_ratio, dropout)

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        features = features + self.mixer(self.mixer_norm(features), valid_mask)
        features = _apply_mask(features, valid_mask)
        features = features + self.mlp(self.mlp_norm(features))
        return _apply_mask(features, valid_mask)


class SVTRStage(nn.Module):
    def __init__(
        self,
        dimension: int,
        depth: int,
        num_heads: int,
        mixer: str,
        mlp_ratio: float,
        dropout: float,
        local_kernel_size: int,
    ) -> None:
        super().__init__()
        self.position = nn.Conv2d(
            dimension,
            dimension,
            kernel_size=3,
            padding=1,
            groups=dimension,
        )
        self.blocks = nn.ModuleList(
            [
                SVTRBlock(
                    dimension,
                    num_heads,
                    mixer,
                    mlp_ratio,
                    dropout,
                    local_kernel_size,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        spatial = features.permute(0, 3, 1, 2)
        position = self.position(spatial).permute(0, 2, 3, 1)
        features = _apply_mask(features + position, valid_mask)
        for block in self.blocks:
            features = block(features, valid_mask)
        return features


class SVTRCTC(nn.Module):
    width_reduction = 4

    def __init__(
        self,
        num_classes: int,
        embed_dims: Sequence[int] = (64, 128, 192),
        depths: Sequence[int] = (2, 2, 3),
        num_heads: Sequence[int] = (2, 4, 6),
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        local_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must include blank and at least one label")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if local_kernel_size <= 0 or local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be a positive odd integer")

        self.num_classes = int(num_classes)
        self.embed_dims = _as_three_ints("embed_dims", embed_dims)
        self.depths = _as_three_ints("depths", depths)
        self.num_heads = _as_three_ints("num_heads", num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        self.local_kernel_size = int(local_kernel_size)
        for dimension, heads in zip(self.embed_dims, self.num_heads):
            if dimension % heads:
                raise ValueError(
                    f"embed dimension {dimension} is not divisible by {heads} heads"
                )

        self.patch_embedding = PatchEmbedding(self.embed_dims[0])
        self.stages = nn.ModuleList(
            [
                SVTRStage(
                    self.embed_dims[0],
                    self.depths[0],
                    self.num_heads[0],
                    "local",
                    self.mlp_ratio,
                    self.dropout,
                    self.local_kernel_size,
                ),
                SVTRStage(
                    self.embed_dims[1],
                    self.depths[1],
                    self.num_heads[1],
                    "local",
                    self.mlp_ratio,
                    self.dropout,
                    self.local_kernel_size,
                ),
                SVTRStage(
                    self.embed_dims[2],
                    self.depths[2],
                    self.num_heads[2],
                    "global",
                    self.mlp_ratio,
                    self.dropout,
                    self.local_kernel_size,
                ),
            ]
        )
        self.mergers = nn.ModuleList(
            [
                PatchMerging(self.embed_dims[0], self.embed_dims[1]),
                PatchMerging(self.embed_dims[1], self.embed_dims[2]),
            ]
        )
        self.output_norm = nn.LayerNorm(self.embed_dims[-1])
        self.classifier = nn.Linear(self.embed_dims[-1], self.num_classes)

    def to_config(self) -> dict[str, object]:
        return {
            "num_classes": self.num_classes,
            "embed_dims": list(self.embed_dims),
            "depths": list(self.depths),
            "num_heads": list(self.num_heads),
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "local_kernel_size": self.local_kernel_size,
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "SVTRCTC":
        return cls(
            num_classes=int(config["num_classes"]),
            embed_dims=config["embed_dims"],
            depths=config["depths"],
            num_heads=config["num_heads"],
            mlp_ratio=float(config["mlp_ratio"]),
            dropout=float(config["dropout"]),
            local_kernel_size=int(config["local_kernel_size"]),
        )

    def output_lengths(self, image_widths: torch.Tensor) -> torch.Tensor:
        if image_widths.ndim != 1:
            raise ValueError("image_widths must be one-dimensional")
        if torch.any(image_widths <= 0):
            raise ValueError("all image widths must be positive")
        lengths = torch.div(image_widths + 1, 2, rounding_mode="floor")
        return torch.div(lengths + 1, 2, rounding_mode="floor")

    def forward(
        self, images: torch.Tensor, image_widths: torch.Tensor | None = None
    ) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("images must have shape [batch, 1, height, width]")
        batch, _, _, padded_width = images.shape
        if image_widths is None:
            image_widths = torch.full(
                (batch,), padded_width, dtype=torch.long, device=images.device
            )
        if image_widths.ndim != 1 or image_widths.numel() != batch:
            raise ValueError("one image width is required per batch item")
        if torch.any(image_widths <= 0) or torch.any(image_widths > padded_width):
            raise ValueError("image widths must be in [1, padded image width]")

        pixel_columns = torch.arange(padded_width, device=images.device).unsqueeze(0)
        pixel_valid = pixel_columns < image_widths.to(images.device).unsqueeze(1)
        images = images.masked_fill(~pixel_valid[:, None, None, :], 0.0)

        spatial = self.patch_embedding(images)
        lengths = self.output_lengths(image_widths).to(images.device)
        valid_mask = _valid_width_mask(
            lengths, spatial.shape[2], spatial.shape[3], images.device
        )
        features = _apply_mask(spatial.permute(0, 2, 3, 1), valid_mask)

        features = self.stages[0](features, valid_mask)
        for merger, stage in zip(self.mergers, self.stages[1:]):
            spatial = merger(features.permute(0, 3, 1, 2))
            valid_mask = _valid_width_mask(
                lengths, spatial.shape[2], spatial.shape[3], images.device
            )
            features = _apply_mask(spatial.permute(0, 2, 3, 1), valid_mask)
            features = stage(features, valid_mask)

        features = _apply_mask(self.output_norm(features), valid_mask)
        sequence = features.mean(dim=1)
        logits = self.classifier(sequence)
        return logits.permute(1, 0, 2).log_softmax(dim=2)
