from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != (1, 1) or in_channels != out_channels:
            self.downsample: nn.Module | None = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(output + residual)


class ResNet18Features(nn.Module):
    output_channels = 512
    width_reduction = 4

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                1, 64, kernel_size=7, stride=(2, 2), padding=3, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1),
        )
        self.layer1 = self._make_layer(64, 64, stride=(1, 1))
        self.layer2 = self._make_layer(64, 128, stride=(2, 1))
        self.layer3 = self._make_layer(128, 256, stride=(2, 1))
        self.layer4 = self._make_layer(256, 512, stride=(2, 1))

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int],
    ) -> nn.Sequential:
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.stem(inputs)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        return self.layer4(output)


def _compact_features() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 64, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2, 1)),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2, 1)),
    )


class CRNN(nn.Module):
    width_reduction = 4

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        backbone: str = "resnet18",
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must include blank and at least one label")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if backbone not in {"resnet18", "compact"}:
            raise ValueError("backbone must be 'resnet18' or 'compact'")
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.backbone_name = backbone
        if backbone == "resnet18":
            self.features: nn.Module = ResNet18Features()
            feature_size = ResNet18Features.output_channels
            self._round_lengths_up = True
        else:
            self.features = _compact_features()
            feature_size = 256
            self._round_lengths_up = False
        self.sequence = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def output_lengths(self, image_widths: torch.Tensor) -> torch.Tensor:
        if self._round_lengths_up:
            return torch.div(
                image_widths + self.width_reduction - 1,
                self.width_reduction,
                rounding_mode="floor",
            )
        return torch.div(image_widths, self.width_reduction, rounding_mode="floor")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images).mean(dim=2)
        sequence = features.permute(2, 0, 1)
        sequence, _ = self.sequence(sequence)
        return self.classifier(sequence).log_softmax(dim=2)
