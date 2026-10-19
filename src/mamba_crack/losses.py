from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def soft_dice(prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    intersection = (prediction * target).sum(dim=(1, 2, 3))
    denominator = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * intersection + smooth) / (denominator + smooth)).mean()


def soft_erode(values: torch.Tensor) -> torch.Tensor:
    horizontal = -F.max_pool2d(-values, (1, 3), stride=1, padding=(0, 1))
    vertical = -F.max_pool2d(-values, (3, 1), stride=1, padding=(1, 0))
    return torch.minimum(horizontal, vertical)


def soft_open(values: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(soft_erode(values), 3, stride=1, padding=1)


def soft_skeletonize(values: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    skeleton = F.relu(values - soft_open(values))
    eroded = values
    for _ in range(iterations):
        eroded = soft_erode(eroded)
        opened = soft_open(eroded)
        skeleton = skeleton + F.relu(eroded - opened)
    return skeleton


def cldice(prediction: torch.Tensor, target: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    prediction_skeleton = soft_skeletonize(prediction, iterations)
    target_skeleton = soft_skeletonize(target, iterations)
    precision = (prediction_skeleton * target).sum(dim=(1, 2, 3)) / (prediction_skeleton.sum(dim=(1, 2, 3)) + 1e-6)
    sensitivity = (target_skeleton * prediction).sum(dim=(1, 2, 3)) / (target_skeleton.sum(dim=(1, 2, 3)) + 1e-6)
    return (2 * precision * sensitivity / (precision + sensitivity + 1e-6)).mean()


class TopologyLoss(nn.Module):
    def __init__(self, use_cldice: bool = True) -> None:
        super().__init__()
        self.use_cldice = use_cldice

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = 1 - soft_dice(prediction, target)
        if not self.use_cldice:
            return dice_loss
        return 0.5 * dice_loss + 0.5 * (1 - cldice(prediction, target))