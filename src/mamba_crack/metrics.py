from __future__ import annotations

import numpy as np
from skimage.measure import label
from skimage.morphology import skeletonize


def _components(mask: np.ndarray) -> int:
    return int(label(mask, connectivity=2).max())


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score.ravel())
    sorted_target = target.ravel()[order].astype(np.float64)
    positives = sorted_target.sum()
    if positives == 0:
        return 1.0
    cumulative = np.cumsum(sorted_target)
    precision = cumulative / np.arange(1, len(sorted_target) + 1)
    return float((precision * sorted_target).sum() / positives)


def betti_numbers(mask: np.ndarray) -> tuple[int, int]:
    mask = mask.astype(bool)
    beta_0 = _components(mask)
    background = ~mask
    padded = np.pad(background, 1, constant_values=True)
    beta_1 = max(_components(padded) - 1, 0)
    return beta_0, beta_1


def batch_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction_scores = prediction.astype(np.float32)
    prediction = prediction_scores >= 0.5
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    true_positive = intersection
    precision = true_positive / (prediction.sum() + 1e-8)
    recall = true_positive / (target.sum() + 1e-8)
    dice = 2 * true_positive / (prediction.sum() + target.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    pred_betti = betti_numbers(prediction)
    target_betti = betti_numbers(target)
    prediction_skeleton = skeletonize(prediction)
    target_skeleton = skeletonize(target)
    topology_precision = (prediction_skeleton & target).sum() / (prediction_skeleton.sum() + 1e-8)
    topology_recall = (target_skeleton & prediction).sum() / (target_skeleton.sum() + 1e-8)
    topology_cldice = 2 * topology_precision * topology_recall / (topology_precision + topology_recall + 1e-8)
    true_negative = np.logical_and(~prediction, ~target).sum()
    accuracy = (true_positive + true_negative) / prediction.size
    background_precision = true_negative / ((~prediction).sum() + 1e-8)
    background_recall = true_negative / ((~target).sum() + 1e-8)
    background_f1 = 2 * background_precision * background_recall / (background_precision + background_recall + 1e-8)
    return {
        "iou": float(iou),
        "dice": float(dice),
        "f1": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "map": average_precision(target, prediction.astype(np.float32)),
        "cldice": float(topology_cldice),
        "foreground_pixels": float(target.sum()),
        "background_precision": float(background_precision),
        "background_recall": float(background_recall),
        "background_f1": float(background_f1),
        "foreground_precision": float(precision),
        "foreground_recall": float(recall),
        "foreground_f1": float(dice),
        "betti_error": float(abs(pred_betti[0] - target_betti[0]) + abs(pred_betti[1] - target_betti[1])),
    }