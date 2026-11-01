"""Pixel confusion metrics and maximum-cardinality IoU instance matching."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from .common import CLASSES


def confusion_matrix(truth, pred, n=5):
    valid = (truth >= 0) & (truth < n)
    return np.bincount(
        (n * truth[valid] + pred[valid]).ravel(), minlength=n * n
    ).reshape(n, n)


def pixel_metrics(cm):
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1) - tp
    iou = np.divide(tp, denom, out=np.full(len(cm), np.nan), where=denom > 0)
    f1 = np.divide(
        2 * tp,
        cm.sum(0) + cm.sum(1),
        out=np.zeros(len(cm)),
        where=(cm.sum(0) + cm.sum(1)) > 0,
    )
    return {
        "pixel_accuracy": float(tp.sum() / max(1, cm.sum())),
        "mean_iou_foreground": float(np.nanmean(iou[1:])),
        "mean_iou_all": float(np.nanmean(iou)),
        "per_class": {
            name: {
                "iou": None if np.isnan(iou[i]) else float(iou[i]),
                "dice": float(f1[i]),
                "support_pixels": int(cm[i].sum()),
            }
            for i, name in enumerate(CLASSES)
        },
        "confusion_matrix": cm.tolist(),
    }


def instance_counts(truth_masks, pred_masks, threshold=0.5):
    ng, npred = len(truth_masks), len(pred_masks)
    if not ng or not npred:
        return np.array([0, npred, ng], dtype=int)
    ious = np.zeros((ng, npred))
    for i, g in enumerate(truth_masks):
        for j, p in enumerate(pred_masks):
            union = np.logical_or(g, p).sum()
            ious[i, j] = np.logical_and(g, p).sum() / max(1, union)
    # Cardinality is the primary objective; IoU breaks ties among eligible pairs.
    eligible = ious >= threshold
    scores = eligible * (min(ng, npred) + 1 + ious)
    rows, cols = linear_sum_assignment(-scores)
    tp = int(eligible[rows, cols].sum())
    return np.array([tp, npred - tp, ng - tp], dtype=int)


def count_metrics(counts):
    tp, fp, fn = map(int, counts)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }
