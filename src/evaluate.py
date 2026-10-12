from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


@torch.inference_mode()
def predict(
    model: nn.Module, loader: DataLoader, criterion: nn.Module
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    for features, labels in loader:
        logits = model(features)
        total_loss += criterion(logits, labels).item() * len(labels)
        y_true.append(labels)
        y_pred.append(logits.argmax(dim=1))
    y_true_arr = torch.cat(y_true).numpy()
    y_pred_arr = torch.cat(y_pred).numpy()
    return y_true_arr, y_pred_arr, total_loss / len(y_true_arr)


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, n_classes: int
) -> dict[str, float]:
    y_true, y_pred, loss = predict(model, loader, criterion)
    macro_f1 = f1_score(
        y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0
    )
    return {
        "loss": loss,
        "acc": float((y_true == y_pred).mean()),
        "macro_f1": float(macro_f1),
    }


def group_confusions(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, n_classes: int
) -> np.ndarray:
    """One confusion matrix per group (e.g. speaker): shape (n_groups, n_classes, n_classes)."""
    _, group_ids = np.unique(groups, return_inverse=True)
    n_groups = group_ids.max() + 1
    flat = (group_ids * n_classes + y_true) * n_classes + y_pred
    counts = np.bincount(flat, minlength=n_groups * n_classes * n_classes)
    return counts.reshape(n_groups, n_classes, n_classes)


def confusion_metrics(
    cm: np.ndarray, labels: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Accuracy and macro-F1 over `labels` for one or a stack of confusion matrices.

    cm[..., true, pred]. Macro-F1 matches sklearn's
    f1_score(labels=labels, average="macro", zero_division=0).
    """
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    acc = tp.sum(axis=-1) / cm.sum(axis=(-2, -1))
    # 2TP + FP + FN = (predicted as l) + (truly l)
    denom = (cm.sum(axis=-2) + cm.sum(axis=-1))[..., labels]
    f1 = np.divide(
        2 * tp[..., labels], denom, out=np.zeros(denom.shape), where=denom > 0
    )
    return acc, f1.mean(axis=-1)


def per_class_report(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> dict[str, list]:
    """Confusion matrix (rows = true, columns = predicted) and per-class
    precision, recall and F1. Classes with no support or no predictions get 0."""
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes)
    cm = cm.reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    predicted, actual = cm.sum(axis=0), cm.sum(axis=1)
    precision = np.divide(tp, predicted, out=np.zeros(n_classes), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros(n_classes), where=actual > 0)
    denom = predicted + actual
    f1 = np.divide(2 * tp, denom, out=np.zeros(n_classes), where=denom > 0)
    return {
        "confusion": cm.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": actual.tolist(),
    }


def _speaker_weights(n_speakers: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """How many times each speaker is drawn, per bootstrap sample: (n_boot, n_speakers)."""
    return rng.multinomial(n_speakers, np.full(n_speakers, 1 / n_speakers), size=n_boot)


def speaker_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    speakers: np.ndarray,
    n_classes: int,
    labels: list[int],
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Bootstrap accuracy and macro-F1 by resampling whole speakers with replacement.

    Clips from one speaker are not independent, so resampling clips would
    understate the uncertainty. Returns n_boot draws of each metric.
    """
    cms = group_confusions(y_true, y_pred, speakers, n_classes)
    weights = _speaker_weights(len(cms), n_boot, rng)
    boot_cms = np.einsum("bs,skl->bkl", weights, cms)
    acc, macro_f1 = confusion_metrics(boot_cms, labels)
    return {"acc": acc, "macro_f1": macro_f1}


def paired_speaker_bootstrap(
    y_true: np.ndarray,
    y_preds: list[np.ndarray],
    speakers: np.ndarray,
    n_classes: int,
    labels: list[int],
    n_boot: int,
    rng: np.random.Generator,
) -> list[dict[str, np.ndarray]]:
    """speaker_bootstrap for several models scored on the same clips, with the
    same resampled speakers in every draw for all of them.

    Differences between models are then paired: a speaker that is hard for
    every model moves all of them together and cancels out, so the CI of the
    difference is much tighter than one built from two independent CIs.
    """
    cms = [group_confusions(y_true, y_pred, speakers, n_classes) for y_pred in y_preds]
    weights = _speaker_weights(len(cms[0]), n_boot, rng)
    draws = []
    for cm in cms:
        acc, macro_f1 = confusion_metrics(np.einsum("bs,skl->bkl", weights, cm), labels)
        draws.append({"acc": acc, "macro_f1": macro_f1})
    return draws
