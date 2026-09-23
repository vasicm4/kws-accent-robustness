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
