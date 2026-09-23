from __future__ import annotations

import torch
import torch.nn as nn

from model import KWSNet


def test_output_shape() -> None:
    model = KWSNet()
    out = model(torch.randn(4, 1, 40, 101))
    assert out.shape == (4, 12)


def test_parameter_count_in_planned_range() -> None:
    n_params = sum(p.numel() for p in KWSNet().parameters() if p.requires_grad)
    assert 50_000 < n_params < 300_000


def test_overfits_one_batch() -> None:
    torch.manual_seed(0)
    model = KWSNet(dropout=0.0)
    features = torch.randn(16, 1, 40, 101)
    labels = torch.arange(16) % 12
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(100):
        optimizer.zero_grad()
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.05
