from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from evaluate import confusion_metrics, group_confusions, speaker_bootstrap

N_CLASSES = 12
TARGETS = list(range(10))


def test_confusion_metrics_match_sklearn() -> None:
    rng = np.random.default_rng(0)
    # True labels are targets only, predictions can be any class, as in the
    # accented evaluation; a few targets are never predicted nor present.
    y_true = rng.integers(0, 8, size=500)
    y_pred = np.where(rng.random(500) < 0.7, y_true, rng.integers(0, N_CLASSES, size=500))
    speakers = rng.integers(0, 20, size=500)

    cm = group_confusions(y_true, y_pred, speakers, N_CLASSES).sum(axis=0)
    acc, macro_f1 = confusion_metrics(cm, TARGETS)

    assert acc == pytest.approx((y_true == y_pred).mean())
    assert macro_f1 == pytest.approx(
        f1_score(y_true, y_pred, labels=TARGETS, average="macro", zero_division=0)
    )


def test_bootstrap_resamples_whole_speakers() -> None:
    # Every speaker is either all right or all wrong, with 10 clips each, so
    # any speaker-level draw has an accuracy that is a multiple of 1/n_speakers.
    # A clip-level bootstrap would produce values in between.
    n_speakers, clips = 4, 10
    speakers = np.repeat(np.arange(n_speakers), clips)
    y_true = np.zeros(n_speakers * clips, dtype=int)
    y_pred = np.where(speakers < 2, 0, 11)

    draws = speaker_bootstrap(
        y_true, y_pred, speakers, N_CLASSES, TARGETS, n_boot=500,
        rng=np.random.default_rng(0),
    )

    assert draws["acc"].shape == (500,)
    np.testing.assert_allclose(draws["acc"] * n_speakers, np.round(draws["acc"] * n_speakers))
    assert draws["acc"].min() == 0.0 and draws["acc"].max() == 1.0
    assert draws["acc"].mean() == pytest.approx(0.5, abs=0.05)


def test_bootstrap_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 10, size=200)
    y_pred = rng.integers(0, N_CLASSES, size=200)
    speakers = rng.integers(0, 15, size=200)

    a = speaker_bootstrap(y_true, y_pred, speakers, N_CLASSES, TARGETS, 100, np.random.default_rng(3))
    b = speaker_bootstrap(y_true, y_pred, speakers, N_CLASSES, TARGETS, 100, np.random.default_rng(3))

    np.testing.assert_array_equal(a["macro_f1"], b["macro_f1"])
