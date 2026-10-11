# kws-accent-robustness
Measuring and reducing the accuracy gap of a keyword-spotting CNN on Serbian-accented speech.

# Results
`
best epoch 27 | val_acc 0.9489 | val_f1 0.9490 | test_acc 0.9527 | test_f1 0.9527
MLflow run 6e4f52f76bfa4a4d9edb8cf569a9b127, git 3624e67
`

## Accent gap (E0 baseline)

Scored on the 10 target words only, since the accented set has no
silence/unknown clips. Accented set: 19 speakers, 510 clips
([data_accented/README.md](data_accented/README.md)).

| Test set | Clips | Accuracy | Macro-F1 |
|---|---|---|---|
| Speech Commands test (10 words) | 4074 | 0.9553 | 0.9614 |
| Serbian-accented | 510 | 0.9353 | 0.9574 |
| **Gap** | | **0.0200** | **0.0040** |

- Most errors on accented speech are target words rejected as `_unknown_`
  (23 of 33), not confused with another word.
- Hardest words: `no` and `right` (0.86 each vs. 0.97 and 0.96 on Speech
  Commands), then `on` (0.90 vs. 0.95).
- Speakers vary much more than the average suggests: 10 of 19 at ≥ 0.97, two at
  0.60 and 0.67. With 10–30 clips per speaker these are rough estimates.

The two E0 speed variants (same model, seed 0) give the same picture; the
same two speakers are the weakest in all three:

| Run | Official acc | Accented acc | Acc gap | F1 gap |
|---|---|---|---|---|
| e0_baseline | 0.9553 | 0.9353 | 0.0200 | 0.0040 |
| e0_channels_last | 0.9531 | 0.9392 | 0.0139 | 0.0004 |
| e0_strided_stem | 0.9519 | 0.9392 | 0.0127 | 0.0011 |

```
uv run python scripts/evaluate_accented.py runs/<run dir>
```
