# Next steps

Working notes for the week-4 experiments (state as of 11 Oct 2026).

## Seed policy

Screen with 1 seed, confirm with 3. Single-seed noise is ~0.5 val macro-F1
points, so a difference smaller than that needs 3 seeds to call.

- 1 seed: augmentation settings, the first pass of the LR sweep, sanity checks.
- 3 seeds (0, 1, 2): only configs that go in the final results table
  (E0, the best of E1/E2, E3).
- On the accented set, the speaker bootstrap (±4 pts with 19 speakers)
  dominates seed noise; more seeds don't shrink it, more speakers do.

## 1. Decide the strided stem (E0, 3 seeds) — interrupted, redo

Seed 0 of both variants exists (`runs/20261007-*`). Seeds 1 and 2 are missing.
The `channels_last` seed-1 run was stopped at epoch 29/30
(`runs/20261011-232059_e0_channels_last_seed1`, no `metrics.json`): delete it
and rerun. About 40 min, run sequentially so epoch times stay comparable:

```
for seed in 1 2; do
  uv run python src/train.py --set name=e0_channels_last seed=$seed
  uv run python src/train.py --set name=e0_strided_stem seed=$seed strided_stem=true
done
for r in runs/2026*_e0_*/; do uv run python scripts/evaluate_accented.py $r; done
uv run python scripts/summarize_runs.py e0_channels_last e0_strided_stem
```

Make `strided_stem: true` the default only if its mean val macro-F1 holds.
The winner's 3 seeds are the E0 row of the results table; update the README
result line from them (the current one predates `channels_last`).

## 2. Screen E1 and E2 (1 seed each, on the chosen stem)

```
uv run python src/train.py --set augment=true name=e1_shift_noise
uv run python src/train.py --set augment=true spec_freq_masks=2 spec_time_masks=2 name=e2_specaug
```

Add `strided_stem=true` to both if step 1 picks it. ~10–12 min each.
Check the first epochs' data time: augmentation runs in the workers and
should stay well under 1 s/epoch. Confirm the better one with seeds 1 and 2
only if it beats E0 by more than the seed noise.

## 3. Paired comparison between models

`evaluate_accented.py` gives independent CIs per model. To compare two models
(E1 vs. E0, E3 vs. E0), bootstrap the *difference* on the same resampled
speakers: much tighter than comparing two independent CIs. Reuse
`speaker_bootstrap` / `group_confusions` in `src/evaluate.py`.

## Later this week

- LR sweep (3e-4, 1e-3, 3e-3), 1 seed each.
- `src/evaluate.py`: confusion matrix and per-class F1 on the official set;
  error-analysis notebook.
- Check whether the two weakest accented speakers (0.60, 0.67) are among the
  hand-cleaned recordings `spk07`–`spk11`.
- Recruit more speakers if possible (25–30): the main lever on the accented CI.
- Admin: ask referees this week; check the Eiffel deadline.
