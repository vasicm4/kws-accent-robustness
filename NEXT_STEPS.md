# Next steps

Working notes (state as of 12 Oct 2026, after the week-4/5 experiment batch).

## Done on 12 Oct

- Strided stem decided over 3 seeds: **not adopted** (−0.0036 val macro-F1,
  rule was ≤ 0.002). E0 = `channels_last`, 3 seeds; README result updated.
- E1 (3 seeds) and E2 (1 seed): no gain over E0; E2 is worse on official.
- LR sweep, 1 seed each: 3e-4 clearly worse, **3e-3 better** (see below).
- `evaluate.py`: confusion matrix and per-class report; `evaluate_accented.py`
  now also writes the 12-class report and `predictions.npz`.
- Paired speaker bootstrap: `scripts/compare_runs.py`.
- E3 few-shot fine-tuning: `scripts/finetune_accented.py`, +0.7 points on
  held-out accented speakers, −0.3 on official.
- ONNX export (front end inside the graph), parity test, latency benchmark,
  FastAPI app, Dockerfile (built and tested locally).
- Error-analysis notebook and figures in `reports/figures/`.
- The two weakest speakers are `spk09` (hand-cleaned) and `spk16` (not).

The run logs and orchestration scripts are in `runs/_logs/` (`phase1-3.sh`).

## Seed policy

Screen with 1 seed, confirm with 3. Single-seed noise is ~0.5 val macro-F1
points (test), ~0.1 on validation for E0. Model selection uses validation
macro-F1 only; the accented set is a test set.

## 1. Confirm lr 3e-3 (3 seeds, ~25 min)

One seed of lr 3e-3 beats E0 on val (0.9559 vs. 0.9494 ± 0.0010) and on the
paired official comparison (+0.6 points, CI [+0.07, +1.19]). Its best epoch
was 20, so it may also allow a shorter schedule.

```
for seed in 1 2; do uv run python src/train.py --set name=hp_lr3.0e-3 lr=3.0e-3 seed=$seed; done
for r in runs/*_hp_lr3.0e-3_*/; do uv run python scripts/evaluate_accented.py $r; done
uv run python scripts/compare_runs.py e0_channels_last hp_lr3.0e-3
```

If it holds, make `lr: 3.0e-3` the default, call it the final E0, and rerun
E3 on its 3 seeds (`scripts/finetune_accented.py`, ~10 min). E1 could be
rerun at 3e-3 too, since it still peaked at the last epoch at 1e-3.

## 2. Error analysis: listen

`notebooks/error_analysis.ipynb` section 2 lists the 20 accented clips every
seed gets wrong. Listen (`listen(path)`, then clear the output: never commit
accented audio) and fill in the notes cell: top failure mode, example,
hypothesis. The current hypothesis (README) is that `right`/`on`/`no` are
rejected as `_unknown_` because of the Serbian *r* and vowels.

## Later

- Write `reports/report.md` (outline in the project plan, section 7).
- Recruit more speakers (25–30): the main lever on the accented CI.
- Optional E3 control: the gain may partly be phone-codec adaptation.
- MLflow run `c5765431` (the interrupted 11 Oct run) is still marked
  RUNNING; mark it killed in the UI.
- Admin: ask referees this week; check the Eiffel deadline.
