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

## Done on 13 Oct

- lr 3e-3 confirmed on 3 seeds (`runs/_logs/phase4.sh`): val macro-F1
  0.9559 ± 0.0031 vs. 0.9494 ± 0.0010, official acc +0.8 points (CI excludes
  0), accented and gap unchanged within CI. Now the config default and the
  final model.
- E3 rerun on the 3e-3 seeds (`runs/_logs/e3_finetune_lr3e-3.out`): +1.0
  (k=1) / +1.1 (k=3) points on held-out speakers, 9 of 9 runs improve, no
  official cost.
- Figures and the error-analysis notebook regenerated on the final model;
  README and `reports/report.md` updated.
- Not rerun at 3e-3: E1, E2 (stated as a limitation).

## 2. Error analysis: listen

`notebooks/error_analysis.ipynb` section 2 lists the 18 accented clips every
seed gets wrong (final model). Listen (`listen(path)`, then clear the output: never commit
accented audio) and fill in the notes cell: top failure mode, example,
hypothesis. The current hypothesis (README) is that `right`/`on`/`no` are
rejected as `_unknown_` because of the Serbian *r* and vowels.

## Later

- Report draft done; fill the `TODO(listen)` in section 5 after listening.
- Recruit more speakers (25–30): the main lever on the accented CI.
- Optional E3 control: the gain may partly be phone-codec adaptation.
- MLflow run `c5765431` (the interrupted 11 Oct run) is still marked
  RUNNING; mark it killed in the UI.
- Admin: ask referees this week; check the Eiffel deadline.
