# Profiling one training epoch (week 3)

E0 setup (batch 128, 289 steps/epoch, 36,923 training clips), CPU only, AMD Ryzen 7 7735HS (8 cores / 16 threads), 26 GB RAM.
All numbers come from `scripts/profile_epoch.py`; the raw outputs are the `.txt` files in this folder.

## Hypothesis

The plan assumed that on-the-fly feature extraction (WAV decode + log-mel in `__getitem__`) was the bottleneck.

## Findings

| Measurement | Result | Source |
|---|---|---|
| Time blocked waiting for data, 4 workers | 0.5–0.8 s of a ~29–31 s epoch (**~2%**) | `cprofile_profile_baseline.txt`, `thread_sweep.txt` |
| Step breakdown (50 steps) | backward **63%**, forward 35%, data 1.4%, optimizer 1.1% | `torch_profile_baseline.txt` |
| Top operators by self time | `convolution_backward` 28%, `mkldnn_convolution` 15%, batch-norm backward 11%, max-pool 8% | `torch_profile_baseline.txt` |
| Data pipeline in the main process (`num_workers=0`) | 17.9 s/epoch: log-mel 11.2 s, soundfile read 6.0 s | `cprofile_profile_baseline_workers0.txt` |

**The hypothesis was wrong for the steady state.** With 4 workers the data pipeline (~0.5 ms/clip once the 3.3 GB dataset is in the OS page cache) runs in parallel with training and almost never stalls it. The epoch is bound by model compute, mostly the convolution backward pass. Data loading only matters on a cold cache (first epoch after a reboot: ~8 ms/clip, dominated by disk reads).

## Thread × worker sweep (epoch 2, seconds)

| `num_threads` | 2 workers | 4 workers |
|---|---|---|
| 2 | 54.4 | 50.4 |
| 4 | 33.4 | 32.6 |
| 6 | **27.5** | 28.1 |
| 8 (PyTorch default) | 28.3 | 28.7 |

Using fewer than 8 threads is clearly slower. 6 threads is ~3% faster than the default, which is within run-to-run noise from a single run per cell, so the default (`num_threads: null`) is kept. Thread tuning is not the fix.

## Fixing the compute bottleneck

Breaking the forward pass down by block showed the first block taking ~29 of ~44 ms per batch, despite doing ~4% of the FLOPs: it is the only block that computes a full-resolution 16×40×101 map, so its BN, ReLU and max-pool are memory-bound. Candidates, timed on the training step alone (scratch benchmark, 128-clip batches):

| Change | Step time | Kept? |
|---|---|---|
| Eager baseline | 92 ms | — |
| `torch.compile` | 78 ms | No: 12 s compile, no gain on top of `channels_last`, and it wraps the model (checkpoint keys, ONNX export) |
| `channels_last` weights | 72 ms | Yes (`channels_last: true`) |
| Stride-2 conv in block 1 instead of conv + max-pool | 44 ms (35 ms with `channels_last`) | Opt-in (`strided_stem: true`); changes the model |

Full epochs, two rounds of 3 epochs each (`compute_sweep.txt`, mean of epochs 2–3):

| Setup | Epoch | vs. baseline |
|---|---|---|
| Baseline | 29.3 s | — |
| `channels_last` | 24.2 s | −17% |
| strided stem | 20.6 s | −30% |
| both | **17.7 s** | **−40%** |

## Accuracy check (full 30-epoch runs, seed 0)

| Run | MLflow run | Median epoch | Training time | val macro-F1 | test acc | test macro-F1 |
|---|---|---|---|---|---|---|
| E0 baseline (23 Sep) | `6e4f52f7` | 33.5 s | 16.8 min | 0.9490 | 0.9527 | 0.9527 |
| + `channels_last` | `a1fde9e9` | 22.3 s | 11.2 min | 0.9491 | 0.9482 | 0.9480 |
| + `channels_last` + strided stem | `ee517c63` | 16.0 s | 8.0 min | 0.9464 | 0.9476 | 0.9475 |

`channels_last` computes the same function in a different memory order, yet test macro-F1 moved by 0.47 points: that is the run-to-run noise floor for a single seed. The strided stem is within that noise (−0.27 val, −0.05 test against the `channels_last` run), but one seed per config cannot show a small real drop, so it stays opt-in until it is compared over 3 seeds.

Net result: an E0 training run went from 16.8 to 11.2 min with no model change, or 8.0 min with the strided stem.
