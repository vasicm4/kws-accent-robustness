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

## Next

The remaining time is in the model, so the next changes to measure are on the compute side: `torch.compile`, `channels_last` memory format, and the cost of the first full-resolution conv block.
