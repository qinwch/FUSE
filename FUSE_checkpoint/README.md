# FUSE Checkpoint Bundle

`FUSE_checkpoint.zip` is an external artifact bundle staged under `artifact-source/FUSE_checkpoint/` during release preparation. It should be uploaded to the external artifact release, not committed to Git.

## File

| File | Size | SHA256 |
| --- | ---: | --- |
| `FUSE_checkpoint/FUSE_checkpoint.zip` | 1,517,330,306 bytes | `fb40197fc6a3864a74fd73cc543be4c3eea08b4a05a0c08f7a81c5daf91b772b` |

## Coverage

The archive contains SBIBM FUSE outputs for these tasks:

- `bernoulli_glm`
- `bernoulli_glm_raw`
- `gaussian_linear`
- `gaussian_linear_uniform`
- `gaussian_mixture`
- `lotka_volterra`
- `sir`
- `slcp`
- `slcp_distractors`
- `two_moons`

Simulation budgets:

| Archive directory | Budget | Coverage |
| --- | ---: | --- |
| `FUSE_checkpoint/FUSE_10_3/<task>/` | 1,000 | checkpoints and metric CSVs are present; `posterior_samples_*.npy` files are absent |
| `FUSE_checkpoint/FUSE_10_4/<task>/` | 10,000 | checkpoints, metric CSVs, posterior plots, and `posterior_samples_*.npy` are present |
| `FUSE_checkpoint/FUSE_10_5/<task>/` | 100,000 | checkpoints, metric CSVs, posterior plots, and `posterior_samples_*.npy` are present |

Common files per covered task include:

- `model_latest.pt`
- `settings.yaml`
- `history.txt`
- `c2st.csv`
- `sinkhorn.csv`
- `kl.csv`
- `mmd.csv`
- `meddist.csv`
- `pme.csv`
- `pvr.csv`
- `inference_time.csv`
- `posterior_1.png` through `posterior_10.png`
- `posterior_samples_*.npy` for `FUSE_10_4` and `FUSE_10_5`

The `slcp` directories also include `corner_plot_obs_*.png` files.

## Intended Public Layout

When preparing the external artifact bundle, map the archive paths to the public layout used by `ARTIFACTS.md` and `REPRODUCING.md`:

| Archive path | Intended public path |
| --- | --- |
| `FUSE_checkpoint/FUSE_10_3/<task>/` | `artifacts/sbibm/fuse/<task>/1000/` |
| `FUSE_checkpoint/FUSE_10_4/<task>/` | `artifacts/sbibm/fuse/<task>/10000/` |
| `FUSE_checkpoint/FUSE_10_5/<task>/` | `artifacts/sbibm/fuse/<task>/100000/` |

Open release questions:

- The archive stores `model_latest.pt`, while earlier release notes expected `best_model.pt`. Confirm whether `model_latest.pt` is the paper checkpoint before renaming or documenting it as the canonical file.
- `FUSE_10_3` does not contain `posterior_samples_*.npy`; regenerate these if downstream scripts need sample arrays for the 1,000-simulation budget.
- FK-steered SLCP posterior samples, SLCP ablation outputs, and orbit artifacts are not covered by this bundle.
