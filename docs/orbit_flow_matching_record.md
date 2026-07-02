# Orbit Flow Matching Record

This document records orbit artifacts staged outside Git under `artifact-source/`. These files are release artifacts, not source files, and should be copied to the external artifact bundle only when the artifact registry marks them for public release.

## Directories

| Directory | Interpretation | Key files | Notes |
| --- | --- | --- | --- |
| `artifact-source/orbitize_training/` | Confirmed paper FUSE/MM-DiT orbit checkpoint run | `model_latest.pt`, `settings.yaml` | `model_latest.pt` is the paper `orbit_mmdit.pth` checkpoint. Saved FK sample arrays in this directory are intentionally not included in the first public artifact release. |
| `artifact-source/orbitize_fmpe/` | Candidate FMPE baseline | `best_model.pt`, `model_latest.pt`, `sample.npy`, `corner.png`, `settings.yaml` | Candidate baseline input for Figure 4 and Table 6. |

## Config Summary

| Directory | Model type | Architecture | Training |
| --- | --- | --- | --- |
| `orbitize_training` | `flow_matching_v2_with_v_pred_mcmc` | MM-DiT, hidden dim 256, depth 12, heads 8, context tokens 12, theta tokens 1 | 200 epochs, batch size 4096, lr 0.0005 cosine schedule |
| `orbitize_fmpe` | `flow_matching` | DenseResidualNet, hidden dims 32 to 1024 to 32 | 52 logged epochs, batch size 4096, lr 0.0002 reduce-on-plateau schedule |

## Selected Checksums

| File | Shape if `.npy` | SHA256 |
| --- | --- | --- |
| `orbitize_training/model_latest.pt` | publish as `artifacts/orbit/models/orbit_mmdit.pth` | `96d2b9a328a317c51cd45522821450224ee3ab68e35466a36f5b000c9fe074fa` |
| `orbitize_fmpe/model_latest.pt` |  | `e185c31bccf982aaa0f3acb0658034e3d9bef813311603ddcf119568f01968a3` |
| `orbitize_fmpe/best_model.pt` |  | `674ebddfe85b2ea89cbecd0e88b7a95ff3b93ba30b7018e035895c4a1f46a056` |
| `orbitize_fmpe/sample.npy` | `(4096, 8)` | `ad6a2b4f1ac2c46fe35c6433e460a4530d50e8c6106174bf81921950837515d8` |

## Candidate Mapping

| Paper need | Candidate file(s) | Confidence | Remaining check |
| --- | --- | --- | --- |
| Orbit FUSE checkpoint | `orbitize_training/model_latest.pt` | high | Confirmed by the author as the paper `orbit_mmdit.pth`; publish as `artifacts/orbit/models/orbit_mmdit.pth`. |
| Orbit FMPE baseline | `orbitize_fmpe/best_model.pt`, `orbitize_fmpe/model_latest.pt`, `orbitize_fmpe/sample.npy` | medium | Confirm whether the paper used `best_model.pt` or `model_latest.pt`. |
| Figure 5 FK samples | withheld | high | Final FK samples are intentionally not included in the first public artifact release. |
| Table 6 metrics | withheld | high | FK samples, PTMCMC reference, metric script, and saved IoU / mode-distance outputs are not included in the first public artifact release. |

## Sample Arrays

The public first release should not include the saved FK / Naive Best-of-N sample arrays from `orbitize_training/`. The confirmed FMPE `sample.npy` array has shape `(4096, 8)`.

## Remaining Questions

- Confirm whether the public artifact should include `orbitize_fmpe/best_model.pt`, `orbitize_fmpe/model_latest.pt`, or both.
- Keep Figure 5 FK samples, Naive Best-of-N samples, PTMCMC reference chain, and Table 6 metric source out of the first public release.
