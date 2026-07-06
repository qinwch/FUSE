# Release Verification

Date: 2026-07-06
Branch: `main`
Code verified at commit: `abc2812` before cleanup commit; generated artifact cleanup is verified by GitHub tree audit.
Verification note updated after project-page preservation, branch pruning, and generated-artifact cleanup.

## Checks

| Check | Status | Evidence |
| --- | --- | --- |
| Public release guard | pass | `python3 scripts/check_public_release.py` printed `public release guard passed`. |
| Unit tests | pass | `PYTHONDONTWRITEBYTECODE=1 /opt/conda/bin/conda run -n dingo python -B -m unittest discover -s tests -v` ran 32 tests with `OK`. |
| Python syntax checks | pass | `PYTHONDONTWRITEBYTECODE=1 /opt/conda/bin/conda run -n dingo python -B -m py_compile` completed with exit code 0 for release inventory, public guard, SBIBM helper scripts, and orbit entrypoints. |
| SBIBM files tracked | pass | `git ls-files flow-matching-posterior-estimation/sbi-benchmark` includes `c2st.py`, `evaluate_sbibm.py`, `repro_utils.py`, `run_sbibm.py`, and `settings_MLP.yaml`. |
| Orbit files tracked | pass | `git ls-files orbit_train` includes `generate.py`, `train_mmdit.py`, `mcmc.py`, `path_config.py`, `helpers/mmdit.py`, `helpers/training_mmdit.py`, and `plots/plots.py`. |
| SBIBM command surface help | pass | `run_sbibm.py --help` and `evaluate_sbibm.py --help` exited 0; each help output contains one `usage:` line. |
| Orbit command surface help | pass | `orbit_train/mcmc.py --help` exited 0 and its help output contains one `usage:` line. |
| Private editor/planning paths | pass | `git ls-files` returned no tracked files for `.vscode/launch.json`, `OPEN_QUESTIONS` `.md`, or `docs/superpowers/`. |
| Pre-push sensitive literal audit | pass | Targeted `git grep` for the removed real private credential, SSH alias, and machine paths returned no output. Guard tests now use synthetic fixture strings. |

## Known Untracked Items

These files/directories remain in the remote working tree and were not staged:

- `FUSE_checkpoint/FUSE_checkpoint.zip`
- `asd/`
- `dingo/docs/source/asimov.md`
- `dingo/docs/source/example_gnpe_model.md`
- `dingo/docs/source/example_npe_model.md`
- `dingo/docs/source/example_toy_npe_model.md`
- `dingo/tests/gw/transforms/waveform_data.npy`

## Not Run

- Full SBIBM 10-task training/evaluation.
- SLCP FK-steering posterior generation.
- Beta Pictoris b model training.
- PTMCMC reference generation.
- Orbit Figure 4 assembly, normalized Sinkhorn heatmap, Figure 5, Table 3, and Table 6 generation.
- Zenodo upload.
- Release-environment verification of `orbit_train/generate.py --help` and `orbit_train/train_mmdit.py --help` was not run.

## Current Known Gaps

Withheld or collaborator-owned artifacts are not included in the first public artifact release. Private triage notes remain outside Git.

| Generated artifact tree audit | pass | Public `main` cleanup removes tracked `diffusions-for-sbi/results/` files and `flow-matching-posterior-estimation/mmdit_trainning/` training outputs; the public guard now flags these prefixes if they reappear. |
| GitHub branch pruning | pass | GitHub branch refs were pruned so only `main` remains. Project-page commits on `main` are intentionally retained. |
