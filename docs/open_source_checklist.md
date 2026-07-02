# Open Source Release Checklist

## Repository contents

- [x] `README.md` links to the paper, reproduction guide, artifact registry, license, and citation.
- [x] `REPRODUCING.md` maps every paper figure and table to scripts, configs, artifacts, outputs, and known gaps.
- [x] `ARTIFACTS.md` lists discovered datasets, checkpoints, generated samples, logs, and reference outputs.
- [x] Private release triage notes are kept out of Git and record withheld collaborator-owned files and unresolved release issues.
- [x] `.gitignore` excludes datasets, checkpoints, generated results, caches, and experiment logs.
- [x] `LICENSE` is present.
- [x] Tracked `__pycache__/*.pyc` files are removed from the Git index before publication.
- [ ] Decide whether to commit or replace `dingo/tests/gw/transforms/waveform_data.npy`.
- [ ] Third-party attribution from Dingo, FMPE, SBIBM, orbitize, Lampe, and Zuko is preserved.

## Artifact release

- [ ] Zenodo record exists.
- [ ] Zenodo DOI is added to `README.md`, `REPRODUCING.md`, and `ARTIFACTS.md`.
- [ ] Artifact checksums are computed on uploaded files.
- [x] Required artifacts are marked as present, missing, or collaborator-owned.
- [x] Missing or withheld orbit FK samples, Naive Best-of-N samples, PTMCMC chain, timing logs, and Table 6 metrics are explicitly left out of the first public release.
- [x] Full SBIBM 10-task checkpoints/results are obtained, regenerated, or explicitly left as long-run reproduction steps.

## Verification

- [x] Python unit tests pass.
- [x] Changed scripts pass `python -m py_compile`.
- [x] README and reproduction commands have been checked with `--help` or dry-run where possible.
- [ ] Verify `orbit_train/generate.py --help` and `orbit_train/train_mmdit.py --help` in the release `fuse-artifact` environment, not only the existing remote `dingo` environment.
- [x] Long-running steps are clearly marked and not implied to be quick.
- [x] `git status --short` shows no accidentally staged large artifacts.
