# FUSE: FK-Steered Multi-Modal Flow Matching

This repository contains code, configuration files, and artifact documentation for **FUSE: FK-Steered Multi-Modal Flow Matching for Efficient Simulation-Based Posterior Estimation**.

Paper links:

- ICML 2026 poster: https://icml.cc/virtual/2026/poster/62622
- Project artifact record: Zenodo DOI to be added after upload

Additional arXiv, PMLR, DOI, and Zenodo links will be added when the public records are available.

## What is in this repository

- `dingo/`: Dingo-derived posterior-estimation code with MM-DiT / flow-matching additions used by the experiments.
- `flow-matching-posterior-estimation/`: SBIBM and gravitational-wave experiment scripts inherited from the FMPE workflow and adapted for FUSE experiments.
- `orbit_train/`: beta Pictoris b orbital-characterization data generation, training, MCMC, and plotting code.
- `REPRODUCING.md`: result-by-result reproduction map for the paper.
- `ARTIFACTS.md`: datasets, checkpoints, reference samples, and generated outputs that are stored outside Git.

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate fuse-artifact
```

If editable Dingo installation fails, install it manually:

```bash
cd dingo
pip install -e .
cd ..
```

## Artifacts

Large datasets, trained weights, posterior samples, and long-run reference chains are not committed to Git. They are tracked in `ARTIFACTS.md` and should be downloaded into the documented `artifacts/` paths after the Zenodo record is available.

Important current artifact status:

- Orbit datasets are external release artifacts and should be published under `artifacts/orbit/datasets/`.
- The paper orbit MM-DiT checkpoint should be published as `artifacts/orbit/models/orbit_mmdit.pth`; the private staging source is recorded outside the public repository.
- Figure 5 / Table 6 FK samples, PTMCMC reference chains, timing logs, and Table 6 metric outputs are not included in the first public artifact release.
- SBIBM scripts are present, and a 10-task FUSE checkpoint/result bundle is documented in `FUSE_checkpoint/README.md`; the zip itself is an external artifact and is not committed to Git.
- Gravitational-wave datasets and BBH checkpoints were identified during release preparation and are listed in `ARTIFACTS.md`, but their paper-result ownership still needs confirmation.

## Reproducing paper results

Start with:

```bash
cat REPRODUCING.md
```

The reproduction guide maps each paper figure/table to scripts, configs, required artifacts, expected outputs, and known gaps. The first release targets artifact-level reproducibility rather than a single command that reruns the full paper.

## Quick checks

After installing dependencies, run:

```bash
python -m unittest discover -s tests -v
python -m py_compile scripts/inventory_artifacts.py
```

SBIBM command help:

```bash
python flow-matching-posterior-estimation/sbi-benchmark/run_sbibm.py --help
python flow-matching-posterior-estimation/sbi-benchmark/evaluate_sbibm.py --help
```

Orbit syntax check:

```bash
python -m py_compile orbit_train/generate.py orbit_train/train_mmdit.py orbit_train/mcmc.py
python orbit_train/mcmc.py --help
```

`orbit_train/generate.py --help` and `orbit_train/train_mmdit.py --help` require `orbitize` to be installed because their current top-level imports load orbit helpers before `argparse`. The release environment includes `orbitize`; the existing remote `dingo` environment used during verification does not.

## License

This companion repository is released under the MIT license. Third-party code retains its upstream attribution and license notices.

## Citation

Use the FUSE paper citation when available. Until the final citation record is public, cite the ICML 2026 paper page linked above.
