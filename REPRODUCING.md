# Reproducing FUSE Paper Results

This guide maps each paper result to the current code, configuration files, required artifacts, expected outputs, and known limitations. It targets artifact-level reproducibility: public users should be able to run the documented code paths after installing the environment and downloading the artifact bundle.

Status labels:

- `code ready`: script path and command-line interface are present.
- `requires artifacts`: code path exists, but datasets, checkpoints, posterior samples, or reference chains must be downloaded or supplied.
- `requires long run`: command is documented but expected to require long training, large sweeps, or multi-hour MCMC.
- `needs collaborator artifact`: the release-preparation workspace does not contain the exact saved file needed to reproduce the paper panel/table.
- `needs script confirmation`: the release-preparation workspace does not expose the exact plotting, FK-sampling, metric, or timing script.

## Environment

```bash
conda env create -f environment.yml
conda activate fuse-artifact
```

The release-preparation verification pass also used an existing `dingo` Conda environment for SBIBM unit and help checks. That environment does **not** currently contain `orbitize`, so `orbit_train/generate.py --help` and `orbit_train/train_mmdit.py --help` fail there before `argparse`. The release `environment.yml` includes `orbitize`, `lampe`, `zuko`, and `sbibm`.

## Result Map

| Paper result | Status | Script / entry point | Config | Required artifacts | Expected outputs | Current gap |
| --- | --- | --- | --- | --- | --- | --- |
| Figure 2, Table 1: SBIBM benchmark over 10 tasks | code ready, requires artifacts or long run | `flow-matching-posterior-estimation/sbi-benchmark/run_sbibm.py`, `flow-matching-posterior-estimation/sbi-benchmark/evaluate_sbibm.py` | `flow-matching-posterior-estimation/sbi-benchmark/settings_MLP.yaml`, `flow-matching-posterior-estimation/sbi-benchmark/settings_MMDiT.yaml`; edit `task.name` and `task.num_train_samples` per task/budget | Optional cached datasets under `artifacts/sbibm/<task>/<budget>/`; trained checkpoints under `artifacts/sbibm/fuse/<task>/<budget>/`; external bundle documented in `FUSE_checkpoint/README.md` | `best_model.pt`, `model_latest.pt`, `c2st.csv`, `results.csv`, `posterior_*.png`, per-observation samples | `FUSE_checkpoint.zip` contains `model_latest.pt`, metric CSVs, plots, and posterior samples for all 10 tasks at `1e5`; it does not contain `best_model.pt` or consolidated `results.csv` by those filenames. |
| Figure 3: SLCP FK-steering posterior comparison | needs collaborator artifact, needs script confirmation | SBIBM SLCP training uses `run_sbibm.py`; exact FK sampler / plotting entry point not found | SLCP MM-DiT config derived from `flow-matching-posterior-estimation/sbi-benchmark/settings_MMDiT.yaml` | SLCP FUSE checkpoint, unsteered samples, FK-steered samples, baseline samples | Corner plot comparing NPE, FMPE, Simformer, FUSE without FK, and FUSE | FK sample-generation and plotting script not found in release-preparation workspace. |
| Table 2: SLCP ablation | code ready for base training, requires long run, needs config confirmation | `run_sbibm.py`, `evaluate_sbibm.py` | Ablation variants of hidden dimension, individual tokenization, and tokens per theta | Optional cached SLCP datasets and per-variant checkpoints | C2ST, KL, and inference time per configuration | Exact ablation config files are not present as separate files; values must be reconstructed from the paper or collaborator notes. |
| Figure 4: beta Pictoris b posterior reconstruction | code ready for data/train/reference generation, requires artifacts, needs script confirmation | `orbit_train/generate.py`, `orbit_train/train_mmdit.py`, `orbit_train/mcmc.py`; partial plotting helpers in `orbit_train/plots/plots.py` | Orbit defaults in `orbit_train/path_config.py`; paper checkpoint settings are summarized in `docs/orbit_flow_matching_record.md` | Orbit datasets, confirmed FUSE checkpoint, FMPE baseline candidate, PTMCMC reference chain | Posterior corner plot and normalized Sinkhorn heatmap | The paper FUSE checkpoint should be released as `artifacts/orbit/models/orbit_mmdit.pth`; NPE, FUSE-without-FK, PTMCMC reference chain, full Figure 4 assembly script, and Sinkhorn heatmap script are not included in the first public artifact release. |
| Figure 5: FK vs Naive Best-of-N | code ready for base orbit training, withheld artifacts | Orbit training/reference scripts exist; exact FK and Naive Best-of-N generation script is not part of the first release | FK settings from paper: 8 particles, resampling every 5 steps, step window 20 to 200, noise scale 0.3 | FUSE checkpoint; FK and Naive Best-of-N samples are withheld | Corner plot comparing FK and final-step selection | Figure 5 saved FK / Naive Best-of-N samples are not included in the first public artifact release. |
| Table 3: FK inference overhead | needs collaborator artifact, needs script confirmation | Timing script or logs not found | FK particle/scoring-step settings | Timing logs or reproducible timing command | Runtime table for particles/scoring steps | Raw timing source was not found. |
| Table 4: SBIBM task-wise l-C2ST | code ready, requires artifacts or long run | `evaluate_sbibm.py` | Task-specific SBIBM settings | Saved trained models or rerun outputs for SIR, SLCP, LV | Per-observation `results.csv` summaries | `FUSE_checkpoint.zip` contains SIR, SLCP, and LV checkpoints plus `c2st.csv`; consolidated `results.csv` files are not present by that filename. |
| Table 5: SLCP simulation-budget sweep | code ready, requires artifacts or long run | `run_sbibm.py`, `evaluate_sbibm.py` | SLCP configs with 1e3, 1e4, and 1e5 simulations | Optional cached datasets and checkpoints per budget; external bundle documented in `FUSE_checkpoint/README.md` | l-C2ST mean and standard deviation per budget | `FUSE_checkpoint.zip` contains SLCP checkpoints and metric CSVs for `1e3`, `1e4`, and `1e5`; `1e3` posterior sample arrays are absent. |
| Table 6: beta Pictoris b IoU and posterior mode distance | withheld artifacts, needs script confirmation | Orbit plotting helpers exist; metric script not found | Orbit artifact paths in `orbit_train/path_config.py` | Posterior samples for FMPE, NPE, FUSE without FK, FUSE, and PTMCMC reference | 95% credible-region IoU and mode L2 distance | Table 6 FK samples, PTMCMC reference, metric script, and saved metric outputs are not included in the first public artifact release. |

## Confirmed Dataset and Checkpoint Locations

Known release-preparation files are listed in `ARTIFACTS.md`. The most important confirmed paths are:

- Orbit datasets: `orbit_train/datasets/orbit-train.h5`, `orbit_train/datasets/orbit-val.h5`, `orbit_train/datasets/orbit-test.h5`.
- Intended public orbit dataset paths: `artifacts/orbit/datasets/orbit-train.h5`, `artifacts/orbit/datasets/orbit-val.h5`, `artifacts/orbit/datasets/orbit-test.h5`.
- Confirmed paper orbit checkpoint: `artifact-source/orbitize_training/model_latest.pt`; intended public path `artifacts/orbit/models/orbit_mmdit.pth`.
- Intended public PTMCMC reference path: `artifacts/orbit/references/mcmc_betapic.hdf5`.
- Candidate orbit FMPE baseline artifacts: `artifact-source/orbitize_fmpe/`; see `docs/orbit_flow_matching_record.md`.
- Gravitational-wave local dataset paths: `flow-matching-posterior-estimation/gravitational-waves/datasets/waveform_dataset.hdf5` and `flow-matching-posterior-estimation/gravitational-waves/datasets/asd_dataset_GW150914.hdf5`.
- BBH MM-DiT local checkpoints/SVD files: `flowmatching_record/experiment/MM-DiT/BBH/`.
- Two-moons local checkpoint/plot files: `flowmatching_record/experiment/MM-DiT/two_moons/`.
- SBIBM FUSE checkpoint bundle: `FUSE_checkpoint/FUSE_checkpoint.zip`; see `FUSE_checkpoint/README.md` for archive contents and intended public path mapping.

If a checkpoint or generated output is not listed above or in `ARTIFACTS.md`, it is not included in the first public artifact release.

## Verified Command Interfaces

The following interfaces were checked on the remote release branch.

SBIBM training:

```bash
python flow-matching-posterior-estimation/sbi-benchmark/run_sbibm.py \
  --train_dir artifacts/sbibm/fuse/slcp/100000 \
  --dataset_dir artifacts/sbibm/slcp/100000 \
  --seed 1 \
  --generation_batch_size 1000
```

Expected inputs:

- `artifacts/sbibm/fuse/slcp/100000/settings.yaml`
- Optional cached `artifacts/sbibm/slcp/100000/x.npy`
- Optional cached `artifacts/sbibm/slcp/100000/theta.npy`

Expected outputs:

- `artifacts/sbibm/fuse/slcp/100000/best_model.pt`
- `artifacts/sbibm/fuse/slcp/100000/model_latest.pt`
- `artifacts/sbibm/fuse/slcp/100000/c2st.csv`
- `artifacts/sbibm/fuse/slcp/100000/posterior_1.png` through `posterior_10.png`

SBIBM evaluation:

```bash
python flow-matching-posterior-estimation/sbi-benchmark/evaluate_sbibm.py \
  --train_dir artifacts/sbibm/fuse/slcp/100000 \
  --dataset_dir artifacts/sbibm/slcp/100000 \
  --num_observations 10
```

Expected outputs:

- `artifacts/sbibm/fuse/slcp/100000/results.csv`
- `artifacts/sbibm/fuse/slcp/100000/01/samples.npy` through `10/samples.npy`
- `posterior_log_probs.npy` and `reference_log_probs.npy` for each evaluated observation

Orbit dataset generation:

```bash
python orbit_train/generate.py \
  --size 23 \
  --name orbit \
  --data-dir artifacts/orbit/datasets
```

Expected outputs:

- `artifacts/orbit/datasets/orbit-train.h5`
- `artifacts/orbit/datasets/orbit-val.h5`
- `artifacts/orbit/datasets/orbit-test.h5`

Orbit MM-DiT training:

```bash
python orbit_train/train_mmdit.py \
  --size 23 \
  --dataset-name orbit \
  --data-dir artifacts/orbit/datasets \
  --models-dir artifacts/orbit/models \
  --outputs-dir artifacts/orbit/outputs \
  --epochs 200 \
  --batch-size 4096
```

Expected output:

- `artifacts/orbit/models/orbit_mmdit.pth`

PTMCMC reference generation:

```bash
python orbit_train/mcmc.py \
  --output artifacts/orbit/references/mcmc_betapic.hdf5 \
  --num-temps 20 \
  --num-walkers 1000 \
  --n-orbs 10000000 \
  --num-threads 32
```

Expected output:

- `artifacts/orbit/references/mcmc_betapic.hdf5`

This PTMCMC command is a long-run reference-generation step and is not part of the quick verification suite.

## Verification Notes

- `run_sbibm.py --help` and `evaluate_sbibm.py --help` pass in the release-preparation `dingo` Conda environment.
- `orbit_train/mcmc.py --help` passes because heavy `orbitize` imports are deferred until runtime.
- `orbit_train/generate.py --help` and `orbit_train/train_mmdit.py --help` fail in the release-preparation `dingo` Conda environment with `ModuleNotFoundError: No module named 'orbitize'`. This should pass in the release environment after `environment.yml` installs `orbitize`.
- Full SBIBM sweeps, orbit training, orbit PTMCMC, FK sampling, and paper plotting were not run during release preparation.
