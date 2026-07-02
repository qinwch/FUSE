#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${REPO_ROOT}/flow-matching-posterior-estimation/sbi-benchmark"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/sbibm}"
METHOD="${METHOD:-fuse}"
BUDGET="${BUDGET:-100000}"
TASK="${TASK:-two_moons}"

TRAIN_DIR="${ARTIFACT_ROOT}/${METHOD}/${TASK}/${BUDGET}"
DATASET_DIR="${ARTIFACT_ROOT}/${TASK}/${BUDGET}"
SETTINGS_FILE="${TRAIN_DIR}/settings.yaml"

mkdir -p "${TRAIN_DIR}" "${DATASET_DIR}"

if [[ ! -f "${SETTINGS_FILE}" ]]; then
  {
    echo "error: missing settings file: ${SETTINGS_FILE}"
    echo "Create or copy a task settings YAML into that path, or set ARTIFACT_ROOT, METHOD, TASK, and BUDGET to an existing artifact directory."
  } >&2
  exit 1
fi

cd "${BENCH_DIR}"
python run_sbibm.py \
  --train_dir "${TRAIN_DIR}" \
  --dataset_dir "${DATASET_DIR}" \
  --generation_batch_size "${GENERATION_BATCH_SIZE:-1000}" \
  --seed "${SEED:-1}"
