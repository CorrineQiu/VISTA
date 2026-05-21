#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
PYTHON=${PYTHON:-python3}
CONFIG=${CONFIG:-${REPO_ROOT}/configs/vista_train.yaml}
SPLITS=${SPLITS:-train,val,test}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs/jepa_global_cache_vjepa8f}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
SHARD_ID=${SHARD_ID:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
SAMPLE_CACHE_DIR=${SAMPLE_CACHE_DIR:-}
export EGO4D_STA_CACHE_ROOT=${EGO4D_STA_CACHE_ROOT:-${REPO_ROOT}/outputs/cache}
export DETECTOR_WEIGHTS=${DETECTOR_WEIGHTS:-}

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

args=(
  -m evals.vista_sta.extract_jepa_global_features
  --config "${CONFIG}"
  --output-root "${OUTPUT_ROOT}"
  --splits "${SPLITS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --shard-id "${SHARD_ID}"
  --num-shards "${NUM_SHARDS}"
)

if [[ -n "${SAMPLE_CACHE_DIR}" ]]; then
  args+=(--sample-cache-dir "${SAMPLE_CACHE_DIR}")
fi

"${PYTHON}" "${args[@]}" "$@"
