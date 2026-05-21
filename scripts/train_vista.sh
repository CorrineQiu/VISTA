#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
PYTHON=${PYTHON:-python3}
CONFIG=${CONFIG:-${REPO_ROOT}/configs/vista_train.yaml}
DEVICES=${DEVICES:-"cuda:0 cuda:1 cuda:2 cuda:3"}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/outputs/logs/train}
export DETECTOR_WEIGHTS=${DETECTOR_WEIGHTS:-}

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/runs"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DIST_TIMEOUT_SEC=${TORCH_DIST_TIMEOUT_SEC:-86400}
export PYTHONFAULTHANDLER=1

log_file="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "CONFIG=${CONFIG}"
echo "DEVICES=${DEVICES}"
echo "LOG=${log_file}"

# shellcheck disable=SC2086
"${PYTHON}" -u -m evals.main \
  --fname "${CONFIG}" \
  --devices ${DEVICES} \
  "$@" 2>&1 | tee -a "${log_file}"
