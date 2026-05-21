#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
PYTHON=${PYTHON:-python3}
export EGO4D_STA_CACHE_ROOT=${EGO4D_STA_CACHE_ROOT:-${REPO_ROOT}/outputs/cache}
export STA_BUILD_TEST_CACHE=${STA_BUILD_TEST_CACHE:-1}
if [[ -n "${EGO4D_STA_ROOT:-}" ]]; then
  export STA_TRAIN_ROOT=${STA_TRAIN_ROOT:-${EGO4D_STA_ROOT}/frames_of_video_sta_2fps/train}
  export STA_VAL_ROOT=${STA_VAL_ROOT:-${EGO4D_STA_ROOT}/frames_of_video_sta_2fps/val}
  export STA_TEST_ROOT=${STA_TEST_ROOT:-${EGO4D_STA_ROOT}/frames_of_video_sta_2fps/test_unannotated}
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
"${PYTHON}" -m evals.vista_sta.prebuild_train_val_cache "$@"
