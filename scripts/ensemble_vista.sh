#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
PYTHON=${PYTHON:-python3}
PRED_DIR=${PRED_DIR:-${REPO_ROOT}/outputs/test_epoch016_allheads}
OUTPUT=${OUTPUT:-${PRED_DIR}/vista_dynamic_metric_ensemble.json}
ZIP_OUTPUT=${ZIP_OUTPUT:-${PRED_DIR}/vista_dynamic_metric_ensemble.zip}

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

"${PYTHON}" -m evals.vista_sta.dynamic_metric_head_ensemble \
  --inputs \
    "${PRED_DIR}/epoch_016_merged.json" \
    "${PRED_DIR}/epoch_016_head1_merged.json" \
    "${PRED_DIR}/epoch_016_head2_merged.json" \
    "${PRED_DIR}/epoch_016_head3_merged.json" \
  --output "${OUTPUT}" \
  --zip-output "${ZIP_OUTPUT}" \
  "$@"
