#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
  echo "usage: $0 INPUT.h5ad OUTPUT.npy EXPECTED_ROWS [BATCH_SIZE]" >&2
  exit 2
fi

project="$(cd "$(dirname "$0")/.." && pwd)"
workspace="$(cd "${project}/.." && pwd)"
input="$1"
output="$2"
expected_rows="$3"
batch_size="${4:-32}"

if [[ "${output}" != *.npy ]]; then
  echo "OUTPUT must end in .npy: ${output}" >&2
  exit 2
fi

full_output="${output%.npy}.state_full.npy"
"${project}/scripts/cache_state_embeddings.sh" \
  "${input}" "${full_output}" "${batch_size}"
"${workspace}/external/state/.venv/bin/python" \
  "${project}/scripts/normalize_state_embeddings.py" \
  "${full_output}" "${output}" --expected-rows "${expected_rows}"
