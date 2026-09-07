#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "usage: $0 INPUT.h5ad OUTPUT.npy [BATCH_SIZE]" >&2
  exit 2
fi

project="$(cd "$(dirname "$0")/.." && pwd)"
workspace="$(cd "${project}/.." && pwd)"
state_repo="${workspace}/external/state"
model_dir="${workspace}/external/SE-600M"
batch_size="${3:-16}"
state_bin="${STATE_BIN:-${state_repo}/.venv/bin/state}"

if [[ ! -x "${state_bin}" ]]; then
  echo "STATE CLI is unavailable at ${state_bin}" >&2
  exit 1
fi

cd "${state_repo}"
"${state_bin}" emb transform \
  --model-folder "${model_dir}" \
  --checkpoint "${model_dir}/se600m_epoch16.ckpt" \
  --config "${model_dir}/config.yaml" \
  --protein-embeddings "${model_dir}/protein_embeddings.pt" \
  --input "$1" \
  --output "$2" \
  --batch-size "${batch_size}"
