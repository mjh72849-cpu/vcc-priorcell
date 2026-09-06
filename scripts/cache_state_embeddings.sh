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

cd "${state_repo}"
uv run state emb transform \
  --model-folder "${model_dir}" \
  --checkpoint "${model_dir}/se600m_epoch16.ckpt" \
  --protein-embeddings "${model_dir}/protein_embeddings.pt" \
  --input "$1" \
  --output "$2" \
  --batch-size "${batch_size}"
