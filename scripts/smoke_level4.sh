#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "$0")/.." && pwd)"
workspace="$(cd "${project}/.." && pwd)"
emb="${workspace}/data/state_embeddings"
python_bin="${VCC_PYTHON:-/sde/vcc/vcc2026/x-cell/.venv/bin/python}"

cd "${project}"
"${python_bin}" scripts/verify_level4_embeddings.py
PYTHONPATH=src "${python_bin}" scripts/train_real.py \
  --level context_adaptive \
  --vocabulary artifacts/gene_vocabulary_level4 \
  --priors artifacts/priors_level4 \
  --initialize-from checkpoints/level31_best.pt \
  --perturbation-data \
    "${workspace}/data/vcc_2025/train/adata_Training.h5ad" \
    "${workspace}/data/vcc_2025/test/adata_Test.h5ad" \
    "${workspace}/data/training/replogle_k562_vcc_balanced_raw.h5ad" \
    "${workspace}/data/references/replogle_rpe1/ReplogleWeissman2022_rpe1.h5ad" \
    "${workspace}/data/multicontext/jiang2025/jiang24_processed.h5ad" \
  --state-embeddings \
    "${emb}/vcc2025_train.npy" "${emb}/vcc2025_test.npy" \
    "${emb}/replogle_k562.npy" "${emb}/replogle_rpe1.npy" \
    "${emb}/jiang24.npy" \
  --dataset-weights 0.15 0.10 0.20 0.20 0.35 \
  --vcc-state-embeddings \
    "${emb}/vcc2026_A.npy" "${emb}/vcc2026_B.npy" "${emb}/vcc2026_C.npy" \
  --de-cache artifacts/de_cache_level4 \
  --output checkpoints/level4_smoke.pt \
  --device "${VCC_DEVICE:-cuda:3}" \
  --pretrained-cell-dim 2048 \
  --model-dim 128 --decoder-dim 64 --context-prototypes 8 \
  --factorized-effect \
  --control-steps 0 --perturbation-steps 1 --control-every 4 \
  --context-cells 16 --query-cells 8 \
  --learning-rate 1e-4 --head-learning-rate 3e-4
