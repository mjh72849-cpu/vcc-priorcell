#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "$0")/.." && pwd)"
workspace="$(cd "${project}/.." && pwd)"
emb="${workspace}/data/state_embeddings"
python_bin="${VCC_PYTHON:-/sde/vcc/vcc2026/x-cell/.venv/bin/python}"

cd "${project}"
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
  --state-embeddings \
    "${emb}/vcc2025_train.npy" "${emb}/vcc2025_test.npy" \
    "${emb}/replogle_k562.npy" "${emb}/replogle_rpe1.npy" \
  --dataset-weights 0.20 0.15 0.30 0.35 \
  --vcc-state-embeddings \
    "${emb}/vcc2026_A.npy" "${emb}/vcc2026_B.npy" "${emb}/vcc2026_C.npy" \
  --de-cache artifacts/de_cache_level4 \
  --output checkpoints/level4_multicontext.pt \
  --device "${VCC_DEVICE:-cuda:3}" \
  --model-dim 128 --decoder-dim 64 --context-prototypes 8 \
  --factorized-effect \
  --control-steps 500 --perturbation-steps 12000 --control-every 4 \
  --context-cells 128 --query-cells 64 \
  --freeze-backbone-steps 750 \
  --learning-rate 1e-4 --head-learning-rate 3e-4 \
  --distribution-weight 0.7 --delta-weight 0.8 --direction-weight 0.15 \
  --de-weight 0.25 --magnitude-weight 0.1 --target-weight 0.05 \
  --support-weight 0.15 --ranking-weight 0.05 \
  --support-sign-weight 0.05 --support-magnitude-weight 0.05 \
  --cardinality-weight 0.02 \
  --save-every 500 --keep-step-checkpoints
