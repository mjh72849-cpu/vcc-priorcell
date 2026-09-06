# PriorCell Level 4

Level 4 keeps the Level 3.1 prior and factorised decoder, while changing the
information flow in four places:

1. Frozen STATE SE-600M produces one 512-dimensional vector per cell.
   `PretrainedCellStateAdapter` projects and conservatively gates it into the
   learned raw-count state. Cached features avoid replicating the 600M encoder
   during PriorCell training.
2. `ContextConditionedTargetPrior` adapts the static GO/STRING/Reactome/GRN
   target embedding using the NTC state plus target mean, variance and detection
   rate. Its residual starts at zero.
3. `AdaptiveEffectCalibrator` and `LowRankPopulationResidual` separate
   context-level magnitude from cell-to-cell response heterogeneity.
4. `DenoisedEmpiricalBaseline` predicts empirical NTC plus a calibrated delta.
   The initial 2% decoder-rate mixture can turn sampling zeros on without free
   whole-expression generation.

The forward API adds optional `context_pretrained_cell_state`,
`query_pretrained_cell_state`, and `empirical_baseline_counts`. Without STATE
caches, the model still runs through the count branch for ablation experiments.

## Prepared assets

- STATE source: `../external/state`.
- SE-600M weights: `../external/SE-600M`.
- Replogle RPE1: `../data/references/replogle_rpe1`.
- Jiang et al. six-cell-line/five-stimulus CRISPRi:
  `../data/multicontext/jiang2025`.
- Level-4 vocabulary: `artifacts/gene_vocabulary_level4` (19,301 global genes;
  fixed VCC output remains the ordered 18,533 genes).
- Re-aligned priors: `artifacts/priors_level4`.

Downloads are resumable and publisher MD5 checks are applied where available:

```bash
./scripts/download_level4_assets.sh all
```

## Cache STATE embeddings

The STATE environment is in `../external/state/.venv`. Cache files preserve
H5AD row order; `train_real.py` validates row count and width.

```bash
mkdir -p ../data/state_embeddings
CUDA_VISIBLE_DEVICES=3 ./scripts/cache_state_embeddings.sh INPUT.h5ad OUTPUT.npy 32
```

Create the seven files named in `scripts/train_level4.sh`. Jiang is released as
Seurat RDS; the same directory also contains PerturBench's training-ready H5AD
conversion and split. Its `X` is normalized and raw counts are in
`layers/counts`; the PriorCell loader selects that layer automatically. For
STATE, first gunzip it and use `materialize_raw_count_h5ad.py` to promote the
raw-count layer to `X`. Do not mix `treatment` with the genetic `condition`:
Level 4 groups episodes by `(cell_type, treatment, condition)`.

## Train

```bash
VCC_DEVICE=cuda:3 ./scripts/train_level4.sh
```

The starter mix contains H1 (VCC 2025), K562 and RPE1. Add Jiang contexts after
RDS-to-H5AD conversion and rebuild their DE cache. Select checkpoints with
leave-one-context-out validation; retain VCC 2025 validation as a final holdout.

STATE source and weights have non-commercial terms. Review the license files in
the STATE checkout before use beyond research or competition work.
