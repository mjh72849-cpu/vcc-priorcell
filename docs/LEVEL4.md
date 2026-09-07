# PriorCell Level 4

Level 4 keeps the Level 3.1 prior and factorised decoder, while changing the
information flow in four places:

1. Frozen STATE SE-600M produces a 2,048-dimensional biological cell-state
   vector per cell. The official CLI's additional 10 dataset-ID dimensions are
   dropped before training to reduce dataset/batch leakage.
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
- Level-4 vocabulary: `artifacts/gene_vocabulary_level4` (24,413 global genes;
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
CUDA_VISIBLE_DEVICES=3 ./scripts/cache_state_cell_embeddings.sh \
  INPUT.h5ad OUTPUT.npy EXPECTED_ROWS 32
```

Create the eight files named in `scripts/train_level4.sh` (five perturbation
datasets plus VCC contexts A/B/C). The official STATE CLI emits 2,058 columns:
2,048 cell-state dimensions followed by 10 dataset-ID dimensions. Run
`normalize_state_embeddings.py` on each cache so training receives only the
first 2,048 columns. Jiang's `X` is log-normalized and is consumed directly by
STATE; raw counts remain in `layers/counts`, which the PriorCell trainer selects
automatically. Do not mix `treatment` with the genetic target in `gene`: Level 4
groups Jiang episodes by `(cell_type, treatment, gene)` and recognizes `NT` as
the control label.

## Train

```bash
VCC_DEVICE=cuda:3 ./scripts/train_level4.sh
```

The starter mix contains H1 (VCC 2025), K562, RPE1 and Jiang's six cell lines by
five stimulus contexts. Jiang falls back to online sampled effects when it is
absent from the optional DE cache. Select checkpoints with leave-one-context-out
validation; retain VCC 2025 validation as a final holdout.

STATE source and weights have non-commercial terms. Review the license files in
the STATE checkout before use beyond research or competition work.
