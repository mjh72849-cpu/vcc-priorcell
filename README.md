# PriorCell architecture for VCC

This directory contains the model architecture, reproducible vocabulary/prior
artifacts, real-data loaders, and Level 1/2 training and validation entry points.
Large source matrices remain under `../data/` and are not copied into the package.

The new context-generalising Level 4, including STATE integration, downloads and
a concrete training command, is documented in [docs/LEVEL4.md](docs/LEVEL4.md).

## Architecture ladder

All levels accept the same `forward(...)` arguments and return the same
`ModelOutput`, so they can be compared without changing downstream evaluation.

| Level | Factory name | Model | Purpose |
|---|---|---|---|
| 0 | `control` | `ControlBaselineModel` | unchanged-query no-effect baseline |
| 1 | `global` | `LearnedGlobalEffectModel` | learned target ID; one global latent shift |
| 2 | `cell_state` | `LearnedCellStateModel` | cell-specific FiLM shift; no prior |
| 3 | `functional_prior` | `FunctionalPriorModel` | add GO, STRING and Reactome |
| 3.1 | `functional_prior` + `factorized_effect=True` | `FunctionalPriorModel` | stable DE support/sign/magnitude heads and library-preserving effects |
| 3+ | `full_prior` | `PriorAwarePerturbationModel` | add signed/directed CollecTRI GRN |
| 4 | `context_adaptive` | `ContextAdaptivePriorModel` | frozen STATE cell state, context-adaptive target prior, calibrated empirical delta and low-rank population response |

```python
from vcc_prior_model import ArchitectureLevel, ModelConfig, build_model

config = ModelConfig(
    num_genes=25_000,          # global union vocabulary
    num_output_genes=18_533,   # fixed VCC panel
    num_go_terms=12_000,
    num_pathway_terms=2_000,
)
model = build_model(
    ArchitectureLevel.FULL_PRIOR,
    config,
    output_gene_index=vcc_output_gene_index,
)
output = model(
    context_counts,                # [B, N_context, G_context_dataset]
    query_counts,                  # [B, N_query, G_query_dataset]
    target_gene_index,             # [B], indices in global union vocabulary
    priors,                        # PriorInputs; optional for levels 0--2
    context_observed_gene_mask=None,
    query_observed_gene_mask=None,
    context_cell_mask=None,
    gene_chunk_size=4096,
    context_gene_index=context_gene_index,
    query_gene_index=query_gene_index,
)
```

Levels 1--4 additionally expose the staged, resource-efficient interface:

```python
genes = model.encode_genes(priors)  # [G_global, D_gene]
context = model.encode_context_chunks(
    context_count_chunks,
    genes,
    input_gene_index=context_gene_index,
)
output = model.predict_from_context(
    query_counts,
    target_gene_index,
    context,
    query_gene_index=query_gene_index,
    # target context statistics are cached inside `context`
)
```

The sparse-prior input interface is explicit and independent of file formats:

```python
from vcc_prior_model import MembershipEdges, PriorInputs

priors = PriorInputs(
    go=MembershipEdges(go_gene_index, go_term_index, go_information_weight),
    reactome=MembershipEdges(pathway_gene_index, pathway_index),
    string_edge_index=string_edges,       # [2, E_string], undirected
    string_edge_weight=string_score,      # [E_string]
    grn_edge_index=grn_edges,             # [2, E_grn], source -> target
    grn_edge_sign=grn_sign,               # +1 activation, -1 inhibition
    grn_edge_weight=grn_confidence,
).to(device)
```

The common output interface contains:

```text
mean                    [B, Q, G]  negative-binomial mean
control_mean            [B, Q, G]  reconstructed NTC baseline
log_effect              [B, Q, G]  bounded residual perturbation effect
dispersion                    [G]  gene-wise NB dispersion
output_gene_index             [G]  global indices of decoded genes
context_embedding       [B, D_ctx]
perturbation_embedding  [B, D_pert]
query_latent            [B, Q, D_cell]
delta_latent            [B, Q, D_cell]
perturbed_latent        [B, Q, D_cell]
state_prototypes        [B, K, D_cell]
```

## Data flow

```text
all NTC cells ── expression-weighted cell encoder ── prototype DeepSets ── z_context
                                                               │
target gene ── learned ID + GO + STRING + Reactome + signed GRN ─┤
                                                               ▼
query NTC cell latents ──────────────┬── control NB decoder ── control_mean
                                     │                             │
                                     └── FiLM transition ── residual log_effect
                                                                   │
                                                                   ▼
                                KD mean = control_mean * exp(log_effect)
```

The same fused gene space is shared by the perturbation encoder and decoder.
Every gene always has a learned base embedding. Missing priors are masked rather
than interpreted as biological zeros.

## Level 3.1 DE-aware fine-tuning

Level 3.1 keeps the Level-3 context and functional-prior backbone, while
factorising each gene's perturbation response into a perturbation-level DE
support probability, a cell-level sign, a magnitude adjustment, and a bounded
context/target strength. The new gates are normalised at initialisation, so
loading a Level-3 checkpoint initially preserves its exact `log_effect`.

Training no longer creates a noisy top-200 label from each random minibatch.
`scripts/prepare_de_cache.py` computes deterministic dataset-level
Mann-Whitney/Wilcoxon statistics from fixed capped cell samples, BH-adjusted q-values, log fold changes, DE
masks and confidence weights. `scripts/train_real.py --de-cache ...
--factorized-effect` consumes those artifacts and adds balanced focal support,
hard-negative ranking, sign, magnitude and DE-cardinality objectives. Cache
gene order and dataset paths are checked before training.

For a factorised checkpoint, both the decoder and submission/evaluation path
apply multiplicative effects with per-cell library-size preservation:

```text
raw_i       = baseline_i * exp(delta_i)
prediction  = raw * sum(baseline) / sum(raw)
```

The selected trained artifact is `checkpoints/level31_best.pt`. Its strict
VCC25-validation evaluations are under `evaluations/level31_*_vcc25val50.json`
and `evaluations/level31_*_support_vcc25val50.json`. No VCC submission is
created by Level 3.1 training.

Prior fusion uses three safeguards by default:

```text
per-gene active-source normalization: prior_sum / sqrt(max(active_count, 1))
conservative gate initialization:     sigmoid(-2) ~= 0.119
source-level prior dropout:            0.20
```

The gate's final weights start at zero, so the model begins base-embedding
dominant and must learn to trust each prior. During training, prior dropout
removes an entire source contribution for a forward pass and uses inverted
dropout scaling. Genes with no active source receive exactly zero prior
contribution. These controls are configurable with `normalize_active_priors`,
`prior_gate_init_bias`, and `prior_dropout` in `ModelConfig`.

## Modules

- `GenePriorEncoder`: gated residual fusion of four independent prior encoders.
- `MembershipEncoder`: GO/Reactome gene--term hypergraph encoding without dense
  gene cliques.
- `WeightedGraphEncoder`: weighted, undirected GraphSAGE-style STRING encoder.
- `SignedDirectedGraphEncoder`: distinct incoming/outgoing and
  activation/inhibition CollecTRI relations.
- `ExpressionCellEncoder`: count-normalized projection through shared gene
  embeddings, with library size and detection-rate covariates.
- `PrototypeSetEncoder`: permutation-invariant context encoding with eight cell
  state prototypes. Its statistics are additive, so 18,400 controls can be
  encoded in chunks. The same pass caches per-gene context mean, variance, and
  detection rate for reuse by all perturbation targets.
- `ContextConditionedTransition`: FiLM residual transition applied separately to
  every query cell; it does not collapse 400 cells to one pseudobulk vector.
- `GeneAwareNegativeBinomialDecoder`: independent per-gene control NB rates plus
  a bounded, zero-initialised residual log effect. It has no whole-panel softmax,
  so an unmeasured gene cannot alter another gene through the denominator.
- `reconstruct_control`: bypasses perturbation modules for NTC autoencoding over
  any requested output subset.
- `CompositePerturbationLoss`: defaults to unpaired sliced-Wasserstein set loss,
  pseudobulk delta, direction, DE weighting, and unsupervised-effect shrinkage.
  Pairwise NB is disabled by default and should only be enabled after explicit
  cell matching. `scripts/train_real.py` supplies the real optimiser/checkpoint loop.

## Resource profile

Defaults are sized for VCC and conservative single-GPU use:

```python
ModelConfig(
    num_genes=18_533,        # or a larger global union vocabulary
    num_output_genes=None,   # None means all global genes; set 18_533 for a union
    gene_dim=256,
    cell_dim=256,
    decoder_dim=128,
    graph_layers=2,
    context_prototypes=8,
    normalize_active_priors=True,
    prior_gate_init_bias=-2.0,
    prior_dropout=0.20,
)
```

For 400 query cells, the unavoidable final `400 x 18,533` mean tensor is about
30 MiB in float32 or 15 MiB in bfloat16. Graph computation scales with retained
edges; preprocessing should use STRING score >= 400 and keep the top 32--64
neighbours per gene. Context encoding scales linearly with cells and can use
`encode_context_chunks`, for example 256--1,024 controls per chunk.

The intended multi-target inference pattern is:

1. `gene_embeddings = model.encode_genes(priors)` once.
2. `context = model.encode_context_chunks(...)` once per A/B/C context.
3. `model.predict_from_context(...)` for each perturbation target and its 400
   representative query cells.

This avoids re-encoding all 18,400 controls for every target.

## Heterogeneous gene panels

The model separates three axes:

```text
global node vocabulary = VCC genes U public-dataset-only genes U prior-only genes
dataset input panel     = genes actually measured by this dataset
default output panel    = exactly 18,533 VCC genes
```

`input_gene_index` maps a dataset count column into the global node vocabulary.
This means the 569 Replogle-only genes may inform the cell encoder without
appearing in the VCC submission. `output_gene_index` is stored as a model buffer
and fixes decoder order. During partial-panel training, `decode_gene_index` can
request only the measured VCC intersection; at VCC inference it is omitted so
the full fixed output panel is decoded.

An explicit control objective is available:

```python
control = model.reconstruct_control(
    ntc_counts,
    query_gene_index=dataset_gene_index,
    decode_gene_index=observed_vcc_intersection,
)
loss_control = control_reconstruction_loss(control, aligned_observed_counts)
```

Perturbation output is residual by construction:

```text
mean_KD = control_mean * exp(log_effect)
```

The latent transition and direct perturbation/relation effect paths are
zero-initialised, so an untrained model has `log_effect == 0` and does not invent
changes for the 10,854 VCC genes absent from the local Replogle panel. The
transition's downstream projection remains non-zero so gradients can reach the
zero-initialised transition on the first update. The composite loss accepts
`effect_supervision_mask`; its complement receives a configurable squared-effect
shrinkage penalty (default weight 0.02).

## Tensor contract

- context counts: `[batch, context_cells, context_dataset_genes]`
- query counts: `[batch, query_cells, query_dataset_genes]`
- input gene indices: `[dataset_genes]`, indexing the global vocabulary
- target indices: `[batch]`, indexing the global vocabulary
- STRING/GRN edges: `[2, edges]`
- output gene indices: `[18_533]`, indexing the global vocabulary
- output NB mean: `[batch, query_cells, 18_533]`
- output dispersion: `[18_533]`

Missing genes in a public training source are represented by
`observed_gene_mask`; they are excluded from cell encoding and should also be
masked in the loss. They remain present in the full decoder output.

The real local VCC/Replogle vocabulary and mappings have been generated under
[`artifacts/gene_vocabulary`](artifacts/gene_vocabulary/README.md). Load them
without AnnData using `GeneVocabularyArtifacts`; the resulting tensors plug
directly into `ModelConfig`, `build_model`, and the dataset-specific gene-index
arguments shown above.

## Synthetic-only verification

Use an environment with PyTorch installed:

```bash
cd /sde/vcc/vcc2026/priorcell
PYTHONPATH=src ../x-cell/.venv/bin/python -m pytest tests
PYTHONPATH=src ../x-cell/.venv/bin/python examples/smoke_forward.py
```

If `pytest` is unavailable, the test module also has a dependency-free runner:

```bash
PYTHONPATH=src ../x-cell/.venv/bin/python tests/test_architecture.py
```
