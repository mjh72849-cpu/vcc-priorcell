# Real gene vocabulary artifacts

These files were built from the local VCC 2026 controls and the full local
Replogle K562 GWPS H5AD. The expression matrices were not loaded.

## Canonical ordering

1. Global indices `0..18532` are the 18,533 VCC genes in exact submission order.
2. Global indices `18533..18535` are the three VCC 2025-only expression genes.
3. Global indices `18536..19104` are the 569 Replogle-only expression genes in
   their original H5AD `var` order.
4. Global indices `19105..19196` are the 92 additional Replogle perturbation
   targets that are not expression features in either panel.

Gene identity uses an exact, case-sensitive symbol match after stripping outer
whitespace. No alias substitution or case folding was applied. See
`manifest.json` for source hashes and coverage counts.

## Files

- `global_gene_vocabulary.csv`: one row per global gene node.
- `vcc_output_mapping.csv`: fixed VCC output position to global index.
- `perturbation_target_mapping.csv`: all 300 targets, global/output indices,
  Replogle perturbation availability and observed perturbed-cell count.
- `datasets/*.csv`: local column to global and VCC output indices (`-1` means
  absent from that output panel).
- `arrays/*_gene_index.npy`: pass as `context_gene_index` or `query_gene_index`.
- `arrays/*_global_observed_mask.npy`: presence over all 19,197 global nodes.
- `arrays/*_output_observed_mask.npy`: loss mask over 18,533 VCC outputs.
- `arrays/*_local_to_output_index.npy`: local column to VCC column, or `-1`.
- `arrays/*_output_to_local_index.npy`: VCC column to local column, or `-1`.

## Model use

```python
import torch
from vcc_prior_model import GeneVocabularyArtifacts, ModelConfig, build_model

vocab = GeneVocabularyArtifacts("artifacts/gene_vocabulary")
replogle = vocab.load_dataset("replogle_k562_gwps")

config = ModelConfig(
    num_genes=vocab.num_genes,
    num_output_genes=vocab.num_output_genes,
)
model = build_model("cell_state", config, vocab.output_gene_index)

output = model(
    context_counts,
    query_counts,
    vocab.target_index(["ACLY"]),
    context_gene_index=replogle.gene_index,
    query_gene_index=replogle.gene_index,
)
```

For a Replogle training loss, align the decoder and observed matrix explicitly:

```python
mask = replogle.output_observed_mask
decode_gene_index = vocab.output_gene_index[mask]  # 7,679 global indices
observed_local_index = replogle.output_to_local_index[mask]
observed_counts = replogle_counts[..., observed_local_index]

output = model(
    context_counts,
    query_counts,
    target_index,
    context_gene_index=replogle.gene_index,
    query_gene_index=replogle.gene_index,
    decode_gene_index=decode_gene_index,
)
```

At VCC inference, omit `decode_gene_index` to decode all 18,533 genes. In
`perturbation_target_mapping.csv`, `has_replogle_expression_gene` only says
whether the target itself is one of the 8,248 measured features;
`has_replogle_perturbation` and `replogle_perturbed_cell_count` determine whether
that target has direct CRISPRi supervision.

Rebuild deterministically with:

```bash
cd /sde/vcc/vcc2026/priorcell
PYTHONPATH=src ../x-cell/.venv/bin/python scripts/build_gene_vocabulary.py
```
