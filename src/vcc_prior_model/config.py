"""Configuration for the PriorCell architecture."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Shape and regularisation choices for :class:`PriorAwarePerturbationModel`.

    The defaults target VCC's 18,533-gene output and are deliberately modest:
    a 400-cell decoder output occupies about 30 MiB in float32 (15 MiB in
    bfloat16), before gradients.  Context controls can be encoded in chunks.
    """

    # ``num_genes`` is the global node vocabulary: VCC output genes plus any
    # useful genes found only in public datasets or prior graphs.
    num_genes: int = 18_533
    # Defaults to ``num_genes``. When it is smaller, callers must provide the
    # global indices of the fixed output panel when constructing the model.
    num_output_genes: int | None = None
    num_go_terms: int = 0
    num_pathway_terms: int = 0
    num_perturbation_types: int = 3

    gene_dim: int = 256
    cell_dim: int = 256
    context_dim: int = 256
    perturbation_dim: int = 256
    decoder_dim: int = 128

    graph_layers: int = 2
    context_prototypes: int = 8
    dropout: float = 0.10
    prior_dropout: float = 0.20
    normalize_active_priors: bool = True
    prior_gate_init_bias: float = -2.0
    prototype_temperature: float = 0.50
    max_log_effect: float = 2.0
    # Level 3.1 factorises a perturbation into DE support, sign and magnitude.
    # The reference values normalise the newly-added gates so a Level-3 warm
    # start is function preserving before the new heads receive supervision.
    factorized_effect: bool = False
    support_reference_probability: float = 0.20
    effect_strength_min: float = 0.10
    effect_strength_max: float = 0.80
    effect_strength_reference: float = 0.30
    magnitude_modulation: float = 0.50
    # Level 4 consumes frozen STATE SE embeddings. SE-600M emits 512
    # dimensions; the adapter keeps this dependency outside the core model so
    # embeddings can be cached once and training remains inexpensive.
    pretrained_cell_dim: int = 512
    pretrained_gate_init: float = 0.20
    context_prior_scale: float = 0.10
    baseline_decoder_mix_init: float = 0.02
    population_rank: int = 16
    population_residual_scale: float = 0.10
    min_dispersion: float = 1e-4
    eps: float = 1e-8

    def __post_init__(self) -> None:
        positive = {
            "num_genes": self.num_genes,
            "num_perturbation_types": self.num_perturbation_types,
            "gene_dim": self.gene_dim,
            "cell_dim": self.cell_dim,
            "context_dim": self.context_dim,
            "perturbation_dim": self.perturbation_dim,
            "decoder_dim": self.decoder_dim,
            "graph_layers": self.graph_layers,
            "context_prototypes": self.context_prototypes,
            "prototype_temperature": self.prototype_temperature,
            "min_dispersion": self.min_dispersion,
            "eps": self.eps,
            "pretrained_cell_dim": self.pretrained_cell_dim,
            "population_rank": self.population_rank,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.num_go_terms < 0 or self.num_pathway_terms < 0:
            raise ValueError("term counts cannot be negative")
        if self.num_output_genes is not None:
            if self.num_output_genes <= 0:
                raise ValueError("num_output_genes must be positive")
            if self.num_output_genes > self.num_genes:
                raise ValueError("num_output_genes cannot exceed num_genes")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.prior_dropout < 1.0:
            raise ValueError("prior_dropout must be in [0, 1)")
        if not math.isfinite(self.prior_gate_init_bias):
            raise ValueError("prior_gate_init_bias must be finite")
        if not 0.0 < self.support_reference_probability < 1.0:
            raise ValueError("support_reference_probability must be in (0, 1)")
        if not 0.0 < self.effect_strength_min < self.effect_strength_max:
            raise ValueError("effect strength bounds must satisfy 0 < min < max")
        if not (
            self.effect_strength_min < self.effect_strength_reference < self.effect_strength_max
        ):
            raise ValueError("effect_strength_reference must lie inside its bounds")
        if self.magnitude_modulation < 0.0:
            raise ValueError("magnitude_modulation must be non-negative")
        if not 0.0 < self.pretrained_gate_init < 1.0:
            raise ValueError("pretrained_gate_init must be in (0, 1)")
        if not 0.0 <= self.baseline_decoder_mix_init < 1.0:
            raise ValueError("baseline_decoder_mix_init must be in [0, 1)")
        if self.context_prior_scale < 0.0 or self.population_residual_scale < 0.0:
            raise ValueError("Level-4 residual scales must be non-negative")

    @property
    def output_genes(self) -> int:
        """Number of genes decoded by the default VCC output head."""

        return self.num_genes if self.num_output_genes is None else self.num_output_genes
