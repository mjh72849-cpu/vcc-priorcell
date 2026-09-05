"""Comparable architecture tiers from a no-effect control to the full model."""

from __future__ import annotations

from enum import Enum

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig
from .model import ModelOutput, PriorAwarePerturbationModel, PriorInputs


class ArchitectureLevel(str, Enum):
    """Stable names used by :func:`build_model`."""

    CONTROL = "control"
    GLOBAL = "global"
    CELL_STATE = "cell_state"
    FUNCTIONAL_PRIOR = "functional_prior"
    FULL_PRIOR = "full_prior"


class ControlBaselineModel(nn.Module):
    """Level 0: return each query NTC cell unchanged.

    This has no perturbation capacity and exists as a pipeline/metric baseline.
    Its forward signature and :class:`ModelOutput` match all learned tiers.
    """

    def __init__(self, config: ModelConfig, output_gene_index: Tensor | None = None) -> None:
        super().__init__()
        self.config = config
        if output_gene_index is None:
            if config.output_genes != config.num_genes:
                raise ValueError(
                    "output_gene_index is required when num_output_genes differs from num_genes"
                )
            output_gene_index = torch.arange(config.num_genes)
        if output_gene_index.numel() != config.output_genes:
            raise ValueError(f"output_gene_index must have shape [{config.output_genes}]")
        self.register_buffer("output_gene_index", output_gene_index.to(torch.long))
        self.raw_dispersion = nn.Parameter(torch.full((config.num_genes,), 2.0))

    def forward(
        self,
        context_counts: Tensor,
        query_counts: Tensor,
        target_gene_index: Tensor,
        priors: PriorInputs | None = None,
        perturbation_type: Tensor | None = None,
        context_observed_gene_mask: Tensor | None = None,
        query_observed_gene_mask: Tensor | None = None,
        context_cell_mask: Tensor | None = None,
        gene_chunk_size: int | None = 4096,
        context_gene_index: Tensor | None = None,
        query_gene_index: Tensor | None = None,
        decode_gene_index: Tensor | None = None,
        query_size_factor: Tensor | None = None,
    ) -> ModelOutput:
        del (
            context_counts,
            target_gene_index,
            priors,
            perturbation_type,
            context_observed_gene_mask,
            query_observed_gene_mask,
            context_cell_mask,
            gene_chunk_size,
            context_gene_index,
            query_size_factor,
        )
        output_index = self.output_gene_index if decode_gene_index is None else decode_gene_index
        output_index = output_index.to(query_counts.device)
        if query_gene_index is None:
            if query_counts.shape[-1] != self.config.num_genes:
                raise ValueError(
                    "query_gene_index is required when query counts do not use "
                    "the global vocabulary"
                )
            global_counts = query_counts
        else:
            if query_gene_index.numel() != query_counts.shape[-1]:
                raise ValueError("query_gene_index length does not match query counts")
            global_counts = query_counts.new_zeros(
                (*query_counts.shape[:-1], self.config.num_genes)
            )
            global_counts.index_copy_(
                -1, query_gene_index.to(query_counts.device), query_counts
            )
        mean = global_counts.index_select(-1, output_index)
        batch, query_cells, _ = query_counts.shape
        zeros_context = query_counts.new_zeros((batch, self.config.context_dim))
        zeros_perturbation = query_counts.new_zeros((batch, self.config.perturbation_dim))
        zeros_cells = query_counts.new_zeros((batch, query_cells, self.config.cell_dim))
        zeros_prototypes = query_counts.new_zeros(
            (batch, self.config.context_prototypes, self.config.cell_dim)
        )
        return ModelOutput(
            mean=mean.clamp_min(0.0),
            control_mean=mean.clamp_min(0.0),
            log_effect=mean.new_zeros(mean.shape),
            dispersion=F.softplus(self.raw_dispersion.index_select(0, output_index))
            + self.config.min_dispersion,
            output_gene_index=output_index,
            context_embedding=zeros_context,
            perturbation_embedding=zeros_perturbation,
            query_latent=zeros_cells,
            delta_latent=zeros_cells,
            perturbed_latent=zeros_cells,
            state_prototypes=zeros_prototypes,
        )

    @torch.no_grad()
    def sample_counts(self, output: ModelOutput) -> Tensor:
        theta = output.dispersion.view(1, 1, -1)
        probability = output.mean / (theta + output.mean)
        return torch.distributions.NegativeBinomial(
            total_count=theta, probs=probability.clamp(max=1.0 - self.config.eps)
        ).sample().to(torch.int64)


class LearnedGlobalEffectModel(PriorAwarePerturbationModel):
    """Level 1: learned target IDs and one shared latent shift per cell set.

    It intentionally ignores every supplied biological prior.  All query cells
    receive the same latent delta, making this the simplest trainable ablation.
    """

    def __init__(self, config: ModelConfig, output_gene_index: Tensor | None = None) -> None:
        super().__init__(
            config, enabled_priors=frozenset(), output_gene_index=output_gene_index
        )
        # The global tier replaces, rather than supplements, the cell-wise FiLM block.
        self.transition = nn.Identity()
        self.global_delta = nn.Sequential(
            nn.Linear(config.context_dim + config.perturbation_dim, config.cell_dim),
            nn.GELU(),
            nn.Linear(config.cell_dim, config.cell_dim),
        )
        nn.init.zeros_(self.global_delta[-1].weight)
        nn.init.zeros_(self.global_delta[-1].bias)

    def encode_genes(self, priors: PriorInputs | None = None) -> Tensor:
        del priors
        return super().encode_genes(None)

    def _apply_transition(
        self, query: Tensor, context_embedding: Tensor, perturbation_embedding: Tensor
    ) -> tuple[Tensor, Tensor]:
        global_delta = self.global_delta(
            torch.cat((context_embedding, perturbation_embedding), dim=-1)
        )
        delta = global_delta.unsqueeze(1).expand(-1, query.shape[1], -1)
        return query + delta, delta


class LearnedCellStateModel(PriorAwarePerturbationModel):
    """Level 2: cell-specific transition with learned genes, but no priors."""

    def __init__(self, config: ModelConfig, output_gene_index: Tensor | None = None) -> None:
        super().__init__(
            config, enabled_priors=frozenset(), output_gene_index=output_gene_index
        )

    def encode_genes(self, priors: PriorInputs | None = None) -> Tensor:
        del priors
        return super().encode_genes(None)


class FunctionalPriorModel(PriorAwarePerturbationModel):
    """Level 3: add GO, STRING and Reactome, while excluding signed GRN edges."""

    def __init__(self, config: ModelConfig, output_gene_index: Tensor | None = None) -> None:
        super().__init__(
            config,
            enabled_priors=frozenset(("go", "string", "reactome")),
            output_gene_index=output_gene_index,
        )

    def encode_genes(self, priors: PriorInputs | None = None) -> Tensor:
        if priors is None:
            return super().encode_genes(None)
        functional = PriorInputs(
            go=priors.go,
            reactome=priors.reactome,
            string_edge_index=priors.string_edge_index,
            string_edge_weight=priors.string_edge_weight,
        )
        return super().encode_genes(functional)


def build_model(
    level: ArchitectureLevel | str,
    config: ModelConfig,
    output_gene_index: Tensor | None = None,
) -> ControlBaselineModel | PriorAwarePerturbationModel:
    """Construct one architecture tier with a common forward contract."""

    level = ArchitectureLevel(level)
    implementations: dict[ArchitectureLevel, type[nn.Module]] = {
        ArchitectureLevel.CONTROL: ControlBaselineModel,
        ArchitectureLevel.GLOBAL: LearnedGlobalEffectModel,
        ArchitectureLevel.CELL_STATE: LearnedCellStateModel,
        ArchitectureLevel.FUNCTIONAL_PRIOR: FunctionalPriorModel,
    }
    if level is ArchitectureLevel.FULL_PRIOR:
        return PriorAwarePerturbationModel(config, output_gene_index=output_gene_index)
    return implementations[level](config, output_gene_index)  # type: ignore[return-value]
