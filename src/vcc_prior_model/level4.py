"""Level-4 modules: pretrained cell state and context-adaptive biology."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .model import (
    ContextEncoding,
    ControlOutput,
    ModelOutput,
    PriorAwarePerturbationModel,
    PriorInputs,
    apply_library_preserving_effect,
)


class PretrainedCellStateAdapter(nn.Module):
    """Conservatively fuse cached frozen-encoder features with count features.

    The STATE model is deliberately not owned by this module. Its embeddings
    are cached to ``.npy``/AnnData once, making Level-4 DDP training independent
    of the 600M-parameter encoder and its license/runtime dependencies.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Sequential(
            nn.LayerNorm(config.pretrained_cell_dim),
            nn.Linear(config.pretrained_cell_dim, config.cell_dim),
            nn.GELU(),
            nn.LayerNorm(config.cell_dim),
        )
        self.gate = nn.Linear(2 * config.cell_dim, config.cell_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(
            self.gate.bias,
            math.log(config.pretrained_gate_init / (1.0 - config.pretrained_gate_init)),
        )
        self.output_norm = nn.LayerNorm(config.cell_dim)

    def forward(self, count_state: Tensor, pretrained_state: Tensor | None) -> Tensor:
        if pretrained_state is None:
            return count_state
        if pretrained_state.shape[:-1] != count_state.shape[:-1]:
            raise ValueError("pretrained cell embeddings must match [batch, cells]")
        if pretrained_state.shape[-1] != self.config.pretrained_cell_dim:
            raise ValueError(
                f"expected pretrained embedding width {self.config.pretrained_cell_dim}, "
                f"got {pretrained_state.shape[-1]}"
            )
        state = self.projection(pretrained_state.to(count_state.dtype))
        gate = torch.sigmoid(self.gate(torch.cat((count_state, state), dim=-1)))
        return self.output_norm(count_state + gate * state)


class ContextConditionedTargetPrior(nn.Module):
    """Adapt a static target-gene prior to its basal context.

    Only the queried target is adapted, avoiding a dense B x 19k x D tensor.
    The zero-initialised residual makes a Level-3.1 warm start function
    preserving, while gene expression/detection statistics let the same graph
    prior behave differently across cell lines or stimuli.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(config.gene_dim + config.context_dim + 3, config.gene_dim),
            nn.GELU(),
            nn.Linear(config.gene_dim, config.gene_dim),
        )
        self.gate = nn.Linear(config.context_dim + 3, config.gene_dim)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        self.output_norm = nn.LayerNorm(config.gene_dim)

    def forward(self, gene: Tensor, context: Tensor, statistics: Tensor) -> Tensor:
        condition = torch.cat((context, statistics), dim=-1)
        residual = torch.tanh(self.network(torch.cat((gene, condition), dim=-1)))
        gate = torch.sigmoid(self.gate(condition))
        return self.output_norm(gene + self.config.context_prior_scale * gate * residual)


class AdaptiveEffectCalibrator(nn.Module):
    """Learn bounded target/context-specific effect scaling around identity."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(config.context_dim + config.perturbation_dim + 4, config.cell_dim),
            nn.GELU(),
            nn.Linear(config.cell_dim, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self, context: Tensor, perturbation: Tensor, target_stats: Tensor, observed: Tensor
    ) -> Tensor:
        feature = torch.cat(
            (context, perturbation, target_stats, observed.to(context.dtype).unsqueeze(-1)),
            dim=-1,
        )
        return 1.0 + 0.5 * torch.tanh(self.head(feature))


class LowRankPopulationResidual(nn.Module):
    """Low-rank cell-by-gene residual for heterogeneous perturbation response."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.cell_factor = nn.Linear(config.cell_dim, config.population_rank)
        self.gene_factor = nn.Linear(config.gene_dim, config.population_rank, bias=False)
        self.condition = nn.Linear(config.perturbation_dim, config.population_rank)
        nn.init.zeros_(self.cell_factor.weight)
        nn.init.zeros_(self.cell_factor.bias)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)

    def forward(self, cells: Tensor, genes: Tensor, perturbation: Tensor) -> Tensor:
        cell_factor = torch.tanh(
            self.cell_factor(cells) + self.condition(perturbation).unsqueeze(1)
        )
        gene_factor = self.gene_factor(genes)
        return (
            self.config.population_residual_scale
            * torch.einsum("bqr,gr->bqg", cell_factor, gene_factor)
            / math.sqrt(self.config.population_rank)
        )


class DenoisedEmpiricalBaseline(nn.Module):
    """Mix empirical NTC with a small decoder rate to denoise sampling zeros."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        init = config.baseline_decoder_mix_init
        logit = -20.0 if init == 0.0 else math.log(init / (1.0 - init))
        self.global_logit = nn.Parameter(torch.tensor(logit))
        self.evidence = nn.Linear(3, 1)
        nn.init.zeros_(self.evidence.weight)
        nn.init.zeros_(self.evidence.bias)

    def forward(
        self, empirical: Tensor, decoded: Tensor, gene_statistics: Tensor
    ) -> tuple[Tensor, Tensor]:
        probability = torch.sigmoid(
            self.global_logit + self.evidence(gene_statistics).squeeze(-1)
        ).unsqueeze(1)
        return (1.0 - probability) * empirical + probability * decoded, probability


class ContextAdaptivePriorModel(PriorAwarePerturbationModel):
    """Level 4: STATE + context-adaptive priors + calibrated empirical delta."""

    def __init__(self, config: ModelConfig, output_gene_index: Tensor | None = None) -> None:
        super().__init__(config, output_gene_index=output_gene_index)
        self.pretrained_adapter = PretrainedCellStateAdapter(config)
        self.context_target_prior = ContextConditionedTargetPrior(config)
        self.effect_calibrator = AdaptiveEffectCalibrator(config)
        self.population_residual = LowRankPopulationResidual(config)
        self.empirical_baseline = DenoisedEmpiricalBaseline(config)

    def _encode_cells(
        self,
        counts: Tensor,
        genes: Tensor,
        observed_mask: Tensor | None,
        gene_index: Tensor | None,
        pretrained: Tensor | None,
    ) -> Tensor:
        count_state = self.cell_encoder(counts, genes, observed_mask, gene_index)
        return self.pretrained_adapter(count_state, pretrained)

    def encode_context(
        self,
        context_counts: Tensor,
        gene_embeddings: Tensor,
        observed_gene_mask: Tensor | None = None,
        cell_mask: Tensor | None = None,
        input_gene_index: Tensor | None = None,
        pretrained_cell_state: Tensor | None = None,
    ) -> ContextEncoding:
        cells = self._encode_cells(
            context_counts,
            gene_embeddings,
            observed_gene_mask,
            input_gene_index,
            pretrained_cell_state,
        )
        context, prototypes = self.context_encoder(cells, cell_mask)
        panel_stats = self._gene_context_statistics(context_counts, observed_gene_mask, cell_mask)
        gene_stats, gene_observed = self._scatter_context_statistics(
            panel_stats, input_gene_index, gene_embeddings.shape[0]
        )
        return ContextEncoding(context, prototypes, gene_embeddings, gene_stats, gene_observed)

    def encode_context_chunks(
        self,
        count_chunks: Iterable[Tensor],
        gene_embeddings: Tensor,
        observed_gene_mask: Tensor | None = None,
        cell_mask_chunks: Iterable[Tensor | None] | None = None,
        input_gene_index: Tensor | None = None,
        pretrained_cell_state_chunks: Iterable[Tensor | None] | None = None,
    ) -> ContextEncoding:
        """Stream a large NTC set together with row-aligned STATE features."""

        mask_iterator = iter(cell_mask_chunks) if cell_mask_chunks is not None else None
        state_iterator = (
            iter(pretrained_cell_state_chunks)
            if pretrained_cell_state_chunks is not None
            else None
        )
        accumulated = None
        accumulated_genes = None
        for counts in count_chunks:
            mask = None if mask_iterator is None else next(mask_iterator)
            state = None if state_iterator is None else next(state_iterator)
            cells = self._encode_cells(
                counts, gene_embeddings, observed_gene_mask, input_gene_index, state
            )
            statistics = self.context_encoder.statistics(cells, mask)
            accumulated = statistics if accumulated is None else accumulated + statistics
            gene_statistics = self._gene_context_statistics(counts, observed_gene_mask, mask)
            accumulated_genes = (
                gene_statistics
                if accumulated_genes is None
                else accumulated_genes + gene_statistics
            )
        if accumulated is None or accumulated_genes is None:
            raise ValueError("count_chunks must contain at least one tensor")
        context, prototypes = self.context_encoder.finalize(accumulated)
        gene_stats, gene_observed = self._scatter_context_statistics(
            accumulated_genes, input_gene_index, gene_embeddings.shape[0]
        )
        return ContextEncoding(context, prototypes, gene_embeddings, gene_stats, gene_observed)

    def predict_from_context(
        self,
        query_counts: Tensor,
        target_gene_index: Tensor,
        context_encoding: ContextEncoding,
        target_context_stats: Tensor | None = None,
        perturbation_type: Tensor | None = None,
        query_observed_gene_mask: Tensor | None = None,
        gene_chunk_size: int | None = 4096,
        query_gene_index: Tensor | None = None,
        decode_gene_index: Tensor | None = None,
        query_size_factor: Tensor | None = None,
        query_pretrained_cell_state: Tensor | None = None,
        empirical_baseline_counts: Tensor | None = None,
    ) -> ModelOutput:
        batch = query_counts.shape[0]
        if perturbation_type is None:
            perturbation_type = torch.zeros(batch, dtype=torch.long, device=query_counts.device)
        genes = context_encoding.gene_embeddings
        if target_context_stats is None:
            gather = target_gene_index[:, None, None].expand(-1, 1, 3)
            target_context_stats = context_encoding.gene_context_stats.gather(1, gather).squeeze(1)
        target_observed = context_encoding.gene_context_observed.gather(
            1, target_gene_index[:, None]
        ).squeeze(1)
        static_target = genes.index_select(0, target_gene_index)
        target_gene = self.context_target_prior(
            static_target, context_encoding.context_embedding, target_context_stats
        )
        perturbation = self.perturbation_encoder(
            target_gene,
            context_encoding.context_embedding,
            target_context_stats,
            perturbation_type,
        )
        query = self._encode_cells(
            query_counts,
            genes,
            query_observed_gene_mask,
            query_gene_index,
            query_pretrained_cell_state,
        )
        perturbed, delta = self._apply_transition(
            query, context_encoding.context_embedding, perturbation
        )
        size_factor = self._resolve_size_factor(
            query_counts, query_observed_gene_mask, query_size_factor
        )
        output_index = self._resolve_output_gene_index(decode_gene_index, genes.device)
        decoded = self.decoder(
            query,
            delta,
            genes,
            output_index,
            target_gene,
            perturbation,
            size_factor,
            gene_chunk_size,
        )
        (
            _,
            control_mean,
            raw_log_effect,
            dispersion,
            support_logits,
            support_probability,
            sign_logits,
            effect_magnitude,
            effect_strength,
        ) = decoded
        calibration = self.effect_calibrator(
            context_encoding.context_embedding,
            perturbation,
            target_context_stats,
            target_observed,
        )
        selected_genes = genes.index_select(0, output_index)
        population = self.population_residual(query, selected_genes, perturbation)
        log_effect = (
            raw_log_effect * calibration[:, None, :] + population
        ).clamp(-self.config.max_log_effect, self.config.max_log_effect)

        baseline = control_mean
        baseline_probability = None
        if empirical_baseline_counts is not None:
            if empirical_baseline_counts.shape != control_mean.shape:
                raise ValueError("empirical_baseline_counts must match decoded [B, Q, G]")
            gene_stats = context_encoding.gene_context_stats.index_select(1, output_index)
            baseline, baseline_probability = self.empirical_baseline(
                empirical_baseline_counts, control_mean, gene_stats
            )
        mean = apply_library_preserving_effect(baseline, log_effect, self.config.eps)
        return ModelOutput(
            mean=mean,
            control_mean=baseline,
            log_effect=log_effect,
            dispersion=dispersion,
            output_gene_index=output_index,
            context_embedding=context_encoding.context_embedding,
            perturbation_embedding=perturbation,
            query_latent=query,
            delta_latent=delta,
            perturbed_latent=perturbed,
            state_prototypes=context_encoding.state_prototypes,
            effect_support_logits=support_logits,
            effect_support_probability=support_probability,
            effect_sign_logits=sign_logits,
            effect_magnitude=effect_magnitude,
            effect_strength=effect_strength,
            effect_calibration=calibration,
            baseline_mix_probability=baseline_probability,
            population_residual=population,
        )

    def reconstruct_control(
        self,
        query_counts: Tensor,
        priors: PriorInputs | None = None,
        gene_embeddings: Tensor | None = None,
        query_observed_gene_mask: Tensor | None = None,
        query_gene_index: Tensor | None = None,
        decode_gene_index: Tensor | None = None,
        query_size_factor: Tensor | None = None,
        gene_chunk_size: int | None = 4096,
        query_pretrained_cell_state: Tensor | None = None,
    ) -> ControlOutput:
        genes = self.encode_genes(priors) if gene_embeddings is None else gene_embeddings
        cells = self._encode_cells(
            query_counts,
            genes,
            query_observed_gene_mask,
            query_gene_index,
            query_pretrained_cell_state,
        )
        size_factor = self._resolve_size_factor(
            query_counts, query_observed_gene_mask, query_size_factor
        )
        output_index = self._resolve_output_gene_index(decode_gene_index, genes.device)
        mean, dispersion = self.decoder.decode_control(
            cells, genes, output_index, size_factor, gene_chunk_size
        )
        return ControlOutput(mean, dispersion, cells, output_index)

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
        context_pretrained_cell_state: Tensor | None = None,
        query_pretrained_cell_state: Tensor | None = None,
        empirical_baseline_counts: Tensor | None = None,
    ) -> ModelOutput:
        genes = self.encode_genes(priors)
        context = self.encode_context(
            context_counts,
            genes,
            context_observed_gene_mask,
            context_cell_mask,
            context_gene_index,
            context_pretrained_cell_state,
        )
        return self.predict_from_context(
            query_counts,
            target_gene_index,
            context,
            perturbation_type=perturbation_type,
            query_observed_gene_mask=query_observed_gene_mask,
            gene_chunk_size=gene_chunk_size,
            query_gene_index=query_gene_index,
            decode_gene_index=decode_gene_index,
            query_size_factor=query_size_factor,
            query_pretrained_cell_state=query_pretrained_cell_state,
            empirical_baseline_counts=empirical_baseline_counts,
        )
