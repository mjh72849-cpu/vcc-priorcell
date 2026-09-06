"""Prior-aware, context-conditioned perturbation prediction architecture."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig
from .graph import MembershipEncoder, SignedDirectedGraphEncoder, WeightedGraphEncoder


@dataclass(frozen=True)
class MembershipEdges:
    """Sparse memberships for a gene--term bipartite graph."""

    gene_index: Tensor
    term_index: Tensor
    weight: Tensor | None = None

    def to(self, device: torch.device | str) -> MembershipEdges:
        return MembershipEdges(
            self.gene_index.to(device),
            self.term_index.to(device),
            None if self.weight is None else self.weight.to(device),
        )


@dataclass(frozen=True)
class PriorInputs:
    """All optional prior graphs, indexed in the model's gene order.

    ``string_edge_index`` and ``grn_edge_index`` have shape ``[2, E]``.
    STRING edges are treated as undirected.  GRN rows are ``source -> target``;
    ``grn_edge_sign`` must contain positive values for activation and non-positive
    values for inhibition.
    """

    go: MembershipEdges | None = None
    reactome: MembershipEdges | None = None
    string_edge_index: Tensor | None = None
    string_edge_weight: Tensor | None = None
    grn_edge_index: Tensor | None = None
    grn_edge_sign: Tensor | None = None
    grn_edge_weight: Tensor | None = None

    def to(self, device: torch.device | str) -> PriorInputs:
        """Return a copy with every supplied sparse prior tensor on ``device``."""

        def move(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.to(device)

        return PriorInputs(
            go=None if self.go is None else self.go.to(device),
            reactome=None if self.reactome is None else self.reactome.to(device),
            string_edge_index=move(self.string_edge_index),
            string_edge_weight=move(self.string_edge_weight),
            grn_edge_index=move(self.grn_edge_index),
            grn_edge_sign=move(self.grn_edge_sign),
            grn_edge_weight=move(self.grn_edge_weight),
        )


@dataclass
class PrototypeStatistics:
    """Additive sufficient statistics for streaming set aggregation."""

    feature_sum: Tensor
    cell_count: Tensor
    prototype_sum: Tensor
    prototype_mass: Tensor

    def __add__(self, other: PrototypeStatistics) -> PrototypeStatistics:
        return PrototypeStatistics(
            self.feature_sum + other.feature_sum,
            self.cell_count + other.cell_count,
            self.prototype_sum + other.prototype_sum,
            self.prototype_mass + other.prototype_mass,
        )


@dataclass
class GeneContextStatistics:
    """Additive per-gene statistics accumulated over context cells."""

    value_sum: Tensor
    squared_sum: Tensor
    detected_sum: Tensor
    observation_count: Tensor

    def __add__(self, other: GeneContextStatistics) -> GeneContextStatistics:
        return GeneContextStatistics(
            self.value_sum + other.value_sum,
            self.squared_sum + other.squared_sum,
            self.detected_sum + other.detected_sum,
            self.observation_count + other.observation_count,
        )

    def finalize(self) -> Tensor:
        count = self.observation_count.clamp_min(1.0)
        mean = self.value_sum / count
        variance = (self.squared_sum / count - mean.square()).clamp_min(0.0)
        detection_rate = self.detected_sum / count
        return torch.stack((mean, torch.log1p(variance), detection_rate), dim=-1)


@dataclass(frozen=True)
class ContextEncoding:
    """Reusable encoding of one or more NTC cell sets."""

    context_embedding: Tensor
    state_prototypes: Tensor
    gene_embeddings: Tensor
    gene_context_stats: Tensor
    gene_context_observed: Tensor


@dataclass(frozen=True)
class ControlOutput:
    """Negative-binomial parameters for NTC reconstruction."""

    mean: Tensor
    dispersion: Tensor
    cell_latent: Tensor
    output_gene_index: Tensor


@dataclass(frozen=True)
class ModelOutput:
    """Parameters and latent states produced by a model forward pass."""

    mean: Tensor
    control_mean: Tensor
    log_effect: Tensor
    dispersion: Tensor
    output_gene_index: Tensor
    context_embedding: Tensor
    perturbation_embedding: Tensor
    query_latent: Tensor
    delta_latent: Tensor
    perturbed_latent: Tensor
    state_prototypes: Tensor


class GenePriorEncoder(nn.Module):
    """Fuse learned gene identities with separately encoded biological priors."""

    PRIOR_NAMES = ("go", "string", "reactome", "grn")

    def __init__(
        self, config: ModelConfig, enabled_priors: frozenset[str] | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.enabled_priors = (
            frozenset(self.PRIOR_NAMES) if enabled_priors is None else enabled_priors
        )
        unknown = self.enabled_priors.difference(self.PRIOR_NAMES)
        if unknown:
            raise ValueError(f"unknown prior sources: {sorted(unknown)}")
        dim = config.gene_dim
        self.base_embedding = nn.Embedding(config.num_genes, dim)
        nn.init.normal_(self.base_embedding.weight, std=0.02)

        self.go_encoder = (
            MembershipEncoder(config.num_go_terms, dim, config.dropout, config.eps)
            if "go" in self.enabled_priors and config.num_go_terms > 0
            else None
        )
        self.reactome_encoder = (
            MembershipEncoder(config.num_pathway_terms, dim, config.dropout, config.eps)
            if "reactome" in self.enabled_priors and config.num_pathway_terms > 0
            else None
        )
        self.string_encoder = (
            WeightedGraphEncoder(dim, config.graph_layers, config.dropout, config.eps)
            if "string" in self.enabled_priors
            else None
        )
        self.grn_encoder = (
            SignedDirectedGraphEncoder(
                dim, config.graph_layers, config.dropout, config.eps
            )
            if "grn" in self.enabled_priors
            else None
        )
        self.prior_projection = nn.ModuleDict(
            {name: nn.Linear(dim, dim, bias=False) for name in self.enabled_priors}
        )
        # A higher-level model initialised from Level 2 must begin as the exact
        # Level-2 function. Prior residuals then enter gradually through learned
        # projections instead of perturbing a good checkpoint at step zero.
        for projection in self.prior_projection.values():
            nn.init.zeros_(projection.weight)
        self.gates = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(2 * dim + 1, dim // 2),
                    nn.GELU(),
                    nn.Linear(dim // 2, 1),
                )
                for name in self.enabled_priors
            }
        )
        # Begin with a base-embedding-dominant model. Zero final weights make
        # every active gate sigmoid(prior_gate_init_bias) at initialisation;
        # with the default -2 this is approximately 0.119.
        for gate in self.gates.values():
            nn.init.zeros_(gate[-1].weight)
            nn.init.constant_(gate[-1].bias, config.prior_gate_init_bias)
        self.output_norm = nn.LayerNorm(dim)

    def _empty_prior(self, base: Tensor) -> tuple[Tensor, Tensor]:
        return torch.zeros_like(base), torch.zeros(
            base.shape[0], dtype=torch.bool, device=base.device
        )

    def _source_dropout(self, value: Tensor) -> Tensor:
        if not self.training or self.config.prior_dropout == 0.0:
            return value
        keep_probability = 1.0 - self.config.prior_dropout
        keep = torch.empty((), device=value.device).bernoulli_(keep_probability)
        return value * keep / keep_probability

    def _normalize_prior_sum(self, prior_sum: Tensor, active_count: Tensor) -> Tensor:
        """Control contribution growth for genes covered by many prior sources."""

        if not self.config.normalize_active_priors:
            return prior_sum
        return prior_sum / active_count.clamp_min(1.0).sqrt()

    def forward(self, priors: PriorInputs | None = None) -> Tensor:
        priors = priors or PriorInputs()
        base = self.base_embedding.weight
        sources: dict[str, tuple[Tensor, Tensor]] = {}

        if "go" in self.enabled_priors and self.go_encoder is not None and priors.go is not None:
            sources["go"] = self.go_encoder(
                base, priors.go.gene_index, priors.go.term_index, priors.go.weight
            )
        elif "go" in self.enabled_priors:
            sources["go"] = self._empty_prior(base)

        if self.string_encoder is not None and priors.string_edge_index is not None:
            sources["string"] = self.string_encoder(
                base, priors.string_edge_index, priors.string_edge_weight
            )
        elif "string" in self.enabled_priors:
            sources["string"] = self._empty_prior(base)

        if (
            "reactome" in self.enabled_priors
            and self.reactome_encoder is not None
            and priors.reactome is not None
        ):
            sources["reactome"] = self.reactome_encoder(
                base,
                priors.reactome.gene_index,
                priors.reactome.term_index,
                priors.reactome.weight,
            )
        elif "reactome" in self.enabled_priors:
            sources["reactome"] = self._empty_prior(base)

        if self.grn_encoder is not None and priors.grn_edge_index is not None:
            if priors.grn_edge_sign is None:
                raise ValueError("grn_edge_sign is required when grn_edge_index is provided")
            sources["grn"] = self.grn_encoder(
                base,
                priors.grn_edge_index,
                priors.grn_edge_sign,
                priors.grn_edge_weight,
            )
        elif "grn" in self.enabled_priors:
            sources["grn"] = self._empty_prior(base)

        prior_sum = torch.zeros_like(base)
        active_count = base.new_zeros((base.shape[0], 1))
        for name in self.PRIOR_NAMES:
            if name not in self.enabled_priors:
                continue
            prior_state, mask = sources[name]
            projected = self.prior_projection[name](prior_state)
            mask_column = mask.to(base.dtype).unsqueeze(-1)
            gate_input = torch.cat((base, projected, mask_column), dim=-1)
            gate = torch.sigmoid(self.gates[name](gate_input)) * mask_column
            prior_sum = prior_sum + self._source_dropout(gate * projected)
            active_count = active_count + mask_column
        fused = base + self._normalize_prior_sum(prior_sum, active_count)
        return self.output_norm(fused)


class ExpressionCellEncoder(nn.Module):
    """Project a count vector through shared gene embeddings."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.cell_network = nn.Sequential(
            nn.Linear(config.gene_dim + 2, config.cell_dim),
            nn.GELU(),
            nn.LayerNorm(config.cell_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.cell_dim, config.cell_dim),
            nn.GELU(),
            nn.LayerNorm(config.cell_dim),
        )

    def normalize(self, counts: Tensor, observed_gene_mask: Tensor | None = None) -> Tensor:
        if observed_gene_mask is not None:
            observed_gene_mask = self._broadcast_gene_mask(observed_gene_mask, counts)
            counts = counts * observed_gene_mask.to(dtype=counts.dtype)
        library = counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.log1p(counts * (10_000.0 / library))

    @staticmethod
    def _broadcast_gene_mask(mask: Tensor, counts: Tensor) -> Tensor:
        """Accept masks shaped [G], [B, G], or already broadcastable to counts."""

        if mask.ndim == 2 and counts.ndim == 3 and mask.shape[0] == counts.shape[0]:
            return mask.unsqueeze(1)
        return mask

    def forward(
        self,
        counts: Tensor,
        gene_embeddings: Tensor,
        observed_gene_mask: Tensor | None = None,
        input_gene_index: Tensor | None = None,
    ) -> Tensor:
        if input_gene_index is None:
            selected_gene_embeddings = gene_embeddings
        else:
            input_gene_index = input_gene_index.to(device=gene_embeddings.device)
            selected_gene_embeddings = gene_embeddings.index_select(0, input_gene_index)
        if counts.shape[-1] != selected_gene_embeddings.shape[0]:
            raise ValueError(
                f"count dimension {counts.shape[-1]} != number of genes "
                f"selected for this dataset ({selected_gene_embeddings.shape[0]})"
            )
        normalized = self.normalize(counts, observed_gene_mask)
        pooled = normalized @ selected_gene_embeddings
        pooled = pooled / normalized.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
        library = torch.log1p(counts.sum(dim=-1, keepdim=True))
        detected = (counts > 0).to(counts.dtype)
        if observed_gene_mask is not None:
            observed_gene_mask = self._broadcast_gene_mask(observed_gene_mask, counts)
            detected = detected * observed_gene_mask.to(counts.dtype)
            denominator = observed_gene_mask.to(counts.dtype).sum(dim=-1, keepdim=True)
        else:
            denominator = counts.new_tensor(counts.shape[-1])
        detected_fraction = detected.sum(dim=-1, keepdim=True) / denominator.clamp_min(1.0)
        return self.cell_network(torch.cat((pooled, library, detected_fraction), dim=-1))


class PrototypeSetEncoder(nn.Module):
    """Permutation-invariant, linearly scaling context-set encoder.

    Learned prototype queries softly partition cells into state summaries.  Its
    sufficient statistics are additive, enabling exact chunked encoding of all
    18,400 context controls without retaining every cell latent on the GPU.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.cell_dim
        self.phi = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim), nn.Dropout(config.dropout)
        )
        self.assignment_prototypes = nn.Parameter(
            torch.randn(config.context_prototypes, dim) / math.sqrt(dim)
        )
        self.output = nn.Sequential(
            nn.Linear((config.context_prototypes + 1) * dim, config.context_dim),
            nn.GELU(),
            nn.LayerNorm(config.context_dim),
        )

    def statistics(self, cells: Tensor, cell_mask: Tensor | None = None) -> PrototypeStatistics:
        features = self.phi(cells)
        batch, num_cells, _ = features.shape
        if cell_mask is None:
            cell_mask = torch.ones(
                (batch, num_cells), dtype=torch.bool, device=features.device
            )
        mask = cell_mask.to(features.dtype)
        logits = torch.einsum(
            "bnd,kd->bnk",
            F.normalize(features, dim=-1),
            F.normalize(self.assignment_prototypes, dim=-1),
        ) / self.config.prototype_temperature
        assignment = torch.softmax(logits, dim=-1) * mask.unsqueeze(-1)
        return PrototypeStatistics(
            feature_sum=torch.einsum("bn,bnd->bd", mask, features),
            cell_count=mask.sum(dim=1, keepdim=True),
            prototype_sum=torch.einsum("bnk,bnd->bkd", assignment, features),
            prototype_mass=assignment.sum(dim=1),
        )

    def finalize(self, stats: PrototypeStatistics) -> tuple[Tensor, Tensor]:
        mean = stats.feature_sum / stats.cell_count.clamp_min(1.0)
        prototypes = stats.prototype_sum / stats.prototype_mass.clamp_min(
            self.config.eps
        ).unsqueeze(-1)
        flat = torch.cat((mean, prototypes.flatten(start_dim=1)), dim=-1)
        return self.output(flat), prototypes

    def forward(self, cells: Tensor, cell_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        return self.finalize(self.statistics(cells, cell_mask))


class BiologicalPerturbationEncoder(nn.Module):
    """Condition a target-gene representation on context and basal target state."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(
            config.num_perturbation_types, config.gene_dim // 4
        )
        input_dim = config.gene_dim + config.context_dim + 3 + config.gene_dim // 4
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.perturbation_dim),
            nn.GELU(),
            nn.LayerNorm(config.perturbation_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.perturbation_dim, config.perturbation_dim),
            nn.GELU(),
            nn.LayerNorm(config.perturbation_dim),
        )

    def forward(
        self,
        target_gene_embedding: Tensor,
        context_embedding: Tensor,
        target_context_stats: Tensor,
        perturbation_type: Tensor,
    ) -> Tensor:
        type_embedding = self.type_embedding(perturbation_type)
        return self.network(
            torch.cat(
                (target_gene_embedding, context_embedding, target_context_stats, type_embedding),
                dim=-1,
            )
        )


class ContextConditionedTransition(nn.Module):
    """A FiLM-modulated residual transition that preserves cell heterogeneity."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.cell_norm = nn.LayerNorm(config.cell_dim)
        self.condition = nn.Sequential(
            nn.Linear(config.context_dim + config.perturbation_dim, 2 * config.cell_dim),
            nn.GELU(),
            nn.Linear(2 * config.cell_dim, 2 * config.cell_dim),
        )
        self.delta = nn.Sequential(
            nn.Linear(config.cell_dim + config.perturbation_dim, config.cell_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.cell_dim, config.cell_dim),
        )
        # Start from identity, so an untrained model does not invent a large effect.
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(
        self, cells: Tensor, context_embedding: Tensor, perturbation_embedding: Tensor
    ) -> tuple[Tensor, Tensor]:
        condition = torch.cat((context_embedding, perturbation_embedding), dim=-1)
        gamma, beta = self.condition(condition).chunk(2, dim=-1)
        modulated = self.cell_norm(cells) * (1.0 + 0.1 * torch.tanh(gamma).unsqueeze(1))
        modulated = modulated + beta.unsqueeze(1)
        perturbation = perturbation_embedding.unsqueeze(1).expand(-1, cells.shape[1], -1)
        delta = self.delta(torch.cat((modulated, perturbation), dim=-1))
        return cells + delta, delta


class GeneAwareNegativeBinomialDecoder(nn.Module):
    """Independent control rates plus a zero-initialised residual perturbation.

    Unlike a whole-panel softmax decoder, each gene's log rate is independent of
    which other genes were measured by a dataset. This is essential when public
    training datasets have different feature panels.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.control_cell_projection = nn.Linear(
            config.cell_dim, config.decoder_dim, bias=False
        )
        self.control_gene_projection = nn.Linear(
            config.gene_dim, config.decoder_dim, bias=False
        )
        self.effect_cell_projection = nn.Linear(
            config.cell_dim, config.decoder_dim, bias=False
        )
        self.effect_gene_projection = nn.Linear(
            config.gene_dim, config.decoder_dim, bias=False
        )
        self.perturbation_projection = nn.Linear(
            config.perturbation_dim, config.decoder_dim, bias=False
        )
        self.target_relation = nn.Linear(config.gene_dim, config.decoder_dim, bias=False)
        self.gene_bias = nn.Parameter(torch.zeros(config.num_genes))
        self.raw_dispersion = nn.Parameter(torch.full((config.num_genes,), 2.0))

        # At initialisation KD mean == control mean. Non-zero effects must be
        # supported by perturbation supervision rather than random decoder noise.
        # ``effect_cell_projection`` intentionally keeps its ordinary random
        # initialization. The transition already emits an exact zero residual,
        # and a non-zero downstream Jacobian lets that zero-initialized layer
        # receive gradients on the first optimization step.
        for layer in (self.perturbation_projection, self.target_relation):
            nn.init.zeros_(layer.weight)

    def _control_parameters(
        self,
        cells: Tensor,
        selected_embeddings: Tensor,
        selected_gene_index: Tensor,
        size_factor: Tensor,
        gene_chunk_size: int | None,
    ) -> tuple[Tensor, Tensor]:
        cell_query = self.control_cell_projection(cells)
        chunk_size = gene_chunk_size or selected_embeddings.shape[0]
        log_rate_parts = []
        for start in range(0, selected_embeddings.shape[0], chunk_size):
            stop = min(start + chunk_size, selected_embeddings.shape[0])
            gene_key = self.control_gene_projection(selected_embeddings[start:stop])
            score = torch.einsum("bqd,gd->bqg", cell_query, gene_key)
            bias = self.gene_bias.index_select(0, selected_gene_index[start:stop])
            log_rate_parts.append(score / math.sqrt(self.config.decoder_dim) + bias)
        log_rate = torch.cat(log_rate_parts, dim=-1)
        log_rate = log_rate + torch.log(size_factor.clamp_min(self.config.eps))
        control_mean = torch.exp(log_rate.clamp(min=-12.0, max=12.0))
        dispersion = F.softplus(
            self.raw_dispersion.index_select(0, selected_gene_index)
        ) + self.config.min_dispersion
        return control_mean, dispersion

    def decode_control(
        self,
        cells: Tensor,
        gene_embeddings: Tensor,
        output_gene_index: Tensor,
        size_factor: Tensor,
        gene_chunk_size: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        selected = gene_embeddings.index_select(0, output_gene_index)
        return self._control_parameters(
            cells, selected, output_gene_index, size_factor, gene_chunk_size
        )

    def forward(
        self,
        query_cells: Tensor,
        delta_cells: Tensor,
        gene_embeddings: Tensor,
        output_gene_index: Tensor,
        target_gene_embedding: Tensor,
        perturbation_embedding: Tensor,
        size_factor: Tensor,
        gene_chunk_size: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        selected = gene_embeddings.index_select(0, output_gene_index)
        control_mean, dispersion = self._control_parameters(
            query_cells, selected, output_gene_index, size_factor, gene_chunk_size
        )
        effect_query = self.effect_cell_projection(delta_cells)
        effect_query = effect_query + self.perturbation_projection(
            perturbation_embedding
        ).unsqueeze(1)
        target_query = self.target_relation(target_gene_embedding)

        chunk_size = gene_chunk_size or selected.shape[0]
        effect_parts = []
        for start in range(0, selected.shape[0], chunk_size):
            stop = min(start + chunk_size, selected.shape[0])
            gene_key = self.effect_gene_projection(selected[start:stop])
            cell_effect = torch.einsum("bqd,gd->bqg", effect_query, gene_key)
            relation = torch.einsum("bd,gd->bg", target_query, gene_key).unsqueeze(1)
            effect_parts.append((cell_effect + relation) / math.sqrt(self.config.decoder_dim))
        raw_effect = torch.cat(effect_parts, dim=-1)
        log_effect = self.config.max_log_effect * torch.tanh(raw_effect)
        mean = control_mean * torch.exp(log_effect)
        return mean, control_mean, log_effect, dispersion


class PriorAwarePerturbationModel(nn.Module):
    """End-to-end architecture from NTC sets and a target gene to NB counts.

    The class exposes a staged API so a context can be encoded once and reused
    for many perturbation targets.  No method performs data loading or training.
    """

    def __init__(
        self,
        config: ModelConfig,
        enabled_priors: frozenset[str] | None = None,
        output_gene_index: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if output_gene_index is None:
            if config.output_genes != config.num_genes:
                raise ValueError(
                    "output_gene_index is required when num_output_genes differs from num_genes"
                )
            output_gene_index = torch.arange(config.num_genes)
        output_gene_index = output_gene_index.to(dtype=torch.long)
        if output_gene_index.ndim != 1 or output_gene_index.numel() != config.output_genes:
            raise ValueError(
                f"output_gene_index must have shape [{config.output_genes}]"
            )
        if output_gene_index.unique().numel() != output_gene_index.numel():
            raise ValueError("output_gene_index must not contain duplicates")
        if output_gene_index.numel() and (
            output_gene_index.min() < 0 or output_gene_index.max() >= config.num_genes
        ):
            raise ValueError("output_gene_index contains an index outside the global vocabulary")
        self.register_buffer("output_gene_index", output_gene_index, persistent=True)
        self.gene_encoder = GenePriorEncoder(config, enabled_priors)
        self.cell_encoder = ExpressionCellEncoder(config)
        self.context_encoder = PrototypeSetEncoder(config)
        self.perturbation_encoder = BiologicalPerturbationEncoder(config)
        self.transition = ContextConditionedTransition(config)
        self.decoder = GeneAwareNegativeBinomialDecoder(config)

    def encode_genes(self, priors: PriorInputs | None = None) -> Tensor:
        return self.gene_encoder(priors)

    def encode_context(
        self,
        context_counts: Tensor,
        gene_embeddings: Tensor,
        observed_gene_mask: Tensor | None = None,
        cell_mask: Tensor | None = None,
        input_gene_index: Tensor | None = None,
    ) -> ContextEncoding:
        cells = self.cell_encoder(
            context_counts, gene_embeddings, observed_gene_mask, input_gene_index
        )
        context, prototypes = self.context_encoder(cells, cell_mask)
        panel_stats = self._gene_context_statistics(
            context_counts, observed_gene_mask, cell_mask
        )
        gene_stats, gene_observed = self._scatter_context_statistics(
            panel_stats, input_gene_index, gene_embeddings.shape[0]
        )
        return ContextEncoding(
            context, prototypes, gene_embeddings, gene_stats, gene_observed
        )

    def encode_context_chunks(
        self,
        count_chunks: Iterable[Tensor],
        gene_embeddings: Tensor,
        observed_gene_mask: Tensor | None = None,
        cell_mask_chunks: Iterable[Tensor | None] | None = None,
        input_gene_index: Tensor | None = None,
    ) -> ContextEncoding:
        """Encode a large context exactly from chunks along the cell dimension."""

        accumulated: PrototypeStatistics | None = None
        accumulated_genes: GeneContextStatistics | None = None
        for counts, mask in _pair_chunks(count_chunks, cell_mask_chunks):
            cells = self.cell_encoder(
                counts, gene_embeddings, observed_gene_mask, input_gene_index
            )
            stats = self.context_encoder.statistics(cells, mask)
            accumulated = stats if accumulated is None else accumulated + stats
            gene_stats = self._gene_context_statistics(counts, observed_gene_mask, mask)
            accumulated_genes = (
                gene_stats if accumulated_genes is None else accumulated_genes + gene_stats
            )
        if accumulated is None:
            raise ValueError("count_chunks must contain at least one tensor")
        if accumulated_genes is None:  # Kept explicit for static type checkers.
            raise RuntimeError("gene statistics were not accumulated")
        context, prototypes = self.context_encoder.finalize(accumulated)
        gene_stats, gene_observed = self._scatter_context_statistics(
            accumulated_genes, input_gene_index, gene_embeddings.shape[0]
        )
        return ContextEncoding(
            context, prototypes, gene_embeddings, gene_stats, gene_observed
        )

    def _scatter_context_statistics(
        self,
        statistics: GeneContextStatistics,
        input_gene_index: Tensor | None,
        global_gene_count: int,
    ) -> tuple[Tensor, Tensor]:
        panel_stats = statistics.finalize()
        panel_observed = statistics.observation_count > 0
        if input_gene_index is None:
            if panel_stats.shape[1] != global_gene_count:
                raise ValueError(
                    "input_gene_index is required when a context panel does not match "
                    "the global gene vocabulary"
                )
            return panel_stats, panel_observed
        input_gene_index = input_gene_index.to(device=panel_stats.device)
        if input_gene_index.numel() != panel_stats.shape[1]:
            raise ValueError("input_gene_index length does not match the context gene panel")
        global_stats = panel_stats.new_zeros(
            (panel_stats.shape[0], global_gene_count, panel_stats.shape[-1])
        )
        global_observed = torch.zeros(
            (panel_stats.shape[0], global_gene_count),
            dtype=torch.bool,
            device=panel_stats.device,
        )
        global_stats.index_copy_(1, input_gene_index, panel_stats)
        global_observed.index_copy_(1, input_gene_index, panel_observed)
        return global_stats, global_observed

    def _gene_context_statistics(
        self,
        context_counts: Tensor,
        observed_gene_mask: Tensor | None,
        cell_mask: Tensor | None,
    ) -> GeneContextStatistics:
        normalized = self.cell_encoder.normalize(context_counts, observed_gene_mask)
        if cell_mask is None:
            valid = torch.ones_like(context_counts[..., :1])
        else:
            valid = cell_mask.to(context_counts.dtype).unsqueeze(-1)
        if observed_gene_mask is not None:
            gene_mask = self.cell_encoder._broadcast_gene_mask(
                observed_gene_mask, context_counts
            ).to(context_counts.dtype)
            valid = valid * gene_mask
        # Expand only logically; broadcasting avoids materialising a cell x gene mask.
        observation_count = valid.sum(dim=1)
        if observation_count.shape[-1] == 1:
            observation_count = observation_count.expand(-1, context_counts.shape[-1])
        return GeneContextStatistics(
            value_sum=(normalized * valid).sum(dim=1),
            squared_sum=(normalized.square() * valid).sum(dim=1),
            detected_sum=((context_counts > 0).to(context_counts.dtype) * valid).sum(dim=1),
            observation_count=observation_count,
        )

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
    ) -> ModelOutput:
        batch = query_counts.shape[0]
        if perturbation_type is None:
            perturbation_type = torch.zeros(
                batch, dtype=torch.long, device=query_counts.device
            )
        genes = context_encoding.gene_embeddings
        target_gene = genes.index_select(0, target_gene_index)
        if target_context_stats is None:
            gather_index = target_gene_index[:, None, None].expand(-1, 1, 3)
            target_context_stats = context_encoding.gene_context_stats.gather(
                1, gather_index
            ).squeeze(1)
        perturbation = self.perturbation_encoder(
            target_gene,
            context_encoding.context_embedding,
            target_context_stats,
            perturbation_type,
        )
        query = self.cell_encoder(
            query_counts, genes, query_observed_gene_mask, query_gene_index
        )
        perturbed, delta = self._apply_transition(
            query, context_encoding.context_embedding, perturbation
        )
        size_factor = self._resolve_size_factor(
            query_counts, query_observed_gene_mask, query_size_factor
        )
        output_index = self._resolve_output_gene_index(decode_gene_index, genes.device)
        mean, control_mean, log_effect, dispersion = self.decoder(
            query,
            delta,
            genes,
            output_index,
            target_gene,
            perturbation,
            size_factor,
            gene_chunk_size,
        )
        return ModelOutput(
            mean=mean,
            control_mean=control_mean,
            log_effect=log_effect,
            dispersion=dispersion,
            output_gene_index=output_index,
            context_embedding=context_encoding.context_embedding,
            perturbation_embedding=perturbation,
            query_latent=query,
            delta_latent=delta,
            perturbed_latent=perturbed,
            state_prototypes=context_encoding.state_prototypes,
        )

    def _apply_transition(
        self, query: Tensor, context_embedding: Tensor, perturbation_embedding: Tensor
    ) -> tuple[Tensor, Tensor]:
        return self.transition(query, context_embedding, perturbation_embedding)

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
    ) -> ControlOutput:
        """Reconstruct NTC cells without constructing a fake perturbation."""

        genes = self.encode_genes(priors) if gene_embeddings is None else gene_embeddings
        cells = self.cell_encoder(
            query_counts, genes, query_observed_gene_mask, query_gene_index
        )
        size_factor = self._resolve_size_factor(
            query_counts, query_observed_gene_mask, query_size_factor
        )
        output_index = self._resolve_output_gene_index(decode_gene_index, genes.device)
        mean, dispersion = self.decoder.decode_control(
            cells, genes, output_index, size_factor, gene_chunk_size
        )
        return ControlOutput(mean, dispersion, cells, output_index)

    def _resolve_output_gene_index(
        self, decode_gene_index: Tensor | None, device: torch.device
    ) -> Tensor:
        index = self.output_gene_index if decode_gene_index is None else decode_gene_index
        index = index.to(device=device, dtype=torch.long)
        if index.ndim != 1:
            raise ValueError("decode_gene_index must be one-dimensional")
        if index.numel() and (index.min() < 0 or index.max() >= self.config.num_genes):
            raise ValueError("decode_gene_index contains an index outside the global vocabulary")
        return index

    def _resolve_size_factor(
        self,
        counts: Tensor,
        observed_gene_mask: Tensor | None,
        supplied_size_factor: Tensor | None,
    ) -> Tensor:
        if supplied_size_factor is not None:
            if supplied_size_factor.shape != counts.shape[:-1] + (1,):
                raise ValueError("query_size_factor must have shape [B, Q, 1]")
            return supplied_size_factor
        if observed_gene_mask is None:
            observed_count: Tensor | float = float(counts.shape[-1])
            masked_counts = counts
        else:
            mask = self.cell_encoder._broadcast_gene_mask(observed_gene_mask, counts).to(
                counts.dtype
            )
            observed_count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            masked_counts = counts * mask
        return masked_counts.sum(dim=-1, keepdim=True) / observed_count

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
        genes = self.encode_genes(priors)
        context = self.encode_context(
            context_counts,
            genes,
            context_observed_gene_mask,
            context_cell_mask,
            context_gene_index,
        )
        return self.predict_from_context(
            query_counts,
            target_gene_index,
            context,
            target_context_stats=None,
            perturbation_type=perturbation_type,
            query_observed_gene_mask=query_observed_gene_mask,
            gene_chunk_size=gene_chunk_size,
            query_gene_index=query_gene_index,
            decode_gene_index=decode_gene_index,
            query_size_factor=query_size_factor,
        )

    @torch.no_grad()
    def sample_counts(self, output: ModelOutput) -> Tensor:
        """Sample integer raw counts from the decoded negative-binomial model."""

        theta = output.dispersion.view(1, 1, -1)
        probability = output.mean / (theta + output.mean)
        distribution = torch.distributions.NegativeBinomial(
            total_count=theta, probs=probability.clamp(max=1.0 - self.config.eps)
        )
        return distribution.sample().to(torch.int64)


def _repeat_none() -> Iterable[None]:
    while True:
        yield None


def _pair_chunks(
    count_chunks: Iterable[Tensor], cell_mask_chunks: Iterable[Tensor | None] | None
) -> Iterable[tuple[Tensor, Tensor | None]]:
    if cell_mask_chunks is None:
        yield from zip(count_chunks, _repeat_none(), strict=False)
    else:
        yield from zip(count_chunks, cell_mask_chunks, strict=True)
