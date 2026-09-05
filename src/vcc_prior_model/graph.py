"""Small sparse graph modules implemented with core PyTorch operations."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _weighted_mean(
    values: Tensor,
    source_index: Tensor,
    destination_index: Tensor,
    destination_count: int,
    weight: Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Aggregate source rows into destination rows with a weighted mean."""

    if source_index.numel() == 0:
        output = values.new_zeros((destination_count, values.shape[-1]))
        mass = values.new_zeros(destination_count)
        return output, mass

    if weight is None:
        weight = values.new_ones(source_index.shape[0])
    else:
        weight = weight.to(device=values.device, dtype=values.dtype)

    messages = values.index_select(0, source_index) * weight.unsqueeze(-1)
    output = values.new_zeros((destination_count, values.shape[-1]))
    output.index_add_(0, destination_index, messages)
    mass = values.new_zeros(destination_count)
    mass.index_add_(0, destination_index, weight.abs())
    output = output / mass.clamp_min(eps).unsqueeze(-1)
    return output, mass


class MembershipEncoder(nn.Module):
    """Encode a gene--term bipartite graph (GO or Reactome).

    Term nodes receive both a learned identity and the mean of their member
    genes.  The updated term states are then pooled back to genes.  This is a
    lightweight hypergraph layer and avoids expanding a term into a gene clique.
    """

    def __init__(self, num_terms: int, dim: int, dropout: float, eps: float) -> None:
        super().__init__()
        self.num_terms = num_terms
        self.eps = eps
        self.term_embedding = nn.Embedding(num_terms, dim)
        self.gene_to_term = nn.Linear(dim, dim, bias=False)
        self.output = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
        )
        nn.init.normal_(self.term_embedding.weight, std=0.02)

    def forward(
        self,
        gene_state: Tensor,
        gene_index: Tensor,
        term_index: Tensor,
        weight: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        term_from_genes, _ = _weighted_mean(
            gene_state,
            gene_index,
            term_index,
            self.num_terms,
            weight,
            self.eps,
        )
        term_state = self.term_embedding.weight + self.gene_to_term(term_from_genes)
        gene_from_terms, gene_mass = _weighted_mean(
            term_state,
            term_index,
            gene_index,
            gene_state.shape[0],
            weight,
            self.eps,
        )
        return self.output(gene_from_terms), gene_mass > 0


class WeightedGraphLayer(nn.Module):
    """A weighted GraphSAGE-style layer for an undirected graph."""

    def __init__(self, dim: int, dropout: float, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.self_linear = nn.Linear(dim, dim, bias=False)
        self.neighbour_linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, state: Tensor, edge_index: Tensor, edge_weight: Tensor | None
    ) -> Tensor:
        source, destination = edge_index
        both_source = torch.cat((source, destination))
        both_destination = torch.cat((destination, source))
        both_weight = None if edge_weight is None else torch.cat((edge_weight, edge_weight))
        neighbours, _ = _weighted_mean(
            state,
            both_source,
            both_destination,
            state.shape[0],
            both_weight,
            self.eps,
        )
        update = self.self_linear(state) + self.neighbour_linear(neighbours)
        return self.norm(state + self.dropout(torch.nn.functional.gelu(update)))


class WeightedGraphEncoder(nn.Module):
    """Multi-layer encoder for STRING-style weighted functional edges."""

    def __init__(self, dim: int, layers: int, dropout: float, eps: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            WeightedGraphLayer(dim, dropout, eps) for _ in range(layers)
        )

    def forward(
        self, state: Tensor, edge_index: Tensor, edge_weight: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        source, destination = edge_index
        degree = state.new_zeros(state.shape[0])
        ones = state.new_ones(source.shape[0])
        degree.index_add_(0, source, ones)
        degree.index_add_(0, destination, ones)
        for layer in self.layers:
            state = layer(state, edge_index, edge_weight)
        return state, degree > 0


class SignedDirectedGraphLayer(nn.Module):
    """One layer over activation/inhibition and incoming/outgoing GRN relations."""

    RELATIONS = ("incoming_pos", "incoming_neg", "outgoing_pos", "outgoing_neg")

    def __init__(self, dim: int, dropout: float, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.self_linear = nn.Linear(dim, dim, bias=False)
        self.relation_linear = nn.ModuleDict(
            {name: nn.Linear(dim, dim, bias=False) for name in self.RELATIONS}
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        state: Tensor,
        edge_index: Tensor,
        edge_sign: Tensor,
        edge_weight: Tensor | None,
    ) -> Tensor:
        source, target = edge_index
        positive = edge_sign > 0
        relation_specs = {
            "incoming_pos": (source[positive], target[positive], positive),
            "incoming_neg": (source[~positive], target[~positive], ~positive),
            "outgoing_pos": (target[positive], source[positive], positive),
            "outgoing_neg": (target[~positive], source[~positive], ~positive),
        }
        update = self.self_linear(state)
        for name, (rel_source, rel_target, selector) in relation_specs.items():
            rel_weight = None if edge_weight is None else edge_weight[selector]
            aggregated, _ = _weighted_mean(
                state,
                rel_source,
                rel_target,
                state.shape[0],
                rel_weight,
                self.eps,
            )
            update = update + self.relation_linear[name](aggregated)
        return self.norm(state + self.dropout(torch.nn.functional.gelu(update)))


class SignedDirectedGraphEncoder(nn.Module):
    """Multi-layer encoder for CollecTRI-style signed directed edges."""

    def __init__(self, dim: int, layers: int, dropout: float, eps: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            SignedDirectedGraphLayer(dim, dropout, eps) for _ in range(layers)
        )

    def forward(
        self,
        state: Tensor,
        edge_index: Tensor,
        edge_sign: Tensor,
        edge_weight: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        source, target = edge_index
        degree = state.new_zeros(state.shape[0])
        ones = state.new_ones(source.shape[0])
        degree.index_add_(0, source, ones)
        degree.index_add_(0, target, ones)
        for layer in self.layers:
            state = layer(state, edge_index, edge_sign, edge_weight)
        return state, degree > 0

