"""Instantiate PriorCell and run one tiny synthetic forward pass."""

from __future__ import annotations

import torch

from vcc_prior_model import MembershipEdges, ModelConfig, PriorAwarePerturbationModel, PriorInputs


def main() -> None:
    config = ModelConfig(
        num_genes=101,
        num_go_terms=12,
        num_pathway_terms=8,
        gene_dim=64,
        cell_dim=64,
        context_dim=64,
        perturbation_dim=64,
        decoder_dim=32,
        context_prototypes=4,
    )
    model = PriorAwarePerturbationModel(config).eval()
    priors = PriorInputs(
        go=MembershipEdges(
            gene_index=torch.arange(24),
            term_index=torch.arange(24) % config.num_go_terms,
        ),
        string_edge_index=torch.stack((torch.arange(50), torch.arange(1, 51))),
        string_edge_weight=torch.linspace(0.4, 1.0, 50),
        grn_edge_index=torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]]),
        grn_edge_sign=torch.tensor([1.0, -1.0, 1.0, -1.0]),
    )
    context_counts = torch.poisson(torch.full((1, 32, config.num_genes), 1.2))
    query_counts = context_counts[:, :8]
    with torch.inference_mode():
        output = model(context_counts, query_counts, torch.tensor([3]), priors)
        samples = model.sample_counts(output)
    print("NB mean:", tuple(output.mean.shape), output.mean.dtype)
    print("raw-count sample:", tuple(samples.shape), samples.dtype)
    print("context:", tuple(output.context_embedding.shape))
    print("perturbation:", tuple(output.perturbation_embedding.shape))


if __name__ == "__main__":
    main()

