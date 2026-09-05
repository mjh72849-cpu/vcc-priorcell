"""Synthetic tensor tests only: no dataset is opened by this test module."""

from __future__ import annotations

from dataclasses import replace

import torch

from vcc_prior_model import (
    ArchitectureLevel,
    CompositePerturbationLoss,
    MembershipEdges,
    ModelConfig,
    PriorAwarePerturbationModel,
    PriorInputs,
    build_model,
    control_reconstruction_loss,
)


def _small_model() -> tuple[PriorAwarePerturbationModel, PriorInputs]:
    torch.manual_seed(7)
    config = ModelConfig(
        num_genes=23,
        num_go_terms=6,
        num_pathway_terms=4,
        gene_dim=32,
        cell_dim=32,
        context_dim=32,
        perturbation_dim=32,
        decoder_dim=16,
        graph_layers=2,
        context_prototypes=4,
        dropout=0.0,
        prior_dropout=0.0,
    )
    model = PriorAwarePerturbationModel(config).eval()
    priors = PriorInputs(
        go=MembershipEdges(
            gene_index=torch.tensor([0, 1, 2, 3, 4, 8, 12, 20]),
            term_index=torch.tensor([0, 0, 1, 1, 2, 3, 4, 5]),
            weight=torch.tensor([1.0, 0.8, 1.2, 1.0, 0.7, 1.0, 0.9, 1.0]),
        ),
        reactome=MembershipEdges(
            gene_index=torch.tensor([0, 2, 5, 7, 9, 11]),
            term_index=torch.tensor([0, 0, 1, 2, 2, 3]),
        ),
        string_edge_index=torch.tensor([[0, 1, 3, 5], [1, 2, 4, 6]]),
        string_edge_weight=torch.tensor([0.9, 0.7, 0.8, 0.6]),
        grn_edge_index=torch.tensor([[0, 2, 4, 7], [3, 3, 8, 9]]),
        grn_edge_sign=torch.tensor([1.0, -1.0, 1.0, -1.0]),
    )
    return model, priors


def test_forward_shapes_and_nb_sampling() -> None:
    model, priors = _small_model()
    context = torch.poisson(torch.full((2, 11, 23), 1.5))
    query = context[:, :5].clone()
    target = torch.tensor([2, 7])

    output = model(context, query, target, priors, gene_chunk_size=7)

    assert output.mean.shape == (2, 5, 23)
    assert output.dispersion.shape == (23,)
    assert output.delta_latent.shape == (2, 5, 32)
    assert output.state_prototypes.shape == (2, 4, 32)
    assert torch.isfinite(output.mean).all()
    assert (output.mean >= 0).all()
    assert torch.equal(output.log_effect, torch.zeros_like(output.log_effect))
    assert torch.equal(output.mean, output.control_mean)
    sampled = model.sample_counts(output)
    assert sampled.shape == output.mean.shape
    assert sampled.dtype == torch.int64
    assert (sampled >= 0).all()


def test_context_is_permutation_invariant() -> None:
    model, priors = _small_model()
    counts = torch.poisson(torch.full((1, 13, 23), 1.2))
    genes = model.encode_genes(priors)
    original = model.encode_context(counts, genes)
    permuted = model.encode_context(counts[:, torch.randperm(counts.shape[1])], genes)
    assert torch.allclose(
        original.context_embedding, permuted.context_embedding, atol=1e-6
    )
    assert torch.allclose(original.state_prototypes, permuted.state_prototypes, atol=1e-6)


def test_chunked_context_matches_dense_context() -> None:
    model, priors = _small_model()
    counts = torch.poisson(torch.full((1, 13, 23), 1.2))
    genes = model.encode_genes(priors)
    dense = model.encode_context(counts, genes)
    chunked = model.encode_context_chunks(
        [counts[:, :4], counts[:, 4:9], counts[:, 9:]], genes
    )
    assert torch.allclose(dense.context_embedding, chunked.context_embedding, atol=1e-6)
    assert torch.allclose(dense.state_prototypes, chunked.state_prototypes, atol=1e-6)
    assert torch.allclose(dense.gene_context_stats, chunked.gene_context_stats, atol=1e-6)
    assert torch.equal(dense.gene_context_observed, chunked.gene_context_observed)


def test_missing_priors_and_gene_mask_are_safe() -> None:
    model, _ = _small_model()
    context = torch.poisson(torch.full((1, 7, 23), 1.0))
    query = context[:, :3]
    gene_mask = torch.ones((1, 23))
    gene_mask[:, -5:] = 0
    output = model(
        context,
        query,
        torch.tensor([0]),
        priors=None,
        context_observed_gene_mask=gene_mask,
        query_observed_gene_mask=gene_mask,
    )
    assert output.mean.shape[-1] == 23
    assert torch.isfinite(output.mean).all()


def test_loss_interface_is_finite() -> None:
    model, priors = _small_model()
    context = torch.poisson(torch.full((1, 9, 23), 1.5))
    query = context[:, :4]
    output = model(context, query, torch.tensor([3]), priors)
    observed = torch.poisson(output.mean.detach().clamp_min(0.1))
    supervised = torch.ones(23, dtype=torch.bool)
    supervised[-5:] = False
    losses = CompositePerturbationLoss()(
        output, observed, query, effect_supervision_mask=supervised
    )
    assert set(losses) == {
        "loss",
        "expression",
        "distribution",
        "delta",
        "direction",
        "de",
        "unsupervised_effect",
    }
    assert all(torch.isfinite(value) for value in losses.values())

    control = model.reconstruct_control(query, priors)
    assert control.mean.shape == query.shape
    assert torch.isfinite(control_reconstruction_loss(control, query))


def test_all_architecture_levels_share_forward_contract() -> None:
    _, priors = _small_model()
    config = ModelConfig(
        num_genes=23,
        num_go_terms=6,
        num_pathway_terms=4,
        gene_dim=32,
        cell_dim=32,
        context_dim=32,
        perturbation_dim=32,
        decoder_dim=16,
        context_prototypes=4,
        dropout=0.0,
        prior_dropout=0.0,
    )
    context = torch.poisson(torch.full((1, 8, 23), 1.0))
    query = context[:, :3]
    target = torch.tensor([2])
    for level in ArchitectureLevel:
        model = build_model(level, config).eval()
        output = model(context, query, target, priors)
        assert output.mean.shape == (1, 3, 23), level
        assert output.dispersion.shape == (23,), level
        assert torch.isfinite(output.mean).all(), level
    control = build_model(ArchitectureLevel.CONTROL, config)
    assert torch.equal(control(context, query, target).mean, query)


def test_tiers_use_only_their_declared_priors() -> None:
    _, priors = _small_model()
    config = ModelConfig(
        num_genes=23,
        num_go_terms=6,
        num_pathway_terms=4,
        gene_dim=32,
        cell_dim=32,
        context_dim=32,
        perturbation_dim=32,
        decoder_dim=16,
        context_prototypes=4,
        dropout=0.0,
        prior_dropout=0.0,
    )
    functional = build_model(ArchitectureLevel.FUNCTIONAL_PRIOR, config).eval()
    without_grn = PriorInputs(
        go=priors.go,
        reactome=priors.reactome,
        string_edge_index=priors.string_edge_index,
        string_edge_weight=priors.string_edge_weight,
    )
    assert torch.equal(functional.encode_genes(priors), functional.encode_genes(without_grn))

    full = build_model(ArchitectureLevel.FULL_PRIOR, config).eval()
    assert not torch.equal(full.encode_genes(priors), full.encode_genes(without_grn))

    learned = build_model(ArchitectureLevel.CELL_STATE, config).eval()
    assert torch.equal(learned.encode_genes(priors), learned.encode_genes(None))


def test_conservative_gate_initialization() -> None:
    model, _ = _small_model()
    expected_gate = torch.sigmoid(torch.tensor(model.config.prior_gate_init_bias))
    for gate in model.gene_encoder.gates.values():
        final = gate[-1]
        assert torch.count_nonzero(final.weight) == 0
        assert torch.allclose(final.bias, torch.full_like(final.bias, -2.0))
        probe = torch.randn(5, final.in_features)
        assert torch.allclose(torch.sigmoid(final(probe)), expected_gate.expand(5, 1))


def test_active_prior_count_normalization() -> None:
    model, _ = _small_model()
    prior_sum = torch.tensor([[4.0, 8.0], [3.0, 6.0], [0.0, 0.0]])
    active_count = torch.tensor([[4.0], [1.0], [0.0]])
    normalized = model.gene_encoder._normalize_prior_sum(prior_sum, active_count)
    expected = torch.tensor([[2.0, 4.0], [3.0, 6.0], [0.0, 0.0]])
    assert torch.equal(normalized, expected)


def test_source_level_prior_dropout() -> None:
    model, _ = _small_model()
    dropout_model = build_model(
        ArchitectureLevel.FULL_PRIOR,
        ModelConfig(
            num_genes=23,
            num_go_terms=6,
            num_pathway_terms=4,
            gene_dim=32,
            cell_dim=32,
            context_dim=32,
            perturbation_dim=32,
            decoder_dim=16,
            context_prototypes=4,
            dropout=0.0,
            prior_dropout=0.5,
        ),
    ).train()
    torch.manual_seed(17)
    value = torch.ones(4, 3)
    observed_sums = {
        float(dropout_model.gene_encoder._source_dropout(value).sum()) for _ in range(16)
    }
    # Inverted source dropout either removes the whole source or scales it by 1 / keep.
    assert observed_sums == {0.0, 24.0}
    model.eval()
    assert torch.equal(model.gene_encoder._source_dropout(value), value)


def test_global_vocabulary_dataset_panel_and_fixed_output_are_decoupled() -> None:
    config = ModelConfig(
        num_genes=29,
        num_output_genes=7,
        gene_dim=32,
        cell_dim=32,
        context_dim=32,
        perturbation_dim=32,
        decoder_dim=16,
        context_prototypes=4,
        dropout=0.0,
        prior_dropout=0.0,
    )
    output_index = torch.tensor([0, 2, 4, 6, 8, 10, 12])
    panel_index = torch.tensor([0, 2, 5, 23, 24])
    model = PriorAwarePerturbationModel(
        config, output_gene_index=output_index
    ).eval()
    context = torch.poisson(torch.full((1, 9, panel_index.numel()), 1.2))
    query = context[:, :3]
    output = model(
        context,
        query,
        torch.tensor([23]),
        context_gene_index=panel_index,
        query_gene_index=panel_index,
    )
    assert output.mean.shape == (1, 3, 7)
    assert torch.equal(output.output_gene_index, output_index)
    assert torch.equal(output.mean, output.control_mean)

    genes = model.encode_genes()
    encoded = model.encode_context(context, genes, input_gene_index=panel_index)
    assert encoded.gene_context_stats.shape == (1, 29, 3)
    assert encoded.gene_context_observed.shape == (1, 29)
    assert encoded.gene_context_observed[0, panel_index].all()
    missing = torch.ones(29, dtype=torch.bool)
    missing[panel_index] = False
    assert not encoded.gene_context_observed[0, missing].any()
    assert torch.equal(
        encoded.gene_context_stats[0, missing],
        torch.zeros_like(encoded.gene_context_stats[0, missing]),
    )


def test_control_reconstruction_can_decode_only_observed_intersection() -> None:
    config = ModelConfig(
        num_genes=29,
        num_output_genes=7,
        gene_dim=32,
        cell_dim=32,
        context_dim=32,
        perturbation_dim=32,
        decoder_dim=16,
        context_prototypes=4,
    )
    output_index = torch.tensor([0, 2, 4, 6, 8, 10, 12])
    panel_index = torch.tensor([0, 2, 5, 23, 24])
    model = PriorAwarePerturbationModel(config, output_gene_index=output_index).eval()
    counts = torch.poisson(torch.full((1, 4, panel_index.numel()), 1.0))
    # Only global genes 0 and 2 are both measured and part of the VCC output panel.
    reconstructed = model.reconstruct_control(
        counts,
        query_gene_index=panel_index,
        decode_gene_index=torch.tensor([0, 2]),
    )
    assert reconstructed.mean.shape == (1, 4, 2)
    assert torch.equal(reconstructed.output_gene_index, torch.tensor([0, 2]))


def test_all_tiers_accept_dataset_panels_and_fixed_output() -> None:
    config = ModelConfig(
        num_genes=29,
        num_output_genes=7,
        gene_dim=32,
        cell_dim=32,
        context_dim=32,
        perturbation_dim=32,
        decoder_dim=16,
        context_prototypes=4,
        dropout=0.0,
        prior_dropout=0.0,
    )
    output_index = torch.tensor([0, 2, 4, 6, 8, 10, 12])
    panel_index = torch.tensor([0, 2, 5, 23, 24])
    context = torch.poisson(torch.full((1, 8, panel_index.numel()), 1.0))
    query = context[:, :3]
    for level in ArchitectureLevel:
        model = build_model(level, config, output_index).eval()
        output = model(
            context,
            query,
            torch.tensor([23]),
            context_gene_index=panel_index,
            query_gene_index=panel_index,
        )
        assert output.mean.shape == (1, 3, 7), level
        assert torch.equal(output.output_gene_index, output_index), level


def test_independent_gene_rate_does_not_depend_on_decode_panel() -> None:
    model, priors = _small_model()
    query = torch.poisson(torch.full((1, 4, 23), 1.0))
    one_gene = model.reconstruct_control(
        query, priors, decode_gene_index=torch.tensor([3])
    )
    larger_panel = model.reconstruct_control(
        query, priors, decode_gene_index=torch.tensor([1, 3, 7, 11])
    )
    assert torch.allclose(one_gene.mean[..., 0], larger_panel.mean[..., 1])
    assert torch.allclose(one_gene.dispersion[0], larger_panel.dispersion[1])


def test_unsupervised_effect_shrinkage_uses_mask_complement() -> None:
    model, priors = _small_model()
    context = torch.poisson(torch.full((1, 6, 23), 1.0))
    query = context[:, :4]
    output = model(context, query, torch.tensor([2]), priors)
    synthetic_effect = torch.zeros_like(output.log_effect)
    synthetic_effect[..., :18] = 10.0
    synthetic_effect[..., 18:] = 2.0
    output = replace(output, log_effect=synthetic_effect)
    supervised = torch.ones(23, dtype=torch.bool)
    supervised[18:] = False
    losses = CompositePerturbationLoss()(
        output,
        output.mean.detach(),
        output.control_mean.detach(),
        effect_supervision_mask=supervised,
        projection_directions=torch.eye(23)[:, :8],
    )
    assert torch.allclose(losses["unsupervised_effect"], torch.tensor(4.0))


def test_zero_effect_initialization_does_not_block_transition_gradient() -> None:
    model, priors = _small_model()
    model.train()
    context = torch.poisson(torch.full((1, 6, 23), 1.0))
    query = context[:, :3]
    output = model(context, query, torch.tensor([2]), priors)
    assert torch.count_nonzero(output.log_effect) == 0
    output.mean.mean().backward()
    final_transition = model.transition.delta[-1]
    assert final_transition.weight.grad is not None
    assert torch.count_nonzero(final_transition.weight.grad) > 0


if __name__ == "__main__":
    checks = sorted(name for name in globals() if name.startswith("test_"))
    for check in checks:
        print(f"RUN {check}")
        globals()[check]()
    print(f"passed {len(checks)} synthetic architecture checks")
