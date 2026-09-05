"""Tests for the generated real vocabulary and its runtime loader."""

from __future__ import annotations

from pathlib import Path

import torch

from vcc_prior_model import GeneVocabularyArtifacts, ModelConfig, build_model

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts/gene_vocabulary"


def test_real_vocabulary_counts_and_vcc_order() -> None:
    vocab = GeneVocabularyArtifacts(ARTIFACTS)
    assert vocab.num_genes == 19_197
    assert vocab.num_output_genes == 18_533
    assert torch.equal(vocab.output_gene_index, torch.arange(18_533))
    assert vocab.output_gene_symbols[:3] == ("TSPAN6", "TNMD", "DPM1")


def test_real_dataset_mappings() -> None:
    vocab = GeneVocabularyArtifacts(ARTIFACTS)
    vcc = vocab.load_dataset("context_A")
    replogle = vocab.load_dataset("replogle_k562_gwps")
    assert vcc.gene_index.numel() == 18_533
    assert vcc.output_observed_mask.all()
    assert replogle.gene_index.numel() == 8_248
    assert int(replogle.output_observed_mask.sum()) == 7_679
    assert int(replogle.global_observed_mask.sum()) == 8_248
    assert int((replogle.local_to_output_index < 0).sum()) == 569
    assert int((replogle.output_to_local_index < 0).sum()) == 10_854


def test_target_symbol_mapping_uses_global_indices() -> None:
    vocab = GeneVocabularyArtifacts(ARTIFACTS)
    target = vocab.target_index(["ACLY", "ABCD1"])
    assert target.tolist() == [6_219, 2_524]


def test_all_replogle_targets_are_encodable() -> None:
    vocab = GeneVocabularyArtifacts(ARTIFACTS)
    target_only = vocab.target_index(["NOMO3"])
    assert target_only.item() >= 18_533


def test_real_mapping_plugs_into_model_contract() -> None:
    vocab = GeneVocabularyArtifacts(ARTIFACTS)
    replogle = vocab.load_dataset("replogle_k562_gwps")
    model = build_model(
        "cell_state",
        ModelConfig(
            num_genes=vocab.num_genes,
            num_output_genes=vocab.num_output_genes,
            gene_dim=16,
            cell_dim=16,
            context_dim=16,
            perturbation_dim=16,
            decoder_dim=8,
            context_prototypes=2,
            dropout=0.0,
        ),
        vocab.output_gene_index,
    ).eval()
    context = torch.ones((1, 3, replogle.gene_index.numel()))
    query = context[:, :2]
    with torch.inference_mode():
        output = model(
            context,
            query,
            vocab.target_index(["ACLY"]),
            context_gene_index=replogle.gene_index,
            query_gene_index=replogle.gene_index,
        )
    assert output.mean.shape == (1, 2, 18_533)
    assert torch.equal(output.output_gene_index, vocab.output_gene_index)


if __name__ == "__main__":
    checks = sorted(name for name in globals() if name.startswith("test_"))
    for check in checks:
        print(f"RUN {check}")
        globals()[check]()
    print(f"passed {len(checks)} real vocabulary checks")
