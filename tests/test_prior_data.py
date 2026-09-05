"""Tests for real prior artifacts and their tensor loader."""

from __future__ import annotations

from pathlib import Path

import torch

from vcc_prior_model import GeneVocabularyArtifacts
from vcc_prior_model.prior_data import PriorArtifacts

ROOT = Path(__file__).resolve().parents[1]


def test_real_prior_artifacts_match_vocabulary_and_tensor_contract() -> None:
    vocabulary = GeneVocabularyArtifacts(ROOT / "artifacts/gene_vocabulary")
    artifacts = PriorArtifacts(ROOT / "artifacts/priors", vocabulary)
    priors = artifacts.load()
    assert artifacts.num_go_terms == 17_819
    assert artifacts.num_pathway_terms == 2_298
    assert priors.go is not None
    assert priors.reactome is not None
    assert priors.go.gene_index.numel() == 285_083
    assert priors.go.gene_index.max().item() < vocabulary.num_genes
    assert priors.go.term_index.max().item() < artifacts.num_go_terms
    assert priors.string_edge_index is not None
    assert tuple(priors.string_edge_index.shape) == (2, 358_194)
    assert priors.grn_edge_index is not None
    assert tuple(priors.grn_edge_index.shape) == (2, 42_213)
    assert priors.grn_edge_sign is not None
    assert set(torch.unique(priors.grn_edge_sign).tolist()) == {-1.0, 1.0}

