"""Backed real-data loader tests using tiny dense and sparse AnnData files."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from vcc_prior_model import GeneVocabularyArtifacts
from vcc_prior_model.real_data import BackedH5adDataset

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts/gene_vocabulary"


def write_toy(path: Path, *, as_sparse: bool = False, fractional: bool = False) -> None:
    values = np.asarray(
        [
            [1.0, 0.0, 9.0],
            [2.0, 3.0, 8.0],
            [4.0, 5.0, 7.0],
            [6.0, 0.0, 6.0],
        ],
        dtype=np.float32,
    )
    if fractional:
        values[0, 0] = 0.5
    matrix = sparse.csr_matrix(values) if as_sparse else values
    obs = pd.DataFrame(
        {"target_gene": pd.Categorical(["NTC", "NTC", "ACLY", "ACLY"])},
        index=[f"cell_{index}" for index in range(len(values))],
    )
    var = pd.DataFrame(index=pd.Index(["TSPAN6", "DPM1", "not_in_vocab"]))
    ad.AnnData(X=matrix, obs=obs, var=var).write_h5ad(path)


@pytest.mark.parametrize("as_sparse", [False, True])
def test_backed_loader_aligns_and_reads_duplicate_rows(
    tmp_path: Path, as_sparse: bool
) -> None:
    path = tmp_path / f"toy_{as_sparse}.h5ad"
    write_toy(path, as_sparse=as_sparse)
    vocabulary = GeneVocabularyArtifacts(ARTIFACTS)
    with BackedH5adDataset(path, vocabulary) as dataset:
        assert dataset.num_cells == 4
        assert dataset.control_label == "NTC"
        assert dataset.perturbation_targets == ("ACLY",)
        assert dataset.alignment.input_gene_index.tolist() == [0, 2]
        assert dataset.alignment.output_position.tolist() == [0, 2]
        rows = dataset.read_rows(
            np.asarray([1, 1, 0], dtype=np.int64),
            dataset.alignment.input_local_index,
        )
        np.testing.assert_array_equal(rows, [[2, 3], [2, 3], [1, 0]])
        chunks = list(
            dataset.input_chunks(np.asarray([0, 1, 2], dtype=np.int64), chunk_size=2)
        )
        assert [tuple(chunk.shape) for chunk in chunks] == [(2, 2), (1, 2)]
        dataset.validate_raw_counts()


def test_raw_count_validation_rejects_fractional_values(tmp_path: Path) -> None:
    path = tmp_path / "fractional.h5ad"
    write_toy(path, fractional=True)
    vocabulary = GeneVocabularyArtifacts(ARTIFACTS)
    with (
        BackedH5adDataset(path, vocabulary) as dataset,
        pytest.raises(ValueError, match="not raw integer counts"),
    ):
        dataset.validate_raw_counts()


def test_multicontext_loader_recognizes_nt_controls(tmp_path: Path) -> None:
    path = tmp_path / "multicontext.h5ad"
    obs = pd.DataFrame(
        {
            "gene": pd.Categorical(["NT", "ACLY", "NT", "ACLY"]),
            "cell_type": pd.Categorical(["a549", "a549", "k562", "k562"]),
            "treatment": pd.Categorical(["IFNB", "IFNB", "IFNG", "IFNG"]),
        },
        index=[f"cell_{index}" for index in range(4)],
    )
    var = pd.DataFrame(index=pd.Index(["TSPAN6", "DPM1"]))
    ad.AnnData(X=np.ones((4, 2), dtype=np.float32), obs=obs, var=var).write_h5ad(path)

    vocabulary = GeneVocabularyArtifacts(ARTIFACTS)
    with BackedH5adDataset(path, vocabulary) as dataset:
        assert dataset.control_label == "NT"
        assert dataset.context_control_indices["a549|IFNB"].tolist() == [0]
        assert dataset.context_control_indices["k562|IFNG"].tolist() == [2]
        assert dataset.episode_groups[("a549|IFNB", "ACLY")].tolist() == [1]
        assert dataset.episode_groups[("k562|IFNG", "ACLY")].tolist() == [3]
