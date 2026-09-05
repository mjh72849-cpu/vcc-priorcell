"""Load preprocessed biological-prior arrays into the model tensor contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .model import MembershipEdges, PriorInputs
from .vocabulary import GeneVocabularyArtifacts


class PriorArtifacts:
    """Validated loader for artifacts written by ``prepare_priors.py``."""

    def __init__(
        self,
        directory: str | Path,
        vocabulary: GeneVocabularyArtifacts,
    ) -> None:
        self.directory = Path(directory)
        with (self.directory / "manifest.json").open(encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        expected = vocabulary.manifest["global_vocabulary"]["symbols_sha256"]
        if self.manifest["global_vocabulary_sha256"] != expected:
            raise ValueError("prior artifacts were built for a different gene vocabulary")
        if self.manifest["global_gene_count"] != vocabulary.num_genes:
            raise ValueError("prior artifact gene count does not match the vocabulary")

    @property
    def num_go_terms(self) -> int:
        return int(self.manifest["go"]["terms"])

    @property
    def num_pathway_terms(self) -> int:
        return int(self.manifest["reactome"]["pathways"])

    def _tensor(self, filename: str, dtype: torch.dtype) -> torch.Tensor:
        values = np.load(self.directory / filename, allow_pickle=False)
        return torch.from_numpy(values).to(dtype=dtype)

    def load(self, device: torch.device | str = "cpu") -> PriorInputs:
        priors = PriorInputs(
            go=MembershipEdges(
                self._tensor("go_gene_index.npy", torch.long),
                self._tensor("go_term_index.npy", torch.long),
            ),
            reactome=MembershipEdges(
                self._tensor("reactome_gene_index.npy", torch.long),
                self._tensor("reactome_term_index.npy", torch.long),
            ),
            string_edge_index=self._tensor("string_edge_index.npy", torch.long),
            string_edge_weight=self._tensor("string_edge_weight.npy", torch.float32),
            grn_edge_index=self._tensor("grn_edge_index.npy", torch.long),
            grn_edge_sign=self._tensor("grn_edge_sign.npy", torch.float32),
            grn_edge_weight=self._tensor("grn_edge_weight.npy", torch.float32),
        )
        return priors.to(device)

