"""Prior-aware virtual-cell architecture.

The runtime package contains model code and a lightweight vocabulary-artifact
loader. It has no expression-matrix reader, training loop, or AnnData dependency.
"""

from .config import ModelConfig
from .losses import (
    CompositePerturbationLoss,
    LossConfig,
    control_reconstruction_loss,
    negative_binomial_nll,
    sliced_wasserstein_distance,
)
from .model import (
    ContextEncoding,
    ControlOutput,
    MembershipEdges,
    ModelOutput,
    PriorAwarePerturbationModel,
    PriorInputs,
)
from .tiers import (
    ArchitectureLevel,
    ControlBaselineModel,
    FunctionalPriorModel,
    LearnedCellStateModel,
    LearnedGlobalEffectModel,
    build_model,
)
from .vocabulary import DatasetGeneMapping, GeneVocabularyArtifacts

__all__ = [
    "CompositePerturbationLoss",
    "ContextEncoding",
    "ControlOutput",
    "ControlBaselineModel",
    "DatasetGeneMapping",
    "ArchitectureLevel",
    "FunctionalPriorModel",
    "GeneVocabularyArtifacts",
    "LearnedCellStateModel",
    "LearnedGlobalEffectModel",
    "LossConfig",
    "MembershipEdges",
    "ModelConfig",
    "ModelOutput",
    "PriorAwarePerturbationModel",
    "PriorInputs",
    "build_model",
    "control_reconstruction_loss",
    "negative_binomial_nll",
    "sliced_wasserstein_distance",
]
