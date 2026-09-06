"""Prior-aware virtual-cell architecture.

The runtime package contains model code and a lightweight vocabulary-artifact
loader. It has no expression-matrix reader, training loop, or AnnData dependency.
"""

from .config import ModelConfig
from .de_cache import DifferentialExpressionCache, DifferentialExpressionTarget
from .level4 import (
    AdaptiveEffectCalibrator,
    ContextAdaptivePriorModel,
    ContextConditionedTargetPrior,
    DenoisedEmpiricalBaseline,
    LowRankPopulationResidual,
    PretrainedCellStateAdapter,
)
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
    apply_library_preserving_effect,
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
    "ContextAdaptivePriorModel",
    "ContextConditionedTargetPrior",
    "ControlOutput",
    "ControlBaselineModel",
    "DatasetGeneMapping",
    "DifferentialExpressionCache",
    "DifferentialExpressionTarget",
    "ArchitectureLevel",
    "FunctionalPriorModel",
    "GeneVocabularyArtifacts",
    "DenoisedEmpiricalBaseline",
    "LearnedCellStateModel",
    "LearnedGlobalEffectModel",
    "LossConfig",
    "MembershipEdges",
    "ModelConfig",
    "ModelOutput",
    "PriorAwarePerturbationModel",
    "PriorInputs",
    "PretrainedCellStateAdapter",
    "AdaptiveEffectCalibrator",
    "LowRankPopulationResidual",
    "apply_library_preserving_effect",
    "build_model",
    "control_reconstruction_loss",
    "negative_binomial_nll",
    "sliced_wasserstein_distance",
]
