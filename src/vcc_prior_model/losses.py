"""Loss interfaces for the architecture; no optimisation loop is included."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import ControlOutput, ModelOutput


def _masked_mean(value: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return value.mean()
    if mask.ndim == 2 and value.ndim == 3 and mask.shape[0] == value.shape[0]:
        mask = mask.unsqueeze(1)
    mask = torch.broadcast_to(mask.to(value.dtype), value.shape)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def negative_binomial_nll(
    counts: Tensor,
    mean: Tensor,
    dispersion: Tensor,
    gene_mask: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Mean negative log likelihood under an NB(mean, dispersion) model."""

    theta = dispersion
    while theta.ndim < counts.ndim:
        theta = theta.unsqueeze(0)
    log_likelihood = (
        torch.lgamma(counts + theta)
        - torch.lgamma(theta)
        - torch.lgamma(counts + 1.0)
        + theta * (torch.log(theta + eps) - torch.log(theta + mean + eps))
        + counts * (torch.log(mean + eps) - torch.log(theta + mean + eps))
    )
    return _masked_mean(-log_likelihood, gene_mask)


def control_reconstruction_loss(
    output: ControlOutput,
    observed_counts: Tensor,
    observed_gene_mask: Tensor | None = None,
) -> Tensor:
    """NB reconstruction objective for NTC pretraining."""

    return negative_binomial_nll(
        observed_counts, output.mean, output.dispersion, observed_gene_mask
    )


def _log_normalize(counts: Tensor, eps: float = 1e-8) -> Tensor:
    library = counts.sum(dim=-1, keepdim=True).clamp_min(eps)
    return torch.log1p(counts * (10_000.0 / library))


def sliced_wasserstein_distance(
    predicted: Tensor,
    observed: Tensor,
    num_projections: int = 32,
    projection_directions: Tensor | None = None,
    gene_mask: Tensor | None = None,
) -> Tensor:
    """Distribution loss for two unpaired cell sets with equal sampled size.

    Inputs have shape ``[B, cells, genes]``. Training code should sample the
    same number of predicted and observed cells per episode. Supplying fixed
    projection directions makes validation deterministic.
    """

    if predicted.shape != observed.shape:
        raise ValueError(
            "sliced_wasserstein_distance requires equally sized [B, cells, genes] sets"
        )
    if gene_mask is not None:
        if gene_mask.ndim == 2:
            gene_mask = gene_mask.unsqueeze(1)
        predicted = predicted * gene_mask.to(predicted.dtype)
        observed = observed * gene_mask.to(observed.dtype)
    predicted = _log_normalize(predicted)
    observed = _log_normalize(observed)
    if projection_directions is None:
        projection_directions = torch.randn(
            predicted.shape[-1],
            num_projections,
            device=predicted.device,
            dtype=predicted.dtype,
        )
        projection_directions = F.normalize(projection_directions, dim=0)
    predicted_projection = predicted @ projection_directions
    observed_projection = observed @ projection_directions
    predicted_projection = predicted_projection.sort(dim=1).values
    observed_projection = observed_projection.sort(dim=1).values
    return (predicted_projection - observed_projection).square().mean()


@dataclass(frozen=True)
class LossConfig:
    # Pairwise NB is disabled by default because Perturb-seq cell sets are
    # unpaired. It may be enabled after an explicit OT matching step.
    expression_weight: float = 0.0
    distribution_weight: float = 1.0
    delta_weight: float = 0.5
    direction_weight: float = 0.1
    de_weight: float = 0.2
    unsupervised_effect_weight: float = 0.02
    distribution_projections: int = 32


class CompositePerturbationLoss(nn.Module):
    """NB expression plus pseudobulk effect, direction, and optional DE losses."""

    def __init__(self, config: LossConfig | None = None) -> None:
        super().__init__()
        self.config = config or LossConfig()

    def forward(
        self,
        output: ModelOutput,
        observed_perturbed_counts: Tensor,
        control_counts: Tensor,
        observed_gene_mask: Tensor | None = None,
        de_gene_mask: Tensor | None = None,
        effect_supervision_mask: Tensor | None = None,
        projection_directions: Tensor | None = None,
    ) -> dict[str, Tensor]:
        expression = negative_binomial_nll(
            observed_perturbed_counts,
            output.mean,
            output.dispersion,
            observed_gene_mask,
        )
        distribution = sliced_wasserstein_distance(
            output.mean,
            observed_perturbed_counts,
            self.config.distribution_projections,
            projection_directions,
            observed_gene_mask,
        )
        predicted_delta = torch.log1p(output.mean.mean(dim=1)) - torch.log1p(
            output.control_mean.mean(dim=1)
        )
        observed_delta = torch.log1p(observed_perturbed_counts.mean(dim=1)) - torch.log1p(
            control_counts.mean(dim=1)
        )
        delta_elementwise = F.smooth_l1_loss(
            predicted_delta, observed_delta, reduction="none"
        )
        delta = _masked_mean(delta_elementwise, observed_gene_mask)

        predicted_norm = predicted_delta.norm(dim=-1)
        observed_norm = observed_delta.norm(dim=-1)
        cosine = F.cosine_similarity(predicted_delta, observed_delta, dim=-1, eps=1e-6)
        # The residual head starts at exactly zero. Activating cosine loss at
        # that singular point produces an enormous 1 / eps gradient; delta and
        # distribution losses first move it away from zero, then direction joins.
        active = (observed_norm > 1e-6) & (predicted_norm > 1e-6)
        direction = (
            (1.0 - cosine[active]).mean()
            if active.any()
            else predicted_delta.new_zeros(())
        )
        if de_gene_mask is None:
            de = predicted_delta.new_zeros(())
        else:
            de = _masked_mean(delta_elementwise, de_gene_mask)

        if effect_supervision_mask is None:
            unsupervised_effect = predicted_delta.new_zeros(())
        else:
            unsupervised_mask = ~effect_supervision_mask.to(torch.bool)
            unsupervised_effect = _masked_mean(
                output.log_effect.square(), unsupervised_mask
            )

        total = (
            self.config.expression_weight * expression
            + self.config.distribution_weight * distribution
            + self.config.delta_weight * delta
            + self.config.direction_weight * direction
            + self.config.de_weight * de
            + self.config.unsupervised_effect_weight * unsupervised_effect
        )
        return {
            "loss": total,
            "expression": expression,
            "distribution": distribution,
            "delta": delta,
            "direction": direction,
            "de": de,
            "unsupervised_effect": unsupervised_effect,
        }
