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
    # Match the global perturbation-effect scale so the identity solution is not
    # rewarded merely because most genes are unchanged.
    magnitude_weight: float = 0.0
    # Explicitly supervise the perturbed gene whenever it is measured.
    target_weight: float = 0.0
    # Penalise wrong signs on the strongest observed DE genes.
    de_direction_weight: float = 0.0
    # Level 3.1 objectives. These remain disabled for earlier checkpoints.
    support_weight: float = 0.0
    ranking_weight: float = 0.0
    support_sign_weight: float = 0.0
    support_magnitude_weight: float = 0.0
    cardinality_weight: float = 0.0
    calibration_regularization_weight: float = 0.0
    population_regularization_weight: float = 0.0
    baseline_mix_regularization_weight: float = 0.0
    focal_gamma: float = 2.0
    ranking_margin: float = 0.5
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
        target_gene_index: Tensor | None = None,
        observed_delta_target: Tensor | None = None,
        de_confidence: Tensor | None = None,
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
        # Match the challenge-style comparison in library-normalised expression
        # space. Raw pseudobulk deltas otherwise learn dataset-specific library
        # size shifts instead of gene-specific perturbation responses.
        predicted_delta = _log_normalize(output.mean).mean(dim=1) - _log_normalize(
            output.control_mean
        ).mean(dim=1)
        sampled_observed_delta = _log_normalize(observed_perturbed_counts).mean(
            dim=1
        ) - _log_normalize(control_counts).mean(dim=1)
        observed_delta = (
            sampled_observed_delta
            if observed_delta_target is None
            else observed_delta_target.to(
                device=predicted_delta.device, dtype=predicted_delta.dtype
            )
        )
        delta_elementwise = F.smooth_l1_loss(predicted_delta, observed_delta, reduction="none")
        delta = _masked_mean(delta_elementwise, observed_gene_mask)

        predicted_norm = predicted_delta.norm(dim=-1)
        observed_norm = observed_delta.norm(dim=-1)
        cosine = F.cosine_similarity(predicted_delta, observed_delta, dim=-1, eps=1e-6)
        # The residual head starts at exactly zero. Activating cosine loss at
        # that singular point produces an enormous 1 / eps gradient; delta and
        # distribution losses first move it away from zero, then direction joins.
        active = (observed_norm > 1e-6) & (predicted_norm > 1e-6)
        direction = (1.0 - cosine[active]).mean() if active.any() else predicted_delta.new_zeros(())
        if de_gene_mask is None:
            de = predicted_delta.new_zeros(())
            de_direction = predicted_delta.new_zeros(())
        else:
            de = _masked_mean(delta_elementwise, de_gene_mask)
            # Require the predicted change to reach the observed sign with a
            # conservative, data-dependent margin capped at 0.1 log units.
            direction_margin = observed_delta.abs().clamp(max=0.1)
            signed_prediction = observed_delta.sign() * predicted_delta
            de_direction = _masked_mean(F.relu(direction_margin - signed_prediction), de_gene_mask)

        if observed_gene_mask is None:
            scale_mask = torch.ones_like(predicted_delta, dtype=torch.bool)
        else:
            scale_mask = torch.broadcast_to(
                observed_gene_mask.to(torch.bool), predicted_delta.shape
            )
        scale_count = scale_mask.sum(dim=-1).clamp_min(1)
        # The perturbation head starts at exact zero. A bare sqrt has an
        # infinite derivative there (0 * inf becomes NaN in backprop), so keep
        # the scale norm smooth around the identity initialisation.
        scale_eps = 1e-8
        predicted_rms = (
            (predicted_delta.square() * scale_mask).sum(dim=-1) / scale_count + scale_eps
        ).sqrt()
        observed_rms = (
            (observed_delta.square() * scale_mask).sum(dim=-1) / scale_count + scale_eps
        ).sqrt()
        magnitude = F.smooth_l1_loss(predicted_rms, observed_rms)

        target_gene_loss = predicted_delta.new_zeros(())
        if target_gene_index is not None:
            target_gene_index = target_gene_index.to(output.output_gene_index.device)
            matches = target_gene_index[:, None] == output.output_gene_index[None, :]
            present = matches.any(dim=-1)
            if present.any():
                positions = matches.to(torch.long).argmax(dim=-1)
                batch_index = torch.arange(predicted_delta.shape[0], device=predicted_delta.device)
                target_gene_loss = F.smooth_l1_loss(
                    predicted_delta[batch_index[present], positions[present]],
                    observed_delta[batch_index[present], positions[present]],
                )

        if effect_supervision_mask is None:
            unsupervised_effect = predicted_delta.new_zeros(())
        else:
            unsupervised_mask = ~effect_supervision_mask.to(torch.bool)
            unsupervised_effect = _masked_mean(output.log_effect.square(), unsupervised_mask)

        support = predicted_delta.new_zeros(())
        ranking = predicted_delta.new_zeros(())
        support_sign = predicted_delta.new_zeros(())
        support_magnitude = predicted_delta.new_zeros(())
        cardinality = predicted_delta.new_zeros(())
        if de_gene_mask is not None and output.effect_support_logits is not None:
            labels = torch.broadcast_to(
                de_gene_mask.to(torch.bool), output.effect_support_logits.shape
            )
            supervision = torch.ones_like(labels)
            if observed_gene_mask is not None:
                supervision &= torch.broadcast_to(observed_gene_mask.to(torch.bool), labels.shape)
            # In low-cell screens, zero discoveries indicate insufficient
            # statistical power rather than evidence that every gene is a
            # true negative. Exclude such rows from support/cardinality losses.
            informative_batch = labels.any(dim=-1)
            supervision &= informative_batch.unsqueeze(-1)
            probabilities = output.effect_support_logits.sigmoid()
            gamma = self.config.focal_gamma
            positive_loss = -(
                (1.0 - probabilities).pow(gamma) * F.logsigmoid(output.effect_support_logits)
            )
            negative_loss = -(
                probabilities.pow(gamma) * F.logsigmoid(-output.effect_support_logits)
            )
            if de_confidence is not None:
                confidence = torch.broadcast_to(
                    de_confidence.to(predicted_delta), labels.shape
                ).clamp(0.0, 1.0)
                positive_loss = positive_loss * confidence
            positive_mask = labels & supervision
            negative_mask = ~labels & supervision
            # Balance classes explicitly: otherwise ~18k negatives overwhelm
            # the few hundred genes relevant to Jaccard/Reach.
            support = 0.5 * (
                _masked_mean(positive_loss, positive_mask)
                + _masked_mean(negative_loss, negative_mask)
            )

            ranking_terms = []
            for batch_index in range(labels.shape[0]):
                positive = torch.nonzero(positive_mask[batch_index], as_tuple=False).flatten()
                negative = torch.nonzero(negative_mask[batch_index], as_tuple=False).flatten()
                if positive.numel() == 0 or negative.numel() == 0:
                    continue
                logits = output.effect_support_logits[batch_index]
                hard_count = min(int(negative.numel()), max(32, 2 * int(positive.numel())))
                hard_negative = negative.index_select(
                    0, logits.index_select(0, negative).topk(hard_count).indices
                )
                ranking_terms.append(
                    F.relu(
                        self.config.ranking_margin
                        - logits.index_select(0, positive)[:, None]
                        + logits.index_select(0, hard_negative)[None, :]
                    ).mean()
                )
            if ranking_terms:
                ranking = torch.stack(ranking_terms).mean()

            predicted_count = (probabilities * supervision).sum(dim=-1)
            observed_count = positive_mask.sum(dim=-1).to(predicted_count)
            if informative_batch.any():
                cardinality = F.smooth_l1_loss(
                    torch.log1p(predicted_count[informative_batch]),
                    torch.log1p(observed_count[informative_batch]),
                )

            if output.effect_sign_logits is not None:
                sign_target = observed_delta.sign().unsqueeze(1)
                sign_target = sign_target.expand_as(output.effect_sign_logits)
                sign_mask = positive_mask.unsqueeze(1).expand_as(sign_target)
                support_sign = _masked_mean(
                    F.softplus(-sign_target * output.effect_sign_logits), sign_mask
                )
            support_magnitude = _masked_mean(
                F.smooth_l1_loss(predicted_delta.abs(), observed_delta.abs(), reduction="none"),
                positive_mask,
            )

        calibration_regularization = predicted_delta.new_zeros(())
        if output.effect_calibration is not None:
            calibration_regularization = (output.effect_calibration - 1.0).square().mean()
        population_regularization = predicted_delta.new_zeros(())
        if output.population_residual is not None:
            population_regularization = output.population_residual.square().mean()
        baseline_mix_regularization = predicted_delta.new_zeros(())
        if output.baseline_mix_probability is not None:
            baseline_mix_regularization = output.baseline_mix_probability.mean()

        total = (
            self.config.expression_weight * expression
            + self.config.distribution_weight * distribution
            + self.config.delta_weight * delta
            + self.config.direction_weight * direction
            + self.config.de_weight * de
            + self.config.magnitude_weight * magnitude
            + self.config.target_weight * target_gene_loss
            + self.config.de_direction_weight * de_direction
            + self.config.support_weight * support
            + self.config.ranking_weight * ranking
            + self.config.support_sign_weight * support_sign
            + self.config.support_magnitude_weight * support_magnitude
            + self.config.cardinality_weight * cardinality
            + self.config.unsupervised_effect_weight * unsupervised_effect
            + self.config.calibration_regularization_weight * calibration_regularization
            + self.config.population_regularization_weight * population_regularization
            + self.config.baseline_mix_regularization_weight * baseline_mix_regularization
        )
        return {
            "loss": total,
            "expression": expression,
            "distribution": distribution,
            "delta": delta,
            "direction": direction,
            "de": de,
            "magnitude": magnitude,
            "target_gene_loss": target_gene_loss,
            "de_direction": de_direction,
            "support": support,
            "ranking": ranking,
            "support_sign": support_sign,
            "support_magnitude": support_magnitude,
            "cardinality": cardinality,
            "unsupervised_effect": unsupervised_effect,
            "calibration_regularization": calibration_regularization,
            "population_regularization": population_regularization,
            "baseline_mix_regularization": baseline_mix_regularization,
        }
