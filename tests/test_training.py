"""Training schedule invariants."""

from __future__ import annotations

import pytest

from vcc_prior_model.training import training_schedule


def test_schedule_preserves_requested_update_counts() -> None:
    phases = training_schedule(control_steps=3, perturbation_steps=10, control_every=4)
    assert phases[:3] == ["control"] * 3
    assert phases.count("perturbation") == 10
    assert phases.count("control") == 5
    assert phases[3:] == [
        "perturbation",
        "perturbation",
        "perturbation",
        "perturbation",
        "control",
        "perturbation",
        "perturbation",
        "perturbation",
        "perturbation",
        "control",
        "perturbation",
        "perturbation",
    ]


def test_zero_control_every_disables_replay() -> None:
    assert training_schedule(1, 2, 0) == ["control", "perturbation", "perturbation"]


def test_negative_step_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        training_schedule(-1, 2, 1)

