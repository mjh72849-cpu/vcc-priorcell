"""Small framework-independent helpers shared by real training entry points."""

from __future__ import annotations


def training_schedule(
    control_steps: int, perturbation_steps: int, control_every: int
) -> list[str]:
    """Build a schedule with exactly ``perturbation_steps`` perturbation updates.

    A control replay update is inserted between perturbation blocks, so replay
    never silently replaces one of the requested perturbation updates.
    """

    if control_steps < 0 or perturbation_steps < 0 or control_every < 0:
        raise ValueError("training step counts must be non-negative")
    phases = ["control"] * control_steps
    for update in range(1, perturbation_steps + 1):
        phases.append("perturbation")
        if control_every and update % control_every == 0 and update < perturbation_steps:
            phases.append("control")
    return phases

