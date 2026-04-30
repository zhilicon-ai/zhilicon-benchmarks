"""Chip lifecycle FSM.

Defines the canonical six-stage lifecycle and the legal transitions
between stages. Includes a transition log so program management can audit
how a part progressed.

Stages
------
1. ``design``      — RTL/architecture authoring.
2. ``verify``      — UVM/formal/coverage gates.
3. ``sim``         — gate-level + post-PR simulation.
4. ``silicon``     — first-silicon bring-up at the lab.
5. ``pilot``       — limited customer engagements.
6. ``production``  — high-volume manufacturing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Set


# Canonical ordered stages
STAGES: List[str] = [
    "design",
    "verify",
    "sim",
    "silicon",
    "pilot",
    "production",
]

# Allowed forward + recovery transitions.
_FORWARD: Dict[str, Set[str]] = {
    "design": {"verify"},
    "verify": {"sim", "design"},          # may regress on bugs
    "sim": {"silicon", "verify"},          # may regress on bugs
    "silicon": {"pilot", "sim"},           # may regress for ECO
    "pilot": {"production", "silicon"},
    "production": set(),                   # terminal
}


@dataclass
class TransitionRecord:
    """A single stage transition (immutable record)."""

    from_stage: str
    to_stage: str
    actor: str
    note: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class IllegalTransition(ValueError):
    """Raised when an illegal lifecycle transition is requested."""


class ChipLifecycle:
    """Tracks a single chip's progress through the lifecycle.

    Examples
    --------
    >>> lc = ChipLifecycle("zh-leo-a0")
    >>> lc.advance("verify", actor="ramesh")
    >>> lc.stage
    'verify'
    """

    def __init__(self, chip_id: str, *, initial_stage: str = "design") -> None:
        if initial_stage not in STAGES:
            raise ValueError(f"unknown stage {initial_stage!r}")
        if not chip_id:
            raise ValueError("chip_id must not be empty")
        self.chip_id = chip_id
        self._stage = initial_stage
        self._history: List[TransitionRecord] = []

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def history(self) -> List[TransitionRecord]:
        return list(self._history)

    @property
    def is_terminal(self) -> bool:
        return not _FORWARD[self._stage]

    def can_transition_to(self, target: str) -> bool:
        if target not in STAGES:
            return False
        return target in _FORWARD[self._stage]

    def advance(self, target: str, *, actor: str, note: str = "") -> TransitionRecord:
        """Transition to ``target`` if legal; else raise."""
        if not actor:
            raise ValueError("actor must not be empty")
        if target not in STAGES:
            raise IllegalTransition(f"unknown target stage {target!r}")
        if target == self._stage:
            raise IllegalTransition(f"already in stage {target!r}")
        if target not in _FORWARD[self._stage]:
            raise IllegalTransition(
                f"cannot move from {self._stage!r} to {target!r}; "
                f"allowed: {sorted(_FORWARD[self._stage])}"
            )
        rec = TransitionRecord(
            from_stage=self._stage, to_stage=target, actor=actor, note=note
        )
        self._stage = target
        self._history.append(rec)
        return rec

    def progress_index(self) -> int:
        """Return the 0-based index of the current stage in ``STAGES``."""
        return STAGES.index(self._stage)

    def progress_pct(self) -> float:
        """Return progress as a percentage (design=0, production=100)."""
        return 100.0 * self.progress_index() / (len(STAGES) - 1)
