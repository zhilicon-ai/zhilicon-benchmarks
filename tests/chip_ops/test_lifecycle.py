"""Tests for chip_ops.lifecycle."""
from __future__ import annotations

import pytest

from chip_ops.lifecycle import (
    STAGES,
    ChipLifecycle,
    IllegalTransition,
    TransitionRecord,
)


def test_default_stage_is_design() -> None:
    lc = ChipLifecycle("zh-x")
    assert lc.stage == "design"
    assert lc.history == []


def test_initial_stage_override() -> None:
    lc = ChipLifecycle("zh-x", initial_stage="silicon")
    assert lc.stage == "silicon"


def test_invalid_initial_stage_raises() -> None:
    with pytest.raises(ValueError):
        ChipLifecycle("zh-x", initial_stage="bogus")


def test_empty_chip_id_raises() -> None:
    with pytest.raises(ValueError):
        ChipLifecycle("")


def test_advance_records_history() -> None:
    lc = ChipLifecycle("zh-x")
    rec = lc.advance("verify", actor="ramesh", note="rtl ready")
    assert lc.stage == "verify"
    assert isinstance(rec, TransitionRecord)
    assert lc.history == [rec]
    assert rec.actor == "ramesh"


def test_full_walk_to_production() -> None:
    lc = ChipLifecycle("zh-x")
    walk = ["verify", "sim", "silicon", "pilot", "production"]
    for s in walk:
        lc.advance(s, actor="r")
    assert lc.stage == "production"
    assert [h.to_stage for h in lc.history] == walk
    assert lc.is_terminal
    assert lc.progress_pct() == pytest.approx(100.0)


def test_illegal_transition_raises() -> None:
    lc = ChipLifecycle("zh-x")
    with pytest.raises(IllegalTransition):
        lc.advance("production", actor="r")


def test_self_transition_raises() -> None:
    lc = ChipLifecycle("zh-x")
    with pytest.raises(IllegalTransition):
        lc.advance("design", actor="r")


def test_unknown_target_raises() -> None:
    lc = ChipLifecycle("zh-x")
    with pytest.raises(IllegalTransition):
        lc.advance("nope", actor="r")


def test_actor_required() -> None:
    lc = ChipLifecycle("zh-x")
    with pytest.raises(ValueError):
        lc.advance("verify", actor="")


def test_can_transition_to() -> None:
    lc = ChipLifecycle("zh-x")
    assert lc.can_transition_to("verify")
    assert not lc.can_transition_to("production")
    assert not lc.can_transition_to("nope")


def test_recovery_path() -> None:
    """verify -> design is allowed for bug-fix regressions."""
    lc = ChipLifecycle("zh-x")
    lc.advance("verify", actor="r")
    lc.advance("design", actor="r", note="critical bug")
    assert lc.stage == "design"
    assert len(lc.history) == 2


def test_progress_index_and_pct() -> None:
    lc = ChipLifecycle("zh-x", initial_stage="sim")
    assert lc.progress_index() == STAGES.index("sim")
    assert lc.progress_pct() == pytest.approx(40.0)


def test_history_is_a_copy() -> None:
    lc = ChipLifecycle("zh-x")
    lc.advance("verify", actor="r")
    h = lc.history
    h.clear()
    assert lc.history  # internal record still intact


def test_all_stages_listed() -> None:
    assert STAGES == [
        "design",
        "verify",
        "sim",
        "silicon",
        "pilot",
        "production",
    ]
