"""Tests for chip_ops.mailbox."""
from __future__ import annotations

from pathlib import Path

import pytest

from chip_ops.mailbox import Mailbox, Message


def test_message_validation() -> None:
    with pytest.raises(ValueError):
        Message(sender="r", topic="t", body="hi", priority="extreme")
    with pytest.raises(ValueError):
        Message(sender="", topic="t", body="hi")
    with pytest.raises(ValueError):
        Message(sender="r", topic="", body="hi")


def test_send_and_messages() -> None:
    mb = Mailbox()
    mb.send(Message(sender="r", topic="design", body="hello"))
    msgs = mb.messages("design")
    assert len(msgs) == 1
    assert msgs[0].body == "hello"


def test_messages_empty_topic() -> None:
    mb = Mailbox()
    assert mb.messages("nope") == []


def test_subscribe_and_unsubscribe() -> None:
    mb = Mailbox()
    mb.subscribe("design", "alice")
    mb.subscribe("design", "bob")
    mb.subscribe("design", "alice")  # idempotent
    assert mb.subscribers("design") == ["alice", "bob"]
    assert mb.unsubscribe("design", "alice")
    assert mb.subscribers("design") == ["bob"]
    assert not mb.unsubscribe("design", "alice")


def test_unread_and_mark_read() -> None:
    mb = Mailbox()
    msg = mb.send(Message(sender="r", topic="t", body="hi"))
    assert mb.unread("t", "alice") == [msg]
    ok = mb.mark_read("t", msg.msg_id, "alice")
    assert ok
    assert mb.unread("t", "alice") == []


def test_mark_read_unknown_msg() -> None:
    mb = Mailbox()
    mb.send(Message(sender="r", topic="t", body="hi"))
    assert not mb.mark_read("t", "no-such-id", "alice")


def test_by_priority() -> None:
    mb = Mailbox()
    mb.send(Message(sender="r", topic="t", body="x", priority="high"))
    mb.send(Message(sender="r", topic="t", body="y", priority="low"))
    high = mb.by_priority("t", "high")
    assert len(high) == 1
    assert high[0].body == "x"


def test_total_messages_and_summary() -> None:
    mb = Mailbox()
    mb.send(Message(sender="r", topic="a", body="1"))
    mb.send(Message(sender="r", topic="a", body="2"))
    mb.send(Message(sender="r", topic="b", body="3"))
    assert mb.total_messages() == 3
    assert mb.topic_summary() == {"a": 2, "b": 1}


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    mb = Mailbox()
    mb.subscribe("design", "alice")
    mb.send(Message(sender="r", topic="design", body="hello"))
    mb.send(
        Message(sender="r", topic="fab", body="urgent", priority="urgent")
    )
    out = tmp_path / "mb.json"
    mb.save(out)
    rebuilt = Mailbox.load(out)
    assert rebuilt.subscribers("design") == ["alice"]
    assert rebuilt.messages("design")[0].body == "hello"
    assert rebuilt.messages("fab")[0].priority == "urgent"


def test_latest() -> None:
    mb = Mailbox()
    assert mb.latest("t") is None
    mb.send(Message(sender="r", topic="t", body="first"))
    last = mb.send(Message(sender="r", topic="t", body="second"))
    assert mb.latest("t") is last


def test_message_default_id_unique() -> None:
    a = Message(sender="r", topic="t", body="x")
    b = Message(sender="r", topic="t", body="x")
    assert a.msg_id != b.msg_id
