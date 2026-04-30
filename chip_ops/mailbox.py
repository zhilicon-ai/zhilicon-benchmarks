"""In-process inter-team messaging stub.

A simple message broker that supports:
  * named topics ("design", "verify", "fab", ...)
  * subscriber lists per topic
  * priority and read-receipt tracking
  * snapshot persistence to JSON

The intent is to provide a deterministic mock so workflow tooling can be
unit-tested without depending on a real Slack/Email/Kafka backend.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """A single message in the mailbox."""

    sender: str
    topic: str
    body: str
    priority: str = "normal"  # one of: low | normal | high | urgent
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    sent_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    read_by: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.priority not in {"low", "normal", "high", "urgent"}:
            raise ValueError(f"unknown priority {self.priority!r}")
        if not self.sender or not self.topic:
            raise ValueError("sender and topic must not be empty")


class Mailbox:
    """Simple multi-topic mailbox."""

    def __init__(self) -> None:
        self._topics: Dict[str, List[Message]] = {}
        self._subscribers: Dict[str, List[str]] = {}

    # ---- subscriptions -----------------------------------------------------
    def subscribe(self, topic: str, subscriber: str) -> None:
        subs = self._subscribers.setdefault(topic, [])
        if subscriber not in subs:
            subs.append(subscriber)

    def unsubscribe(self, topic: str, subscriber: str) -> bool:
        subs = self._subscribers.get(topic, [])
        if subscriber in subs:
            subs.remove(subscriber)
            return True
        return False

    def subscribers(self, topic: str) -> List[str]:
        return list(self._subscribers.get(topic, []))

    # ---- send/receive ------------------------------------------------------
    def send(self, msg: Message) -> Message:
        """Append a message to its topic."""
        self._topics.setdefault(msg.topic, []).append(msg)
        return msg

    def messages(self, topic: str) -> List[Message]:
        return list(self._topics.get(topic, []))

    def unread(self, topic: str, subscriber: str) -> List[Message]:
        return [
            m for m in self._topics.get(topic, []) if subscriber not in m.read_by
        ]

    def mark_read(self, topic: str, msg_id: str, subscriber: str) -> bool:
        for m in self._topics.get(topic, []):
            if m.msg_id == msg_id:
                if subscriber not in m.read_by:
                    m.read_by.append(subscriber)
                return True
        return False

    def by_priority(self, topic: str, priority: str) -> List[Message]:
        return [m for m in self._topics.get(topic, []) if m.priority == priority]

    # ---- introspection -----------------------------------------------------
    def total_messages(self) -> int:
        return sum(len(msgs) for msgs in self._topics.values())

    def topic_summary(self) -> Dict[str, int]:
        return {t: len(ms) for t, ms in self._topics.items()}

    # ---- persistence -------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "subscribers": {t: list(s) for t, s in self._subscribers.items()},
            "topics": {
                t: [
                    {
                        "sender": m.sender,
                        "topic": m.topic,
                        "body": m.body,
                        "priority": m.priority,
                        "msg_id": m.msg_id,
                        "sent_at": m.sent_at,
                        "read_by": list(m.read_by),
                    }
                    for m in msgs
                ]
                for t, msgs in self._topics.items()
            },
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Mailbox":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        mb = cls()
        for topic, subs in data.get("subscribers", {}).items():
            mb._subscribers[topic] = list(subs)
        for topic, msgs in data.get("topics", {}).items():
            mb._topics[topic] = [Message(**m) for m in msgs]
        return mb

    def latest(self, topic: str) -> Optional[Message]:
        msgs = self._topics.get(topic, [])
        return msgs[-1] if msgs else None
