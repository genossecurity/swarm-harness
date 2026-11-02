"""Append-only JSONL store for feedback events.

Append-only on purpose: the raw signal is the one thing you cannot
regenerate. Training sets are derived and disposable; this log is not.
One file per day keeps it greppable and makes nightly builds trivial.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Iterator

from .events import FeedbackEvent


class EventStore:
    def __init__(self, root: str = "data/raw"):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path_for(self, ts: float) -> str:
        day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(self.root, f"events-{day}.jsonl")

    def append(self, ev: FeedbackEvent) -> None:
        path = self._path_for(ev.ts)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev.to_json(), ensure_ascii=False) + "\n")

    def iter_all(self) -> Iterator[FeedbackEvent]:
        if not os.path.isdir(self.root):
            return
        for name in sorted(os.listdir(self.root)):
            if not name.endswith(".jsonl"):
                continue
            with open(os.path.join(self.root, name), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield FeedbackEvent.from_json(json.loads(line))

    def iter_since(self, day: str) -> Iterator[FeedbackEvent]:
        """day = 'YYYY-MM-DD' inclusive lower bound on the file date."""
        for name in sorted(os.listdir(self.root)):
            if not name.endswith(".jsonl"):
                continue
            file_day = name[len("events-"):-len(".jsonl")]
            if file_day >= day:
                with open(os.path.join(self.root, name), encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield FeedbackEvent.from_json(json.loads(line))
