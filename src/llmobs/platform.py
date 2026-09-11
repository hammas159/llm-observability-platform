"""The observatory: record calls, ask questions about them.

Windows are time-based rather than count-based. A count window compares last week's
thousand calls against today's thousand and calls a traffic change a quality change;
a time window does not.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .alerts.rules import Alert, AlertEngine
from .drift.psi import interpret, psi_categorical, psi_numeric
from .drift.signature import signature
from .metrics.aggregate import group_by, most_expensive, summarise
from .types import LLMCall


@dataclass
class Observatory:
    calls: list[LLMCall] = field(default_factory=list)
    engine: AlertEngine = field(default_factory=AlertEngine)
    # Append-only sink. Written per call, so a process that dies still leaves data.
    sink: Path | None = None

    def record(self, call: LLMCall, *, prompt: str | None = None) -> LLMCall:
        if prompt is not None:
            # Derived here and immediately discarded: the prompt is never stored.
            call.prompt_signature = signature(prompt)
            call.prompt_length = len(prompt.split())
        self.calls.append(call)
        if self.sink is not None:
            self.sink.parent.mkdir(parents=True, exist_ok=True)
            with self.sink.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(call.to_dict(), default=str) + "\n")
        return call

    def window(self, seconds: float, *, ending: float | None = None) -> list[LLMCall]:
        end = ending if ending is not None else time.time()
        return [c for c in self.calls if end - seconds <= c.at <= end]

    def summary(self, seconds: float = 3600.0) -> dict:
        return summarise(self.window(seconds))

    def by(self, key: str, seconds: float = 3600.0) -> dict[str, dict]:
        return group_by(self.window(seconds), key)

    def top_spend(self, key: str = "feature", seconds: float = 3600.0, limit: int = 5):
        return most_expensive(self.window(seconds), key, limit)

    def drift(self, *, window_seconds: float = 3600.0) -> dict:
        """Compare the most recent window against the one before it.

        Adjacent windows, not window-against-all-history: history includes the
        current window and dilutes exactly the change being looked for.
        """
        now = time.time()
        current = self.window(window_seconds, ending=now)
        reference = self.window(window_seconds, ending=now - window_seconds)

        if len(reference) < 30 or len(current) < 30:
            return {
                "status": "insufficient data",
                "reference": len(reference),
                "current": len(current),
            }

        scores = {
            "prompt_length": psi_numeric(
                [c.prompt_length for c in reference], [c.prompt_length for c in current]
            ),
            "prompt_signature": psi_categorical(
                [c.prompt_signature for c in reference], [c.prompt_signature for c in current]
            ),
            "model": psi_categorical([c.model for c in reference], [c.model for c in current]),
            "latency": psi_numeric(
                [c.latency_ms for c in reference], [c.latency_ms for c in current]
            ),
            "cost": psi_numeric([c.usd for c in reference], [c.usd for c in current]),
        }
        return {
            "status": "ok",
            "reference_calls": len(reference),
            "current_calls": len(current),
            "psi": scores,
            "verdict": {k: interpret(v) for k, v in scores.items()},
            "worst": max(scores, key=scores.get),
        }

    def check_alerts(
        self, *, window_seconds: float = 3600.0, per_feature: bool = True
    ) -> list[Alert]:
        """Evaluate rules globally and per feature.

        Per feature as well as globally, because one broken feature is invisible in
        an aggregate dominated by a healthy one.
        """
        now = time.time()
        current = self.window(window_seconds, ending=now)
        baseline = summarise(self.window(window_seconds, ending=now - window_seconds))
        baseline = baseline if baseline.get("calls", 0) >= 30 else None

        fired = self.engine.evaluate(summarise(current), baseline=baseline, scope="global")

        if per_feature:
            for feature, stats in group_by(current, "feature").items():
                fired.extend(self.engine.evaluate(stats, scope=f"feature:{feature}"))

        return fired

    @staticmethod
    def load(path: str | Path) -> Observatory:
        obs = Observatory()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                obs.calls.append(LLMCall(**json.loads(line)))
        return obs

    def ingest(self, calls: Iterable[LLMCall]) -> None:
        self.calls.extend(calls)
