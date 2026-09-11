"""Alert rules.

Every rule carries a **minimum sample count** and a **cooldown**. Both exist for the
same reason: an alerting system people mute is worse than no alerting system, and both
noise sources - firing on three requests, and firing every thirty seconds about the
same thing - are entirely preventable.

Rules are relative to a baseline where one exists. "Error rate above 5%" is the wrong
alarm when the service normally runs at 6%, and the right one when it normally runs at
0.1%.
"""

from __future__ import annotations

import enum
import time
from collections.abc import Callable
from dataclasses import dataclass, field


class Severity(enum.StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    rule: str
    severity: Severity
    message: str
    value: float
    threshold: float
    scope: str = ""
    at: float = field(default_factory=time.time)


@dataclass
class Rule:
    name: str
    severity: Severity
    # (summary, baseline) -> (fired, value, threshold, message)
    evaluate: Callable[[dict, dict | None], tuple[bool, float, float, str]]
    min_samples: int = 30
    cooldown_seconds: float = 300.0


def _rate_rule(name, field_name, threshold, severity, label, *, multiplier=None):
    def evaluate(summary: dict, baseline: dict | None):
        value = summary.get(field_name) or 0.0
        limit = threshold
        if multiplier and baseline and baseline.get(field_name):
            # Relative to the baseline where one exists, absolute otherwise.
            limit = max(threshold, baseline[field_name] * multiplier)
        return value > limit, value, limit, f"{label} {value:.2%} exceeds {limit:.2%}"

    return Rule(name=name, severity=severity, evaluate=evaluate)


DEFAULT_RULES: list[Rule] = [
    _rate_rule("error_rate", "error_rate", 0.05, Severity.CRITICAL, "error rate", multiplier=2.0),
    _rate_rule(
        "refusal_rate", "refusal_rate", 0.30, Severity.WARNING, "refusal rate", multiplier=2.0
    ),
    _rate_rule(
        "fallback_rate", "fallback_rate", 0.20, Severity.WARNING, "fallback rate", multiplier=3.0
    ),
    _rate_rule(
        "tool_error_rate",
        "tool_error_rate",
        0.10,
        Severity.WARNING,
        "tool error rate",
        multiplier=2.0,
    ),
    _rate_rule(
        "ungrounded_rate",
        "ungrounded_rate",
        0.20,
        Severity.CRITICAL,
        "ungrounded answer rate",
        multiplier=2.0,
    ),
]


def _latency_rule(max_p95_ms: float = 5000.0) -> Rule:
    def evaluate(summary: dict, baseline: dict | None):
        value = summary.get("p95_ms") or 0.0
        limit = max_p95_ms
        if baseline and baseline.get("p95_ms"):
            # A doubling is a regression even well under the absolute ceiling.
            limit = min(limit, baseline["p95_ms"] * 2)
        return value > limit, value, limit, f"p95 {value:.0f}ms exceeds {limit:.0f}ms"

    return Rule("p95_latency", Severity.WARNING, evaluate)


def _cost_rule(max_usd_per_call: float = 0.05) -> Rule:
    def evaluate(summary: dict, baseline: dict | None):
        value = summary.get("usd_per_call") or 0.0
        limit = max_usd_per_call
        if baseline and baseline.get("usd_per_call"):
            limit = min(limit, baseline["usd_per_call"] * 2)
        return value > limit, value, limit, f"${value:.4f}/call exceeds ${limit:.4f}"

    return Rule("cost_per_call", Severity.WARNING, evaluate)


@dataclass
class AlertEngine:
    rules: list[Rule] = field(
        default_factory=lambda: [
            *DEFAULT_RULES,
            _latency_rule(),
            _cost_rule(),
        ]
    )
    _last_fired: dict[str, float] = field(default_factory=dict)
    history: list[Alert] = field(default_factory=list)

    def evaluate(
        self, summary: dict, *, baseline: dict | None = None, scope: str = ""
    ) -> list[Alert]:
        now = time.time()
        fired: list[Alert] = []

        for rule in self.rules:
            if summary.get("calls", 0) < rule.min_samples:
                continue
            key = f"{rule.name}:{scope}"
            if now - self._last_fired.get(key, 0.0) < rule.cooldown_seconds:
                continue

            triggered, value, threshold, message = rule.evaluate(summary, baseline)
            if not triggered:
                continue

            alert = Alert(
                rule=rule.name,
                severity=rule.severity,
                message=message,
                value=value,
                threshold=threshold,
                scope=scope,
            )
            self._last_fired[key] = now
            self.history.append(alert)
            fired.append(alert)

        return fired
