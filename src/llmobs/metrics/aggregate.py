"""Aggregation over a window of calls.

Two rules run through everything here.

**Percentiles, never means.** A mean latency of 400ms is compatible with every user
waiting 400ms and with 95% waiting 100ms while 5% wait 6 seconds. Only one of those is
a problem, and the mean cannot distinguish them.

**Cost is attributed, not totalled.** A single number for monthly spend tells you
nothing actionable. Cost per *feature* tells you what to switch off; cost per
*successful outcome* tells you whether it was worth running at all.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from ..types import LLMCall


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank. Unambiguous, and on small samples it does not invent a value
    that no request actually had."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(p / 100 * len(ordered))) - 1))
    return round(ordered[index], 3)


def summarise(calls: Iterable[LLMCall]) -> dict:
    calls = list(calls)
    if not calls:
        return {"calls": 0}

    served = [c for c in calls if c.ok]
    latencies = [c.latency_ms for c in served]
    # Cache hits are excluded from latency percentiles: a 2ms cache hit makes the
    # model look fast and hides the tail that users on a miss actually experience.
    uncached = [c.latency_ms for c in served if not c.cached]

    total_usd = sum(c.usd for c in calls)
    successes = [c for c in served if not c.refused]

    grounded = [c.grounding for c in calls if c.grounding is not None]

    return {
        "calls": len(calls),
        "errors": sum(1 for c in calls if c.error),
        "error_rate": round(sum(1 for c in calls if c.error) / len(calls), 4),
        "refusal_rate": round(sum(1 for c in calls if c.refused) / len(calls), 4),
        "cache_hit_rate": round(sum(1 for c in calls if c.cached) / len(calls), 4),
        "fallback_rate": round(sum(1 for c in calls if c.fallback_used) / len(calls), 4),
        "p50_ms": percentile(uncached or latencies, 50),
        "p95_ms": percentile(uncached or latencies, 95),
        "p99_ms": percentile(uncached or latencies, 99),
        "tokens": sum(c.total_tokens for c in calls),
        "total_usd": round(total_usd, 6),
        "usd_per_call": round(total_usd / len(calls), 8),
        # The number that decides whether a feature stays switched on.
        "usd_per_success": round(total_usd / len(successes), 8) if successes else None,
        "tool_error_rate": (
            round(sum(c.tool_errors for c in calls) / sum(c.tool_calls for c in calls), 4)
            if sum(c.tool_calls for c in calls)
            else 0.0
        ),
        # Averaged over calls that reported one. A call with no grounding score is
        # not a call with a grounding score of zero.
        "mean_grounding": round(sum(grounded) / len(grounded), 4) if grounded else None,
        "ungrounded_rate": (
            round(sum(1 for g in grounded if g < 0.5) / len(grounded), 4) if grounded else None
        ),
    }


def group_by(calls: Iterable[LLMCall], key: str) -> dict[str, dict]:
    """Summarise per feature, tenant, model or provider."""
    buckets: dict[str, list[LLMCall]] = defaultdict(list)
    for call in calls:
        buckets[str(getattr(call, key, "unknown"))].append(call)
    return {name: summarise(group) for name, group in sorted(buckets.items())}


def most_expensive(calls: Iterable[LLMCall], key: str = "feature", limit: int = 5) -> list[dict]:
    """Ranked by spend, with the per-success figure alongside.

    Ranking by total spend alone promotes whatever is merely popular. The pair is what
    identifies a feature that is both costly and ineffective.
    """
    grouped = group_by(calls, key)
    ranked = sorted(grouped.items(), key=lambda kv: kv[1].get("total_usd", 0), reverse=True)
    return [
        {
            key: name,
            "total_usd": stats["total_usd"],
            "calls": stats["calls"],
            "usd_per_success": stats["usd_per_success"],
            "error_rate": stats["error_rate"],
        }
        for name, stats in ranked[:limit]
    ]
