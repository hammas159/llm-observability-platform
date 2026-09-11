"""Observability tests.

Synthetic call streams rather than recorded traffic, so every scenario worth checking
— a drifting input distribution, a feature that is expensive and ineffective, a shared
outage — can be constructed exactly and asserted on.
"""

from __future__ import annotations

import time

import pytest

from llmobs.alerts.rules import AlertEngine, Severity
from llmobs.drift.psi import interpret, psi, psi_categorical, psi_numeric
from llmobs.drift.signature import length_bucket, question_type, signature
from llmobs.metrics.aggregate import group_by, most_expensive, percentile, summarise
from llmobs.platform import Observatory
from llmobs.types import LLMCall


def call(**kw) -> LLMCall:
    kw.setdefault("feature", "search")
    kw.setdefault("model", "local-small")
    kw.setdefault("latency_ms", 100.0)
    return LLMCall(**kw)


class TestPercentiles:
    def test_tail_is_not_hidden_by_the_mean(self):
        """A mean of 400ms fits both 'everyone waits 400ms' and '5% wait 6 seconds'."""
        values = [100.0] * 95 + [6000.0] * 5
        assert percentile(values, 50) == 100.0
        assert percentile(values, 99) == 6000.0

    def test_empty_is_zero_not_an_error(self):
        assert percentile([], 95) == 0.0

    def test_single_value(self):
        assert percentile([42.0], 95) == 42.0


class TestSummarise:
    def test_empty_window(self):
        assert summarise([])["calls"] == 0

    def test_rates_are_counted(self):
        calls = [call(error="boom"), call(refused=True), call(cached=True), call()]
        s = summarise(calls)
        assert s["error_rate"] == 0.25
        assert s["refusal_rate"] == 0.25
        assert s["cache_hit_rate"] == 0.25

    def test_cache_hits_are_excluded_from_latency(self):
        """A 2ms cache hit makes the model look fast and hides the real tail."""
        calls = [call(latency_ms=2.0, cached=True) for _ in range(90)]
        calls += [call(latency_ms=900.0) for _ in range(10)]
        assert summarise(calls)["p95_ms"] == 900.0

    def test_usd_per_success_excludes_refusals(self):
        """The number that decides whether a feature stays switched on."""
        calls = [call(usd=1.0, refused=True), call(usd=1.0)]
        assert summarise(calls)["usd_per_success"] == 2.0

    def test_missing_grounding_is_not_treated_as_zero(self):
        """A call with no grounding score is not a call scoring zero."""
        calls = [call(grounding=0.9), call(grounding=0.8), call()]
        assert summarise(calls)["mean_grounding"] == pytest.approx(0.85, abs=1e-4)

    def test_grounding_is_none_when_nothing_reported_one(self):
        assert summarise([call(), call()])["mean_grounding"] is None

    def test_tool_error_rate_is_per_tool_call_not_per_request(self):
        calls = [call(tool_calls=10, tool_errors=1), call(tool_calls=10, tool_errors=1)]
        assert summarise(calls)["tool_error_rate"] == 0.1

    def test_no_tool_calls_does_not_divide_by_zero(self):
        assert summarise([call()])["tool_error_rate"] == 0.0


class TestAttribution:
    def test_group_by_feature(self):
        calls = [call(feature="search"), call(feature="search"), call(feature="summarise")]
        grouped = group_by(calls, "feature")
        assert grouped["search"]["calls"] == 2
        assert grouped["summarise"]["calls"] == 1

    def test_most_expensive_reports_efficiency_alongside_spend(self):
        """Ranking on total spend alone just promotes whatever is popular."""
        calls = [call(feature="cheap", usd=0.001) for _ in range(100)]
        calls += [call(feature="costly", usd=1.0, refused=True) for _ in range(5)]
        top = most_expensive(calls, "feature")
        assert top[0]["feature"] == "costly"
        # Refused every time, so there is no cost-per-success to report.
        assert top[0]["usd_per_success"] is None


class TestPSI:
    def test_identical_distributions_score_zero(self):
        assert psi([0.5, 0.5], [0.5, 0.5]) == 0.0

    def test_shifted_distribution_scores_high(self):
        assert psi([0.9, 0.1], [0.1, 0.9]) > 0.25

    def test_zero_bucket_does_not_produce_infinity(self):
        """A category present in one window and absent in the other is normal."""
        assert psi([1.0, 0.0], [0.5, 0.5]) < float("inf")

    def test_mismatched_bucket_counts_are_rejected(self):
        with pytest.raises(ValueError):
            psi([0.5, 0.5], [0.3, 0.3, 0.4])

    def test_numeric_drift_is_detected(self):
        reference = [float(i % 10) for i in range(500)]
        current = [float(50 + i % 10) for i in range(500)]
        assert psi_numeric(reference, current) > 0.25

    def test_stable_numeric_data_is_not_flagged(self):
        reference = [float(i % 10) for i in range(500)]
        current = [float(i % 10) for i in range(500)]
        assert psi_numeric(reference, current) < 0.10

    def test_constant_reference_has_no_distribution_to_shift(self):
        assert psi_numeric([5.0] * 100, [9.0] * 100) == 0.0

    def test_categorical_drift(self):
        assert psi_categorical(["a"] * 100, ["b"] * 100) > 0.25

    def test_categorical_stable(self):
        assert psi_categorical(["a", "b"] * 50, ["a", "b"] * 50) < 0.10

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.05, "stable"), (0.15, "moderate shift"), (0.40, "significant shift")],
    )
    def test_interpretation_bands(self, value, expected):
        assert interpret(value) == expected


class TestSignature:
    def test_prompt_text_is_not_recoverable(self):
        """The prompt is never stored; only its shape."""
        sig = signature("What is the capital of Pakistan?")
        assert "capital" not in sig and "Pakistan" not in sig

    def test_same_shape_different_content_matches(self):
        """Two prompts asking the same kind of question about different things."""
        assert signature("What is the capital of Pakistan?") == signature(
            "What is the capital of Morocco?"
        )

    def test_different_shapes_differ(self):
        assert signature("What is X?") != signature("Why did X happen and how?")

    @pytest.mark.parametrize(
        ("prompt", "expected"),
        [
            ("What is X", "factual"),
            ("Why did it fail", "causal"),
            ("How do I do this", "procedural"),
            ("Is this correct", "yes_no"),
            ("Write me a poem", "imperative"),
            ("xyz", "other"),
        ],
    )
    def test_question_types(self, prompt, expected):
        assert question_type(prompt) == expected

    def test_length_buckets(self):
        assert length_bucket("hi there") == "xs"
        assert length_bucket(" ".join(["word"] * 500)) == "xl"


class TestObservatory:
    def test_prompt_is_not_stored(self):
        obs = Observatory()
        recorded = obs.record(call(), prompt="my secret medical question about X")
        assert "secret" not in recorded.prompt_signature
        assert recorded.prompt_length == 6

    def test_window_is_time_based(self):
        """A count window calls a traffic change a quality change."""
        obs = Observatory()
        now = time.time()
        obs.ingest([call(at=now - 10_000) for _ in range(100)])
        obs.ingest([call(at=now) for _ in range(5)])
        assert obs.summary(3600)["calls"] == 5

    def test_drift_needs_both_windows(self):
        obs = Observatory()
        obs.ingest([call() for _ in range(50)])
        assert obs.drift()["status"] == "insufficient data"

    def test_drift_detects_a_moved_input_distribution(self):
        obs = Observatory()
        now = time.time()
        # Reference window: short factual questions.
        obs.ingest([call(at=now - 5400, prompt_length=8,
                         prompt_signature="factual:xs:aaa") for _ in range(100)])
        # Current window: long procedural ones. Nothing errored; quality would just
        # quietly fall, which is the failure nobody instruments.
        obs.ingest([call(at=now - 60, prompt_length=250,
                         prompt_signature="procedural:l:bbb") for _ in range(100)])
        result = obs.drift(window_seconds=3600)
        assert result["status"] == "ok"
        assert result["psi"]["prompt_signature"] > 0.25
        assert result["verdict"]["prompt_signature"] == "significant shift"

    def test_stable_traffic_is_not_flagged(self):
        obs = Observatory()
        now = time.time()
        for offset in (5400, 60):
            obs.ingest([call(at=now - offset, prompt_length=10 + i % 5,
                             prompt_signature="factual:xs:aaa") for i in range(100)])
        assert obs.drift(window_seconds=3600)["verdict"]["prompt_signature"] == "stable"

    def test_sink_is_written_per_call(self, tmp_path):
        """A process that dies must still leave its data."""
        sink = tmp_path / "calls.jsonl"
        obs = Observatory(sink=sink)
        obs.record(call())
        obs.record(call())
        assert len(sink.read_text(encoding="utf-8").strip().splitlines()) == 2

    def test_reload_from_disk(self, tmp_path):
        sink = tmp_path / "calls.jsonl"
        obs = Observatory(sink=sink)
        obs.record(call(usd=0.5))
        assert Observatory.load(sink).summary()["total_usd"] == 0.5


class TestAlerts:
    def test_no_alert_below_the_sample_floor(self):
        """An alerting system people mute is worse than no alerting system."""
        engine = AlertEngine()
        assert engine.evaluate(summarise([call(error="x") for _ in range(5)])) == []

    def test_error_rate_alert_fires(self):
        engine = AlertEngine()
        calls = [call(error="x") for _ in range(50)] + [call() for _ in range(50)]
        fired = engine.evaluate(summarise(calls))
        assert any(a.rule == "error_rate" and a.severity is Severity.CRITICAL for a in fired)

    def test_threshold_is_relative_to_baseline(self):
        """5% errors is the wrong alarm for a service that normally runs at 6%."""
        engine = AlertEngine()
        calls = [call(error="x") for _ in range(8)] + [call() for _ in range(92)]
        baseline = summarise([call(error="x") for _ in range(10)]
                             + [call() for _ in range(90)])
        assert not any(a.rule == "error_rate"
                       for a in engine.evaluate(summarise(calls), baseline=baseline))

    def test_cooldown_prevents_repeat_firing(self):
        engine = AlertEngine()
        calls = [call(error="x") for _ in range(50)] + [call() for _ in range(50)]
        first = engine.evaluate(summarise(calls), scope="global")
        second = engine.evaluate(summarise(calls), scope="global")
        assert first and not second

    def test_different_scopes_alert_independently(self):
        engine = AlertEngine()
        calls = [call(error="x") for _ in range(50)] + [call() for _ in range(50)]
        assert engine.evaluate(summarise(calls), scope="feature:a")
        assert engine.evaluate(summarise(calls), scope="feature:b")

    def test_ungrounded_answers_are_critical(self):
        engine = AlertEngine()
        calls = [call(grounding=0.1) for _ in range(50)] + [call(grounding=0.9) for _ in range(50)]
        fired = engine.evaluate(summarise(calls))
        assert any(a.rule == "ungrounded_rate" and a.severity is Severity.CRITICAL
                   for a in fired)

    def test_one_broken_feature_is_visible_despite_a_healthy_aggregate(self):
        """A feature failing every single call, completely invisible in the total.

        Sized so the global error rate (50/2050 = 2.4%) stays under the 5% threshold
        while the broken feature sits at 100%. That gap is the whole argument for
        evaluating rules per feature and not only in aggregate.
        """
        obs = Observatory()
        now = time.time()
        obs.ingest([call(at=now, feature="healthy") for _ in range(2000)])
        obs.ingest([call(at=now, feature="broken", error="boom") for _ in range(50)])

        assert obs.summary()["error_rate"] < 0.05  # aggregate looks fine

        fired = obs.check_alerts()
        assert any(a.scope == "feature:broken" and a.rule == "error_rate" for a in fired)
        assert not any(a.scope == "global" and a.rule == "error_rate" for a in fired)
