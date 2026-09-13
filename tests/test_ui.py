"""Smoke tests for the Streamlit demo (the `ui` dependency group)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "ui" / "app.py")


def _app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    assert not at.exception
    return at


def test_app_loads_without_exceptions():
    _app()


def test_cost_tab_reports_cost_per_success():
    at = _app()
    labels = [m.label for m in at.metric]
    assert "Cost per success" in labels
    assert "Cache hit rate" in labels


def test_identical_windows_do_not_report_drift():
    """The false positive this demo was originally built with: at 150 calls per
    window, PSI reported a latency shift between two windows drawn from the same
    distribution. Guards the sample size that fixed it."""
    at = _app()
    assert any("No significant drift" in s.value for s in at.success)


def test_injected_drift_is_detected():
    at = _app()
    at.checkbox[0].set_value(True).run(timeout=60)
    assert not at.exception
    assert any("Drift detected" in e.value for e in at.error)
