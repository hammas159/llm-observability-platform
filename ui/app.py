"""The Streamlit demo the `ui` dependency group declared but never shipped.

Three tabs: the cost/latency view an operator actually needs (cost per *success*,
percentiles that exclude cache hits), a per-feature breakdown that says what to switch
off, and prompt drift detected by PSI without a single prompt being stored.

Generates a synthetic but deterministic call stream on load — no provider, no API key.

Run: streamlit run ui/app.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llmobs.platform import Observatory  # noqa: E402
from llmobs.types import LLMCall  # noqa: E402

st.set_page_config(page_title="llm-observability-platform demo", layout="wide")
st.title("llm-observability-platform")
st.caption(
    "Monitoring built for LLM applications rather than retrofitted from web "
    "monitoring: cost per success, percentiles that exclude cache hits, and drift "
    "detected without ever storing a prompt."
)

FEATURES = ["chat", "summarise", "rag_search", "code_assist"]
MODELS = ["local-small", "local-large", "frontier"]
PROMPTS = {
    "chat": "how do I {} my account",
    "summarise": "summarise the following document about {}",
    "rag_search": "find the policy covering {}",
    "code_assist": "write a python function that {}",
}
TOPICS = ["billing", "shipping", "returns", "access", "payroll"]


def _make_call(rng: random.Random, at: float, *, drifted: bool = False) -> tuple[LLMCall, str]:
    feature = rng.choice(FEATURES)
    model = rng.choices(MODELS, weights=[0.6, 0.3, 0.1])[0]
    cached = rng.random() < 0.25
    errored = rng.random() < 0.04
    latency = 2.0 if cached else rng.gauss(600 if model == "frontier" else 180, 60)
    usd = 0.0 if cached else {"local-small": 0.0, "local-large": 0.0, "frontier": 0.004}[model]
    prompt = PROMPTS[feature].format(rng.choice(TOPICS))
    if drifted:
        # Users started pasting whole documents in - the shape of the traffic changed
        # without any code changing.
        prompt = prompt + " " + "context " * rng.randint(40, 120)
        latency *= 2.5
    call = LLMCall(
        feature=feature,
        tenant=rng.choice(["acme", "globex", "initech"]),
        model=model,
        provider="demo",
        prompt_tokens=len(prompt.split()),
        completion_tokens=rng.randint(20, 200),
        usd=usd,
        latency_ms=max(1.0, latency),
        error="timeout" if errored else "",
        refused=rng.random() < 0.05,
        cached=cached,
        grounding=round(rng.uniform(0.4, 1.0), 2) if feature == "rag_search" else None,
        at=at,
    )
    return call, prompt


# 500 calls per window, not 150. LLM latency is bimodal - a cache hit is ~2ms and a
# miss is hundreds - and PSI on a bimodal metric is noisy at small n: at 150 per window
# this demo reported a "moderate shift" in latency between two windows drawn from an
# identical distribution. That is a false positive worth knowing about rather than
# tuning away, and it is why the drift tab says what sample size it is working from.
CALLS_PER_WINDOW = 500


@st.cache_resource
def _observatory(drift_injected: bool) -> Observatory:
    """One hour of reference traffic, then one hour of current traffic that either
    looks the same or has drifted."""
    obs = Observatory()
    rng = random.Random(11)
    now = time.time()
    step = 1800 / CALLS_PER_WINDOW
    for i in range(CALLS_PER_WINDOW):  # reference window: 2h..1h ago
        call, prompt = _make_call(rng, now - 5400 + i * step)
        obs.record(call, prompt=prompt)
    for i in range(CALLS_PER_WINDOW):  # current window: last hour
        call, prompt = _make_call(rng, now - 1800 + i * step, drifted=drift_injected)
        obs.record(call, prompt=prompt)
    return obs


tab_cost, tab_features, tab_drift = st.tabs(["Cost & latency", "By feature", "Prompt drift (PSI)"])

inject_drift = st.sidebar.checkbox("Inject prompt drift in the current window", value=False)
obs = _observatory(inject_drift)

with tab_cost:
    summary = obs.summary(seconds=3600)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calls (last hour)", summary["calls"])
    c2.metric("Total spend (USD)", f"{summary['total_usd']:.4f}")
    c3.metric(
        "Cost per success",
        f"{summary['usd_per_success']:.6f}" if summary["usd_per_success"] else "—",
        help="Not cost per call. A refused or errored call still costs money.",
    )
    c4.metric("Cache hit rate", f"{summary['cache_hit_rate']:.0%}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("p50 latency (ms)", round(summary["p50_ms"]))
    c6.metric("p95 latency (ms)", round(summary["p95_ms"]))
    c7.metric("p99 latency (ms)", round(summary["p99_ms"]))
    c8.metric("Error rate", f"{summary['error_rate']:.1%}")

    st.info(
        "Latency percentiles **exclude cache hits** — a 2ms cache hit makes the model "
        "look fast and hides the tail a user on a miss actually experiences."
    )
    st.json(summary, expanded=False)

with tab_features:
    st.markdown(
        "Cost per *tenant* tells you who to bill. Cost per **feature** tells you what "
        "is not worth running — the breakdown most systems omit."
    )
    by_feature = obs.by("feature", seconds=3600)
    st.dataframe(
        [
            {
                "feature": name,
                "calls": stats["calls"],
                "total_usd": stats["total_usd"],
                "usd_per_success": stats["usd_per_success"],
                "p95_ms": round(stats["p95_ms"]),
                "refusal_rate": stats["refusal_rate"],
            }
            for name, stats in sorted(by_feature.items())
        ],
        hide_index=True,
    )
    st.subheader("Spend by feature")
    st.bar_chart({name: stats["total_usd"] for name, stats in by_feature.items()})

    st.subheader("Spend by tenant")
    by_tenant = obs.by("tenant", seconds=3600)
    st.bar_chart({name: stats["total_usd"] for name, stats in by_tenant.items()})

with tab_drift:
    st.markdown(
        "PSI between the last hour and the hour before it — **adjacent** windows, "
        "because comparing against all history includes the current window and "
        "dilutes the very change being looked for. The prompt itself is never stored: "
        "only a structural signature and a length."
    )
    result = obs.drift(window_seconds=3600)
    if result.get("status") != "ok":
        st.warning(f"Not enough data to compare: {result}")
    else:
        st.write(
            f"Comparing **{result['current_calls']}** current calls against "
            f"**{result['reference_calls']}** reference calls."
        )
        st.caption(
            "Sample size matters more than it looks: at 150 calls per window this "
            "same generator reported a moderate latency shift between two windows "
            "drawn from an identical distribution. Cache hits make LLM latency "
            "bimodal, and PSI on a bimodal metric is noisy when n is small."
        )
        rows = [
            {"feature": k, "psi": round(v, 4), "verdict": result["verdict"][k]}
            for k, v in result["psi"].items()
        ]
        st.dataframe(rows, hide_index=True)
        st.bar_chart({r["feature"]: r["psi"] for r in rows})
        if any(r["verdict"] != "stable" for r in rows):
            st.error("Drift detected — tick/untick the sidebar box to compare.")
        else:
            st.success("No significant drift between the two windows.")
