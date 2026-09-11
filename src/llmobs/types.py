"""What an LLM call looks like once it is worth monitoring.

Deliberately not a generic span. Generic tracing tells you a request took 800ms; it
does not tell you that 800ms cost $0.04, was served by a fallback model, returned an
ungrounded answer, and came from the feature that is losing money. Those are the
questions an LLM application actually raises, so they are fields.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LLMCall:
    # Attribution. `feature` is the one most systems omit and the one that decides
    # what to switch off: cost per tenant tells you who to bill, cost per feature
    # tells you what is not worth running.
    feature: str = "unknown"
    tenant: str = "default"
    model: str = ""
    provider: str = ""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    latency_ms: float = 0.0

    # Outcome, not just success/failure. An LLM call can succeed at the HTTP level
    # and still be a failure of the thing you built.
    error: str = ""
    refused: bool = False
    cached: bool = False
    fallback_used: bool = False
    # Grounding score where the caller has one (RAG). None means not applicable -
    # which is different from zero and must not be averaged in.
    grounding: float | None = None
    tool_calls: int = 0
    tool_errors: int = 0

    # A cheap, privacy-preserving fingerprint of the prompt shape, for drift.
    # The prompt text itself is deliberately not stored.
    prompt_length: int = 0
    prompt_signature: str = ""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    at: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def ok(self) -> bool:
        return not self.error

    def to_dict(self) -> dict:
        return asdict(self)
