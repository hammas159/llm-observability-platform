"""PSI on two windows drawn from the same distribution, at three sample sizes.

    python demo.py

Nothing drifts here. Both windows are generated from identical parameters,
by the same generator, with no shift applied. The only variable is n.

This is the failure worth knowing before wiring an alert to a PSI threshold:
LLM latency is bimodal because cache hits return in milliseconds and misses
do not, and PSI bins a bimodal metric badly at small n.
"""

import random
import sys

sys.path.insert(0, "src")

from llmobs.drift.psi import psi_numeric

CACHE_HIT_RATE = 0.30
HIT_MS = (1.0, 4.0)
MISS_MS = (180.0, 900.0)


def latencies(n: int, rng: random.Random) -> list[float]:
    """One window of LLM call latencies. Bimodal: cache hits are ~1000x faster."""
    out = []
    for _ in range(n):
        if rng.random() < CACHE_HIT_RATE:
            out.append(rng.uniform(*HIT_MS))
        else:
            out.append(rng.uniform(*MISS_MS))
    return out


print("INPUT")
print("   two windows per row, both from the same generator")
print(f"   cache hit rate     {CACHE_HIT_RATE:.0%}  ->  {HIT_MS[0]}-{HIT_MS[1]} ms")
print(f"   cache miss         {1 - CACHE_HIT_RATE:.0%}  ->  {MISS_MS[0]}-{MISS_MS[1]} ms")
print("   no shift is applied to either window at any sample size")
print()

TRIALS = 300

print("OUTPUT")
print(f"   {TRIALS} independent trials per row. A 'false alarm' is PSI >= 0.10")
print("   -- the conventional 'moderate shift' threshold -- on two windows that")
print("   are drawn from the same distribution.")
print()
print(f"   {'n':>6}  {'median psi':>11}  {'worst psi':>10}  {'false alarms':>13}")
print("   " + "-" * 50)
rng = random.Random(20260916)
for n in (150, 300, 500, 2000):
    scores = sorted(psi_numeric(latencies(n, rng), latencies(n, rng)) for _ in range(TRIALS))
    alarms = sum(1 for s in scores if s >= 0.10)
    print(
        f"   {n:>6}  {scores[len(scores) // 2]:>11.4f}  {scores[-1]:>10.4f}  "
        f"{alarms:>6} / {TRIALS}  ({alarms / TRIALS:>5.1%})"
    )

print()
print("   Nothing drifted in any of these trials. The alarms are the metric")
print("   reacting to sample size, not to a change in the system.")
print("   A PSI threshold on a low-traffic feature is a scheduled false alarm.")
