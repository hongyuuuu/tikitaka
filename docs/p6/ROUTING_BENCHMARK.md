# Routing benchmark — always-generative vs deterministic

Owner: Person 1. Requested by Person 2: *"run and benchmark the existing
selective API-routing mode… decide when the LLM materially helps."*

Date: 2026-08-30. Route `primary/gpt-5.6-terra` at `xhigh`, list price $2.00/1M
input and $12.00/1M output. Stratified sample, seed 2026, 50 sessions
(browsing 20, buying 19, intent_override 8, boundary 3), both arms run over the
**same** sessions.

## Result

| | Hit@10 | MRR | MTTC | TechnicalScore | cost |
|---|---:|---:|---:|---:|---:|
| Deterministic | **0.880** | 0.5614 | 6.00 | **0.7084** | $0 |
| Always-generative | 0.700 | 0.5177 | 6.46 | 0.5961 | $2.77 |

Per scenario:

| Scenario | n | API | Deterministic |
|---|---:|---:|---:|
| browsing | 20 | 0.850 | 0.950 |
| buying | 19 | 0.632 | 0.895 |
| intent_override | 8 | 0.500 | 0.625 |
| boundary | 3 | 0.667 | 1.000 |

**The API route loses in every scenario.** 404 calls, zero repairs, zero
fallbacks: the route worked exactly as designed and produced worse answers.

The 0.180 gap in Hit@10 exceeds the ±0.127 interval at n=50, so this is
unlikely to be noise, though a firm conclusion wants a few hundred sessions.
The largest deficit is on **buying** — the biggest sample and a 0.26 gap — and
the route also loses on **intent_override**, which is where the build plan
predicted the LLM would earn its cost, because that is where the heuristic
mangles overrides. That prediction is not supported.

## What that means for "when does the LLM materially help"

On this evidence: **nowhere yet measured.** Not on overrides, which was the
strongest prior. Person 2's instruction not to select always-generative blindly
is correct, and the reason is now measured rather than assumed.

The honest reading is narrower than "the LLM is worse". The LLM route replaces
heuristic extraction with model extraction, and the deterministic extractor is
tuned against the public simulator's templates. A model that paraphrases a
constraint the retriever then fails to match will lose on this set while
generalising better to a private simulator that phrases things differently.
That is Risk 5 in the build plan pointing the other way, and it cannot be
settled on the public set at all.

## The selective arm is void

`artifacts/bench_selective.json` must not be reported. The account ran out of
credits partway through:

```
HTTP 429: "You have no credits remaining. Add credits to continue using the API."
```

60 of 63 escalations failed and fell back, so its 0.860 Hit@10 is the
deterministic route with three LLM calls, not a measurement of selective
routing. The file is retained as evidence of the failure mode, marked
`valid_route_measurement: false`.

Selective routing therefore remains **unmeasured**. Offline it escalates on
15.7% of turns after the recalibration below, projecting roughly $1.91 per 200
sessions against $11.07 for always-generative.

## Tooling change this forced

The probe already refused a run that consumed zero tokens. That catches total
degradation and misses the dangerous case: a route that fails *most* turns
still produces tokens, still completes, and still prints a plausible hit rate
that actually belongs to the fallback. The selective arm printed
`hit@10 api 0.860` next to `fallback turns 60` and would have been read as a
success.

The probe now computes the degraded share and, above 20%, prints a blunt
banner saying the quality figures are not a measurement of that route. The
report carries `valid_route_measurement` so a downstream reader cannot miss it.

## SELECTIVE was miscalibrated before this run

Its main signal was a constant. The heuristic emits exactly two
`mode_confidence` values across the public set — 0.00 on 27% of turns and 0.60
on the other 73% — and the threshold was 0.65, above both, so
`low_mode_confidence` fired on 100% of turns and SELECTIVE escalated on 82.3%
of them. Recalibrated to 0.30, between the two observed values, escalation
falls to 15.7%.

| Policy | API turns | Share | Projected /200 |
|---|---:|---:|---:|
| always-generative | 1133 | 100% | $11.07 |
| SELECTIVE as shipped (0.65) | 933 | 82.3% | $10.02 |
| SELECTIVE recalibrated (0.30) | 178 | 15.7% | $1.91 |

A test now asserts the threshold sits strictly between the two observed values.

## Spend

| Run | Cost | Valid |
|---|---:|---|
| 10-session probe (pre-fix) | $0.34 | yes — found the prompt/schema defect |
| 10-session probe (post-fix) | $0.61 | yes |
| 50-session always-generative | $2.77 | yes |
| 50-session selective | $0.16 | **no** — credits exhausted |
| **Total** | **$3.87** | |

## Is the selective arm worth re-running?

Probably not, and the decision does not wait on it.

The void arm still carries usable paired data: across the same 50 sessions the
three escalations that *did* succeed produced **0 sessions won and 1 lost**
against deterministic. Combined with the valid arm, the exposure ladder reads:

| LLM exposure | Hit@10 |
|---|---:|
| 0% (deterministic) | 0.880 |
| ~1% (3 calls, void arm) | 0.860 |
| 100% (always-generative) | 0.700 |

For selective at 15.7% to beat deterministic, the LLM would have to help
specifically on the turns selective picks. It loses in all four scenarios,
including intent_override — where selective's override trigger fires and where
the build plan predicted the strongest gain. On a proportional reading, 15.7%
exposure projects roughly 0.852, still below 0.880.

So the $0.16 re-run would most likely confirm "slightly worse than
deterministic, far cheaper than always-generative", which changes nothing about
what ships. Worth running if credits are topped up for other reasons; not worth
topping up for.

## What no amount of public-set spending can settle

Whether the deterministic extractor is overfit to the public simulator's
templates. It is tuned against them, and a model that paraphrases a constraint
the retriever then fails to match loses here while possibly generalising better
to a private simulator. That is Risk 5, and the public set cannot answer it at
any budget. Only the private run can, and that arrives with the result.

This is the argument for keeping the API route built, tested, and switchable
rather than deleting it — not for enabling it.

## Decision

1. **Deterministic ships. The API route stays off by default.** Established by
   the valid arm, not assumed.
2. Keep the route and SELECTIVE maintained and reachable, as the hedge against
   Risk 5 described above.
3. No further live measurement is required for this decision. Credits are a
   blocker only on optional confirmation.
