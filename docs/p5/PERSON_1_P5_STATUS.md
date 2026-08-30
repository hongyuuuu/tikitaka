# Person 1 — P5 Status: Fault and Degraded-Path Hardening

Owner: Person 1. Phase reference: `docs/PERSON_1_BUILD_PLAN.md` section 10.

## Exit gate

| Requirement | Status |
|---|---|
| Zero uncaught exceptions across the fault matrix | met |
| Operation flood truncated deterministically | met |
| Constraints per attribute capped | met |
| Network-free run produces valid deterministic output | met |
| No secret in traces, reports, or committed fixtures | met |
| Degraded-path quality delta recorded | met — see `docs/p6/ROUTING_BENCHMARK.md` |

## Fault matrix

`tests/test_fault_matrix.py` drives every fault through `build_agent` and the
official `respond` contract, not through Person 1 internals. The bar is not that
the agent answers *well* under failure but that it answers *validly*: the
evaluator scores a malformed response as a miss and an exception as a zero.

Covered: malformed JSON, empty response, prose instead of JSON, a JSON array
where an object was required, truncated JSON, provider refusal, timeout,
connection error, rate limiting, server error, an unexpected component
exception, a 500-operation flood, and a hallucinated attribute.

Each response is asserted against the official contract: string `message`,
`ask_attribute` from the closed enum or null, at most ten unique catalog-valid
recommendations, DG-01 mutual exclusion, and non-negative integer usage.

Two further cases: a route that fails on every turn still completes all ten
turns validly, and a transient failure does not permanently pin the agent to
the fallback.

## Network-free run

Full official evaluator, unchanged, with no credential in the environment:

```text
metric           baseline   keyless    delta
hit_rate@10      0.1250     0.8950     +0.7700
mrr              0.0680     0.5424     +0.4744
mttc             9.81       5.74       -4.08
efficiency       0.1190     0.5265     +0.4075
TechnicalScore   0.1067     0.7155     +0.6088

reported tokens  prompt 0, completion 0
runtime          ~58s
```

Zero reported tokens is itself evidence: no provider call occurred, and the
usage plumbing reports honestly rather than inventing numbers.

Per scenario:

| Scenario | n | Hit@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| browsing | 80 | 0.9875 | 0.6519 | 5.51 |
| buying | 80 | 0.8625 | 0.4737 | 5.41 |
| boundary | 10 | 0.9000 | 0.5961 | 6.60 |
| intent_override | 30 | 0.7333 | 0.4155 | 6.90 |

**This is the full 200-session public set, i.e. the tuning split. It is not a
held-out number and must not be reported as one.**

## Two findings worth the team's attention

1. **Intent Override is the weakest scenario by a clear margin** — 0.733 Hit@10
   against 0.988 for browsing, and the worst MTTC at 6.9. That is exactly where
   `test_api_route_recovers_an_override_the_heuristic_mangles` shows the
   deterministic route shredding the message, so it is the most likely place
   for the LLM route to earn its cost.

2. **The deterministic route already scores 0.7155.** Risk 5 in the build plan —
   that the heuristic may be competitive because the public simulator is
   templated — is no longer hypothetical. The ten-session probe should
   establish whether `gpt-5.6-terra` beats this before any budget is committed,
   and the comparison must be made on the held-out split rather than here.

## Hardening added during this phase

`tikitaka/state/trace.py` now redacts credential-shaped text from the `failure`
field. Our transport never echoes a credential, but a trace is written to disk
and pasted into reports, and a future provider error or dependency exception
might carry one. Verified before and after: previously a key in error text
landed in the trace verbatim.

A test also scans every committed fixture for credential-shaped strings, so a
real key cannot reach the repository through test data.

## What remains

Nothing. The API-versus-heuristic quality delta was the last open item and was
measured on 2026-08-30 over 50 paired sessions:

| | Hit@10 | TechnicalScore |
|---|---:|---:|
| Deterministic | 0.880 | 0.7084 |
| API, always-generative | 0.700 | 0.5961 |

The degraded path is not a degradation. It is the better route on every
scenario measured, and the deterministic route is what ships. Full method,
caveats and cost in `docs/p6/ROUTING_BENCHMARK.md`.
