# Person 3 — P6 decision and ranking release status

Date: 2026-08-31

Status: Person 3's no-cost Phase 6 release work is complete. The measured
decision/ranking fallback is frozen and guarded against configuration drift.
No runtime scoring code changed in this phase.

## Frozen measured configuration

The Phase 5 tuning winner and one-time held-out confirmation remain the only
measured Person 3 release configuration:

| Field | Frozen value |
|---|---|
| Arm | `conservative-questions-deterministic` |
| Arm version | `person3-phase5-v1` |
| Arm fingerprint | `127edd88cdc561f3c150002258e1b1502c0c7c832b740f52f8af3be8e4dec251` |
| Question policy | `p3/conservative-official-proxy-v1` |
| Information-gain threshold | `0.07` |
| Reranker | `p3/deterministic-v2` |
| Profile weight | `0` |
| LLM reranking | disabled in the measured arm |
| Held-out report fingerprint | `6fc74f8cf6fbc3974138837a2e98a5053c66d276f821e61143d620eb5912b027` |

`tests/test_person3_tuning.py` now fails if the executable arm drifts from the
canonical report, if the selected report no longer points to the one-time
held-out confirmation, or if the recorded fixed-ask/LLM controls no longer lose
on the primary Hit Rate@10 objective.

## Selection evidence

Threshold `0.07` won the 140-session tuning sweep with Hit Rate@10 `0.900000`,
MRR `0.502738`, MTTC `5.657143`, and Technical Score `0.707679`. The matched
controls were rejected on tuning:

- fixed-order asking: Hit Rate@10 `0.800000`, MRR `0.378260`;
- anchored `gpt-5.6-terra` at historical `xhigh`: Hit Rate@10 `0.671429`,
  MRR `0.401015`, MTTC `6.735714`, with 169 interpreter and 131 reranker
  fallbacks.

The frozen deterministic finalist then passed the single allowed 60-session
held-out run:

| Scope | Samples | HR@10 | MRR | MTTC | Technical Score |
|---|---:|---:|---:|---:|---:|
| Overall | 60 | 0.933333 | 0.590245 | 5.000000 | 0.763740 |
| Buying | 24 | 0.916667 | 0.497338 | 4.708333 | 0.733368 |
| Browsing | 24 | 1.000000 | 0.679266 | 4.958333 | 0.824613 |
| Intent Override | 9 | 0.888889 | 0.630688 | 5.222222 | 0.749206 |
| Boundary | 3 | 0.666667 | 0.500000 | 7.000000 | 0.563333 |

The held-out run made zero model calls and used zero tokens and API cost.
Canonical evidence is in `reports/p5-threshold-070-held-out.json` and
`reports/p5-selection.json`. Held-out was not reopened during Phase 6.

## Fallback and safety verification

The focused Person 3 and owner-boundary suite verifies:

- deterministic pool diagnostics and stable ranking tie-breaks;
- unique, supplied-shortlist-only IDs and strict `top_k` limits;
- reliable hard-constraint enforcement while retaining unknown metadata;
- turn-10 recommendation and valid turn-9 clarification behavior;
- suppression of answered, no-preference, exhausted, and already-asked
  attributes;
- same-intent repetition handling and eligibility restoration after an intent
  version change;
- deterministic recommendation fallback after decision-component failure;
- LLM timeout/exception, malformed output, hallucination, duplication,
  omission, and partial-output normalization;
- preservation of deterministic eligibility before any semantic reorder;
- deterministic operation without credentials, hidden labels, provider SDKs,
  or network access.

Validation on this checkout:

- focused decision, ranking, tuning, orchestration, and runtime integration:
  121 tests passed after adding the three release-freeze guards;
- repository suite excluding the six known Person 2 dense-memory-map modules:
  446 tests passed and one unrelated test was skipped;
- complete repository suite: 483 tests ran, with ten existing Windows
  `vectors.f32` cleanup errors and one skip. Every error is the documented
  Person 2 `np.memmap` handle issue; there is no Person 3 failure.

## API-primary release route

The settled production default remains API-first `gpt-5.6-terra` at `medium`
reasoning when credentials exist, with deterministic fallback. The historical
`xhigh` LLM arm above was rejected and must not be presented as evidence for
the unmeasured `medium` route. Person 3 therefore makes no live quality, cost,
latency, or promotion claim for `medium`; the deterministic path is the only
held-out-confirmed ranking policy.

## Remaining dependencies and limits

- The external 50,000-product catalog is absent from this checkout, so Phase 6
  did not rerun or alter the frozen held-out evaluation.
- Person 2 still owns the production 1024-dimensional dense artifact, its
  explicitly approved tuning-only hybrid query run, and the Windows memory-map
  cleanup defect.
- A paid `medium` API run requires explicit approval and cannot reopen held-out
  or replace the frozen Person 3 selection using held-out inspection.

Within Person 3 ownership, Phase 6 is complete. Any further decision or ranking
change should be limited to a demonstrated correctness defect and must receive
new tuning evidence before release.
