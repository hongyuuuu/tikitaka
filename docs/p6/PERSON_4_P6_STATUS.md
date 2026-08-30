# Person 4 — P6 release, evidence, and integration status

Date: 2026-08-31

Status: all no-cost Person 4 P6 work is complete. The only remaining numerical
release dependency is Person 2's production 1024-dimensional dense artifact and
the explicitly approved tuning-only hybrid query run.

## Completed release work

- Shared contracts, isolated orchestration, the thin official adapter, stable
  evaluation split, canonical reporting, leakage guards, and release packaging
  are integrated.
- The owner-selected default is API-first `gpt-5.6-terra` at `medium`, with an
  automatic deterministic fallback and no local generative LLM.
- The `medium` route has no live quality, latency, token, or cost claim.
  Submission documentation now labels all measured API figures as historical
  `xhigh` evidence.
- The canonical offline package audit completed all 200 sessions with no
  credential, zero socket/DNS attempts, zero model tokens, no participant
  exceptions, and TechnicalScore `0.705672`.
- Package policy excludes the catalog, evaluator, public labels, credentials,
  generated indexes, tests, reports, caches, and transient output.
- The committed Buying, Browsing, Intent Override, and Boundary demo traces are
  non-empty, label-free, credential-free, network-free, byte-reproducible, and
  contract-valid. Intent Override reaches intent version 2. A miss remains a
  valid failure-analysis trace and is not replaced by a favorable sample.
- Fake-only accounting reconciliation proves that interpreter, repair, and
  reranker usage events aggregate without local arithmetic loss and reports
  per-component tokens, calls, repairs, and estimated cost.
- The prepared sparse control command was executed on the tuning partition
  only: Hit Rate@10 `0.900000`, MRR `0.502738`, MTTC `5.657143`, Efficiency
  `0.534286`, and TechnicalScore `0.707679`. Held-out remained `not_run`.

## Historical accounting gap

The historical `xhigh` provider bill cannot be reconciled retroactively because
the billing total covered a shared day rather than an isolated run window.
Local fake tests rule out simple merge/aggregation arithmetic as the cause, but
they cannot distinguish provider usage omitted from response bodies, unrelated
account activity, or an unrecorded production call path. A future controlled
check requires provider totals immediately before and after a known call set.
No such paid check is required for the current release gate.

## Demo trace verification

| Scenario | Turns | Hit | Final intent version | Model calls |
|---|---:|---:|---:|---:|
| Buying | 7 | yes | 1 | 0 |
| Browsing | 6 | yes | 1 | 0 |
| Intent Override | 10 | no | 2 | 0 |
| Boundary | 6 | yes | 1 | 0 |

The capture command is network-free by default even if the shell contains an
API credential. `--allow-api` is required to permit billable trace capture.

## Person 4 contribution summary

- Owned the dependency-light contracts and fake-first integration spine.
- Implemented session isolation, response normalization, failure containment,
  usage aggregation, and the official adapter boundary.
- Built tuning/held-out splits, experiment fingerprints, reporting, P5
  selection gates, and per-scenario evidence.
- Added label-leakage guards, deterministic demo traces, clean-package
  reproduction, secret/artifact audits, and offline network denial.
- Coordinated the API-default `medium` owner decision and preserved historical
  `xhigh` evidence without relabeling it.
- Published the production-index acceptance and tuning-only comparison gate for
  Person 2's handoff.

## Remaining dependency

Follow `docs/p6/PERSON_4_PRODUCTION_INDEX_HANDOFF.md` when Person 2 delivers the
artifact. Do not run paid embedding or model jobs without explicit approval,
and do not reopen held-out for route selection.
