# Person 4 Phased Build Plan

## 1. Role and outcome

Person 4 owns the integration spine of TikiTaka:

- dependency-light shared contracts;
- end-to-end orchestration and isolated sessions;
- the thin official `starter/agent.py` adapter;
- evaluation splits, experiments, reporting, and reproducibility;
- integration tests, release documentation, and submission evidence.

The goal is to make useful vertical slices available early, then integrate the
other three workstreams without absorbing their implementation ownership.
Person 4 should not implement provider logic, retrieval algorithms, decision
policy, or reranking inside the orchestrator.

This plan follows the repository authority order. If an implementation detail
conflicts with an official contract or `ARCHITECTURE.md`, stop, record the
conflict in `docs/IMPLEMENTATION_PLAN.md`, and ask the owner before changing a
settled rule.

## 2. Current starting point

As of the initial checkout:

- `starter/agent.py` is a self-contained stateless BM25 baseline;
- `tikitaka/` and the proposed internal packages do not yet exist;
- the official evaluator and public dataset must remain unchanged;
- `python3 -m unittest` passes 3 baseline tests;
- the initial baseline metrics are already recorded in
  `docs/baseline_results.json`;
- all implementation milestones are still ahead.

Person 4 should preserve this baseline until the first delegated vertical slice
passes the same official tests.

## 3. Working rules for Person 4

1. Announce proposed changes to shared contracts, dependency manifests,
   `ARCHITECTURE.md`, `AGENTS.md`, and other coordination surfaces to all four
   owners before merging.
2. Land contracts and fakes before asking another owner to integrate a concrete
   provider.
3. Keep imports one-way: `starter` delegates to orchestration; orchestration
   depends on protocols; provider implementations satisfy those protocols.
4. Never pass an evaluator sample, scenario label, ground truth, intent card, or
   evaluator object into participant code.
5. Keep evaluation-only knowledge under `tikitaka/evaluation/` and out of all
   runtime request objects.
6. Require exact experiment configuration and per-scenario results for every
   score claim.
7. Do not edit `evaluator/`, the frozen catalog, `data/public_set.jsonl`, or
   public labels to obtain a better result.
8. Use tiny synthetic fixtures for unit and integration tests. Full-catalog and
   live-model runs are explicit integration or evaluation jobs.
9. Keep Python 3.10 compatibility and credentials in environment variables.
10. Treat API failure as a runtime route change to the deterministic path, not
    permission to introduce a local generative LLM.

## 4. Phase summary

| Phase | Person 4 outcome | External dependency | Exit gate |
|---|---|---|---|
| P0 | Baseline and contract proposal | All owners acknowledge interfaces | Untouched tests and baseline reproduced |
| P1 | Shared contracts plus fake components | Contract review from Persons 1-3 | Contract and malformed-input tests pass |
| P2 | Deterministic orchestration vertical slice | Minimal fakes; then owner implementations | Official lifecycle works end to end |
| P3 | Honest experiment and reporting harness | Stable evaluator-facing configuration | Reproducible tuning/held-out report |
| P4 | Workstream integration and hardening | Deliverables from Persons 1-3 | All integration and leakage tests pass |
| P5 | Ablation and configuration selection | Full retrieval/model routes | Selected config has held-out evidence |
| P6 | Offline contingency and release | Frozen artifacts and dependencies | Clean, one-command submission run |

## 5. Phase P0 — Baseline, boundaries, and contract proposal

### Outcome

Establish a known-good baseline and publish the smallest interface proposal that
lets all four people work independently.

### Tasks

- Record the current commit, Python version, catalog checksum, dataset checksum,
  catalog row count, and baseline configuration.
- Run `python3 -m unittest` without modifying official files.
- Re-run `python3 -m evaluator.local_evaluator` and compare it with
  `docs/baseline_results.json` before replacing or recording any baseline.
- Confirm branch and file ownership with all four people.
- Propose the shared dataclasses and protocols in a short contract review:
  `Constraint`, `StateOperation`, `StateDelta`, `Candidate`, `TurnDecision`,
  `Usage`, `SearchPlan`, and the interpreter/retriever/decision/reranker
  protocols.
- Resolve naming, optional fields, enums, and serialization boundaries before
  provider-specific work begins.
- Define a contract version and the migration rule for later changes.
- Agree that candidate product IDs originate only from catalog validation and
  that LLM reranking may only reorder a supplied shortlist.

### Person 4 artifacts

- contract proposal and owner acknowledgements;
- tiny synthetic catalog and conversation fixture design;
- baseline evidence stored outside official artifacts;
- integration status checklist for Persons 1-3.

### Exit gate

- Current official tests pass unchanged.
- The untouched baseline is reproducible.
- Persons 1-3 acknowledge the protocol fields they consume or produce.
- No unresolved contract conflict is hidden in implementation code.

## 6. Phase P1 — Shared contracts and fake-first scaffold

### Outcome

Create a dependency-light internal API and deterministic fakes so orchestration
can be built before live models or full indexes exist.

### Owned implementation

Create:

```text
tikitaka/
  __init__.py
  config.py
  contracts/
    __init__.py
    domain.py
    model.py
  orchestration/
    __init__.py
  evaluation/
    __init__.py
tests/
  fixtures/
    tiny_catalog.jsonl
  fakes/
    components.py
  test_contracts.py
```

Contract design requirements:

- use enums or strict literals for operations, polarity, strength, inferred
  mode, and action;
- make confidence validation and clamping behavior explicit;
- represent missing product metadata as unknown, not false;
- retain per-route ranks and product evidence on every `Candidate`;
- keep `Usage` provider-neutral while preserving model, provider, reasoning,
  latency, tokens, and estimated cost;
- keep experiment pins separate from runtime automatic-routing decisions;
- couple an embedding route to index identity and manifest metadata;
- avoid importing provider SDKs, evaluator code, or large data libraries from
  `tikitaka/contracts/`.

Fake component requirements:

- deterministic intent interpreter producing scripted `StateDelta` values;
- deterministic retriever returning catalog-valid evidence-bearing candidates;
- deterministic decision policy supporting `CLARIFY` and `RECOMMEND`;
- deterministic reranker that can only reorder its input shortlist;
- malformed and exception-throwing variants for failure tests;
- fake usage records for accounting tests.

### Tests

- construction and validation of every shared contract;
- rejection of unknown operations, attributes, actions, and modes;
- confidence boundary behavior;
- usage values cannot become negative;
- candidate evidence and shortlist identity are preserved;
- reranker fakes cannot introduce an out-of-shortlist ID;
- fake outputs are stable for a fixed seed and configuration.

### Exit gate

- `tikitaka/contracts/` imports without optional provider dependencies.
- All fake components satisfy the published protocols.
- Persons 1-3 can write their owned modules without importing each other's
  implementations.
- Contract tests pass on Python 3.10 or later.

## 7. Phase P2 — Deterministic orchestration vertical slice

### Outcome

Replace the monolithic baseline entry point with a thin adapter over a complete,
fake-driven per-turn pipeline.

### Owned implementation

Create:

```text
tikitaka/orchestration/
  shopping_agent.py
  sessions.py
starter/agent.py
tests/
  test_orchestration.py
  test_agent_contract.py
```

Implement this sequence in `ShoppingAgent.respond`:

```text
validate request
  -> load isolated session
  -> interpret visible message
  -> validate/apply state delta through the Person 1 interface
  -> build search plan
  -> retrieve candidates
  -> choose exactly one action
  -> rerank only for RECOMMEND
  -> normalize and validate official response
  -> record asked/shown/usage history
```

Orchestration rules:

- `reset` creates or replaces only the named session and retains a defensive
  snapshot of the supplied profile.
- `respond` accepts only `session_id`, `user_message`, `turn`, and `top_k`.
- Reject or safely handle calls before reset, invalid turns, and unknown
  sessions without exposing internal data.
- Enforce DG-01 centrally: `CLARIFY` has one allowed `ask_attribute` and no
  recommendations; `RECOMMEND` has `ask_attribute = null`.
- Normalize recommendations to the first 10 unique catalog-valid IDs in ranked
  order, even if a component returns duplicates or malformed items.
- Do not trust model-generated product IDs, scores, operations, or attributes.
- Do not spend turn 10 on a clarification that cannot receive a reply.
- Keep same-intent shown-product history distinct from history for a new
  `intent_version`.
- Aggregate usage once per component call and prevent retry double counting.
- On component failure, return a contract-valid response; once local retrieval
  is integrated, prefer the deterministic recommendation path over an empty
  response.
- Keep `starter/agent.py` limited to configuration, construction, and delegation.

### Tests

- official reset/respond lifecycle and session isolation;
- overwriting a session through a second reset;
- output schema and allowed `ask_attribute` values;
- mutual exclusivity of clarification and recommendation;
- first-10, unique, valid-ID normalization;
- malformed component outputs and raised exceptions;
- turn bounds and final-turn recommendation behavior;
- shown history within an intent and eligibility after an intent-version change;
- non-negative, attributable usage aggregation;
- a spy proving participant components receive no `ground_truth`,
  `scenario_type`, intent card, or evaluator internals.

### Exit gate

- `starter.Agent` delegates to the internal orchestrator.
- The fake pipeline completes representative Buying, Browsing, Override, and
  Boundary traces.
- `python3 -m unittest` passes with the official tests unchanged.
- A deterministic end-to-end run emits only contract-valid responses.

## 8. Phase P3 — Evaluation, splits, experiments, and reports

### Outcome

Build an evaluation layer that can compare configurations honestly without
leaking labels into the Agent.

### Owned implementation

Create:

```text
tikitaka/evaluation/
  splits.py
  experiment.py
  reporting.py
  ablations.py
scripts/
  run_experiment.py
tests/
  test_evaluation.py
```

Evaluation design:

- generate a stable, versioned tuning/held-out split using public `sample_id`
  values;
- use `scenario_type` only in the evaluation layer for stratification and
  reporting, never as an Agent feature;
- store split membership or its deterministic derivation so results can be
  reproduced exactly;
- keep ground truth inside scoring code only;
- define immutable experiment configuration covering prompt/schema versions,
  runtime versus pinned routing, model/provider/reasoning, embedding/index
  identity, reranker, fusion parameters, profile weight, question policy, seed,
  catalog checksum, and code revision;
- place caches only at deterministic, versioned boundaries and include all
  behavior-changing inputs in cache keys;
- record aggregate and per-scenario Hit Rate@10, MRR, MTTC, Efficiency, and
  TechnicalScore;
- record question counts and asked-attribute distribution;
- record prompt/completion/total tokens, latency, model identity, provider,
  reasoning level, estimated cost, failures, retries, and fallback activations;
- preserve per-session results for error analysis while excluding secrets and
  raw credentials from reports;
- reject comparisons whose catalog, split, or index identity differs unless the
  difference is the declared experiment variable.

### Tests

- same split seed/version produces identical membership;
- tuning and held-out membership are disjoint and complete;
- scenario stratification is stable;
- labels are unavailable at the Agent call boundary;
- same deterministic config produces byte-stable normalized metrics;
- usage and latency aggregate to the correct component and route;
- report generation includes all required metrics and configuration fields;
- cache invalidates on prompt, model, embedding, index, catalog, or code-version
  changes.

### Exit gate

- One command runs a named experiment and writes a reproducible report.
- Tuning and held-out results are shown separately.
- Every score can be traced to configuration, split, code revision, and index
  manifest.
- The original evaluator and official tests remain unchanged and passing.

## 9. Phase P4 — Integrate Persons 1, 2, and 3

### Outcome

Replace fakes one boundary at a time while keeping the vertical slice runnable
after every merge.

### Integration order

1. **Person 1 state boundary:** integrate the fake interpreter, validated state
   reducer, and query builder before the live API adapter. Verify state
   accumulation, retraction, replacement, exclusion, no-preference, reset,
   mode changes, and dependency-aware intent override.
2. **Person 2 retrieval boundary:** integrate catalog validation, lexical and
   structured retrieval, then the dense route and fusion. Verify candidate
   evidence, catalog validity, deterministic order, missing-metadata safeguards,
   and embedding/index compatibility.
3. **Person 3 policy boundary:** integrate deterministic decision/ranking,
   generality and question-value policy, then LLM shortlist reranking. Verify
   DG-01, no repeated low-value questions, final-turn behavior, constraints,
   deduplication, and intent-version eligibility.
4. **Person 1 live route:** enable `gpt-5.6-terra` through the main API with
   `xhigh` reasoning behind configuration. Verify environment-only credentials,
   structured-output validation, usage telemetry, and deterministic fallback.

### Handoff checklist for every owner

- protocol version implemented;
- deterministic fake retained;
- normal, boundary, malformed, and timeout tests supplied;
- configuration schema and defaults documented;
- no direct import of another provider implementation;
- limitations and fallback behavior stated;
- evidence required by reporting exposed in a structured form.

### Person 4 integration tests

- full multi-turn traces for all four scenario classes;
- Buying-to-Browsing and Browsing-to-Buying transitions;
- explicit intent override and `intent_version` increment;
- Boundary no-preference response without a repeated question;
- hard-constraint behavior when catalog metadata is missing;
- hallucinated or duplicate reranker IDs cannot escape validation;
- API timeout and malformed JSON produce a valid deterministic response;
- query embedding cannot load an index for a different embedding model;
- concurrent/interleaved sessions never share state;
- official evaluator can instantiate the thin `starter.Agent` directly.

### Exit gate

- The primary and deterministic paths both execute end to end.
- All required tests in `AGENTS.md` have an owning test module and pass.
- Full-catalog retrieval returns only catalog-valid IDs.
- No provider SDK leaks into orchestration, retrieval contracts, or evaluation.

## 10. Phase P5 — Ablations and configuration selection

### Outcome

Select the release configuration from held-out evidence, prioritizing coverage,
then exact rank, then useful-turn efficiency.

### Required experiment matrix

- BM25 alone versus dense alone versus hybrid retrieval;
- deterministic versus LLM shortlist reranking;
- clarification policy versus never-ask and fixed-ask baselines;
- profile weight `0` versus each candidate soft, decaying profile weight;
- automatic runtime routing versus fully pinned routes;
- embedding and fusion configurations, always with matching indexes;
- state tracking and override handling enabled versus controlled alternatives;
- same-intent deduplication and intent-version reset behavior;
- API-primary path versus deterministic degraded path.

### Selection discipline

- Tune thresholds on the tuning split only.
- Use the held-out split for decision gates, not repeated parameter searching.
- Report overall and Buying, Browsing, Intent Override, and Boundary deltas.
- Do not accept an aggregate gain that conceals a material scenario collapse.
- Preserve the exact configuration and per-session output for each candidate
  release.
- Set profile weight to `0` unless a profile-enabled option improves held-out
  results.
- Prefer a simpler configuration when score differences are within observed
  run variance and the simpler route is more reproducible.

### Exit gate

- The selected primary configuration has a complete ablation table.
- The selected deterministic fallback has a measured quality delta.
- Model, token, latency, estimated cost, and failure evidence are complete.
- Every index used in a reported run has a matching manifest.

## 11. Phase P6 — Network-degraded path, packaging, and release

### Outcome

Produce a clean submission that works through the official interface, documents
its API dependency, and remains valid without network access.

### Tasks

- Run a full evaluator smoke test with API credentials absent and network use
  disabled; confirm automatic selection of the deterministic path.
- Confirm local indexes load from documented paths and validate their catalog
  checksum, text version, embedding model, dimensionality, and normalization.
- Freeze Python/dependency versions and verify setup in a clean environment.
- Run `python3 -m unittest` and `python3 -m evaluator.local_evaluator` without
  modifying the official evaluator or dataset.
- Verify the public repository contains no secret, generated result containing
  a secret, private data, large unapproved index, or undeclared service
  dependency.
- Document setup, environment variables, API-primary behavior, offline
  behavior, one-command reproduction, limitations, model choice, latency,
  tokens, estimated cost, and team contributions.
- Prepare one vague-to-specific trace and one Intent Override trace for the
  demo, with structured state and ranking evidence but no hidden evaluator
  information.
- Package the exact entry point and required local helper modules.
- Update remotes only after explicit owner authorization.

### Final release gates

- [ ] Official tests pass unchanged.
- [ ] Official evaluator completes from a clean environment.
- [ ] Catalog checksum and 50,000 unique IDs are verified.
- [ ] Agent inputs contain only official fields.
- [ ] All responses are schema-valid and catalog-valid.
- [ ] Primary `gpt-5.6-terra` route uses `xhigh` reasoning.
- [ ] Credentials are environment-only and absent from source and reports.
- [ ] Deterministic no-network execution produces valid results.
- [ ] Tuning and held-out metrics are reported separately.
- [ ] Aggregate and per-scenario metrics accompany the selected configuration.
- [ ] Model, embedding, reranker, prompt/schema, index, and routing identities
      are recorded.
- [ ] Token, latency, cost, retry, failure, and fallback evidence is disclosed.
- [ ] Submission instructions reproduce the measured run with one command.
- [ ] Limitations and quality loss in degraded mode are explicit.

## 12. Recommended merge sequence

Keep each merge small and leave `main` runnable:

1. contracts, protocols, fixtures, and fakes;
2. session registry, output normalization, and orchestration tests;
3. thin `starter.Agent` delegation;
4. stable split and experiment configuration;
5. reporting and reproducibility metadata;
6. Person 1 state integration;
7. Person 2 sparse/structured retrieval integration;
8. Person 3 deterministic policy/ranking integration;
9. dense/fusion integration;
10. live LLM interpretation and shortlist reranking integration;
11. fallback hardening, ablations, and release documentation.

Each merge must include its tests. Any score-changing merge must additionally
include the exact configuration and aggregate plus per-scenario deltas.

## 13. Person 4 definition of done

Person 4's work is complete only when:

- contracts are documented, versioned, acknowledged, and covered by malformed
  as well as normal-input tests;
- the thin official Agent delegates to a tested orchestrator;
- sessions are isolated and label-leakage guards prove the official boundary;
- every integrated component can be replaced by a deterministic fake;
- the evaluator runs unchanged with primary and no-network configurations;
- experiment reports reproduce the selected score and expose per-scenario
  effects;
- configuration, usage, latency, cost, limitations, and fallback behavior are
  documented; and
- the clean submission bundle satisfies the official interface and release
  checklist.
