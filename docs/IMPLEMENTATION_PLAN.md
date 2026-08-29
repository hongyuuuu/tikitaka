# TikiTaka Four-Person Implementation Plan

## 1. Purpose

This plan turns the recorded Shopping Copilot architecture into four parallel
workstreams with stable interfaces, integration milestones, evaluation gates,
and submission evidence. It is a build map, not permission to change official
data or resolve open owner decisions silently.

The system must transform a vague or evolving conversation into a current
search plan, choose whether more information is worth a turn, and retrieve and
rank the exact hidden purchased product as early and as highly as possible.

## 2. Success measures

The official objective is:

```text
TechnicalScore = 0.50 * HitRate@10 + 0.30 * MRR + 0.20 * Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

Every experiment must report:

- overall Hit Rate@10, MRR, MTTC, Efficiency and TechnicalScore;
- Buying, Browsing, Intent Override and Boundary metrics;
- questions asked per session and attribute distribution;
- model/provider/reasoning configuration;
- embedding and reranker configuration;
- prompt, completion and total tokens;
- latency and estimated cost;
- tuning-split and held-out-split results separately.

The official weak baseline is the first comparison point: Hit Rate@10 `0.125`,
MRR `0.068034`, and MTTC `9.81` on the released 200-session set.

## 3. Known data facts

The released data has been verified locally:

- 50,000 catalog rows and 50,000 unique `parent_asin` values;
- 200 public sessions and 200 unique public targets;
- all 200 public targets are present in the catalog;
- scenario mix: 80 Buying, 80 Browsing, 30 Intent Override, 10 Boundary;
- all 200 sessions contain complete, non-empty anonymized profiles;
- 125 distinct complete profile combinations;
- no stable raw `user_id` is supplied;
- `purchase_frequency` is constant in the public data and is not useful as a
  discriminative ranking feature by itself.

Catalog metadata is incomplete and must be treated accordingly. In the local
scan, only 10,527 of 50,000 products have a non-empty price and 26,113 have a
non-empty description. Missing metadata must not be treated as a failed hard
constraint without corroborating evidence.

## 4. Decision gates

These decisions must be settled by the owner before their dependent behavior is
locked. Implement interfaces that keep them configurable until then.

### DG-01 — Turn action policy — settled

Although the official contract permits asking and recommending simultaneously,
the owner selected mutually exclusive actions. `CLARIFY` returns one structured
attribute and no recommendations. `RECOMMEND` returns ranked products and
`ask_attribute = null`. Keep response composition behind one policy interface
so the rule is enforced consistently.

### DG-02 — Profile influence — settled

The supplied profile may be used only as a soft, decaying session-local signal.
Its numeric weight is chosen through held-out comparison against profile weight
`0`. It never becomes a hard filter and never overrides explicit current
conversation. If the profile-enabled variant does not improve held-out results,
the selected weight is `0`.

### DG-03 — Override clearing scope — settled

Clearing is dependency-aware. A direct correction replaces only its attribute.
An explicit restart clears all conversation-derived constraints. A major
category or product-type change creates a new intent version, removes old
category-derived or incompatible constraints, preserves still-applicable
universal constraints such as budget, and marks ambiguous constraints for
revalidation. The supplied profile remains separate from this state.

### DG-04 — Model-selection behavior — settled

The model-selection segment automatically routes at runtime across configured
API model options, embedding routes, and reranking strategies. Evaluation runs
can pin all routes for reproducibility. Embedding selection is coupled to the
matching precomputed product index; the router must never compare vectors from
different embedding models.

## 5. Target package structure

```text
starter/
  agent.py                         official thin entry point

tikitaka/
  config.py
  contracts/
    domain.py                      Constraint, StateDelta, Candidate, TurnDecision
    model.py                       model and embedding protocols
  models/
    api_llm.py                     primary API adapter
    fake.py                        deterministic unit-test adapter
    selector.py
    usage.py
  state/
    schema.py                      strict LLM output schema
    extractor.py                   message -> StateDelta
    reducer.py                     validated state mutation
    query_builder.py               active state -> retrieval query
  retrieval/
    catalog.py                     read-only catalog access and normalized records
    lexical.py                     BM25/sparse index
    dense.py                       embedding build/load/search
    structured.py                  reliable filters and boosts
    fusion.py                      rank fusion and evidence merging
  decision/
    intent_router.py               Buying/Browsing runtime mode
    generality.py                  uncertainty and overload sensor
    question_value.py              expected ranking-change policy
    response_policy.py             DG-01 action composition
  ranking/
    deterministic.py               reproducible scorer/fallback
    llm.py                         semantic shortlist reranker
    constraints.py                 hard/soft/negative enforcement
    diversity.py                   early coverage and repetition policy
  orchestration/
    shopping_agent.py              end-to-end per-turn control flow
    sessions.py                    isolated SessionState registry
  evaluation/
    splits.py                      stable tuning/held-out split
    experiment.py                  configuration runner
    reporting.py                   aggregate/per-scenario/model evidence
    ablations.py

scripts/
  build_lexical_index.py
  build_embeddings.py
  run_experiment.py

tests/
  fixtures/
  test_contracts.py
  test_state.py
  test_retrieval.py
  test_decision.py
  test_ranking.py
  test_orchestration.py
```

Large generated indexes and secrets must remain uncommitted. Commit index
manifests, schemas and reproducible build commands instead.

## 6. Shared interfaces and integration contract

### 6.1 State interpretation

```python
class IntentInterpreter(Protocol):
    def interpret(
        self,
        message: str,
        state: SessionState,
    ) -> tuple[StateDelta, Usage]: ...
```

The LLM returns structured operations. Required operations include:

- `ADD`: introduce a new active constraint;
- `REMOVE`: retract a known constraint without replacing it;
- `REPLACE`: supersede an active value;
- `EXCLUDE`: express a negative constraint such as “not leather”;
- `NO_PREFERENCE`: mark an attribute irrelevant and suppress repeated asks;
- `RESET`: discard the requested scope and create a new intent version.

The reducer, not the LLM, validates and mutates state.

### 6.2 Retrieval

```python
class Retriever(Protocol):
    def search(
        self,
        plan: SearchPlan,
        limit: int,
    ) -> list[Candidate]: ...
```

Candidates must preserve route evidence, not only a fused score. Missing
metadata is represented explicitly as unknown.

### 6.3 Decision

```python
class DecisionPolicy(Protocol):
    def choose(
        self,
        state: SessionState,
        candidates: list[Candidate],
        turn: int,
    ) -> TurnDecision: ...
```

The decision policy cannot read evaluator labels. It uses current state,
candidate uncertainty, expected ranking change, already-asked attributes and
remaining turn budget.

### 6.4 Reranking

```python
class Reranker(Protocol):
    def rank(
        self,
        state: SessionState,
        candidates: list[Candidate],
        top_k: int,
    ) -> tuple[list[str], Usage]: ...
```

Validate every returned ID against the supplied shortlist and catalog. The
reranker cannot invent or retrieve products.

### 6.5 Orchestration

```text
reset
  -> create isolated SessionState and retain supplied profile snapshot

respond
  -> interpret message
  -> validate and reduce state
  -> build active search plan
  -> retrieve and fuse candidates
  -> measure over-generality and question value
  -> select action through response policy
  -> rerank when recommending
  -> normalize official response
  -> update asked/shown/usage history
```

## 7. Person 1 — Model gateway and conversation state

### Objective

Provide one trusted boundary between natural-language models and deterministic
session state. Support the primary API first without coupling the rest of the
agent to one provider.

### Responsibilities

1. Implement model protocols, usage metadata and provider-neutral errors.
2. Implement the primary `gpt-5.6-terra` API adapter with environment-only
   credentials and `xhigh` reasoning configuration.
3. Implement a fake adapter for deterministic tests before live API work.
4. Define strict structured output for intent mode, confidence, generality and
   state operations.
5. Implement JSON parsing, schema validation, retry/repair boundaries and
   malformed-output handling.
6. Implement `SessionState`, constraint provenance and intent-version history.
7. Implement the state reducer and active query builder.
8. Keep the supplied profile snapshot separate from explicit conversation
   constraints so DG-02 can be tested cleanly.
9. Implement runtime routing inputs and decisions while allowing experiment
   configuration to pin the generative route.

### Required tests

- Accumulate multiple compatible constraints.
- Replace a conflicting color/material/category.
- Retract a budget without creating a negative budget.
- Exclude a value such as leather or red.
- Record no-preference and suppress future asks for that attribute.
- Full reset versus single-attribute replacement.
- Category change clears incompatible/dependent constraints, preserves an
  applicable budget, and flags ambiguous constraints for revalidation.
- Reject unknown attributes and operations.
- Clamp invalid confidence values.
- Recover or fail safely on malformed JSON.
- Prove sessions do not share state.
- Prove explicit dialogue state outranks profile input.

### Deliverables

- Stable model and state protocols.
- Fake and primary API adapters.
- State reducer with unit tests.
- Prompt/schema version identifiers.
- Example structured traces for Buying, Browsing, Override and Boundary.

### Done when

A synthetic multi-turn conversation produces the same validated state through
the fake and API adapters, and no provider code leaks into retrieval or ranking.

## 8. Person 2 — Catalog processing and hybrid retrieval

### Objective

Maximize target coverage by combining lexical precision, semantic meaning and
reliable structured evidence while keeping the entire retrieval path local and
reproducible.

### Responsibilities

1. Load and validate the frozen catalog without mutating it.
2. Define normalized product text from title, categories, features, details,
   description and store.
3. Extract reliable structured values and retain unknown/missing states.
4. Implement a deterministic sparse/BM25 index.
5. Implement embedding preprocessing, batching, manifest versioning and
   in-memory similarity search.
6. Implement the embedding-model protocol, runtime route-selection hooks, and
   benchmark pinning. Couple every route to its matching product index.
7. Implement structured filters/boosts for category, material, color, size,
   brand, budget, features and use cases where evidence permits.
8. Implement Reciprocal Rank Fusion or a configurable calibrated alternative.
9. Preserve per-route evidence for every fused candidate.
10. Expose candidate-pool diagnostics required by Person 3’s generality sensor.

### Required tests

- Load exactly 50,000 unique IDs from the official catalog.
- Deterministic text construction and index manifests.
- Exact lexical match retrieval.
- Semantic paraphrase retrieval on synthetic examples.
- Category and budget constraint behavior.
- Missing-price and missing-description safeguards.
- Stable RRF ordering for fixed route ranks.
- No duplicate or invalid candidate IDs.
- Query embedding and product embedding normalization consistency.
- Index mismatch detection for catalog/model/version changes.

### Deliverables

- Catalog normalization pipeline.
- Sparse and dense index builders/loaders.
- Structured evidence extractor.
- Hybrid retriever with route diagnostics.
- Reproducible index manifest and build instructions.

### Done when

The hybrid retriever runs locally over the full catalog, returns only valid
evidence-bearing candidates, and can be compared with BM25 alone on held-out
retrieval coverage.

## 9. Person 3 — Decision policy and reranking

### Objective

Ask only when missing information is likely to change the ranking, then push the
exact target as high as possible once the state is sufficiently specific.

### Responsibilities

1. Implement the runtime Buying/Browsing mode policy using visible state only.
2. Implement the over-generality sensor using constraint coverage, score
   concentration, route disagreement, rank margin and metadata availability.
3. Estimate question value from expected Top-10 membership/order changes.
4. Exclude answered, no-preference and already-asked attributes.
5. Implement candidate-aware but non-misleading clarification phrasing.
6. Implement `response_policy.py` so DG-01 is isolated in one place.
7. Implement deterministic reranking and hard/soft/negative constraint logic.
8. Implement LLM shortlist reranking through Person 1’s model protocol.
9. Implement early-turn diversity and same-intent repetition handling.
10. Reset shown-product eligibility on a confirmed new intent version.

### Required tests

- Buying and Browsing route changes as constraints accumulate.
- High-generality request chooses a valuable unasked attribute.
- Low-information question is suppressed.
- No-preference attribute is not asked again.
- Turn budget prevents an unanswered final clarification.
- Hard contradictions cannot win through semantic similarity alone.
- Missing metadata does not become an automatic contradiction.
- LLM reranker output is shortlist-valid and duplicate-free.
- Same-intent repeats are penalized or excluded.
- Prior products become eligible after a new intent version.
- DG-01 behavior changes through configuration, not duplicated branching.

### Deliverables

- Generality diagnostics and question-value policy.
- Deterministic and LLM rerankers.
- Constraint enforcement and diversity selection.
- Explainable turn-decision records for demo and error analysis.

### Done when

Question policy beats fixed asking/never-asking baselines on held-out MTTC or
Hit Rate/MRR, and LLM reranking improves MRR over deterministic ordering without
violating hard constraints.

## 10. Person 4 — Contracts, orchestration, evaluation and release

### Objective

Keep all workstreams integrable, preserve evaluator integrity, measure every
change honestly, and produce the final reproducible submission.

### Responsibilities

1. Land shared contracts and tiny fake components first.
2. Keep `starter/agent.py` as a thin official-interface adapter.
3. Implement isolated session registry and end-to-end orchestration.
4. Validate and normalize official response payloads.
5. Create a stable tuning/held-out session split without exposing labels to the
   Agent.
6. Build experiment configuration, caching boundaries and result reporting.
7. Record aggregate, per-scenario, per-model, token, latency and cost evidence.
8. Add label-leakage guards and evaluator-integrity checks.
9. Coordinate integration fixtures and contract-version changes.
10. Record automatic model-routing decisions and support pinned experiment
    configurations.
11. Package API-primary operation, deterministic no-network contingency,
    dependencies, README, demo trace and final ablations.
12. Maintain the writable fork as `origin` and organizer repository as
    `upstream` once the owner authorizes the remote update.

### Required tests

- Official reset/respond lifecycle.
- Output schema and allowed `ask_attribute` values.
- First-10 unique valid recommendation behavior.
- Exceptions and malformed component outputs produce valid safe responses.
- No `ground_truth` or `scenario_type` reaches participant components.
- Sessions terminate correctly by hit or turn budget.
- Usage aggregation remains non-negative and attributable.
- Same configuration produces reproducible deterministic results.
- Network-free smoke test before final submission.
- Unchanged official tests and evaluator continue to run.

### Deliverables

- Shared contracts and fake end-to-end scaffold.
- Thin official Agent entry point.
- Experiment and reporting harness.
- Integration test suite.
- Reproducible submission instructions and demo evidence.

### Done when

The full agent can run from the official evaluator with one command, reports the
configuration needed to reproduce its score, and remains valid when API access
is unavailable.

## 11. Parallel build sequence

### Milestone M0 — Contract freeze

**All four people, first working block**

- Agree shared data classes and protocols.
- Create tiny catalog and conversation fixtures.
- Confirm all settled decision gates in shared contracts and tests.
- Confirm branch/file ownership.
- Reproduce the untouched baseline.

Exit evidence: official tests pass; fake orchestrator returns a valid response.

### Milestone M1 — Deterministic vertical slice

Work in parallel:

- Person 1: fake interpreter, schema and state reducer.
- Person 2: lexical plus structured retrieval.
- Person 3: deterministic decision and reranker.
- Person 4: orchestrator, split and reporting harness.

Integrate:

```text
message -> fake/heuristic state -> BM25/structured retrieval
        -> deterministic decision/rank -> official response
```

Exit evidence: end-to-end deterministic evaluation and per-scenario report.

### Milestone M2 — Dense hybrid retrieval

- Person 2 builds and validates embeddings and rank fusion.
- Person 1 exposes embedding/model configuration contracts.
- Person 3 consumes retrieval diagnostics in generality scoring.
- Person 4 runs BM25 versus dense versus hybrid ablations.

Exit evidence: hybrid held-out retrieval coverage compared with BM25 baseline.

### Milestone M3 — Primary LLM vertical slice

- Person 1 integrates API intent/state extraction.
- Person 3 integrates LLM shortlist reranking and clarification generation.
- Person 2 supplies compact candidate evidence and query embeddings.
- Person 4 captures tokens, cost, latency and failure traces.

Exit evidence: live API run completes representative scenarios with valid
structured outputs and no leaked secrets.

### Milestone M4 — Policy and score tuning

- Tune generality/question-value thresholds.
- Compare profile weight `0` versus soft prior.
- Compare deterministic and LLM reranking.
- Compare embedding and fusion configurations.
- Test override-clearing alternatives.
- Compare automatic runtime routing with pinned-route evaluation runs.
- Report all changes on held-out and per-scenario metrics.

Exit evidence: selected configuration has documented ablations and no hidden
scenario collapse.

### Milestone M5 — Network-degraded deterministic path

- Person 1 validates deterministic handling when the API is unavailable.
- Person 2 confirms local embedding availability and index reproduction.
- Person 3 confirms deterministic ranking and decision fallbacks.
- Person 4 runs the full evaluator without network access.

Exit evidence: valid deterministic degraded results and documented quality
delta from the API configuration.

### Milestone M6 — Submission and demo

- Freeze dependencies and model/index manifests.
- Run official tests and evaluator from a clean environment.
- Produce final aggregate, scenario and model-selection tables.
- Record one vague-to-specific flow and one Intent Override flow.
- Document limitations, network behavior, cost and team contributions.

Exit evidence: public repository, reproducible command, demo video material and
final Devpost evidence are ready.

## 12. Suggested three-day allocation

### Day 1 — Contracts and deterministic baseline

- Morning: M0 contract freeze and fake scaffold.
- Afternoon: four parallel M1 workstreams.
- Evening: deterministic vertical slice and baseline report.

Do not end Day 1 with four disconnected modules. The orchestrator must run.

### Day 2 — Dense and LLM quality

- Morning: M2 dense/hybrid integration.
- Afternoon: M3 API state extraction and reranking.
- Evening: first M4 ablations and scenario failure analysis.

Prioritize complete measured flows over additional model providers.

### Day 3 — Tuning, deterministic fallback and submission

- Morning: finish M4 thresholds, profile and override ablations.
- Midday: M5 deterministic no-network verification.
- Afternoon: M6 clean run, documentation and demo capture.
- Final block: freeze changes except score-critical defects.

## 13. Integration cadence

- Hold a 10-minute contract sync at the start and midpoint of each day.
- Merge the smallest working vertical slices; avoid long-lived mega-branches.
- Person 4 publishes the current contract version and integration status.
- Every owner supplies one fake implementation so integration never waits for a
  live API or full index.
- Any shared-contract change includes migration notes and updated fixtures.
- Any scoring change includes the exact configuration and per-scenario delta.
- Any owner decision updates `ARCHITECTURE.md` before dependent code is locked.

## 14. Final acceptance checklist

- [ ] Official tests pass unchanged.
- [ ] Catalog checksum and row count verified.
- [ ] No official data or evaluator modifications.
- [ ] No label or scenario leakage into Agent inputs.
- [ ] Exactly contract-valid responses for all test turns.
- [ ] Intent accumulation, override, exclusion, no-preference and reset tested.
- [ ] Buying/Browsing routing uses visible state only.
- [ ] Sparse, dense and structured retrieval ablated.
- [ ] Generality/question policy compared with simple baselines.
- [ ] Deterministic and LLM reranking compared.
- [ ] Profile weight `0` compared with any profile-enabled configuration.
- [ ] Aggregate and per-scenario metrics reported.
- [ ] Model, embedding and reranker selection evidence reported.
- [ ] Tokens, cost and latency disclosed.
- [ ] API secrets absent from source and reports.
- [ ] Deterministic no-network path produces valid outputs.
- [ ] One-command reproduction documented and verified.
- [ ] Demo shows vague intent convergence and Intent Override recovery.
