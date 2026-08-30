# Person 3 Build Plan — Decision Policy and Reranking

## 1. Purpose and ownership

Person 3 owns `tikitaka/decision/`, `tikitaka/ranking/`, and their matching
tests. The objective is to spend a clarification turn only when the expected
ranking improvement justifies losing a recommendation opportunity, then rank a
validated shortlist for Hit Rate@10 first and MRR second.

This plan follows the repository authority order. The official contracts and
competition constraints remain authoritative, followed by `ARCHITECTURE.md`
and `docs/IMPLEMENTATION_PLAN.md`. Person 3 does not modify shared contracts,
retrieval output, orchestration, evaluator code, or public labels without the
appropriate owner review.

## 2. Corrections to the reference architecture

The attached reference is useful as a conceptual sketch, but the implementation
must not copy these details:

- Do not choose `CLARIFY` from candidate count alone. A large, concentrated
  candidate set may already have a stable Top 10, while a small ambiguous set
  may still benefit from one decisive attribute.
- Do not equate high facet entropy with high question value. Entropy measures
  diversity, not the expected change in Top-10 membership or order.
- Only turn 10 is forced to `RECOMMEND`. A turn-9 clarification can still be
  answered and used on turn 10.
- Do not hard-filter a candidate merely because metadata is absent. Unknown is
  distinct from contradiction.
- Do not merge profile signals into conversation-derived constraints. The
  profile stays session-local, soft, decaying, and subordinate to explicit
  statements.
- Do not add alternative generative models or a local generative model. The
  selected generative route is `gpt-5.6-terra` through the main API with
  `medium` reasoning; deterministic ranking is the network-free fallback.
- Do not infer or consume evaluator scenario labels. Buying/Browsing behavior
  uses only the validated visible-session state.

## 3. Contract dependencies to settle before implementation

Person 3 should acknowledge shared contract version `0.1.0` only after the
following inputs are available through dependency-light records or views.

### 3.1 State view consumed by Person 3

The decision and ranking code needs a read-only view containing:

- inferred mode and confidence;
- active constraints, exclusions, confidence, strength, and source turn;
- attributes marked no-preference or awaiting revalidation;
- already-asked attributes;
- current intent version and products shown in that version;
- session-local profile soft signals kept separate from explicit constraints;
- current turn and remaining turn budget.

Person 1 owns mutable state. Person 4 owns the orchestration-facing view.
Person 3 must not import a mutable state implementation into shared contracts.

### 3.2 Candidate evidence consumed by Person 3

Each candidate must expose enough deterministic evidence to support ranking and
question simulation:

- catalog-valid `parent_asin`;
- sparse, dense, structural, and fused rank/score evidence when present;
- normalized known attribute values for the allowed clarification attributes;
- explicit matched, contradicted, and unknown constraint evidence;
- reliability or provenance for evidence used as a hard contradiction;
- compact product text/evidence suitable for a bounded LLM reranking prompt.

Person 2 owns extraction and retrieval evidence. Missing values must be encoded
as unknown, not as an empty value that Person 3 might interpret as a mismatch.

### 3.3 Decision diagnostics

`TurnDecision.expected_information_gain` should be a deterministic normalized
score in `[0.0, 1.0]`. `reason` should include a stable machine-readable code
and may include human-readable detail. Initial reason codes:

- `FINAL_TURN`;
- `VALUABLE_CLARIFICATION`;
- `LOW_QUESTION_VALUE`;
- `NO_ELIGIBLE_ATTRIBUTE`;
- `RANKING_STABLE`;
- `INSUFFICIENT_EVIDENCE`;
- `COMPONENT_FALLBACK`.

Any addition to shared records requires Person 4 to update the contract and
affected owners to acknowledge it. Until then, richer pool-level diagnostics
remain private types inside `tikitaka/decision/`.

## 4. Target modules

```text
tikitaka/
  decision/
    diagnostics.py       candidate-pool uncertainty and route disagreement
    generality.py        deterministic over-generality score
    question_value.py    expected Top-10 change by eligible attribute
    response_policy.py   single CLARIFY-or-RECOMMEND decision boundary
    phrasing.py           candidate-grounded deterministic question fallback
  ranking/
    constraints.py       match/unknown/contradiction enforcement
    deterministic.py     reproducible shortlist scorer and stable tie-break
    llm.py               bounded semantic reranking through model protocol
    diversity.py         early coverage and same-intent repetition policy

tests/
  test_decision.py
  test_ranking.py
  fixtures/
    decision_catalog.jsonl
```

Do not create provider adapters in these packages. The LLM reranker consumes
Person 1's provider-neutral model protocol.

## 5. Phased implementation

### P3.0 — Contract review and executable fakes

1. Record answers to the Person 3 contract questions.
2. Confirm shortlist-only reranking, DG-01 mutual exclusion, and label-free
   runtime behavior.
3. Agree with Person 2 on normalized facet values, evidence reliability,
   unknown representation, and stable retrieval tie-breaking.
4. Agree with Person 4 on decision reason codes and the `[0, 1]` information-
   gain scale.
5. Build tiny in-memory state and candidate fixtures against the shared
   interfaces before production algorithms.

Exit gate: contract tests and fakes can express match, contradiction, unknown,
route disagreement, asked/no-preference state, and intent-version changes.

### P3.1 — Constraint safety and deterministic reranking

Implement ranking in two stages:

1. **Eligibility and safety:** reject only catalog-invalid IDs and confirmed
   contradictions to explicit hard constraints with reliable evidence. Retain
   unknown metadata with an uncertainty penalty or neutral treatment.
2. **Ordering:** combine fused retrieval evidence, explicit constraint matches,
   soft constraints, route agreement, profile soft signals, and repetition
   penalties. Use a documented stable tie-break ending in `parent_asin`.

The scorer must be configuration-driven and deterministic for fixed inputs.
Profile weight defaults to zero until a held-out ablation supports a non-zero
value. Products already shown in the same intent version are excluded or
strongly penalized; a new intent version makes them eligible again.

Exit gate: the scorer returns only supplied shortlist IDs, enforces confirmed
hard contradictions, preserves unknowns, is duplicate-free, and is stable
across repeated runs.

### P3.2 — Candidate-pool diagnostics and generality

Compute diagnostics over a competitive prefix rather than the full arbitrary
retrieval limit:

- active-constraint coverage and confidence;
- score concentration and effective candidate mass;
- lead margin and Top-10 boundary margin;
- overlap/disagreement among sparse, dense, and structural routes;
- known-value coverage and distribution for each eligible attribute;
- current Top-10 stability under small deterministic score perturbations;
- metadata sufficiency for safe question simulation.

Normalize each diagnostic before combining it into a generality score. Keep
the weights configurable. A missing diagnostic lowers confidence; it must not
automatically force clarification.

Exit gate: synthetic concentrated, diffuse, route-conflicted, and metadata-
sparse pools produce explainable stable diagnostics.

### P3.3 — Expected ranking-change question policy

For every allowed attribute, first remove attributes that are:

- already answered with sufficient confidence;
- explicitly no-preference;
- already asked in the current intent version without new ambiguity;
- unsupported by enough known candidate metadata;
- unlikely to be answerable from the candidate evidence.

For each remaining attribute, simulate plausible answers from the known value
distribution among competitive candidates. For every answer branch:

1. apply the branch as a temporary constraint without mutating session state;
2. rerun the deterministic scorer;
3. measure normalized change in Top-10 membership and rank-weighted order;
4. weight the change by branch probability, metadata coverage, constraint
   confidence, and remaining turn budget.

The resulting expected change is the normalized information-gain score. Ask
only when the best score exceeds a configured threshold and the generality
sensor indicates meaningful uncertainty. Otherwise recommend. Turn 10 always
recommends; turn 9 may clarify when its expected benefit exceeds the threshold.

Exit gate: the policy suppresses high-entropy but rank-irrelevant attributes,
selects low-cardinality attributes that materially change the Top 10, never
repeats a no-preference question, and never clarifies on turn 10.

### P3.4 — Response policy and clarification phrasing

Put DG-01 in one `ResponsePolicy` implementation:

- `CLARIFY`: exactly one allowed `ask_attribute`, no recommendations;
- `RECOMMEND`: `ask_attribute = None`, ranked shortlist IDs;
- component failure: deterministic `RECOMMEND` fallback with a stable reason.

Question phrasing must remain consistent with the structured attribute. It may
mention one or two candidate-supported examples, but it must not ask several
attributes or promise unavailable products. The deterministic template is the
network-free path; optional LLM phrasing uses the approved model gateway and
must not alter the chosen attribute.

Exit gate: every decision maps to a schema-valid mutually exclusive action and
the prose cannot disagree with `ask_attribute`.

### P3.5 — Bounded LLM shortlist reranking

Run semantic reranking only after deterministic eligibility enforcement. Pass
a bounded shortlist with IDs, compact visible product evidence, active state,
and explicit ranking instructions. Never pass the full catalog, hidden labels,
or evaluator internals.

Validate structured output as untrusted input:

- discard unknown and out-of-shortlist IDs;
- keep the first occurrence of duplicates;
- preserve deterministic hard-constraint eligibility;
- fill omitted positions from deterministic order;
- fall back completely on timeout, malformed output, or empty valid output;
- record provider, model, reasoning level, prompt/schema version, tokens,
  latency, estimated cost, and route.

The only generative route is `gpt-5.6-terra` through the main API at `medium`.

Exit gate: malformed, duplicate, hallucinated, partial, and timed-out outputs
all produce valid deterministic rankings; valid LLM output can only reorder the
validated shortlist.

### P3.6 — Diversity, repetition, and integration

Apply diversity only as a controlled coverage strategy after relevance and
hard-constraint safety. Compare diversification at early vague turns with no
diversification. Do not force category diversity after the user specifies a
category. Ensure products shown in an old intent version can return after a
confirmed override.

Integrate through Person 4's orchestrator in this order:

1. deterministic reranker;
2. deterministic diagnostics and response policy;
3. question-value simulation and phrasing;
4. repetition/diversity handling;
5. LLM shortlist reranker;
6. network-free fallback verification.

Exit gate: the official lifecycle stays valid across Buying/Browsing mode
changes, override, boundary/no-preference, malformed model output, and turn 10.

## 6. Test matrix

### Decision tests

- A vague pool with a rank-changing material split chooses `material`.
- High color entropy with unchanged Top-10 order suppresses `color`.
- An answered, excluded, no-preference, or already-asked attribute is not
  selected again.
- Missing facet metadata lowers question confidence without becoming a
  contradiction.
- Route disagreement and weak margins raise generality deterministically.
- Buying/Browsing mode changes behavior using visible state only.
- Turn 9 may clarify; turn 10 cannot clarify.
- Every `TurnDecision` has a normalized score and stable reason code.
- DG-01 produces no mixed clarify/recommend response.

### Ranking tests

- A confirmed hard contradiction cannot win on semantic similarity.
- Unknown size, price, or material is not treated as contradiction.
- Soft preference and profile evidence cannot override explicit dialogue.
- Deterministic scoring and tie-breaking are reproducible.
- Only supplied, catalog-valid, unique IDs are returned.
- Same-intent shown products are penalized or excluded.
- Shown products become eligible after an intent-version change.
- LLM duplicates, hallucinations, omissions, malformed JSON, and timeout all
  normalize to a valid deterministic result.
- Top-k truncation preserves input ranking and never exceeds the requested
  boundary.

### Integration and leakage tests

- Person 3 modules cannot access `ground_truth`, `scenario_type`, public labels,
  or evaluator internals.
- No provider SDK is imported by deterministic decision or ranking modules.
- No network is required for unit tests or deterministic degraded execution.
- Usage is recorded only for actual model calls.

## 7. Experiment ladder and decision gates

Use Person 4's stable tuning/held-out split. Never tune on all 200 public
targets. Save every configuration and report aggregate plus per-scenario
results.

Run controlled additions in this order:

1. fused retrieval order, never ask;
2. deterministic constraint-aware reranking;
3. generality gate with the `fixed-ask-baseline`: choose the first eligible
   attribute in the immutable contract order, without selecting the attribute
   that has the highest estimated information gain;
4. expected ranking-change attribute selection;
5. repetition and early-turn diversity;
6. LLM shortlist reranking;
7. selected combined configuration;
8. deterministic network-free configuration.

### Phase 5 fixed-ask control

The fixed-ask arm is a control for the value of adaptive question selection,
not permission to ask invalid or repeated questions. It must:

- reuse the same generality, turn-budget and clarify-versus-recommend utility
  gates as the adaptive deterministic arm;
- choose the first eligible attribute in `ALLOWED_ATTRIBUTES` order rather
  than the attribute with the largest expected ranking change;
- permanently skip no-preference and exhausted attributes, and skip answered
  or already-asked attributes unless an explicit revalidation flag applies;
- recommend when no fixed-order attribute is eligible and always recommend on
  turn 10;
- use deterministic reranking, profile weight `0` and no LLM calls so question
  selection is the only changed variable; and
- have its own stable question-policy ID, arm fingerprint, normal/boundary/
  malformed tests and tuning plus held-out report.

Person 3 owns the decision configuration, arm and unit tests. Person 4 owns
running and reporting the arm through the existing P5 experiment harness. No
evaluator, orchestration or shared-contract branch should be added for this
control.

After the no-information-state correction, tune only the clarification
threshold on the tuning split using the pre-registered grid `0.05`, `0.06`,
`0.07`, `0.08`, and `0.09`. Keep the official-proxy question-value weights,
clarification cost and late-turn cost fixed. Each threshold must expose:

- an adaptive deterministic arm;
- a contract-order fixed-ask arm; and
- an otherwise identical anchored-LLM reranking arm.

Use the threshold-matched arms so the held-out comparison changes exactly one
axis at a time. Profile weight remains `0`; a non-zero override is not a
release finalist unless it first wins on tuning and then improves held-out
evidence. Freeze the final policy by Hit Rate@10, then MRR, then MTTC, subject
to the per-scenario collapse safeguard.

Track:

- candidate Recall@N before ranking;
- conditional MRR when the target is in the shortlist;
- Hit Rate@10, MRR, MTTC, Efficiency, and TechnicalScore;
- questions per session, asked-attribute distribution, repeated-question rate,
  and clarification-to-hit conversion;
- hard-constraint violation and invalid-ID rates;
- tokens, latency, cost, model identity, prompt/schema version, and route;
- Buying, Browsing, Intent Override, and Boundary results separately.

Promotion gates:

- A reranker cannot compensate for inadequate shortlist recall; retrieval recall
  is reported before interpreting ranking results.
- A decision-policy variant must beat both `never_ask` and a fixed-question
  baseline on held-out TechnicalScore, with no unexplained scenario collapse.
- LLM reranking is selected only if held-out MRR improves over deterministic
  ordering without reducing catalog validity or violating hard constraints.
- Diversity is selected only if held-out Hit Rate@10 improves enough to offset
  any MRR loss under the official weighted score.
- A non-zero profile weight is selected only after held-out improvement over
  profile weight zero.
- No aggregate gain is accepted without configuration, seed, per-scenario
  metrics, and failure examples.

## 8. Coordination points

### Person 1

- Provide validated mode/state views and the provider-neutral LLM protocol.
- Keep LLM output untrusted and usage attributable per call.
- Notify Person 3 when state confidence, constraint semantics, or prompt schema
  changes.

### Person 2

- Provide stable shortlist ordering, normalized facet values, route evidence,
  evidence reliability, and explicit unknowns.
- Expose enough evidence for temporary question branches without letting Person
  3 reach into catalog internals.
- Notify Person 3 when text construction, fusion, or index identity changes.

### Person 4

- Land and version shared contracts, fakes, normalization, experiment pins,
  and reporting fields.
- Enforce catalog validity and the first 10 unique recommendations again at the
  official boundary.
- Integrate Person 3 modules without duplicating DG-01 policy logic.

## 9. Definition of done

Person 3's workstream is done only when:

- contracts and reason/score semantics are documented and acknowledged;
- normal, boundary, malformed, and network-free tests pass;
- every recommendation is a unique ID from the validated supplied shortlist;
- every clarification is allowed, non-repeated, and impossible on turn 10;
- unknown metadata is never silently converted into contradiction;
- deterministic fallback is reproducible and requires no network;
- held-out aggregate and per-scenario effects are reported for every selected
  policy and reranker;
- limitations, thresholds, prompt/schema versions, and fallback behavior are
  recorded; and
- the full agent integrates through the orchestrator without touching official
  evaluator behavior or exposing hidden labels.
