# Person 1 Phased Build Plan

## 1. Role and outcome

Person 1 owns the trusted boundary between untrusted natural-language model
output and deterministic session state:

- provider-neutral model, embedding, and usage protocols;
- the primary `gpt-5.6-terra` API adapter at `xhigh` reasoning;
- a deterministic fake adapter that unblocks every other workstream;
- the strict structured-output schema for intent, mode, generality, and state
  operations;
- `SessionState`, constraint provenance, and intent-version history;
- the validating state reducer and the active query builder;
- profile isolation so DG-02 can be ablated cleanly.

Owned paths: `tikitaka/models/`, `tikitaka/state/`, `tests/test_state.py`,
`tests/test_models.py`, and Person 1 fixtures under `tests/fixtures/`.

Person 1 does not implement retrieval, scoring, question policy, or response
composition. The reducer produces state; it does not decide turns. If an
implementation detail conflicts with an official contract or `ARCHITECTURE.md`,
stop, record the conflict in `docs/IMPLEMENTATION_PLAN.md`, and ask the owner
before changing a settled rule.

## 2. Current starting point

As of this plan:

- `tikitaka/` does not exist; `starter/agent.py` is a self-contained stateless
  BM25 baseline with no state, no LLM, and no session memory beyond a set of
  session IDs;
- `docs/agent_api_contract.json` fixes the external contract: `turn` is 1-10,
  `top_k` is exactly 10, and `ask_attribute` is one of `category`, `material`,
  `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`,
  or `null`;
- `data/public_set.jsonl` is present (200 sessions); `data/catalog.jsonl` is a
  gitignored download and is not required by any Person 1 unit test;
- `python -m unittest` passes the 3 official evaluator tests; they build a
  synthetic catalog in a temporary directory, so they pass without the
  gitignored `data/catalog.jsonl` download. The local interpreter is Python
  3.13.3 while the target floor is Python 3.10;
- DG-01 through DG-04 are settled and must be implemented as configuration,
  not as hardcoded branching;
- Person 4's shared contracts in `tikitaka/contracts/` do not exist yet, so
  Phase P0 must not assume them.

### 2.1 Simulator facts that constrain extraction

The local evaluator generates customer turns from templates. Person 1's
extractor and fixtures must be built against these exact shapes, because they
are the only messages the agent will ever see on the public set:

| Situation | Message |
|---|---|
| Buying, turn 1 | `I'm looking for {category}. A key requirement is: {constraint}.` |
| Browsing, turn 1 | `I'm looking for {category}, but I'm still exploring.` |
| Intent Override, turn 1 | `I'm looking for {category}. {old_value}` |
| Answer to a clarify | `For that, what matters is: {v1}; {v2}.` |
| Nothing left for that attribute | `I don't have an additional preference for {attribute}.` |
| Boundary | `I don't have a preference for {attribute}; please use your judgment.` |
| Recommend turn that missed | `Those options are not quite right yet. Ask me about one specific attribute.` |
| Override turn (3 or 4) | the override message from the hidden behavior record |

The simulator discloses a constraint only when `ask_attribute` matches the
bucket its own classifier assigns to that constraint string. The classifier is
a fixed keyword cascade, checked in this order: `budget` (the word "budget", or
`$`/`<=`/"under" before a digit), `material` (cotton, polyester, nylon,
leather, wool, spandex, silk, rayon, fabric), `color` (the word "color" or
black, white, blue, red, pink, green), `size` (size, sizing, width, wide,
narrow), `style` (department, style, fit, sleeve, neck), `use_case` (hiking,
running, gym, winter, outdoor, work), and `feature` as the fallback for
everything else.

Four consequences Person 1 must record and hand to Persons 3 and 4 rather than
solve alone:

1. The classifier never emits `category` or `brand`. Asking either returns the
   "no additional preference" reply and buys nothing. This is a question-policy
   fact owned by Person 3; Person 1's contribution is to make the possible
   replies unambiguously distinguishable in the `StateDelta`.
2. `ask_attribute = "other"` bypasses classification entirely: the simulator
   matches any undisclosed constraint and returns up to two of them. On the
   public simulator it is a strictly dominant ask. Person 1 records the fact;
   Person 3 decides whether to use it, weighed against risk 5 below.
3. `feature` is the fallback bucket, so it absorbs every constraint no earlier
   rule claims. After `other`, it is the highest-yield single ask.
4. A reply of `I don't have an additional preference for X` is exhaustion of
   that attribute, not a `NO_PREFERENCE` boundary answer. The boundary reply is
   the `please use your judgment` form. Conflating them will either suppress a
   useful re-ask or waste the Boundary scenario. Emit distinct operations.

## 3. Working rules for Person 1

1. LLM output is untrusted input. Parse through a strict schema, reject unknown
   operations and attributes, clamp confidence, and never let a model mutate
   state directly.
2. The reducer owns mutation. The extractor proposes; the reducer validates,
   applies, and records provenance.
3. No provider SDK import outside `tikitaka/models/`. Retrieval, decision,
   ranking, and evaluation code must never learn the provider's name.
4. Credentials come from environment variables only. Never in source, logs,
   traces, prompts echoed into reports, or committed configuration.
5. Every unit test runs offline, in under a second, with no catalog and no
   network. Live-API runs are explicit integration jobs.
6. Keep Python 3.10 compatibility: no reliance on 3.11+ features, no
   `typing.Self`, no PEP 695 generics.
7. API failure is a runtime route change to the deterministic path. It is not
   permission to add a local generative LLM.
8. The supplied profile is a separate snapshot field. It never enters
   `active_constraints`, never becomes a hard filter, and its weight must be
   settable to `0` from configuration.
9. Never read `scenario_type`, `ground_truth`, `category_bucket`, or
   `difficulty_bucket`. Mode is inferred from visible conversation only.
10. Every prompt and schema change bumps `PROMPT_VERSION` / `SCHEMA_VERSION`
    and is reported with any score claim that used it.

## 4. Phase summary

| Phase | Person 1 outcome | External dependency | Exit gate |
|---|---|---|---|
| P0 | Grounding, schema and prompt proposal | Person 4 contract review | Owners acknowledge the state and model interfaces |
| P1 | Fake adapter, strict schema, parse/repair boundary | Person 4's `contracts/` landed or shimmed | Schema and malformed-output tests pass |
| P2 | `SessionState`, provenance, validating reducer | None | Full state test matrix passes offline |
| P3 | Active query builder and profile isolation | `SearchPlan` agreed with Persons 2 and 4 | Person 2 retrieves from a built plan, not raw text |
| P4 | Live `gpt-5.6-terra` adapter with usage accounting | Owner confirms endpoint and credentials | Live and fake adapters produce identical validated state |
| P5 | Malformed, timeout, and degraded-path hardening | Person 4's orchestrator | Zero uncaught exceptions across the fault matrix |
| P6 | Routing inputs, pinning, ablation support, freeze | Person 4's experiment harness | Routes pin reproducibly; profile weight ablatable |

Phases P0-P2 have no external blockers. Start them immediately and do not wait
on Person 4's contract merge.

## 5. Phase P0 — Grounding, schema and prompt proposal

### Outcome

Publish the smallest interface proposal that lets Persons 2, 3, and 4 write
against Person 1 before any model code exists.

### Tasks

- Record the interpreter's inputs and outputs and circulate them at the M0
  contract sync: `IntentInterpreter`, `StateDelta`, `Constraint`,
  `StateOperation`, `SearchPlan`, `Usage`, `ModelRoute`.
- Confirm with the owner, before P4: the exact API base URL, SDK or raw HTTP
  transport, the environment variable name for the credential, how `xhigh`
  reasoning is expressed in the request, and whether structured output is
  requested via a JSON schema parameter or enforced by prompt alone. None of
  this is verifiable from the repository and it must not be guessed in code.
- Agree with Person 2 and Person 4 on `SearchPlan` field names before writing
  `query_builder.py`; it is the only contract Person 1 hands downstream.
- Agree with Person 3 that `StateDelta.generality` is an input signal to the
  generality sensor, not the sensor itself. Person 1 reports what the model
  said; Person 3 decides what it means.
- Build the conversation fixture set directly from the simulator templates in
  section 2.1, one fixture file per scenario: Buying, Browsing, Intent
  Override, Boundary.
- Write down the attribute vocabulary as a single enum shared with the official
  `ask_attribute` list, plus internal-only attributes if any are needed. Any
  internal attribute must map to `other` at the boundary.

### Artifacts

- `docs/person1_contract_notes.md` with the schema proposal, the prompt plan,
  and the open owner questions above;
- four scenario fixture files under `tests/fixtures/conversations/`;
- acknowledgement from Persons 2, 3, and 4 on the fields they consume.

### Exit gate

- Official tests still pass unchanged.
- No provider-specific assumption is embedded anywhere yet.
- The open API questions are asked, with a recorded owner answer or an explicit
  "blocked" note.

## 6. Phase P1 — Fake adapter, strict schema, parse boundary

### Outcome

A deterministic interpreter and a hardened parser exist before any live call,
so the other three workstreams can integrate on day one.

### Owned implementation

```text
tikitaka/
  models/
    __init__.py
    base.py          protocols, ModelRoute, provider-neutral errors
    fake.py          deterministic scripted + heuristic interpreter
    usage.py         Usage accumulation, cost estimation, redaction
  state/
    __init__.py
    schema.py        strict structured-output schema + validator
```

`base.py` defines the provider-neutral surface:

- `IntentInterpreter.interpret(message, state) -> tuple[StateDelta, Usage]`;
- `TextModel.complete_structured(prompt, schema, route) -> tuple[dict, Usage]`;
- `EmbeddingModel.embed(texts, route) -> tuple[list[list[float]], Usage]`,
  declared here for Person 2 to satisfy; Person 1 does not build indexes;
- an error taxonomy: `ModelUnavailable`, `ModelTimeout`, `ModelRefused`,
  `MalformedModelOutput`, `SchemaViolation`, `CredentialMissing`. Every one
  carries the route identity and never the credential.

`schema.py` defines the only accepted model payload:

```json
{
  "inferred_mode": "buying | browsing | unknown",
  "mode_confidence": 0.0,
  "generality": 0.0,
  "query_summary": "",
  "operations": [
    {
      "operation": "add | remove | replace | exclude | no_preference | reset",
      "attribute": "category | material | color | size | style | brand | budget | feature | use_case | other",
      "old_value": null,
      "new_value": null,
      "polarity": "include | exclude",
      "strength": "hard | soft",
      "scope": "attribute | category | all",
      "confidence": 0.0
    }
  ]
}
```

Validation rules, all tested:

- unknown `operation`, `attribute`, `polarity`, `strength`, or `scope` values
  drop that single operation and increment a `rejected_operations` counter;
  they never abort the turn;
- `confidence`, `mode_confidence`, and `generality` clamp to `[0.0, 1.0]`;
  a non-numeric value becomes the configured default, not an exception;
- `replace` without `new_value` is rejected; `remove` carrying a `new_value` is
  rejected; `reset` requires an explicit `scope`;
- `budget` values parse to a normalized numeric bound with a currency-free
  representation, and a failed parse downgrades the operation to `other`
  rather than inventing a number;
- string values are trimmed, case-folded for comparison, and length-capped,
  while the original form is retained for display;
- an empty or fully rejected operation list is a valid delta, not a failure.

`fake.py` provides three interpreters:

- `ScriptedInterpreter`: replays a fixture list of deltas by turn index, for
  Person 4's orchestration tests;
- `HeuristicInterpreter`: regex and keyword extraction over the section 2.1
  templates, producing real deltas with no model. Seed its attribute
  vocabularies from the classifier cascade in section 2.1, since the constraint
  strings it will parse are the same strings that cascade bucketed. This
  doubles as the P5 degraded path, so build it as production code, not as a
  stub;
- `FaultyInterpreter`: configurable to return malformed JSON, an unknown
  operation, an out-of-range confidence, a timeout, or an exception.

### Tests — `tests/test_models.py`

- every schema field validates and every invalid variant is rejected;
- confidence clamping at `-1`, `0`, `0.5`, `1`, `2`, `"high"`, and `None`;
- malformed JSON, truncated JSON, JSON wrapped in prose, and a JSON array where
  an object was required all produce a safe empty delta plus a recorded error;
- an unknown attribute or operation is dropped individually while valid sibling
  operations survive;
- `Usage` cannot go negative and accumulates across calls;
- the fake interpreters are stable for a fixed seed and fixture.

### Exit gate

- `tikitaka/state/schema.py` imports with no provider dependency.
- Person 4 can run an end-to-end fake turn using `HeuristicInterpreter`.
- Every malformed-output test passes without raising.

## 7. Phase P2 — SessionState, provenance, and the reducer

### Outcome

The deterministic core of the workstream. This is the highest-value phase and
the one most likely to decide the Intent Override and Boundary scores.

### Owned implementation

```text
tikitaka/state/
  session.py       SessionState, Constraint, intent versions
  reducer.py       validated mutation
  extractor.py     message -> StateDelta via the configured interpreter
```

`SessionState` matches `ARCHITECTURE.md` section 5 field for field:
`session_id`, `turn`, `mode`, `intent_version`, `active_constraints`,
`constraint_history`, `no_preference`, `asked_attributes`, `shown_by_intent`,
`candidate_set`, `profile_seed`, `active_query_summary`. Do not rename fields;
Persons 3 and 4 read them.

Each `Constraint` carries: `attribute`, `value`, `normalized_value`,
`polarity`, `strength`, `source_turn`, `confidence`, `intent_version`,
`status` (`active` | `replaced` | `retracted` | `needs_revalidation`), and
`category_dependent`.

### Reducer rules

Applied in a fixed order so the result is reproducible:

1. `RESET` with `scope = all` first, then `REPLACE`, then `REMOVE` and
   `EXCLUDE`, then `ADD`, then `NO_PREFERENCE`. The order is a documented
   decision because a single turn can carry a correction plus an addition.
2. `ADD` on an attribute that already holds an equal normalized value is a
   no-op that refreshes `source_turn` and keeps the higher confidence.
3. `ADD` on an attribute holding a different value for a single-valued
   attribute becomes a `REPLACE`, with the old constraint moved to
   `constraint_history` with `status = replaced`. Multi-valued attributes
   (`feature`, `use_case`) accumulate up to a configured cap.
4. `REMOVE` retracts without creating a negative. Removing a budget must never
   yield a `budget < 0` or an `exclude` budget constraint.
5. `EXCLUDE` stores `polarity = exclude` and never silently removes a matching
   include; if both exist for the same normalized value, the exclude wins and
   the include is retracted with a recorded reason.
6. `NO_PREFERENCE` adds the attribute to `no_preference`, clears any pending
   ask, and is permanent for the current intent version. It creates no
   constraint.
7. Attribute exhaustion (`I don't have an additional preference for X`) marks
   the attribute as asked-and-dry in `asked_attributes` with a distinct flag,
   which is not the same as `no_preference`.

### Dependency-aware clearing (DG-03)

Implement as a single function with an explicit attribute-dependency table:

- `UNIVERSAL = {budget}` survives a category change unchanged;
- `CATEGORY_DEPENDENT = {size, style, material, feature, use_case}` is retracted
  on a major category change;
- `AMBIGUOUS = {color, brand}` survives with `status = needs_revalidation`, and
  Person 3's question policy may spend a turn confirming one.

The four DG-03 paths, each with its own test:

1. a direct correction replaces only its attribute, with no version bump;
2. an explicit restart clears all conversation-derived constraints, bumps
   `intent_version`, and leaves `profile_seed` untouched;
3. a major category change bumps `intent_version`, applies the dependency table
   above, and preserves an applicable budget;
4. ambiguity is flagged rather than resolved by the reducer.

Every version bump appends to `constraint_history` with the triggering turn and
operation so Person 4 can render an explainable trace, and signals Person 3 to
re-enable previously shown products for the new version.

### Tests — `tests/test_state.py`

Mapped one to one against the required tests in `IMPLEMENTATION_PLAN.md`
section 7:

- `test_accumulates_compatible_constraints`
- `test_replace_conflicting_color`, `..._material`, `..._category`
- `test_remove_budget_does_not_create_negative_budget`
- `test_exclude_leather_and_exclude_red`
- `test_no_preference_suppresses_future_asks`
- `test_full_reset_versus_single_attribute_replacement`
- `test_category_change_clears_dependent_preserves_budget_flags_ambiguous`
- `test_rejects_unknown_attributes_and_operations`
- `test_clamps_invalid_confidence`
- `test_malformed_json_recovers_or_fails_safely`
- `test_sessions_do_not_share_state`
- `test_explicit_dialogue_state_outranks_profile`
- `test_attribute_exhaustion_is_distinct_from_no_preference`
- `test_reducer_is_order_stable_for_a_multi_operation_delta`

Session isolation must be proven with mutable defaults: build two sessions from
the same profile dict, mutate one, and assert the other is unchanged.

### Exit gate

- The full matrix passes offline in under a second.
- No test imports a provider, a catalog, or the evaluator.
- Person 3 can read `no_preference`, `asked_attributes`, and `intent_version`
  without touching reducer internals.

## 8. Phase P3 — Active query builder and profile isolation

### Outcome

Retrieval consumes a structured plan built from active state, never a
concatenation of raw conversation turns.

### Owned implementation

`tikitaka/state/query_builder.py` produces the `SearchPlan` agreed in P0:

- `text_query`: the rewritten active-state summary, also stored on
  `SessionState.active_query_summary`;
- `must_terms` from hard include constraints;
- `should_terms` from soft include constraints;
- `exclude_terms` from exclude constraints;
- `filters`: normalized budget bound, category path, and any structured value
  Person 2 can evidence;
- `attribute_values`: the normalized attribute map, so Person 2 can boost
  without re-parsing text;
- `mode`, `intent_version`, and `revalidation_flags`;
- `profile_bias`: the profile tags plus the configured weight, carried
  separately so a weight of `0` removes its influence entirely.

Two rewrite routes sit behind one interface: a deterministic template assembler
that needs no model, and an LLM rewrite for Browsing sessions where the visible
text is thin. The deterministic route is the default until an ablation shows
the LLM rewrite wins on held-out results.

Profile isolation for DG-02:

- `profile_seed` is written once at `reset` and never mutated by the reducer;
- profile tags never enter `active_constraints` and never become `must_terms`;
- a decay factor reduces profile weight as explicit constraints accumulate;
- `profile_weight = 0` must produce a `SearchPlan` identical to one built with
  no profile at all. That equality is the test.

### Tests

- an empty state produces a valid, non-crashing plan;
- an explicit constraint that contradicts a profile tag wins, and the profile
  tag does not appear in `must_terms`;
- the `profile_weight = 0` equality test described above;
- exclusions appear in `exclude_terms` and never in `should_terms`;
- a plan built at intent version 2 contains no retracted version 1 constraint;
- plan construction is deterministic for a fixed state.

### Exit gate

- Person 2 retrieves using only `SearchPlan` fields.
- DG-02 can be ablated by changing one configuration value.

## 9. Phase P4 — Primary API adapter

### Outcome

`gpt-5.6-terra` at `xhigh` reasoning drives interpretation, with full usage
accounting and no behavioral difference in the validated state.

### Owned implementation

`tikitaka/models/api_llm.py`:

- construction from configuration plus an environment credential; a missing
  credential raises `CredentialMissing` at construction, never at turn time;
- one structured-output call per turn, with the P1 schema enforced at the
  request level where the API supports it and re-validated locally regardless;
- timeout, one bounded repair retry, and a jittered backoff on transport
  errors, all configurable and all defaulting to values that keep a
  200-session run tractable;
- the repair retry resends the malformed output with the schema and an explicit
  correction instruction; it is attempted at most once per turn;
- after a failed repair, fall through to `HeuristicInterpreter` and record the
  route change in the trace;
- `Usage` records prompt tokens, completion tokens, model identity, provider,
  reasoning level, latency, and estimated cost; reasoning tokens are recorded
  separately and included in the completion total if the API bills them so;
- prompt assembly is a pure function of `(message, state, PROMPT_VERSION)` so a
  trace can be replayed;
- an optional on-disk response cache keyed by that tuple, so repeated
  evaluation runs during tuning do not re-bill identical turns. The cache is
  off by default in reported runs and its state is recorded in the experiment
  configuration.

Prompt design notes:

- give the model the active state, not the raw transcript; summarizing the
  transcript is the reducer's job and it has already happened;
- state the closed attribute vocabulary in the prompt and require operations
  only from the enum;
- require `no_preference` for the boundary template and a distinct exhaustion
  signal for the "no additional preference" template;
- never place catalog products in the interpretation prompt; that token budget
  belongs to Person 3's reranker.

### Tests

Unit tests use a transport double, not the network:

- a well-formed response produces the expected delta and non-zero usage;
- a malformed response triggers exactly one repair, then the heuristic
  fallback;
- a timeout produces the fallback and a recorded `ModelTimeout`, not an
  exception;
- a hallucinated attribute outside the enum is dropped;
- the credential never appears in `repr`, logs, traces, or error messages;
- `Usage` totals are attributable to a route identity.

One live integration job, run explicitly and kept out of the unit suite: ten
representative sessions across all four scenarios, capturing the structured
traces required by section 7 of the implementation plan.

### Exit gate

The synthetic multi-turn conversation from the P0 fixtures produces the same
validated state through the fake and API adapters. Divergence is either a
prompt defect or a schema defect, and must be fixed before P5.

## 10. Phase P5 — Fault and degraded-path hardening

### Outcome

No model failure can produce an invalid official response or an exception that
reaches the evaluator.

### Tasks

- Run the fault matrix through Person 4's orchestrator: malformed JSON, empty
  response, refusal, timeout, connection error, missing credential, rate limit,
  and an operation list of 500 entries.
- Cap operations per delta and constraints per attribute; a flood must be
  truncated deterministically, not accepted.
- Confirm that with the credential unset, the whole agent still runs the full
  evaluator through the heuristic route and produces valid responses. This is
  the M5 evidence Person 1 owns.
- Measure and record the quality delta between the API route and the heuristic
  route on the held-out split, so the submission can state it honestly.
- Verify no secret appears in any trace, report, or committed fixture.

### Exit gate

- Zero uncaught exceptions across the fault matrix.
- A network-free run produces valid deterministic output.
- The degraded-path quality delta is a recorded number, not an estimate.

## 11. Phase P6 — Routing, pinning, ablation support, freeze

### Outcome

The model-selection layer behaves per DG-04 and every Person 1 knob is
reproducibly pinnable.

### Tasks

- Implement `tikitaka/models/selector.py` routing inputs: task type, state
  confidence, candidate uncertainty from Person 3, and remaining turn budget.
  Person 1 supplies the routing decision and its recorded reason; Person 4
  records it in the experiment report.
- Guarantee an embedding route is never used against an index built by another
  embedding model: the route carries the index identity and the selector
  refuses a mismatch loudly. Person 2 owns the index; Person 1 owns the refusal.
- Expose a pin configuration that fixes every route for reproducible runs.
- Support the ablations Person 4 will run: profile weight `0` versus soft
  prior, deterministic versus LLM query rewrite, heuristic versus API
  interpretation, and a reasoning-level comparison if the owner authorizes the
  cost.
- Freeze `PROMPT_VERSION` and `SCHEMA_VERSION` and record them in the final
  report alongside the model identity and reasoning level.

### Exit gate

- The same pinned configuration reproduces the same validated state.
- An index/route mismatch is impossible to reach silently.

## 12. Interfaces Person 1 publishes

Other owners depend on exactly these. Changing any of them requires an
announcement and updated fixtures.

| Consumer | Surface |
|---|---|
| Person 2 | `SearchPlan`, the `EmbeddingModel` protocol, route/index identity |
| Person 3 | `SessionState` read fields, `StateDelta.generality`, `mode`, `mode_confidence`, `no_preference`, `asked_attributes`, `intent_version`, and `TextModel` for the reranker |
| Person 4 | `IntentInterpreter`, the fake interpreters, `Usage`, the error taxonomy, prompt and schema versions |

## 13. Risks and owner questions

1. **API details are unverified in-repo.** Base URL, transport, credential
   variable name, and how `xhigh` is expressed are unknown. P4 is blocked until
   the owner answers. P0-P3 are not.
2. **DG-01 makes every CLARIFY turn a guaranteed miss** for that turn, because
   a clarify carries no recommendations and the evaluator only checks the
   returned list. This is a settled owner decision and Person 1 implements it
   as given; it is flagged here because it sets the true MTTC cost of a
   question for Person 3's policy.
3. **Override sessions discard pre-override hits.** The evaluator ignores a
   correct product until the override lands on turn 3 or 4. Person 1's job is
   to make the version bump unmistakable on that turn.
4. **The cost of `xhigh` across 200 sessions is unbudgeted.** Report
   per-session token cost from the first live run so the owner can cap it
   before M4 tuning consumes the budget.
5. **Public-simulator overfitting is the sharpest risk in this workstream.**
   Because the simulator is templated, regex extraction could match the LLM on
   the public set while generalizing worse to the 800 private sessions, and the
   `other` wildcard in section 2.1 could collapse against a private simulator
   that classifies differently. Report the heuristic and API routes separately
   and honestly; do not select the heuristic, or a wildcard ask strategy, on
   public-set evidence alone.

## 14. Definition of done

- [x] Model and state protocols are stable and documented.
- [x] Fake, heuristic, faulty, and primary API adapters all satisfy them.
- [x] The full `tests/test_state.py` and `tests/test_models.py` matrices pass
      offline. Verified by static scan for 3.11+ syntax; the local interpreter
      is 3.13, so the 3.10 floor is checked rather than executed.
- [x] A synthetic multi-turn conversation yields identical validated state
      through the fake and API adapters (`tests/test_route_equivalence.py`).
- [x] No provider code is reachable from retrieval, ranking, or evaluation
      (AST import scan, clean).
- [x] Profile influence is isolated and `profile_weight = 0` is provably inert.
- [x] Structured traces exist for Buying, Browsing, Intent Override, Boundary
      (`artifacts/traces/`, produced by `scripts/capture_traces.py`).
- [x] Prompt and schema versions are recorded with every reported score.
- [x] Credentials are environment-only and absent from all output.
- [x] The network-free heuristic route produces valid responses, with a
      recorded quality delta (`docs/p6/ROUTING_BENCHMARK.md`).

## 15. Mapping to the shared schedule

| Shared milestone | Person 1 phase |
|---|---|
| M0 contract freeze | P0 |
| M1 deterministic vertical slice | P1, P2 |
| M2 dense hybrid retrieval | P3, plus embedding-protocol support to Person 2 |
| M3 primary LLM vertical slice | P4 |
| M4 policy and score tuning | P6 ablation support |
| M5 network-degraded path | P5 |
| M6 submission and demo | traces, versions, cost disclosure |

Against the three-day allocation: P0-P2 on Day 1, P3 and P4 on Day 2, P5 and P6
on Day 3. Do not end Day 1 without the heuristic interpreter and the reducer
running inside Person 4's orchestrator.
