# P0 Owner Acknowledgments

Contract proposal: `docs/p0/CONTRACT_PROPOSAL.md`

Proposed version: `0.1.0`

Each affected owner should review the proposal, answer their review questions,
and replace `pending` with an acknowledgment plus the reviewing commit hash.
Acknowledgment means the interface is sufficient to begin implementation; it
does not transfer file ownership.

| Owner | Scope | Status | Reviewing commit | Notes |
|---|---|---|---|---|
| Person 1 | Models, state, query construction, usage | acknowledged with requested changes | `f0dd286` | Sufficient to start P1/P2. Three blocking changes before `0.1.0` freezes; see Person 1 review responses below. |
| Person 2 | Catalog, retrieval, evidence, index identity | pending | — | — |
| Person 3 | Decision policy, reranking, diagnostics | pending | — | — |
| Person 4 | Contracts, orchestration, evaluation boundary | proposed | `3db9434` | Awaiting affected-owner review |

## Coordination checklist

- [ ] Branch and primary file ownership confirmed.
- [ ] Enum values and allowed attributes accepted.
- [ ] State validation failure granularity settled.
- [ ] Candidate evidence and diagnostics settled.
- [ ] Search-plan and index identity fields settled.
- [ ] Decision diagnostics and information-gain scale settled.
- [ ] Usage attribution fields settled.
- [ ] No circular dependency introduced by `SessionStateView`.
- [ ] Fake implementation obligations accepted by every owner.
- [ ] Contract version `0.1.0` approved for P1 implementation.

Person 1 has not ticked any box unilaterally. Every remaining item needs either
Person 4's decision on the requested changes or acknowledgment from Persons 2
and 3. Person 1's own positions on each are recorded below.

## Person 1 review responses

Reviewing commit: `f0dd286`. Person 1 build plan: `docs/PERSON_1_BUILD_PLAN.md`.

The design boundaries, versioning process, and runtime normalization invariants
are right, and P1/P2 can start against this proposal. Three changes are needed
before `0.1.0` is frozen, because as written the state contract cannot express
DG-03 and `StateOperation` cannot construct a `Constraint`.

### Blocking 1 — `StateOperation` cannot produce a `Constraint`

`Constraint` requires `polarity`, `strength`, and `confidence`. `StateOperation`
carries none of them, so the reducer would have to invent all three. Defaulting
`strength` is not safe: hard versus soft is the model's judgment and it drives
Person 3's hard-constraint enforcement and the missing-metadata safeguard.

Requested: add `polarity`, `strength`, and `confidence` to `StateOperation`.
Structured model output already produces them per operation.

### Blocking 2 — `Constraint` cannot express dependency-aware clearing (DG-03)

`Constraint` has nowhere to record that a constraint was superseded, retracted,
or flagged for revalidation, and no intent version. DG-03 rule 4 requires
ambiguous constraints to be marked for revalidation rather than silently kept or
dropped, and Person 3's question policy reads that flag. `constraint_history`
also has to distinguish replaced from retracted.

Requested additions to `Constraint`:

- `normalized_value` — `value: object` is too loose to validate centrally; a
  budget needs a parsed numeric bound alongside its original display form;
- `intent_version: int`;
- `status: Literal["active", "replaced", "retracted", "needs_revalidation"]`;
- `category_dependent: bool`, driving the category-change dependency table:
  universal `budget` survives, `size`/`style`/`material`/`feature`/`use_case`
  are retracted, `color`/`brand` become `needs_revalidation`.

Without these, DG-03 either lives in a parallel Person 1 structure that diverges
from the shared contract, or cannot be implemented as settled.

### Blocking 3 — `StateReducer.apply` returns the wrong type

`apply(...) -> SessionStateView` implies the reducer hands back a read-only
view, leaving orchestration nothing mutable to persist. `ARCHITECTURE.md`
section 5 defines `SessionState` as mutable, with `set` and `dict` members.

Requested: the reducer takes and returns the concrete `SessionState`, which
Person 1 owns. `SessionStateView` remains the narrowed thing handed to Persons 2
and 3. Otherwise the view grows mutation methods or Person 4 keeps a parallel
copy of state.

Two coordination items belong in the same decision:

1. Person 1 defines the `SessionState` type; Person 4 owns the registry holding
   instances. Requesting explicit confirmation.
2. `asked_attributes` and `shown_by_intent` are populated after the decision,
   which is Person 4's half of the turn, but the Person 1 rule is that the
   reducer owns all mutation. Either Person 4 calls a reducer method Person 1
   exposes, or Person 1 concedes those two fields to the orchestrator. Person 1
   prefers the former.

### Answers to the Person 1 review questions

**Is `SessionStateView` sufficient to prevent a contracts/state import cycle?**
Yes, under three conditions: it is a `Protocol` in `tikitaka/contracts/` with
read-only members; `tikitaka/state/` imports contracts and never the reverse;
and no annotation inside contracts names the concrete `SessionState`. Structural
typing, no inheritance. See Blocking 3 for the reducer signature.

**Should one invalid state operation reject the entire delta or only itself?**
Only itself. Reject the operation, keep valid siblings, count the rejection. An
override turn carries a correction plus an addition in one delta; whole-delta
rejection on a single bad operation would drop the intent-version bump and
damage the 15 percent Intent Override slice specifically. A top-level schema
violation — not an object, unparseable — yields an empty delta, which is valid
and not an error. Both behaviors are deterministic and will be tested.

Requested for reporting: add `rejected_operations: int` and `schema_version:
str` to `StateDelta`.

**Which query-builder fields must be stable for Person 2?** Proposing these
concrete names now, since Person 1 and Person 2 both code against them:
`text_query`, `must_terms`, `should_terms`, `exclude_terms`, `filters` (parsed
budget bound and category path), `attribute_values` (the normalized attribute
map, so retrieval boosts without re-parsing text), `mode`, `intent_version`,
`revalidation_flags`, `no_preference`, `profile_bias` (tags plus weight, carried
separately so weight `0` is provably inert), and the route pin or index identity
when a dense route is selected. Person 2 to confirm or rename before P1 lands.

**Can all provider usage be represented without provider-specific types?** Yes,
with four additions:

- `calls: int` and `repairs: int`. Orchestration aggregates once per actual call
  and prevents retry double counting, but the Person 1 adapter makes at most one
  repair retry per turn, so a single logical `interpret()` can be two HTTP
  calls. Counters make that reconcilable without provider types.
- `reasoning_tokens: int`. `xhigh` may bill reasoning separately from
  completion, while the official response carries only `prompt_tokens` and
  `completion_tokens`. Whether reasoning rolls into the reported completion
  count should be a deliberate, reversible decision, and the cost disclosure
  has to be honest either way.
- `cache_hit: bool`. Person 1 plans an optional replay cache for tuning runs; a
  cached turn must not add cost or tokens to a reported total.

Also state units and currency for `estimated_cost`.

### All-owner acknowledgements

- Mutual-exclusion action invariant (DG-01): acknowledged. `CLARIFY` carries one
  allowed attribute and no recommendations; `RECOMMEND` carries
  `ask_attribute = null`. Enforced centrally by Person 4, not duplicated in
  Person 1 modules.
- Catalog-valid, shortlist-only reranking: acknowledged. Nothing Person 1 owns
  may introduce a product ID.
- Label-free participant runtime boundary: acknowledged. `interpret()` takes
  `(message, state)` only; `scenario_type`, `ground_truth`, `category_bucket`,
  and `difficulty_bucket` are never read.
- Fields forcing a cross-ownership import: only the `StateReducer.apply` return
  type in Blocking 3.

### Ownership and obligations

Person 1 claims `tikitaka/models/`, `tikitaka/state/`, `tests/test_state.py`,
`tests/test_models.py`, and `tests/fixtures/conversations/`.

Fake obligation accepted: `ScriptedInterpreter`, `HeuristicInterpreter`, and a
`FaultyInterpreter` covering malformed JSON, unknown operation, out-of-range
confidence, timeout, and raised exception, all by M1, so orchestration never
waits on the API.

### One evaluator finding that affects Person 3 and reporting

The simulator discloses a constraint only when `ask_attribute` matches the
bucket its keyword classifier assigns. That classifier never emits `category` or
`brand`, and `ask_attribute = "other"` bypasses classification entirely,
returning up to two undisclosed constraints of any kind. On the public set
`other` is therefore a strictly dominant ask. Recorded in
`docs/PERSON_1_BUILD_PLAN.md` section 2.1 rather than acted on: it is Person 3's
policy call, and an obvious overfitting risk against the 800 private sessions.

## Change log

Record review-driven changes here before freezing the contract.

| Date | Proposed by | Change | Affected owners | Status |
|---|---|---|---|---|
| 2026-08-29 | Person 4 | Initial P0 proposal | Persons 1–4 | pending review |
| 2026-08-29 | Person 1 | Add `polarity`, `strength`, `confidence` to `StateOperation` | Persons 1, 3, 4 | requested |
| 2026-08-29 | Person 1 | Add `normalized_value`, `intent_version`, `status`, `category_dependent` to `Constraint` | Persons 1, 3, 4 | requested |
| 2026-08-29 | Person 1 | `StateReducer.apply` takes and returns concrete `SessionState`, not `SessionStateView` | Persons 1, 4 | requested |
| 2026-08-29 | Person 1 | Add `rejected_operations`, `schema_version` to `StateDelta` | Persons 1, 4 | proposed |
| 2026-08-29 | Person 1 | Add `calls`, `repairs`, `reasoning_tokens`, `cache_hit` to `Usage`; state cost units | Persons 1, 4 | proposed |
| 2026-08-29 | Person 1 | Concrete `SearchPlan` field names for Person 2 review | Persons 1, 2, 4 | proposed |
