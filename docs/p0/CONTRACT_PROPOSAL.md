# P0 Shared Contract Proposal

Status: **Draft for acknowledgment**

Proposed contract version: `0.1.0`

Owner: Person 4

This document is the P0 review artifact. It does not freeze the shared Python
contracts by itself. Persons 1, 2, and 3 must acknowledge the portions their
workstreams produce or consume before Person 4 lands the P1 contract module.

## Design boundaries

- Shared contracts live in `tikitaka/contracts/` and remain dependency-light.
- LLM output is untrusted and cannot mutate state directly.
- Catalog identifiers become candidates only after catalog validation.
- A reranker may reorder or remove supplied candidates but cannot add IDs.
- Evaluator labels and hidden fields never appear in runtime contracts.
- Profile data remains session-local and separate from explicit constraints.
- Runtime routing may be automatic; evaluation configuration may pin all routes.
- An embedding query route must match the product-index embedding identity.
- Missing product metadata is unknown, not a failed hard constraint.
- DG-01 is represented by one `TurnDecision.action`: clarification and
  recommendation are mutually exclusive.

## Proposed enums

```python
ConstraintPolarity = Literal["include", "exclude"]
ConstraintStrength = Literal["hard", "soft"]
StateOperationKind = Literal[
    "add", "remove", "replace", "exclude", "no_preference", "reset"
]
InferredMode = Literal["buying", "browsing", "unknown"]
TurnAction = Literal["clarify", "recommend"]
```

Attributes accepted from structured model output must be drawn from a single
allowlist. The initial allowlist matches the official clarification attributes:
`category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`,
`use_case`, and `other`. Adding an internal-only attribute requires a contract
review and an explicit mapping at the official response boundary.

## Proposed domain records

### `Constraint`

```python
@dataclass(frozen=True)
class Constraint:
    attribute: str
    value: object
    polarity: ConstraintPolarity
    strength: ConstraintStrength
    source_turn: int
    confidence: float
```

Validation rules:

- `source_turn` is within the official 1-to-10 range;
- confidence is normalized deterministically into `[0.0, 1.0]`;
- unknown attributes are rejected before state mutation;
- absence of catalog evidence does not invert or falsify the constraint.

### `StateOperation`

```python
@dataclass(frozen=True)
class StateOperation:
    operation: StateOperationKind
    attribute: str | None
    old_value: object | None
    new_value: object | None
    scope: str | None
```

Validation rules:

- operation-specific required and forbidden fields are checked centrally;
- `reset` has an explicit scope and cannot silently mean global reset;
- `replace` names the affected attribute and does not imply unrelated clearing;
- category-change dependency clearing is performed by the validated reducer,
  not encoded as arbitrary model-created operations.

### `StateDelta`

```python
@dataclass(frozen=True)
class StateDelta:
    inferred_mode: InferredMode
    mode_confidence: float
    operations: tuple[StateOperation, ...]
    generality: float
```

Unknown operations or attributes invalidate the affected operation. Person 1
should propose whether one bad operation rejects the full delta or only that
operation; the chosen behavior must be deterministic and tested.

### `ProductEvidence`

```python
@dataclass(frozen=True)
class ProductEvidence:
    matched_fields: tuple[str, ...]
    matched_constraints: tuple[str, ...]
    contradicted_constraints: tuple[str, ...]
    unknown_fields: tuple[str, ...]
    route_details: Mapping[str, object]
```

Evidence must be compact enough for bounded shortlist reranking and structured
enough for failure analysis. It must not include hidden target data.

### `Candidate`

```python
@dataclass(frozen=True)
class Candidate:
    parent_asin: str
    product_evidence: ProductEvidence
    sparse_rank: int | None
    dense_rank: int | None
    structural_score: float | None
    fused_score: float
```

Candidate construction requires a catalog-valid `parent_asin`. Route ranks are
positive when present. Stable tie-breaking must be specified by Person 2.

### `TurnDecision`

```python
@dataclass(frozen=True)
class TurnDecision:
    action: TurnAction
    ask_attribute: str | None
    reason: str
    expected_information_gain: float
```

Invariants:

- `clarify` requires one allowed `ask_attribute`;
- `recommend` requires `ask_attribute is None`;
- turn 10 cannot choose an unanswered clarification;
- `reason` is diagnostic evidence, not authority to bypass validation.

### `Usage`

```python
@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    provider: str | None = None
    model: str | None = None
    reasoning_level: str | None = None
    estimated_cost: float | None = None
    route: str | None = None
```

Numeric usage values are non-negative. Orchestration aggregates one record per
actual call while evaluation preserves component and route attribution.

### `SearchPlan`

`SearchPlan` is the provider-neutral snapshot consumed by retrieval. Person 1
and Person 2 must jointly acknowledge its final fields. At minimum it carries:

- current intent version and inferred mode;
- normalized active constraints and exclusions;
- no-preference attributes;
- query text or query components built from active state;
- session-local profile soft signals kept separate from explicit constraints;
- route pins or automatic-routing inputs;
- embedding/index identity when a dense route is selected.

## Proposed protocols

```python
class IntentInterpreter(Protocol):
    def interpret(
        self,
        message: str,
        state: SessionStateView,
    ) -> tuple[StateDelta, Usage]: ...


class StateReducer(Protocol):
    def apply(
        self,
        state: SessionStateView,
        delta: StateDelta,
        turn: int,
    ) -> SessionStateView: ...


class QueryBuilder(Protocol):
    def build(self, state: SessionStateView) -> SearchPlan: ...


class Retriever(Protocol):
    def search(self, plan: SearchPlan, limit: int) -> list[Candidate]: ...


class DecisionPolicy(Protocol):
    def choose(
        self,
        state: SessionStateView,
        candidates: Sequence[Candidate],
        turn: int,
    ) -> TurnDecision: ...


class Reranker(Protocol):
    def rank(
        self,
        state: SessionStateView,
        candidates: Sequence[Candidate],
        top_k: int,
    ) -> tuple[list[str], Usage]: ...
```

`SessionStateView` must expose only the state needed by a consumer. Person 1
owns the mutable state implementation; Person 4 owns the orchestration-facing
view and boundary. The final design must avoid circular imports between
contracts and `tikitaka/state/`.

## Runtime response normalization

Person 4 will enforce these invariants after component execution:

- response contains a string `message`;
- `ask_attribute` is allowed or `null`;
- clarification has no recommendations;
- recommendation has `ask_attribute = null`;
- recommendations contain the first 10 unique catalog-valid IDs in input order;
- optional usage contains non-negative prompt and completion token counts;
- component exceptions or malformed outputs fall back to a valid deterministic
  response.

## Versioning and change process

- `0.1.0` becomes active only after all affected owners acknowledge it.
- Backward-compatible optional fields increment the minor version.
- Breaking field, invariant, or semantic changes increment the major version.
- Every change includes migration notes, updated fakes, and contract tests.
- Person 4 announces a proposed change before merge and records acknowledgements
  in `docs/p0/OWNER_ACKNOWLEDGEMENTS.md` or its successor.

## Review questions

### Person 1

- Is `SessionStateView` sufficient to prevent a contracts/state import cycle?
- Should one invalid state operation reject the entire delta or only itself?
- Which query-builder fields must be stable for Person 2?
- Can all provider usage be represented without provider-specific types?

### Person 2

- What exact evidence and route diagnostics are required on `Candidate`?
- What stable tie-break rule will retrieval expose?
- Which index identity fields belong in `SearchPlan` versus the index manifest?
- Which structured values need an explicit unknown state?

### Person 3

- Which candidate diagnostics are required by generality and question value?
- Is `expected_information_gain` a normalized score or an unbounded diagnostic?
- Does deterministic reranking need richer constraint-match evidence?
- Which decision reasons must be machine-readable for reporting?

### All owners

- Acknowledge the mutual-exclusion action invariant.
- Acknowledge catalog-valid shortlist-only reranking.
- Acknowledge label-free participant runtime boundaries.
- Identify any field that would force an import across ownership boundaries.
