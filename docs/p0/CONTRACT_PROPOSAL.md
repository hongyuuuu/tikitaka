# P0 Shared Contract Specification

Status: **Accepted and frozen for P1 implementation**

Contract version: `0.1.0`

Contract owner: Person 4

Accepted: 2026-08-29

The filename is retained so existing review links remain valid. This document
supersedes the initial proposal in commit `f0dd286` and incorporates the
recorded reviews from Persons 1, 2, and 3. The Python records and protocols land
in P1; their implementation must preserve the semantics frozen here.

## 1. Authority and boundaries

- Shared contracts live in `tikitaka/contracts/` and remain dependency-light.
- Person 1 owns `tikitaka/models/`, the concrete mutable `SessionState`, state
  validation/reduction, query construction, and all state mutation methods.
- Person 2 owns catalog normalization, retrieval, candidate evidence, index
  manifests, and deterministic retrieval ordering.
- Person 3 owns decision diagnostics, question value, response policy,
  deterministic ranking, and bounded shortlist reranking.
- Person 4 owns shared-contract versions, the isolated session registry,
  orchestration, official response normalization, evaluation, and release.
- LLM output is untrusted. Deterministic validation runs before state mutation.
- Catalog identifiers become candidates only after catalog validation.
- A reranker may reorder or remove supplied candidates but cannot add IDs.
- Runtime participant contracts never include evaluator labels or hidden data.
- Profile data remains session-local, soft, decaying, and separate from explicit
  dialogue constraints.
- Runtime routing may be automatic; evaluation configurations may pin every
  route.
- A query embedding is searched only against its matching product index.
- Missing product metadata is `unknown`, never an implicit contradiction.
- DG-01 is enforced centrally: clarification and recommendation are mutually
  exclusive actions.

## 2. Shared aliases and closed values

The P1 Python implementation may use enums instead of `Literal`, but serialized
values and validation semantics must remain these values.

```python
Attribute = Literal[
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
]

ConstraintPolarity = Literal["include", "exclude"]
ConstraintStrength = Literal["hard", "soft"]
ConstraintStatus = Literal[
    "active", "replaced", "retracted", "needs_revalidation",
]
StateOperationKind = Literal[
    "add", "remove", "replace", "exclude", "no_preference", "reset",
]
OperationScope = Literal["attribute", "conversation", "intent"]
InferredMode = Literal["buying", "browsing", "unknown"]
TurnAction = Literal["clarify", "recommend"]
EvidenceOutcome = Literal["match", "contradiction", "unknown"]
RoutePolicy = Literal["auto", "sparse", "dense", "hybrid"]
DecisionReasonCode = Literal[
    "final_turn",
    "valuable_clarification",
    "low_question_value",
    "no_eligible_attribute",
    "ranking_stable",
    "insufficient_evidence",
    "component_fallback",
]
```

Adding a closed value is a contract change. An internal-only attribute also
requires an explicit mapping or rejection rule at the official response
boundary.

## 3. Domain records

### 3.1 `Constraint`

```python
@dataclass(frozen=True)
class Constraint:
    attribute: Attribute
    value: object
    normalized_value: object
    polarity: ConstraintPolarity
    strength: ConstraintStrength
    source_turn: int
    confidence: float
    intent_version: int
    status: ConstraintStatus = "active"
    category_dependent: bool = False
```

Invariants:

- `source_turn` is within the official 1-to-10 range.
- `intent_version` is positive.
- Confidence is deterministically clamped into `[0.0, 1.0]` before the record
  becomes trusted state.
- `value` preserves the user-facing/source representation;
  `normalized_value` is the deterministic representation used for matching.
- Replaced, retracted, and revalidation constraints remain available as
  history but are not treated as active hard filters.
- `category_dependent` is provenance used by dependency-aware clearing. It is
  not permission to apply one unconditional attribute-clearing table.
- A category change preserves applicable universal constraints such as budget,
  retracts constraints proven incompatible or derived from the old category,
  and marks ambiguous survivors `needs_revalidation`.
- Missing catalog evidence never changes a constraint into a contradiction.

### 3.2 `StateOperation`

```python
@dataclass(frozen=True)
class StateOperation:
    operation: StateOperationKind
    attribute: Attribute | None
    old_value: object | None
    new_value: object | None
    scope: OperationScope
    polarity: ConstraintPolarity | None
    strength: ConstraintStrength | None
    confidence: float | None
```

Operation rules:

- `add` and `exclude` require an attribute, new value, polarity, strength, and
  confidence. `exclude` normalizes polarity to `exclude`.
- `replace` requires an attribute, old value, new value, strength, confidence,
  and the polarity of the replacement.
- `remove` requires an attribute or an unambiguous old value and does not create
  a negative constraint.
- `no_preference` requires an attribute and suppresses repeated questions for
  that attribute in the current intent version.
- `reset` requires explicit `conversation` or `intent` scope. It cannot silently
  mean a global or cross-session reset.
- Fields irrelevant to an operation must be `None`; the strict schema rejects
  contradictory field combinations.
- The validated reducer, not the model, performs dependency-aware clearing and
  intent-version increments.

### 3.3 `StateDelta`

```python
@dataclass(frozen=True)
class StateDelta:
    inferred_mode: InferredMode
    mode_confidence: float
    operations: tuple[StateOperation, ...]
    generality: float
    rejected_operations: int
    schema_version: str
```

Validation semantics:

- Mode confidence and generality are normalized into `[0.0, 1.0]`.
- An invalid operation is rejected individually; valid sibling operations are
  retained and `rejected_operations` is incremented.
- An unparseable or top-level-invalid model response yields a deterministic
  empty fallback delta plus a model-call failure diagnostic. It never mutates
  state directly.
- `schema_version` identifies the structured-output schema used for parsing and
  participates in experiment and cache identity.

### 3.4 `SessionStateView`

`SessionStateView` is a structural, read-only protocol in
`tikitaka/contracts/`. It does not inherit from or import Person 1's concrete
state class.

```python
class SessionStateView(Protocol):
    @property
    def session_id(self) -> str: ...

    @property
    def turn(self) -> int: ...

    @property
    def mode(self) -> InferredMode: ...

    @property
    def mode_confidence(self) -> float: ...

    @property
    def intent_version(self) -> int: ...

    @property
    def active_constraints(self) -> tuple[Constraint, ...]: ...

    @property
    def revalidation_constraints(self) -> tuple[Constraint, ...]: ...

    @property
    def no_preference(self) -> frozenset[Attribute]: ...

    @property
    def asked_attributes(self) -> frozenset[Attribute]: ...

    @property
    def shown_product_ids(self) -> frozenset[str]: ...

    @property
    def profile_seed(self) -> Mapping[str, object]: ...
```

Person 1 owns the concrete mutable `SessionState`. Person 4 owns the registry
that stores one instance per session. Person 4 never mutates its fields
directly; orchestration calls Person 1-owned reducer methods to apply a delta
and record asked/shown history. Person 2 and Person 3 receive only
`SessionStateView`.

### 3.5 `ProfileBias`

```python
@dataclass(frozen=True)
class ProfileBias:
    terms: tuple[str, ...] = ()
    weight: float = 0.0
```

The weight is configured through held-out ablation, clamped into an accepted
range, and defaults to `0.0`. Explicit dialogue always wins.

### 3.6 `SearchPlan`

```python
@dataclass(frozen=True)
class SearchPlan:
    text_query: str
    must_terms: tuple[str, ...]
    should_terms: tuple[str, ...]
    exclude_terms: tuple[str, ...]
    filters: Mapping[str, object]
    attribute_values: Mapping[Attribute, tuple[object, ...]]
    mode: InferredMode
    intent_version: int
    revalidation_flags: frozenset[Attribute]
    no_preference: frozenset[Attribute]
    profile_bias: ProfileBias
    route_policy: RoutePolicy = "auto"
    embedding_route_id: str | None = None
    index_id: str | None = None
```

Rules:

- The plan represents current validated state, not a raw transcript.
- Exclusions are structured and do not rely on negative words hidden in text.
- Profile bias remains visibly separate from dialogue terms and filters.
- Dense retrieval requires both `embedding_route_id` and `index_id`.
- The selected identities must match the loaded immutable index manifest.
- A mismatch fails the dense route closed and records a route failure; it never
  compares incompatible vectors.

The index manifest, not `SearchPlan`, owns catalog checksum and row count,
ordered-ID checksum, product-text schema version, provider/model/route identity,
dimension, vector dtype, normalization, document count, artifact format, build
time, and artifact checksums.

### 3.7 `ProductEvidence`

```python
@dataclass(frozen=True)
class ProductEvidence:
    matched_fields: tuple[str, ...]
    supporting_snippets: tuple[str, ...]
    constraint_outcomes: Mapping[Attribute, EvidenceOutcome]
    attribute_values: Mapping[Attribute, tuple[object, ...]]
    evidence_reliability: Mapping[Attribute, float]
    unknown_fields: tuple[str, ...]
    route_details: Mapping[str, object]
    profile_contribution: float = 0.0
```

Rules:

- Every supported constraint attribute has tri-state evidence: match,
  contradiction, or unknown.
- Missing metadata is represented as unknown.
- Reliability values are normalized into `[0.0, 1.0]`.
- A hard contradiction requires reliable positive evidence; an absent value is
  not enough.
- Evidence is compact enough for a bounded reranker prompt and contains no
  hidden target or evaluator data.
- Profile contribution remains separately identifiable from dialogue evidence.

### 3.8 `Candidate`

```python
@dataclass(frozen=True)
class Candidate:
    parent_asin: str
    product_evidence: ProductEvidence
    sparse_rank: int | None
    sparse_score: float | None
    dense_rank: int | None
    dense_score: float | None
    structural_score: float
    fused_score: float
```

Candidate construction requires a catalog-valid `parent_asin`. Present ranks
are positive. For fixed inputs and configuration, retrieval orders candidates
by:

1. fused score descending;
2. structural score descending;
3. best present route rank ascending;
4. sparse rank ascending;
5. dense rank ascending;
6. `parent_asin` ascending.

A missing rank sorts after every present rank. Person 3 may rerank this validated
shortlist but may not introduce IDs.

### 3.9 `TurnDecision`

```python
@dataclass(frozen=True)
class TurnDecision:
    action: TurnAction
    ask_attribute: Attribute | None
    reason_code: DecisionReasonCode
    reason: str
    expected_information_gain: float
```

Invariants:

- `clarify` requires one allowed `ask_attribute` and produces no
  recommendations.
- `recommend` requires `ask_attribute is None`.
- Turn 10 cannot choose an unanswered clarification.
- `expected_information_gain` is normalized into `[0.0, 1.0]` and represents
  expected rank-weighted Top-10 membership/order change, not raw entropy.
- Reports group by `reason_code`; they never parse human-readable `reason`.

### 3.10 `Usage`

```python
@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0
    repairs: int = 0
    latency_ms: float = 0.0
    provider: str | None = None
    model: str | None = None
    reasoning_level: str | None = None
    estimated_cost: float | None = None
    cost_currency: str = "USD"
    route: str | None = None
    cache_hit: bool = False
```

Rules:

- Counts, latency, tokens, and estimated cost are non-negative.
- Token fields are token counts, latency is milliseconds, and estimated cost is
  denominated by the ISO-4217 `cost_currency` value.
- `calls` counts actual provider calls, including repair attempts;
  `repairs <= calls`.
- Reasoning tokens remain separately attributable. The official response maps
  provider-billed output tokens into `completion_tokens` according to the
  selected adapter's documented accounting rule.
- A replay-cache hit sets `cache_hit = True` and adds no provider call, tokens,
  latency, or cost for the cached result to the current run.
- Orchestration aggregates each usage record once; evaluation preserves
  component and route attribution.

## 4. Provider-neutral protocols

### 4.1 Interpretation, query, retrieval, decision, and reranking

```python
class IntentInterpreter(Protocol):
    def interpret(
        self,
        message: str,
        state: SessionStateView,
    ) -> tuple[StateDelta, Usage]: ...


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

### 4.2 State mutation boundary

The shared package does not import or name the concrete state type. Person 1
defines the concrete reducer beside `SessionState` in `tikitaka/state/`:

```python
class StateReducer:
    def apply(
        self,
        state: SessionState,
        delta: StateDelta,
        turn: int,
    ) -> SessionState: ...

    def record_decision(
        self,
        state: SessionState,
        decision: TurnDecision,
        shown_product_ids: Sequence[str],
    ) -> SessionState: ...
```

This keeps all state mutation with Person 1 while Person 4 retains registry and
control-flow ownership. The concrete state structurally satisfies
`SessionStateView` when passed to shared consumers.

### 4.3 Embedding boundary

```python
EmbeddingVector = tuple[float, ...]
EmbeddingBatch = tuple[EmbeddingVector, ...]


class Embedder(Protocol):
    @property
    def route_id(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch: ...

    def embed_query(self, text: str) -> EmbeddingVector: ...
```

Person 1 owns provider calls and credentials. Person 2 owns document
preparation, batching, artifact generation, manifest validation, and similarity
search. Provider instrumentation produces the shared `Usage` records; provider
SDK types do not cross this protocol.

## 5. Runtime response normalization

Person 4 enforces these invariants after component execution:

- response contains a string `message`;
- `ask_attribute` is allowed or `null`;
- clarification has no recommendations;
- recommendation has `ask_attribute = null`;
- recommendations contain the first 10 unique catalog-valid IDs in input order;
- optional official usage contains non-negative prompt and completion counts;
- malformed component output and exceptions produce a contract-valid
  deterministic response;
- once local retrieval is available, the deterministic recommendation path is
  preferred over an empty failure response.

## 6. Validation and deterministic failure behavior

- Unknown enum values and attributes are rejected.
- Numeric confidence, reliability, generality, and information-gain values are
  clamped only at the untrusted-input boundary; trusted internal constructors
  reject out-of-range values.
- The state parser retains valid sibling operations and counts rejected ones.
- Catalog validity is checked again at the official response boundary.
- Reranker duplicates and out-of-shortlist IDs are discarded; omitted positions
  are filled from deterministic shortlist order.
- Query/index identity mismatch disables the incompatible dense route and is
  recorded. Sparse/structured fallback may continue.
- No unit test requires network access, the full catalog, or a live model.

## 7. Versioning and change process

- Version `0.1.0` is the frozen P1 implementation target.
- P1 code exports the contract version and structured-output schema version.
- Backward-compatible optional fields increment the minor version.
- Breaking field, invariant, or semantic changes increment the major version.
- Every change includes migration notes, updated fakes, and contract tests.
- Person 4 announces proposed changes before merge and records acknowledgment
  from every affected owner.
- Provider-specific convenience fields remain private unless all affected
  consumers need them.

## 8. Resolved review decisions

The accepted specification resolves every recorded request:

- `StateOperation` carries polarity, strength, and confidence.
- `Constraint` carries normalized value, intent version, lifecycle status, and
  category-dependency provenance.
- Invalid operations are rejected individually and counted on `StateDelta`.
- The concrete reducer takes and returns Person 1's `SessionState`.
- Person 4 owns the registry; Person 1 owns every state mutation method.
- `SessionStateView` is structural and read-only, avoiding a circular import.
- `SearchPlan` uses the field names accepted by Persons 1 and 2 and carries
  dense route/index identity.
- `ProductEvidence` exposes normalized values, tri-state outcomes, and evidence
  reliability.
- `Candidate` includes route ranks/scores and a deterministic tie-break.
- Information gain is normalized and `TurnDecision` has a closed reason code.
- `Usage` records calls, repairs, reasoning tokens, cache state, cost currency,
  and explicit units.
- Fake obligations and file ownership remain with their recorded owners.
