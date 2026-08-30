# Role 2 Build Plan — Catalog Processing and Hybrid Retrieval

## 0. Document purpose

This is the portable implementation authority for **Person 2**. It preserves
the decisions already made for TikiTaka, maps them to the current repository,
and gives enough detail to resume work on another computer without depending on
chat history.

This document is a build plan, not evidence that the planned retrieval system
already exists. Status labels are used deliberately:

- **Verified:** observed in the repository or locally checked against the
  released data.
- **Settled:** an owner decision already recorded in `ARCHITECTURE.md` or
  `AGENTS.md`.
- **Planned:** work that still needs implementation or integration.
- **Provisional:** a useful initial value or interface that may change at the
  M0 contract freeze.
- **Unknown:** a dependency or decision that another owner still needs to
  supply.

The source point for this plan is repository commit `bf42a92` on `main`. The
working branch is `role2/catalog-retrieval`.

## 1. Mission in one sentence

Turn the current structured shopping intent into a high-recall, evidence-rich,
duplicate-free candidate pool by combining BM25, dense embeddings, and reliable
structured metadata over the immutable 50,000-product catalog.

Role 2 optimizes the retrieval boundary of the official score:

1. Get the purchased `parent_asin` into the candidate pool and ultimately the
   scored Top 10 (**Hit Rate@10 / coverage**).
2. Give Person 3 enough evidence to push it toward rank 1 (**MRR / precision**).
3. Expose uncertainty and attribute-distribution diagnostics so the agent asks
   only ranking-changing questions (**MTTC / efficiency**).

Retrieval must be local and reproducible once its artifacts exist. It must not
send all 50,000 products to an LLM.

## 2. System mental model and Role 2 boundary

```text
Official turn input
  -> Person 1: LLM interpretation + validated state delta
  -> Person 1: deterministic active state and query summary
  -> Person 2: sparse + dense + structured retrieval and fusion
  -> Person 3: question-value decision + constraint-aware reranking
  -> Person 4: orchestration + official response validation + evaluation
```

The logic tiers are intentional:

| Tier | Owner | Responsibility |
|---|---|---|
| 0. Contract boundary | Person 4 | Official input/output validation and shared types |
| 1. Intent/state | Person 1 | Understand the message and deterministically update isolated session state |
| 2. Search plan | Person 1 + shared contract | Represent only the currently active intent, including replacements and exclusions |
| 3. Candidate retrieval | **Person 2** | BM25, dense embeddings, structured evidence, fusion, diagnostics |
| 4. Ask-or-rank policy | Person 3 | Decide whether clarification is worth a turn; rerank the shortlist |
| 5. Evaluation/release | Person 4 | Run the official lifecycle, report metrics, consolidate milestones |

### Role 2 owns

- catalog loading, validation, normalization, and immutable product documents;
- searchable text construction and versioning;
- sparse/BM25 indexing and search;
- structured metadata extraction, tri-state constraint evidence, filters, and
  boosts;
- dense product-index building, loading, query-vector search, and manifest
  validation;
- rank fusion and per-route evidence;
- retrieval diagnostics used by the generality/question-value sensor;
- Role 2 fixtures, correctness tests, index-build scripts, and documentation.

### Role 2 does not own

- conversation history or session registries;
- LLM intent extraction and state mutation;
- whether a turn is `CLARIFY` or `RECOMMEND`;
- final shortlist reranking or same-intent recommendation policy;
- official Agent payload normalization;
- ground-truth labels, scenario labels, or evaluator behavior;
- API credentials or direct provider SDK integration;
- cross-session user profiles or identity.

Role 2 should be **stateless per request**: a fixed search plan, fixed indexes,
and fixed configuration must produce the same retrieval result. This property
makes intent overrides safe. The retriever sees the newly reduced active state,
not an ever-growing concatenation of prior user messages.

## 3. Competition and project rules that constrain this work

### Verified official boundaries

- Catalog: frozen 50,000-product Amazon Clothing, Shoes & Jewelry catalog.
- Session: maximum 10 turns.
- Scoring: only the first 10 unique catalog-valid `parent_asin` values count.
- Hit: exact `parent_asin` equality only.
- Scenarios: 40% Buying, 40% Browsing, 15% Intent Override, 5% Boundary.
- Agent-visible inputs: `session_id`, anonymized `user_profile`, user message,
  turn, and `top_k`.
- The Agent must never receive `ground_truth`, `scenario_type`, hidden intent
  cards, or simulator state.
- Dense, sparse, and hybrid retrieval are allowed.
- A heavyweight external vector database is out of scope. An in-process index
  or local generated artifact is acceptable.
- The catalog cannot be modified to improve results.
- Final scoring may disable network access.

### Settled project policy

- The primary generative route is the main API using `gpt-5.6-terra` with
  `medium` reasoning.
- Do not add Qwen, MLX, or another local generative LLM.
- The LLM is used for intent understanding, state-delta extraction, query
  rewriting, clarification planning, and bounded-shortlist semantic reranking.
- Deterministic code validates all model output and owns state mutation,
  catalog validity, constraints, turn limits, and output normalization.
- Retrieval is hybrid: sparse/BM25 + dense embeddings + structured evidence.
- Model selection routes automatically at runtime, but evaluation can pin every
  route. A query vector must never be searched against an index produced by a
  different embedding route.
- Accuracy, exact rank, and fewer questions are the initial priorities. Latency,
  tokens, and cost are still logged.
- Every response chooses one action:
  - `CLARIFY`: one valid `ask_attribute`, no recommendations;
  - `RECOMMEND`: recommendations, `ask_attribute = null`.
- A supplied profile is a soft, decaying, session-local input only. It cannot
  override an explicit conversation constraint or become cross-session memory.
- Intent changes use dependency-aware clearing rather than indiscriminately
  retaining or deleting all constraints.

### Why intent handling matters to retrieval

The current query is dependent on the current state. If a user changes from
“red leather boots” to “actually running shoes, any color,” retrieval must not
continue using red, leather, or boots. Person 1's reducer creates the corrected
active state; Role 2 searches only that state.

The settled state operations are:

- `ADD`
- `REMOVE`
- `REPLACE`
- `EXCLUDE`
- `NO_PREFERENCE`
- `RESET`

Clearing rules:

1. Direct attribute correction replaces only that attribute.
2. Explicit “start over” clears conversation-derived constraints and begins a
   new intent version.
3. Major category/product-type change begins a new intent version, removes
   incompatible/category-derived constraints, and preserves still-applicable
   universal constraints such as budget.
4. Ambiguous surviving constraints are marked for revalidation and must not be
   treated as certain hard filters.
5. Profile hints remain separate soft evidence; they are not dialogue state.

Role 2 must therefore accept `intent_version` and active constraints in the
search plan, but it must not implement a second state reducer.

## 4. Verified current repository and data state

### Repository state at plan creation

- `starter/agent.py` is a weak, stateless, in-memory SQLite FTS5/BM25 agent.
- `starter/agent.py` indexes title, categories, features, details, store, and
  description with field weights `6.0, 4.0, 2.5, 2.5, 1.5, 1.0`.
- The baseline searches only the current raw user message and has no dense,
  structured-state, clarification, or reranking layer.
- Official tests live in `tests/test_evaluator.py` and currently cover evaluator
  normalization, miss-turn calculation, and hidden-field materialization.
- The released baseline scores Hit Rate@10 `0.125`, MRR `0.068034`, and MTTC
  `9.81` on the public set.
- No internal `tikitaka/` package or Role 2 implementation exists yet.
- `data/catalog.jsonl` is intentionally ignored by Git.

### Locally verified data facts

| Fact | Value |
|---|---:|
| Catalog rows | 50,000 |
| Unique catalog `parent_asin` values | 50,000 |
| Public sessions | 200 |
| Buying / Browsing / Override / Boundary | 80 / 80 / 30 / 10 |
| Public targets present in catalog | 200 / 200 |
| Non-empty title | 49,998 |
| Non-empty features | 44,781 |
| Non-empty description | 26,113 |
| Non-empty categories | 50,000 |
| Non-empty details | 48,330 |
| Non-empty store | 49,686 |
| Non-empty price | 10,527 |
| Non-empty rating fields | 50,000 |

Implications:

- Missing price is common. A missing price means **unknown**, not “over budget.”
- Description is absent for nearly half the catalog, so dense text cannot depend
  on it alone.
- Title and category are the most consistently available semantic anchors.
- `details` keys are heterogeneous and must be normalized without assuming one
  fixed schema.
- Public profiles are populated but contain no stable user ID. They cannot be
  joined across sessions.

The catalog record fields observed in the official specification are:

```text
parent_asin, title, features, description, price, categories, details,
average_rating, rating_number, store
```

## 5. Team and collaboration model

| Person | Handle | Workstream |
|---|---|---|
| 1 | `hongyuuuuu` | API/model gateway, LLM interpretation, state reducer, query builder |
| 2 | **current owner; GitHub handle not yet recorded** | Catalog processing and hybrid retrieval |
| 3 | `azora04` | Clarification policy, constraint-aware reranking, final selection |
| 4 | `joelyrk` | Shared contracts, orchestration, evaluator integration, milestone consolidation |

Branch convention:

```text
role1/<work>
role2/<work>
role3/<work>
role4/<work>
```

Everyone has merge authority. Person 4 coordinates milestone integration and
merge order; Person 4 is not the only person permitted to merge. Assume Person
4 will implement the assigned integration work normally.

### Role 2 dependency map

| Dependency | Owner | Can Role 2 start without it? | Integration requirement |
|---|---|---:|---|
| Catalog schema and released data | Official repository | Yes; verified | Do not mutate or commit the catalog |
| Shared `SearchPlan` / `Candidate` contracts | Person 4 + all owners | Partly | Start internals, then adapt at M0 freeze |
| Active state/query summary | Person 1 | Yes, using fixtures | Agree normalized constraint semantics before integration |
| Embedding API adapter and credentials | Person 1 | Sparse work: yes; dense live calls: no | Depend on a provider-neutral embedding protocol |
| Candidate evidence required by reranker | Person 3 | Yes, with proposed schema | Review evidence payload before M2 |
| Orchestrator/evaluation harness | Person 4 | Unit work: yes | Needed for milestone score experiments |

API and credential work is explicitly dependent on Person 1. Role 2 must not
copy credentials, call a provider SDK directly, or block sparse retrieval work
while waiting for the API.

## 6. Proposed shared contracts for M0

Person 4 owns the final dependency-light shared contract module. The following
is Role 2's proposed shape and should be reviewed by Persons 1, 3, and 4 before
it is frozen. Names are provisional; semantics are not.

### 6.1 Search input

```python
@dataclass(frozen=True)
class SearchPlan:
    query_text: str
    mode: Literal["buying", "browsing", "unknown"]
    intent_version: int
    constraints: tuple[Constraint, ...]
    no_preference: frozenset[str]
    profile_terms: tuple[str, ...] = ()
    profile_weight: float = 0.0
    route_policy: Literal["auto", "sparse", "dense", "hybrid"] = "auto"
    pinned_embedding_route: str | None = None
```

Rules:

- `query_text` is an active-state summary, not the entire raw transcript.
- Explicit active constraints are separate from profile terms.
- `profile_weight` is clamped to an evaluated configuration and defaults to 0.
- Exclusions are represented by constraint polarity, not negative words hidden
  inside `query_text`.
- `intent_version` participates in logging and cache keys.
- A pinned embedding route selects both its query embedder and matching index.

### 6.2 Normalized catalog product

```python
@dataclass(frozen=True)
class ProductDocument:
    parent_asin: str
    title: str
    categories: tuple[str, ...]
    features: tuple[str, ...]
    details: tuple[tuple[str, str], ...]
    description: tuple[str, ...]
    store: str
    price: Decimal | None
    average_rating: float | None
    rating_number: int | None
    sparse_fields: Mapping[str, str]
    dense_text: str
    structured: "StructuredProductEvidence"
```

Keep the raw catalog untouched. Normalization creates derived immutable
documents and index artifacts.

### 6.3 Candidate output

```python
@dataclass(frozen=True)
class Candidate:
    parent_asin: str
    sparse_rank: int | None
    sparse_score: float | None
    dense_rank: int | None
    dense_score: float | None
    structural_score: float
    fused_score: float
    product_evidence: "ProductEvidence"

@dataclass(frozen=True)
class RetrievalResult:
    candidates: tuple[Candidate, ...]
    diagnostics: "RetrievalDiagnostics"
    index_manifest_ids: tuple[str, ...]
```

Evidence must be compact enough for Person 3's bounded reranker prompt but rich
enough to explain why each route found a product.

Suggested `ProductEvidence` fields:

- title and leaf categories;
- price when known;
- matched sparse terms by field;
- structured constraint outcomes: `match`, `contradiction`, or `unknown`;
- compact feature/detail snippets supporting each match;
- sparse/dense ranks and scores;
- profile contribution kept visibly separate from dialogue contribution.

### 6.4 Retrieval protocol

```python
class Retriever(Protocol):
    def retrieve(self, plan: SearchPlan, limit: int) -> RetrievalResult:
        ...
```

The method has no session mutation. Recommendation history is a Person 3 policy
input, not hidden retriever state. When `intent_version` changes, Person 3 can
make prior products eligible again without rebuilding retrieval indexes.

### 6.5 Embedding boundary

```python
class Embedder(Protocol):
    @property
    def route_id(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch: ...
    def embed_query(self, text: str) -> EmbeddingVector: ...
```

Person 1 owns provider request/credential handling and automatic model routing.
Role 2 owns document preparation, batching strategy, artifact generation,
manifest checks, and similarity search. Person 4 places the final protocol in
the shared contract package.

## 7. Planned repository layout

Do not create all files empty. Add each file only when its milestone begins.

```text
tikitaka/
  retrieval/
    __init__.py
    catalog.py            # stream/load/validate immutable products
    text.py               # sparse fields and versioned dense text
    structured.py         # extraction and tri-state constraint evidence
    sparse.py             # SQLite FTS5/BM25 build/load/search
    dense.py              # vector artifact load and similarity search
    manifests.py          # catalog/text/model/index compatibility checks
    fusion.py             # RRF and route evidence
    hybrid.py             # public stateless Retriever implementation
scripts/
  build_sparse_index.py
  build_dense_index.py
  inspect_retrieval.py
tests/
  fixtures/
    catalog_small.jsonl
  test_catalog.py
  test_retrieval_text.py
  test_structured_retrieval.py
  test_sparse_retrieval.py
  test_dense_retrieval.py
  test_fusion.py
  test_hybrid_retrieval.py
```

Person 4 may choose a different shared package root during M0. Role 2 should
keep imports shallow so moving shared contracts is mechanical.

Do not edit:

- `data/public_set.jsonl`;
- `evaluator/`;
- official contract/specification/config files;
- ground-truth labels;
- the downloaded catalog.

Coordinate before editing:

- `starter/agent.py`;
- `AGENTS.md`, `ARCHITECTURE.md`, or the shared implementation plan;
- dependency manifests;
- shared contracts.

## 8. Detailed implementation design

### 8.1 Catalog loader and validation

Implement a streaming JSONL reader first. It must:

1. reject blank/malformed records with a useful line number;
2. require a non-empty `parent_asin`;
3. detect duplicate IDs;
4. normalize optional scalar/list/dict fields without changing the source file;
5. distinguish missing, null, and present values where constraint safety needs
   that distinction;
6. expose a stable product map by `parent_asin` plus an ordered sequence;
7. record row count and source-file checksum in derived artifact manifests.

Use small helpers with one defined behavior for dicts and lists. Do not rely on
Python's default `str(dict)` because ordering/format changes would silently
invalidate embeddings.

Initial correctness gates:

- tiny fixture loads expected records;
- malformed JSON reports its line;
- empty/missing/duplicate ID fails;
- official catalog loads 50,000 unique IDs;
- loading never writes into `data/catalog.jsonl`.

### 8.2 Versioned searchable text

Sparse fields and dense text serve different purposes.

Sparse/BM25 fields should stay separate so field weights remain configurable:

```text
title | categories | features | details | store | description
```

Dense text should be a deterministic labeled representation, for example:

```text
TITLE: Columbia Men's Thistletown Park Crew
CATEGORY: Men > Clothing > Shirts > T-Shirts
BRAND_OR_STORE: Columbia
FEATURES: 67% polyester, 33% cotton | UPF 15 | moisture wicking | ...
DETAILS: department: mens | manufacturer: Columbia
DESCRIPTION: ...
PRICE: 27.99
```

Requirements:

- normalize whitespace and Unicode consistently;
- preserve meaningful numbers, sizes, materials, and hyphenation;
- cap each verbose field deterministically before embedding;
- sort heterogeneous detail keys only if source order is not part of the
  declared text version;
- publish a text schema ID such as `product_text_v1`;
- rebuild dense artifacts whenever that version changes.

Do **not** generate the 50,000 product embeddings until `product_text_v1` and
the embedding route are settled. Otherwise a small text change forces an
expensive rebuild and makes query/index compatibility ambiguous.

### 8.3 Sparse/BM25 route

Start by extracting and improving the verified SQLite FTS5 baseline. It uses
only Python's standard library and has already indexed the full catalog.

Initial implementation:

- one FTS5 table with one row per `parent_asin`;
- explicit column weights, starting from the baseline values;
- safe query construction from the structured active query;
- deduplicated, normalized terms with a documented cap;
- bounded route depth; **provisional** `sparse_k = 200`;
- deterministic tie-break by `parent_asin` after route score/rank;
- returned rank, raw BM25 score, and matched-field evidence;
- optional generated on-disk SQLite artifact after the in-memory slice works.

Do not concatenate previous turns. Search the current `SearchPlan.query_text`
plus active positive constraints. Handle negative constraints in structured
evidence rather than inserting negated terms into an OR query.

The starter's weights are a baseline, not a final truth. Weight changes belong
to post-vertical-slice experiments and must be reported with held-out retrieval
coverage.

### 8.4 Structured evidence and safe constraints

Create a normalized evidence record with provenance. Suggested attributes:

- category/product type;
- material;
- color;
- size/fit;
- brand/store;
- price/budget;
- style;
- feature;
- use case.

Every constraint comparison returns one of:

```text
MATCH          reliable evidence supports the constraint
CONTRADICTION  reliable evidence conflicts with the constraint
UNKNOWN        the catalog does not establish either outcome
```

Safety rules:

- Hard-filter only a reliable `CONTRADICTION` against an explicit hard
  conversation constraint.
- Never filter a missing field as a contradiction.
- Missing price cannot fail a budget constraint; it remains `UNKNOWN` and may
  receive no price-match boost.
- Profile preferences are always soft boosts and never hard filters.
- A user exclusion such as “not leather” is a contradiction only when reliable
  evidence positively identifies leather.
- Ambiguous constraints flagged for revalidation cannot hard-filter.
- Preserve the source field/snippet for audit and Person 3 explanations.

Prefer high-precision extractors first: normalized category path, known price,
store/manufacturer, exact material/color terms, and explicit size/detail keys.
Broader semantic feature/use-case matching can use the dense route rather than
fragile regex expansion.

### 8.5 Dense embedding route

Dense embeddings are required in the planned hybrid system. They complement
BM25 when user language is a paraphrase of catalog language.

Sequence:

1. Freeze `product_text_v1`.
2. Receive one working embedding route through Person 1's provider-neutral
   adapter. Do not begin with a multi-model benchmark.
3. Embed a tiny fixture and validate dimensions/normalization.
4. Build the 50,000-product artifact in deterministic batches with resumable
   progress and explicit failure reporting.
5. Store vectors in a local array plus an ID mapping and manifest.
6. Implement exact cosine search first; benchmark memory and latency.
7. Add a lightweight in-process ANN index only if exact search is materially
   too slow. Do not add an external vector service.
8. Embed the active search query through the exact matching route.
9. Return **provisional** `dense_k = 200` with cosine score and rank.

An embedding “candidate” means an embedding model/route to compare later—not a
product candidate. Model comparisons are postponed until one full path works.

Required manifest fields:

```text
artifact_format_version
catalog_source_sha256
catalog_row_count
ordered_parent_asin_sha256
product_text_schema_version
embedding_provider
embedding_model
embedding_route_id
embedding_dimension
vector_dtype
normalization
document_count
build_timestamp
artifact_checksums
```

The loader must fail closed on manifest mismatch. It must never quietly combine
a query vector and product matrix from different routes.

Generated embeddings/indexes should remain ignored by Git unless the team later
chooses a submission-safe artifact distribution mechanism. The build script and
manifest schema are committed; secrets and large generated files are not.

### 8.6 Hybrid fusion

Begin with Reciprocal Rank Fusion because BM25 and cosine scores are not on the
same scale:

```text
RRF(product) = sum(route_weight / (rrf_k + route_rank))
```

Use stable deterministic ordering. **Provisional** initial values:

```text
rrf_k = 60
sparse_weight = 1.0
dense_weight = 1.0
fused_candidate_limit = 100
```

Structured evidence can:

- remove only definite hard contradictions;
- boost explicit reliable matches;
- apply smaller boosts for soft dialogue preferences;
- apply a separately logged, decaying profile contribution;
- leave unknown metadata neutral.

Buying mode should initially favor explicit constraint precision. Browsing mode
should retain broader semantic coverage and candidate diversity. Route weights
must be configuration, not scattered conditionals.

Every fused candidate retains its per-route ranks and evidence. Person 3 must
be able to distinguish “found by both routes” from “found only semantically.”

### 8.7 Retrieval diagnostics for clarification policy

Person 3 needs more than a list. Return diagnostics such as:

- total candidates before/after definite contradiction filtering;
- sparse and dense result counts;
- route overlap at 10/50/100;
- top score/rank margin and score concentration;
- route disagreement on the leading products;
- candidate distributions for known material, color, category, size, brand,
  budget band, and other reliable attributes;
- missing-metadata rate per attribute among competitive candidates;
- active constraint match/unknown/contradiction counts;
- retrieval route, model, and manifest identifiers;
- elapsed time per route.

Do not decide the question in Role 2. Provide the evidence that lets Person 3
estimate whether knowing an attribute can change Top-10 membership or ordering.

### 8.8 Profile signal

The official profile is allowed, but the project policy is deliberately weak:

- Person 1 keeps the supplied snapshot separate from explicit state.
- Role 2 may accept normalized `profile_terms` as a distinct soft route or
  contribution.
- The initial and fallback weight is `0.0`.
- Any nonzero weight must be selected by a held-out ablation against weight 0.
- Explicit user constraints always win.
- Weight decays as the conversation becomes more specific.
- No profile is persisted, joined, or inferred across session IDs.

The public set has no stable user identity, so cross-session personalization is
both technically unsupported and contrary to the isolation rule.

### 8.9 Intent changes and recommendation duplication

Retrieval itself should return unique `parent_asin` values. It must not keep a
hidden “already shown” session set.

Person 3 owns same-intent display deduplication using
`shown_by_intent[intent_version]`:

- within one intent version, prior recommendations are excluded or strongly
  penalized so the Top 10 is not wasted on repeats;
- after a confirmed intent change creates a new version, those products become
  eligible again if they match the new state.

Role 2 supports this by returning a reproducible broad pool and accepting only
the current active intent. If a future shared contract passes display exclusions
to retrieval for efficiency, they must be explicit function inputs and included
in diagnostic logs—not mutable retriever memory.

### 8.10 Network-degraded path

Primary development assumes the API works. Failure engineering is not the
first milestone, but final scoring may be offline.

Role 2's deterministic degraded behavior is:

- load local catalog and sparse/structured indexes;
- skip dense query search when the required query embedder is unavailable;
- return valid sparse + structured candidates and diagnostics;
- never fail the whole turn because an embedding call failed;
- report the route failure to orchestration without leaking secrets.

Do not add a local generative LLM. A later locally executable embedding model is
an evaluation option only if the model-selection plan explicitly chooses it;
it is not required for the initial slice.

## 9. Milestone execution plan

### M0 — Contract freeze and reproducible baseline

**Goal:** agree on the smallest interfaces that let all four people work in
parallel.

Role 2 tasks:

1. Review the proposed `SearchPlan`, `Candidate`, `RetrievalResult`, and
   `Embedder` semantics with Persons 1, 3, and 4.
2. Agree which module owns each shared type and how contract versions change.
3. Create a tiny synthetic catalog fixture covering:
   - exact lexical match;
   - paraphrase-only semantic match;
   - known and missing price;
   - explicit material/color;
   - exclusion contradiction;
   - missing metadata;
   - duplicate-ID validation.
4. Reproduce official tests and the untouched BM25 baseline.
5. Record Python version, catalog row count, and checksums.

Handoff to Person 4:

- fixture path and schema;
- proposed contracts and compatibility notes;
- baseline command/result;
- no dependency additions yet.

Exit gate:

- shared contracts accepted;
- official tests pass unchanged;
- tiny fake end-to-end orchestration can consume a fake retrieval result.

### M1 — Deterministic lexical + structured vertical slice

**Goal:** make the full agent work without waiting for embeddings or live APIs.

Role 2 tasks:

1. Implement catalog loader and immutable product map.
2. Implement deterministic sparse field/text construction.
3. Extract the existing FTS5 baseline into `SparseRetriever`.
4. Search from a fixture `SearchPlan`, not a raw transcript.
5. Implement high-confidence structured evidence and tri-state comparison.
6. Combine sparse rank with structured filtering/boosts.
7. Return compact evidence and diagnostics through the accepted contract.
8. Provide a fake/config-pinned retriever to Person 4.

Correctness checks during implementation:

- full catalog is 50,000 unique IDs;
- exact lexical test product is retrieved;
- a known over-budget item is contradictory;
- a missing-price item is unknown, not filtered;
- explicit negative material behaves correctly;
- results contain no duplicate/invalid IDs;
- fixed input returns stable ordering.

Integration exit:

```text
message -> fake/heuristic active state -> sparse/structured retrieval
        -> deterministic decision/rank -> official response
```

No metric tuning is required before this vertical slice runs.

### M2 — Dense retrieval and hybrid fusion

**Goal:** improve semantic catalog coverage while preserving exact reproducibility.

Role 2 tasks:

1. Freeze and test `product_text_v1`.
2. Integrate one embedding route from Person 1.
3. Build/validate fixture embeddings and matching manifest.
4. Build the full 50,000-product artifact.
5. Implement exact in-memory cosine search.
6. Implement manifest mismatch failures.
7. Add RRF and route-specific diagnostics.
8. Expose configuration pins for sparse-only, dense-only, and hybrid.
9. Hand artifact build/load instructions to Person 4.

Integration with Person 3:

- confirm candidate evidence fits the deterministic and LLM rerankers;
- confirm route disagreement and attribute distributions are sufficient for the
  generality sensor;
- keep the LLM prompt shortlist bounded.

Exit evidence, produced after integration:

- sparse-only vs dense-only vs hybrid retrieval coverage on the held-out split;
- per-scenario retrieval coverage;
- route overlap and target rank before final reranking;
- artifact manifest and reproduction command.

### M3 — Primary LLM vertical slice

**Goal:** connect live intent/query interpretation and semantic reranking to the
retrieval engine.

Role 2 tasks:

1. Accept Person 1's validated active query/state for Buying, Browsing,
   Override, and Boundary traces.
2. Confirm intent replacement causes a fresh retrieval from corrected state.
3. Supply compact evidence for Person 3's LLM reranker.
4. Keep model/provider SDKs outside retrieval modules.
5. Log query embedding route, index manifest ID, route timing, and candidate
   counts for Person 4.
6. Confirm embedding/API failure degrades to sparse + structured output.

Exit gate:

- representative live API sessions produce contract-valid candidate pools;
- no secret or full catalog enters an LLM prompt;
- stale constraints disappear after override;
- no mismatched embedding route/index is possible.

### M4 — Policy and score experiments

**Goal:** choose configurations with evidence, after a full path exists.

Testing chronology is important:

1. **During implementation:** small correctness tests only.
2. **At milestone consolidation:** integration tests against the vertical slice.
3. **After the vertical slice works:** public/held-out performance experiments.

Role 2 experiments:

- baseline FTS5 weights vs revised field weights;
- sparse-only vs dense-only vs hybrid;
- one embedding route vs later candidate routes, if time and budget permit;
- RRF depths, `rrf_k`, and route weights;
- hard contradiction policy vs boost-only policy for unreliable attributes;
- candidate-pool depths into deterministic and LLM reranking;
- Buying/Browsing route weights;
- profile weight `0` vs a soft decaying weight;
- exact cosine vs ANN only if latency warrants it.

Every experiment records:

- code commit and configuration ID;
- catalog/text/index manifest IDs;
- aggregate Hit Rate@10, MRR, MTTC, Efficiency, TechnicalScore;
- scenario-specific metrics;
- retrieval target coverage/rank before final rerank;
- latency and any embedding cost;
- failures, missing artifacts, and route decisions.

Do not tune on all 200 public sessions. Person 4 owns a stable tuning/held-out
split. Report both and prefer held-out decisions.

### M5 — Deterministic no-network contingency

Role 2 tasks:

1. Force embedding/API unavailability.
2. Confirm sparse + structured artifact loading from a clean environment.
3. Confirm every query still returns only valid catalog IDs.
4. Measure and document the quality delta against primary hybrid retrieval.
5. Make missing/corrupt index failures explicit and recoverable.

Exit gate: the official evaluator completes without network and without a local
generative LLM.

### M6 — Submission and demo

Role 2 tasks:

1. Freeze index/text/config versions.
2. Verify index reproduction from the documented catalog source.
3. Document artifact sizes, memory, build time, query time, API requirements,
   and degraded behavior.
4. Supply retrieval traces for:
   - vague trip-shoe request becoming a useful search plan;
   - explicit constraints narrowing the pool;
   - intent override clearing stale dependencies;
   - Boundary/no-preference behavior;
   - one sparse-only and one hybrid comparison.
5. Confirm no catalog, secret, temporary result, or oversized generated artifact
   is accidentally committed.

## 10. Role 2 test plan

Tests are part of implementation correctness; score experiments happen only
after integration.

### Unit tests

- catalog loader handles normal/missing/malformed fields;
- exactly one immutable record per unique ID;
- deterministic sparse and dense text construction;
- manifest changes when catalog/text/model/dimension/normalization changes;
- exact lexical search;
- structured category/material/color/brand/budget matches;
- missing price/description produces `UNKNOWN` safeguards;
- dense paraphrase match on a fake embedding fixture;
- normalized query/product vector consistency;
- stable RRF order with deterministic ties;
- duplicate and invalid ID rejection;
- mode/profile weights remain configuration inputs;
- no profile hard-filter can override explicit dialogue state.

### Contract tests

- accepts the shared `SearchPlan` from Person 1;
- returns Person 3-compatible `Candidate` evidence;
- route failure remains a valid `RetrievalResult` with diagnostics;
- pinned embedding route rejects the wrong index;
- candidate limit is respected;
- retriever has no cross-session mutable state.

### Integration tests

- official Agent lifecycle through Person 4's orchestrator;
- Buying/Browsing use visible state rather than `scenario_type`;
- direct replacement removes stale retrieval constraints;
- category override increments intent and re-retrieves;
- `NO_PREFERENCE` does not create a false negative filter;
- same-intent display dedup occurs in Person 3, while a new intent restores
  eligibility;
- network-off path completes with sparse/structured results;
- official evaluator and data remain unchanged.

### Performance/evaluation reports

- target recall at candidate K before reranking;
- target rank distribution before/after fusion;
- official Hit Rate@10, MRR, and MTTC after integration;
- per-scenario breakdown;
- build/query latency and memory;
- tokens/cost for embedding calls when applicable;
- profile and route ablations.

## 11. Risks and mitigations

| Risk | Effect | Mitigation |
|---|---|---|
| Text/model/index mismatch | Dense results become meaningless | Strict route + manifest validation; fail closed |
| Missing price/description | Valid targets are over-filtered | Tri-state evidence; unknown is neutral |
| Stale conversational text | Override queries retain old intent | Retrieve only from reduced active `SearchPlan` |
| Duplicate recommendations | Wastes scored Top-10 positions | Unique candidate IDs; Person 3 tracks shown IDs per intent |
| Overfitting 200 public targets | Public gain fails private evaluation | Stable held-out split and per-scenario reports |
| Embedding API unavailable | Dense route cannot embed query | Sparse + structured deterministic degraded path |
| API/provider coupling | Blocks testing and route changes | Shared protocol; fake embedder; no SDK in retrieval |
| Large generated artifacts | Repository/submission becomes unusable | Ignore artifacts; commit scripts/manifests; document distribution |
| Broad regex extraction | False contradictions remove target | High-precision evidence first; hard-filter only reliable conflicts |
| Role 2/P3 policy overlap | Conflicting dedup/ranking behavior | Role 2 returns evidence; Person 3 owns final policy |
| Contract drift | Four branches stop integrating | Small shared types, migration notes, milestone consolidation |
| Premature benchmarking | Time spent optimizing disconnected code | Correctness first, vertical slice second, experiments third |

## 12. Fresh-computer bootstrap

### 12.1 Clone and select the branch

```bash
git clone https://github.com/hongyuuuu/tikitaka.git
cd tikitaka
git fetch --all --prune
git switch role2/catalog-retrieval
git remote -v
```

Expected remotes:

```text
origin    https://github.com/hongyuuuu/tikitaka.git
upstream  https://github.com/TechJam2026/techjam-conversational-search.git
```

If `upstream` is absent:

```bash
git remote add upstream https://github.com/TechJam2026/techjam-conversational-search.git
```

### 12.2 Python environment

The verified starter uses only the standard library and requires Python 3.10+
at minimum.

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m unittest
```

Do not invent a `pip install` command until a dependency manifest is committed.
When one exists, install exactly from that manifest.

### 12.3 Download the ignored catalog

```bash
curl -L -O https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz
shasum -a 256 catalog.jsonl.gz
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

The previously verified published SHA-256 for `catalog.jsonl.gz` is:

```text
07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8
```

Also compare it with the release's `SHA256SUMS`. Then verify:

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("data/catalog.jsonl")
count = 0
identifiers = set()
with path.open(encoding="utf-8") as handle:
    for line in handle:
        product = json.loads(line)
        count += 1
        identifiers.add(str(product["parent_asin"]))
print({"rows": count, "unique_parent_asin": len(identifiers)})
PY
```

Expected result:

```text
{'rows': 50000, 'unique_parent_asin': 50000}
```

`data/catalog.jsonl` is ignored by Git. Do not force-add it.

### 12.4 Establish a clean baseline

```bash
git status --short --branch
python3 -m unittest
python3 -m evaluator.local_evaluator
```

The evaluator writes `results.json`, which is a generated local result and
should not be committed unless the team explicitly establishes a documented
results-artifact policy.

### 12.5 Secrets and API configuration

Role 2's M1 work needs no secret. For dense API integration, obtain environment
variable names and setup from Person 1. Transfer secrets through an approved
secret manager or manual environment setup, never through Git, this plan, chat
logs committed to the repository, or copied `.env` files.

## 13. First implementation session

The first coding block can begin immediately; no live API or Person 4 code is
required for the following tasks.

1. Sync with latest `main` and confirm `role2/catalog-retrieval` is clean.
2. Ask Person 4 to schedule the M0 contract review, but do not wait to implement
   internal catalog parsing behind a narrow adapter.
3. Add the tiny catalog fixture.
4. Implement `catalog.py` with streaming validation and immutable documents.
5. Add loader unit tests, including duplicate/missing IDs and null fields.
6. Implement deterministic `text.py` plus `product_text_v1` tests.
7. Extract the starter FTS5 behavior into `sparse.py` without changing weights.
8. Prove the fixture and full 50,000-row catalog can be indexed and queried.
9. Commit the smallest working slice and send the contract/evidence questions
   below to Persons 1, 3, and 4.

Do not start the full embedding build in this session unless the text version,
embedding route ID, normalization, and artifact manifest are settled.

## 14. Required cross-owner questions at M0

These are integration questions, not blockers for loader/BM25 work.

### For Person 1 (`hongyuuuuu`)

- What exact normalized `SearchPlan` fields will the state/query layer emit?
- Will explicit exclusions and ambiguous constraints be structured separately?
- What provider-neutral `Embedder` API and route ID will be supplied?
- Which environment variable names and usage metadata shape are authoritative?
- How will automatic embedding routing be pinned during evaluation?

### For Person 3 (`azora04`)

- What maximum shortlist size and evidence budget can the rerankers accept?
- Which retrieval diagnostics are required for question-value estimation?
- Does final dedup use exclusion or a large penalty within one intent version?
- Which structured outcomes must be exposed for hard-constraint enforcement?

### For Person 4 (`joelyrk`)

- Where will shared contracts live and how are they versioned?
- What stable tuning/held-out split and experiment config format will be used?
- Where should generated index artifacts live locally and in submission setup?
- What are milestone consolidation times and merge order?
- Which dependency additions are acceptable for vector math/indexing?

### Still unknown

- Person 2's GitHub handle for the ownership table.
- The initial embedding provider/model/route ID.
- Whether exact NumPy cosine is sufficient or a lightweight ANN library is
  necessary.
- Final artifact distribution method if prebuilt embeddings are required.
- The dependency/environment manifest Person 4 will select.

Unknowns must remain labeled rather than silently invented.

## 15. Commit, handoff, and consolidation protocol

Prefer small reviewable commits:

```text
docs: add portable role 2 build plan
feat(retrieval): add catalog loader and product documents
test(retrieval): cover catalog validation and missing metadata
feat(retrieval): extract deterministic sparse retrieval
feat(retrieval): add structured product evidence
feat(retrieval): add dense index manifests and search
feat(retrieval): fuse sparse dense and structured routes
```

Before pushing any implementation commit:

```bash
python3 -m unittest
git diff --check
git status --short
git push -u origin role2/catalog-retrieval
```

Each milestone handoff should contain:

```text
Branch/commit:
Contract version:
What is implemented:
What remains planned:
Commands run and results:
Generated artifacts/manifests:
Configuration:
Known limitations:
Inputs needed from another owner:
Integration/migration notes:
```

At each milestone, all four owners consolidate. Person 4 coordinates the
integration branch/status and merge order. Anyone merging a shared change must
announce it and preserve the exact test/evaluation evidence.

## 16. Definition of done for Role 2

Role 2 is complete only when:

- the frozen catalog loads as 50,000 unique immutable products;
- text construction and index manifests are deterministic and versioned;
- sparse, dense, and structured routes work through provider-neutral contracts;
- query embedding and product index compatibility is enforced;
- hybrid retrieval returns only unique, valid catalog IDs;
- every candidate carries compact evidence and per-route ranks;
- missing metadata cannot silently remove a valid target;
- retrieval uses the corrected active state after intent changes;
- no cross-session profile or recommendation memory exists in retrieval;
- Person 3 receives the diagnostics needed for clarification and reranking;
- sparse-only, dense-only, and hybrid routes can be pinned and ablated;
- the full-catalog hybrid route runs locally from documented artifacts;
- the network-degraded sparse/structured route remains valid;
- correctness, integration, held-out, and per-scenario evidence is recorded;
- secrets, the catalog, and large generated artifacts are absent from Git;
- setup and reproduction succeed on a clean second computer.

## 17. Decision recap

- We are solving Challenge 4, not building a generic shopping UI.
- Dense embeddings are part of the planned industry-grade hybrid architecture.
- The generative LLM is API-only: main `gpt-5.6-terra`, `medium`.
- Automatic runtime routing is planned; benchmark runs can pin routes.
- The active state is structured because users add, retract, replace, exclude,
  remove preference, reset, and change their intent across turns.
- Ask and recommend are mutually exclusive project actions.
- The profile is an allowed session-local snapshot, not persistent identity; it
  is a soft decaying signal and must beat weight 0 on held-out evaluation.
- Each evaluator session is isolated even though one Agent instance may serve
  many sessions; isolation means no state leakage, not necessarily a new Python
  process.
- Role 2 can start now on catalog loading, normalization, BM25, structured
  evidence, fixtures, and contracts.
- Dense generation waits only for the text format and embedding route to be
  fixed, not for the rest of the product to be finished.
- Correctness checks happen during implementation; integration tests happen at
  consolidation; score experiments happen after a working vertical slice.
- Person 4 coordinates consolidation, while all four people retain merge
  authority.

## 18. Plan-creation verification record

The following checks were run on 2026-08-29 before this plan was committed:

| Check | Result |
|---|---|
| `python3 -m unittest` | 3 tests passed |
| Markdown fence/structure check | passed; all code fences balanced |
| `git diff --check` | passed |
| Catalog scan | 50,000 rows and 50,000 unique IDs |
| Public target join | 200 / 200 targets present; 200 unique targets |
| Profile identifier scan | no identifier-like profile key |
| Release `SHA256SUMS` fetch | `catalog.jsonl.gz` checksum matches this plan |
| Unchanged starter evaluator | Hit Rate@10 `0.125`, MRR `0.068034`, MTTC `9.81`, TechnicalScore `0.10671` |

Verified starter scenario metrics:

| Scenario | Samples | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Buying | 80 | 0.2375 | 0.126508 | 8.625 |
| Browsing | 80 | 0.025 | 0.004514 | 10.75 |
| Intent Override | 30 | 0.133333 | 0.104167 | 10.066667 |
| Boundary | 10 | 0.0 | 0.0 | 11.0 |

These are baseline facts, not claims about the planned Role 2 implementation.
