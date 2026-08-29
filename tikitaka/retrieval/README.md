# Role 2 retrieval

This package implements the local, stateless retrieval boundary for Challenge
4. It consumes the current validated `SearchPlan`; it does not own conversation
state, ask/recommend decisions, shown-product history, or official response
normalization.

## Routes

- `SparseStructuredRetriever` is the deterministic no-network M1 route.
- `HybridRetriever` combines SQLite FTS5/BM25, an optional matching dense
  artifact, reliable structured evidence, and reciprocal-rank fusion.
- `ContractRetrieverAdapter` constructs Person 4's canonical `Candidate` and
  `ProductEvidence` types once their factories are supplied. This package does
  not redefine those shared contracts.

Dense failures never mix incompatible vectors. A requested route ID and index
ID must match the loaded manifest. An unavailable, timed-out, corrupt, or
mismatched dense route fails closed and the turn uses sparse/structured
retrieval instead.

## Embedder boundary

The build and query embedder must structurally provide:

```python
route_id: str
embed_documents(texts: Sequence[str]) -> Sequence[Sequence[float]]
embed_query(text: str) -> Sequence[float]
```

Provider SDKs, credentials, retry policy, token accounting, and automatic model
routing remain Person 1 responsibilities. The same `route_id` must be used for
document and query embeddings. Credentials belong in environment variables;
they must not appear in the factory specification or repository.

## Building a dense artifact

The factory is an importable zero-argument callable that returns Person 1's
Embedder-compatible adapter:

```bash
python3 scripts/build_dense_index.py \
  --output /path/outside-the-repository/dense-index \
  --embedder-factory package.module:create_embedder \
  --provider PROVIDER_ID \
  --model MODEL_ID
```

The build is deterministic and resumable at batch boundaries. The output is:

```text
manifest.json
ids.jsonl
vectors.f32
```

The manifest binds the artifact to the catalog source checksum, ordered ASIN
checksum, row count, `product_text_v1`, provider, model, route ID, dimension,
float dtype, normalization, document count, timestamp, and artifact checksums.
Loading fails if any compatibility or integrity field disagrees.

Exact cosine search uses a read-only NumPy memory map when NumPy is available
and otherwise retains a standard-library exact-search fallback. NumPy remains
optional until Person 4 approves and records the final dependency manifest;
the selected backend is exposed in retrieval diagnostics.

Generated artifacts can be large and are not source files. Keep them outside
the repository until Person 4 establishes the submission artifact policy.

## Inspecting hybrid retrieval

```bash
python3 scripts/inspect_hybrid_retrieval.py \
  "comfortable water-resistant shoes for long walks" \
  --artifact /path/to/dense-index \
  --embedder-factory package.module:create_embedder \
  --route hybrid \
  --mode buying \
  --category shoes \
  --max-price 80
```

The output includes sparse/dense ranks, manifest IDs, route overlap, filtering
counts, score concentration, effective candidate count, attribute
distributions, missing-metadata rates, constraint outcomes, route failures, and
route timings. These are inputs to Person 3's clarification sensor; retrieval
does not decide whether to ask a question.

## Current dependency status

The repository does not yet contain Person 1's production embedding adapter or
Person 4's canonical Python contract package. Tests use a deliberately small
deterministic semantic fixture only to prove artifact, compatibility, fusion,
fallback, and adapter behavior. Fixture results are not model-quality evidence
and must never be reported as a production embedding benchmark.
