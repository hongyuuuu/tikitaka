# Role 2 retrieval

This package implements the local, stateless retrieval boundary for Challenge
4. It consumes the current validated `SearchPlan`; it does not own conversation
state, ask/recommend decisions, shown-product history, or official response
normalization.

## Routes

- `SparseStructuredRetriever` is the deterministic no-network M1 route.
- `HybridRetriever` combines SQLite FTS5/BM25, an optional matching dense
  artifact, reliable structured evidence, and reciprocal-rank fusion.
- Both public retrievers satisfy Person 4's canonical `Retriever` protocol and
  return canonical `Candidate` plus `ProductEvidence` records directly.
- `ContractRetrieverAdapter` remains only as a compatibility wrapper.

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
routing remain Person 1 responsibilities. `GatewayEmbedder` bridges Person 1's
provider-neutral batch `EmbeddingModel` and `ModelRoute` to this interface. It
validates usage attribution and every declared route/provider/model/index
identity. The same route must be used for document and query embeddings.
Credentials belong in environment variables; they must not appear in the
factory specification or repository.

## Building a dense artifact

The factory is an importable zero-argument callable that returns a canonical
Embedder. A `GatewayEmbedder` declares its provider and model, so the CLI reads
those values from the route and prevents duplicated configuration from
drifting. Legacy structural embedders must supply `--provider` and `--model`:

```bash
python3 scripts/build_dense_index.py \
  --output /path/outside-the-repository/dense-index \
  --embedder-factory package.module:create_embedder
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

The output includes sparse/dense ranks, manifest IDs, attributable embedding
usage, route overlap, filtering counts, score concentration, effective
candidate count, attribute
distributions, missing-metadata rates, constraint outcomes, route failures, and
route timings. These are inputs to Person 3's clarification sensor; retrieval
does not decide whether to ask a question.

## Split-aware retrieval benchmarking

`scripts/benchmark_retrieval.py` compares sparse, dense, and hybrid coverage on
a strict JSONL case file supplied by Person 4. Every split must contain Buying,
Browsing, Intent Override, and Boundary cases, and every file must contain
explicit `tuning` and `heldout` splits. Targets must be disjoint across those
splits; the loader rejects cross-split leakage. The benchmark never derives
cases from evaluator internals and never passes a target ID into retrieval; it
compares the target only after a candidate pool has been returned.

Each JSONL record uses this retrieval-only boundary:

```json
{
  "case_id": "heldout_001",
  "split": "heldout",
  "scenario": "intent_override",
  "target_parent_asin": "B000...",
  "request": {
    "text_query": "current corrected intent",
    "intent_version": 2,
    "mode": "buying",
    "constraints": [
      {"attribute": "budget", "values": [80], "strength": "hard", "operator": "lte"}
    ]
  }
}
```

The case producer—not this benchmark—owns the stable split and converts
validated conversation state into the active request. Case targets remain
offline evaluation labels and must never enter `Agent.reset()` or
`Agent.respond()`.

```bash
python3 scripts/benchmark_retrieval.py \
  --cases /path/to/versioned-retrieval-cases.jsonl \
  --artifact /path/to/dense-index \
  --embedder-factory package.module:create_embedder \
  --routes sparse,dense,hybrid \
  --ks 10,50,100,200 \
  --output /path/outside-the-repository/retrieval-report.json
```

Reports include the case-file checksum, catalog and dense-index identities,
complete retrieval configuration, usage, latency, overall metrics, and
per-scenario metrics for each split. Candidate counts, hard-filter counts,
sparse/dense overlap, and route timings are also summarized overall and per
scenario. Dense or hybrid fallback fails a benchmark by default so a sparse
fallback cannot masquerade as dense-model quality. Use
`--allow-route-degradation` only for an explicit failure-path experiment.

## Current dependency status

Person 4's canonical contracts and Role 2's provider-neutral gateway bridge are
integrated. Person 1 has not yet supplied the live embedding provider adapter,
credential configuration, or initial production embedding route. Tests use a
deliberately small deterministic semantic fixture only to prove artifact,
compatibility, usage, fusion, fallback, and adapter behavior. Fixture results
are not model-quality evidence and must never be reported as a production
embedding benchmark.
