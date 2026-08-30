# Method and feasibility report

## Method

TikiTaka maintains deterministic state per isolated session. An API-primary
interpreter converts each visible message into validated state operations; a
deterministic reducer applies add, remove, replace, exclude, no-preference, and
reset operations. Major intent changes increment the intent version and remove
stale category-dependent constraints while retaining still-applicable universal
constraints such as budget.

The current official entry point retrieves a bounded shortlist from the frozen
catalog with SQLite FTS5/BM25 plus structured metadata evidence. Candidate IDs
are catalog-validated, duplicate-free, and limited before output. The dense
implementation uses a catalog-pinned local float32 index, exact cosine search,
and reciprocal-rank fusion, but the production 1024-dimensional artifact is not
part of this source-only bundle.

The selected generative route is `gpt-5.6-terra` with `xhigh` reasoning through
the main API. It interprets intent and reranks only a bounded shortlist. Its
outputs are treated as untrusted and validated before state or output changes.

## Reproduced offline result

With `OPENAI_API_KEY` absent and Python socket activity denied, the full
200-session public evaluator completed with:

| Metric | Result |
|---|---:|
| Hit Rate@10 | 0.885000 |
| MRR | 0.529240 |
| MTTC | 5.780000 |
| Efficiency | 0.522000 |
| TechnicalScore | 0.705672 |
| Model prompt/completion tokens | 0 / 0 |

The clean reproduction imports participant code only from an extracted bundle,
not from the source repository. Network attempts, Agent exceptions, and raw
response violations were all zero in the dedicated offline evidence run.

## Latency, token, and cost disclosure

- Offline route: zero model calls, zero model tokens, and zero model cost.
- API route: latency and token use vary by turn and are reported through the
  Agent usage payload. Estimated cost is recorded by the model gateway.
- Production embeddings: actual build time, index size, query time, and cost
  remain pending until the real `text-embedding-3-large`/1024 artifact is built.
- Fixture hybrid timings are mechanics evidence only and are not reported as
  production performance.

## Limitations

- The source-only package intentionally excludes the catalog, evaluator,
  public labels, generated dense index, and credentials.
- Without credentials, semantic interpretation and LLM reranking degrade to a
  deterministic heuristic route.
- Without a compatible production index, retrieval remains sparse/structured.
- Public-set metrics are development evidence and do not predict private-set
  performance.
