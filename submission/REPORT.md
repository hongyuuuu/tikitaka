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

**Offline route.** Zero model calls, zero model tokens, zero model cost. This
is the route the reproduced result above was scored on.

**API route.** Measured over a 10-session live probe against
`gpt-5.6-terra` at `xhigh`, 90 calls, list price $2.00/1M input and
$12.00/1M output:

| Measure | Value |
|---|---:|
| Prompt tokens per call | 1,422 |
| Completion tokens per call | 327 |
| — of which reasoning | 255 (78%) |
| Latency, mean | 7.2 s |
| Latency, p95 | 30.0 s |
| Cost per session | $0.0609 |
| Projected, 200 public sessions | $12.17 |
| Projected, 800 private sessions | $48.69 |

Three qualifications, none of them cosmetic:

- The p95 latency of 30.0 s sits at the configured 30 s request timeout. Turns
  at that tail degrade to the deterministic route rather than failing, but the
  headroom is nil.
- Cached input is billed at $0.20/1M, a tenth of standard, and the prompt
  prefix is large and stable. No field records cache hits, so these figures
  assume none and are therefore an upper bound.
- Ten sessions fixes cost, tokens and latency; it does not establish quality.
  The interval on Hit Rate@10 at that sample size is about ±0.31.

**Production embeddings.** Route pinned to `text-embedding-3-large` at 1024
dimensions. Estimated one-off build volume is 14.5M input tokens across ~196
requests for the 50,000-document catalog. Actual build time, index size and
query latency remain pending until the artifact is built.

**Fixture hybrid timings** are mechanics evidence only and are not reported as
production performance.

## Limitations

- The source-only package intentionally excludes the catalog, evaluator,
  public labels, generated dense index, and credentials.
- Without credentials, semantic interpretation and LLM reranking degrade to a
  deterministic heuristic route.
- Without a compatible production index, retrieval remains sparse/structured.
- Public-set metrics are development evidence and do not predict private-set
  performance.
