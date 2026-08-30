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

The selected generative route is `gpt-5.6-terra` with `medium` reasoning through
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

**API route.** Measured against provider billing, not derived from client-side
token counts. The distinction matters: our own accounting understated the bill
by 4.8x, for the reason given below.

| Measure | Value |
|---|---:|
| Billed input tokens | 3.086 M |
| Billed output tokens | 1.035 M |
| Billed cost, all live work | **$18.59** |
| Cost per session, always-generative | **$0.266** |
| Projected, 200 public sessions | **$53.14** |
| Projected, 800 private sessions | **$212.57** |
| Latency, mean | 6.6 s |
| Latency, p95 | 30.0 s |

**Client-side token counts understate the bill and must not be used alone.**
The agent's `Usage` recorded 0.838 M input and 0.183 M output for the same work
the provider billed at 3.086 M and 1.035 M — **79% of the real cost was
invisible to our instrumentation**, and output is understated more than input
(5.65x against 3.68x).

**The cause is not established.** Timeout-and-retry was the obvious candidate
and it does not fit: `ApiInterpreter` does not retry a timeout, and the
50-session run recorded zero fallbacks and zero repairs across 404 clean calls.
Remaining candidates, none confirmed: provider-side usage not surfaced in the
response body, activity on the account outside these runs on the same billing
day, or a component making calls whose usage is never recorded. Until it is
identified, the billed figures above are the ones to trust and the derived ones
should not be used for budgeting.

The p95 latency of 30.0 s sits exactly at the configured 30 s request timeout,
which is worth fixing on its own merits, but it is not demonstrated to be the
cause of the accounting gap.

Cached input is billed at $0.20/1M, a tenth of standard, and nothing here
records cache hits; any caching benefit is already reflected in the billed
figures above.

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
