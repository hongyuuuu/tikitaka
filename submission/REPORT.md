# Method and feasibility report

## Method

TikiTaka maintains deterministic state per isolated session. An API-primary
interpreter converts each visible message into validated state operations; a
deterministic reducer applies add, remove, replace, exclude, no-preference, and
reset operations. Major intent changes increment the intent version and remove
stale category-dependent constraints while retaining still-applicable universal
constraints such as budget.

The current official entry point retrieves a bounded shortlist from the frozen
catalog with SQLite FTS5/BM25, a catalog-pinned local float32 dense index, exact
cosine search, structured metadata evidence, and reciprocal-rank fusion. The
selected hybrid weights are sparse `1.0` and dense `0.5`. Candidate IDs are
catalog-validated, duplicate-free, and limited before output. The production
1024-dimensional artifact is supplied externally through
`TIKITAKA_DENSE_ARTIFACT`; it is not part of this source-only bundle.

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

**Historical API route (`xhigh`).** The table below describes the earlier
`gpt-5.6-terra` `xhigh` run, not the current `medium` submission default. It was
measured against provider billing rather than derived from client-side token
counts. The distinction matters: our own accounting understated the bill by
4.8x, for the reason given below.

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

The current `medium` default has not been run live and therefore has no claimed
quality, latency, token, or cost measurement. Official model documentation
lists `medium` as the default reasoning effort, but that does not establish a
workload-specific savings factor. The historical projections above must not be
relabeled as `medium` estimates.

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
day, or a component making calls whose usage is never recorded. Fake-only
reconciliation tests now prove local aggregation across interpreter, repair,
and reranker events and report per-component token totals. They cannot resolve
the historical discrepancy because no isolated provider snapshot exists for
that run. Until a controlled same-window snapshot is available, the billed
historical figures above are the ones to trust and client-derived figures
should not be used for budgeting.

The p95 latency of 30.0 s sits exactly at the configured 30 s request timeout,
which is worth fixing on its own merits, but it is not demonstrated to be the
cause of the accounting gap.

Cached input is billed at $0.20/1M, a tenth of standard, and nothing here
records cache hits; any caching benefit is already reflected in the billed
figures above.

**Production embeddings.** Route pinned to `text-embedding-3-large` at 1024
dimensions. The index is built: 50,000 documents at 1,024 dimensions,
12,740,612 `cl100k_base` input tokens over 391 successful batched requests at
batch size 128, 922 s wall time including transient HTTP 429 windows and one
recovered client timeout, for **$1.65627956** at $0.13/1M input tokens. The
local token count matched the provider's exactly on the final 9,552-document
segment (2,401,640 tokens). Artifact totals 205,450,751 bytes
(204,800,000 of normalized little-endian float32 vectors); load and checksum
validation passed. Index ID `dense-285ef587d363de24212f`, bound to catalog
SHA-256 `da979b05…c69a67`. The figure excludes a separate six-token route probe
and any provider-side charge for the request whose response timed out.

**Pinned hybrid-versus-sparse comparison.** Both arms ran on the same 140
tuning samples at the same revision, deterministic interpretation and
reranking, conservative question policy, profile weight zero. Held-out was not
opened.

| Metric | Sparse | Hybrid | Hybrid − sparse |
|---|---:|---:|---:|
| Hit Rate@10 | 0.900000 | 0.892857 | -0.007143 |
| MRR | 0.502738 | 0.486613 | -0.016125 |
| MTTC | 5.657143 | 5.700000 | +0.042857 |
| Efficiency | 0.534286 | 0.530000 | -0.004286 |
| Technical score | 0.707679 | 0.698412 | -0.009267 |
| Questions asked | 448 | 465 | +17 |

Hybrid query embeddings cost $0.007088 over 783 calls and 54,525 tokens, with
no failed calls and no fallback activations. This original pinned hybrid arm
did not earn selection on quality.

**Owner-selected hybrid configuration.** A later tuning-only run with sparse
weight `1.0` and dense weight `0.5` used
769 query-embedding calls, 53,223 input tokens, and an estimated `$0.006919`.
Its metrics were Hit Rate@10 `0.892857`, MRR `0.486071`, MTTC `5.600000`,
Efficiency `0.540000`, and TechnicalScore `0.700250`. Held-out remained
`not_run`. It also trails the sparse control; hybrid is selected because the
owner requires it for the hackathon, not because it improved measured quality.

An earlier equal-weight attempt reported Hit Rate@10 `0.892857` and MRR
`0.501845`, but it recorded zero retrieval calls because DNS/TLS failures sent
every turn through `sparse_fallback`. It is retained only as failure-path
diagnostics and is not hybrid quality evidence.

**Fixture hybrid timings** are mechanics evidence only and are not reported as
production performance.

## Limitations

- The source-only package intentionally excludes the catalog, evaluator,
  public labels, generated dense index, and credentials.
- Without credentials, semantic interpretation and LLM reranking degrade to a
  deterministic heuristic route.
- Without a compatible production index, retrieval remains sparse/structured.
- The selected hybrid tuning result trails the sparse control on Hit Rate@10,
  MRR, and TechnicalScore. The owner nevertheless requires hybrid retrieval for
  the hackathon submission; this override is disclosed rather than relabeled as
  a score improvement.
- The `medium` API default has no live score or cost claim; all API measurements
  in this report are explicitly historical `xhigh` evidence.
- Public-set metrics are development evidence and do not predict private-set
  performance.
