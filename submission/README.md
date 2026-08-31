# TikiTaka TechJam submission

## Runtime

- Python 3.10 or later.
- No mandatory third-party packages.
- The organizer supplies the frozen catalog at `data/catalog.jsonl`.
- The entry point is `agent.py`, which exports `Agent`.

Run inside the organizer harness with:

```bash
python3 -m evaluator.local_evaluator
```

## Models and network behavior

`OPENAI_API_KEY` is optional. When present, the agent uses the declared
`gpt-5.6-terra` API route by default for every eligible intent-interpretation
and shortlist-reranking task. Selective or deterministic routing is used only
when explicitly pinned for an experiment. When the credential is absent, or
an API call fails, execution remains valid through the deterministic
`heuristic/local` fallback; no local generative LLM is used.

The production dense route is `text-embedding-3-large` at 1024 dimensions. The
index has been built and validated (50,000 x 1,024, $1.6563, index ID
`dense-285ef587d363de24212f`). Although both measured hybrid configurations
trailed the sparse control on tuning, the owner requires hybrid retrieval for
the hackathon submission. This override is recorded in
`reports/p6-hybrid-selection.json` and does not claim a score improvement.
Set `TIKITAKA_DENSE_ARTIFACT` to the external production artifact directory to
activate the selected hybrid route. Reciprocal-rank fusion uses sparse weight
`1.0` and dense weight `0.5`. The generated index is not included in this
source-only bundle. Without a compatible index or credential, retrieval remains
valid through the deterministic sparse/structured fallback. The index manifest
binds the catalog checksum, ordered product IDs, text schema, provider, model,
route, dimension, normalization, and artifact checksums, and mismatches fail
closed.

The runtime preserves an explicitly configured `SSL_CERT_FILE`. Otherwise it
selects an installed trusted CA bundle when available; TLS verification is
never disabled.

## Known limitations

- The selected hybrid route trails sparse on the measured tuning set and is an
  explicit hackathon requirement, not a measured quality win.
- Live API latency, token use, and cost depend on the conversation and provider.
- Exact dense search over 50,000 1024-dimensional vectors is substantially
  slower than sparse fallback in the reference Python backend.
- The frozen catalog is intentionally external to the participant bundle.

See `manifest.json` for the exact code revision, file hashes, frozen retrieval
identities, and package policy used to build this archive.

The current `medium` API configuration has not been measured live. Cost and
latency figures in `REPORT.md` are labeled historical `xhigh` evidence and must
not be interpreted as measurements of the current default.
