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
index has been built and validated (50,000 x 1,024, $1.66, index ID
`dense-285ef587d363de24212f`), but it is **not** the release route: the pinned
hybrid arm lost to sparse on 140 tuning sessions, so retrieval runs on the
deterministic sparse/structured route by decision rather than by omission. See
`reports/p6-production-index-handoff.json`. The generated index is not included
in this source-only bundle. The index manifest binds the catalog checksum,
ordered product IDs, text schema, provider, model, route, dimension,
normalization, and artifact checksums, and mismatches fail closed.

## Known limitations

- Sparse retrieval is the measured selection, not a degraded stand-in for the
  dense index; the hybrid arm was tested and lost. It was rejected at one
  pinned configuration only.
- Live API latency, token use, and cost depend on the conversation and provider.
- The frozen catalog is intentionally external to the participant bundle.

See `manifest.json` for the exact code revision, file hashes, frozen retrieval
identities, and package policy used to build this archive.

The current `medium` API configuration has not been measured live. Cost and
latency figures in `REPORT.md` are labeled historical `xhigh` evidence and must
not be interpreted as measurements of the current default.
