# Person 2 — Production 1,024-dimensional index handoff

Date: 2026-08-31

Status: artifact accepted locally; production hybrid not selected on tuning.

The production dense artifact has been built and validated at:

```text
/Users/kevinyongcj/Programming/tikitaka-artifacts/tikitaka-dense-1024
```

It is intentionally outside the repository. The complete machine-readable
evidence is in `reports/p6-production-index-handoff.json`.

## Artifact acceptance

- Source revision: `ec07837d77ef8872a6f98b97174fc819f76e2ea3`
- Catalog SHA-256:
  `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`
- Index ID: `dense-285ef587d363de24212f`
- Route: `openai/text-embedding-3-large/dimensions-1024`
- Shape: 50,000 documents × 1,024 dimensions
- Encoding: normalized little-endian float32, `dense-f32-v1`
- Vector bytes: 204,800,000
- Total artifact bytes: 205,450,751
- Load and checksum validation: passed

The canonical builder emits `ids.jsonl`, not `ids.json`; the latter name in
Person 4's request is a documentation typo. The manifest checksum for
`ids.jsonl` is
`434c4af5626625892f78c3ce59303dca7faae6755ddb24118a0a87d30e198f6b`,
and the `vectors.f32` checksum is
`d7992b1442fdc7a01f61ccaf85e06ca099247dd0ad52b298c3d97f6eb0b77a1e`.

## Build evidence

The resumable build took 922 seconds wall time, including transient HTTP 429
windows and one recovered client timeout. It issued 391 successful document
embedding requests at batch size 128.

The 50,000 unique product texts contain 12,740,612 `cl100k_base` tokens. At
$0.13 per million tokens, the unique-artifact embedding cost is $1.65627956.
The local count for the final 9,552 documents is 2,401,640 tokens, exactly
matching the provider-reported count for that segment. This cost excludes the
separate six-token route probe and any possible provider-side charge for the
request whose response timed out.

## Pinned tuning comparison

Both arms ran on 140 tuning samples at the same revision with deterministic
interpretation/reranking, the conservative question policy, and profile weight
zero. Held-out status remained `not_run`.

| Metric | Sparse | Hybrid | Hybrid − sparse |
| --- | ---: | ---: | ---: |
| Hit Rate@10 | 0.900000 | 0.892857 | -0.007143 |
| MRR | 0.502738 | 0.486613 | -0.016125 |
| MTTC | 5.657143 | 5.700000 | +0.042857 |
| Efficiency | 0.534286 | 0.530000 | -0.004286 |
| Technical score | 0.707679 | 0.698412 | -0.009267 |
| Questions | 448 | 465 | +17 |

Hybrid query embeddings made 783 calls over 54,525 tokens, with 268,146 ms
aggregate provider latency and $0.007088 estimated cost. There were no failed
query calls or fallback activations.

The hybrid arm's technical-score deltas were -0.006857 for Buying, -0.009550
for Browsing, -0.001848 for Intent Override, and -0.048521 for Boundary.
Although intent-override MRR and MTTC improved, the complete technical score did
not.

## Release decision

Keep sparse retrieval as the release route. The dense artifact is valid and
ready for future fusion/text-schema tuning, but this pinned hybrid configuration
does not earn selection. Do not open held-out to tune it.

No API credential, `.env` file, generated vector file, or artifact directory is
tracked by Git.
