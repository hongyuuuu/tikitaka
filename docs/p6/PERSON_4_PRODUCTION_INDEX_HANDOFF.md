# Person 4 — Production index handoff and evaluation gate

Date: 2026-08-31

Owner boundary: Person 2 builds and validates the production dense artifact.
Person 4 accepts the artifact, integrates it through the official lifecycle,
runs the predeclared comparison, and publishes release evidence. Person 4 does
not rebuild or silently repair an artifact that fails identity validation.

## Required Person 2 delivery

The handoff is incomplete unless all of these are supplied:

- artifact directory outside the repository;
- `manifest.json`, `ids.json`, and `vectors.f32` produced by the canonical
  builder;
- catalog SHA-256
  `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`;
- 50,000 documents and 1,024 float32 dimensions;
- provider `openai`, model `text-embedding-3-large`, and route
  `openai/text-embedding-3-large/dimensions-1024`;
- normalized vectors and `dense-f32-v1` artifact format;
- actual build duration, artifact bytes, prompt tokens, request count, and
  embedding cost; and
- confirmation that no credential or generated vector file was committed.

Person 2's canonical build command is shown for reproducibility only. It is a
paid operation and must not be run by Person 4 without explicit spend approval:

```bash
python3 scripts/build_dense_index.py \
  --catalog data/catalog.jsonl \
  --expected-count 50000 \
  --output /absolute/path/outside/repository/tikitaka-dense-1024 \
  --embedder-factory tikitaka.retrieval.openai_embeddings:openai_embedder_from_env \
  --provider openai \
  --model text-embedding-3-large
```

## Person 4 acceptance checks

Before evaluating quality:

1. Confirm the source worktree is clean and record its revision.
2. Load the artifact through `load_dense_index`; never edit its manifest.
3. Refuse a catalog checksum, ordered-ID checksum, route, dimension, dtype,
   normalization, count, or file-checksum mismatch.
4. Record artifact bytes and build usage from Person 2's machine-readable
   builder output.
5. Keep the artifact outside the repository and submission ZIP.

## Pinned quality comparison

The held-out partition has already been opened for the frozen P5 finalist. It
must not be reused to tune or select the production hybrid route. Run both arms
on the tuning partition only, at the same clean revision, with deterministic
interpretation/reranking, the conservative question policy, and profile weight
zero.

The sparse control is free:

```bash
python3 scripts/run_experiment.py \
  --name p6-production-sparse-control \
  --output /tmp/p6-production-sparse-control.json \
  --retrieval-policy sparse \
  --generative-policy deterministic \
  --decision-arm conservative-questions-deterministic \
  --profile-weight 0 \
  --stage tuning
```

The hybrid arm makes query-embedding API calls and requires explicit spend
approval:

```bash
python3 scripts/run_experiment.py \
  --name p6-production-hybrid \
  --output /tmp/p6-production-hybrid.json \
  --retrieval-policy hybrid \
  --generative-policy deterministic \
  --decision-arm conservative-questions-deterministic \
  --profile-weight 0 \
  --stage tuning \
  --artifact /absolute/path/outside/repository/tikitaka-dense-1024 \
  --embedder-factory tikitaka.retrieval.openai_embeddings:openai_embedder_from_env
```

Do not add `--stage held_out`, `--stage both`, or `--confirm-held-out`.

## Acceptance evidence

Publish, without selecting on held-out:

- sparse and hybrid tuning Hit Rate@10, MRR, MTTC, Efficiency, and
  TechnicalScore;
- Buying, Browsing, Intent Override, and Boundary deltas;
- query-embedding tokens, calls, latency, and cost;
- index bytes and build duration;
- manifest/index identities and route-execution diagnostics; and
- limitations, including that tuning-set gains do not establish private-set
  gains.

If the hybrid arm loses, fails identity checks, or degrades routes, keep sparse
as the release retrieval route and report the result. Do not alter the frozen
catalog, evaluator, split, or held-out evidence.
