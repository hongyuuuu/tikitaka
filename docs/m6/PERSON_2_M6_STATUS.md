# Person 2 — M6 Submission Status

Evidence revision: `92938963e57773b1375fdfb11735b9acd3f630e9`.

Status: every source-only M6 gate is implemented. Two numerical fields remain
explicitly pending on the real `text-embedding-3-large` 1024-dimensional index:
production index build/size/query/cost measurements and the production-hybrid
quality delta. No fixture result is substituted for either field.

## Exit-gate status

| Requirement | Status | Evidence |
|---|---|---|
| Freeze text/index/config versions | met | generated package `manifest.json` records contract, schema, sparse, structured, hybrid, embedding, dimension and artifact-format identities |
| Reproduce from documented catalog | met | builder requires the external 50,000-row catalog and records its SHA-256 without bundling it |
| Build lightweight participant bundle | met | deterministic source-only ZIP; strict audit produced 68 files / 155,729 compressed bytes |
| Supply entry file, dependencies, setup and report | met | `submission/agent.py`, `requirements.txt`, `README.md`, and `REPORT.md` |
| Prove clean-machine-style execution | met | extracted temporary harness, repository absent from import path, all 200 sessions completed |
| Prove network-free degraded behavior | met | credential removed, Python socket audit guard active, zero attempts, route `heuristic/local`, zero model tokens |
| Enforce artifact policy | met | package whitelist, tracked-file audit, size limits, path traversal/symlink checks, secret scan and per-file hashes |
| Provide retrieval traces | met | five full-catalog sparse cases plus a pinned fixture-hybrid mechanics comparison |
| Measure production dense index | pending | requires the production 1024-dimensional artifact |
| Measure production hybrid quality delta | pending | requires the production artifact and matching primary run |

## Package boundary

The generated archive includes only:

- root `agent.py`, `README.md`, `REPORT.md`, `requirements.txt`, and generated
  `manifest.json`;
- Python sources under `starter/` and `tikitaka/`.

It excludes the frozen catalog, public labels, evaluator, tests, reports,
credentials, environment files, generated dense artifacts, caches, and
temporary output. The catalog remains an organizer-supplied read-only input at
`data/catalog.jsonl`.

Every bundled source file is individually hashed. ZIP member order, timestamps,
permissions, and compression settings are fixed so the same revision and
catalog identity reproduce the same archive.

## Clean reproduction

The verifier performs the following without relying on the live source tree:

1. builds and audits the archive;
2. extracts it into a fresh temporary directory;
3. copies the unchanged public evaluator, dataset, and frozen catalog into that
   external harness as organizer-supplied inputs;
4. removes `OPENAI_API_KEY` and `PYTHONPATH`;
5. installs a socket/DNS denial audit hook;
6. runs all 200 sessions and verifies all participant imports resolve inside
   the temporary harness;
7. rejects nonzero model tokens, network attempts, wrong routes, missing
   scenarios, or an incomplete session count.

The reproduced offline metrics are Hit Rate@10 `0.885000`, MRR `0.529240`, MTTC
`5.780000`, Efficiency `0.522000`, and TechnicalScore `0.705672` with zero model
tokens. The final machine-readable report is `reports/m6-release-audit.json`.
The isolated evaluator wall time was `24,832.056 ms`; this is an end-to-end
reproduction measurement, not a production per-query latency claim.

## Retrieval traces

`artifacts/m6/retrieval-traces.json` contains label-free, hand-authored visible
queries over the actual 50,000-product catalog:

1. a vague “shoes for a trip” request;
2. explicit water resistance, comfort, long-walking, and budget constraints;
3. the old red/leather/boots intent before an override;
4. the new running-shoes intent at version 2, with old dependent constraints
   absent and color recorded as no-preference;
5. a Boundary-style no-material-preference request.

The same explicit query is also run through a catalog-pinned deterministic
fixture dense index. That comparison proves dense identity checks, query
embedding, exact search, fusion, diagnostics, and evidence plumbing. It is
marked `fixture_mechanics_only` and makes no production-quality claim.

## Reproduction commands

Run from a clean committed worktree with the frozen catalog present. Write the
archive and temporary report outside the repository:

```bash
python3 scripts/build_submission.py \
  --output /tmp/tikitaka-submission.zip

python3 scripts/verify_m6_submission.py \
  --archive /tmp/tikitaka-submission.zip \
  --output /tmp/tikitaka-m6-release-audit.json
```

Regenerate the retrieval trace set when the full fixture index is available:

```bash
python3 scripts/capture_m6_retrieval_traces.py \
  --output artifacts/m6/retrieval-traces.json \
  --hybrid-artifact /path/outside/repository/fixture-index \
  --embedder-factory tests.retrieval_fakes:create_gateway_semantic_embedder
```

## Remaining production-index insertion

When the real 1024-dimensional artifact arrives, no redesign is required:

1. build/verify it using `scripts/build_dense_index.py` and its canonical
   manifest;
2. record actual artifact bytes, build duration, query timings, and embedding
   usage/cost;
3. run the pinned production hybrid experiment against the same evaluation
   split;
4. calculate primary-hybrid minus offline deltas for Hit Rate@10, MRR, MTTC,
   Efficiency and TechnicalScore;
5. replace only the fields currently marked `pending_production_1024_index`.

## Coordination note

`submission/` is a release coordination surface normally consolidated by
Person 4. All four owners must be told about this M6 package boundary before the
branch is merged. The branch may be pushed for review, but should not be merged
until that acknowledgement and Kevin's explicit merge instruction.
