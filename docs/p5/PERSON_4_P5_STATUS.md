# Person 4 — P5 Ablation Harness Status

Evidence revision: `4d486853fba0a62a48aa6c07e1a8dce6aa6520fc`

Status: the P5 decision-policy gate is complete. Threshold `0.07` with sparse
retrieval, deterministic interpretation and reranking, and profile weight `0`
is the frozen selection. The one-time held-out partition has been opened only
for that predeclared finalist.

## Implemented

- The orchestration composition root accepts an injected canonical retriever,
  allowing sparse, dense, and hybrid routes to run through the official Agent
  lifecycle rather than through a retrieval-only benchmark.
- `scripts/run_experiment.py` records explicit retrieval, generative-routing,
  Person 3 decision/reranking, and profile-weight arms.
- Dense and hybrid arms require both a versioned artifact and an importable
  matching embedder factory. A missing or mismatched route fails before scoring.
- Tuning is the default execution stage. Held-out execution requires both an
  explicit stage and `--confirm-held-out`.
- P5 arm parameters have their own fingerprint and are included in the
  experiment fingerprint and canonical report.
- Release selection follows Hit Rate@10, then MRR, then MTTC, and rejects a
  candidate whose held-out per-scenario Hit Rate drops materially against the
  declared baseline.

## Threshold tuning

The five pre-registered adaptive deterministic thresholds completed on the
140-session tuning partition with no network access:

| Threshold | Hit Rate@10 | MRR | MTTC | Questions |
|---:|---:|---:|---:|---:|
| 0.05 | 0.885714 | 0.504515 | 5.850000 | 490 |
| 0.06 | 0.885714 | 0.500060 | 5.771429 | 468 |
| **0.07** | **0.900000** | 0.502738 | 5.657143 | 448 |
| 0.08 | 0.892857 | 0.499226 | 5.550000 | 419 |
| 0.09 | 0.892857 | 0.497381 | 5.378571 | 388 |

Threshold `0.07` wins by the declared Hit Rate@10, MRR, then MTTC objective
order. Its per-scenario Hit Rates are Buying `0.875000`, Browsing `0.982143`,
Intent Override `0.714286`, and Boundary `1.000000`.

## Threshold-matched controls

The fixed-order control reduced Hit Rate@10 to `0.800000`, MRR to `0.378260`,
and Intent Override Hit Rate to `0.428571`. It is rejected despite asking only
308 questions.

The pinned `gpt-5.6-terra` `xhigh` anchored-LLM arm completed all 140 tuning
sessions after a guarded live preflight. It consumed `1,859,870` tokens at an
estimated cost of `$6.720220`, but reduced Hit Rate@10 to `0.671429`, MRR to
`0.401015`, and increased MTTC to `6.735714`. It recorded 169 interpreter
fallbacks and 131 reranker fallbacks and is rejected.

The first attempted live run encountered a local TLS trust-store failure before
the provider received a request. Its zero-token fallback report was overwritten
by the valid CA-configured run and is not used as evidence.

## One-time held-out confirmation

The finalist was frozen in `docs/p5/PERSON_4_P5_FINALIST_FREEZE.md` before
held-out was opened. The 60-session held-out result is:

| Metric | Value |
|---|---:|
| Hit Rate@10 | 0.933333 |
| MRR | 0.590245 |
| MTTC | 5.000000 |
| Efficiency | 0.600000 |
| Technical Score | 0.763740 |

Held-out per-scenario Hit Rates are Buying `0.916667` (`n=24`), Browsing
`1.000000` (`n=24`), Intent Override `0.888889` (`n=9`), and Boundary
`0.666667` (`n=3`). The Boundary result has high sampling uncertainty and must
not be generalized beyond those three sessions.

The selected route made no model calls and consumed no API tokens or cost.

## Usage

Run tuning first:

```bash
python3 scripts/run_experiment.py \
  --name sparse-deterministic \
  --retrieval-policy sparse \
  --generative-policy deterministic \
  --decision-arm adaptive-deterministic
```

After tuning choices are frozen, run the held-out arm explicitly:

```bash
python3 scripts/run_experiment.py \
  --name sparse-deterministic-held-out \
  --retrieval-policy sparse \
  --generative-policy deterministic \
  --decision-arm adaptive-deterministic \
  --stage held_out \
  --confirm-held-out
```

Dense and hybrid arms additionally require `--artifact` and
`--embedder-factory module.path:callable`.

Select only from completed held-out reports, with the declared baseline first:

```bash
python3 scripts/select_p5_config.py \
  reports/baseline-held-out.json \
  reports/candidate-a-held-out.json \
  reports/candidate-b-held-out.json \
  --output reports/p5-selection.json
```

## Canonical evidence

- `reports/p5-threshold-050.json`
- `reports/p5-threshold-060.json`
- `reports/p5-threshold-070.json`
- `reports/p5-threshold-080.json`
- `reports/p5-threshold-090.json`
- `reports/p5-fixed-ask-070.json`
- `reports/p5-llm-anchored-070.json`
- `reports/p5-threshold-070-held-out.json`
- `reports/p5-selection.json`
- `docs/p5/PERSON_4_P5_FINALIST_FREEZE.md`

## Remaining cross-workstream dependency

P5 policy selection is complete. The remaining major release-quality evidence
is the production 1024-dimensional `text-embedding-3-large` artifact and the
clean-revision sparse/dense/hybrid runtime sweep owned jointly with Person 2.
