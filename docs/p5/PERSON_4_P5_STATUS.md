# Person 4 — P5 Ablation Harness Status

Branch: `person4/p5-ablation-config-selection`

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

## Verified deterministic tuning run

The full 140-session tuning partition completed on the sparse deterministic
arm with no network access:

| Metric | Value |
|---|---:|
| Hit Rate@10 | 0.892857 |
| MRR | 0.522341 |
| MTTC | 5.771429 |
| Efficiency | 0.522857 |
| Technical score | 0.707702 |

The smoke report was written to `/tmp`, not committed as release evidence.

A separate one-session full-catalog hybrid wiring smoke test loaded a matching
50,000-row fixture index, executed hybrid retrieval through orchestration, and
attributed eight embedding calls to the recorded embedding route. The fixture
embedder is structural test evidence only, not model-quality evidence.

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

## Remaining P5 evidence dependencies

- Person 1 has not supplied a production embedding adapter/route, so the
  full-catalog dense artifact and its model-quality evidence do not yet exist.
- The existing Person 3 arms provide adaptive and never-ask policies, but no
  fixed-ask baseline. Person 3 must provide that policy; Person 4 should not
  invent decision behavior inside the evaluation harness.
- The complete tuning matrix and one-time held-out evaluation have not been
  run. Therefore no release configuration is selected and the P5 exit gate is
  not yet claimed.
