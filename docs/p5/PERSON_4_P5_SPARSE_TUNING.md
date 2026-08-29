# Person 4 — P5 Sparse Tuning Decision

Date: 2026-08-29  
Code revision: `c6a27e0`  
Split: `public-v1`, seed `2026`, tuning fraction `0.7`  
Scope: 140 tuning sessions only; held-out was not run.

## Decision

Keep `adaptive-deterministic` as the sparse shortlist baseline.

- Reject `always-recommend-baseline`: clarification is materially necessary
  for catalog coverage on the tuning partition.
- Do not promote `official-proxy-deterministic`: its MRR is slightly higher,
  but Hit Rate@10 is lower and Boundary Hit Rate drops from `1.0000` to
  `0.8571`.
- Do not promote `conservative-questions-deterministic` despite its best
  aggregate Technical Score and MTTC. Relative to adaptive, Intent Override
  Hit Rate drops by `0.0952` and Boundary drops by `0.1429`, both beyond the
  P5 selection safeguard of a maximum `0.05` scenario drop.
- Reject profile weight `0.10`: it lowers both Hit Rate@10 and MRR. Keep the
  profile weight at `0` unless a later held-out candidate improves quality.

This is a tuning decision, not a release selection. No held-out claim is made.

## Aggregate tuning results

| Arm | Hit Rate@10 | MRR | MTTC | Efficiency | Technical Score | Questions |
|---|---:|---:|---:|---:|---:|---:|
| Always recommend | 0.514286 | 0.167537 | 7.107143 | 0.389286 | 0.385261 | 0 |
| Adaptive | **0.892857** | 0.522341 | 5.771429 | 0.522857 | 0.707702 | 502 |
| Official proxy | 0.885714 | **0.531389** | 5.721429 | 0.527857 | 0.707845 | 489 |
| Conservative | **0.892857** | 0.522891 | **5.471429** | **0.552857** | **0.713867** | 440 |
| Profile 0.10 | 0.885714 | 0.464314 | 5.750000 | 0.525000 | 0.687151 | 491 |

The question totals count recommendation-free clarification turns across all
140 sessions. Fewer questions alone cannot override a material scenario
coverage regression.

## Per-scenario Hit Rate@10

| Arm | Buying (n=56) | Browsing (n=56) | Intent Override (n=21) | Boundary (n=7) |
|---|---:|---:|---:|---:|
| Always recommend | 0.642857 | 0.517857 | 0.190476 | 0.428571 |
| Adaptive | 0.839286 | 0.982143 | **0.761905** | **1.000000** |
| Official proxy | 0.839286 | **1.000000** | 0.714286 | 0.857143 |
| Conservative | **0.892857** | 0.982143 | 0.666667 | 0.857143 |
| Profile 0.10 | 0.857143 | 0.964286 | 0.714286 | **1.000000** |

## Canonical evidence

- `reports/sparse-always-recommend.json`
- `reports/sparse-adaptive.json`
- `reports/sparse-official-proxy.json`
- `reports/sparse-conservative.json`
- `reports/sparse-profile-010.json`

Every report contains its full configuration, experiment and arm fingerprints,
split manifest, aggregate and per-scenario metrics, usage, question counts, and
per-session outputs. Each report records held-out as `not_run`.

## Remaining tuning dependencies

1. Person 3 must supply the fixed-ask baseline.
2. Person 1 must supply a production embedding route so Person 2's matching
   full-catalog dense artifact can be built and sparse/dense/hybrid quality can
   be compared.
3. API interpreter/reranker experiments require an explicit cost-authorized
   run. The earlier ten-session probe is not sufficient release evidence.
