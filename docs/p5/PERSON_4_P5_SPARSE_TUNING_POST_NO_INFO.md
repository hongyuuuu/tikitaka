# Person 4 — P5 Sparse Tuning After No-Information Guard

Date: 2026-08-30  
Code revision: `fc61002`  
Split: `public-v1`, seed `2026`, tuning fraction `0.7`  
Scope: 140 tuning sessions only; held-out was not run.

## Decision

Promote `conservative-questions-deterministic` as the sparse tuning baseline.
This supersedes the pre-guard tuning choice recorded at revision `c6a27e0`.

The no-information guard removed an accidental constraint-count signal that
had caused some clean no-information replies to switch from CLARIFY to
RECOMMEND. After removing that pollution, the conservative policy:

- has the highest Hit Rate@10 (`0.900000`);
- improves every scenario Hit Rate relative to adaptive;
- reduces MTTC from `5.985714` to `5.657143`;
- asks 65 fewer questions than adaptive; and
- has no scenario Hit Rate regression relative to adaptive.

The conservative arm's MRR is `0.000357` below adaptive, but the official
objective prioritizes Hit Rate@10 before MRR and MTTC. Its coverage gain is
therefore decisive.

Keep profile weight at `0`. The `0.10` profile arm trails conservative on
Hit Rate@10, MRR, and Intent Override coverage.

This remains a tuning decision, not a release selection. No held-out claim is
made.

## Aggregate post-fix tuning results

| Arm | Hit Rate@10 | MRR | MTTC | Efficiency | Technical Score | Questions |
|---|---:|---:|---:|---:|---:|---:|
| Always recommend | 0.514286 | 0.167537 | 7.107143 | 0.389286 | 0.385261 | 0 |
| Adaptive | 0.864286 | 0.503095 | 5.985714 | 0.501429 | 0.683357 | 513 |
| Official proxy | 0.885714 | **0.509872** | 5.907143 | 0.509286 | 0.697676 | 501 |
| Conservative | **0.900000** | 0.502738 | **5.657143** | **0.534286** | **0.707679** | 448 |
| Profile 0.10 | 0.871429 | 0.470794 | 5.735714 | 0.526429 | 0.682238 | 481 |

## Per-scenario Hit Rate@10

| Arm | Buying (n=56) | Browsing (n=56) | Intent Override (n=21) | Boundary (n=7) |
|---|---:|---:|---:|---:|
| Always recommend | 0.642857 | 0.517857 | 0.190476 | 0.428571 |
| Adaptive | 0.839286 | 0.982143 | 0.619048 | 0.857143 |
| Official proxy | 0.839286 | **1.000000** | 0.666667 | **1.000000** |
| Conservative | **0.875000** | 0.982143 | **0.714286** | **1.000000** |
| Profile 0.10 | 0.857143 | 0.964286 | 0.619048 | **1.000000** |

Relative to adaptive, conservative changes scenario Hit Rate by:

- Buying: `+0.035714`
- Browsing: `0.000000`
- Intent Override: `+0.095238`
- Boundary: `+0.142857`

## Effect of the guard

The always-recommend arm is byte-for-byte unchanged in score, which isolates
the behavioral effect to policies that spend clarification turns. Adaptive
lost `0.028571` Hit Rate@10 after the pollution was removed, concentrated in
Intent Override and Boundary. Conservative recovers clean coverage without
reintroducing the invalid constraint.

## Canonical evidence

- `reports/sparse-post-no-info-always-recommend.json`
- `reports/sparse-post-no-info-adaptive.json`
- `reports/sparse-post-no-info-official-proxy.json`
- `reports/sparse-post-no-info-conservative.json`
- `reports/sparse-post-no-info-profile-010.json`

Every report records revision `fc61002`, its full configuration, experiment
and arm fingerprints, split manifest, aggregate and per-scenario metrics,
usage, question counts, and per-session outputs. Each report records held-out
as `not_run`.

## Remaining P5 dependencies

1. Person 3 must supply the fixed-ask baseline.
2. Person 1 must supply a production embedding route so a matching
   full-catalog dense artifact can be built.
3. API interpreter/reranker experiments require an explicitly cost-authorized
   tuning run.
