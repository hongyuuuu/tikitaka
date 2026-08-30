# Person 4 — P5 Finalist Freeze

Date: 2026-08-30

Code revision: `4d486853fba0a62a48aa6c07e1a8dce6aa6520fc`

Split: `public-v1`, seed `2026`, tuning fraction `0.7`

Scope: tuning evidence only; this decision was recorded before held-out was opened.

## Frozen finalist

Advance exactly one configuration to the one-time held-out run:

- sparse retrieval;
- deterministic interpreter;
- adaptive deterministic question policy at threshold `0.07`;
- deterministic reranker; and
- profile weight `0`.

The canonical tuning report is `reports/p5-threshold-070.json`.

Threshold `0.07` wins by the project's declared objective order. Its tuning
Hit Rate@10 is `0.900000`, ahead of thresholds `0.05` and `0.06` at
`0.885714` and thresholds `0.08` and `0.09` at `0.892857`. It also has no
per-scenario Hit Rate collapse: Buying `0.875000`, Browsing `0.982143`, Intent
Override `0.714286`, and Boundary `1.000000`.

## Rejected controls

The threshold-matched fixed-order control is rejected. It reduced Hit Rate@10
to `0.800000`, MRR to `0.378260`, and Intent Override Hit Rate to `0.428571`.
Its lower question count does not override the primary coverage objective.

The pinned `gpt-5.6-terra` `xhigh` anchored-LLM arm is also rejected. It used
`1,859,870` tokens at an estimated cost of `$6.720220`, but reduced Hit Rate@10
to `0.671429`, MRR to `0.401015`, and increased MTTC to `6.735714`. It recorded
169 interpreter fallbacks and 131 reranker fallbacks. The canonical report is
`reports/p5-llm-anchored-070.json`.

## Held-out gate

No alternative may be selected by looking at held-out. The one-time held-out
run may confirm this frozen finalist or reject it back to the already-declared
deterministic baseline; it must not re-rank thresholds or reopen the fixed-order
or LLM controls.

## One-time held-out result

The frozen finalist was subsequently run on the 60-session held-out partition
and confirmed with Hit Rate@10 `0.933333`, MRR `0.590245`, MTTC `5.000000`,
Efficiency `0.600000`, and Technical Score `0.763740`. It used no model calls,
tokens, or API cost.

Held-out per-scenario Hit Rate@10 is Buying `0.916667` (`n=24`), Browsing
`1.000000` (`n=24`), Intent Override `0.888889` (`n=9`), and Boundary
`0.666667` (`n=3`). The Boundary estimate is based on only three sessions and
must be reported with that sample-size limitation rather than generalized.

Canonical evidence:

- `reports/p5-threshold-070-held-out.json`
- `reports/p5-selection.json`
