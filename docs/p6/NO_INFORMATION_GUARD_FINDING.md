# Finding — the third no-information template is not suppressed

Date: 2026-08-31

Status: **defect confirmed, fix measured and rejected.** No runtime code
changed. Evidence: `reports/ask-me-guard-fix.json`.

## The defect

`evaluator.local_evaluator.customer_reply` can return four things. Three are
no-information replies and one carries real constraints:

| Simulator reply | Carries a constraint? | Suppressed by `carries_no_new_constraint`? |
|---|---|---|
| `I don't have a preference for {attr}; please use your judgment.` | no | yes |
| `I don't have an additional preference for {attr}.` | no | yes |
| `Those options are not quite right yet. Ask me about one specific attribute.` | no | **no** |
| `For that, what matters is: X; Y.` | yes | n/a, correctly parsed |

The third is sent whenever the agent recommends without asking and does not
hit. `VisibleMessageInterpreter` in `tikitaka/orchestration/runtime.py` treats
an empty delta as a parse failure and preserves the raw text as a soft `other`
constraint. `carries_no_new_constraint` exists precisely to stop that, and it
does not recognise this template.

The result is visible in `artifacts/traces/intent_override.jsonl`, turns 7-10:

```text
other = those options are not quite right yet. ask me about one specific
        attribute.  [soft, from turn 7]
```

Every subsequent BM25 query searches the catalog for those words. The state
stops changing, and turns 7 through 10 are byte-identical.

## The fix that was tried

Add the template to the no-information vocabulary in `tikitaka/models/fake.py`
alongside the other two, following the existing discipline: suppress only when
the recognised phrase is the entire message, so a private simulator that
combines it with a real requirement still preserves the requirement.

Six-case behavioural check passed, including the combined-message case.

## Why it was not shipped

Measured with `scripts/run_experiment.py` on the frozen arm
(`conservative-questions-deterministic`, profile weight 0, sparse,
deterministic), tuning split, held-out untouched:

| Metric | Baseline | With fix | Delta |
|---|---:|---:|---:|
| Hit Rate@10 | 0.900000 | 0.878571 | **-0.021429** |
| MRR | 0.502738 | 0.499770 | -0.002968 |
| MTTC | 5.657143 | 5.821429 | +0.164286 |
| Technical score | 0.707679 | 0.692788 | **-0.014891** |

The baseline arm reproduces `0.707679` exactly, matching
`reports/p6-production-index-handoff.json`, so this is a clean same-revision
A/B rather than a cross-machine comparison.

**Removing the pollution makes the measured score worse.** The most likely
mechanism: the junk constraint mutated state on every otherwise-empty turn, and
the question-value threshold `0.07` was tuned with that mutation present.
Remove it and the state stops changing on those turns, so the policy repeats
itself instead of being nudged. The threshold is calibrated around the defect.

A correct fix therefore has to re-tune the question-value threshold with the
guard in place. That would replace Person 3's frozen selection, and the single
allowed held-out confirmation has already been spent on the current arm — a
re-tuned threshold could not be held-out confirmed before submission. Shipping
an unconfirmed configuration that also loses on tuning is worse than shipping
a known, documented defect.

## A trap worth recording

Measured on all 200 public sessions the same change looks like an improvement:

| Instrument | Baseline | With fix | Delta |
|---|---:|---:|---:|
| Tuning split, 140 | 0.707679 | 0.692788 | **-0.014891** |
| Full public set, 200 | 0.705672 | 0.708828 | +0.003156 |

The full set includes the 60 reserved sessions. Reading the change as a gain
requires scoring on data the split exists to protect, and it inverts the sign
of the decision. The tuning-split number is the one that governs.

## If this is picked up later

1. Apply the guard in `tikitaka/models/fake.py`.
2. Re-sweep the question-value threshold on the tuning split only.
3. Compare the best re-tuned arm against `0.707679`.
4. Only if it wins, and only with owner sign-off, consider whether a held-out
   confirmation is available.

Do not apply the guard on its own.
