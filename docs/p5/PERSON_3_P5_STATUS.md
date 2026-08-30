# Person 3 — P5 Fixed-Ask and Policy Selection Status

Branch: `person3/p5-fixed-ask-baseline`

## Implemented locally

- Added an explicit contract-order question-selection strategy. It computes the
  same eligibility and expected ranking-change evidence as the adaptive policy,
  but selects the first eligible attribute in `ALLOWED_ATTRIBUTES` order rather
  than the attribute with the largest estimated gain.
- Added `fixed-ask-baseline` at the current post-no-information threshold
  `0.07`, with deterministic reranking, profile weight `0`, and no model call.
- Suppressed exhausted attributes in question selection. No-preference and
  exhausted attributes remain permanently ineligible; an explicit revalidation
  may reopen an answered or previously asked attribute.
- Preserved the fingerprints of all previously reported Phase 4 arms. The new
  selector is omitted from fingerprint payloads only when it is the historical
  default, whose behavior is unchanged.
- Added the pre-registered post-no-information threshold grid `0.05`, `0.06`,
  `0.07`, `0.08`, and `0.09`.
- Added a threshold-matched fixed-order arm and an otherwise identical anchored
  LLM-reranking arm for every threshold. This lets reports change one axis at a
  time after the tuning threshold is selected.

## Executable comparison ladder

For each threshold, the P5 harness can now run:

| Threshold | Adaptive deterministic | Fixed order | Anchored LLM reranker |
|---:|---|---|---|
| 0.05 | `post-no-info-threshold-050-deterministic` | `fixed-ask-threshold-050` | `post-no-info-threshold-050-llm-anchored` |
| 0.06 | `post-no-info-threshold-060-deterministic` | `fixed-ask-threshold-060` | `post-no-info-threshold-060-llm-anchored` |
| 0.07 | `conservative-questions-deterministic` | `fixed-ask-baseline` | `conservative-llm-anchored` |
| 0.08 | `post-no-info-threshold-080-deterministic` | `fixed-ask-threshold-080` | `post-no-info-threshold-080-llm-anchored` |
| 0.09 | `post-no-info-threshold-090-deterministic` | `fixed-ask-threshold-090` | `post-no-info-threshold-090-llm-anchored` |

The existing `always-recommend-baseline` remains the never-ask control.

## Evaluation discipline

1. Run only the five adaptive deterministic threshold arms on the tuning split.
2. Freeze the winning threshold by Hit Rate@10, then MRR, then MTTC, subject to
   the existing per-scenario collapse safeguard.
3. At that threshold, run the matching fixed-order control and the deterministic
   versus anchored-LLM reranker comparison while holding retrieval, interpreter,
   question policy, shortlist, and profile weight constant.
4. Freeze the finalist set before using `--stage held_out --confirm-held-out`.
5. Keep profile weight `0`. A non-zero override is not held-out eligible unless
   it first beats `0` on tuning with the selected question policy.
6. Record candidate Recall@N, conditional shortlist MRR, invalid/duplicate ID
   rate, hard-constraint violations, usage, latency, cost, and all four scenario
   results before promoting a model reranker.

Person 4 owns executing and reporting these arms through
`scripts/run_experiment.py`; Person 3 does not modify evaluator or orchestration
code. LLM arms require an explicitly authorized API run.

## Local evidence and remaining gates

- Focused decision, ranking, P5-arm, and runtime integration suite: `112` tests
  passed after implementation.
- Repository-wide suite: `465` tests ran with no Person 3 regression. The `10`
  errors are the existing Person 2 Windows `vectors.f32` memory-map cleanup
  failures; one unrelated test is skipped. Excluding those six affected
  retrieval modules, `428` tests pass with the same one skip.
- The full 50,000-product catalog is not present in this workspace, so no new
  tuning or held-out score is claimed from this branch.
- The committed post-no-information reports remain tuning-only. Held-out must
  not be opened until the threshold sweep and finalist set are frozen.
- The primary API versus degraded-path quality delta remains pending a real,
  cost-attributed API run.
