# Person 1 — P6 Status: Routing, Pinning, Ablation Support, Freeze

Owner: Person 1. Phase reference: `docs/PERSON_1_BUILD_PLAN.md` section 11.

## Exit gate

| Requirement | Status |
|---|---|
| The same pinned configuration reproduces the same validated state | met |
| An index/route mismatch is impossible to reach silently | met |
| Routing decision and its reason are recorded for Person 4 | met |
| Every route pinnable for a reproducible run | met |
| `PROMPT_VERSION` and `SCHEMA_VERSION` frozen and reported | met |
| Ablation hooks for profile weight, rewrite, interpretation, reasoning | met |
| Structured traces exist for the four scenarios | met |
| Routing reason recorded durably for Person 4 | met |

## What routing can actually see

`IntentInterpreter` is called as `interpret(message, state)` **before** retrieval
runs. At routing time there are no candidates and no ranking scores. The build
plan lists "candidate uncertainty from Person 3" as a routing input; that value
does not exist at the point where the decision is made, and a router that asked
for it would have silently never fired. The routing inputs are therefore what
`SessionStateView` and the message actually provide:

| Signal | Source |
|---|---|
| `mode_confidence` | `state.mode_confidence` |
| `remaining_turns` | `MAX_TURNS - state.turn` |
| `constraint_count` | `len(state.active_constraints)` |
| `observed_turns` | `state.turn` |
| `override_suspected` | regex on the message |

**This is a deviation from section 11 of the build plan and is deliberate.**
Candidate uncertainty can only be a routing input for a task that runs after
ranking — a rerank or a follow-up turn — not for interpretation.

### The opening-turn trap

The interpreter runs before the current turn is reduced, so on turn 1 every
session alive has `mode_confidence == 0.0` and zero constraints. An earlier
draft escalated on those, which is not a signal at all: it is a provider call on
turn 1 of all 200 sessions, dressed up as a measurement. `RoutingSignals.has_evidence`
now gates both, and a test asserts the opening turn does not escalate while an
opening-turn *override* still does.

## Routing policy, and why the default is what it is

`RoutingThresholds.always_generative` defaults to **True**: if a generative route
is configured, it handles every turn. That is exactly what `build_agent` did
before routing existed, so this phase changes no behaviour by default.

`SELECTIVE` is the cost-saving policy and it is fully built and tested — engage
the LLM only on an override turn, an unconfident mode, a state nothing was
extracted from, or a nearly spent turn budget. It is **not** the default, and
that is the point: turning it on would be a quality decision made with zero
measurement of the API route. Risk 5 in the build plan names this trap directly.
Switch it on when the live run says what the API route is worth.

Escalation reasons are checked in fixed precedence, so identical signals always
record the same reason. Two runs that disagree on *why* they escalated are not
comparable even when they picked the same route.

## Reproducibility

`routing_mode` reports `pinned` only when **every** task is pinned. A partial pin
is still runtime routing — something is free to vary, and calling that run
reproducible would be a lie in the report.

Verified end to end: two independent `build_agent` builds with the same pins,
run over the same four-turn conversation, produce byte-identical responses and
identical resulting session state (intent version, mode, constraints, asked
attributes).

## Index/route refusal

`SearchPlan` already requires `embedding_route_id` and `index_id` to be set
together, but *present* is not *correct*. A query embedded by one model against
an index built by another produces scores that look entirely ordinary and are
meaningless. `ModelSelector.select_embedding` raises `RouteMismatch` on a
differing index **and** on a route carrying no index identity at all — absence
of identity is a refusal, not a waiver, because the unverifiable route is the
one most likely to reach a run unnoticed.

Person 2 owns the index and its manifest validation; this is the route-level
half of the same guarantee.

## Regression evidence

The composition root changed, so the deterministic path was re-measured rather
than assumed. Full official evaluator, no credential in the environment:

```text
metric           P5 (docs/p5)   P6         delta
hit_rate@10      0.8950         0.8950     0.0000
mrr              0.5424         0.542393   0.0000
mttc             5.74           5.735      0.000
efficiency       0.5265         0.5265     0.0000
TechnicalScore   0.7155         0.715518   0.0000

reported tokens  prompt 0, completion 0
```

Per scenario, unchanged: browsing 0.9875, buying 0.8625, boundary 0.9000,
intent_override 0.7333.

Test suite: 355 tests, zero new failures. The 9 remaining errors are the
pre-existing Windows `np.memmap` handle leak in `retrieval/dense.py` (Person 2),
present at baseline and unrelated to this phase.

## Changes to another owner's file

`tikitaka/orchestration/runtime.py` is Person 4's. The diff is small and
deliberately behaviour-preserving:

- `RuntimeConfig` gains `selector: ModelSelector | None = None`.
- `build_agent` composes `RoutingInterpreter` instead of choosing one route at
  build time. A degraded selector has no generative route, so it always chooses
  the fallback and the deterministic path is unchanged — which the 0.715518
  above demonstrates rather than asserts.
- `ResilientInterpreter` is left in place and still exported. `RoutingInterpreter`
  subsumes it, but removing a public class from another owner's module is his
  call, not mine.

Three of Person 4's tests in `tests/test_runtime_integration.py` encode the
assumption that the primary interpreter runs on every turn. That assumption is
now explicit in `RoutingThresholds.always_generative` and preserved by default;
his tests pass unchanged.

**Note for Person 4:** `runtime.py` defines `_VISIBLE_OVERRIDE_RE`, and
`selector.py` now has an equivalent `looks_like_override`. Two owners carrying
the same regex will drift. Proposing his module import mine.

## Frozen for submission

| Field | Value |
|---|---|
| `PROMPT_VERSION` | `intent-interpreter/1` |
| `SCHEMA_VERSION` | `0.1.0` |
| Primary route | `primary/gpt-5.6-terra`, reasoning `xhigh` |
| Fallback route | `heuristic/local` |

`ModelSelector.identity()` returns these plus the ablation state, keyed to match
`ExperimentConfig` field names. `selector_from_env()` builds the selector matching
what the environment can actually reach, so with no credential `identity()`
reports `degraded` rather than naming a model no call could have used.

## Traces, and the three defects they exposed

`capture()` had never been called outside tests, so the four scenario traces the
definition of done requires did not exist. `scripts/capture_traces.py` produces
them, mirroring `evaluator.local_evaluator.evaluate` including its break on
first hit, and refusing to write if any evaluator label or the hidden target
reaches a trace. Output is committed under `artifacts/traces/`.

Writing the artifact once found three real bugs that tests had not:

1. **`SessionState.active_query_summary` was dead.** Declared, read by the
   trace, written by nothing — every trace carried an empty string where the
   query belongs. Field removed; the summary is derived in `trace.py`, which
   keeps the read-only state read-only.

2. **`exhausted_attributes` was empty in every trace.** `state/extractor.py`
   owns the spent-question/no-preference distinction and is *not in the running
   agent* — the live path is `ShoppingAgent` → `RoutingInterpreter` →
   `VisibleMessageInterpreter` → reducer. `RoutingInterpreter` now carries the
   note, since it is the only Person 1 component in that path that sees both the
   message and the session. Exhaustion stays distinct from no-preference.

3. **No-information replies were becoming search constraints.** An empty
   `StateDelta` means either "understood, nothing to add" or "failed to parse".
   `VisibleMessageInterpreter` reads it as the second and injects the raw
   message, so a Boundary reply searched on the customer's refusal to state a
   preference. `carries_no_new_constraint()` lets a caller tell the two apart;
   the guard itself is Person 4's file and went to him as PR #21.

### The measurement that matters

Removing the pollution **costs 0.0098** on the public set:

| Variant | TechnicalScore |
|---|---|
| Raw sentence injected (current behaviour) | 0.715518 |
| Same constraint, meaningless placeholder text | 0.713204 |
| Constraint removed (correct behaviour) | 0.705672 |

77% of the gap is the constraint merely existing, not its text. It is a count
effect, not semantic matching: the extra constraint pushes the policy from
CLARIFY to RECOMMEND, and under DG-01 a CLARIFY turn is a guaranteed miss.

The reading is that **the agent over-clarifies by roughly 0.010 of score, and
accidental pollution has been masking it.** That gain is real and recoverable in
Person 3's threshold; the pollution is not, because it depends on one public
template failing to match a regex. Risk 5, live.

### Two paths that disagree

`state/extractor.py` and the live interpreter chain both claim to own message
ingestion, and only one of them runs. That is how defect 2 survived. The
extractor is left in place because a lot of test coverage depends on it, but the
duplication should be consolidated before submission rather than carried.

## What remains

Cost rates in `ApiConfig` are still `0.0`, so `estimated_cost` is structurally
zero and the M6 cost disclosure cannot be produced. Real per-token rates for
`gpt-5.6-terra` are an owner input, not something to estimate.

The ten-session live run, still deferred to avoid cost during the build phase.
It is what decides whether `SELECTIVE` should be switched on and whether the API
route is worth its budget. Cost rates in `ApiConfig` are still `0.0` and need
filling before that run is reportable.
