# TikiTaka Repository Instructions

## Mission

Build the TikTok TechJam 2026 Challenge 4 Shopping Copilot: a reproducible,
headless Python agent that identifies the hidden purchased product from the
frozen 50,000-product catalog within at most 10 turns.

Optimize the official objective in this order of influence:

1. Hit Rate@10 / catalog coverage.
2. MRR / exact product rank.
3. MTTC / useful turns to conversion.

The system is an LLM-assisted, stateful, hybrid retrieval pipeline. It is not a
UI project, a full recommender-model training project, or a production vector
database project.

## Authority order

When sources disagree, follow this order:

1. Official competition contracts and constraints:
   - `docs/agent_api_contract.json`
   - `docs/competition_specification.md`
   - `docs/evaluation_config.json`
   - `docs/submission_rules.md`
2. Settled owner decisions in `ARCHITECTURE.md`.
3. Work breakdown and decision gates in `docs/IMPLEMENTATION_PLAN.md`.
4. Existing implementation and tests.

Do not silently resolve a conflict. Record the conflict in the implementation
plan and ask the owner before changing an already-settled rule.

## Non-negotiable competition boundaries

- Never modify `evaluator/`, `data/public_set.jsonl`, public labels, or the
  frozen catalog to improve a score.
- Never expose `ground_truth`, `scenario_type`, hidden intent-card data, or
  evaluator internals to `Agent.reset()` or `Agent.respond()`.
- The Agent may receive only the official inputs: `session_id`, the supplied
  anonymized `user_profile`, `user_message`, `turn`, and `top_k`.
- The catalog is read-only. Return only catalog-valid `parent_asin` values.
- Only the first 10 unique valid recommendations are scored, in returned order.
- A session ends at a valid hit or after turn 10.
- `ask_attribute` must be one contract value or `null`. The simulator uses the
  structured value, not the prose, to choose its reply.
- Treat every session as independent. Do not create cross-session user memory
  or infer a stable user identity.
- Keep catalog retrieval in memory. Do not add an infrastructure-heavy external
  vector database.
- Use text and structured metadata only. Do not add multimodal processing.
- Do not full-parameter fine-tune a foundation model.
- Never commit secrets. API keys and endpoints come from environment variables.
- Assume final scoring may disable network access. Disclose the API dependency
  and preserve a valid deterministic degraded path; do not add a local
  generative LLM.
- Keep participant code compatible with Python 3.10 or later.

## Settled technical direction

- A generative LLM is required by this project even though a paid API is not
  required by the official challenge.
- The only generative route is `gpt-5.6-terra` through the main API with
  `medium` reasoning. Do not add or plan a local generative LLM.
- Model selection covers generative LLMs, embedding models, and rerankers.
- Model selection routes automatically at runtime, while evaluation
  configurations may pin every route for reproducible comparisons. Never mix a
  query embedding with a product index built by another embedding model.
- The LLM performs intent interpretation, structured state-delta extraction,
  query rewriting, clarification planning, and shortlist semantic reranking.
- Deterministic code validates LLM output and owns state mutation, catalog
  validity, hard constraints, turn limits, and output normalization.
- Retrieval combines sparse/BM25, dense embeddings, and reliable structured
  evidence.
- Buying and Browsing are runtime modes inferred from visible conversation;
  never read the public `scenario_type` label.
- Accuracy, exact rank, and fewer useful questions take priority over latency
  during initial development. Still measure latency, tokens, and cost.
- The supplied profile is a session-local input snapshot, not persistent user
  memory. It may contribute only as a soft, decaying signal whose weight is
  chosen through held-out ablation; explicit session statements always win.
- Each turn chooses one action. `CLARIFY` returns an allowed `ask_attribute`
  with no recommendations; `RECOMMEND` returns ranked products with
  `ask_attribute = null`.
- Intent overrides use dependency-aware clearing: replace only an explicitly
  corrected attribute, fully clear conversation-derived state only for an
  explicit restart, and on a major category change remove incompatible or
  category-derived constraints while preserving still-applicable universal
  constraints such as budget. Revalidate ambiguous constraints before use.

## Shared domain contracts

Define shared contracts before implementing providers. Keep them in a small
dependency-light module owned by Person 4; changes require acknowledgment from
every affected owner.

Minimum contracts:

```python
Constraint
    attribute
    value
    polarity       # include | exclude
    strength       # hard | soft
    source_turn
    confidence

StateOperation
    operation      # add | remove | replace | exclude | no_preference | reset
    attribute
    old_value
    new_value
    scope

StateDelta
    inferred_mode  # buying | browsing | unknown
    mode_confidence
    operations
    generality

Candidate
    parent_asin
    product_evidence
    sparse_rank
    dense_rank
    structural_score
    fused_score

TurnDecision
    action         # clarify | recommend
    ask_attribute
    reason
    expected_information_gain
```

LLM output is untrusted input. Parse it through a strict schema, reject unknown
operations and attributes, clamp confidence values, and never accept product
IDs that were not in the validated shortlist.

## Four-person ownership

Each person owns a workstream and its tests. Avoid concurrent edits to another
owner's files; integrate through the shared contracts.

| Person | Workstream | Primary ownership |
|---|---|---|
| 1 | Model gateway and conversation state | `tikitaka/models/`, `tikitaka/state/`, matching tests |
| 2 | Catalog processing and hybrid retrieval | `tikitaka/retrieval/`, preprocessing scripts, matching tests |
| 3 | Decision policy and reranking | `tikitaka/decision/`, `tikitaka/ranking/`, matching tests |
| 4 | Contracts, orchestration, evaluation, integration and release | `tikitaka/contracts/`, `tikitaka/orchestration/`, `tikitaka/evaluation/`, `starter/agent.py`, integration tests and submission docs |

`ARCHITECTURE.md`, `AGENTS.md`, official files, dependency manifests, and shared
contracts are coordination surfaces. Changes to them must be announced to all
four owners before merging.

## Integration rules

1. Work contract-first. Person 4 lands interfaces and fake implementations
   before provider-specific modules are integrated.
2. Every module must be testable with a tiny synthetic catalog and fake model;
   unit tests must not require network access or a 27B model.
3. Keep the official `starter/agent.py` thin. It should construct and delegate
   to the internal orchestrator.
4. Keep model adapters replaceable. Retrieval, state, and evaluation code must
   not import a provider SDK directly.
5. Keep retrieval deterministic for a fixed index, query, and configuration.
6. Version index metadata with the catalog checksum, text-construction version,
   embedding model, dimensionality, and normalization setting.
7. Return evidence with candidates so reranking and failure analysis can explain
   why a product was retrieved.
8. Do not place all 50,000 products in an LLM prompt. Retrieve a bounded
   shortlist first.
9. Keep hard filters conservative when catalog metadata is missing. An absent
   field is not proof of a contradiction.
10. Track prompt/completion tokens, latency, model identity, provider, reasoning
    level, and estimated cost for each evaluated model call.
11. Do not optimize on all 200 public targets. Maintain a held-out subset and
    report it separately from tuning results.
12. Never merge a score improvement without its configuration and per-scenario
    results. Aggregate improvements may conceal an Override or Boundary failure.

## Required tests

At minimum, cover:

- Official Agent output schema and normalization.
- State accumulation, retraction, replacement, exclusion, no-preference and
  full reset.
- Buying-to-Browsing and Browsing-to-Buying mode changes.
- Explicit Intent Override and intent-version increment.
- Boundary reply handling without repeated questions.
- Over-generality detection and question suppression when information gain is
  low.
- Sparse, dense and structured candidate retrieval.
- Rank fusion determinism and valid-ID enforcement.
- Hard constraint enforcement with missing-metadata safeguards.
- Same-intent recommendation deduplication.
- Previously shown product eligibility after an intent-version change.
- Malformed, timed-out and hallucinated model output.
- API usage accounting and environment-only credentials.
- Network-free deterministic degraded execution before final submission.
- A guard proving `Agent` cannot access `ground_truth` or `scenario_type`.

Run the official tests with:

```bash
python3 -m unittest
```

Run the official evaluator only against an unchanged evaluator and dataset:

```bash
python3 -m evaluator.local_evaluator
```

Add project-specific commands to the README only after they exist and are
verified.

## Definition of done

A feature is done only when:

- its contract is documented;
- unit tests cover normal, boundary and malformed input;
- it runs without exposing labels or secrets;
- it records the configuration needed to reproduce its result;
- it integrates through the orchestrator without modifying official scoring;
- its held-out aggregate and per-scenario effects are reported; and
- its limitations and fallback behavior are explicit.
