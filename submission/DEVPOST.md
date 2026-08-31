# TikiTaka — Conversational Shopping Copilot

**Challenge 4: Conversational E-Commerce Search.** Find a customer's hidden
target product inside a frozen 50,000-product Amazon catalog within at most 10
conversational turns, asking as few questions as possible along the way.

## Demo video

<!-- Replace with the public YouTube watch URL before submitting. -->
**▶ https://youtube.com/watch?v=REPLACE_ME**

A three-minute walkthrough: one full session replayed turn by turn, the scored
run over all 200 public sessions, and the clean-room audit that reproduces the
result with the API credential deleted and network access blocked.

---

## How our solution addresses the problem statement

The problem statement is a search problem wearing a conversation's clothes. The
customer starts vague ("I need shoes for a trip"), the target is one exact
`parent_asin` out of 50,000, and every turn spent asking a question is a turn
not spent recommending. Scoring rewards three things in tension: coverage
(Hit Rate@10), precision (MRR), and efficiency (MTTC). Asking more questions
helps the first two and hurts the third.

TikiTaka addresses this with four design commitments:

**1. State, not transcript.** We never concatenate the raw conversation into a
query. Each visible message is converted into a validated *state delta* —
structured `add` / `remove` / `replace` / `exclude` / `no_preference` / `reset`
operations over typed constraints (attribute, value, polarity, hard/soft
strength, source turn, confidence). A deterministic reducer applies them and
rebuilds the query from the resulting active state. This is what makes the
Intent Override scenario tractable: when a customer replaces an earlier
preference on turn 3, the stale constraint is *erased*, not stacked on top of
its replacement. Clearing is dependency-aware — a direct correction replaces
one attribute, an explicit restart clears all conversation-derived state, and a
major category change drops category-derived constraints while keeping
still-applicable universal ones such as budget. Each override increments an
`intent_version`, which also makes previously shown products eligible again if
they fit the new intent.

**2. Hybrid retrieval, not one index.** Candidate generation runs three
complementary routes and fuses them with Reciprocal Rank Fusion, so BM25 and
cosine scores never need a shared numerical scale:

| Route | Responsibility |
|---|---|
| Sparse | SQLite FTS5/BM25 over title, categories, features, details, store, description |
| Dense | exact cosine over a catalog-pinned float32 embedding index |
| Structural | filters and boosts on reliable category, material, color, size, brand, budget, feature, use-case evidence |

Hard filters apply only when the customer constraint is explicit *and* the
catalog field is reliable, so sparse metadata can never silently delete a valid
product. Browsing weights semantic coverage and diversity; Buying weights
explicit constraints and precision.

**3. Ask only when the question changes the ranking.** The common failure mode
is an agent that interrogates the customer because the result list is long. Our
question-value sensor asks only when the expected information gain is real. It
scores constraint coverage and confidence, effective candidate mass and score
concentration, the margin between leading candidates, disagreement between the
sparse/dense/structural routes, attribute uncertainty among competitive
candidates, the predicted change in Top-10 membership or ordering if an
attribute were known, and the remaining turn budget against already-asked
attributes. Each turn commits to exactly one action: `CLARIFY` returns one
structured `ask_attribute` and no recommendations; `RECOMMEND` returns up to 10
ranked products with `ask_attribute = null`. Because a clarification turn
forfeits a hit opportunity, the threshold has to earn the turn — and that
threshold was chosen by held-out sweep, not by taste.

**4. LLM in the loop, deterministic code in control.** The model interprets
intent, extracts state deltas, detects negation and overrides, rewrites the
active query, phrases clarifications, and semantically reranks a *bounded*
shortlist. It never sees all 50,000 products and it never mutates state
directly. Every model output is treated as untrusted and validated against the
catalog and the shortlist before it can affect anything. That boundary is also
what makes the system survive a network-disabled judging environment: with no
credential, the same pipeline runs on its deterministic heuristic route and
still emits contract-valid responses.

### Results

Reproduced on the 200-session public evaluator with `OPENAI_API_KEY` removed
and Python socket/DNS access denied by an audit hook — i.e. the pessimistic,
fully offline route, imported from an extracted submission bundle rather than
the source tree:

| Metric | Weak BM25 baseline | TikiTaka (offline route) |
|---|---:|---:|
| Hit Rate@10 | 0.125 | **0.885** |
| MRR | 0.068034 | **0.529240** |
| MTTC | 9.81 | **5.78** |
| Efficiency | 0.119 | **0.522** |
| TechnicalScore | 0.10671 | **0.705672** |

Per scenario: Browsing 0.9875 HR@10, Buying 0.8625, Boundary 0.800, Intent
Override 0.700. Network attempts, agent exceptions, and raw response contract
violations were all zero in that run.

Two disclosures we would rather make in our own words than have discovered: the
`medium` reasoning default has **not** been measured live, so we make no live
quality, latency, or cost claim for it — the API figures in our report are
labelled historical `xhigh` evidence, measured against provider billing
($18.59 for 3.086M input / 1.035M output tokens, 6.6 s mean latency). And our
client-side token accounting understated that bill by 4.8x; we could not
establish the cause, so we report the billed numbers and tell people not to
budget from our instrumentation. The production 1024-dimensional dense artifact
is likewise not built, so the scored result above is sparse/structured
retrieval only — the dense route is implemented, tested, and manifest-pinned,
but we are not claiming a number we did not measure.

---

## Development tools used

- **Python 3.10+** — the entire system; no build step, no service to stand up.
- **Git / GitHub** — four-owner repository with PR-per-milestone review; branch
  ownership split across models/state, retrieval, decision policy, and
  release/contracts.
- **Claude Code** (Opus 5) and **OpenAI Codex** — AI pair-programming across
  the four owners, governed by a checked-in `AGENTS.md` that encodes the
  competition's non-negotiable boundaries (never modify the evaluator, never
  leak `ground_truth` or `scenario_type` into the agent, never commit secrets).
- **Python `unittest`** — 44 test modules run as `python3 -m unittest`,
  including fault-injection, route-equivalence, and prompt-contract suites.
- **SQLite (FTS5)** — bundled with CPython; our BM25 index runs in-memory, so
  there is no external search service to deploy.
- **Custom release tooling** — `scripts/build_submission.py` and
  `scripts/verify_m6_submission.py`, which rebuild the bundle and re-score it in
  an isolated temp harness with the credential forcibly removed and socket/DNS
  calls blocked by a Python audit hook, then emit a JSON audit recording code
  revision, file hashes, package policy, and metrics.
- **Local evaluator** (`evaluator/local_evaluator.py`, organizer-supplied) —
  deterministic 200-session simulator used for every ablation.

## APIs used

- **OpenAI Chat Completions API** — `gpt-5.6-terra` at `medium` reasoning
  effort, called over plain HTTPS. This is the single generative route: intent
  interpretation, structured state-delta extraction, override detection, query
  rewriting, clarification phrasing, and shortlist reranking. Optional by
  design — an absent or failing credential degrades to the deterministic route.
- **OpenAI Embeddings API** — `text-embedding-3-large` pinned at 1024
  dimensions for the dense product index. The integration, batching, manifest
  binding, and query-side embedder are implemented and tested; the production
  50,000-document index build (~14.5M input tokens across ~196 requests) is
  costed but not yet run, so the submitted result does not depend on it.

No other external API is called. Credentials come from environment variables
only and appear in no source file, config, log, or report.

## Libraries and frameworks used

**Runtime dependencies: none.** `submission/requirements.txt` deliberately
lists no packages. The agent runs on the Python standard library alone, which
is what lets the bundle drop into an unknown judging environment with no
install step and no version conflict.

Standard library carrying real weight:

- `sqlite3` (FTS5) — BM25 sparse retrieval
- `urllib.request` / `urllib.error` — the entire HTTP client for both APIs,
  written by hand rather than pulling in an SDK, so the agent adds zero
  dependency surface
- `array`, `math` — float32 dense vector storage and exact cosine search
- `hashlib`, `json` — catalog/index/artifact checksums and manifest binding
- `dataclasses`, `typing`, `pathlib`, `decimal`, `collections`, `re`,
  `unicodedata` — contracts, text normalization, cost arithmetic
- `socket`, `zipfile`, `subprocess`, `importlib` — release verification and the
  network-denial guard

Optional, developer-side only (never imported on the scored path):

- **NumPy** — an alternative `numpy-exact` backend for dense search and index
  building; the pure-stdlib backend is the default.
- **tiktoken** — used by `scripts/estimate_cost.py` for offline token and cost
  estimation.

No ML training framework is used. We do not fine-tune, and there is no
PyTorch / TensorFlow / Transformers dependency — retrieval quality comes from
hybrid search, state management, and reranking rather than from a model we
trained ourselves.

## Datasets and assets used

- **Amazon Reviews 2023**, McAuley Lab, UCSD — https://amazon-reviews-2023.github.io/
  The organizer-frozen competition package derives from the
  `Clothing_Shoes_and_Jewelry` category, joined on `parent_asin`.
  - **Frozen catalog:** 50,000 products, read-only, text and structured
    metadata only. Visible fields: `parent_asin`, `title`, `features`,
    `description`, `price`, `categories`, `details`, `average_rating`,
    `rating_number`, `store`. Its SHA-256 is pinned in our release audit
    (`da979b05…c69a67`) and bound into every dense index manifest, so a
    mismatched catalog fails closed rather than silently scoring against the
    wrong corpus.
  - **Public session set:** 200 labeled development sessions
    (`data/public_set.jsonl`, SHA-256 `857259f7…8f7579`) across the fixed
    scenario mix — 40% Buying, 40% Browsing, 15% Intent Override, 5% Boundary.
    We held out a portion of these for honest comparison and tuned thresholds
    on the held-out split rather than on the full set.
  - **Private sessions:** 800, retained by the organizer. Never seen by us.
  - **Anonymized `user_profile`:** the only user-side input, containing
    purchase-frequency and rating summaries plus controlled preference tags. We
    treat it as a session-local snapshot and a soft, decaying signal whose
    weight was chosen by held-out ablation; an explicit statement in the
    conversation always overrides it. No cross-session memory is built.

- **No manually labelled data, no scraped data, no external corpus.** We added
  no annotations to the competition data and used no outside product source.

- **Synthetic test fixtures** — small hand-written catalogs and fake model
  clients under `tests/fixtures/` and `tests/fakes/`, used to exercise retrieval
  mechanics, fault handling, and token accounting deterministically without
  network access or the real catalog. Fixture timings are mechanics evidence
  only and are never reported as production performance.

- **Deliberately absent from our bundle:** the catalog, the evaluator, public
  labels, the generated dense index, and any credential. The submitted archive
  is 68 files / 155 KB of source, checked against a forbidden-contents policy on
  every build.

## Known limitations

- The offline route lacks the semantic recall of a production dense index and
  the interpretation quality of the API route; the scored 0.7057 is the
  *floor*, achieved with the model switched off.
- The `medium` API default carries no live measurement — no score, latency,
  token, or cost claim.
- Client-side token accounting is known to understate provider billing; use the
  billed historical figures for budgeting.
- Public-set metrics are development evidence and do not predict private-set
  performance.
