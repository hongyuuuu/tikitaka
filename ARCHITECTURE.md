# TikiTaka Shopping Copilot — Architecture Authority

## 1. Objective

Build the Challenge 4 Shopping Copilot: a multi-turn agent that finds the
customer's hidden purchased product from the frozen 50,000-product Amazon
catalog within at most 10 turns.

The optimization priorities are the official metrics:

1. **Coverage (Hit Rate@10):** retrieve the purchased product within the scored
   Top 10.
2. **Precision (MRR / Top-K Hit Rate):** place the purchased product as high as
   possible, ideally at rank 1.
3. **Efficiency (MTTC):** identify the product with as few clarification turns
   as possible.

A clarification question is valuable only when the missing information can
materially change the ranking. The agent must also determine what information
to ask for next, because its prompts influence the customer's subsequent
request and the information revealed.

## 2. Official execution boundary

- The catalog is read-only and contains 50,000 products.
- A session has no more than 10 turns.
- Although the official contract permits both fields in one response, the
  project policy is mutually exclusive: a `CLARIFY` turn returns one structured
  `ask_attribute` and no recommendations; a `RECOMMEND` turn returns up to 10
  ranked recommendations with `ask_attribute = null`.
- `ask_attribute` is exactly one of `category`, `material`, `color`, `size`,
  `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`.
- The simulator uses the single structured `ask_attribute`, not the prose, to
  choose its reply. Customer-facing prose may sound natural, but must not rely
  on receiving answers to several attributes in one turn.
- The evaluator calls one `Agent` across sessions and creates isolated
  per-session state through `reset(session_id, user_profile)`.
- The official rules permit external or local models; this project selects an
  external API for the generative LLM. Dense retrieval, hybrid retrieval,
  semantic reranking, and the supplied anonymized profile are permitted.
- The organizer may disable network access during final scoring. The submission
  must disclose its API dependency. A deterministic non-LLM contingency may
  produce valid degraded outputs when live API credentials are unavailable.

## 3. Per-turn pipeline

Every call to `respond()` follows this control flow:

```text
user message
  -> LLM understanding and structured state delta
  -> state update and intent-override handling
  -> Buying/Browsing route selection
  -> sparse + dense + structural candidate retrieval
  -> candidate fusion and constraint-aware reranking
  -> over-generality / question-value sensor
  -> one proactive clarification when valuable OR ranked Top 10
  -> memory and measurement update
```

The LLM is a required component of the primary system. Deterministic logic
validates and applies its structured output.

## 4. Model-selection layer

All model-dependent components sit behind provider-neutral interfaces. The
selector covers generative LLMs, embedding models, and rerankers.

The selector routes automatically at runtime. Routing may consider task type,
state confidence, candidate uncertainty, and whether semantic reranking is
needed. Evaluation configurations can pin every route for reproducible
comparison. An embedding route is always coupled to its matching precomputed
product index.

### 4.1 Generative-model route

- **Generative route:** `gpt-5.6-terra` through the main API at `xhigh`
  reasoning.
- No local generative LLM is part of the design.
- API integration and primary-path correctness come first. A deterministic
  non-LLM fallback remains planned backup behavior, but failure engineering is
  not the first implementation priority.

### 4.2 LLM responsibilities

The LLM participates in:

- intent and Buying/Browsing interpretation;
- structured slot and constraint extraction;
- negation and intent-override detection;
- active-state query rewriting;
- selection and natural phrasing of proactive clarification;
- semantic reranking of a retrieved shortlist.

The LLM never searches all 50,000 products directly. Retrieval produces a
bounded candidate set before semantic reranking.

### 4.3 Selection evidence

Every evaluated model configuration records:

- Hit Rate@10, MRR, MTTC, Efficiency, and TechnicalScore;
- per-scenario results for Buying, Browsing, Intent Override, and Boundary;
- prompt, completion, and total token usage;
- latency and estimated monetary cost;
- model, provider, reasoning level, embedding model, and reranker identity.

Accuracy, ranking quality, and fewer useful questions take priority over
latency at this stage. Latency is still measured for the final comparison.
External APIs may receive the conversation, anonymized profile, and bounded
product evidence needed to maximize results; privacy minimization is not a
current optimization constraint. Secrets must still remain outside source,
logs, prompts displayed in reports, and committed configuration.

## 5. Session state

```python
@dataclass
class SessionState:
    session_id: str
    turn: int = 0
    mode: str = "unknown"  # buying | browsing | unknown
    intent_version: int = 1
    active_constraints: dict = field(default_factory=dict)
    constraint_history: list = field(default_factory=list)
    no_preference: set = field(default_factory=set)
    asked_attributes: set = field(default_factory=set)
    shown_by_intent: dict[int, set[str]] = field(default_factory=dict)
    candidate_set: list = field(default_factory=list)
    profile_seed: dict = field(default_factory=dict)
    active_query_summary: str = ""
```

Each constraint carries its attribute, normalized value, hard/soft status,
source turn, confidence, and active/replaced status. The query is constructed
from the active state rather than by concatenating the raw conversation.

## 6. Over-generality and question-value sensor

The agent must actively clarify when the request is too generic, but it must
not ask merely because the retrieved list has a fixed size. The sensor uses:

- active constraint coverage and confidence;
- effective candidate mass and score concentration;
- margin between leading candidates;
- disagreement between sparse, dense, and structural routes;
- attribute uncertainty among competitive candidates;
- the predicted change in Top-10 membership or ordering if an attribute were
  known;
- remaining turn budget and already-asked attributes.

The selected question targets the missing attribute with the greatest expected
effect on the ranking. Repeated or low-value questions are suppressed. A
clarification turn returns no recommendations; the question-value threshold
must therefore justify spending a turn without a hit opportunity.

## 7. Hybrid retrieval

Candidate generation uses three complementary routes:

| Route | Responsibility |
|---|---|
| Sparse | BM25/lexical matches over title, category, features, details, store, and description |
| Dense | cosine or approximate-nearest-neighbor retrieval over precomputed product embeddings and an embedding of the active intent state |
| Structural | filters and boosts for reliable category, material, color, size, brand, budget, feature, and use-case evidence |

Initial fusion uses Reciprocal Rank Fusion so BM25 and cosine scores do not
need to share a numerical scale. Hard filters are applied only when the user
constraint is explicit and the relevant catalog field is reliable; sparse
metadata must not cause valid products to disappear silently.

Browsing gives more weight to semantic coverage and diversity. Buying gives
more weight to explicit constraints and precision.

## 8. Reranking and recommendation selection

The LLM reranker receives only a bounded shortlist with product IDs and compact
evidence. Its structured output is validated against the shortlist. A
deterministic scorer remains available for comparison and later fallback.

Reranking rewards semantic relevance, lexical evidence, explicit constraint
matches, and appropriate soft preferences. It penalizes contradictions and
same-intent repetition. Early vague turns favor coverage and diversity; later
well-specified turns concentrate probability at the top to improve MRR.

Within an unchanged intent version, already-shown products are excluded or
strongly penalized to avoid wasting recommendation positions. When an intent
override creates a new `intent_version`, previously shown products become
eligible again if they fit the new active state.

## 9. Intent override

The LLM emits structured add, remove, replace, no-preference, and reset
operations. Outdated constraints must be erased or rewritten rather than
stacked with their replacements, and retrieval is recomputed from the resulting
active state.

Constraint clearing is dependency-aware:

1. A direct attribute correction replaces only that attribute.
2. An explicit “start over” clears all conversation-derived constraints and
   begins a new intent version.
3. A major category or product-type change begins a new intent version, clears
   constraints inferred from or incompatible with the old category, and keeps
   category-independent constraints such as budget when still applicable.
4. Ambiguous constraints are marked for revalidation rather than silently
   enforced or discarded. The question-value policy may clarify one when it
   would materially change ranking.
5. The supplied profile remains separate from conversation-derived state and is
   not erased by an intent override.

Every new intent version recomputes retrieval from active constraints and makes
previously shown products eligible again when they fit the new intent.

## 10. Demonstration flow

The core story is that a vague request becomes a useful search plan:

```text
CUSTOMER
"I need shoes for a trip."

AGENT
Returns its best current products and asks the single highest-value scorable
clarification.

CUSTOMER
"Water-resistant, comfortable and under $80."

STATE
travel | long walking | water-resistant | comfort | budget <= $80

ACTION
search -> ask when valuable -> remember -> rerank -> Top 10
```

The intended information progression is:

```text
category -> use case -> material -> style -> budget
```

This is an adaptive ordering, not a mandatory questionnaire. A better question
can be more valuable than another unchanged retrieval call, while unnecessary
questions directly harm MTTC.

## 11. Evaluation discipline

- Do not expose `ground_truth` or hidden intent-card data to `Agent`.
- Keep a held-out portion of the 200 public sessions for honest local
  comparison.
- Report aggregate and per-scenario metrics.
- Compare model, embedding, reranking, state-tracking, clarification, and
  deduplication variants through controlled ablations.
- Tune for the official weighted TechnicalScore while preserving the evidence
  needed to explain improvements in Coverage, Precision, and Efficiency.
