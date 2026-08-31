# Person 3 MRR analysis and tuning candidate

## Outcome

Person 3's strongest safe tuning candidate raises HR@10 from `0.900000` to
`0.978571`, MRR from `0.502738` to `0.822687`, and lowers MTTC from
`5.657143` to `4.407143` on the 140-profile tuning split. Its technical score
rises from `0.707679` to `0.867949`.

The improvement is general runtime logic, not a target lookup. The ranker now
rewards complete, rare phrase matches, product-evidence richness, and a bounded
catalog popularity prior to break otherwise weak retrieval-score ties. It still
validates every returned ID against the candidate shortlist, uses profile weight
`0`, and works without network access. The new components default to weight `0`,
so existing frozen behavior remains unchanged unless the named evaluation arm is
selected.

The full offline mapping from public profile ID to target `parent_asin`, title,
category, derived clue types, and split is recorded in
`reports/person3-public-target-audit.json`. It is analysis-only and must never
be imported by Agent runtime code.

## Score and rank changes

The official technical score is:

`0.50 * HR@10 + 0.30 * MRR + 0.20 * efficiency`

The candidate improves every scenario's HR@10 relative to the reproduced
baseline. It ranks 104 targets first instead of 51 and reduces tuning misses
from 14 to 3. Boundary reaches MRR `0.904762`, Browsing reaches `0.844069`,
Buying reaches `0.819494`, and Intent Override reaches `0.746825`.

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| HR@10 | 0.900000 | 0.978571 | +0.078571 |
| MRR | 0.502738 | 0.822687 | +0.319949 |
| MTTC | 5.657143 | 4.407143 | -1.250000 |
| Technical score | 0.707679 | 0.867949 | +0.160270 |

The reproducible arm is
`mrr-evidence-popularity-011-deterministic`, with fingerprint
`7fd7ee8006e5d24ea0f0a7cd075dbb3df583daa3ed6d7bae0920e772eed40fdc`.

## Why the popularity prior works

An offline candidate-pool audit shows that retrieval exposes 137 of 140 tuning
targets to Person 3, an oracle pool recall and oracle MRR of `0.978571`. Only
three targets never reach any recommendation candidate pool. Ranking, rather
than retrieval coverage, is therefore the main tractable bottleneck.

The catalog's `rating_number` is a legitimate, participant-visible purchase
prior: the tuning targets have a median rating count of `5,910` compared with
`12` for the full catalog. The ranker applies its bounded log-scaled popularity
signal only after the semantic evidence features; it cannot bypass hard
constraint checks or return products outside the validated candidate shortlist.

Some public dialogues still do not identify one unique product. In
36 of 47 inspected target-present low-rank turns, the hidden target ties for the
best phrase score with other catalog products. Some intent cards provide only
generic clues such as a category, common material, and a broad feature. Many
catalog rows share exactly those visible facts. Choosing the hidden target
first in those cases would require unavailable information or memorizing public
labels, which would overfit the 200 profiles and likely fail the 800 private
profiles.

The achieved overall `0.822687` exceeds the target without target-ID lookup or
held-out probing. Future improvements should focus on richer visible state
extraction, better candidate evidence from retrieval, and a valid selective-LLM
comparison rather than more tuning against public target IDs.

## Implemented scoring evidence

The evidence phrase feature compares active, sufficiently confident session
constraints with each candidate's bounded evidence text. It combines rare-token
coverage, ordered bigram coverage, and exact multi-token phrase coverage.
Low-confidence canned fallback replies are ignored.

The evidence specificity feature rewards candidates whose shortlist evidence
contains more unique descriptive terms, matched fields, and known structured
attributes. Its bounded logarithmic score breaks ties without allowing long
descriptions to dominate.

The popularity feature receives the catalog's public `rating_number` as
read-only route evidence. It uses `log1p(rating_number)` normalized within the
shortlist, so it is a conservative tie-breaker rather than an absolute ranking
replacement. The independently reproduced arm uses weight `0.11`.

The ranker also supports independent sparse/dense route-rank evidence and the
question policy supports guarded answerability priors. Both remain available,
but neither is enabled in the recommended candidate because it did not improve
the HR-first tuning result.

## Validation and release decision

The 60-profile held-out split was not reopened. No provider API credential was
available, so this run evaluates the required deterministic degraded path; it
does not claim an LLM result. All new behavior has focused unit coverage,
including malformed/low-confidence evidence and default-off compatibility.

The frozen release policy is unchanged because promoting a tuning winner
without untouched confirmation would violate the evaluation contract. Run the
single recommended arm once on the next untouched or private profile set, using
HR@10 as the first gate, MRR second, and MTTC third. Full reproducible metrics,
per-scenario results, catalog checksums, and limitations are recorded in
`reports/p6-person3-mrr-analysis.json`.
