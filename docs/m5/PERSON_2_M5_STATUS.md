# Person 2 — M5 Network-Degraded Retrieval Status

Evidence revision: `30c9f980aa1a112e10296ed7920a7597bc6fc3a3`

Status: the network-free runtime and artifact-failure gates are met. The
numerical quality delta versus the primary hybrid route remains pending because
the real `text-embedding-3-large` 1024-dimensional artifact has not been built.
Fixture embeddings are not substituted for production evidence.

## Exit-gate evidence

| Requirement | Status | Evidence |
|---|---|---|
| Force API/embedding unavailability | met | `OPENAI_API_KEY` removed for the run |
| Prevent network use | met | Python audit hook denied socket connect, DNS and send events; zero attempts |
| Complete the official 200-session evaluator | met | 200 sessions, all four scenarios |
| Use the deterministic route | met | `heuristic/local`, `degraded: true`, zero model tokens |
| Return contract-valid raw outputs | met | 1,133 responses, zero violations and zero Agent exceptions |
| Missing dense artifact is recoverable | met | `dense_artifact_unavailable` |
| Corrupt dense artifact is recoverable | met | `dense_artifact_invalid` |
| Quality delta versus primary hybrid | pending | requires the production 1024-dimensional artifact and primary run |

The machine-readable evidence is `reports/m5-offline.json`. It records the
catalog, dataset and code identities rather than relying on a prose claim.

## Why the dedicated runner exists

The official evaluator catches Agent exceptions and truncates recommendation
lists before scoring. A process exit code of zero therefore cannot prove that
the raw Agent stayed valid or that a failed network request was not hidden as a
miss. `scripts/run_m5_offline.py` adds three fail-closed checks around the
unchanged evaluator:

1. a Python audit hook refuses and counts network connection, DNS and send
   events;
2. an observer validates every raw response before evaluator normalization;
3. the report is accepted only from a clean committed revision with the
   credential absent and the deterministic route selected.

The observer exposed a real orchestration defect during the first run: 124 raw
responses contained 11 recommendations. The evaluator silently kept the first
10. `ShoppingAgent._normalize_ids` now checks the limit before appending fallback
candidates, and a regression test covers a full ten-item reranker result that
omits the first fallback candidate.

## Offline metrics

| Metric | Result |
|---|---:|
| Hit Rate@10 | 0.885000 |
| MRR | 0.529240 |
| MTTC | 5.780000 |
| Efficiency | 0.522000 |
| TechnicalScore | 0.705672 |
| Prompt/completion tokens | 0 / 0 |

| Scenario | Samples | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Buying | 80 | 0.862500 | 0.482763 | 5.375000 |
| Browsing | 80 | 0.987500 | 0.610288 | 5.625000 |
| Intent Override | 30 | 0.700000 | 0.409299 | 6.866667 |
| Boundary | 10 | 0.800000 | 0.612500 | 7.000000 |

## Reproduction

Run from a clean committed worktree with the frozen catalog present:

```bash
python3 scripts/run_m5_offline.py \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --expected-catalog-count 50000 \
  --expected-sample-count 200 \
  --output /path/outside-the-repository/m5-offline.json
```

The runner removes and later restores `OPENAI_API_KEY`; it never prints or
serializes its value. No local generative LLM is used.

## Remaining dependency

After the production M4 run exists, compute the primary-minus-offline delta for
Hit Rate@10, MRR, MTTC, Efficiency and TechnicalScore and append it to the M5
report. This is the only unmet M5 evidence row. It must not be filled from the
fixture mechanics run.
