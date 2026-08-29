# P0 Status — Baseline and Contract Proposal

Branch: `person4/p0-baseline-contract`

Base commit: `3db9434cc12354bc15bee42f4cf36253f7a5605c`

Owner: Person 4

## Completed locally

- [x] Created a dedicated Person 4 P0 branch.
- [x] Recorded the base commit and available Python runtimes.
- [x] Recorded checksums for the public dataset, official contracts, starter,
      evaluator, and published baseline result.
- [x] Verified 200 public rows and the published 80/80/30/10 scenario mix.
- [x] Ran the unchanged 3-test suite successfully with Python 3.14.2.
- [x] Ran the unchanged 3-test suite successfully with Python 3.12.
- [x] Confirmed the frozen catalog is ignored and absent from the checkout.
- [x] Queried GitHub Releases and confirmed no release is currently published
      for `hongyuuuu/tikitaka`.
- [x] Located the separately downloaded official `catalog.jsonl.gz` and
      `SHA256SUMS` artifacts.
- [x] Verified the archive checksum and gzip integrity.
- [x] Decompressed the catalog to the ignored `data/catalog.jsonl` path.
- [x] Verified 50,000 catalog rows, 50,000 unique `parent_asin` values, and no
      missing IDs.
- [x] Re-ran `python3 -m evaluator.local_evaluator` against the untouched
      evaluator and dataset.
- [x] Reproduced every published aggregate baseline metric with zero delta.
- [x] Drafted shared domain records, protocols, invariants, versioning, and
      owner-specific review questions.
- [x] Preserved the official evaluator, dataset, starter, and contract files.

## Blocked or awaiting people

- [x] Confirm branch/file ownership with Persons 1, 2, and 3.
- [x] Obtain Person 1 acknowledgment of the model/state-facing contracts.
- [x] Obtain Person 2 acknowledgment of the retrieval-facing contracts.
- [x] Obtain Person 3 acknowledgment of the decision/ranking-facing contracts.
- [ ] Resolve review comments and mark contract version `0.1.0` accepted.

## Baseline reproduction

The frozen archive passed its published SHA-256 check and gzip integrity test.
The decompressed catalog contains exactly 50,000 rows and 50,000 unique,
non-empty `parent_asin` values. It remains ignored by Git.

```text
archive SHA-256:      07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8
catalog SHA-256:      da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67
rows:                 50000
unique parent_asins:  50000
```

The unchanged official evaluator reproduced the published values exactly:

| Metric | Published | Reproduced | Delta |
|---|---:|---:|---:|
| Sample count | 200 | 200 | 0 |
| Hit Rate@10 | 0.125 | 0.125 | 0 |
| MRR | 0.068034 | 0.068034 | 0 |
| MTTC | 9.81 | 9.81 | 0 |
| Efficiency | 0.119 | 0.119 | 0 |
| TechnicalScore | 0.10671 | 0.10671 | 0 |

The source archive was supplied separately in `Downloads`; the repository still
had no published GitHub Release at validation time. The release-distribution
gap does not affect the local baseline result but should be resolved before
other owners need to provision the catalog.

## P0 exit decision

P0 is **not yet closed**. All machine-verifiable local baseline work is
complete, and Persons 1, 2, and 3 have acknowledged their affected interfaces
and confirmed ownership. The remaining exit gate is Person 4's resolution of
the recorded review changes and final acceptance of contract version `0.1.0`.

P1 scaffolding may be prepared in parallel, but shared contracts must not be
declared frozen or merged as final until those conditions are satisfied.
