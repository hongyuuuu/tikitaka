"""Shared, dependency-light domain contracts.

Ownership: Person 4. This package is a faithful transcription of the frozen
`0.1.0` specification in `docs/p0/CONTRACT_PROPOSAL.md`, landed by Person 1 to
unblock the P1 model and state workstream while Person 4's own module is in
flight. If Person 4 lands `tikitaka/contracts/`, prefer their version wholesale;
the semantics here are copied from the same frozen source and nothing in
`tikitaka/models/` or `tikitaka/state/` depends on this file's authorship.

Nothing here may import a provider SDK, the evaluator, the catalog, or any
owner's implementation package.
"""

from __future__ import annotations

CONTRACT_VERSION = "0.1.0"

__all__ = ["CONTRACT_VERSION"]
