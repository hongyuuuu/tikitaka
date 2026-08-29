"""Conversation state. Person 1.

The extractor proposes, the reducer disposes. No model output reaches
`SessionState` without passing the strict schema in `schema.py` first.
"""

from __future__ import annotations

from tikitaka.state.schema import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION"]
