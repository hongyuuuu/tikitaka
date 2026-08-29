"""Message ingestion: interpret, then reduce.

The extractor owns two things the frozen contract cannot express inside a
`StateDelta`: routing a spent question to exhaustion rather than
no-preference, and falling back to the deterministic interpreter when the
model route fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tikitaka.contracts.domain import Usage
from tikitaka.models.base import ModelError
from tikitaka.models.fake import HeuristicInterpreter, detect_exhaustion
from tikitaka.models.usage import merge
from tikitaka.state.reducer import StateReducer
from tikitaka.state.schema import empty_delta
from tikitaka.state.session import SessionState


@dataclass
class IngestResult:
    usage: Usage
    used_fallback: bool = False
    failure: str = ""
    exhausted_attribute: str | None = None


@dataclass
class Extractor:
    interpreter: object
    reducer: StateReducer = field(default_factory=StateReducer)
    fallback: object = field(default_factory=HeuristicInterpreter)

    def ingest(self, state: SessionState, message: str, turn: int) -> IngestResult:
        exhausted = detect_exhaustion(message or "")
        if exhausted is not None:
            self.reducer.note_exhausted(state, exhausted)

        used_fallback = False
        failure = ""
        try:
            delta, usage = self.interpreter.interpret(message, state)
        except ModelError as error:
            delta, usage, used_fallback, failure = self._degrade(state, message, error)
        except Exception as error:  # component defect, never reaches the evaluator
            delta, usage, used_fallback, failure = self._degrade(state, message, error)

        self.reducer.apply(state, delta, turn)
        return IngestResult(
            usage=usage,
            used_fallback=used_fallback,
            failure=failure,
            exhausted_attribute=exhausted,
        )

    def _degrade(
        self,
        state: SessionState,
        message: str,
        error: Exception,
    ) -> tuple[object, Usage, bool, str]:
        # A failed call may still have burned tokens. Carrying its usage
        # forward keeps the cost disclosure honest about work that was paid
        # for and thrown away.
        spent = getattr(error, "usage", None)
        spent = spent if isinstance(spent, Usage) else Usage()

        if self.fallback is None:
            return empty_delta(state.mode), spent, False, str(error)
        try:
            delta, usage = self.fallback.interpret(message, state)
            return delta, merge(spent, usage), True, str(error)
        except Exception:
            return empty_delta(state.mode), spent, False, str(error)


__all__ = ["Extractor", "IngestResult"]
