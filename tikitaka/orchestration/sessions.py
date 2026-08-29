"""Thread-safe ownership of isolated, opaque per-session state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Callable, Generic, Mapping, TypeVar

from tikitaka.contracts import Usage


StateT = TypeVar("StateT")


@dataclass(frozen=True)
class ComponentUsage:
    """One attributable component usage event."""

    component: str
    usage: Usage


@dataclass
class _SessionRecord(Generic[StateT]):
    state: StateT
    profile_snapshot: Mapping[str, object]
    usage_events: list[ComponentUsage] = field(default_factory=list)


class SessionRegistry(Generic[StateT]):
    """Store states without knowing or mutating their concrete representation."""

    def __init__(self, state_factory: Callable[[str, Mapping[str, object]], StateT]) -> None:
        self._state_factory = state_factory
        self._records: dict[str, _SessionRecord[StateT]] = {}
        self._lock = RLock()

    def reset(self, session_id: str, user_profile: Mapping[str, object]) -> StateT:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(user_profile, Mapping):
            raise TypeError("user_profile must be a mapping")
        snapshot = deepcopy(dict(user_profile))
        protected_snapshot = MappingProxyType(snapshot)
        state = self._state_factory(session_id, protected_snapshot)
        with self._lock:
            self._records[session_id] = _SessionRecord(state, protected_snapshot)
        return state

    def get(self, session_id: str) -> StateT | None:
        with self._lock:
            record = self._records.get(session_id)
            return None if record is None else record.state

    def replace(self, session_id: str, state: StateT) -> None:
        with self._lock:
            record = self._records.get(session_id)
            if record is None:
                raise KeyError(session_id)
            record.state = state

    def record_usage(self, session_id: str, component: str, usage: Usage) -> None:
        if not isinstance(usage, Usage):
            raise TypeError("usage must be a Usage record")
        with self._lock:
            record = self._records.get(session_id)
            if record is None:
                raise KeyError(session_id)
            record.usage_events.append(ComponentUsage(component=component, usage=usage))

    def usage_events(self, session_id: str) -> tuple[ComponentUsage, ...]:
        with self._lock:
            record = self._records.get(session_id)
            return () if record is None else tuple(record.usage_events)

    def profile_snapshot(self, session_id: str) -> Mapping[str, object] | None:
        with self._lock:
            record = self._records.get(session_id)
            return None if record is None else record.profile_snapshot
