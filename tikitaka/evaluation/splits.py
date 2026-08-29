"""Stable, versioned public-set tuning and held-out splits."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class SplitSpec:
    version: str = "public-v1"
    seed: int = 2026
    tuning_fraction: float = 0.7

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("split version must be non-empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("split seed must be an integer")
        if not 0.0 < self.tuning_fraction < 1.0:
            raise ValueError("tuning_fraction must be between 0 and 1")


@dataclass(frozen=True)
class SplitManifest:
    spec: SplitSpec
    tuning_ids: tuple[str, ...]
    held_out_ids: tuple[str, ...]
    scenario_counts: tuple[tuple[str, int, int], ...]

    def __post_init__(self) -> None:
        tuning = set(self.tuning_ids)
        held_out = set(self.held_out_ids)
        if len(tuning) != len(self.tuning_ids) or len(held_out) != len(self.held_out_ids):
            raise ValueError("split membership contains duplicate sample IDs")
        if tuning & held_out:
            raise ValueError("tuning and held-out membership must be disjoint")

    def membership(self, sample_id: str) -> str:
        if sample_id in self.tuning_ids:
            return "tuning"
        if sample_id in self.held_out_ids:
            return "held_out"
        raise KeyError(sample_id)

    def to_dict(self) -> dict:
        return {
            "version": self.spec.version,
            "seed": self.spec.seed,
            "tuning_fraction": self.spec.tuning_fraction,
            "tuning_ids": list(self.tuning_ids),
            "held_out_ids": list(self.held_out_ids),
            "scenario_counts": {
                scenario: {"tuning": tuning, "held_out": held_out}
                for scenario, tuning, held_out in self.scenario_counts
            },
        }


def _ordering_key(spec: SplitSpec, scenario: str, sample_id: str) -> tuple[str, str]:
    material = f"{spec.version}\0{spec.seed}\0{scenario}\0{sample_id}".encode("utf-8")
    return hashlib.sha256(material).hexdigest(), sample_id


def create_split(samples: Iterable[Mapping[str, object]], spec: SplitSpec = SplitSpec()) -> SplitManifest:
    """Stratify by evaluator-only scenario labels with deterministic membership."""

    grouped: dict[str, list[str]] = {}
    seen: set[str] = set()
    for sample in samples:
        sample_id = sample.get("sample_id")
        scenario = sample.get("scenario_type")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("every sample requires a non-empty sample_id")
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("every sample requires a non-empty scenario_type")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        grouped.setdefault(scenario, []).append(sample_id)

    tuning: list[str] = []
    held_out: list[str] = []
    counts: list[tuple[str, int, int]] = []
    for scenario in sorted(grouped):
        ordered = sorted(grouped[scenario], key=lambda item: _ordering_key(spec, scenario, item))
        tuning_count = round(len(ordered) * spec.tuning_fraction)
        if len(ordered) > 1:
            tuning_count = min(len(ordered) - 1, max(1, tuning_count))
        tuning.extend(ordered[:tuning_count])
        held_out.extend(ordered[tuning_count:])
        counts.append((scenario, tuning_count, len(ordered) - tuning_count))

    return SplitManifest(
        spec=spec,
        tuning_ids=tuple(sorted(tuning)),
        held_out_ids=tuple(sorted(held_out)),
        scenario_counts=tuple(counts),
    )


def partition_samples(
    samples: Iterable[Mapping[str, object]], manifest: SplitManifest
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    tuning_ids = set(manifest.tuning_ids)
    held_out_ids = set(manifest.held_out_ids)
    tuning: list[Mapping[str, object]] = []
    held_out: list[Mapping[str, object]] = []
    observed: set[str] = set()
    for sample in samples:
        sample_id = sample.get("sample_id")
        if sample_id in tuning_ids:
            tuning.append(sample)
        elif sample_id in held_out_ids:
            held_out.append(sample)
        else:
            raise ValueError(f"sample is absent from split manifest: {sample_id!r}")
        observed.add(str(sample_id))
    expected = tuning_ids | held_out_ids
    if observed != expected:
        raise ValueError("samples do not completely match split manifest")
    return tuning, held_out
