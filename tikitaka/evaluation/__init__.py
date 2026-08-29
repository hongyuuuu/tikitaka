"""Evaluation-only package kept outside participant runtime contracts."""

from tikitaka.evaluation.experiment import ExperimentConfig, evaluate_samples
from tikitaka.evaluation.reporting import build_report, canonical_report_json
from tikitaka.evaluation.splits import SplitManifest, SplitSpec, create_split, partition_samples

__all__ = [
    "ExperimentConfig",
    "SplitManifest",
    "SplitSpec",
    "build_report",
    "canonical_report_json",
    "create_split",
    "evaluate_samples",
    "partition_samples",
]
