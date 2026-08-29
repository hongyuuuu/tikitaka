"""Model gateway. Person 1.

The only package permitted to import a provider SDK. Retrieval, decision,
ranking, and evaluation code must never learn the provider's name.
"""

from __future__ import annotations

from tikitaka.models.base import (
    CredentialMissing,
    MalformedModelOutput,
    ModelError,
    ModelRefused,
    ModelRoute,
    ModelTimeout,
    ModelUnavailable,
    SchemaViolation,
)

__all__ = [
    "CredentialMissing",
    "MalformedModelOutput",
    "ModelError",
    "ModelRefused",
    "ModelRoute",
    "ModelTimeout",
    "ModelUnavailable",
    "SchemaViolation",
]
