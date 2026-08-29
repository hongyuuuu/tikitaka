"""Catalog normalization and deterministic retrieval components."""

from .catalog import (
    CatalogIdentity,
    CatalogValidationError,
    ProductCatalog,
    ProductDocument,
    load_catalog,
)
from .request import RetrievalConstraint, RetrievalRequest, request_from_search_plan
from .retriever import RetrievalHit, SparseStructuredRetriever
from .sparse import SparseHit, SparseIndex, SparseIndexConfig
from .structured import (
    AttributeEvidence,
    ConstraintEvaluation,
    StructuredProductEvidence,
    extract_structured_evidence,
)
from .text import PRODUCT_TEXT_SCHEMA_VERSION, SparseFields, build_dense_text, build_sparse_fields

__all__ = [
    "AttributeEvidence",
    "CatalogIdentity",
    "CatalogValidationError",
    "ConstraintEvaluation",
    "PRODUCT_TEXT_SCHEMA_VERSION",
    "ProductCatalog",
    "ProductDocument",
    "RetrievalConstraint",
    "RetrievalHit",
    "RetrievalRequest",
    "SparseFields",
    "SparseHit",
    "SparseIndex",
    "SparseIndexConfig",
    "SparseStructuredRetriever",
    "StructuredProductEvidence",
    "build_dense_text",
    "build_sparse_fields",
    "extract_structured_evidence",
    "load_catalog",
    "request_from_search_plan",
]
