"""Catalog normalization and deterministic retrieval components."""

from tikitaka.contracts import IndexManifest

from .adapters import ContractRetrieverAdapter, contract_candidate, contract_product_evidence
from .catalog import (
    CatalogIdentity,
    CatalogValidationError,
    ProductCatalog,
    ProductDocument,
    load_catalog,
)
from .dense import (
    DenseArtifactError,
    DenseHit,
    DenseIndex,
    DenseLoadResult,
    DenseRouteError,
    build_dense_artifact,
    embed_query_for_index,
    load_dense_index,
    load_dense_index_safe,
    normalize_embedding,
)
from .fusion import FusedRouteHit, RRFConfig, reciprocal_rank_fusion, route_overlap
from .hybrid import (
    HybridConfig,
    HybridDiagnostics,
    HybridRetrievalHit,
    HybridRetrievalResult,
    HybridRetriever,
)
from .manifests import (
    DENSE_ARTIFACT_FORMAT,
    ManifestValidationError,
    assert_dense_manifest_compatible,
    dense_manifest_as_dict,
    dense_manifest_from_dict,
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
from .text import (
    DENSE_QUERY_SCHEMA_VERSION,
    PRODUCT_TEXT_SCHEMA_VERSION,
    SparseFields,
    build_dense_query,
    build_dense_text,
    build_sparse_fields,
)

__all__ = [
    "AttributeEvidence",
    "CatalogIdentity",
    "CatalogValidationError",
    "ContractRetrieverAdapter",
    "ConstraintEvaluation",
    "DENSE_ARTIFACT_FORMAT",
    "DENSE_QUERY_SCHEMA_VERSION",
    "DenseArtifactError",
    "DenseHit",
    "DenseIndex",
    "DenseLoadResult",
    "DenseRouteError",
    "FusedRouteHit",
    "HybridConfig",
    "HybridDiagnostics",
    "HybridRetrievalHit",
    "HybridRetrievalResult",
    "HybridRetriever",
    "IndexManifest",
    "ManifestValidationError",
    "PRODUCT_TEXT_SCHEMA_VERSION",
    "ProductCatalog",
    "ProductDocument",
    "RetrievalConstraint",
    "RetrievalHit",
    "RetrievalRequest",
    "RRFConfig",
    "SparseFields",
    "SparseHit",
    "SparseIndex",
    "SparseIndexConfig",
    "SparseStructuredRetriever",
    "StructuredProductEvidence",
    "build_dense_text",
    "build_dense_query",
    "build_dense_artifact",
    "build_sparse_fields",
    "extract_structured_evidence",
    "embed_query_for_index",
    "dense_manifest_as_dict",
    "dense_manifest_from_dict",
    "assert_dense_manifest_compatible",
    "load_catalog",
    "load_dense_index",
    "load_dense_index_safe",
    "normalize_embedding",
    "reciprocal_rank_fusion",
    "route_overlap",
    "contract_candidate",
    "contract_product_evidence",
    "request_from_search_plan",
]
