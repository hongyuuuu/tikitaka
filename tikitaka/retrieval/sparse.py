"""Deterministic in-memory SQLite FTS5/BM25 catalog retrieval."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from threading import RLock
from typing import Iterable

from .catalog import ProductCatalog, ProductDocument
from .text import PRODUCT_TEXT_SCHEMA_VERSION, SparseFields, build_sparse_fields, fts5_expression, query_terms


@dataclass(frozen=True, slots=True)
class SparseIndexConfig:
    title_weight: float = 6.0
    category_weight: float = 4.0
    feature_weight: float = 2.5
    detail_weight: float = 2.5
    store_weight: float = 1.5
    description_weight: float = 1.0
    max_query_terms: int = 40
    batch_size: int = 1_000

    def __post_init__(self) -> None:
        weights = self.field_weights
        if any(weight < 0 for weight in weights):
            raise ValueError("BM25 field weights must be non-negative")
        if self.max_query_terms <= 0:
            raise ValueError("max_query_terms must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @property
    def field_weights(self) -> tuple[float, ...]:
        return (
            self.title_weight,
            self.category_weight,
            self.feature_weight,
            self.detail_weight,
            self.store_weight,
            self.description_weight,
        )


@dataclass(frozen=True, slots=True)
class SparseIndexManifest:
    engine: str
    catalog_sha256: str
    ordered_parent_asin_sha256: str
    catalog_row_count: int
    product_text_schema_version: str
    field_weights: tuple[float, ...]
    tokenizer: str

    def as_dict(self) -> dict[str, object]:
        return {
            "engine": self.engine,
            "catalog_sha256": self.catalog_sha256,
            "ordered_parent_asin_sha256": self.ordered_parent_asin_sha256,
            "catalog_row_count": self.catalog_row_count,
            "product_text_schema_version": self.product_text_schema_version,
            "field_weights": list(self.field_weights),
            "tokenizer": self.tokenizer,
        }


@dataclass(frozen=True, slots=True)
class SparseHit:
    parent_asin: str
    rank: int
    score: float
    matched_fields: tuple[str, ...]
    supporting_snippets: tuple[str, ...]


def _matched_evidence(
    fields: SparseFields,
    terms: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    named_fields = (
        ("title", fields.title),
        ("categories", fields.categories),
        ("features", fields.features),
        ("details", fields.details),
        ("store", fields.store),
        ("description", fields.description),
    )
    matched: list[str] = []
    snippets: list[str] = []
    term_set = set(terms)
    for field_name, value in named_fields:
        tokens = set(query_terms(value, max_terms=max(1, len(value))))
        if value and term_set.intersection(tokens):
            matched.append(field_name)
            snippets.append(f"{field_name}: {value[:240]}")
    return tuple(matched), tuple(snippets[:4])


class SparseIndex:
    """Catalog-derived index with no session or recommendation-history state."""

    def __init__(
        self,
        catalog: ProductCatalog,
        *,
        config: SparseIndexConfig | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self.catalog = catalog
        self.config = config or SparseIndexConfig()
        self.connection = connection or sqlite3.connect(":memory:", check_same_thread=False)
        self._owns_connection = connection is None
        self._closed = False
        self._lock = RLock()
        self._build()
        self.manifest = SparseIndexManifest(
            engine="sqlite-fts5-bm25-v1",
            catalog_sha256=catalog.identity.source_sha256,
            ordered_parent_asin_sha256=catalog.identity.ordered_parent_asin_sha256,
            catalog_row_count=catalog.identity.row_count,
            product_text_schema_version=PRODUCT_TEXT_SCHEMA_VERSION,
            field_weights=self.config.field_weights,
            tokenizer="unicode61 remove_diacritics 2",
        )

    def _build(self) -> None:
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                "CREATE VIRTUAL TABLE products USING fts5("
                "parent_asin UNINDEXED, title, categories, features, details, store, description, "
                "tokenize='unicode61 remove_diacritics 2')"
            )
        except sqlite3.OperationalError as error:
            raise RuntimeError("SQLite FTS5 support is required for sparse retrieval") from error
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        for product in self.catalog:
            fields = build_sparse_fields(product)
            batch.append((product.parent_asin, *fields.ordered_values()))
            if len(batch) >= self.config.batch_size:
                cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("sparse index is closed")

    def search(
        self,
        text_query: str,
        *,
        must_terms: tuple[str, ...] = (),
        should_terms: tuple[str, ...] = (),
        limit: int = 200,
    ) -> list[SparseHit]:
        with self._lock:
            self._ensure_open()
            if limit <= 0:
                raise ValueError("retrieval limit must be positive")
            expression = fts5_expression(
                text_query=text_query,
                must_terms=must_terms,
                should_terms=should_terms,
                max_terms=self.config.max_query_terms,
            )
            if not expression:
                return []
            weights = ", ".join(format(value, ".12g") for value in (0.0, *self.config.field_weights))
            rows = self.connection.execute(
                "SELECT parent_asin, title, categories, features, details, store, description, "
                f"bm25(products, {weights}) AS rank_score "
                "FROM products WHERE products MATCH ? "
                "ORDER BY rank_score ASC, parent_asin ASC LIMIT ?",
                (expression, limit),
            ).fetchall()
        terms = query_terms(text_query, *must_terms, *should_terms, max_terms=self.config.max_query_terms)
        hits: list[SparseHit] = []
        seen: set[str] = set()
        for row in rows:
            parent_asin = str(row[0])
            if parent_asin in seen or parent_asin not in self.catalog:
                continue
            seen.add(parent_asin)
            fields = SparseFields(*[str(value or "") for value in row[1:7]])
            matched_fields, snippets = _matched_evidence(fields, terms)
            hits.append(
                SparseHit(
                    parent_asin=parent_asin,
                    rank=len(hits) + 1,
                    score=-float(row[7]),
                    matched_fields=matched_fields,
                    supporting_snippets=snippets,
                )
            )
        return hits

    def close(self) -> None:
        with self._lock:
            if not self._closed and self._owns_connection:
                self.connection.close()
            self._closed = True

    def __enter__(self) -> "SparseIndex":
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def documents_for_hits(catalog: ProductCatalog, hits: Iterable[SparseHit]) -> tuple[ProductDocument, ...]:
    return tuple(catalog.require(hit.parent_asin) for hit in hits)
