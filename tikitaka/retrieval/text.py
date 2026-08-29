"""Deterministic sparse fields and dense product text construction."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .catalog import ProductDocument


PRODUCT_TEXT_SCHEMA_VERSION = "product_text_v1"
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "please",
        "some",
        "that",
        "the",
        "this",
        "to",
        "want",
        "with",
        "would",
        "you",
        "looking",
    }
)
SINGLE_CHARACTER_SIZE_TERMS = frozenset({"s", "m", "l"})


@dataclass(frozen=True, slots=True)
class SparseFields:
    title: str
    categories: str
    features: str
    details: str
    store: str
    description: str

    def ordered_values(self) -> tuple[str, ...]:
        return (
            self.title,
            self.categories,
            self.features,
            self.details,
            self.store,
            self.description,
        )


def normalize_text(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value))
    return " ".join(normalized.split())


def _join(values: tuple[str, ...], separator: str = " | ") -> str:
    return separator.join(text for value in values if (text := normalize_text(value)))


def _details(product: ProductDocument) -> str:
    return " | ".join(
        f"{normalize_text(key)}: {normalize_text(value)}" for key, value in product.details
    )


def _searchable_categories(product: ProductDocument) -> tuple[str, ...]:
    """Drop Amazon's broad navigation root when a specific path is present."""

    if len(product.categories) > 1:
        return product.categories[1:]
    return product.categories


def build_sparse_fields(product: ProductDocument) -> SparseFields:
    return SparseFields(
        title=normalize_text(product.title),
        categories=_join(_searchable_categories(product), " > "),
        features=_join(product.features),
        details=_details(product),
        store=normalize_text(product.store),
        description=_join(product.description),
    )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    shortened = text[:limit].rsplit(" ", 1)[0].rstrip(" |:;,-")
    return shortened or text[:limit]


def build_dense_text(product: ProductDocument) -> str:
    """Build the frozen v1 representation used by future product embeddings."""

    fields = build_sparse_fields(product)
    sections = (
        ("TITLE", _truncate(fields.title, 500)),
        ("CATEGORY", _truncate(fields.categories, 500)),
        ("BRAND_OR_STORE", _truncate(fields.store, 200)),
        ("FEATURES", _truncate(fields.features, 2_000)),
        ("DETAILS", _truncate(fields.details, 1_500)),
        ("DESCRIPTION", _truncate(fields.description, 2_000)),
        ("PRICE", "" if product.price is None else format(product.price, "f")),
    )
    return "\n".join(f"{label}: {value}" for label, value in sections if value)


def query_terms(*texts: str, max_terms: int = 40) -> tuple[str, ...]:
    if max_terms <= 0:
        return ()
    result: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for token in TOKEN_RE.findall(normalize_text(text).casefold()):
            if token in STOPWORDS:
                continue
            if len(token) == 1 and token not in SINGLE_CHARACTER_SIZE_TERMS and not token.isdigit():
                continue
            if token not in seen:
                seen.add(token)
                result.append(token)
                if len(result) >= max_terms:
                    return tuple(result)
    return tuple(result)


def fts5_expression(
    *,
    text_query: str,
    must_terms: tuple[str, ...] = (),
    should_terms: tuple[str, ...] = (),
    max_terms: int = 40,
) -> str:
    """Build an injection-safe FTS5 query from sanitized alphanumeric tokens."""

    required = query_terms(*must_terms, max_terms=max_terms)
    optional_limit = max(0, max_terms - len(required))
    optional = tuple(
        term
        for term in query_terms(text_query, *should_terms, max_terms=optional_limit)
        if term not in required
    )
    quoted_required = [f'"{term}"' for term in required]
    quoted_optional = [f'"{term}"' for term in optional]
    if quoted_required and quoted_optional:
        return " AND ".join(quoted_required) + " AND (" + " OR ".join(quoted_optional) + ")"
    if quoted_required:
        return " AND ".join(quoted_required)
    return " OR ".join(quoted_optional)
