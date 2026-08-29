"""Immutable loading and validation for the frozen product catalog."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping


class CatalogValidationError(ValueError):
    """Raised when a catalog record violates the participant-visible schema."""


@dataclass(frozen=True, slots=True)
class ProductDocument:
    """Normalized, immutable participant-visible product metadata."""

    parent_asin: str
    title: str
    features: tuple[str, ...]
    description: tuple[str, ...]
    price: Decimal | None
    categories: tuple[str, ...]
    details: tuple[tuple[str, str], ...]
    average_rating: float | None
    rating_number: int | None
    store: str
    present_fields: frozenset[str]


@dataclass(frozen=True, slots=True)
class CatalogIdentity:
    """Stable identity used by generated retrieval artifacts."""

    source_sha256: str
    ordered_parent_asin_sha256: str
    row_count: int


@dataclass(frozen=True, slots=True)
class ProductCatalog:
    """Read-only catalog with deterministic source order and ID lookup."""

    products: tuple[ProductDocument, ...]
    identity: CatalogIdentity
    _by_id: Mapping[str, ProductDocument] = field(repr=False, compare=False)

    @classmethod
    def from_products(
        cls,
        products: tuple[ProductDocument, ...],
        identity: CatalogIdentity,
    ) -> "ProductCatalog":
        by_id = {product.parent_asin: product for product in products}
        if len(by_id) != len(products):
            raise CatalogValidationError("catalog contains duplicate parent_asin values")
        return cls(products=products, identity=identity, _by_id=MappingProxyType(by_id))

    def __len__(self) -> int:
        return len(self.products)

    def __iter__(self) -> Iterator[ProductDocument]:
        return iter(self.products)

    def __contains__(self, parent_asin: object) -> bool:
        return isinstance(parent_asin, str) and parent_asin in self._by_id

    @property
    def by_id(self) -> Mapping[str, ProductDocument]:
        return self._by_id

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(self._by_id)

    def get(self, parent_asin: str) -> ProductDocument | None:
        return self._by_id.get(parent_asin)

    def require(self, parent_asin: str) -> ProductDocument:
        try:
            return self._by_id[parent_asin]
        except KeyError as error:
            raise KeyError(f"unknown catalog parent_asin: {parent_asin}") from error


def _clean_scalar(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    items = value if isinstance(value, (list, tuple)) else (value,)
    cleaned = tuple(text for item in items if (text := _clean_scalar(item)))
    return cleaned


def _detail_tuple(value: object) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        pairs = ((_clean_scalar(key), _clean_scalar(item)) for key, item in value.items())
    elif isinstance(value, (list, tuple)):
        pairs = ((str(index), _clean_scalar(item)) for index, item in enumerate(value))
    else:
        pairs = (("value", _clean_scalar(value)),)
    return tuple(sorted(((key, item) for key, item in pairs if key and item), key=lambda pair: pair[0].casefold()))


def _decimal_or_none(value: object, *, line_number: int) -> Decimal | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise CatalogValidationError(f"line {line_number}: price must be numeric or null")
    if isinstance(value, str):
        normalized = _clean_scalar(value).casefold()
        # The released catalog uses an em dash for an unavailable price and a
        # handful of "from X" values for variant-dependent prices. A lower
        # bound is not an exact purchasable price, so both remain unknown for
        # hard budget filtering rather than creating false matches.
        if normalized in {"-", "—", "–", "n/a", "na", "none", "null"}:
            return None
        if normalized.startswith("from "):
            return None
    try:
        result = Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError) as error:
        raise CatalogValidationError(f"line {line_number}: invalid price {value!r}") from error
    if not result.is_finite() or result < 0:
        raise CatalogValidationError(f"line {line_number}: price must be finite and non-negative")
    return result


def _float_or_none(value: object, *, field_name: str, line_number: int) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise CatalogValidationError(f"line {line_number}: {field_name} must be numeric or null")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise CatalogValidationError(f"line {line_number}: invalid {field_name} {value!r}") from error
    if not math.isfinite(result):
        raise CatalogValidationError(f"line {line_number}: {field_name} must be finite")
    return result


def _int_or_none(value: object, *, field_name: str, line_number: int) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise CatalogValidationError(f"line {line_number}: {field_name} must be an integer or null")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise CatalogValidationError(f"line {line_number}: invalid {field_name} {value!r}") from error
    if result < 0:
        raise CatalogValidationError(f"line {line_number}: {field_name} must be non-negative")
    return result


def _normalize_record(payload: object, *, line_number: int) -> ProductDocument:
    if not isinstance(payload, dict):
        raise CatalogValidationError(f"line {line_number}: catalog record must be a JSON object")
    raw_id = payload.get("parent_asin")
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise CatalogValidationError(f"line {line_number}: parent_asin must be a non-empty string")
    parent_asin = raw_id.strip()
    visible_fields = {
        "parent_asin",
        "title",
        "features",
        "description",
        "price",
        "categories",
        "details",
        "average_rating",
        "rating_number",
        "store",
    }
    present_fields = frozenset(
        field_name
        for field_name in visible_fields
        if field_name in payload and payload[field_name] not in (None, "", [], {})
    )
    return ProductDocument(
        parent_asin=parent_asin,
        title=_clean_scalar(payload.get("title")),
        features=_string_tuple(payload.get("features")),
        description=_string_tuple(payload.get("description")),
        price=_decimal_or_none(payload.get("price"), line_number=line_number),
        categories=_string_tuple(payload.get("categories")),
        details=_detail_tuple(payload.get("details")),
        average_rating=_float_or_none(
            payload.get("average_rating"), field_name="average_rating", line_number=line_number
        ),
        rating_number=_int_or_none(
            payload.get("rating_number"), field_name="rating_number", line_number=line_number
        ),
        store=_clean_scalar(payload.get("store")),
        present_fields=present_fields,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(
    path: str | Path,
    *,
    expected_count: int | None = None,
) -> ProductCatalog:
    """Load the catalog without modifying it and validate unique identifiers."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"catalog not found: {source}")
    products: list[ProductDocument] = []
    seen: set[str] = set()
    ordered_ids = hashlib.sha256()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise CatalogValidationError(f"line {line_number}: blank catalog record")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise CatalogValidationError(
                    f"line {line_number}: malformed JSON at column {error.colno}: {error.msg}"
                ) from error
            product = _normalize_record(payload, line_number=line_number)
            if product.parent_asin in seen:
                raise CatalogValidationError(
                    f"line {line_number}: duplicate parent_asin {product.parent_asin!r}"
                )
            seen.add(product.parent_asin)
            products.append(product)
            ordered_ids.update(product.parent_asin.encode("utf-8"))
            ordered_ids.update(b"\n")
    if expected_count is not None and len(products) != expected_count:
        raise CatalogValidationError(
            f"catalog row count mismatch: expected {expected_count}, found {len(products)}"
        )
    identity = CatalogIdentity(
        source_sha256=_file_sha256(source),
        ordered_parent_asin_sha256=ordered_ids.hexdigest(),
        row_count=len(products),
    )
    return ProductCatalog.from_products(tuple(products), identity)
