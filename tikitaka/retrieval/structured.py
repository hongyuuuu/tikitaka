"""High-precision structured product evidence with unknown-safe matching."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Mapping, Sequence

from .catalog import ProductDocument
from .text import build_sparse_fields, normalize_text


ATTRIBUTE_NAMES = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
)
EVIDENCE_OUTCOMES = frozenset({"match", "contradiction", "unknown"})
MATERIALS = (
    "canvas",
    "cashmere",
    "cotton",
    "denim",
    "fabric",
    "fleece",
    "leather",
    "linen",
    "mesh",
    "nylon",
    "polyester",
    "rayon",
    "rubber",
    "silk",
    "spandex",
    "suede",
    "wool",
)
COLORS = (
    "beige",
    "black",
    "blue",
    "brown",
    "burgundy",
    "cream",
    "gold",
    "gray",
    "green",
    "grey",
    "navy",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "tan",
    "white",
    "yellow",
)
USE_CASES = (
    "casual",
    "gym",
    "hiking",
    "outdoor",
    "running",
    "sports",
    "travel",
    "walking",
    "winter",
    "work",
)
MATERIAL_RE = re.compile(r"\b(" + "|".join(map(re.escape, MATERIALS)) + r")\b", re.IGNORECASE)
COLOR_RE = re.compile(r"\b(" + "|".join(map(re.escape, COLORS)) + r")\b", re.IGNORECASE)
USE_CASE_RE = re.compile(r"\b(" + "|".join(map(re.escape, USE_CASES)) + r")\b", re.IGNORECASE)
SIZE_RE = re.compile(
    r"\b(?:size|width)\s*[:#-]?\s*((?:xxs|xs|s|m|l|xl|xxl|xxxl)|(?:\d{1,2}(?:\.5)?))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class AttributeEvidence:
    values: tuple[object, ...] = ()
    reliability: float = 0.0
    snippets: tuple[str, ...] = ()

    @property
    def known(self) -> bool:
        return bool(self.values)


@dataclass(frozen=True, slots=True)
class StructuredProductEvidence:
    by_attribute: Mapping[str, AttributeEvidence]

    def for_attribute(self, attribute: str) -> AttributeEvidence:
        return self.by_attribute.get(attribute, AttributeEvidence())

    @property
    def unknown_fields(self) -> tuple[str, ...]:
        return tuple(attribute for attribute in ATTRIBUTE_NAMES if not self.for_attribute(attribute).known)


@dataclass(frozen=True, slots=True)
class ConstraintEvaluation:
    attribute: str
    outcome: str
    reliability: float
    matched_values: tuple[object, ...] = ()
    snippets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.outcome not in EVIDENCE_OUTCOMES:
            raise ValueError(f"invalid evidence outcome: {self.outcome}")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("reliability must be within [0.0, 1.0]")


def _ordered_unique(values: Sequence[object]) -> tuple[object, ...]:
    result: list[object] = []
    seen: set[str] = set()
    for value in values:
        key = normalize_text(value).casefold()
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _matching_snippets(product: ProductDocument, terms: Sequence[str], limit: int = 3) -> tuple[str, ...]:
    fields = build_sparse_fields(product)
    candidates = (
        fields.title,
        fields.categories,
        fields.features,
        fields.details,
        fields.store,
        fields.description,
    )
    lowered_terms = tuple(term.casefold() for term in terms)
    snippets: list[str] = []
    for candidate in candidates:
        lowered = candidate.casefold()
        if candidate and any(term in lowered for term in lowered_terms):
            snippets.append(candidate[:240])
            if len(snippets) >= limit:
                break
    return tuple(snippets)


def _lexicon_evidence(
    product: ProductDocument,
    pattern: re.Pattern[str],
    *,
    reliability: float,
) -> AttributeEvidence:
    fields = build_sparse_fields(product)
    corpus = " ".join(fields.ordered_values())
    values = _ordered_unique(tuple(match.casefold() for match in pattern.findall(corpus)))
    terms = tuple(str(value) for value in values)
    return AttributeEvidence(
        values=values,
        reliability=reliability if values else 0.0,
        snippets=_matching_snippets(product, terms),
    )


def _category_evidence(product: ProductDocument) -> AttributeEvidence:
    values: list[str] = []
    # Amazon category arrays are ordered from a broad navigation root to a
    # specific leaf. The root (for example "Clothing, Shoes & Jewelry") is
    # not product-type evidence: treating its words as exact categories makes
    # every bag and coat a shoe. Prefer descendants; retain a single category
    # only when no more-specific hierarchy exists.
    categories = product.categories[1:] if len(product.categories) > 1 else product.categories
    for raw in categories:
        normalized = normalize_text(raw).casefold()
        if normalized:
            values.append(normalized)
        for part in re.split(r"\s*(?:,|>)\s*", normalized):
            if part:
                values.append(part)
    unique = _ordered_unique(values)
    return AttributeEvidence(
        values=unique,
        reliability=1.0 if unique else 0.0,
        snippets=tuple(categories[:3]),
    )


def _brand_evidence(product: ProductDocument) -> AttributeEvidence:
    values: list[str] = []
    snippets: list[str] = []
    if product.store:
        values.append(normalize_text(product.store).casefold())
        snippets.append(product.store)
    for key, value in product.details:
        if any(marker in key.casefold() for marker in ("brand", "manufacturer")):
            values.append(normalize_text(value).casefold())
            snippets.append(f"{key}: {value}")
    unique = _ordered_unique(values)
    return AttributeEvidence(
        values=unique,
        reliability=0.95 if unique else 0.0,
        snippets=tuple(snippets[:3]),
    )


def _size_evidence(product: ProductDocument) -> AttributeEvidence:
    values: list[str] = []
    snippets: list[str] = []
    for key, value in product.details:
        if any(marker in key.casefold() for marker in ("size", "width", "fit")):
            values.append(normalize_text(value).casefold())
            snippets.append(f"{key}: {value}")
    fields = build_sparse_fields(product)
    for match in SIZE_RE.findall(" ".join((fields.title, fields.features, fields.details))):
        values.append(normalize_text(match).casefold())
    unique = _ordered_unique(values)
    return AttributeEvidence(
        values=unique,
        reliability=0.85 if unique else 0.0,
        snippets=tuple(snippets[:3]) or _matching_snippets(product, ("size", "width")),
    )


def _style_evidence(product: ProductDocument) -> AttributeEvidence:
    values: list[str] = []
    snippets: list[str] = []
    for key, value in product.details:
        if any(marker in key.casefold() for marker in ("department", "style", "fit", "sleeve", "neck")):
            values.append(normalize_text(value).casefold())
            snippets.append(f"{key}: {value}")
    unique = _ordered_unique(values)
    return AttributeEvidence(
        values=unique,
        reliability=0.8 if unique else 0.0,
        snippets=tuple(snippets[:3]),
    )


def _feature_evidence(product: ProductDocument) -> AttributeEvidence:
    values = _ordered_unique(tuple(normalize_text(value).casefold() for value in product.features))
    return AttributeEvidence(
        values=values,
        reliability=0.65 if values else 0.0,
        snippets=tuple(value[:240] for value in product.features[:3]),
    )


def extract_structured_evidence(product: ProductDocument) -> StructuredProductEvidence:
    budget = AttributeEvidence(
        values=() if product.price is None else (product.price,),
        reliability=0.0 if product.price is None else 1.0,
        snippets=() if product.price is None else (f"price: {format(product.price, 'f')}",),
    )
    use_case = _lexicon_evidence(product, USE_CASE_RE, reliability=0.75)
    evidence = {
        "category": _category_evidence(product),
        "material": _lexicon_evidence(product, MATERIAL_RE, reliability=0.9),
        "color": _lexicon_evidence(product, COLOR_RE, reliability=0.85),
        "size": _size_evidence(product),
        "style": _style_evidence(product),
        "brand": _brand_evidence(product),
        "budget": budget,
        "feature": _feature_evidence(product),
        "use_case": use_case,
        "other": AttributeEvidence(),
    }
    return StructuredProductEvidence(by_attribute=MappingProxyType(evidence))


def _decimal(value: object) -> Decimal | None:
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    text = normalize_text(value).replace("$", "").replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        result = Decimal(match.group(0))
    except InvalidOperation:
        return None
    return result if result.is_finite() else None


def _string_match(known: object, desired: object) -> bool:
    known_text = normalize_text(known).casefold()
    desired_text = normalize_text(desired).casefold()
    if not known_text or not desired_text:
        return False
    if known_text == desired_text or desired_text in known_text or known_text in desired_text:
        return True
    desired_tokens = set(re.findall(r"[a-z0-9]+", desired_text))
    known_tokens = set(re.findall(r"[a-z0-9]+", known_text))
    return bool(desired_tokens) and desired_tokens.issubset(known_tokens)


def evaluate_constraint(
    evidence: StructuredProductEvidence,
    *,
    attribute: str,
    desired_values: Sequence[object],
    polarity: str = "include",
    operator: str = "eq",
) -> ConstraintEvaluation:
    """Evaluate one retrieval constraint without treating absence as conflict."""

    if attribute not in ATTRIBUTE_NAMES:
        raise ValueError(f"unknown retrieval attribute: {attribute}")
    if polarity not in {"include", "exclude"}:
        raise ValueError(f"invalid constraint polarity: {polarity}")
    item = evidence.for_attribute(attribute)
    desired = tuple(value for value in desired_values if normalize_text(value))
    if not desired or not item.known:
        return ConstraintEvaluation(attribute, "unknown", 0.0, snippets=item.snippets)

    if attribute == "budget":
        price = _decimal(item.values[0])
        bound = _decimal(desired[0])
        if price is None or bound is None:
            return ConstraintEvaluation(attribute, "unknown", 0.0, snippets=item.snippets)
        comparisons = {
            "lte": price <= bound,
            "lt": price < bound,
            "gte": price >= bound,
            "gt": price > bound,
            "eq": price == bound,
        }
        if operator not in comparisons:
            raise ValueError(f"invalid budget operator: {operator}")
        matched = comparisons[operator]
        if polarity == "exclude":
            matched = not matched
        return ConstraintEvaluation(
            attribute,
            "match" if matched else "contradiction",
            item.reliability,
            matched_values=(price,) if matched else (),
            snippets=item.snippets,
        )

    matched_values = tuple(
        known
        for known in item.values
        if any(_string_match(known, wanted) for wanted in desired)
    )
    if matched_values:
        outcome = "contradiction" if polarity == "exclude" else "match"
        return ConstraintEvaluation(
            attribute,
            outcome,
            item.reliability,
            matched_values=matched_values,
            snippets=item.snippets,
        )

    authoritative = attribute in {"category", "brand"}
    if authoritative:
        outcome = "match" if polarity == "exclude" else "contradiction"
        return ConstraintEvaluation(attribute, outcome, item.reliability, snippets=item.snippets)
    return ConstraintEvaluation(attribute, "unknown", 0.0, snippets=item.snippets)
