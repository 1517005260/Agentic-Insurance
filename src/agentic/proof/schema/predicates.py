"""Pydantic models for predicate specs at the tool boundary.

The discriminated union covers the ten registered primitives plus the
recursive ``and`` composite. Each primitive's args model carries only
the keys the predicate registry treats as required, with ``extra='allow'``
so optional/canonicalisable knobs (e.g. ``case_sensitive`` for
``contains_string``, ``flags`` for ``regex_match``) round-trip without
a schema-level rejection.

The AND variant absorbs the legacy ``args.conjuncts`` shape into a
canonical top-level ``conjuncts`` via a ``BeforeValidator`` so both
shapes the plant historically accepted continue to validate.
"""
from typing import Annotated, Any, List, Literal, Union

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
)


_ARGS_CONFIG = ConfigDict(extra="allow")


# --------------------------------------------------------------- args


class ContainsStringArgs(BaseModel):
    model_config = _ARGS_CONFIG
    pattern: str = Field(min_length=1)
    case_sensitive: bool = False


class RegexMatchArgs(BaseModel):
    model_config = _ARGS_CONFIG
    pattern: str = Field(min_length=1)
    flags: str = ""


class FieldEqualsArgs(BaseModel):
    model_config = _ARGS_CONFIG
    field_path: str = Field(min_length=1)
    value: Any = None


class NumericCompareArgs(BaseModel):
    model_config = _ARGS_CONFIG
    field_path: str = Field(min_length=1)
    op: str
    value: float


class DateCompareArgs(BaseModel):
    model_config = _ARGS_CONFIG
    field_path: str = Field(min_length=1)
    op: str
    value: str


class TypeIsArgs(BaseModel):
    model_config = _ARGS_CONFIG
    unit_type: str = Field(min_length=1)


class TableCellContainsArgs(BaseModel):
    model_config = _ARGS_CONFIG
    column_name: str = Field(min_length=1)
    pattern: str = Field(min_length=1)


class SectionTitleContainsArgs(BaseModel):
    model_config = _ARGS_CONFIG
    pattern: str = Field(min_length=1)


class RangeInArgs(BaseModel):
    model_config = _ARGS_CONFIG
    field_path: str = Field(min_length=1)
    lo: float
    hi: float


class ListContainsArgs(BaseModel):
    model_config = _ARGS_CONFIG
    field_path: str = Field(min_length=1)
    item_pattern: str = Field(min_length=1)


# --------------------------------------------------------------- variants


class ContainsStringPredicate(BaseModel):
    model_config = _ARGS_CONFIG
    name: Literal["contains_string"]
    args: ContainsStringArgs


class RegexMatchPredicate(BaseModel):
    model_config = _ARGS_CONFIG
    name: Literal["regex_match"]
    args: RegexMatchArgs


class FieldEqualsPredicate(BaseModel):
    model_config = _ARGS_CONFIG
    name: Literal["field_equals"]
    args: FieldEqualsArgs


class NumericComparePredicate(BaseModel):
    model_config = _ARGS_CONFIG
    name: Literal["numeric_compare"]
    args: NumericCompareArgs


class DateComparePredicate(BaseModel):
    model_config = _ARGS_CONFIG
    name: Literal["date_compare"]
    args: DateCompareArgs


class TypeIsPredicate(BaseModel):
    model_config = _ARGS_CONFIG
    name: Literal["type_is"]
    args: TypeIsArgs


class TableCellContainsPredicate(BaseModel):
    model_config = _ARGS_CONFIG
    name: Literal["table_cell_contains"]
    args: TableCellContainsArgs


class SectionTitleContainsPredicate(BaseModel):
    model_config = _ARGS_CONFIG
    name: Literal["section_title_contains"]
    args: SectionTitleContainsArgs


class RangeInPredicate(BaseModel):
    model_config = _ARGS_CONFIG
    name: Literal["range_in"]
    args: RangeInArgs


class ListContainsPredicate(BaseModel):
    model_config = _ARGS_CONFIG
    name: Literal["list_contains"]
    args: ListContainsArgs


# --------------------------------------------------------------- AND


def _absorb_legacy_and(data: Any) -> Any:
    """Collapse legacy ``{"name":"and","args":{"conjuncts":[...]}}`` into
    the canonical ``{"name":"and","conjuncts":[...]}`` so the discriminator
    sees a uniform shape regardless of how the LLM emitted it."""
    if isinstance(data, dict) and data.get("name") == "and":
        if "conjuncts" not in data and isinstance(data.get("args"), dict):
            inner = data["args"].get("conjuncts")
            if inner is not None:
                merged = {k: v for k, v in data.items() if k != "args"}
                merged["conjuncts"] = inner
                data = merged
    return data


PredicateSpecField = Annotated[
    Annotated[
        Union[
            ContainsStringPredicate,
            RegexMatchPredicate,
            FieldEqualsPredicate,
            NumericComparePredicate,
            DateComparePredicate,
            TypeIsPredicate,
            TableCellContainsPredicate,
            SectionTitleContainsPredicate,
            RangeInPredicate,
            ListContainsPredicate,
            "AndPredicate",
        ],
        Field(discriminator="name"),
    ],
    BeforeValidator(_absorb_legacy_and),
]


class AndPredicate(BaseModel):
    model_config = _ARGS_CONFIG
    name: Literal["and"]
    conjuncts: List[PredicateSpecField] = Field(min_length=1)


AndPredicate.model_rebuild()


__all__ = [
    "ContainsStringArgs",
    "RegexMatchArgs",
    "FieldEqualsArgs",
    "NumericCompareArgs",
    "DateCompareArgs",
    "TypeIsArgs",
    "TableCellContainsArgs",
    "SectionTitleContainsArgs",
    "RangeInArgs",
    "ListContainsArgs",
    "ContainsStringPredicate",
    "RegexMatchPredicate",
    "FieldEqualsPredicate",
    "NumericComparePredicate",
    "DateComparePredicate",
    "TypeIsPredicate",
    "TableCellContainsPredicate",
    "SectionTitleContainsPredicate",
    "RangeInPredicate",
    "ListContainsPredicate",
    "AndPredicate",
    "PredicateSpecField",
]
