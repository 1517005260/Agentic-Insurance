"""Per-primitive modules. Each module exports a ``SCHEMA`` constant
plus its module-private ``_<name>_universal`` / ``_<name>_eval``
helpers. The registry assembles ``_PRIMITIVES`` by importing the
``SCHEMA`` constants below; the evaluation dispatcher imports the
``_eval`` callables directly from each module.
"""
from agentic.proof.predicate.primitives import (
    contains_string,
    date_compare,
    field_equals,
    list_contains,
    numeric_compare,
    range_in,
    regex_match,
    section_title_contains,
    table_cell_contains,
    type_is,
)

__all__ = [
    "contains_string",
    "date_compare",
    "field_equals",
    "list_contains",
    "numeric_compare",
    "range_in",
    "regex_match",
    "section_title_contains",
    "table_cell_contains",
    "type_is",
]
