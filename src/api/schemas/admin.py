"""Pydantic schemas for the admin config-center routes.

Kept separate from the algorithm-layer ``ConfigEntry`` dataclass so the
web layer can shape its responses (camelCase, FastAPI docs, etc.)
without polluting the framework-free :mod:`config.config_store`.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConfigEntrySchema(BaseModel):
    """One row in the admin UI's config table."""

    key: str
    type: str                # "int" | "str" | "float" | "bool"
    default: Any
    description: str = ""
    group: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None


class ConfigSnapshotResponse(BaseModel):
    """Shape returned by ``GET /admin/config``."""

    snapshot: Dict[str, Any] = Field(
        ..., description="Flat key→value map of the effective config."
    )
    schema_: List[ConfigEntrySchema] = Field(
        ..., alias="schema",
        description=(
            "Registered entries with their defaults / bounds. The admin "
            "UI uses these to render labels, helptext, and validation."
        ),
    )

    model_config = {"populate_by_name": True}


class ConfigPatchRequest(BaseModel):
    """Body for ``PATCH /admin/config`` — flat ``{key: value}`` map."""

    updates: Dict[str, Any] = Field(
        ...,
        description=(
            "Flat key→new-value map. Validated against the schema "
            "before any DB write — the patch is all-or-nothing."
        ),
    )


class ConfigPatchResponse(BaseModel):
    """Per-key diff returned after a successful PATCH."""

    diffs: Dict[str, Dict[str, Any]] = Field(
        ...,
        description=(
            "Per-key ``{old, new}`` pairs. Mirrors what landed in the "
            "audit_log so the frontend can render an undo affordance."
        ),
    )
    snapshot: Dict[str, Any] = Field(
        ..., description="Full effective config after the patch lands."
    )


__all__ = [
    "ConfigEntrySchema",
    "ConfigSnapshotResponse",
    "ConfigPatchRequest",
    "ConfigPatchResponse",
]
