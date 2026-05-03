"""Pydantic models for ScopeRef and ScoreSpec at the tool boundary."""
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ScopeRefModel(BaseModel):
    """Tool-boundary scope shape. ``section_ids=null`` means "whole files".

    The plant freezes ``sealed`` at create — it is part of the obligation's
    identity, not a runtime knob.
    """

    model_config = ConfigDict(extra="forbid")

    file_ids: List[str] = Field(min_length=1)
    section_ids: Optional[List[str]] = None
    sealed: bool = False


class ScoreSpecModel(BaseModel):
    """Score extractor reference. ``args`` is registry-checked downstream
    so we keep the schema permissive (``extra='allow'``) and only validate
    the discriminator key here."""

    model_config = ConfigDict(extra="forbid")

    name: Literal[
        "numeric_amount",
        "percentage",
        "integer_count",
        "date_iso",
        "text_field",
    ]
    args: dict[str, Any] = Field(default_factory=dict)


__all__ = ["ScopeRefModel", "ScoreSpecModel"]
