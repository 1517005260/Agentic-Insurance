"""Schema entry — one row in :data:`config_store.schema.CONFIG_ENTRIES`.

Each entry pairs a key with: its python type, the algorithm-layer
default, and a validator that's a single source of truth for both the
admin-route 422s and the in-process ``patch()`` guard. We deliberately
keep this trivial — no pydantic field machinery, just a dataclass with
a ``validate()`` method — so the algorithm side has no web-framework
dependency.
"""
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ConfigEntry:
    key: str
    type: str            # "int" | "str" | "float" | "bool" | "list_str" | "dict_str_float"
    default: Any         # imported live from the algorithm layer
    description: str = ""
    # Range / length bounds. Inclusive. For ``int`` / ``float`` both
    # apply; for ``str`` only ``max_length`` (and ``min_length`` if you
    # want a non-empty constraint); ``bool`` ignores them; ``list_str``
    # uses ``min_length`` / ``max_length`` against the list length.
    min: Optional[float] = None
    max: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    # Optional grouping label for the admin UI.
    group: str = ""

    def validate(self, value: Any) -> Any:
        """Coerce + range-check ``value``. Raise ``ValueError`` on rejection.

        Returns the (possibly coerced) value the caller should store.
        Coercion is intentionally narrow:

        * ``int`` accepts a JSON number only if it has no fractional
          part; bool is rejected (Python bool is an int subclass).
        * ``float`` accepts JSON int OR float.
        * ``bool`` requires a real bool — no truthy-coercion.
        * ``str`` requires str.

        We don't auto-cast strings like ``"5"`` because the admin UI
        sends ``application/json`` and the schema is the contract —
        ambiguity costs more than it saves.
        """
        if self.type == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"{self.key}: expected int, got {type(value).__name__}"
                )
            if self.min is not None and value < self.min:
                raise ValueError(f"{self.key}: {value} < min {self.min}")
            if self.max is not None and value > self.max:
                raise ValueError(f"{self.key}: {value} > max {self.max}")
            return value
        if self.type == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{self.key}: expected float, got {type(value).__name__}"
                )
            value = float(value)
            if self.min is not None and value < self.min:
                raise ValueError(f"{self.key}: {value} < min {self.min}")
            if self.max is not None and value > self.max:
                raise ValueError(f"{self.key}: {value} > max {self.max}")
            return value
        if self.type == "bool":
            if not isinstance(value, bool):
                raise ValueError(
                    f"{self.key}: expected bool, got {type(value).__name__}"
                )
            return value
        if self.type == "str":
            if not isinstance(value, str):
                raise ValueError(
                    f"{self.key}: expected str, got {type(value).__name__}"
                )
            if self.min_length is not None and len(value) < self.min_length:
                raise ValueError(
                    f"{self.key}: length {len(value)} < min_length {self.min_length}"
                )
            if self.max_length is not None and len(value) > self.max_length:
                raise ValueError(
                    f"{self.key}: length {len(value)} > max_length {self.max_length}"
                )
            return value
        if self.type == "list_str":
            # JSON arrays of strings — used for open-set NER prompt
            # lists. ``str`` is rejected to force the admin UI's
            # JSON-encoded payload contract; we don't auto-split a
            # comma-separated string for the same reason ``int``
            # doesn't auto-cast ``"5"``.
            if not isinstance(value, list):
                raise ValueError(
                    f"{self.key}: expected list[str], got {type(value).__name__}"
                )
            for i, item in enumerate(value):
                if not isinstance(item, str):
                    raise ValueError(
                        f"{self.key}[{i}]: expected str, got {type(item).__name__}"
                    )
            if self.min_length is not None and len(value) < self.min_length:
                raise ValueError(
                    f"{self.key}: length {len(value)} < min_length {self.min_length}"
                )
            if self.max_length is not None and len(value) > self.max_length:
                raise ValueError(
                    f"{self.key}: length {len(value)} > max_length {self.max_length}"
                )
            return list(value)  # defensive copy — the store snapshot is immutable
        if self.type == "dict_str_float":
            # JSON objects with string keys and numeric values.
            # Used for per-label GLiNER thresholds (e.g. {"concept": 0.5}).
            if not isinstance(value, dict):
                raise ValueError(
                    f"{self.key}: expected dict[str, float], got {type(value).__name__}"
                )
            result = {}
            for k, v in value.items():
                if not isinstance(k, str):
                    raise ValueError(
                        f"{self.key}: key {k!r} must be str, got {type(k).__name__}"
                    )
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(
                        f"{self.key}[{k!r}]: expected float, got {type(v).__name__}"
                    )
                result[k] = float(v)
            return result
        raise ValueError(f"{self.key}: unsupported entry type {self.type!r}")

    def to_public_dict(self) -> dict:
        """Shape for ``GET /admin/config/schema``."""
        out: dict = {
            "key": self.key,
            "type": self.type,
            "default": self.default,
            "description": self.description,
            "group": self.group,
        }
        for name in ("min", "max", "min_length", "max_length"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        return out
