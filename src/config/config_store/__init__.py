"""Public API for the runtime config store.

Both the FastAPI lifespan (web overrides) and experiment scripts (pure
schema defaults) construct a :class:`ConfigStore` via the constructor
classmethods. Keep imports here narrow — the algorithm layer should
not pay an admin-route import cost.
"""
from config.config_store.entry import ConfigEntry
from config.config_store.schema import CONFIG_ENTRIES, CONFIG_ENTRIES_BY_KEY
from config.config_store.store import ConfigStore


__all__ = [
    "ConfigEntry",
    "ConfigStore",
    "CONFIG_ENTRIES",
    "CONFIG_ENTRIES_BY_KEY",
]
