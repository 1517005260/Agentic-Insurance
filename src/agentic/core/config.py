"""YAML / JSON config loader with dotted-key access."""

import json
from typing import Any, Dict


class Config:
    """Nested dict with `get('a.b.c', default)` accessors."""

    def __init__(self, config_dict: Dict[str, Any] = None):
        self._config = config_dict or {}

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}
        return cls(config_dict)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls(config_dict)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        return self._config.copy()

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None
