"""Centralized settings and constants. Importing this package loads .env."""

from config import settings
from config.settings import *  # noqa: F401,F403
from config.linear_rag import LinearRAGConfig

__all__ = ["settings", "LinearRAGConfig"]
