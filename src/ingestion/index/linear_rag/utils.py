"""Hashing helper used by ``EmbeddingStore`` to key (text → embedding) pairs."""

from hashlib import md5


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()
