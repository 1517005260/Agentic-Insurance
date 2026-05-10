"""End-to-end ingestion → index pipeline.

This package intentionally re-exports nothing. Importing
:mod:`pipeline` would otherwise chain-load
:mod:`pipeline.parse_and_index`, which pulls spaCy / torch / faiss /
igraph (~600 MB resident) for callers that only want the lightweight
``parse_only`` / ``index_parsed`` entry points. Import the specific
submodule directly: ``from pipeline.parse_and_index import
parse_and_index``.
"""

__all__: list[str] = []
