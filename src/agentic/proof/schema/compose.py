"""Tool-description composition.

A Phase D placeholder: today the existing prose carries everything the
LLM needs (worked examples, remediation hints), so this is a thin
pass-through. Centralising the call site lets us splice in
auto-generated sections (e.g. registry-derived predicate enums) later
without touching every tool file.
"""
def compose_tool_description(model: type, prose: str) -> str:
    """Return the description prose used in a tool's ``parameters`` block.

    Future enhancement may inject a "see schema" tail derived from
    ``model.model_json_schema()``; for now the LLM-facing prose is
    unchanged.
    """
    return prose


__all__ = ["compose_tool_description"]
