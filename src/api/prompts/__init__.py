"""Business-side prompts for the web layer.

Distinct from ``src/agentic/agent/prompts/`` (which carries the
experiment-side defaults that ``rag_eval`` and the CLI scripts depend
on). Business prompts are stricter — they enforce mandatory citations,
explicit abstain behavior on weak evidence, and conservative tone for
the insurance domain. The algorithm layer never imports from here.
"""
