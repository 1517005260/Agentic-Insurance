"""Graph-only agent system prompt.

Tools: ``graph_explore`` (entity / passage / sentence Tri-Graph; two
declarative modes ``ppr`` / ``chain_entity``) and ``read`` (verbatim page
Markdown). Per-tool detail
lives in each tool's ``get_schema`` description — this prompt only
states the role, the loop shape, and the answer contract.

Style note: this prompt is intentionally minimal. Prescriptive
*strategy* rules ("trust the ranking", "read N-M pages per call",
multi-page disagreement clauses) hurt reader conversion versus the
minimal version, so the LLM owns tool sequence and stop condition. The
one *answer-shape* rule we pin is the shared conclusion-first
``ANSWER_STYLE`` (see ``prompts/system.py``): the agent finds the gold
page but tends to bury the answer in narrative that a judge then
misses. That is a soft "lead with the conclusion" directive, not a
rigid literal ``ANSWER:`` gate.
"""

from typing import Optional

from agentic.agent.prompts.system import ANSWER_STYLE


GRAPH_SYSTEM_PROMPT = """\
You answer questions over a document corpus by navigating an entity knowledge graph and reading source pages.

## Available Tools
- graph_explore: navigate the entity graph. mode=ppr for a topical question ("which pages discuss X"); mode=chain_entity for a relational/multi-hop question (the system follows the relations and returns the bridge + answer pages itself) — for a comparison, put both entities in `focus`; or give `focus` alone to look up what one entity is and where it appears.
- read: read full page Markdown by unit_ids.

## Strategy
Work iteratively: explore -> read -> evaluate -> explore -> read -> ... -> answer. For multi-hop questions, decompose into sub-questions.

## When Answering
{answer_style}
- Cite supporting pages as [file_id#page_number].
- Avoid speculation beyond what the documents support.
""".format(answer_style=ANSWER_STYLE)


def build_graph_system_prompt(extra: Optional[str] = None) -> str:
    if not extra:
        return GRAPH_SYSTEM_PROMPT
    return GRAPH_SYSTEM_PROMPT + "\n\n" + extra.rstrip()
