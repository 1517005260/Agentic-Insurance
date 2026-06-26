"""The shared answer-shape directive for the agent prompts.

Conclusion-first: the agent reliably finds the evidence but tends to bury the
answer in narrative, which a judge then misses. This pins "lead with the
conclusion" without a rigid literal gate.
"""

ANSWER_STYLE = """\
Lead with the conclusion: give the direct answer first — the shortest \
span, value, or verdict that resolves the question — then the supporting \
reasoning and evidence. Reply in the user's language. Do not narrate your \
search process or restate the question."""
