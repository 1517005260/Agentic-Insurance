# CLAUDE.md

Use first principles thinking. Don't always assume that I clearly know what I want and how to get it. Be cautious, and start from the fundamental needs and problems. If the motivations or goals are unclear, stop and discuss them with me. If the goal is clear but the path isn't the shortest, let me know and suggest a better approach.

When writing code, avoid the simplest implementation; instead, aim for the most logical and elegant one.

**Behavioral Guidelines**: These guidelines aim to reduce common LLM coding mistakes. Merge them with project-specific instructions as needed.

**Tradeoff:** These guidelines favor caution over speed. For trivial tasks, use your judgment.

## 1. Think Before Coding

**Don’t assume. Don’t hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them—don’t pick silently.
- If a simpler approach exists, mention it. Push back when warranted.
- If something is unclear, stop. Identify what's confusing and ask for clarification.

## 2. Simplicity First

**Write the minimum code necessary to solve the problem. Avoid speculative work.**

- Don’t add features beyond what’s requested.
- Avoid unnecessary abstractions for single-use code.
- Don’t introduce flexibility or configurability that wasn’t requested.
- Skip error handling for impossible scenarios.
- If your solution is overly complicated (e.g., 200 lines when 50 would suffice), refactor.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify it.

## 3. Surgical Changes

**Touch only what is necessary. Clean up only your own mess.**

When editing existing code:

- Don’t “improve” adjacent code, comments, or formatting unless requested.
- Don’t refactor things that aren’t broken.
- Match the existing style, even if you would do it differently.
- If you notice unrelated dead code, mention it, but don’t delete it.

When your changes create orphans:

- Remove imports, variables, or functions that your changes rendered unused.
- Don’t remove pre-existing dead code unless specifically asked to.

Test: Every modified line should directly address the user’s request.

## 4. Goal-Driven Execution

**Define success criteria and loop until verified.**

Turn tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs and make them pass."
- "Fix the bug" → "Write a test that reproduces the bug, then make it pass."
- "Refactor X" → "Ensure tests pass before and after the refactor."

For multi-step tasks, provide a brief plan:

```

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

```

Clear success criteria allow you to loop independently. Weak criteria (e.g., "make it work") will require constant clarification.

---

**These guidelines are working if:** You see fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and more clarifying questions before implementation rather than after mistakes.
