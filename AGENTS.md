# Working agreement

Prefer correctness, readability, and the smallest coherent change.
Keep designs simple, short and easy to modify (KISS).

## Workflow

- Read relevant code, callers, and tests before editing; investigate discoverable facts before asking.
- In auto-mode, For nontrivial work, state a checkable goal and short `step -> check` plan. 
- Skip ceremony for trivial edits.
- Search narrowly with `rg`; read relevant sections and bound noisy output.
- Flag concrete design concerns or a substantially simpler approach before coding.
- After two failed approaches to the same problem, stop and report.
- Run jobs longer than a few minutes in the background with one waiter that exits when all jobs end (success, failure, or timeout) and prints a short summary. Don't poll while it runs.

## Ask

- Ask when missing requirements or unresolved choices could materially affect scope, behavior, architecture, compatibility, or data safety. 
- Stating an assumption does not replace asking me.
- Explain the uncertainty and tradeoffs; recommend an option when useful.
- Handle routine implementation details within agreed requirements.
  Skip tutorials and questions about settled decisions.

## Plan

- In plan mode, write plans a fresh session can execute alone. Include stakes, my decisions with reasons, what is ruled out, `step -> check`, what to report.
- Before presenting, have an Opus subagent read only the plan and repo, and list what it cannot determine.
- Fix the gaps that would change the work or how results are read; ask me about those only I can answer. Repeat until none remain.

## Delegate

- Delegate bounded, independent exploration or review to subagents when context savings or parallelism justify the overhead; keep small lookups and tightly coupled implementation local.
- Never more than 3 subagents at once.
- Use available lightweight (Haiku) models for straightforward retrieval and stronger models (Sonnet / Opus) for complex investigation or review; re-check uncertain findings with a stronger model.
- Default workers to read-only. Provide a bounded question, scope, and constraints; require concise findings, file references, and uncertainty.
- The main agent owns synthesis, integration, discussion with me, and verification; inspect decisive evidence directly.
- Avoid duplicate exploration and overlapping edits.

## Scope

- Fix the cause within scope, including necessary callers and tests.
- Match existing style; avoid unrelated refactoring or formatting. 
- Remove what your change makes unused.
- No speculative features, configuration, or extension points.
- Prefer existing dependencies and the standard library.
- Validate external inputs and handle credible boundary failures. Do not hide errors or add
  speculative fallbacks, retries, logging, or redundant internal checks.

## Design

- Functions and simple records first. Classes need cohesive state, resource ownership,
  or a meaningful interface. Prefer composition to inheritance.
- Prefer pure computation with explicit inputs and outputs; keep I/O and shared mutation
  at boundaries. Local mutation is fine when clearer or materially cheaper.
- Keep dependencies explicit and cohesive code together. Extract functions for a clear responsibility, simpler reasoning, or reuse; avoid pass-through helpers and unnecessary indirection.
- Prefer early returns and straightforward loops. Group parameters only around a real domain concept; do not invent abstractions to meet size or nesting quotas.
- Keep critical checks easy to trace; avoid unnecessary metaprogramming and hidden control flow.

## Names, types, and documentation

- Use precise domain names: one concept, one term. Prefer readable words to opaque abbreviations. Use conventional names when available.
- Annotate parameters and returns in new or changed Python functions with precise types
  and modern syntax supported by the project's Python version. Prefer precise types; confine `Any` to unavoidable scenarios. Do not hide type errors with unjustified casts or suppressions.
- The code expresses its intent. Let names, types, and structure explain the code. Do not add verbose comments: confine them to explain non-obvious constraints, contracts, or reasons; omit narration, banners, commented-out code, and edit notes.
- Give public Python functions and methods one-line summary docstrings. Private helpers (`_name`) may omit them. Do not repeat signatures or add parameter/return sections.
- Leave unrelated comments alone; correct any your change makes false.

## Tools and verification

- Use `uv run` and existing project configuration; do not change dependencies or tooling incidentally.
- Tests: `uv run pytest`. Lint: `uv run ruff check`. Types: `uv run ty check`.
- Format with `uv run ruff format`; verify with `uv run ruff format --check`. Scope Ruff commands to changed files; start pytest with affected tests.
- For bugs, add a regression test when practical; establish that it fails for the relevant
  reason before the fix, then passes afterward.
- Test observable behavior, not helper structure. Never weaken checks to conceal a defect.
- Broaden checks for integration risk or project gates; repeat when code or evidence changes.
- Report tests or lint failures that existed before your change; never fix them silently.
- Review the final diff. Report actual check results and blockers; never claim verification you did not perform.

## Git

- Run `git status` first: if the worktree is dirty, work in a new worktree under `.worktrees/`; otherwise, work in place.
- Never commit, rebase, reset, switch, checkout, or push without my explicit instruction.