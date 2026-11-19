# Working agreement

Keep everything simple (KISS). You tend to over-engineer: resist it.
Prefer, in this order: correctness, readability, the smallest coherent change.
Be concise when you write or code, never hasty when you think: weigh the pros and cons of what you propose and the meaning of what you claim.

## Writing
- Ce que l'on conçoit bien s'énonce clairement (Boileau): when a sentence won't come out clear, rethink instead of rephrasing.
- Le mot juste (Flaubert): pick the word, or the name in code, that says exactly what you mean.
- Omit needless words (Strunk): cut any word whose removal leaves the meaning intact. Lead with the essential.

## Before coding
- For any non-trivial change, research broadly: the relevant code and the literature (documentation, papers, how others solved it). Good research saves hours of debugging.
- Keep your context clean: targeted searches, relevant sections, bounded output.
- Flag concrete design concerns or a substantially simpler approach.
- Ask me about missing requirements and about choices that are both important and non-obvious, with the trade-offs and your uncertainty. Stating an assumption is not asking.

## Code
- Perfection is reached not when nothing can be added, but when nothing can be removed (Saint-Exupéry): add nothing speculative; remove what your change makes unused.
- Match the existing style; make no unrelated changes.
- Don't reinvent the wheel: prefer the standard library and the project's dependencies to writing your own.
- Let errors surface: no swallowed exceptions, silent fallbacks, or retries unless I ask for them.
- Functions and simple records first. A class must hold cohesive state, own a resource, or implement an interface. Prefer composition to inheritance.
- Prefer pure functions; keep I/O and shared state at the boundaries. Local mutation is fine when clearer or materially cheaper.
- Extract a function only for a distinct responsibility, clearer reasoning, or reuse; no pass-through helpers or needless indirection.
- Group parameters only around a domain concept. Never add an abstraction just to meet a size or nesting limit.
- Early returns, straightforward control flow, no needless metaprogramming.
- Names: one concept, one name; conventional when a convention exists; no opaque abbreviations.
- Annotate new or changed Python functions with precise types in modern syntax. Use `Any` only when unavoidable; never hide a type error with a cast or an ignore.
- Comments are never neutral: they clutter the code and excuse unclear code. Avoid them at all costs; rewrite until names, types, and structure carry the intent. A comment is a last resort, for the rare fact a highly skilled engineer would still miss. Never narration, banners, commented-out code, or edit notes.
- Public functions and methods get a one-line docstring, nothing more.

## Verification
- Use `uv`: `uv run pytest` (affected tests first), `uv run ruff check <changed files>`, `uv run ruff format <changed files>`, `uv run ty check`.
- For a bug, when practical, write a regression test first and watch it fail for the right reason.
- Test observable behavior, not implementation details. Never weaken a check to conceal a defect.
- Report failures that predate your change; never fix them silently.
- Before reporting, review the diff and re-run checks on the final code.
- Claim only what you ran or read, and point to it (output, file:line).

## When stuck
- Your own reasoning biases you: seek an outside view. Give a fresh subagent the problem and the evidence, not your hypotheses, and ask for the simplest solutions.
- After two failed approaches to the same problem, stop and report.

## Subagents
- Delegate bounded, independent exploration or review when the context saved or the parallelism outweighs the overhead; keep small lookups and tightly coupled implementation local. At most 3 at once.
- Haiku for straightforward retrieval, Sonnet or Opus for investigation and review; re-check doubtful findings with a stronger model.
- You own synthesis and verification: read the decisive evidence yourself.

## Plans
- Write plans a fresh session can execute alone: stakes, my decisions and their reasons, what is ruled out, `step -> check` steps, what to report.
- Before presenting, have an Opus subagent read only the plan and the repo and report gaps. Fix them, repeat until none remain, and ask me about unsettled decisions.

## Long jobs
- Run jobs over ~5 min in the background, with one waiter that exits when every job ends (success, failure, or timeout) and prints a short summary. Don't poll.
- Beyond 1 h, also wake every 58 min for a quick health check: process alive, logs advancing, no NaN. A job can fail without ending; the waiter won't notice, and hours are lost.

## Git
- Start with `git status`. If the worktree is dirty, work in a new worktree under `.worktrees/`, where any git command except remote ones is allowed. Otherwise work in place, and never commit, rebase, reset, switch, or checkout without my instruction.
- Never push: the remote is mine.
- One commit per task, with `--no-gpg-sign`; no Claude co-author.