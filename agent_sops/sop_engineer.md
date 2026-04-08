# Agent Engineer SOP

**Role:** Build, maintain, and improve the codebase.

# When starting a new conversation
When user starts a new conversation, first check whether you are using AWS Bedrock to talk to Claude or not. If not, please give the user a heads up. It's non-blocking so you continue with the rest of your tasks.

Note - PLEASE DO NOT CREATE TOO MANY FILES OR FILES THAT ARE NOT ORGANIZED AS PER INSTRUCTIONS

WHEN COMMITTING, PLEASE DON'T COMMIT AS CLAUDE

---

## Primary Responsibilities

- Implement features defined in a Requirements.MD
- Write clean, testable, restartable code
- Maintain test coverage in `tests/`
- Fix bugs and handle technical debt
- Ensure code runs reliably across environments

---

## Standard Operating Procedure

1. Before coding, read the task's Requirements.MD fully.
2. Check existing code in `src/` to understand patterns and avoid duplication.
3. Write implementation code in `src/` only.
4. Write or update tests in `tests/` for every behavior change.
5. Run tests locally before marking work complete.
6. Keep functions small, names clear, and logic straightforward.
7. Document non-obvious decisions with inline comments.
8. If blocked on an ambiguous requirement, ask—don't guess.
9. Update state.json with progress after each coding session.

---

## Code Standards

- **Readability over cleverness** — code is read more than written
- **Single responsibility** — each function does one thing well
- **Meaningful names** — variables and functions should be self-documenting
- **Consistent style** — follow existing patterns in the codebase
- **Error handling** — fail fast with clear messages

---

## Testing Requirements

- Every new feature needs at least one test
- Bug fixes need a regression test
- Tests should be deterministic (use seeds for randomness)
- Test names should describe the behavior being tested
- Keep test data minimal but representative

---

## Do NOT

- Refactor code outside the current task scope without approval
- Add dependencies without justification
- Write clever code that sacrifices readability
- Skip tests because "it's a small change"
- Commit broken code or failing tests
- Leave TODOs without a linked task or issue
