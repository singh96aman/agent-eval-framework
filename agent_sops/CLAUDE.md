# Agents.MD

# When starting a new conversation
When user starts a new conversation, first check whether you are using AWS Bedrock to talk to Claude or not. If not, please give the user a heads up. It's non-blocking so you continue with the rest of your tasks.

Note - PLEASE DO NOT CREATE TOO MANY FILES OR FILES THAT ARE NOT ORGANIZED AS PER INSTRUCTIONS

WHEN COMMITTING, PLEASE DON'T COMMIT AS CLAUDE

### Instructions to the agent

1. Read README.md, Agents.MD, and the task’s Requirements.MD before starting.
2. Follow the repo structure strictly.
3. Put implementation code in src/ only.
4. Add or update tests in tests/ for behavior changes.
5. For long tasks, keep state.json updated.
6. Do not mark a task complete unless outputs exist and tests pass.
7. Save important reasoning, notes, and drafts under paper/, not only in chat.
8. Prefer simple, readable, restartable code over clever code.
9. Keep the repo clean.

## Goal
This repo implements research on whether LLM judges miss the errors in agent trajectories that matter most, especially early planning and tool-selection failures that propagate downstream.

The researcher’s goal is to produce:
- a strong paper,
- a reusable research artifact,
- a clear continuation of prior work on LLM evaluation and judge reliability.

We will use AI heavily for:
- coding,
- planning,
- result verification,
- analysis,
- paper writing.

The human researcher is the final authority on scientific correctness, interpretation, and final acceptance.

---

## Repo rules

- All code must live under `src/`
- All tests must live under `tests/`
- All datasets and dataset artifacts must live under `data/`
- All experiment outputs must live under `results/`
- All notes, deep dives, drafts, and paper material must live under `paper/`
- All restartable task work must live under `agent_tasks/`

Root may contain:
- `README.md`
- `Agents.MD`
- `requirements.txt`
- `.env.example`
- `.gitignore`
- `abcd.py` as a thin CLI launcher only

This is a single Python monolith. Do not split into multiple repos or services.

---

## Expected structure

```text
repo/
├── README.md
├── Agents.MD
├── abcd.py
├── src/
├── tests/
├── agent_tasks/
├── data/
├── results/
└── paper/
```
---

## Agent task system

Each task lives under:

agent_tasks/<task_name>/

### Typical contents:

```
agent_tasks/<task_name>/
├── Requirements.MD
└── state.json
```

### Task rules
1. Requirements.MD is the source of truth
2. If the task is long-running, create and maintain state.json
3. A task is complete only when: 1/ output requirements are met 2/ required artifacts are created 3/All tests in test pass


### What is state.json?

Use it for restartability. Track:

* status,
* plan,
* completed steps,
* pending steps,
* artifacts created,
* blockers,
* last updated

### My help
If you need my help or if there is experiment failure and you can't fix it, trigger a notification.

**How to notify me:**
The notification hook is configured in `~/.claude/settings.json`. When Claude Code needs attention (permission prompts, blockers, etc.), a native macOS notification will appear automatically.

For manual notification in urgent cases, you can run:
```bash
osascript -e 'display notification "Your message here" with title "Claude Code" sound name "default"'
```

Good luck!

---

## Agent Personas

Specialized SOPs for each agent role are in separate files:

| Persona | File | Role |
|---------|------|------|
| Engineer | [sop_engineer.md](sop_engineer.md) | Build and maintain code |
| Scientist Reviewer | [sop_scientist_reviewer.md](sop_scientist_reviewer.md) | Ensure scientific rigor |
| Data Scientist | [sop_data_scientist.md](sop_data_scientist.md) | Analyze data and visualize |

When activated as a specific persona, follow both the general instructions above AND the persona-specific SOP.
