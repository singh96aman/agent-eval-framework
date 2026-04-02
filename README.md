
## README

## What this repo is

This repository studies whether LLM judges fail to penalize the most consequential errors in agentic trajectories.

Core idea:
some trajectory errors are much more damaging than others, but judges may not score them in proportion to their true downstream impact.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API credentials
```

Required environment variables:
- `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (for Claude Bedrock)
- `GPT_OSS_ENDPOINT`, `GPT_OSS_API_KEY` (for GPT-OSS 120B)

### 3. Set Up Datasets

Place ToolBench and GAIA datasets in their respective directories:
- `data/toolbench/` - ToolBench trajectories (JSON/JSONL)
- `data/gaia/` - GAIA benchmark data (JSON/JSONL)

See [data/README.md](data/README.md) for detailed setup instructions.

### 4. Verify Pre-Requisites

```bash
python src/prereq_check.py
```

This checks:
- ✓ Directory structure
- ✓ Datasets available and parseable
- ✓ API access (Claude Bedrock, GPT-OSS)
- ✓ Python dependencies installed

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## Repo layout

- `src/` — all code
  - `data/` — dataset loaders and schema
  - `perturbations/` — perturbation generation (Phase 2)
  - `judges/` — LLM judge API integration (Phase 3)
  - `annotation/` — human annotation tools (Phase 4)
  - `metrics/` — CCG computation (Phase 4)
  - `visualization/` — plotting and reporting (Phase 5)
- `tests/` — all tests
- `agent_tasks/` — restartable task folders
- `data/` — datasets and dataset artifacts
- `results/` — experiment outputs
- `paper/` — notes, deep dives, outlines, drafts

---

## Current Status

**Completed:**
- ✅ Task 01: Research Foundation (literature review + POC design)
- ✅ Phase 1 of Task 02: Pre-requisite checks + dataset loaders

**In Progress:**
- 🔄 Task 02: POC Implementation (Phase 2-5)

**Next:**
- Phase 2: Perturbation generation
- Phase 3: Judge API integration
- Phase 4: Annotation tools + CCG metrics
- Phase 5: Experiment runner + visualization

---

## How to start or restart work

Read in this order:
1. `README.md` (this file)
2. `Agents.MD` (project rules and instructions)
3. `agent_tasks/<task_name>/Requirements.MD`
4. `agent_tasks/<task_name>/state.json` if present

That should be enough to resume most work.

---

## Task format

Example:

```text
agent_tasks/create_scaffolding_for_mongodb/
├── Requirements.MD
├── state.json
├── inputs/
├── outputs/
└── artifacts/