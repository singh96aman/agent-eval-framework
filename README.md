# Agent Judge Calibration Research

## What This Is

This repository studies whether LLM judges fail to penalize the most consequential errors in agent trajectories.

**Core hypothesis:** Judges systematically miscalibrate to error criticality, over-penalizing visible local mistakes while under-penalizing early structural failures (planning/tool selection) that cause larger downstream degradation.

**Example:** A judge might rate a typo in step 9 as worse than a wrong strategy in step 1, even though the step 1 error cascaded and ruined all subsequent steps.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your credentials
```

**Required:**
- `MONGODB_URI` - MongoDB Atlas connection string
- `HUGGINGFACE_TOKEN` - HuggingFace API token
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` - AWS credentials for Bedrock

### 3. Verify Setup
```bash
python src/prereq_check.py
# Should show: ✅ PASSED: 6/6 (100%)
```

### 4. Run Experiment
```bash
# Run entire experiment (all 6 phases)
python main.py --config poc_experiment --runner all

# Or run specific phases
python main.py --config poc_experiment --runner load
python main.py --config poc_experiment --runner load,perturb
python main.py --config poc_experiment --runner judge,ccg,analyze
```

---

## Experiment Configuration

### Design: 1 Config = 1 Complete Experiment

Each JSON config file in `config/experiments/` defines a complete experiment. Use the `--runner` parameter to control which phases execute.

### Example: `config/experiments/poc_experiment.json`

```json
{
  "experiment": {
    "name": "POC: Judge Calibration to Trajectory Criticality",
    "experiment_id": "exp_poc_full_v1_20260402",
    "description": "Full POC experiment with 50 trajectories"
  },

  "datasets": {
    "toolbench": {
      "num_trajectories": 25,
      "filters": { "min_steps": 3, "max_steps": 15 }
    },
    "gaia": {
      "num_trajectories": 25,
      "filters": { "min_steps": 2, "max_steps": 12 }
    }
  },

  "storage": {
    "backend": "mongodb",
    "database": "agent_judge_experiment"
  },

  "judges": {
    "models": [
      {"name": "claude-sonnet-4.5", "model_id": "..."},
      {"name": "gpt-oss-120b", "model_id": "..."}
    ],
    "samples_per_trajectory": 3
  },

  "perturbations": {
    "types": ["planning", "tool_selection", "parameter"],
    "positions": ["early", "middle", "late"],
    "num_per_condition": 5
  },

  "execution": {
    "runner": "all",
    "dry_run": false,
    "verbose": true
  }
}
```

### Config Sections Explained

#### `experiment`
- **`experiment_id`**: Unique identifier (manual or auto-generated hash)
- **`name`**: Human-readable display name
- **`description`**: What this experiment does

#### `datasets`
- **`num_trajectories`**: How many trajectories to load from each benchmark
- **`filters`**: Length constraints (min/max steps) and success criteria
- **`random_seed`**: For reproducibility (same seed = same sample)

#### `storage`
- **`backend`**: Database type (mongodb)
- **`database`**: Database name where results are stored

#### `judges`
- **`models`**: Which LLM judges to use (Claude, GPT-OSS, etc.)
- **`samples_per_trajectory`**: How many times to run each judge (for variance measurement)

#### `perturbations`
- **`types`**: Error types to inject (planning, tool_selection, parameter)
- **`positions`**: Where to inject (early/middle/late in trajectory)
- **`num_per_condition`**: Samples per (type × position) combination

#### `execution`
- **`runner`**: Which phases to run (`"all"`, `"load"`, `"load,perturb"`, etc.)
- **`dry_run`**: If true, runs without saving to database
- **`verbose`**: Enable detailed output

---

## Running Experiments

### The `--runner` Parameter

Controls which phases execute:

| Runner | Phases | Description |
|--------|--------|-------------|
| `all` | 1→2→3→4→5→6 | Full experiment pipeline |
| `load` | 1 | Load trajectories from HuggingFace |
| `perturb` | 2 | Generate perturbed versions |
| `annotate` | 3 | Human annotation interface |
| `judge` | 4 | Run LLM judge evaluations |
| `ccg` | 5 | Compute CCG metrics |
| `analyze` | 6 | Generate visualizations |

### Common Workflows

**Full experiment (end-to-end):**
```bash
python main.py --config poc_experiment --runner all
```

**Staged execution:**
```bash
# Stage 1: Load baseline data
python main.py --config poc_experiment --runner load

# Stage 2: Generate perturbations
python main.py --config poc_experiment --runner perturb

# Stage 3: Annotate (human)
python main.py --config poc_experiment --runner annotate

# Stage 4-6: Judge evaluation + analysis
python main.py --config poc_experiment --runner judge,ccg,analyze
```

**Test mode (no database writes):**
```bash
python main.py --config poc_experiment --runner load --dry-run
```

---

## The 6 Experiment Phases

| Phase | Name | Input | Output | Time |
|-------|------|-------|--------|------|
| 1 | Load Trajectories | HuggingFace datasets | 50 baseline trajectories in MongoDB | 5-10 min |
| 2 | Generate Perturbations | Baseline trajectories | 50 perturbed trajectories (9 conditions) | 1-2 hours (manual) |
| 3 | Annotate | Perturbed trajectories | True criticality scores (TSD, SER) | 12-17 hours (manual) |
| 4 | Evaluate Judges | Trajectories + perturbations | Judge penalty scores (JPS) | 20-30 min (API) |
| 5 | Compute CCG | TCS + JPS | CCG = (JPS - TCS) / TCS | 1-2 min |
| 6 | Analyze | CCG results | Heatmaps, statistical tests, reports | 5-10 min |

---

## Repo Structure

```
repo/
├── README.md                    # This file
├── Agents.MD                    # Project rules and instructions
├── main.py                      # CLI entry point
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
│
├── src/                         # All implementation code
│   ├── data/                    # Dataset loaders (ToolBench, GAIA)
│   ├── storage/                 # MongoDB integration
│   ├── perturbations/           # Error injection logic
│   ├── judges/                  # LLM judge API clients
│   ├── annotation/              # Human annotation tools
│   ├── metrics/                 # CCG computation
│   ├── visualization/           # Plotting and analysis
│   ├── experiment_runner.py    # Main orchestrator
│   └── prereq_check.py          # Prerequisite verification
│
├── tests/                       # Test suite (58 tests)
│   ├── test_loaders.py
│   ├── test_storage.py
│   ├── test_prerequisites.py
│   └── test_integration_pipeline.py
│
├── config/                      # Experiment configurations
│   └── experiments/
│       ├── poc_experiment.json  # Full POC experiment
│       └── test_load.json       # Small test config
│
├── agent_tasks/                 # Task tracking for agents
│   ├── 01_research_foundation/
│   └── 02_poc_implementation/
│
├── data/                        # Datasets and artifacts
├── results/                     # Experiment outputs
└── paper/                       # Papers, notes, literature review
    ├── literature_review.md
    └── POC_REQUIREMENTS.MD
```

---

## Key Concepts

### Trajectories
A **trajectory** is a sequence of steps an AI agent takes to solve a task.

Example:
```
Task: "What's the population of Tokyo?"
Step 1: [Think] "I need to search"
Step 2: [Tool] search_api(query="Tokyo population 2023")
Step 3: [Observe] "14.09 million"
Step 4: [Answer] "The population is 14.09 million"
```

### Perturbations
A **perturbation** is an intentionally injected error to test judge sensitivity.

**Types:**
- **Planning error**: Wrong goal or strategy
- **Tool selection error**: Wrong tool for the task
- **Parameter error**: Right tool, wrong arguments

**Positions:**
- **Early** (steps 1-2): Planning phase
- **Middle** (steps 3-5): Execution phase
- **Late** (steps 6+): Validation phase

### Criticality Metrics

**True Criticality Score (TCS):**
```
TCS = (Task Success Degradation × 100) + (Subsequent Error Rate × 10)
```
- Measured by human annotation
- High TCS = error broke everything downstream

**Judge Penalty Score (JPS):**
```
JPS = 100 - judge_overall_score
```
- Extracted from LLM judge ratings
- High JPS = judge penalized heavily

**Criticality-Calibration Gap (CCG):**
```
CCG = (JPS - TCS) / TCS
```
- **CCG < 0**: Judge under-penalizes (missed critical error)
- **CCG > 0**: Judge over-penalizes (overreacted to minor error)
- **CCG ≈ 0**: Well-calibrated

### Expected Results

**Hypothesis:** Judges exhibit position-dependent miscalibration.

| Position | Expected CCG | Interpretation |
|----------|--------------|----------------|
| Early | < -0.3 | Under-penalizes critical errors by 30%+ |
| Middle | -0.2 to +0.1 | Moderately calibrated |
| Late | > +0.2 | Over-penalizes minor errors by 20%+ |

**Heatmap visualization:**
```
           Early    Middle    Late
Planning   -0.65    -0.30    +0.15  (blue → red)
Tool       -0.50    -0.20    +0.25
Parameter  -0.40    -0.10    +0.35

Blue (negative) = Under-penalizes
Red (positive) = Over-penalizes
```

---

## Data Storage

Results are stored in MongoDB Atlas:

**Collections:**
- `experiments` - Experiment metadata and configs
- `trajectories` - Baseline trajectories
- `perturbed_trajectories` - Perturbed versions with metadata
- `annotations` - Human annotations (TSD, SER, TCS)
- `judge_evaluations` - Judge ratings (JPS)
- `ccg_results` - Computed CCG metrics

**Experiment ID tracking:**
All data for an experiment is linked by `experiment_id` from the config. This enables:
- Querying all data for one experiment
- Resuming failed runs
- Comparing across experiments

---

## Current Status

**Completed:**
- ✅ Infrastructure (loaders, storage, runner system)
- ✅ Test suite (58 tests passing)
- ✅ Prerequisites (MongoDB, HF, AWS Bedrock verified)
- ✅ Phase 1 implementation (trajectory loading)

**In Progress:**
- 🔄 Phase 2: Perturbation generation
- 🔄 Phase 3: Annotation interface
- 🔄 Phase 4: Judge API integration
- 🔄 Phase 5: CCG computation
- 🔄 Phase 6: Visualization

---

## CLI Reference

### List available configs
```bash
python main.py --list-configs
```

### Run experiment
```bash
python main.py --config <config_name> --runner <phases>
```

**Options:**
- `--config <name>` - Config file name (in `config/experiments/`)
- `--runner <phases>` - Which phases to run (`all`, `load`, `load,perturb`, etc.)
- `--dry-run` - Test without saving to database
- `--resume` - Resume from last checkpoint
- `--verbose` - Enable detailed output

**Examples:**
```bash
# Run all phases
python main.py --config poc_experiment --runner all

# Run specific phases
python main.py --config poc_experiment --runner load,perturb

# Test mode
python main.py --config poc_experiment --runner load --dry-run

# Resume after failure
python main.py --config poc_experiment --resume
```

---

## Development

### Run tests
```bash
pytest tests/ -v
```

### Check prerequisites
```bash
python src/prereq_check.py
```

### Verify MongoDB connection
```bash
python src/data/verify_atlas_connection.py
```

---

## How to Start Work

Read in this order:
1. `README.md` (this file)
2. `Agents.MD` (project rules)
3. `agent_tasks/<task_name>/Requirements.MD`
4. `agent_tasks/<task_name>/state.json`

---

## Research Context

**Literature review:** [paper/literature_review.md](paper/literature_review.md)  
**POC design:** [paper/POC_REQUIREMENTS.MD](paper/POC_REQUIREMENTS.MD)

**Key related work:**
- G-Eval (judge evaluation)
- AgentProcessBench (step-level evaluation)
- Position-Weighted Consistency (PWC)
- Information Fidelity in Tool-Using Agents
- Effort-based metrics (autonomous driving)

**Our novelty:**
First systematic study of judge calibration to trajectory criticality with position-dependent error weighting.

---

## Contributing

This is a research project. AI agents are heavily used for implementation. See `Agents.MD` for project-specific instructions.

**Important rules:**
- All code in `src/`
- All tests in `tests/`
- All paper material in `paper/`
- Never commit as Claude (update git config)
- Keep repo clean and organized
