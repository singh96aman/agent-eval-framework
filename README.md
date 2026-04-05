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

## Dataset Generation

### Reproduce the Full Study Dataset

The study uses 600 trajectories (400 ToolBench + 100 GAIA + 100 SWE-bench) with 492 perturbations. To reproduce from scratch:

**Single command (recommended):**
```bash
# Generate trajectories, perturbations, and validate
python main.py --config dataset_full_study --runner load,perturb,validate
```

**Or run phases individually:**
```bash
python main.py --config dataset_full_study --runner load      # Load trajectories
python main.py --config dataset_full_study --runner perturb   # Generate perturbations
python main.py --config dataset_full_study --runner validate  # Validate quality
```

### Dataset Files

After generation, find the data in:
```
data/
├── sampled/
│   ├── toolbench_400.json    # 400 ToolBench trajectories
│   ├── gaia_100.json         # 100 GAIA trajectories
│   └── swebench_100.json     # 100 SWE-bench trajectories
└── perturbed/
    ├── toolbench_perturbations.json
    ├── gaia_perturbations.json
    └── swebench_perturbations.json
```

### Config Options for Dataset Loading

The config supports two data sources:

**1. Load from pre-sampled JSON (faster, reproducible):**
```json
{
  "datasets": {
    "toolbench": {
      "enabled": true,
      "source": "json",
      "json_path": "data/sampled/toolbench_400.json",
      "num_trajectories": 400
    }
  }
}
```

**2. Load fresh from HuggingFace/local:**
```json
{
  "datasets": {
    "toolbench": {
      "enabled": true,
      "source": "local",
      "num_trajectories": 50,
      "filters": {
        "min_steps": 3,
        "max_steps": 15,
        "filter_successful": false
      }
    }
  }
}
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

  "annotation": {
    "num_samples": 25,
    "annotator_id": "researcher1",
    "sampling_strategy": "stratified",
    "skip_annotated": true,
    "random_seed": 42
  },

  "ccg": {
    "judges": ["claude-sonnet-4.5", "gpt-oss-120b"]
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

#### `annotation`
- **`num_samples`**: How many perturbations to annotate (default: 25)
- **`annotator_id`**: Your identifier (for tracking who annotated)
- **`sampling_strategy`**: "stratified" (balanced) or "random"
- **`skip_annotated`**: Skip already annotated perturbations (default: true)
- **`random_seed`**: For reproducible sampling

#### `ccg`
- **`judges`**: Which judges to analyze (if omitted, uses all from judges config)

#### `execution`
- **`runner`**: Which phases to run (`"all"`, `"load"`, `"load,perturb"`, `"annotate"`, `"judge"`, `"ccg"`, etc.)
- **`dry_run`**: If true, runs without saving to database
- **`verbose`**: Enable detailed output

---

## Running Experiments

### The `--runner` Parameter

Controls which phases execute:

| Runner | Phases | Description |
|--------|--------|-------------|
| `all` | 1→2→3→4→5→6→7 | Full experiment pipeline |
| `load` | 1 | Load trajectories from JSON or HuggingFace |
| `perturb` | 2 | Generate perturbed versions |
| `validate` | 3 | Validate perturbation quality |
| `annotate` | 4 | Human annotation interface |
| `judge` | 5 | Run LLM judge evaluations |
| `ccg` | 6 | Compute CCG metrics |
| `analyze` | 7 | Generate visualizations |

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
| 1 | **Load Trajectories** | JSON files or HuggingFace | 600 baseline trajectories | 1-2 min |
| 2 | **Generate Perturbations** | Baseline trajectories | 492 perturbed trajectories (11 conditions) | 2-5 min |
| 3 | **Validate Perturbations** | Perturbed trajectories | Quality report (pass/fail) | <1 min |
| 4 | **Annotate Criticality** | Perturbed trajectories | Human annotations (TSD, SER, TCS) | ~2 hours for 50 samples |
| 5 | **Evaluate Judges** | Trajectories + perturbations | Judge ratings (JPS) | ~4 hours for 1500 evals |
| 6 | **Compute CCG** | Annotations + Judge ratings | CCG scores by condition | 1-2 min |
| 7 | **Analyze Results** | CCG results | Heatmaps, statistical tests, reports | 5-10 min |

### Phase 3: Annotation Details

**Interactive CLI interface** for human researchers to assess perturbation criticality:

```bash
# Annotate 25 samples (stratified sampling across 9 conditions)
python main.py --config poc_experiment_toolbench --runner annotate
```

**For each perturbation, you'll answer:**
1. **Task Success Degradation (TSD):** Did the perturbation cause task failure? (yes=1, no=0)
2. **Subsequent Error Rate (SER):** How many errors occurred *after* the perturbation? (count)

**Automatically computes True Criticality Score:**
```
TCS = (TSD × 100) + (SER × 10)
```

**Configuration options** (in experiment JSON):
```json
{
  "annotation": {
    "num_samples": 25,           // How many to annotate
    "annotator_id": "researcher1", // Your identifier
    "sampling_strategy": "stratified", // or "random"
    "skip_annotated": true,       // Don't re-annotate
    "random_seed": 42             // For reproducibility
  }
}
```

**Output:** Annotations saved to `data/annotations/<perturbation_id>.json`

### Phase 5: CCG Computation

**Computes calibration metrics** from annotations and judge evaluations:

```bash
# Compute CCG for all configured judges
python main.py --config poc_experiment_toolbench --runner ccg
```

**What it does:**
1. Loads human annotations (TCS)
2. Loads judge evaluations (scores → JPS)
3. Computes CCG = (JPS - TCS) / TCS for each perturbation
4. Aggregates by type, position, and condition
5. Runs ANOVA statistical tests
6. Exports to CSV and JSON

**Output:** `results/<experiment_id>/`
- `ccg_results_<judge>.csv` - Per-perturbation scores
- `ccg_summary_<judge>.json` - Aggregated stats and ANOVA results

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
├── tests/                       # Test suite (93 tests)
│   ├── test_loaders.py
│   ├── test_storage.py
│   ├── test_prerequisites.py
│   ├── test_perturbations.py
│   ├── test_judges.py
│   ├── test_annotation.py
│   ├── test_ccg.py
│   └── test_integration_pipeline.py
│
├── config/                      # Experiment configurations
│   └── experiments/
│       ├── dataset_full_study.json  # Full study dataset generation
│       ├── poc_experiment_toolbench.json  # POC with ToolBench only
│       └── test_load.json       # Small test config
│
├── agent_tasks/                 # Task tracking for agents
│   ├── 01_research_foundation/
│   └── 02_poc_implementation/
│
├── data/                        # Datasets and artifacts
├── results/                     # Experiment outputs
├── notebooks/                   # Jupyter notebooks for analysis
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

**MongoDB Collections:**
- `experiments` - Experiment metadata and configs
- `trajectories` - Baseline trajectories from datasets
- `perturbations` - Perturbed trajectories with metadata
- `judge_evaluations` - Judge ratings and scores (JPS)

**File Storage:**
- `data/annotations/` - Human annotations (TSD, SER, TCS) in JSON
- `results/<experiment_id>/` - CCG results (CSV + JSON summaries)

**Experiment ID tracking:**
All data for an experiment is linked by `experiment_id` from the config. This enables:
- Querying all data for one experiment
- Resuming failed runs
- Comparing across experiments

---

## Current Status

**Dataset Ready:**
- ✅ 600 trajectories sampled (400 ToolBench + 100 GAIA + 100 SWE-bench)
- ✅ 492 perturbations generated (all 4 types, all 3 positions)
- ✅ Expert validation passed (Realism=3.75, Detectability=3.58)

**Completed Phases:**
- ✅ Phase 1: Load Trajectories (600 trajectories from 3 benchmarks)
- ✅ Phase 2: Generate Perturbations (492 perturbations, 11 conditions)
- ✅ Phase 3: Annotation Interface (interactive CLI, stratified sampling)
- ✅ Phase 4: Judge Evaluation (852 Claude evaluations, 64% coverage)
- ✅ Phase 5: CCG Computation (full statistical analysis)

**Infrastructure:**
- ✅ Test suite (93 tests passing)
- ✅ MongoDB Atlas storage (5 collections)
- ✅ AWS Bedrock integration (Claude + GPT-OSS)
- ✅ Experiment runner (load, perturb, annotate, judge, ccg)
- ✅ JSON-based dataset loading for reproducibility

**Ready for:**
- 🎯 LLM judge evaluation (experiment_id: exp_20260403_v2)
- 🎯 Human annotation (50 samples recommended)
- 🔄 Phase 6: Visualization (heatmaps, scatter plots)

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
