# Task 02: POC Implementation - Experimental Infrastructure

## Objective

Build the experimental infrastructure to execute the proof-of-concept defined in [paper/POC_REQUIREMENTS.MD](../../paper/POC_REQUIREMENTS.MD).

This task implements the complete pipeline from dataset loading through CCG computation, enabling the human researcher to run the experiment and validate the hypothesis.

---

## Context

**Previous Task:** Task 01 (Research Foundation) - Literature review and POC design completed, approved by human researcher

**Hypothesis:** LLM judges systematically miscalibrate to error criticality in agent trajectories, under-penalizing early structural failures and over-penalizing late local mistakes.

**Experimental Design:**
- Dual benchmarks: ToolBench (25 trajectories) + GAIA (25 trajectories)
- 3 perturbation types × 3 trajectory positions = 9 conditions
- 2 judges: Claude-3.5-Sonnet (Bedrock) + GPT-OSS 120B
- Metric: Criticality-Calibration Gap (CCG)
- POC-first: 50 trajectories to validate system, scale to 150+ if promising

---

## Deliverables

### 1. Pre-Requisite Verification System
**Location:** `src/prereq_check.py` and `tests/test_prerequisites.py`

**Functionality:**
- Check ToolBench dataset availability and format
- Check GAIA dataset availability and format
- Verify Claude-3.5-Sonnet Bedrock API access
- Verify GPT-OSS 120B endpoint access
- Test sample trajectory execution
- Output: PASS/FAIL report with actionable error messages

**Acceptance Criteria:**
- All checks automated (run via `python src/prereq_check.py`)
- Clear error messages for each failure case
- Test suite in `tests/test_prerequisites.py` passes
- Documented in README section

### 2. Dataset Loaders
**Location:** `src/data/loaders.py`

**Functionality:**
- `load_toolbench_trajectories()` - Load and parse ToolBench format
- `load_gaia_trajectories()` - Load and parse GAIA format
- Unified trajectory schema (compatible with both benchmarks)
- Filter by length, complexity, success criteria
- Sample N trajectories with stratified sampling

**Acceptance Criteria:**
- Loads both datasets successfully
- Returns unified format: `Trajectory` dataclass with steps, metadata, ground_truth
- Tests in `tests/test_loaders.py` pass
- Handles missing files gracefully

### 3. Perturbation Generation Framework
**Location:** `src/perturbations/generator.py`

**Functionality:**
- `PerturbationGenerator` class with methods:
  - `generate_planning_error(trajectory, position)` - Type A
  - `generate_tool_selection_error(trajectory, position)` - Type B
  - `generate_parameter_error(trajectory, position)` - Type C
- Position mapping: early (steps 1-2), middle (steps 3-5), late (steps 6+)
- Plausibility checks (agent could have made this error)
- Save perturbation metadata (type, position, original step, perturbed step)

**Acceptance Criteria:**
- Generates all 9 perturbation types
- Outputs perturbed trajectory + metadata JSON
- Tests in `tests/test_perturbations.py` validate format
- Manual review tool to inspect perturbations

### 4. Judge API Integration
**Location:** `src/judges/api.py`

**Functionality:**
- `ClaudeBedrockJudge` class:
  - Connect to Bedrock (`anthropic.claude-3-5-sonnet-20241022-v2:0`)
  - Handle auth via boto3
  - Retry logic for rate limits
- `GPTOSSJudge` class:
  - Connect to GPT-OSS 120B endpoint
  - Handle API key auth
- Unified interface: `judge.evaluate(trajectory) -> JudgeOutput`
- Prompt templates from POC requirements (Appendix C)
- Parse judge outputs (overall score, per-step errors, severity ratings)

**Acceptance Criteria:**
- Both judges callable via unified interface
- Output format: `JudgeOutput(overall_score, errors: List[StepError])`
- Tests with mock responses in `tests/test_judges.py`
- Handles API failures gracefully (retry, timeout, error messages)
- Cost tracking (token counts, API call logs)

### 5. Ground Truth Annotation Tools
**Location:** `src/annotation/tools.py`

**Functionality:**
- `AnnotationInterface` for human researcher:
  - Display baseline vs. perturbed trajectory side-by-side
  - Prompt for Task Success Degradation (binary: pass/fail)
  - Prompt for Subsequent Error Rate (count errors after perturbation)
  - Save annotations to JSON
- `AnnotationReviewer` for quality control:
  - Load previous annotations
  - Flag inconsistencies for re-review
  - Compute self-agreement metrics

**Acceptance Criteria:**
- CLI tool: `python src/annotation/tools.py --trajectory <id>`
- Annotations saved to `data/annotations/<trajectory_id>.json`
- Simple, fast interface (minimize annotation time)
- Tests validate annotation format in `tests/test_annotation.py`

### 6. CCG Metric Computation
**Location:** `src/metrics/ccg.py`

**Functionality:**
- Compute True Criticality Score (TCS):
  - `TCS = (TSD × 100) + (SER × 10)`
- Compute Judge Penalty Score (JPS):
  - `JPS = 100 - judge_overall_score`
- Compute Criticality-Calibration Gap (CCG):
  - `CCG = (JPS - TCS) / TCS`
- Aggregate by condition (type × position)
- Statistical analysis: ANOVA, post-hoc tests, effect sizes

**Acceptance Criteria:**
- Functions: `compute_tcs()`, `compute_jps()`, `compute_ccg()`
- Tests in `tests/test_ccg.py` with known examples
- Output: CSV with per-trajectory CCG + aggregated stats
- Handles edge cases (TCS = 0, missing judge ratings)

### 7. Experiment Runner Script
**Location:** `src/run_experiment.py`

**Functionality:**
- Orchestrates full pipeline:
  1. Load datasets (ToolBench + GAIA)
  2. Sample 50 trajectories (25 per benchmark)
  3. Generate 50 perturbed trajectories (9 conditions)
  4. Save perturbations for human annotation
  5. Load annotations (after human completes)
  6. Run judge evaluations (2 judges × 3 samples × 50 = 300 calls)
  7. Compute CCG scores
  8. Generate output reports
- Checkpointing: Resume from any stage
- Logging: Track progress, API calls, costs

**Acceptance Criteria:**
- CLI: `python src/run_experiment.py --stage <stage_name>`
- Stages: `sample`, `perturb`, `annotate`, `judge`, `analyze`
- Each stage saves state to `results/poc_run_<timestamp>/`
- Tests validate each stage independently in `tests/test_experiment.py`

### 8. Visualization and Reporting
**Location:** `src/visualization/plots.py`

**Functionality:**
- Generate heatmap: CCG by perturbation type (rows) × position (columns)
- Generate scatter plots: TCS vs. JPS per condition
- Generate summary statistics table
- Save outputs to `results/poc_run_<timestamp>/figures/`

**Acceptance Criteria:**
- Functions: `plot_ccg_heatmap()`, `plot_tcs_jps_scatter()`
- Outputs saved as PNG and PDF
- Tests verify plot generation in `tests/test_visualization.py`

---

## Implementation Phases

### Phase 1: Foundation (Days 1-3)
- Pre-requisite verification system
- Dataset loaders (ToolBench + GAIA)
- Unified trajectory schema
- Tests for loaders

**Checkpoint:** Pre-requisite checks pass, datasets load successfully

### Phase 2: Perturbations (Days 4-6)
- Perturbation generator for all 9 conditions
- Metadata tracking
- Manual review tool
- Tests for perturbations

**Checkpoint:** 50 perturbed trajectories generated and reviewable

### Phase 3: Judge Integration (Days 7-9)
- Claude Bedrock API integration
- GPT-OSS 120B API integration
- Unified judge interface
- Tests with mocks

**Checkpoint:** Both judges callable, test trajectories evaluated

### Phase 4: Annotation & Metrics (Days 10-12)
- Annotation tools for human researcher
- CCG metric computation
- Statistical analysis functions
- Tests for metrics

**Checkpoint:** Annotation tool usable, CCG computable from sample data

### Phase 5: Experiment Runner (Days 13-15)
- Orchestration script with checkpointing
- Logging and cost tracking
- Visualization and reporting
- End-to-end integration tests

**Checkpoint:** Full pipeline runnable end-to-end

---

## Repository Structure After Implementation

```
repo/
├── src/
│   ├── prereq_check.py          # Pre-requisite verification
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py           # ToolBench + GAIA loaders
│   │   └── schema.py            # Unified trajectory schema
│   ├── perturbations/
│   │   ├── __init__.py
│   │   └── generator.py         # Perturbation generation
│   ├── judges/
│   │   ├── __init__.py
│   │   └── api.py               # Claude + GPT-OSS integration
│   ├── annotation/
│   │   ├── __init__.py
│   │   └── tools.py             # Human annotation interface
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── ccg.py               # CCG computation
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py             # Heatmaps, scatter plots
│   └── run_experiment.py        # Main orchestration script
├── tests/
│   ├── test_prerequisites.py
│   ├── test_loaders.py
│   ├── test_perturbations.py
│   ├── test_judges.py
│   ├── test_annotation.py
│   ├── test_ccg.py
│   ├── test_experiment.py
│   └── test_visualization.py
├── data/
│   ├── toolbench/               # ToolBench dataset (user provides)
│   ├── gaia/                    # GAIA dataset (user provides)
│   ├── perturbed/               # Generated perturbed trajectories
│   └── annotations/             # Human annotations
├── results/
│   └── poc_run_<timestamp>/     # Experiment outputs
│       ├── perturbations.json
│       ├── judge_outputs.json
│       ├── ccg_scores.csv
│       └── figures/
│           ├── ccg_heatmap.png
│           └── tcs_jps_scatter.png
└── requirements.txt             # Updated with dependencies
```

---

## Dependencies to Add

**Add to `requirements.txt`:**
```
boto3>=1.34.0           # AWS Bedrock SDK
anthropic>=0.25.0       # Claude API (if direct API used)
openai>=1.0.0           # For GPT-OSS endpoint compatibility
pandas>=2.0.0           # Data manipulation
numpy>=1.24.0           # Numerical operations
matplotlib>=3.7.0       # Plotting
seaborn>=0.12.0         # Statistical visualizations
scipy>=1.10.0           # Statistical tests
pytest>=7.4.0           # Testing framework
pytest-mock>=3.11.0     # Mocking for tests
python-dotenv>=1.0.0    # Environment variables (.env)
tqdm>=4.65.0            # Progress bars
```

---

## Success Criteria

This task is complete when:

1. ✅ All 8 deliverables implemented and tested
2. ✅ Pre-requisite checks pass on development machine
3. ✅ All tests in `tests/` pass (run via `pytest tests/`)
4. ✅ Experiment runner executes end-to-end (with mock data)
5. ✅ Code is documented (docstrings, type hints)
6. ✅ README updated with:
   - Setup instructions
   - How to run pre-requisite checks
   - How to run experiment
   - How to review perturbations
   - How to annotate trajectories
7. ✅ Human researcher can run the pipeline without code modification
8. ✅ All code committed to repo with passing tests

---

## Testing Strategy

**Unit Tests:**
- Each module has corresponding test file in `tests/`
- Test happy paths and error cases
- Use mocks for external APIs (judges, dataset loading)

**Integration Tests:**
- `tests/test_experiment.py` runs full pipeline with sample data
- Validates checkpointing and resumption
- Checks output formats

**Manual Tests:**
- Human researcher runs pre-requisite checks
- Human researcher generates pilot perturbations (12 samples)
- Human researcher annotates 2-3 sample trajectories
- Human researcher reviews outputs and visualizations

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **ToolBench/GAIA dataset format unknown** | Start with schema exploration, create adapters for format variations |
| **Claude Bedrock auth issues** | Pre-requisite check catches early, document IAM setup in README |
| **GPT-OSS 120B endpoint unavailable** | Support multiple backends (local, hosted), fallback to GPT-4 if needed |
| **Judge output parsing fragile** | Strict prompt templates, validation, graceful degradation |
| **Annotation tool too complex** | Keep CLI simple, prioritize speed over features |
| **Perturbations not realistic** | Manual review tool, human researcher validates pilot batch |

---

## Next Steps After This Task

Once implementation is complete:
- **Task 03:** Execute POC experiment (human annotation + judge evaluation)
- **Task 04:** Analyze results and decide on scaling
- **Task 05:** Write paper draft (if results promising)

---

## References

- [paper/POC_REQUIREMENTS.MD](../../paper/POC_REQUIREMENTS.MD) - Detailed experimental protocol
- [paper/LITERATURE_REVIEW.MD](../../paper/LITERATURE_REVIEW.MD) - Related work and positioning
- [agent_tasks/01_research_foundation/](../01_research_foundation/) - Previous task context
