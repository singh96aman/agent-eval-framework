# MongoDB Schema Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                       │
│                        EXPERIMENT CONFIG (JSON)                       │
│                    config/experiments/poc.json                        │
│                                                                       │
│  {                                                                    │
│    datasets: {toolbench: {...}, gaia: {...}},                        │
│    perturbations: {types: [...], positions: [...]},                  │
│    judges: {models: [...]},                                          │
│    annotation: {...}                                                  │
│  }                                                                    │
│                                                                       │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                │ Defines experiment
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        EXPERIMENTS COLLECTION                         │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  experiment_id: "exp_poc_toolbench_20260402" ◄─ Primary Key │   │
│  │  config: { ...full JSON... }                                │   │
│  │  status: "in_progress"                                      │   │
│  │  progress: {                                                │   │
│  │    trajectories_loaded: 50,                                 │   │
│  │    perturbations_generated: 450,                            │   │
│  │    annotations_completed: 25,                               │   │
│  │    evaluations_completed: 150                               │   │
│  │  }                                                           │   │
│  │  created_at: "2026-04-02T12:00:00Z"                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                │ References trajectories via
                                │ downstream entities (perturbations, etc.)
                                ▼
        ┌─────────────────────────────────────────────────┐
        │  Pure Cache Layer - NO experiment references!   │
        └─────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    TRAJECTORIES (Pure Cache)                          │
│                   ⚠️  NO experiment_id field!                         │
├─────────────────────────────────────────────────────────────────────┤
│  trajectory_id (PK) - Unique identifier                              │
│  cache_key {...} - What makes this trajectory unique                 │
│  trajectory_data {...} - Steps, ground truth, etc.                   │
│  benchmark - "toolbench", "gaia", etc.                               │
│  is_perturbed - true/false                                           │
│  original_trajectory_id - (if perturbed, FK to original)             │
│  stored_at - When cached                                             │
│                                                                       │
│  Used by multiple experiments with NO coupling!                      │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                │ Referenced by experiment-scoped entities
                                ▼
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
   PERTURBATIONS          ANNOTATIONS            JUDGE_EVALUATIONS
   (has exp_id)          (has exp_id)           (has exp_id)
```

---

## Key Design Principle

**Trajectories = Pure Cache (Zero Experiment Coupling)**

- Trajectories are stored once and referenced by multiple experiments
- All experiment context lives in **downstream entities**
- Query pattern: `db.perturbations.find({experiment_id: "exp_B"})` → get trajectory_ids → lookup trajectories

---

## Detailed Collection Schemas

### 1️⃣ EXPERIMENTS Collection (Metadata Only)

```
┌────────────────────────────────────────────────────────────────┐
│  Collection: experiments                                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Document Structure                                  │     │
│  ├──────────────────────────────────────────────────────┤     │
│  │                                                      │     │
│  │  experiment_id: "exp_poc_toolbench_20260402"       │     │
│  │  └─ User-provided or auto-generated                │     │
│  │  └─ UNIQUE INDEX                                    │     │
│  │                                                      │     │
│  │  config: {                                          │     │
│  │    datasets: {...},                                 │     │
│  │    perturbations: {...},                            │     │
│  │    judges: {...},                                   │     │
│  │    annotation: {...}                                │     │
│  │  }                                                   │     │
│  │  └─ Full config stored for reproducibility          │     │
│  │                                                      │     │
│  │  status: "in_progress"                              │     │
│  │  └─ created | in_progress | completed | failed      │     │
│  │                                                      │     │
│  │  progress: {                                        │     │
│  │    trajectories_loaded: 50,      ◄─ Counts only    │     │
│  │    perturbations_generated: 450,                    │     │
│  │    annotations_completed: 25,                       │     │
│  │    evaluations_completed: 150                       │     │
│  │  }                                                   │     │
│  │                                                      │     │
│  │  created_at: "2026-04-02T12:00:00Z"                │     │
│  │  updated_at: "2026-04-02T15:30:00Z"                │     │
│  │                                                      │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
│  Indexes:                                                      │
│    - experiment_id (unique)                                    │
│    - status                                                    │
│    - created_at (desc)                                         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

### 2️⃣ TRAJECTORIES Collection (Pure Cache - NO Experiment References!)

```
┌────────────────────────────────────────────────────────────────┐
│  Collection: trajectories                                      │
│  ⚠️  NO experiment_id field - Pure cache layer!                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Original Trajectory                                 │     │
│  ├──────────────────────────────────────────────────────┤     │
│  │                                                      │     │
│  │  trajectory_id: "traj_toolbench_123"               │     │
│  │  └─ UNIQUE INDEX (Primary Key)                     │     │
│  │                                                      │     │
│  │  cache_key: {                                       │     │
│  │    benchmark: "toolbench",                          │     │
│  │    hf_index: 123,                                   │     │
│  │    split: "train",                                  │     │
│  │    filters: {                                       │     │
│  │      min_steps: 5,                                  │     │
│  │      max_steps: 10                                  │     │
│  │    },                                               │     │
│  │    is_perturbed: false,                             │     │
│  │    perturbation: null                               │     │
│  │  }                                                   │     │
│  │  └─ What makes this trajectory unique               │     │
│  │                                                      │     │
│  │  trajectory_data: {                                 │     │
│  │    steps: [                                         │     │
│  │      {                                              │     │
│  │        step_number: 1,                              │     │
│  │        step_type: "reasoning",                      │     │
│  │        content: "I need to search...",              │     │
│  │        tool_name: null                              │     │
│  │      },                                             │     │
│  │      {                                              │     │
│  │        step_number: 2,                              │     │
│  │        step_type: "tool_execution",                 │     │
│  │        content: "Use Search",                       │     │
│  │        tool_name: "Search",                         │     │
│  │        tool_input: {"query": "..."},                │     │
│  │        tool_output: "..."                           │     │
│  │      }                                              │     │
│  │    ],                                               │     │
│  │    ground_truth: {                                  │     │
│  │      task_description: "...",                       │     │
│  │      expected_answer: "...",                        │     │
│  │      task_success: true                             │     │
│  │    }                                                │     │
│  │  }                                                   │     │
│  │                                                      │     │
│  │  benchmark: "toolbench"  ◄─ Indexed for queries     │     │
│  │  is_perturbed: false                                │     │
│  │  original_trajectory_id: null                       │     │
│  │                                                      │     │
│  │  stored_at: "2026-04-02T12:00:00Z"                 │     │
│  │                                                      │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Perturbed Trajectory (Still Pure Cache!)           │     │
│  ├──────────────────────────────────────────────────────┤     │
│  │                                                      │     │
│  │  trajectory_id: "traj_perturbed_e8f2a3b4"          │     │
│  │                                                      │     │
│  │  cache_key: {                                       │     │
│  │    original_trajectory_id: "traj_toolbench_123",   │     │
│  │    perturbation_type: "planning",                   │     │
│  │    perturbation_position: "early",                  │     │
│  │    perturbation_params: {                           │     │
│  │      method: "goal_substitution"                    │     │
│  │    }                                                │     │
│  │  }                                                   │     │
│  │                                                      │     │
│  │  trajectory_data: {                                 │     │
│  │    steps: [                                         │     │
│  │      { ... modified step 2 ... },                   │     │
│  │      { ... },                                       │     │
│  │      ...                                            │     │
│  │    ],                                               │     │
│  │    ground_truth: { ... }                           │     │
│  │  }                                                   │     │
│  │                                                      │     │
│  │  benchmark: "toolbench"                             │     │
│  │  is_perturbed: true                                 │     │
│  │  original_trajectory_id: "traj_toolbench_123"      │     │
│  │                                                      │     │
│  │  stored_at: "2026-04-02T13:00:00Z"                 │     │
│  │                                                      │     │
│  │  ⚠️  NO experiment_id - Experiment context is in    │     │
│  │      the perturbations collection!                  │     │
│  │                                                      │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
│  Indexes:                                                      │
│    - trajectory_id (unique) ◄─ Primary key                    │
│    - benchmark                                                 │
│    - is_perturbed                                              │
│    - original_trajectory_id ◄─ Link perturbed → original      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

### 3️⃣ PERTURBATIONS Collection (NEW! - Experiment-Scoped)

```
┌────────────────────────────────────────────────────────────────┐
│  Collection: perturbations                                     │
│  Links experiments → original → perturbed trajectories         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │                                                      │     │
│  │  perturbation_id: "pert_a1b2c3d4"                   │     │
│  │  └─ UNIQUE INDEX (Primary Key)                     │     │
│  │                                                      │     │
│  │  experiment_id: "exp_poc_toolbench_20260402"       │     │
│  │  └─ FK to experiments ◄─ INDEXED                   │     │
│  │                                                      │     │
│  │  original_trajectory_id: "traj_toolbench_123"      │     │
│  │  └─ FK to trajectories (original)                  │     │
│  │                                                      │     │
│  │  perturbed_trajectory_id: "traj_perturbed_e8f2"    │     │
│  │  └─ FK to trajectories (perturbed version)         │     │
│  │                                                      │     │
│  │  perturbation_type: "planning"                      │     │
│  │  └─ planning | tool_selection | parameter           │     │
│  │                                                      │     │
│  │  perturbation_position: "early"                     │     │
│  │  └─ early | middle | late                           │     │
│  │                                                      │     │
│  │  perturbation_config: {                             │     │
│  │    method: "goal_substitution",                     │     │
│  │    step_number: 2,                                  │     │
│  │    original_step: {...},                            │     │
│  │    modified_step: {...}                             │     │
│  │  }                                                   │     │
│  │  └─ Full details of the perturbation               │     │
│  │                                                      │     │
│  │  created_at: "2026-04-02T13:00:00Z"                │     │
│  │                                                      │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
│  Indexes:                                                      │
│    - perturbation_id (unique)                                  │
│    - experiment_id ◄─ CRITICAL for experiment queries         │
│    - original_trajectory_id                                    │
│    - perturbed_trajectory_id                                   │
│    - (perturbation_type, perturbation_position) compound       │
│                                                                │
│  Query Pattern:                                                │
│    # Get all perturbations for experiment B                    │
│    perturbations = db.perturbations.find({                     │
│      "experiment_id": "exp_B"                                  │
│    })                                                          │
│                                                                │
│    # Get trajectory data                                       │
│    for pert in perturbations:                                  │
│      original = db.trajectories.find_one({                     │
│        "trajectory_id": pert["original_trajectory_id"]         │
│      })                                                        │
│      perturbed = db.trajectories.find_one({                    │
│        "trajectory_id": pert["perturbed_trajectory_id"]        │
│      })                                                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

### 4️⃣ ANNOTATIONS Collection (Experiment-Scoped)

```
┌────────────────────────────────────────────────────────────────┐
│  Collection: annotations                                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │                                                      │     │
│  │  annotation_id: "ann_f9a3b5c2"                      │     │
│  │  └─ UNIQUE INDEX (Primary Key)                     │     │
│  │                                                      │     │
│  │  experiment_id: "exp_poc_toolbench_20260402"       │     │
│  │  └─ FK to experiments ◄─ INDEXED                   │     │
│  │                                                      │     │
│  │  trajectory_id: "traj_perturbed_e8f2"              │     │
│  │  └─ FK to trajectories ◄─ INDEXED                  │     │
│  │                                                      │     │
│  │  annotator: "human_researcher"                      │     │
│  │                                                      │     │
│  │  task_success_degradation: 1.0                      │     │
│  │  └─ 0.0 = no impact, 1.0 = failed                   │     │
│  │                                                      │     │
│  │  subsequent_errors: [                               │     │
│  │    {                                                │     │
│  │      step_number: 3,                                │     │
│  │      error_type: "wrong_tool",                      │     │
│  │      severity: "high"                               │     │
│  │    },                                               │     │
│  │    {                                                │     │
│  │      step_number: 4,                                │     │
│  │      error_type: "cascade",                         │     │
│  │      severity: "critical"                           │     │
│  │    }                                                │     │
│  │  ]                                                   │     │
│  │  subsequent_error_rate: 0.67                        │     │
│  │                                                      │     │
│  │  true_criticality_score: 110.0                      │     │
│  │  └─ TCS = (TSD × 100) + (SER × 10)                  │     │
│  │                                                      │     │
│  │  annotated_at: "2026-04-02T14:00:00Z"              │     │
│  │                                                      │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
│  Indexes:                                                      │
│    - annotation_id (unique)                                    │
│    - experiment_id ◄─ Query all annotations for experiment    │
│    - trajectory_id                                             │
│    - annotator                                                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

### 5️⃣ JUDGE_EVALUATIONS Collection (Experiment-Scoped)

```
┌────────────────────────────────────────────────────────────────┐
│  Collection: judge_evaluations                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │                                                      │     │
│  │  evaluation_id: "eval_g1b4c6d3"                     │     │
│  │  └─ UNIQUE INDEX (Primary Key)                     │     │
│  │                                                      │     │
│  │  experiment_id: "exp_poc_toolbench_20260402"       │     │
│  │  └─ FK to experiments ◄─ INDEXED                   │     │
│  │                                                      │     │
│  │  trajectory_id: "traj_perturbed_e8f2"              │     │
│  │  └─ FK to trajectories ◄─ INDEXED                  │     │
│  │                                                      │     │
│  │  judge_model: "claude-sonnet-4.5"                   │     │
│  │  judge_provider: "aws_bedrock"                      │     │
│  │  sample_number: 1                                   │     │
│  │  └─ For multiple samples                            │     │
│  │                                                      │     │
│  │  overall_score: 85.0                                │     │
│  │  └─ 0-100 scale                                     │     │
│  │                                                      │     │
│  │  errors_identified: [                               │     │
│  │    {                                                │     │
│  │      step_number: 2,                                │     │
│  │      description: "Imprecise goal",                 │     │
│  │      severity: 4.0,                                 │     │
│  │      └─ 0-10 scale                                  │     │
│  │      reasoning: "..."                               │     │
│  │    }                                                │     │
│  │  ]                                                   │     │
│  │                                                      │     │
│  │  judge_penalty_score: 15.0                          │     │
│  │  └─ JPS = 100 - overall_score                       │     │
│  │                                                      │     │
│  │  raw_response: "The trajectory..."                  │     │
│  │  prompt_version: "v1.0"                             │     │
│  │                                                      │     │
│  │  api_call: {                                        │     │
│  │    tokens_input: 1234,                              │     │
│  │    tokens_output: 567,                              │     │
│  │    cost_usd: 0.025,                                 │     │
│  │    latency_ms: 2345                                 │     │
│  │  }                                                   │     │
│  │                                                      │     │
│  │  evaluated_at: "2026-04-02T15:00:00Z"              │     │
│  │                                                      │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
│  Indexes:                                                      │
│    - evaluation_id (unique)                                    │
│    - experiment_id ◄─ Query all evaluations for experiment    │
│    - (trajectory_id, judge_model) compound                     │
│    - judge_model                                               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

### 6️⃣ CCG_SCORES Collection (Experiment-Scoped)

```
┌────────────────────────────────────────────────────────────────┐
│  Collection: ccg_scores                                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │                                                      │     │
│  │  ccg_id: "ccg_h2c5d7e4"                             │     │
│  │  └─ UNIQUE INDEX (Primary Key)                     │     │
│  │                                                      │     │
│  │  experiment_id: "exp_poc_toolbench_20260402"       │     │
│  │  └─ FK to experiments ◄─ INDEXED                   │     │
│  │                                                      │     │
│  │  trajectory_id: "traj_perturbed_e8f2"              │     │
│  │  └─ FK to trajectories ◄─ INDEXED                  │     │
│  │                                                      │     │
│  │  annotation_id: "ann_f9a3b5c2"                      │     │
│  │  └─ FK to annotations                               │     │
│  │                                                      │     │
│  │  evaluation_id: "eval_g1b4c6d3"                     │     │
│  │  └─ FK to judge_evaluations                         │     │
│  │                                                      │     │
│  │  perturbation_type: "planning"                      │     │
│  │  perturbation_position: "early"                     │     │
│  │  benchmark: "toolbench"                             │     │
│  │  judge_model: "claude-sonnet-4.5"                   │     │
│  │                                                      │     │
│  │  true_criticality_score: 110.0                      │     │
│  │  └─ From annotation                                 │     │
│  │                                                      │     │
│  │  judge_penalty_score: 15.0                          │     │
│  │  └─ From judge evaluation                           │     │
│  │                                                      │     │
│  │  criticality_calibration_gap: -0.86                 │     │
│  │  └─ CCG = (JPS - TCS) / TCS                         │     │
│  │  └─ Negative = under-penalized                      │     │
│  │                                                      │     │
│  │  calibration_status: "under_penalized"              │     │
│  │  gap_magnitude: "severe"                            │     │
│  │                                                      │     │
│  │  computed_at: "2026-04-02T16:00:00Z"               │     │
│  │                                                      │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
│  Indexes:                                                      │
│    - ccg_id (unique)                                           │
│    - experiment_id ◄─ Query all CCG scores for experiment     │
│    - trajectory_id                                             │
│    - annotation_id                                             │
│    - evaluation_id                                             │
│    - (perturbation_type, perturbation_position) compound       │
│    - judge_model                                               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Visualization

```
┌──────────────┐
│ Config JSON  │
│              │
│ poc.json     │
└──────┬───────┘
       │
       │ Create experiment
       ▼
┌──────────────────┐
│ Experiment       │
│ "exp_B"          │
└──────────────────┘
       │
       │ Load trajectories (phase 1)
       ▼
┌─────────────────────────────────┐
│ For each trajectory:            │
│ 1. Check if in cache            │
│ 2. If not, load from HF         │
│ 3. Store in trajectories        │
│    (NO experiment_id!)          │
└────────┬────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ TRAJECTORIES (Pure Cache)          │
│ - traj_001 (original)              │
│ - traj_002 (original)              │
│ - ...                              │
│ - traj_050 (original)              │
│                                    │
│ ⚠️  NO experiment references!      │
└────────┬───────────────────────────┘
         │
         │ Generate perturbations (phase 2)
         ▼
┌────────────────────────────────────┐
│ For each perturbation:             │
│ 1. Create perturbed trajectory     │
│ 2. Store in trajectories           │
│ 3. Create perturbation record      │
│    linking exp_B → original        │
│    → perturbed                     │
└────────┬───────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ PERTURBATIONS (Experiment-Scoped)      │
│ - pert_001: exp_B, traj_001→traj_051   │
│ - pert_002: exp_B, traj_001→traj_052   │
│ - ...                                  │
│ - pert_450: exp_B, traj_050→traj_500   │
└────────┬───────────────────────────────┘
         │
         │ Human annotation (phase 3)
         ▼
┌────────────────────────────┐
│ ANNOTATIONS                │
│ - ann_001: exp_B, traj_051 │
│ - ann_002: exp_B, traj_052 │
│ - ...                      │
└────────┬───────────────────┘
         │
         │ Judge evaluation (phase 4)
         ▼
┌─────────────────────────────┐
│ JUDGE_EVALUATIONS           │
│ - eval_001: exp_B, traj_051 │
│ - eval_002: exp_B, traj_052 │
│ - ...                       │
└────────┬────────────────────┘
         │
         │ Compute CCG (phase 5)
         ▼
┌────────────────────────────┐
│ CCG_SCORES                 │
│ - ccg_001: exp_B, ...      │
│ - ccg_002: exp_B, ...      │
│ - ...                      │
└────────────────────────────┘
```

---

## Multi-Experiment Workflow Example

### Scenario: Same trajectories, different perturbations

**Experiment A (load only):**
```
Phase 1: Load 50 trajectories
  → Store in trajectories collection (traj_001..traj_050)
  → NO experiment_id stored on trajectories
  → Experiment A has NO perturbations
```

**Experiment B (same load + perturbation set 1):**
```
Phase 1: Load 50 trajectories
  → Check cache: traj_001..traj_050 already exist ✓
  → NO writes to trajectories (cache hit!)

Phase 2: Generate 450 perturbations (9 conditions × 50 trajs)
  → Store perturbed trajectories (traj_051..traj_500)
  → Create perturbation records:
      db.perturbations.insert_many([
        {
          experiment_id: "exp_B",
          original_trajectory_id: "traj_001",
          perturbed_trajectory_id: "traj_051",
          perturbation_type: "planning",
          perturbation_position: "early"
        },
        ...
      ])
```

**Experiment C (same load + perturbation set 2):**
```
Phase 1: Load 50 trajectories
  → Cache hit again! traj_001..traj_050 reused

Phase 2: Generate 450 NEW perturbations
  → Store perturbed trajectories (traj_501..traj_950)
  → Create perturbation records:
      db.perturbations.insert_many([
        {
          experiment_id: "exp_C",
          original_trajectory_id: "traj_001",
          perturbed_trajectory_id: "traj_501",
          perturbation_type: "tool_selection",
          perturbation_position: "early"
        },
        ...
      ])
```

**Result:**
- Original trajectories stored ONCE (traj_001..traj_050)
- Experiment B perturbations: traj_051..traj_500
- Experiment C perturbations: traj_501..traj_950
- Zero duplication of originals
- Easy to compare: query by experiment_id

---

## Query Patterns

### Get all data for an experiment

```javascript
// 1. Get experiment metadata
const experiment = db.experiments.find_one({
  experiment_id: "exp_B"
});

// 2. Get all perturbations for this experiment
const perturbations = db.perturbations.find({
  experiment_id: "exp_B"
});

// 3. Get unique original trajectory IDs
const original_ids = [...new Set(
  perturbations.map(p => p.original_trajectory_id)
)];

// 4. Get original trajectories
const originals = db.trajectories.find({
  trajectory_id: { $in: original_ids }
});

// 5. Get all perturbed trajectory IDs
const perturbed_ids = perturbations.map(p => p.perturbed_trajectory_id);

// 6. Get perturbed trajectories
const perturbed = db.trajectories.find({
  trajectory_id: { $in: perturbed_ids }
});

// 7. Get annotations
const annotations = db.annotations.find({
  experiment_id: "exp_B"
});

// 8. Get evaluations
const evaluations = db.judge_evaluations.find({
  experiment_id: "exp_B"
});

// 9. Get CCG scores
const ccg_scores = db.ccg_scores.find({
  experiment_id: "exp_B"
});
```

### Compare experiments

```javascript
// Find which original trajectories are shared
const exp_B_originals = db.perturbations.distinct(
  "original_trajectory_id",
  { experiment_id: "exp_B" }
);

const exp_C_originals = db.perturbations.distinct(
  "original_trajectory_id",
  { experiment_id: "exp_C" }
);

const shared_originals = exp_B_originals.filter(
  id => exp_C_originals.includes(id)
);

// Compare CCG scores for same perturbation type/position
db.ccg_scores.aggregate([
  {
    $match: {
      experiment_id: { $in: ["exp_B", "exp_C"] },
      perturbation_type: "planning",
      perturbation_position: "early"
    }
  },
  {
    $group: {
      _id: "$experiment_id",
      avg_ccg: { $avg: "$criticality_calibration_gap" },
      count: { $sum: 1 }
    }
  }
]);
```

---

## Key Relationships (Foreign Keys)

```
experiments (1) ────────────────── (M) perturbations
                                        │
                                        │ experiment_id
                                        │
                                        
experiments (1) ────────────────── (M) annotations
                                        │
                                        │ experiment_id

experiments (1) ────────────────── (M) judge_evaluations
                                        │
                                        │ experiment_id

experiments (1) ────────────────── (M) ccg_scores
                                        │
                                        │ experiment_id

trajectories (1) ───────────────── (M) perturbations
                                        │
                                        │ original_trajectory_id
                                        │ perturbed_trajectory_id

trajectories (1) ───────────────── (M) annotations
                                        │
                                        │ trajectory_id

trajectories (1) ───────────────── (M) judge_evaluations
                                        │
                                        │ trajectory_id
```

**Why This Design?**
✅ Trajectories cached independently of experiments  
✅ Zero experiment_id writes to trajectories  
✅ No array bottlenecks, no junction table overhead  
✅ Experiment context lives where it's used  
✅ Simple queries: `find({experiment_id: "exp_B"})`  
✅ Easy to share trajectories across experiments  
✅ Clean separation of concerns  

---

## Benefits Summary

| Aspect | Old Design | New Design |
|--------|------------|------------|
| Trajectory reuse | ❌ Must duplicate or update | ✅ Pure cache, zero writes |
| Experiment coupling | ❌ experiment_id in trajectories | ✅ Zero coupling |
| Query pattern | ❌ Complex OR queries | ✅ Simple experiment_id filter |
| Bottlenecks | ❌ Array updates on trajectories | ✅ None |
| Comparison | ❌ Hard to track shared trajs | ✅ Easy via perturbations |
| Schema clarity | ❌ Mixed concerns | ✅ Clear separation |
