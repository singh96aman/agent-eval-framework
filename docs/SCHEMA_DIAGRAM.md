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
                                │ SHA256(entire_config)
                                ▼
                    ┌────────────────────────┐
                    │   experiment_hash      │
                    │   "a3f5b2c1..."       │
                    └────────────────────────┘
                                │
                                │ Used as primary key
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        EXPERIMENTS COLLECTION                         │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  _id: ObjectId("...")                                       │   │
│  │  experiment_hash: "a3f5b2c1..."          ◄─── Primary Key   │   │
│  │                                                              │   │
│  │  config: { ...full JSON... }                                │   │
│  │  status: "in_progress"                                      │   │
│  │                                                              │   │
│  │  ⚠️  NO trajectory_refs array!                              │   │
│  │  ⚠️  Relationship stored via foreign key in trajectories    │   │
│  │                                                              │   │
│  │  progress: {                             ◄─── Only counts   │   │
│  │    trajectories_loaded: 50,                  not IDs!       │   │
│  │    annotations_completed: 25,                               │   │
│  │    evaluations_completed: 150                               │   │
│  │  }                                                           │   │
│  │                                                              │   │
│  │  created_at: "2026-04-02T12:00:00Z"                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                │ Linked via experiment_id (foreign key)
                                │ NOT via array! See trajectories below
                                ▼
        ┌─────────────────────────────────────────────────┐
        │  Query: db.trajectories.find({                  │
        │    experiment_id: "a3f5b2c1..."                 │
        │  })                                             │
        └─────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
   traj_001               traj_002                 traj_003
   experiment_id:         experiment_id:           experiment_id:
   "a3f5b2c1..."         "a3f5b2c1..."           "a3f5b2c1..."
   (Original)             (Perturbed)              (Perturbed)
```

---

## Detailed Collection Schemas

### 1️⃣ EXPERIMENTS Collection (Metadata + References)

```
┌────────────────────────────────────────────────────────────────┐
│  Collection: experiments                                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Document Structure                                  │     │
│  ├──────────────────────────────────────────────────────┤     │
│  │                                                      │     │
│  │  experiment_hash: "a3f5b2c189d7e4f1..."            │     │
│  │  └─ SHA256 of entire config JSON                   │     │
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
│  │  ⚠️  NO ARRAYS OF IDS! Avoids 16MB limit            │     │
│  │                                                      │     │
│  │  To get trajectories for this experiment:           │     │
│  │    db.trajectories.find({                           │     │
│  │      experiment_id: "a3f5b2c1..."                   │     │
│  │    })                                                │     │
│  │  ↑ Uses INDEX for O(1) lookup                       │     │
│  │                                                      │     │
│  │  progress: {                                        │     │
│  │    trajectories_loaded: 50,      ◄─ Just counts    │     │
│  │    annotations_completed: 25,        not IDs!       │     │
│  │    evaluations_completed: 150                       │     │
│  │  }                                                   │     │
│  │                                                      │     │
│  │  created_at: "2026-04-02T12:00:00Z"                │     │
│  │  updated_at: "2026-04-02T15:30:00Z"                │     │
│  │                                                      │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
│  Indexes:                                                      │
│    - experiment_hash (unique)                                  │
│    - status                                                    │
│    - created_at (desc)                                         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

### 2️⃣ TRAJECTORIES Collection (Cached Data)

```
┌────────────────────────────────────────────────────────────────┐
│  Collection: trajectories                                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Original Trajectory                                 │     │
│  ├──────────────────────────────────────────────────────┤     │
│  │                                                      │     │
│  │  trajectory_hash: "d7e9f1a289c4b3e2..."            │     │
│  │  └─ SHA256 of cache_key                            │     │
│  │  └─ UNIQUE INDEX                                    │     │
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
│  │  └─ Parameters that uniquely identify this          │     │
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
│  │  experiment_id: "a3f5b2c1..."  ◄─ Foreign key!       │     │
│  │  └─ Links back to experiments collection            │     │
│  │  └─ INDEXED for fast queries                        │     │
│  │                                                      │     │
│  │  ⚠️  NOT an array! Single value only                │     │
│  │  Each trajectory belongs to ONE experiment          │     │
│  │                                                      │     │
│  │  first_cached: "2026-04-02T12:00:00Z"              │     │
│  │  last_accessed: "2026-04-05T15:30:00Z"             │     │
│  │  access_count: 5                                    │     │
│  │                                                      │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Perturbed Trajectory                                │     │
│  ├──────────────────────────────────────────────────────┤     │
│  │                                                      │     │
│  │  trajectory_hash: "e8f2a3b4c5d6e7f8..."            │     │
│  │                                                      │     │
│  │  cache_key: {                                       │     │
│  │    original_trajectory_hash: "d7e9f1a2...", ◄────┐  │     │
│  │    perturbation_type: "planning",             │  │     │
│  │    perturbation_position: "early",            │  │     │
│  │    perturbation_params: {                     │  │     │
│  │      method: "goal_substitution"              │  │     │
│  │    }                                          │  │     │
│  │  }                                            │  │     │
│  │  └─ Links to original via hash               │  │     │
│  │                                                │  │     │
│  │  trajectory_data: {                            │  │     │
│  │    steps: [                                    │  │     │
│  │      { ... modified step 2 ... },             │  │     │
│  │      { ... },                                  │  │     │
│  │      ...                                       │  │     │
│  │    ],                                          │  │     │
│  │    ground_truth: { ... }                      │  │     │
│  │  }                                             │  │     │
│  │                                                │  │     │
│  │  experiment_id: "a3f5b2c1..."  ◄─ Foreign key  │  │     │
│  │  first_cached: "2026-04-02T13:00:00Z"        │  │     │
│  │  access_count: 1                              │  │     │
│  │                                                │  │     │
│  └────────────────────────────────────────────────│──┘     │
│                                                    │        │
│  Indexes:                                         │        │
│    - trajectory_hash (unique)                     │        │
│    - experiment_id ◄─ CRITICAL for fast queries  │        │
│    - cache_key.benchmark                          │        │
│    - cache_key.original_trajectory_hash  ─────────┘        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

### 3️⃣ ANNOTATIONS Collection

```
┌────────────────────────────────────────────────────────────────┐
│  Collection: annotations                                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │                                                      │     │
│  │  annotation_hash: "f9a3b5c289d7e4f1..."            │     │
│  │  └─ UNIQUE INDEX                                    │     │
│  │                                                      │     │
│  │  experiment_hash: "a3f5b2c1..." ──────┐            │     │
│  │  trajectory_hash: "e8f2a3b4..." ──────┼─ Links     │     │
│  │                                        │            │     │
│  │  annotator: "human_researcher"         │            │     │
│  │                                        │            │     │
│  │  task_success_degradation: 1.0         │            │     │
│  │  └─ 0.0 = no impact, 1.0 = failed      │            │     │
│  │                                        │            │     │
│  │  subsequent_errors: [                  │            │     │
│  │    {                                   │            │     │
│  │      step_number: 3,                   │            │     │
│  │      error_type: "wrong_tool",         │            │     │
│  │      severity: "high"                  │            │     │
│  │    },                                  │            │     │
│  │    {                                   │            │     │
│  │      step_number: 4,                   │            │     │
│  │      error_type: "cascade",            │            │     │
│  │      severity: "critical"              │            │     │
│  │    }                                   │            │     │
│  │  ]                                     │            │     │
│  │  subsequent_error_rate: 0.67           │            │     │
│  │                                        │            │     │
│  │  true_criticality_score: 110.0         │            │     │
│  │  └─ TCS = (TSD × 100) + (SER × 10)     │            │     │
│  │                                        │            │     │
│  │  annotated_at: "2026-04-02T14:00:00Z" │            │     │
│  │                                        │            │     │
│  └────────────────────────────────────────┼────────────┘     │
│                                           │                  │
│  Indexes:                                 │                  │
│    - annotation_hash (unique)             │                  │
│    - experiment_hash ─────────────────────┘                  │
│    - trajectory_hash                                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

### 4️⃣ JUDGE_EVALUATIONS Collection

```
┌────────────────────────────────────────────────────────────────┐
│  Collection: judge_evaluations                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │                                                      │     │
│  │  evaluation_hash: "g1b4c6d389e2f5a7..."            │     │
│  │  └─ UNIQUE INDEX                                    │     │
│  │                                                      │     │
│  │  experiment_hash: "a3f5b2c1..." ──────┐            │     │
│  │  trajectory_hash: "e8f2a3b4..." ──────┼─ Links     │     │
│  │                                        │            │     │
│  │  judge_model: "claude-3.5-sonnet"      │            │     │
│  │  judge_provider: "aws_bedrock"         │            │     │
│  │  sample_number: 1                      │            │     │
│  │  └─ For multiple samples               │            │     │
│  │                                        │            │     │
│  │  overall_score: 85.0                   │            │     │
│  │  └─ 0-100 scale                        │            │     │
│  │                                        │            │     │
│  │  errors_identified: [                  │            │     │
│  │    {                                   │            │     │
│  │      step_number: 2,                   │            │     │
│  │      description: "Imprecise goal",    │            │     │
│  │      severity: 4.0,                    │            │     │
│  │      └─ 0-10 scale                     │            │     │
│  │      reasoning: "..."                  │            │     │
│  │    }                                   │            │     │
│  │  ]                                     │            │     │
│  │                                        │            │     │
│  │  judge_penalty_score: 15.0             │            │     │
│  │  └─ JPS = 100 - overall_score          │            │     │
│  │                                        │            │     │
│  │  raw_response: "The trajectory..."     │            │     │
│  │  prompt_version: "v1.0"                │            │     │
│  │                                        │            │     │
│  │  api_call: {                           │            │     │
│  │    tokens_input: 1234,                 │            │     │
│  │    tokens_output: 567,                 │            │     │
│  │    cost_usd: 0.025,                    │            │     │
│  │    latency_ms: 2345                    │            │     │
│  │  }                                     │            │     │
│  │                                        │            │     │
│  │  evaluated_at: "2026-04-02T15:00:00Z" │            │     │
│  │                                        │            │     │
│  └────────────────────────────────────────┼────────────┘     │
│                                           │                  │
│  Indexes:                                 │                  │
│    - evaluation_hash (unique)             │                  │
│    - experiment_hash ─────────────────────┘                  │
│    - trajectory_hash, judge_model (compound)                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

### 5️⃣ CCG_SCORES Collection

```
┌────────────────────────────────────────────────────────────────┐
│  Collection: ccg_scores                                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │                                                      │     │
│  │  ccg_hash: "h2c5d7e489f1a3b6..."                   │     │
│  │  └─ UNIQUE INDEX                                    │     │
│  │                                                      │     │
│  │  experiment_hash: "a3f5b2c1..." ──────┐            │     │
│  │  trajectory_hash: "e8f2a3b4..." ──────┤            │     │
│  │  annotation_hash: "f9a3b5c2..." ──────┼─ Links     │     │
│  │  evaluation_hash: "g1b4c6d3..." ──────┘            │     │
│  │                                                      │     │
│  │  perturbation_type: "planning"                      │     │
│  │  perturbation_position: "early"                     │     │
│  │  benchmark: "toolbench"                             │     │
│  │  judge_model: "claude-3.5-sonnet"                   │     │
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
│    - ccg_hash (unique)                                         │
│    - experiment_hash                                           │
│    - perturbation_type, perturbation_position (compound)       │
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
       │ SHA256(config)
       ▼
┌──────────────┐
│ Experiment   │───────┐
│ "a3f5b2c1"   │       │
└──────────────┘       │
                       │
                       │ Loads trajectories
                       ▼
           ┌─────────────────────┐
           │ For each trajectory: │
           │ 1. Compute hash      │
           │ 2. Check cache       │
           │ 3. Load if needed    │
           └─────────┬───────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ Traj 1  │  │ Traj 2  │  │ Traj 3  │
  │ (cache  │  │ (cache  │  │ (new)   │
  │  hit)   │  │  hit)   │  │         │
  └────┬────┘  └────┬────┘  └────┬────┘
       │            │            │
       │            │            │ Store in cache
       └────────────┼────────────┘
                    │
                    │ Generate perturbations
                    ▼
       ┌────────────────────────┐
       │ Perturbed trajectories │
       │ - Planning errors      │
       │ - Tool errors          │
       │ - Parameter errors     │
       └────────┬───────────────┘
                │
                │ Human annotation
                ▼
           ┌────────────┐
           │ Annotations│──────┐
           │ TCS values │      │
           └────────────┘      │
                               │
                               │ Judge evaluation
                               ▼
                          ┌────────────┐
                          │ Evaluations│──────┐
                          │ JPS values │      │
                          └────────────┘      │
                                              │
                                              │ Compute CCG
                                              ▼
                                         ┌──────────┐
                                         │ CCG Score│
                                         │ Analysis │
                                         └──────────┘
```

---

## Cache Hit Example

```
User runs: python run.py config/experiments/poc.json

Step 1: Compute experiment_hash
  SHA256(config) → "a3f5b2c1..."

Step 2: Check if experiment exists
  db.experiments.find_one({experiment_hash: "a3f5b2c1"})
  
  ┌─────────────────────────┐
  │ Result: Not found       │ → NEW EXPERIMENT
  │ Continue...             │
  └─────────────────────────┘

Step 3: Load trajectory #1 (ToolBench, index=123)
  cache_key = {
    benchmark: "toolbench",
    hf_index: 123,
    filters: {...}
  }
  
  SHA256(cache_key) → "d7e9f1a2..."
  
  db.trajectories.find_one({trajectory_hash: "d7e9f1a2"})
  
  ┌─────────────────────────┐
  │ Result: Found!          │ → CACHE HIT ✓
  │ Use cached trajectory   │
  │ Update access_count     │
  └─────────────────────────┘

Step 4: Load trajectory #2 (ToolBench, index=456)
  SHA256(cache_key) → "e8f2a3b4..."
  
  db.trajectories.find_one({trajectory_hash: "e8f2a3b4"})
  
  ┌─────────────────────────┐
  │ Result: Not found       │ → CACHE MISS ✗
  │ Load from HuggingFace   │
  │ Store in cache          │
  └─────────────────────────┘

And so on...

Final result:
  - 40 cache hits (instant, $0)
  - 10 cache misses (load from HF, ~$10)
  - Total time: 5 minutes (vs 30 minutes without cache)
  - Total cost: $10 (vs $50 without cache)
```

---

## Key Relationships (Foreign Keys, NOT Arrays!)

```
experiments (1) ────── (M) trajectories
    ▲                      │
    │                      │
    │                      │ experiment_id (foreign key)
    │                      │
    └──────────────────────┘
    
    Query: db.trajectories.find({ experiment_id: "a3f5b2c1..." })
    ↑ Indexed for O(1) lookup, no 16MB limit!


trajectories (1) ───── (M) annotations
    ▲                      │
    │                      │ trajectory_id (foreign key)
    └──────────────────────┘


trajectories (1) ───── (M) judge_evaluations
    ▲                      │
    │                      │ trajectory_id (foreign key)
    └──────────────────────┘


(annotation + evaluation) ───── (1) ccg_score
      (paired to compute CCG via foreign keys)
```

**Why Foreign Keys > Arrays:**
- ✅ No 16MB document size limit
- ✅ Efficient pagination (skip/limit)
- ✅ Fast indexed queries
- ✅ No memory overhead loading experiment metadata
- ❌ Arrays would fail at ~1M trajectory IDs (~40MB)

Is this clearer? Let me know if you want me to:
1. Zoom in on any specific part
2. Show a specific query example
3. Diagram the caching logic in more detail
4. Show multi-experiment scenarios
