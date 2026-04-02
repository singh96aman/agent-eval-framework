# MongoDB Schema Design for Agent Judge Calibration Experiment

## Design Philosophy

**Scalability Goals:**
1. Support multiple experiments with different configurations
2. Enable cross-experiment analysis and comparison
3. Allow incremental data addition without schema changes
4. Facilitate efficient querying by experiment, condition, judge, etc.
5. Support future expansion (new perturbation types, judges, metrics)

**Key Design Decisions:**
- Each document is self-contained (denormalized for read performance)
- `experiment_id` as foreign key for cross-collection queries
- Indexed fields for common query patterns
- Embedded arrays for related data (steps, errors)
- Timestamps for temporal analysis

---

## Collection Schemas

### 1. `experiments` Collection

**Purpose:** Track experiment metadata and configurations

```javascript
{
  _id: ObjectId("..."),
  experiment_id: "poc_2026_04_02_001",  // Unique identifier
  name: "POC: Judge Calibration Study",
  description: "Initial 50-trajectory POC with dual benchmarks",
  
  // Configuration
  config: {
    benchmarks: ["toolbench", "gaia"],
    num_trajectories_per_benchmark: 25,
    perturbation_types: ["planning", "tool_selection", "parameter"],
    perturbation_positions: ["early", "middle", "late"],
    judges: [
      {
        name: "claude-3.5-sonnet",
        model_id: "anthropic.claude-3-5-sonnet-20241022-v2:0",
        provider: "aws_bedrock"
      },
      {
        name: "gpt-oss-120b",
        model_id: "meta.llama3-70b-instruct-v1:0",
        provider: "aws_bedrock"
      }
    ],
    random_seed: 42,
    samples_per_trajectory: 3
  },
  
  // Status tracking
  status: "in_progress",  // created | in_progress | completed | failed
  progress: {
    trajectories_loaded: 50,
    perturbations_generated: 50,
    annotations_completed: 25,
    judge_evaluations_completed: 150,
    ccg_scores_computed: 150
  },
  
  // Timestamps
  created_at: ISODate("2026-04-02T12:00:00Z"),
  started_at: ISODate("2026-04-02T13:00:00Z"),
  completed_at: null,
  updated_at: ISODate("2026-04-02T15:30:00Z"),
  
  // Metadata
  created_by: "human_researcher",
  tags: ["poc", "dual_benchmark", "phase_1"],
  notes: "Initial proof-of-concept experiment"
}
```

**Indexes:**
- `experiment_id` (unique)
- `status`
- `created_at` (descending)
- `tags`

---

### 2. `trajectories` Collection

**Purpose:** Store original and perturbed trajectories

```javascript
{
  _id: ObjectId("..."),
  trajectory_id: "toolbench_hf_123",  // Unique identifier
  
  // Experiment linkage
  experiment_id: "poc_2026_04_02_001",  // Links to experiment
  
  // Trajectory metadata
  benchmark: "toolbench",  // toolbench | gaia
  is_perturbed: false,     // true if this is a perturbed version
  original_trajectory_id: null,  // If perturbed, links to original
  
  // Perturbation metadata (only if is_perturbed = true)
  perturbation: {
    type: "planning",           // planning | tool_selection | parameter
    position: "early",          // early | middle | late
    perturbed_step_number: 2,
    original_step_content: "...",
    perturbed_step_content: "...",
    metadata: {
      // Additional perturbation-specific data
    }
  },
  
  // Trajectory content (unified schema)
  steps: [
    {
      step_id: "toolbench_hf_123_step_1",
      step_number: 1,
      step_type: "reasoning",  // planning | tool_selection | tool_execution | reasoning | validation | final_answer
      content: "I need to search for Tokyo population",
      tool_name: null,
      tool_input: null,
      tool_output: null,
      metadata: {}
    },
    {
      step_id: "toolbench_hf_123_step_2",
      step_number: 2,
      step_type: "tool_execution",
      content: "Use Search tool",
      tool_name: "Search",
      tool_input: {"query": "Tokyo population 2023"},
      tool_output: "14.09 million",
      metadata: {}
    }
  ],
  
  // Ground truth
  ground_truth: {
    task_description: "Find the population of Tokyo in 2023",
    expected_answer: "14.09 million",
    task_success: true,
    success_criteria: "exact_match",
    difficulty: "Level 1",
    domain: "geography"
  },
  
  // Source tracking
  source: {
    dataset: "toolbench",
    hf_index: 123,
    hf_split: "train",
    file_name: null
  },
  
  // Timestamps
  stored_at: ISODate("2026-04-02T12:30:00Z"),
  updated_at: ISODate("2026-04-02T12:30:00Z")
}
```

**Indexes:**
- `trajectory_id` (unique)
- `experiment_id`
- `benchmark`
- `is_perturbed`
- `perturbation.type, perturbation.position` (compound, for perturbed only)
- `original_trajectory_id`

---

### 3. `annotations` Collection

**Purpose:** Store human annotations (ground truth for evaluation)

```javascript
{
  _id: ObjectId("..."),
  annotation_id: "ann_poc_001_123",
  
  // Links
  experiment_id: "poc_2026_04_02_001",
  trajectory_id: "toolbench_hf_123_perturbed",
  
  // Annotator info
  annotator: "human_researcher",
  annotation_method: "manual",  // manual | automated | hybrid
  
  // Ground truth measurements
  task_success_degradation: 1.0,  // 0.0 = no degradation, 1.0 = complete failure
  task_success_before: true,
  task_success_after: false,
  
  subsequent_errors: [
    {
      step_number: 3,
      error_type: "incorrect_tool",
      severity: "high",
      description: "Used wrong tool due to planning error"
    },
    {
      step_number: 4,
      error_type: "cascading_failure",
      severity: "critical",
      description: "Task completely derailed"
    }
  ],
  subsequent_error_rate: 0.67,  // errors / remaining_steps
  
  // Derived metrics
  true_criticality_score: 110.0,  // TCS = (TSD × 100) + (SER × 10)
  
  // Quality control
  confidence: "high",  // low | medium | high
  notes: "Clear cascade from early planning error",
  review_status: "approved",  // pending | approved | flagged
  
  // Timestamps
  annotated_at: ISODate("2026-04-02T14:00:00Z"),
  updated_at: ISODate("2026-04-02T14:00:00Z")
}
```

**Indexes:**
- `annotation_id` (unique)
- `experiment_id`
- `trajectory_id`
- `annotator`

---

### 4. `judge_evaluations` Collection

**Purpose:** Store LLM judge ratings

```javascript
{
  _id: ObjectId("..."),
  evaluation_id: "eval_poc_001_claude_123_s1",
  
  // Links
  experiment_id: "poc_2026_04_02_001",
  trajectory_id: "toolbench_hf_123_perturbed",
  
  // Judge info
  judge_model: "claude-3.5-sonnet",
  judge_model_id: "anthropic.claude-3-5-sonnet-20241022-v2:0",
  judge_provider: "aws_bedrock",
  sample_number: 1,  // For multiple samples per trajectory
  
  // Judge ratings
  overall_score: 85.0,  // 0-100 scale
  overall_penalty: 15.0,  // 100 - overall_score (used in CCG)
  
  errors_identified: [
    {
      step_number: 2,
      error_description: "Imprecise goal formulation",
      severity: 4.0,  // 0-10 scale
      judge_reasoning: "The planning step lacks specificity..."
    }
  ],
  
  // Raw judge output
  raw_response: "The trajectory shows...",
  prompt_used: "You are evaluating...",
  
  // API metadata
  api_call: {
    tokens_input: 1234,
    tokens_output: 567,
    latency_ms: 2345,
    cost_usd: 0.025
  },
  
  // Configuration
  temperature: 0.7,
  max_tokens: 2000,
  
  // Timestamps
  evaluated_at: ISODate("2026-04-02T15:00:00Z")
}
```

**Indexes:**
- `evaluation_id` (unique)
- `experiment_id`
- `trajectory_id, judge_model` (compound)
- `judge_model`

---

### 5. `ccg_scores` Collection

**Purpose:** Store computed CCG scores (analysis results)

```javascript
{
  _id: ObjectId("..."),
  ccg_id: "ccg_poc_001_123_claude",
  
  // Links
  experiment_id: "poc_2026_04_02_001",
  trajectory_id: "toolbench_hf_123_perturbed",
  annotation_id: "ann_poc_001_123",
  evaluation_id: "eval_poc_001_claude_123_s1",
  
  // Condition
  perturbation_type: "planning",
  perturbation_position: "early",
  benchmark: "toolbench",
  
  // Judge info
  judge_model: "claude-3.5-sonnet",
  sample_number: 1,
  
  // CCG computation
  true_criticality_score: 110.0,  // From annotation
  judge_penalty_score: 30.0,      // From judge evaluation (100 - overall_score)
  criticality_calibration_gap: -0.73,  // (JPS - TCS) / TCS
  
  // Interpretation
  calibration_status: "under_penalized",  // under_penalized | well_calibrated | over_penalized
  gap_magnitude: "severe",  // negligible | moderate | severe
  
  // Timestamps
  computed_at: ISODate("2026-04-02T16:00:00Z")
}
```

**Indexes:**
- `ccg_id` (unique)
- `experiment_id`
- `perturbation_type, perturbation_position` (compound)
- `judge_model`
- `calibration_status`

---

## Cross-Experiment Queries

### Example 1: Compare CCG across experiments

```javascript
db.ccg_scores.aggregate([
  {
    $match: {
      perturbation_type: "planning",
      perturbation_position: "early"
    }
  },
  {
    $group: {
      _id: {
        experiment_id: "$experiment_id",
        judge_model: "$judge_model"
      },
      mean_ccg: { $avg: "$criticality_calibration_gap" },
      count: { $sum: 1 }
    }
  }
])
```

### Example 2: Find most miscalibrated conditions across all experiments

```javascript
db.ccg_scores.aggregate([
  {
    $group: {
      _id: {
        perturbation_type: "$perturbation_type",
        perturbation_position: "$perturbation_position",
        judge_model: "$judge_model"
      },
      mean_ccg: { $avg: "$criticality_calibration_gap" },
      experiments: { $addToSet: "$experiment_id" },
      total_samples: { $sum: 1 }
    }
  },
  { $sort: { mean_ccg: 1 } },
  { $limit: 10 }
])
```

### Example 3: Track improvements over time

```javascript
db.experiments.aggregate([
  {
    $lookup: {
      from: "ccg_scores",
      localField: "experiment_id",
      foreignField: "experiment_id",
      as: "ccg_scores"
    }
  },
  {
    $project: {
      experiment_id: 1,
      created_at: 1,
      mean_ccg: { $avg: "$ccg_scores.criticality_calibration_gap" }
    }
  },
  { $sort: { created_at: 1 } }
])
```

---

## Scalability Features

### 1. **Experiment Isolation**
- Each experiment has unique `experiment_id`
- Can run multiple experiments in parallel
- Easy to archive/delete old experiments

### 2. **Flexible Schema**
- `metadata` fields allow custom data without schema changes
- `tags` for categorization and filtering
- Optional fields (e.g., `perturbation` only for perturbed trajectories)

### 3. **Efficient Querying**
- Compound indexes for common query patterns
- Denormalized for read-heavy workloads
- Aggregation pipelines for complex analysis

### 4. **Incremental Updates**
- `progress` field tracks completion
- Can resume interrupted experiments
- `updated_at` timestamps for staleness detection

### 5. **Version Control**
- `config` embedded in experiment doc
- Can reproduce experiments exactly
- Compare configuration changes over time

---

## Discussion Questions

1. **Granularity**: Should we split `trajectories` into separate collections for original vs. perturbed?
   - **Pro**: Cleaner schema, faster queries on originals
   - **Con**: More complex joins, harder to track lineage

2. **Embedding vs. Referencing**: Steps are embedded in trajectories. Alternative: separate `steps` collection?
   - **Current**: Fast read, atomic updates, follows document model
   - **Alternative**: Normalize for very large trajectories (100+ steps)

3. **Aggregated Statistics**: Should we pre-compute and store experiment-level statistics?
   - **Pro**: Faster dashboard queries
   - **Con**: Duplicate data, staleness issues
   - **Proposal**: Materialized views or cached aggregations?

4. **Multi-Annotator Support**: Currently assumes single annotator per trajectory. Scale to multiple?
   - Add `annotations` array to handle inter-annotator agreement
   - Add `consensus_annotation` field for final ground truth

5. **Judge Prompt Versioning**: Prompts may evolve. Track versions?
   - Add `prompt_version` field to `judge_evaluations`
   - Store prompt templates in separate `prompts` collection

**What are your thoughts on this schema? Any concerns about scalability or missing requirements?**
