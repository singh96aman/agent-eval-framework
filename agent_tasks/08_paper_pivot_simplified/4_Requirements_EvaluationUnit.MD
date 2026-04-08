# 4. Evaluation Unit

**Status: REQUIREMENTS DEFINED**

The Evaluation Unit is the atomic unit of analysis in this study. It packages a baseline trajectory with its perturbed variant, along with all metadata needed for evaluation by humans, LLM judges, and outcome verification.

---

## Design Principles

1. **Self-contained**: Each evaluation unit contains everything needed to evaluate it (no external lookups required)
2. **Paired**: Always links baseline and perturbed versions for direct comparison
3. **Blinded**: Supports blinded evaluation where evaluators don't know which trajectory has the perturbation
4. **Traceable**: Full provenance from original trajectory through typed representation to perturbation
5. **Single source of truth**: Canonical data lives in one place; derived fields are explicitly marked and regenerated, never hand-edited

---

## 4.1 Role in the Pipeline

```
Section 2: Typed Representation --> Section 3: Controlled Perturbations
                                                    |
                                                    v
                                          Section 4: Evaluation Unit
                                                    |
                            +-----------------------+-----------------------+
                            |                       |                       |
                            v                       v                       v
                    5A. Human Labels       5B. LLM Judge Outputs    5C. Outcome Evidence
```

The Evaluation Unit serves as the **interface layer** between perturbation generation and evaluation collection. It standardizes what data flows to each downstream evaluation method.

---

## 4.2 Evaluation Unit Schema

### ID Hierarchy

Three distinct ID types prevent overloading:

| ID Type | Scope | Example | Purpose |
|---------|-------|---------|---------|
| `source_trajectory_id` | Original trajectory | `gaia_122` | Immutable reference to source |
| `trajectory_variant_id` | Specific instance | `gaia_122::base`, `gaia_122::pert::001` | Unique id for baseline or perturbed version |
| `evaluation_unit_id` | The pair | `eval::gaia_122::001` | Identifies this baseline+perturbed pair |

**Naming convention:**
- Baseline: `{source_trajectory_id}::base`
- Perturbed: `{source_trajectory_id}::pert::{perturbation_index}`
- Evaluation unit: `eval::{source_trajectory_id}::{perturbation_index}`

### Core Object

```json
{
  // === Identity ===
  "evaluation_unit_id": "string (e.g., eval::gaia_122::001)",
  "created_at": "datetime (ISO 8601)",
  
  // === Source Reference ===
  "source_trajectory_id": "string (immutable original id, e.g., gaia_122)",
  "benchmark": "toolbench | gaia | swebench",
  "task_id": "string",
  "task_text": "string",
  
  // === Baseline Trajectory (CANONICAL) ===
  "baseline": {
    "trajectory_variant_id": "string (e.g., gaia_122::base)",
    "trajectory": "TypedTrajectory (full Section 2 object, SOURCE OF TRUTH)"
  },
  
  // === Perturbed Trajectory (CANONICAL) ===
  "perturbed": {
    "trajectory_variant_id": "string (e.g., gaia_122::pert::001)",
    "trajectory": "TypedTrajectory (with perturbation applied)",
    "perturbation_record": "PerturbationRecord (full Section 3 object, SOURCE OF TRUTH)"
  },
  
  // === Derived Cache (DENORMALIZED - regenerate, never hand-edit) ===
  "derived_cache": {
    "_warning": "These fields are derived from canonical objects. Regenerate via build script.",
    "baseline_outcome": "float (from baseline.trajectory.baseline_outcome)",
    "baseline_num_steps": "int (from len(baseline.trajectory.steps))",
    "perturbed_num_steps": "int (from len(perturbed.trajectory.steps))",
    "perturbation_class": "string (from perturbed.perturbation_record.perturbation_class)",
    "perturbation_family": "string (from perturbed.perturbation_record.perturbation_family)",
    "perturbation_type": "string (from perturbed.perturbation_record.perturbation_type)",
    "target_step_canonical_id": "string (from perturbed.perturbation_record)",
    "expected_impact": "int (from perturbed.perturbation_record.expected_impact)",
    "expected_detectability": "int (from perturbed.perturbation_record.expected_detectability)"
  },
  
  // === Evaluation Capabilities (derived from trajectory properties) ===
  "evaluation_capabilities": {
    "has_objective_verifier": "bool",
    "can_replay": "bool",
    "can_regenerate_downstream": "bool",
    "environment_accessible": "bool",
    "ground_truth_available": "bool"
  },
  
  // === Derived: Replay Tier (computed from capabilities) ===
  "replay_tier": "1 | 2 | 3"
}
```

### Source of Truth Rules

| Data | Source of Truth | Derived/Cached In |
|------|-----------------|-------------------|
| Trajectory structure | `baseline.trajectory`, `perturbed.trajectory` | - |
| Perturbation details | `perturbed.perturbation_record` | `derived_cache.*` |
| Baseline outcome | `baseline.trajectory.baseline_outcome` | `derived_cache.baseline_outcome` |
| Step count | `len(trajectory.steps)` | `derived_cache.*_num_steps` |

**Invariant:** `derived_cache` fields must be regenerated on any edit to canonical objects. Validation scripts must verify consistency.

### Field Descriptions

| Field | Purpose |
|-------|---------|
| `evaluation_unit_id` | Unique identifier for this evaluation unit (the pair) |
| `source_trajectory_id` | Immutable reference to the original trajectory |
| `baseline.trajectory_variant_id` | Unique ID for this baseline instance |
| `perturbed.trajectory_variant_id` | Unique ID for this perturbed instance |
| `baseline.trajectory` | **SOURCE OF TRUTH** - the unperturbed typed trajectory |
| `perturbed.trajectory` | **SOURCE OF TRUTH** - the perturbed typed trajectory |
| `perturbed.perturbation_record` | **SOURCE OF TRUTH** - full perturbation metadata |
| `derived_cache` | Denormalized fields for quick filtering (regenerate, never edit) |
| `evaluation_capabilities` | What kinds of evaluation are possible (capability-based) |
| `replay_tier` | Derived from capabilities, not hardcoded by benchmark |

---

## 4.3 Baseline vs Perturbed Trajectories

### Baseline Trajectory

The baseline is the **original typed trajectory** from Section 2:
- No modifications from Section 3
- Serves as the control/reference for comparison
- Has unique `trajectory_variant_id` (e.g., `gaia_122::base`)

### Perturbed Trajectory

The perturbed trajectory contains:
- The typed trajectory with exactly one perturbation applied
- The full `perturbation_record` from Section 3
- Has unique `trajectory_variant_id` (e.g., `gaia_122::pert::001`)
- Shares `source_trajectory_id` with baseline (same original task)

### Pairing Rules

| Rule | Description |
|------|-------------|
| **One perturbation per unit** | Each evaluation unit has exactly one perturbation applied |
| **Same source** | Baseline and perturbed share the same `source_trajectory_id` |
| **Distinct variant IDs** | Each has a unique `trajectory_variant_id` |
| **Stable step identity** | Steps have `canonical_step_id` for alignment (see 4.3.1) |
| **Multiple units per trajectory** | A single baseline may have multiple evaluation units (different perturbations) |

### 4.3.1 Step Identity (Canonical Step IDs)

**Problem:** `step_index` alone becomes unstable for deletion/truncation perturbations. We need stable identity for:
- Localization evaluation
- Propagation tracking
- Step matching between baseline and perturbed
- Perturbation target alignment

**Solution:** Each step has both `canonical_step_id` and `display_step_index`:

```json
{
  "canonical_step_id": "string (stable, e.g., gaia_122::step::2)",
  "display_step_index": "int (position in this variant, may differ from canonical)"
}
```

**Naming convention:**
- Canonical ID: `{source_trajectory_id}::step::{original_position}`
- This ID is **immutable** — even if a step is deleted, downstream steps keep their canonical IDs

**Example: Deletion Perturbation**

Baseline (5 steps):
| canonical_step_id | display_step_index |
|-------------------|-------------------|
| gaia_122::step::1 | 1 |
| gaia_122::step::2 | 2 |
| gaia_122::step::3 | 3 |
| gaia_122::step::4 | 4 |
| gaia_122::step::5 | 5 |

Perturbed (step 2 deleted, 4 steps):
| canonical_step_id | display_step_index |
|-------------------|-------------------|
| gaia_122::step::1 | 1 |
| gaia_122::step::3 | 2 |
| gaia_122::step::4 | 3 |
| gaia_122::step::5 | 4 |

**Benefits:**
- Can always locate the perturbed step via `canonical_step_id`
- Can track which downstream steps are affected
- Human evaluators see clean `display_step_index` (1, 2, 3, 4)
- Analysis scripts use `canonical_step_id` for matching

### Step Schema Update (Section 2 Amendment)

Add to each step object:
```json
{
  "canonical_step_id": "string (stable across variants)",
  "display_step_index": "int (position in this trajectory variant)",
  // ... existing step fields from Section 2
}
```

---

## 4.4 Evaluation Configuration

### Replay Tier Assignment

Each evaluation unit is assigned a replay tier based on what outcome evidence is feasible:

| Tier | Description | Required Capabilities | Outcome Metrics |
|------|-------------|----------------------|-----------------|
| **1** | Full replay | `can_replay` + `environment_accessible` + `has_objective_verifier` | OD, PD, RC (full) |
| **2** | Downstream regeneration | `can_regenerate_downstream` + `has_objective_verifier` | OD (partial), PD |
| **3** | Final-answer grading only | `ground_truth_available` OR `has_objective_verifier` | OD only |

### Capability-Based Tier Assignment

Tier is derived from capabilities, not hardcoded by benchmark:

```python
def assign_replay_tier(evaluation_unit):
    caps = evaluation_unit["evaluation_capabilities"]
    
    # Tier 1: Full replay possible
    if (caps["can_replay"] and 
        caps["environment_accessible"] and 
        caps["has_objective_verifier"]):
        return 1
    
    # Tier 2: Can regenerate downstream from perturbed prefix
    if (caps["can_regenerate_downstream"] and 
        caps["has_objective_verifier"]):
        return 2
    
    # Tier 3: Final-answer grading only
    if caps["ground_truth_available"] or caps["has_objective_verifier"]:
        return 3
    
    # Fallback: no outcome evidence possible
    return None
```

### Capability Definitions

| Capability | Definition |
|------------|------------|
| `has_objective_verifier` | Automated verification exists (test suite, exact match, etc.) |
| `can_replay` | Can re-execute the trajectory from scratch |
| `can_regenerate_downstream` | Can continue generation from a perturbed prefix |
| `environment_accessible` | External APIs/tools are available and deterministic enough |
| `ground_truth_available` | Expected answer or success label exists |

### Benchmark Capability Defaults

Benchmarks provide **default** capability values, but individual trajectories may override:

| Benchmark | Default Capabilities | Typical Tier |
|-----------|---------------------|--------------|
| SWE-bench | `can_replay=true`, `environment_accessible=true`, `has_objective_verifier=true` | 1 |
| GAIA | `can_regenerate_downstream=varies`, `has_objective_verifier=true`, `ground_truth_available=true` | 2-3 |
| ToolBench | `environment_accessible=false`, `has_objective_verifier=partial`, `ground_truth_available=partial` | 3 |

**Key principle:** Benchmark is a prior for capability defaults, but the actual tier comes from the capability flags. This allows:
- SWE-bench tasks with flaky tests to be downgraded
- GAIA tasks with reproducible tool access to be upgraded
- ToolBench tasks with mocked/cached APIs to be upgraded

### Verifier Types

| Verifier Type | Capability Implication |
|---------------|----------------------|
| Test suite (SWE-bench) | `has_objective_verifier=true`, `can_replay=true` |
| Exact-match answer (GAIA) | `has_objective_verifier=true`, `ground_truth_available=true` |
| Task success heuristic | `has_objective_verifier=partial` (may need human confirmation) |
| None | `has_objective_verifier=false` |

---

## 4.5 Blinding Protocol

### Purpose

Blinding prevents evaluators from knowing which trajectory contains the perturbation, reducing bias in:
- Human detectability ratings
- Human impact assessments
- LLM judge evaluations (optional)

### Three-File Architecture

Blinding information is **never** stored in the evaluation unit payload. Instead:

```
data/evaluation_units/
├── canonical/                    # SOURCE OF TRUTH - full evaluation units
│   ├── toolbench_eval_units.json
│   ├── gaia_eval_units.json
│   └── swebench_eval_units.json
├── blinding_key.json             # PRIVATE - mapping A/B to baseline/perturbed
└── views/                        # PUBLIC - evaluator-facing data
    ├── human/
    └── llm_judge/
```

### Blinding Key (Private, Never Shared with Evaluators)

```json
{
  "generated_at": "datetime",
  "global_seed": "int",
  "assignments": [
    {
      "evaluation_unit_id": "eval::gaia_122::001",
      "trajectory_a_variant_id": "gaia_122::pert::001",
      "trajectory_b_variant_id": "gaia_122::base",
      "is_a_baseline": false,
      "is_a_perturbed": true
    }
  ]
}
```

**Key rule:** `is_a_baseline` lives ONLY in `blinding_key.json`, never in the canonical evaluation unit or evaluator views.

### Public Presentation Object (in Evaluator Views)

Evaluator views contain only the shuffled IDs, not the truth labels:

```json
{
  "evaluation_unit_id": "eval::gaia_122::001",
  "trajectory_a_id": "view_A",
  "trajectory_b_id": "view_B"
}
```

### Blinding Rules

1. **Shuffle assignment**: Use `hash(evaluation_unit_id + global_seed)` to assign baseline/perturbed to A/B
2. **Consistent within unit**: Same seed produces same assignment for all evaluators
3. **Balanced across dataset**: ~50% of units have baseline as A (verified post-generation)
4. **Strict separation**: Blinding key never included in evaluator-facing files
5. **Unblinding**: Join evaluator responses with blinding key only during analysis

### Evaluation Modes

| Mode | Use Case | What Evaluator Sees |
|------|----------|---------------------|
| **Blinded pair** | Human detectability, some LLM judges | Trajectory A and B without labels |
| **Single-trajectory** | Most LLM judges, outcome verification | One trajectory (usually perturbed) |
| **Labeled pair** | Detailed human annotation, post-analysis | Both with baseline/perturbed labels |

---

## 4.6 Views for Different Evaluators

Each evaluation method receives a tailored view of the evaluation unit. Views are **annotation-mode-specific** to prevent bias.

### 4.6.1 Human Evaluator Views (5A)

Human evaluation requires different views depending on the annotation task:

#### Mode A: Detectability / Localization View

**Purpose:** Can the human detect that an error exists and locate it?
**Answer visibility:** NO expected answer (prevents "looks right" bias)

```json
{
  "view_type": "detectability",
  "evaluation_unit_id": "string",
  "task_text": "string",
  "trajectory_a": {
    "trajectory_id": "view_A",
    "steps": ["SimplifiedStep[]"],
    "num_steps": "int"
  },
  "trajectory_b": {
    "trajectory_id": "view_B",
    "steps": ["SimplifiedStep[]"],
    "num_steps": "int"
  },
  "instructions": "Compare these trajectories. Does one contain an error? If so, which trajectory and which step?",
  "rating_schema": {
    "error_detected": "bool",
    "error_trajectory": "A | B | neither | both",
    "error_step_id": "string | null",
    "confidence": "1-5"
  }
}
```

#### Mode B: Consequence / Correctness View

**Purpose:** How severe is the error's impact on task success?
**Answer visibility:** Expected answer MAY be shown (needed to assess correctness)

```json
{
  "view_type": "consequence",
  "evaluation_unit_id": "string",
  "task_text": "string",
  "expected_answer": "string | null (shown only if needed)",
  "trajectory_a": {
    "trajectory_id": "view_A",
    "steps": ["SimplifiedStep[]"],
    "final_answer": "string | null"
  },
  "trajectory_b": {
    "trajectory_id": "view_B",
    "steps": ["SimplifiedStep[]"],
    "final_answer": "string | null"
  },
  "instructions": "Given the expected answer, assess the impact of any errors on task success.",
  "rating_schema": {
    "impact_a": "0-3 (none/minor/moderate/critical)",
    "impact_b": "0-3",
    "correctness_a": "correct | partially_correct | incorrect",
    "correctness_b": "correct | partially_correct | incorrect"
  }
}
```

#### Mode C: Pair Preference View

**Purpose:** Which trajectory is better overall?
**Answer visibility:** Configurable per benchmark

```json
{
  "view_type": "preference",
  "evaluation_unit_id": "string",
  "task_text": "string",
  "expected_answer": "string | null (benchmark-dependent)",
  "trajectory_a": {"steps": ["SimplifiedStep[]"]},
  "trajectory_b": {"steps": ["SimplifiedStep[]"]},
  "instructions": "Which trajectory better accomplishes the task?",
  "rating_schema": {
    "preference": "A | B | tie",
    "reason": "string"
  }
}
```

**SimplifiedStep** (human-readable, common to all modes):
```json
{
  "display_step_index": "int (position in this variant)",
  "canonical_step_id": "string (for analysis matching)",
  "step_role": "string",
  "description": "string (human-readable summary)",
  "tool_name": "string | null",
  "tool_arguments_summary": "string | null",
  "observation_summary": "string | null"
}
```

### 4.6.2 LLM Judge Views (5B)

LLM judges also require mode-specific views for scientifically valid comparison:

#### Judge Mode: Single Trajectory

**Use case:** Standard trajectory evaluation (most common)

```json
{
  "judge_mode": "single_trajectory",
  "evaluation_unit_id": "string",
  "task_text": "string",
  "trajectory": "TypedTrajectory (perturbed, no baseline shown)",
  "judge_config": {
    "judge_model": "string",
    "prompt_template_id": "string",
    "expected_outputs": ["overall_score", "error_detected", "predicted_error_step", "predicted_impact"]
  }
}
```

#### Judge Mode: Blinded Pair

**Use case:** Comparative evaluation without labels

```json
{
  "judge_mode": "blinded_pair",
  "evaluation_unit_id": "string",
  "task_text": "string",
  "trajectory_a": "TypedTrajectory",
  "trajectory_b": "TypedTrajectory",
  "judge_config": {
    "judge_model": "string",
    "prompt_template_id": "string",
    "expected_outputs": ["preference", "error_trajectory", "error_step"]
  }
}
```

#### Judge Mode: Labeled Pair

**Use case:** Explicit baseline vs perturbed comparison (for calibration studies)

```json
{
  "judge_mode": "labeled_pair",
  "evaluation_unit_id": "string",
  "task_text": "string",
  "baseline_trajectory": "TypedTrajectory",
  "perturbed_trajectory": "TypedTrajectory",
  "perturbation_summary": {
    "target_step_canonical_id": "string",
    "perturbation_type": "string"
  },
  "judge_config": {
    "judge_model": "string",
    "prompt_template_id": "string",
    "expected_outputs": ["impact_estimate", "detectability_estimate"]
  }
}
```

**Critical rule:** Analysis must never mix outputs from different `judge_mode` values. They are scientifically incomparable.

### 4.6.3 Outcome Verification View (5C)

Outcome verification always sees both trajectories (no blinding needed):

```json
{
  "evaluation_unit_id": "string",
  "benchmark": "string",
  "task_id": "string",
  "source_trajectory_id": "string",
  "replay_tier": "int",
  "baseline": {
    "trajectory_variant_id": "string",
    "trajectory": "TypedTrajectory"
  },
  "perturbed": {
    "trajectory_variant_id": "string",
    "trajectory": "TypedTrajectory",
    "perturbation_record": "PerturbationRecord"
  },
  "verification_config": {
    "verifier_type": "test_suite | exact_match | heuristic",
    "verifier_params": "object",
    "expected_answer": "string | null"
  }
}
```

---

## 4.7 Data Location

```
data/
├── typed/                              # Input (from Section 2)
├── perturbed/                          # Input (from Section 3)
└── evaluation_units/
    ├── canonical/                      # SOURCE OF TRUTH - full evaluation units
    │   ├── toolbench_eval_units.json
    │   ├── gaia_eval_units.json
    │   ├── swebench_eval_units.json
    │   └── evaluation_unit_index.json
    │
    ├── private/                        # NEVER SHARE with evaluators
    │   └── blinding_key.json
    │
    └── views/                          # PUBLIC - evaluator-facing data
        ├── human/
        │   ├── detectability/          # Mode A views
        │   ├── consequence/            # Mode B views
        │   └── preference/             # Mode C views
        └── llm_judge/
            ├── single_trajectory/      # Single-trajectory mode
            ├── blinded_pair/           # Blinded pair mode
            └── labeled_pair/           # Labeled pair mode (calibration)
```

### Evaluation Unit Index Schema

```json
{
  "total_units": 1700,
  "by_benchmark": {
    "toolbench": 1000,
    "gaia": 350,
    "swebench": 350
  },
  "by_perturbation_class": {
    "placebo": 340,
    "fine_grained": 850,
    "coarse_grained": 510
  },
  "by_perturbation_family": {
    "data_reference": 400,
    "parameter": 350,
    "tool_selection": 200,
    "structural": 400,
    "terminal_flag": 110
  },
  "by_replay_tier": {
    "1": 350,
    "2": 150,
    "3": 1200
  },
  "by_expected_impact": {
    "0": 400,
    "1": 450,
    "2": 500,
    "3": 350
  },
  "by_expected_detectability": {
    "0": 500,
    "1": 700,
    "2": 500
  },
  "units": [
    {
      "evaluation_unit_id": "string",
      "source_trajectory_id": "string",
      "baseline_variant_id": "string",
      "perturbed_variant_id": "string",
      "benchmark": "string",
      "perturbation_class": "string",
      "perturbation_family": "string",
      "expected_impact": "int",
      "expected_detectability": "int",
      "replay_tier": "int",
      "file": "string"
    }
  ]
}
```

### Blinding Key (Private, Never Share)

Located at `evaluation_units/private/blinding_key.json`:

```json
{
  "generated_at": "datetime",
  "global_seed": "int",
  "balance_check": {
    "a_is_baseline_count": 850,
    "a_is_perturbed_count": 850,
    "balance_ratio": 0.50
  },
  "assignments": [
    {
      "evaluation_unit_id": "eval::gaia_122::001",
      "trajectory_a_variant_id": "gaia_122::pert::001",
      "trajectory_b_variant_id": "gaia_122::base",
      "is_a_baseline": false,
      "is_a_perturbed": true
    }
  ]
}
```

**Access control:**
- `canonical/` — accessible to researchers and analysis scripts
- `private/` — accessible only to unblinding script and final analysis
- `views/` — accessible to evaluators (human annotators, LLM judge pipeline)

---

## 4.8 Sampling for Evaluation

Not all 1700 evaluation units need every type of evaluation. Sampling must ensure coverage across multiple dimensions.

### Human Evaluation Sample

**Total target:** ~350 units (~20% of dataset)

#### Primary Stratification (by experimental condition)

| Stratum | Count | Purpose |
|---------|-------|---------|
| High-impact (expected_impact=3) | 100 | Critical error detection |
| Placebo | 75 | Calibration baseline |
| Fine-grained subtle (detectability=0-1) | 100 | Subtle error detection |
| Coarse-grained | 75 | Structural error detection |

#### Required Coverage Minimums (cross-cutting)

| Dimension | Minimum per Cell | Rationale |
|-----------|------------------|-----------|
| **Benchmark** | ≥50 each (ToolBench, GAIA, SWE-bench) | Prevent benchmark bias |
| **Perturbation class** | ≥50 per class | Statistical power for class-level claims |
| **Perturbation family** | ≥20 per family | Coverage of all error types |
| **Detectability band** | ≥75 per band (0, 1, 2) | Test detection across difficulty |

#### Sampling Algorithm

```python
def sample_human_evaluation_set(all_units, target=350):
    # 1. Stratified sample by primary strata
    sample = stratified_sample(all_units, primary_strata, target)
    
    # 2. Check coverage minimums
    for dimension in [benchmark, perturbation_class, perturbation_family, detectability]:
        for cell in dimension.cells:
            if count(sample, cell) < minimum[dimension]:
                # Swap in additional units from underrepresented cell
                sample = rebalance(sample, cell, minimum[dimension])
    
    # 3. Verify no stratum is over-represented
    assert max_ratio(sample, any_cell) <= 0.35  # No single cell > 35%
    
    return sample
```

#### Expected Distribution (approximate)

| Benchmark | Count | % |
|-----------|-------|---|
| ToolBench | ~140 | 40% |
| GAIA | ~105 | 30% |
| SWE-bench | ~105 | 30% |

| Perturbation Family | Count | % |
|---------------------|-------|---|
| data_reference | ~90 | 26% |
| parameter | ~75 | 21% |
| tool_selection | ~50 | 14% |
| structural | ~90 | 26% |
| terminal_flag | ~45 | 13% |

### LLM Judge Evaluation

| Scope | Count | Purpose |
|-------|-------|---------|
| All units (single_trajectory mode) | 1700 | Full judge evaluation |
| Blinded pair mode | 500 | Comparative evaluation |
| High-temperature repeats | 500 | Variance estimation (n=5 per unit) |
| Multi-judge comparison | 500 | Inter-judge agreement |

**Mode coverage:** Each judge must be run in at least one mode per benchmark to enable cross-benchmark comparison.

### Outcome Evidence Collection

| Tier | Count | Composition |
|------|-------|-------------|
| Tier 1 (full replay) | ~350 | All SWE-bench units |
| Tier 2 (regeneration) | ~150 | GAIA units with regeneration capability |
| Tier 3 (grading only) | ~1200 | Remaining GAIA + all ToolBench |

**Note:** Tier assignment is capability-based per unit, not fixed by benchmark (see 4.4).

---

## 4.9 Validation Rules

Each evaluation unit must pass these validation checks:

### ID Validation

| Check | Requirement |
|-------|-------------|
| **evaluation_unit_id format** | Matches pattern `eval::{source}::{index}` |
| **baseline variant_id format** | Matches pattern `{source}::base` |
| **perturbed variant_id format** | Matches pattern `{source}::pert::{index}` |
| **source_trajectory_id consistency** | Extractable from both variant IDs |
| **No ID collisions** | All IDs unique across dataset |

### Structural Validation

| Check | Requirement |
|-------|-------------|
| **Baseline exists** | `baseline.trajectory` is a valid TypedTrajectory |
| **Perturbed exists** | `perturbed.trajectory` is a valid TypedTrajectory |
| **Same source** | Both trajectories derive from same `source_trajectory_id` |
| **Perturbation applied** | `perturbed.trajectory` differs from `baseline.trajectory` |
| **Perturbation record complete** | `perturbed.perturbation_record` has all required fields |

### Step Identity Validation

| Check | Requirement |
|-------|-------------|
| **canonical_step_id present** | All steps have `canonical_step_id` |
| **canonical_step_id format** | Matches pattern `{source}::step::{position}` |
| **canonical_step_id stable** | Same step has same `canonical_step_id` in baseline and perturbed |
| **display_step_index sequential** | `display_step_index` is 1, 2, 3, ... with no gaps |
| **perturbation target valid** | `target_step_canonical_id` exists in baseline |

### Derived Cache Validation

| Check | Requirement |
|-------|-------------|
| **Cache matches source** | `derived_cache.baseline_outcome == baseline.trajectory.baseline_outcome` |
| **Cache matches source** | `derived_cache.perturbation_class == perturbed.perturbation_record.perturbation_class` |
| **All derived fields present** | No null values in `derived_cache` for required fields |
| **Regeneration test** | `rebuild_derived_cache(unit) == unit.derived_cache` |

### Capability & Tier Validation

| Check | Requirement |
|-------|-------------|
| **Capabilities complete** | All capability flags have boolean values |
| **Tier derivable** | `replay_tier` consistent with `evaluation_capabilities` |
| **Tier in range** | `replay_tier` is 1, 2, or 3 (or null if no evaluation possible) |

### Consistency Validation

| Check | Requirement |
|-------|-------------|
| **Metadata match** | `benchmark`, `task_id`, `task_text` consistent across objects |
| **Step alignment** | For non-deletion perturbations, baseline and perturbed have matching `canonical_step_id` sets (minus deleted) |

### Blinding Validation (on blinding_key.json)

| Check | Requirement |
|-------|-------------|
| **All units covered** | Every `evaluation_unit_id` appears in blinding key |
| **No leakage** | `is_a_baseline` does NOT appear in canonical units or views |
| **Balanced assignment** | 45% ≤ `a_is_baseline_count / total` ≤ 55% |
| **Variant IDs match** | `trajectory_a_variant_id` and `trajectory_b_variant_id` match unit's variants |

---

## 4.10 Generation Pipeline

### Phase 1: Unit Assembly

1. Load typed trajectories (Section 2)
2. Load perturbed trajectories (Section 3)
3. For each perturbation record:
   - Load baseline trajectory
   - Load perturbed trajectory
   - Create evaluation unit with paired data
   - Assign evaluation configuration

### Phase 2: Blinding Assignment

1. Generate global seed for reproducibility
2. For each evaluation unit:
   - Use `hash(evaluation_unit_id + global_seed)` to determine A/B assignment
   - Create blinded presentation object
3. Verify ~50% balance

### Phase 3: Tier Assignment

1. For each evaluation unit:
   - Check benchmark and verifier availability
   - Assign replay tier
   - Record in `evaluation_config`

### Phase 4: Index Generation

1. Aggregate statistics by benchmark, class, tier, impact
2. Create unit summaries for quick lookup
3. Write index file

### Phase 5: View Generation (Optional)

1. Generate human evaluator views (stripped to essentials)
2. Generate LLM judge views (templated prompts)
3. Generate outcome verification views (with configs)

---

## 4.11 Benchmark-Specific Considerations

These are **default** capability values. Individual trajectories may override based on their specific properties.

### ToolBench

| Aspect | Default | Override Conditions |
|--------|---------|---------------------|
| `can_replay` | false | true if API responses are cached/mocked |
| `environment_accessible` | false | true if using stable/local APIs |
| `has_objective_verifier` | partial | true if task has clear success criteria |
| `ground_truth_available` | partial | depends on task |
| Typical replay tier | 3 | could be 2 if regeneration is feasible |
| Human evaluation focus | API selection errors, parameter choices | |

### GAIA

| Aspect | Default | Override Conditions |
|--------|---------|---------------------|
| `can_replay` | false | true if all tools are deterministic |
| `can_regenerate_downstream` | varies | true if model access available |
| `has_objective_verifier` | true | exact-match answer grading |
| `ground_truth_available` | true | always has expected answer |
| Typical replay tier | 2-3 | 2 if downstream regeneration feasible |
| Human evaluation focus | Extraction errors, numeric computation | |

### SWE-bench

| Aspect | Default | Override Conditions |
|--------|---------|---------------------|
| `can_replay` | true | false if environment setup fails |
| `environment_accessible` | true | false if dependencies unavailable |
| `has_objective_verifier` | true | test suite available |
| `ground_truth_available` | true | test pass/fail is ground truth |
| Typical replay tier | 1 | 3 if test suite is flaky |
| Human evaluation focus | Patch correctness, localization accuracy | |

---

## 4.12 Failure Modes to Avoid

| Failure Mode | Description | Mitigation |
|--------------|-------------|------------|
| **Broken pairing** | Baseline and perturbed don't match | Validation checks on `trajectory_id` |
| **Missing metadata** | Perturbation summary incomplete | Schema validation |
| **Biased blinding** | Non-random A/B assignment | Use deterministic hash + verify balance |
| **View leakage** | Human view contains perturbation labels | Strip perturbation fields from human view |
| **Tier misassignment** | Wrong replay tier for benchmark | Benchmark-specific assignment logic |
| **Orphan perturbations** | Perturbations without evaluation units | Cross-check against perturbation index |

---

## 4.13 Implementation Plan

### Step 1: Module Structure (src/evaluation/)

```
src/evaluation/
├── __init__.py
├── ids.py                  # ID generation and parsing (variant IDs, canonical step IDs)
├── unit_assembler.py       # Create evaluation units from typed + perturbed
├── derived_cache.py        # Build and rebuild derived_cache fields
├── capabilities.py         # Compute evaluation_capabilities from trajectory properties
├── tier_assignment.py      # Capability-based replay tier assignment
├── blinding.py             # Blinding assignment (writes to private/)
├── validators.py           # All validation checks (IDs, structure, cache, blinding)
├── views/
│   ├── __init__.py
│   ├── human.py            # Generate human views (detectability, consequence, preference)
│   ├── llm_judge.py        # Generate judge views (single, blinded_pair, labeled_pair)
│   └── outcome.py          # Generate outcome verification views
├── sampling.py             # Human evaluation sampling with coverage constraints
└── index.py                # Index generation with multi-dimensional statistics
```

### Step 2: Assembly Script

```bash
python -m src.evaluation.assemble \
  --typed-dir data/typed/ \
  --perturbed-dir data/perturbed/ \
  --output-dir data/evaluation_units/canonical/ \
  --blinding-seed 42
```

Outputs:
- `data/evaluation_units/canonical/{benchmark}_eval_units.json`
- `data/evaluation_units/private/blinding_key.json`

### Step 3: View Generation Script

```bash
python -m src.evaluation.generate_views \
  --input data/evaluation_units/canonical/ \
  --output-dir data/evaluation_units/views/ \
  --human-modes detectability,consequence,preference \
  --judge-modes single_trajectory,blinded_pair,labeled_pair
```

Outputs:
- `data/evaluation_units/views/human/{mode}/*.json`
- `data/evaluation_units/views/llm_judge/{mode}/*.json`

### Step 4: Validation Script

```bash
python -m src.evaluation.validate \
  --canonical-dir data/evaluation_units/canonical/ \
  --blinding-key data/evaluation_units/private/blinding_key.json \
  --views-dir data/evaluation_units/views/ \
  --output data/evaluation_units/validation_report.json
```

Validates:
- ID format and uniqueness
- Structural integrity
- Step identity stability
- Derived cache consistency
- Blinding separation and balance
- View completeness

### Step 5: Sampling Script

```bash
python -m src.evaluation.sample_human \
  --input data/evaluation_units/canonical/ \
  --output data/evaluation_units/human_sample.json \
  --target-size 350 \
  --coverage-config configs/human_coverage_requirements.yaml
```

Outputs sample with verified coverage across benchmark, class, family, and detectability.

---

## 4.14 Dependencies

| Dependency | Section | What's Needed |
|------------|---------|---------------|
| Typed Representation (Section 2) | Input | Typed trajectories as baselines |
| Controlled Perturbations (Section 3) | Input | Perturbed trajectories + perturbation records |
| Human Labels (Section 5A) | Output | Evaluation units as input for human annotation |
| LLM Judge Outputs (Section 5B) | Output | Evaluation units as input for judge evaluation |
| Outcome Evidence (Section 5C) | Output | Evaluation units as input for replay/verification |

---

## 4.15 Success Criteria

Section 4 is complete when:

### Schema & IDs
1. [ ] Evaluation unit schema finalized with three-tier ID hierarchy (`source_trajectory_id`, `trajectory_variant_id`, `evaluation_unit_id`)
2. [ ] All IDs follow naming conventions (e.g., `gaia_122::base`, `gaia_122::pert::001`, `eval::gaia_122::001`)
3. [ ] All steps have `canonical_step_id` and `display_step_index`
4. [ ] `derived_cache` clearly marked as regeneratable, not hand-editable

### Data Integrity
5. [ ] All ~1700 perturbations have corresponding evaluation units
6. [ ] All evaluation units pass ID validation
7. [ ] All evaluation units pass structural validation
8. [ ] All evaluation units pass step identity validation
9. [ ] All evaluation units pass derived cache validation
10. [ ] Validation report shows 0 errors
11. [ ] Cross-check confirms no orphan perturbations

### Capabilities & Tiers
12. [ ] All units have complete `evaluation_capabilities` flags
13. [ ] `replay_tier` derived from capabilities (not hardcoded by benchmark)
14. [ ] Tier distribution reasonable:
    - SWE-bench: mostly Tier 1
    - GAIA: mix of Tier 2/3 based on capabilities
    - ToolBench: mostly Tier 3

### Blinding
15. [ ] `blinding_key.json` stored in `private/` directory
16. [ ] No `is_a_baseline` field in canonical units or views
17. [ ] Blinding balance: 45% ≤ A-is-baseline ≤ 55%

### Views
18. [ ] Human views generated for all three modes (detectability, consequence, preference)
19. [ ] Detectability views have NO `expected_answer`
20. [ ] LLM judge views generated for all three modes (single_trajectory, blinded_pair, labeled_pair)
21. [ ] Outcome verification views generated with verification configs

### Index & Statistics
22. [ ] Evaluation unit index created with summary statistics by:
    - benchmark
    - perturbation_class
    - perturbation_family
    - replay_tier
    - expected_impact
    - expected_detectability
23. [ ] Human sampling plan achieves coverage minimums:
    - ≥50 units per benchmark
    - ≥50 units per perturbation class
    - ≥20 units per perturbation family
    - ≥75 units per detectability band

---

## 4.16 Example: Complete Evaluation Unit

### GAIA Example (gaia_122 with wrong extracted value)

#### Canonical Evaluation Unit (in `canonical/gaia_eval_units.json`)

```json
{
  "evaluation_unit_id": "eval::gaia_122::001",
  "created_at": "2026-04-07T12:00:00Z",
  
  "source_trajectory_id": "gaia_122",
  "benchmark": "gaia",
  "task_id": "122",
  "task_text": "How many more blocks in BERT base encoder than the encoder from Attention is All You Need?",
  
  "baseline": {
    "trajectory_variant_id": "gaia_122::base",
    "trajectory": {
      "trajectory_id": "gaia_122",
      "benchmark": "gaia",
      "task_id": "122",
      "task_text": "How many more blocks...",
      "baseline_outcome": 1.0,
      "num_steps": 5,
      "ground_truth_available": true,
      "environment_type": "qa",
      "steps": [
        {
          "canonical_step_id": "gaia_122::step::1",
          "display_step_index": 1,
          "raw_text": "Search the internet for \"blocks in bert base\"",
          "step_role": "tool_call",
          "tool_name": "web_search",
          "tool_arguments": {"query": "blocks in bert base"}
        },
        {
          "canonical_step_id": "gaia_122::step::2",
          "display_step_index": 2,
          "raw_text": "Examine the search results page to locate the answer (12)",
          "step_role": "extraction",
          "extracted_value": 12,
          "value_type": "integer",
          "source_step": 1
        }
      ]
    }
  },
  
  "perturbed": {
    "trajectory_variant_id": "gaia_122::pert::001",
    "trajectory": {
      "trajectory_id": "gaia_122",
      "num_steps": 5,
      "steps": [
        {
          "canonical_step_id": "gaia_122::step::1",
          "display_step_index": 1,
          "raw_text": "Search the internet for \"blocks in bert base\"",
          "step_role": "tool_call",
          "tool_name": "web_search",
          "tool_arguments": {"query": "blocks in bert base"}
        },
        {
          "canonical_step_id": "gaia_122::step::2",
          "display_step_index": 2,
          "raw_text": "Examine the search results page to locate the answer (10)",
          "step_role": "extraction",
          "extracted_value": 10,
          "value_type": "integer",
          "source_step": 1
        }
      ]
    },
    "perturbation_record": {
      "perturbation_id": "pert::gaia_122::001",
      "perturbation_class": "fine_grained",
      "perturbation_family": "data_reference",
      "perturbation_type": "wrong_value",
      "target_step_canonical_id": "gaia_122::step::2",
      "target_slot": "extracted_value",
      "original_value": 12,
      "perturbed_value": 10,
      "expected_impact": 3,
      "expected_detectability": 0
    }
  },
  
  "derived_cache": {
    "_warning": "Derived from canonical objects. Regenerate, never hand-edit.",
    "baseline_outcome": 1.0,
    "baseline_num_steps": 5,
    "perturbed_num_steps": 5,
    "perturbation_class": "fine_grained",
    "perturbation_family": "data_reference",
    "perturbation_type": "wrong_value",
    "target_step_canonical_id": "gaia_122::step::2",
    "expected_impact": 3,
    "expected_detectability": 0
  },
  
  "evaluation_capabilities": {
    "has_objective_verifier": true,
    "can_replay": false,
    "can_regenerate_downstream": false,
    "environment_accessible": false,
    "ground_truth_available": true
  },
  
  "replay_tier": 3
}
```

**Note:** No `blinded_presentation` in canonical object. Blinding lives only in `private/blinding_key.json`.

#### Blinding Key Entry (in `private/blinding_key.json`)

```json
{
  "evaluation_unit_id": "eval::gaia_122::001",
  "trajectory_a_variant_id": "gaia_122::pert::001",
  "trajectory_b_variant_id": "gaia_122::base",
  "is_a_baseline": false,
  "is_a_perturbed": true
}
```

#### Human Evaluator View — Mode A: Detectability (in `views/human/detectability/`)

```json
{
  "view_type": "detectability",
  "evaluation_unit_id": "eval::gaia_122::001",
  "task_text": "How many more blocks in BERT base encoder than the encoder from Attention is All You Need?",
  
  "trajectory_a": {
    "trajectory_id": "view_A",
    "steps": [
      {"display_step_index": 1, "canonical_step_id": "gaia_122::step::1", "step_role": "tool_call", "description": "Search for 'blocks in bert base'"},
      {"display_step_index": 2, "canonical_step_id": "gaia_122::step::2", "step_role": "extraction", "description": "Extract value: 10"},
      {"display_step_index": 3, "canonical_step_id": "gaia_122::step::3", "step_role": "tool_call", "description": "Search for 'attention is all you need layers'"},
      {"display_step_index": 4, "canonical_step_id": "gaia_122::step::4", "step_role": "extraction", "description": "Extract value: 6"},
      {"display_step_index": 5, "canonical_step_id": "gaia_122::step::5", "step_role": "reasoning", "description": "Calculate: 10 - 6 = 4"}
    ],
    "num_steps": 5
  },
  
  "trajectory_b": {
    "trajectory_id": "view_B",
    "steps": [
      {"display_step_index": 1, "canonical_step_id": "gaia_122::step::1", "step_role": "tool_call", "description": "Search for 'blocks in bert base'"},
      {"display_step_index": 2, "canonical_step_id": "gaia_122::step::2", "step_role": "extraction", "description": "Extract value: 12"},
      {"display_step_index": 3, "canonical_step_id": "gaia_122::step::3", "step_role": "tool_call", "description": "Search for 'attention is all you need layers'"},
      {"display_step_index": 4, "canonical_step_id": "gaia_122::step::4", "step_role": "extraction", "description": "Extract value: 6"},
      {"display_step_index": 5, "canonical_step_id": "gaia_122::step::5", "step_role": "reasoning", "description": "Calculate: 12 - 6 = 6"}
    ],
    "num_steps": 5
  },
  
  "instructions": "Compare these two trajectories. Does one contain an error? If so, identify which trajectory and which step.",
  "rating_schema": {
    "error_detected": "bool",
    "error_trajectory": "A | B | neither | both",
    "error_step_id": "string (canonical_step_id) | null",
    "confidence": "1-5"
  }
}
```

**Key points:**
- No `expected_answer` in detectability view (prevents bias)
- No `is_a_baseline` anywhere in view (lives only in blinding key)
- `canonical_step_id` included for analysis matching after unblinding
- Human sees generic "view_A" / "view_B" labels

---

## 4.17 Section 2 Amendment Required

This section introduces `canonical_step_id` and `display_step_index` as required step fields. Section 2 (Typed Representation) should be updated to include:

### Addition to Step Object — Core Fields

```json
{
  // === Identity (NEW) ===
  "canonical_step_id": "string (stable across variants, e.g., gaia_122::step::2)",
  "display_step_index": "int (position in this trajectory variant)",
  
  // === Existing fields (renamed) ===
  "step_index": "DEPRECATED - use display_step_index",
  // ... rest of Section 2 step schema
}
```

### Rationale

- `canonical_step_id` enables stable step matching across baseline/perturbed variants
- `display_step_index` replaces `step_index` for position-based display
- Required for deletion perturbations where step indices shift
- Enables precise localization evaluation and propagation tracking

### Migration

When generating typed trajectories (Section 2 output):
1. Assign `canonical_step_id` as `{trajectory_id}::step::{original_1_indexed_position}`
2. Set `display_step_index` equal to original position
3. Optionally keep `step_index` as alias for backwards compatibility
