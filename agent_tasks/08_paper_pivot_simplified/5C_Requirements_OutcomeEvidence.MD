# 5C. Outcome Evidence

**Status: REQUIREMENTS DEFINED**

Outcome Evidence provides objective ground truth for the actual impact of perturbations. By replaying, regenerating, or grading trajectories, we measure the real-world consequence of errors — independent of human or judge perception.

---

## Design Principles

1. **Tiered**: Different benchmarks support different evidence levels (replay, regeneration, grading)
2. **Objective**: Metrics derived from automated verifiers, not subjective assessment
3. **Paired**: Always compare baseline outcome to perturbed outcome
4. **Capability-based**: Tier assignment follows trajectory capabilities, not benchmark defaults
5. **Traceable**: Every outcome links to evaluation_unit_id with full provenance

---

## 5C.1 Role in the Pipeline

```
Section 4: Evaluation Unit
         |
         v
    Outcome Verification View (from 4.6.3)
         |
         v
    5C. Outcome Evidence
         |
    +----+----+
    |         |
    v         v
6B. Consequentiality   6C. Main Claim
    Calibration            Validation
```

Outcome Evidence provides:
- Ground truth impact for calibration metrics (6B)
- Objective success/failure for claim validation (6C)
- Baseline for comparing judge predictions against reality

---

## 5C.2 Replay Tiers

### Tier Definitions (from Section 4)

| Tier | Name | Description | Evidence Available |
|------|------|-------------|-------------------|
| **1** | Full Replay | Re-execute trajectory from scratch | OD, PD, RC (full) |
| **2** | Downstream Regeneration | Continue from perturbed prefix | OD (partial), PD |
| **3** | Final-Answer Grading | Grade final output only | OD only |

### Tier Assignment by Capability

Tiers are derived from `evaluation_capabilities` (Section 4), not hardcoded by benchmark:

```python
def assign_outcome_tier(capabilities):
    if (capabilities["can_replay"] and 
        capabilities["environment_accessible"] and 
        capabilities["has_objective_verifier"]):
        return 1  # Full replay
    
    if (capabilities["can_regenerate_downstream"] and 
        capabilities["has_objective_verifier"]):
        return 2  # Downstream regeneration
    
    if (capabilities["ground_truth_available"] or 
        capabilities["has_objective_verifier"]):
        return 3  # Final-answer grading
    
    return None  # No outcome evidence possible
```

### Expected Tier Distribution

| Benchmark | Typical Tier | Count | Notes |
|-----------|--------------|-------|-------|
| SWE-bench | 1 | ~350 | Test suite available |
| GAIA | 2-3 | ~350 | Exact-match grading, some regeneration |
| ToolBench | 3 | ~1000 | Final answer grading only |

---

## 5C.3 Core Outcome Metrics

### Outcome Degradation (OD)

**Definition:** The difference in task success between baseline and perturbed trajectory.

```
OD = Score(baseline) - Score(perturbed)
```

**Score types by benchmark:**
- SWE-bench: `test_pass_rate` (0-1)
- GAIA: `exact_match` (0 or 1)
- ToolBench: `task_success_heuristic` (0-1)

**Interpretation:**
| OD Value | Interpretation |
|----------|---------------|
| OD = 0 | No impact (perturbation had no effect) |
| 0 < OD ≤ 0.25 | Minor impact |
| 0.25 < OD ≤ 0.5 | Moderate impact |
| OD > 0.5 | Critical impact |
| OD < 0 | Perturbed is better (unexpected, flag for review) |

### Propagation Depth (PD) — Tier 1-2 Only

**Definition:** Number of downstream steps whose outputs differ between baseline and perturbed trajectories.

```
PD = count(steps where step_output(baseline) ≠ step_output(perturbed))
```

**Measurement:**
- Compare step-by-step outputs after the perturbation point
- Count steps with semantically different outputs
- Requires regeneration or replay

### Recovery Cost (RC) — Tier 1 Only

**Definition:** Additional resources required to recover from the perturbation.

```
RC = Δsteps + Δtool_calls + Δretries
```

**Components:**
- `Δsteps`: Extra steps needed to complete task
- `Δtool_calls`: Additional tool invocations
- `Δretries`: Retry attempts due to errors

---

## 5C.4 Outcome Evidence Schema

### Outcome Record Object

```json
{
  // === Identity ===
  "outcome_id": "string (uuid)",
  "evaluation_unit_id": "string",
  "created_at": "datetime (ISO 8601)",
  
  // === Tier Information ===
  "replay_tier": "int (1, 2, or 3)",
  "evidence_method": "full_replay | downstream_regeneration | final_answer_grading",
  
  // === Baseline Outcome ===
  "baseline": {
    "trajectory_variant_id": "string",
    "outcome_score": "float (0-1)",
    "outcome_binary": "bool (pass/fail)",
    "verifier_output": "object (verifier-specific details)"
  },
  
  // === Perturbed Outcome ===
  "perturbed": {
    "trajectory_variant_id": "string",
    "outcome_score": "float (0-1)",
    "outcome_binary": "bool (pass/fail)",
    "verifier_output": "object (verifier-specific details)"
  },
  
  // === Core Metrics ===
  "metrics": {
    "outcome_degradation": "float (OD = baseline - perturbed)",
    "outcome_degradation_binary": "int (-1, 0, or 1)",
    "propagation_depth": "int | null (Tier 1-2 only)",
    "recovery_cost": "object | null (Tier 1 only)"
  },
  
  // === Detailed Propagation (Tier 1-2) ===
  "propagation_trace": {
    "first_divergent_step": "string (canonical_step_id) | null",
    "divergent_steps": ["string[] (canonical_step_ids)"],
    "convergent_steps": ["string[] (canonical_step_ids)"]
  },
  
  // === Recovery Details (Tier 1) ===
  "recovery_details": {
    "extra_steps": "int | null",
    "extra_tool_calls": "int | null",
    "retries": "int | null",
    "recovery_successful": "bool | null"
  },
  
  // === Execution Metadata ===
  "execution": {
    "method": "string (e.g., 'swebench_harness', 'gaia_exact_match')",
    "duration_seconds": "float",
    "errors": ["string[] | null"],
    "logs": "string | null"
  }
}
```

### Verifier Output by Benchmark

#### SWE-bench Verifier Output

```json
{
  "test_results": {
    "total_tests": "int",
    "passed_tests": "int",
    "failed_tests": "int",
    "error_tests": "int"
  },
  "test_pass_rate": "float",
  "failing_test_names": ["string[]"],
  "patch_applied": "bool",
  "patch_diff": "string"
}
```

#### GAIA Verifier Output

```json
{
  "expected_answer": "string",
  "predicted_answer": "string",
  "exact_match": "bool",
  "normalized_expected": "string",
  "normalized_predicted": "string",
  "partial_match_score": "float (0-1)"
}
```

#### ToolBench Verifier Output

```json
{
  "task_completed": "bool",
  "success_heuristic": "float (0-1)",
  "final_action": "string",
  "api_errors_count": "int",
  "reached_goal_state": "bool | null"
}
```

---

## 5C.5 Tier-Specific Evidence Collection

### Tier 1: Full Replay (SWE-bench)

**Process:**
1. Set up isolated environment (Docker container with repo state)
2. Apply baseline patch, run test suite
3. Record baseline outcome
4. Reset environment
5. Apply perturbed patch, run test suite
6. Record perturbed outcome
7. Compare step-by-step execution logs

**Metrics available:** OD, PD, RC

**SWE-bench Replay Configuration:**
```json
{
  "environment": "docker",
  "base_image": "swebench/evaluation:latest",
  "timeout_seconds": 300,
  "test_command": "pytest --tb=short",
  "capture_logs": true,
  "capture_step_outputs": true
}
```

### Tier 2: Downstream Regeneration (GAIA with Model Access)

**Process:**
1. Load perturbed trajectory up to perturbation point
2. Regenerate remaining steps using same model
3. Grade regenerated final answer
4. Compare propagation of perturbation

**Metrics available:** OD (via grading), PD

**Regeneration Configuration:**
```json
{
  "model": "gpt-4o",
  "temperature": 0,
  "max_regeneration_steps": 10,
  "stop_on_final_answer": true,
  "tools_available": ["web_search", "calculator"]
}
```

**Note:** Tier 2 provides partial PD — we see which regenerated steps differ, but can't know exact recovery cost without full replay.

### Tier 3: Final-Answer Grading (GAIA, ToolBench)

**Process:**
1. Extract final answer from baseline trajectory
2. Extract final answer from perturbed trajectory
3. Grade each against ground truth
4. Compute OD

**Metrics available:** OD only

**Grading Methods:**
| Method | Use Case | Implementation |
|--------|----------|----------------|
| Exact match | GAIA numeric/factual | Normalize + string compare |
| Fuzzy match | GAIA text | Levenshtein + semantic similarity |
| Heuristic | ToolBench | Task-specific success rules |
| LLM grader | Fallback | GPT-4 as backup grader |

---

## 5C.6 Outcome Degradation Binning

For analysis, OD is binned into discrete impact levels:

| Impact Level | OD Range | Binary | Description |
|--------------|----------|--------|-------------|
| **0 (None)** | OD = 0 | 0 | No outcome change |
| **1 (Minor)** | 0 < OD ≤ 0.25 | 1 | Small degradation |
| **2 (Moderate)** | 0.25 < OD ≤ 0.5 | 1 | Significant degradation |
| **3 (Critical)** | OD > 0.5 | 1 | Task failure |

### Binary Outcome Degradation

```
OD_binary = {
  1   if baseline passes AND perturbed fails
  0   if both pass OR both fail
  -1  if baseline fails AND perturbed passes (unexpected)
}
```

---

## 5C.7 Propagation Analysis (Tier 1-2)

### Divergence Detection

Compare step outputs between baseline and perturbed after the perturbation point:

```python
def compute_propagation(baseline_steps, perturbed_steps, perturbation_step_idx):
    divergent = []
    convergent = []
    
    for i in range(perturbation_step_idx, len(baseline_steps)):
        baseline_output = baseline_steps[i].get("observation") or baseline_steps[i].get("extracted_value")
        perturbed_output = perturbed_steps[i].get("observation") or perturbed_steps[i].get("extracted_value")
        
        if semantically_different(baseline_output, perturbed_output):
            divergent.append(baseline_steps[i]["canonical_step_id"])
        else:
            convergent.append(baseline_steps[i]["canonical_step_id"])
    
    return {
        "first_divergent_step": divergent[0] if divergent else None,
        "divergent_steps": divergent,
        "convergent_steps": convergent,
        "propagation_depth": len(divergent)
    }
```

### Semantic Difference Check

| Output Type | Comparison Method |
|-------------|------------------|
| Numeric | Absolute difference > threshold |
| String (short) | Edit distance > threshold |
| String (long) | Semantic embedding similarity < threshold |
| Structured | Key-by-key recursive comparison |

---

## 5C.8 Recovery Cost Computation (Tier 1)

For full replay, measure resources needed to recover:

```python
def compute_recovery_cost(baseline_trace, perturbed_trace):
    return {
        "extra_steps": len(perturbed_trace.steps) - len(baseline_trace.steps),
        "extra_tool_calls": count_tool_calls(perturbed_trace) - count_tool_calls(baseline_trace),
        "retries": count_retries(perturbed_trace) - count_retries(baseline_trace),
        "recovery_successful": perturbed_trace.final_outcome == baseline_trace.final_outcome
    }
```

**Recovery Cost Interpretation:**
| RC Value | Interpretation |
|----------|---------------|
| RC = 0 | No additional cost (error absorbed) |
| RC = 1-3 | Minor recovery cost |
| RC = 4-10 | Moderate recovery cost |
| RC > 10 | High recovery cost |
| Recovery failed | Unrecoverable error |

---

## 5C.9 Ground Truth Impact Label

The outcome evidence provides the **true impact** for comparison against:
- Judge's `predicted_impact_score`
- Human's `impact_tier`
- Perturbation's `expected_impact`

### True Impact Derivation

```python
def derive_true_impact(outcome_record):
    od = outcome_record["metrics"]["outcome_degradation"]
    
    if od == 0:
        return 0  # No impact
    elif od <= 0.25:
        return 1  # Minor
    elif od <= 0.5:
        return 2  # Moderate
    else:
        return 3  # Critical
```

### Validation Against Expected Impact

Compare `true_impact` to `perturbation_record.expected_impact`:

| Comparison | Interpretation |
|------------|---------------|
| true = expected | Prediction accurate |
| true > expected | Underestimated impact |
| true < expected | Overestimated impact |

---

## 5C.10 Data Location

```
data/outcome_evidence/
├── tier_1/                        # Full replay results (SWE-bench)
│   ├── replay_results.json
│   ├── logs/
│   │   └── {evaluation_unit_id}/
│   │       ├── baseline_log.txt
│   │       └── perturbed_log.txt
│   └── step_outputs/
│       └── {evaluation_unit_id}/
│           ├── baseline_steps.json
│           └── perturbed_steps.json
├── tier_2/                        # Downstream regeneration results
│   ├── regeneration_results.json
│   └── regenerated_trajectories/
├── tier_3/                        # Final-answer grading results
│   └── grading_results.json
├── aggregated/
│   └── outcome_evidence_all.json
└── metadata/
    ├── tier_assignments.json
    └── execution_log.json
```

---

## 5C.11 Implementation Plan

### Module Structure

```
src/outcome_evidence/
├── __init__.py
├── tier_1/
│   ├── __init__.py
│   ├── swebench_replay.py       # SWE-bench full replay
│   ├── docker_env.py            # Container management
│   └── log_parser.py            # Parse execution logs
├── tier_2/
│   ├── __init__.py
│   ├── regeneration.py          # Downstream regeneration
│   └── model_client.py          # LLM client for regeneration
├── tier_3/
│   ├── __init__.py
│   ├── exact_match.py           # Exact match grading
│   ├── fuzzy_match.py           # Fuzzy/semantic grading
│   └── heuristic_grader.py      # ToolBench heuristics
├── metrics.py                   # OD, PD, RC computation
├── propagation.py               # Propagation analysis
├── aggregation.py               # Aggregate all evidence
└── export.py                    # Export for analysis
```

### Scripts

```bash
# Tier 1: Run SWE-bench replay
python -m src.outcome_evidence.tier_1.swebench_replay \
  --evaluation-units data/evaluation_units/canonical/swebench_eval_units.json \
  --output data/outcome_evidence/tier_1/ \
  --parallel 4

# Tier 2: Run downstream regeneration
python -m src.outcome_evidence.tier_2.regeneration \
  --evaluation-units data/evaluation_units/canonical/gaia_eval_units.json \
  --model gpt-4o \
  --output data/outcome_evidence/tier_2/

# Tier 3: Run final-answer grading
python -m src.outcome_evidence.tier_3.grading \
  --evaluation-units data/evaluation_units/canonical/ \
  --output data/outcome_evidence/tier_3/

# Aggregate all evidence
python -m src.outcome_evidence.aggregation \
  --tier-1 data/outcome_evidence/tier_1/ \
  --tier-2 data/outcome_evidence/tier_2/ \
  --tier-3 data/outcome_evidence/tier_3/ \
  --output data/outcome_evidence/aggregated/
```

---

## 5C.12 Error Handling

### Replay Failures

| Failure Type | Handling | Recording |
|--------------|----------|-----------|
| Environment setup failed | Skip unit, log error | `execution.errors` |
| Timeout | Record timeout, use partial results | `execution.errors` |
| Test suite not found | Fall back to Tier 3 grading | Downgrade tier |
| Non-deterministic tests | Run 3x, take majority | `execution.logs` |

### Regeneration Failures

| Failure Type | Handling | Recording |
|--------------|----------|-----------|
| Model API error | Retry 3x with backoff | `execution.errors` |
| Infinite loop | Stop at max steps | `execution.logs` |
| Tool unavailable | Skip tool calls, log | `execution.errors` |

### Grading Failures

| Failure Type | Handling | Recording |
|--------------|----------|-----------|
| No final answer | Grade as failure (score=0) | `verifier_output` |
| Unparseable answer | Fall back to LLM grader | `execution.method` |
| Missing ground truth | Exclude from OD analysis | Flag unit |

---

## 5C.13 Dependencies

| Dependency | Section | What's Needed |
|------------|---------|---------------|
| Evaluation Unit (Section 4) | Input | Outcome verification views |
| Perturbation Record (Section 3) | Reference | Expected impact for validation |
| Consequentiality (Section 6B) | Output | True impact for calibration |
| Agreement Analysis (Section 6C) | Output | Outcome for claim validation |

---

## 5C.14 Success Criteria

Section 5C is complete when:

### Coverage
1. [ ] All ~350 SWE-bench units have Tier 1 replay results
2. [ ] All ~350 GAIA units have Tier 2 or Tier 3 results
3. [ ] All ~1000 ToolBench units have Tier 3 grading results
4. [ ] All 1700 units have outcome_degradation computed

### Quality
5. [ ] Replay success rate >90% for Tier 1 units
6. [ ] Grading coverage >95% for Tier 3 units
7. [ ] No systematic failures by perturbation type
8. [ ] Execution logs captured for debugging

### Metrics
9. [ ] OD computed for all units
10. [ ] PD computed for Tier 1-2 units
11. [ ] RC computed for Tier 1 units
12. [ ] True impact labels derived for all units

### Validation
13. [ ] OD distribution is reasonable (not all 0 or all 1)
14. [ ] Negative OD cases (<5%) flagged and reviewed
15. [ ] True impact vs expected impact correlation computed
16. [ ] Cross-check with perturbation_record.expected_impact

### Output
17. [ ] Aggregated outcome evidence file created
18. [ ] Evidence linked to evaluation_unit_id
19. [ ] Ready for downstream analysis (6B, 6C)

---

## 5C.15 Example Outcome Records

### Tier 1 Example (SWE-bench)

```json
{
  "outcome_id": "outcome_swe_001",
  "evaluation_unit_id": "eval::swebench_tenacity::001",
  "created_at": "2026-04-12T08:00:00Z",
  
  "replay_tier": 1,
  "evidence_method": "full_replay",
  
  "baseline": {
    "trajectory_variant_id": "swebench_tenacity::base",
    "outcome_score": 1.0,
    "outcome_binary": true,
    "verifier_output": {
      "test_results": {
        "total_tests": 45,
        "passed_tests": 45,
        "failed_tests": 0,
        "error_tests": 0
      },
      "test_pass_rate": 1.0,
      "failing_test_names": [],
      "patch_applied": true
    }
  },
  
  "perturbed": {
    "trajectory_variant_id": "swebench_tenacity::pert::001",
    "outcome_score": 0.89,
    "outcome_binary": false,
    "verifier_output": {
      "test_results": {
        "total_tests": 45,
        "passed_tests": 40,
        "failed_tests": 5,
        "error_tests": 0
      },
      "test_pass_rate": 0.89,
      "failing_test_names": ["test_wait_chain_order", "test_wait_chain_reversed", ...],
      "patch_applied": true
    }
  },
  
  "metrics": {
    "outcome_degradation": 0.11,
    "outcome_degradation_binary": 1,
    "propagation_depth": 2,
    "recovery_cost": {
      "extra_steps": 0,
      "extra_tool_calls": 0,
      "retries": 0,
      "recovery_successful": false
    }
  },
  
  "propagation_trace": {
    "first_divergent_step": "swebench_tenacity::step::13",
    "divergent_steps": ["swebench_tenacity::step::13", "swebench_tenacity::step::14"],
    "convergent_steps": []
  },
  
  "recovery_details": {
    "extra_steps": 0,
    "extra_tool_calls": 0,
    "retries": 0,
    "recovery_successful": false
  },
  
  "execution": {
    "method": "swebench_harness",
    "duration_seconds": 127.5,
    "errors": null,
    "logs": "logs/eval_swebench_tenacity_001/"
  }
}
```

### Tier 3 Example (GAIA)

```json
{
  "outcome_id": "outcome_gaia_001",
  "evaluation_unit_id": "eval::gaia_122::001",
  "created_at": "2026-04-12T09:00:00Z",
  
  "replay_tier": 3,
  "evidence_method": "final_answer_grading",
  
  "baseline": {
    "trajectory_variant_id": "gaia_122::base",
    "outcome_score": 1.0,
    "outcome_binary": true,
    "verifier_output": {
      "expected_answer": "6",
      "predicted_answer": "6",
      "exact_match": true,
      "normalized_expected": "6",
      "normalized_predicted": "6",
      "partial_match_score": 1.0
    }
  },
  
  "perturbed": {
    "trajectory_variant_id": "gaia_122::pert::001",
    "outcome_score": 0.0,
    "outcome_binary": false,
    "verifier_output": {
      "expected_answer": "6",
      "predicted_answer": "4",
      "exact_match": false,
      "normalized_expected": "6",
      "normalized_predicted": "4",
      "partial_match_score": 0.0
    }
  },
  
  "metrics": {
    "outcome_degradation": 1.0,
    "outcome_degradation_binary": 1,
    "propagation_depth": null,
    "recovery_cost": null
  },
  
  "propagation_trace": null,
  "recovery_details": null,
  
  "execution": {
    "method": "gaia_exact_match",
    "duration_seconds": 0.5,
    "errors": null,
    "logs": null
  }
}
```

### Tier 3 Example (Placebo - No Impact)

```json
{
  "outcome_id": "outcome_toolbench_placebo",
  "evaluation_unit_id": "eval::toolbench_146416::placebo",
  "created_at": "2026-04-12T10:00:00Z",
  
  "replay_tier": 3,
  "evidence_method": "final_answer_grading",
  
  "baseline": {
    "trajectory_variant_id": "toolbench_146416::base",
    "outcome_score": 0.4,
    "outcome_binary": false,
    "verifier_output": {
      "task_completed": false,
      "success_heuristic": 0.4,
      "final_action": "I apologize, but I was unable to retrieve nutritional information...",
      "api_errors_count": 3,
      "reached_goal_state": false
    }
  },
  
  "perturbed": {
    "trajectory_variant_id": "toolbench_146416::pert::placebo",
    "outcome_score": 0.4,
    "outcome_binary": false,
    "verifier_output": {
      "task_completed": false,
      "success_heuristic": 0.4,
      "final_action": "I apologize, but I was unable to retrieve nutritional information...",
      "api_errors_count": 3,
      "reached_goal_state": false
    }
  },
  
  "metrics": {
    "outcome_degradation": 0.0,
    "outcome_degradation_binary": 0,
    "propagation_depth": null,
    "recovery_cost": null
  },
  
  "propagation_trace": null,
  "recovery_details": null,
  
  "execution": {
    "method": "toolbench_heuristic",
    "duration_seconds": 1.2,
    "errors": null,
    "logs": null
  }
}
```
