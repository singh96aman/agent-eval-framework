# PDR Hierarchy Analysis - dual_mode_v4

**Date:** 2026-04-14  
**Experiment:** dual_mode_v4  
**Status:** Complete

---

## Executive Summary

Ran dual_mode_v4 with improved coarse-grained detection. Results show the PDR hierarchy is still inverted - coarse-grained detection is lower than fine-grained in blinded mode.

---

## Pipeline Execution

| Phase | Status | Details |
|-------|--------|---------|
| Load | Complete | 32 toolbench trajectories (2.8% verification pass rate) |
| Typing | Complete | 32 trajectories typed |
| Perturb | Complete | 50 perturbations generated |
| Judge | Complete | 100 evaluations (50 single + 50 blinded_pair) |
| Compute | Complete | Metrics computed and stored |

### Data Issues

- **ToolBench:** Only 32/75 target (2.8% baseline verification pass rate)
- **SWE-bench:** 0/25 target (0% baseline verification pass rate - all rejected)
- **Small sample size:** 50 perturbations total (limited statistical power)

---

## Perturbation Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Coarse-grained | 23 | 46% |
| Fine-grained | 16 | 32% |
| Placebo | 11 | 22% |

**Class Validation:** 72% match rate (36/50)

---

## Results: All Perturbations

### Blinded Pair Mode (v1_blinded_pair)

| Class | Detection Rate | Count |
|-------|----------------|-------|
| Placebo FP | **0.0%** | 0/11 |
| Fine-grained | 25.0% | 4/16 |
| Coarse-grained | **0.0%** | 0/23 |
| **Overall PDR** | **10.3%** | 4/39 |

### Single Trajectory Mode (v1_single_trajectory)

| Class | Detection Rate | Count |
|-------|----------------|-------|
| Placebo FP | 27.3% | 3/11 |
| Fine-grained | 43.8% | 7/16 |
| Coarse-grained | 17.4% | 4/23 |
| **Overall PDR** | **28.2%** | 11/39 |

---

## Results: Validated Perturbations Only (class_matches=1)

### Blinded Pair Mode

| Class | Detection Rate | Count |
|-------|----------------|-------|
| Placebo FP | 0.0% | 0/9 |
| Fine-grained | 28.6% | 4/14 |
| Coarse-grained | 0.0% | 0/13 |
| **Overall PDR** | **14.8%** | 4/27 |

### Single Trajectory Mode

| Class | Detection Rate | Count |
|-------|----------------|-------|
| Placebo FP | 33.3% | 3/9 |
| Fine-grained | 50.0% | 7/14 |
| Coarse-grained | 15.4% | 2/13 |
| **Overall PDR** | **33.3%** | 9/27 |

---

## Results: Clean Baselines Only (no pre-existing errors)

**Clean Baselines:** 42/50 (84%)

### Blinded Pair Mode

| Class | Detection Rate | Count |
|-------|----------------|-------|
| Placebo FP | 0.0% | 0/9 |
| Fine-grained | 30.8% | 4/13 |
| Coarse-grained | 0.0% | 0/20 |
| **Overall PDR** | **12.1%** | 4/33 |

### Single Trajectory Mode

| Class | Detection Rate | Count |
|-------|----------------|-------|
| Placebo FP | 11.1% | 1/9 |
| Fine-grained | 30.8% | 4/13 |
| Coarse-grained | 5.0% | 1/20 |
| **Overall PDR** | **15.2%** | 5/33 |

---

## Summary Comparison

| Version | Placebo FP | Fine-grained | Coarse-grained | PDR |
|---------|------------|--------------|----------------|-----|
| v1_blinded_pair | 0.0% | 25.0% | 0.0% | 10.3% |
| v1_single_trajectory | 27.3% | 43.8% | 17.4% | 28.2% |

---

## Hierarchy Status

### Target Hierarchy
```
Placebo FP (<10%) < Fine-grained PDR < Coarse-grained PDR
```

### Actual Results (Blinded Pair - Preferred Mode)

| Check | Status | Notes |
|-------|--------|-------|
| Placebo FP < 10% | PASS | 0.0% FP rate |
| Fine > Placebo | PASS | 25% > 0% |
| Coarse > Fine | **FAIL** | 0% < 25% (inverted!) |

### Actual Results (Single Trajectory)

| Check | Status | Notes |
|-------|--------|-------|
| Placebo FP < 10% | **FAIL** | 27.3% FP rate (too high) |
| Fine > Placebo | PASS | 43.8% > 27.3% |
| Coarse > Fine | **FAIL** | 17.4% < 43.8% (inverted!) |

---

## Key Findings

### 1. Coarse-grained PDR is Still Inverted
Despite WRONG_PLAN_PROMPT_V3 and type weights, coarse-grained detection remains lower than fine-grained:
- Blinded: 0% coarse vs 25% fine
- Single: 17.4% coarse vs 43.8% fine

### 2. Blinded Pair Has Low False Positives
- 0% placebo FP in blinded mode (excellent)
- 27.3% placebo FP in single trajectory mode (too high)

### 3. Type Weights May Not Be Working
Config specified weights:
- false_terminal: 30%
- premature_termination: 30%
- wrong_tool_family: 25%
- wrong_plan: 15%

Need to verify actual distribution of coarse-grained types generated.

### 4. Small Sample Size Limits Conclusions
- Only 50 perturbations (target was 300)
- Only 23 coarse-grained samples
- Statistical power is very limited

---

## Root Cause Hypotheses

### Hypothesis A: Coarse-grained perturbations are not visually distinct enough
The judge may not recognize structural errors in blinded comparison because both trajectories look "reasonable" even when one has wrong plan/tool.

### Hypothesis B: Wrong_plan perturbations create suboptimal but not impossible plans
LLM-generated wrong plans may still be plausible enough to look correct to the judge.

### Hypothesis C: Judge focuses on execution quality, not plan correctness
The judge may score based on tool execution and formatting rather than whether the plan would achieve the goal.

---

## Recommendations

### Short-term
1. **Investigate coarse-grained samples:** Manually review the 23 coarse-grained perturbations to understand why they're not being detected
2. **Check type distribution:** Verify which coarse-grained types were actually generated
3. **Review judge reasoning:** Examine judge outputs for coarse-grained units

### Medium-term
1. **Increase sample size:** Run with baseline_verify_min_score lowered to get more trajectories
2. **Add explicit structural checks:** Consider rule-based detection for obvious structural errors
3. **Test different coarse-grained types:** Compare detection rates across false_terminal, premature_termination, wrong_tool_family, wrong_plan

### Long-term
1. **Revise perturbation taxonomy:** Current coarse-grained types may not produce visually distinct errors
2. **Consider outcome-based evaluation:** Use task success as ground truth rather than error detection

---

## Files and Artifacts

| Purpose | Path |
|---------|------|
| Config | config/experiments/v3/poc/dual_mode_v4.json |
| Prompts | src/prompts/perturbation_prompts.py |
| Logs | logs/dual_mode_v4.log |
| MongoDB Collections | trajectories, typed_trajectories, perturbed_trajectories, evaluation_units, judge_eval_outputs, metrics |

---

## Caching Feature Added

During this experiment, added versioned caching to all pipeline phases:
- **Version field** added to all phase configs and data schemas
- **Caching logic** checks for existing data before running expensive operations
- **Verified working:** Re-run of load phase correctly detected 32 cached trajectories

This will speed up future experiments by avoiding redundant work.
