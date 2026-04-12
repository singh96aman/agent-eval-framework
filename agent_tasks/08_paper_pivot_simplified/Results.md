# Section 6 Analysis Results: Expert Review Request

**Experiment:** `exp_trajectory_sampling_v7`  
**Date:** 2026-04-11  
**Total Evaluation Units:** 652  
**Human Labels:** 50  

---

## Executive Summary

This document presents results from our experiment testing the claim:

> **"LLM judges detect errors better than they estimate their downstream consequence."**

We request expert review of both the quantitative findings and qualitative issues with the experimental setup that may affect validity.

---

## 1. Quantitative Results

### 1.1 Main Claim Test

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **PDR** (Perturbation Detection Rate) | 0.909 | Judge detects 91% of injected errors |
| **CCorr** (Consequence Correlation) | -0.076 | Near-zero correlation with outcome degradation |
| **Gap** (PDR - CCorr) | 0.985 | |
| **95% Bootstrap CI** | [0.908, 1.062] | CI excludes zero |
| **Claim Supported?** | **YES** | Statistically significant |

### 1.2 Detection Metrics (6A)

| Metric | Value | Description |
|--------|-------|-------------|
| PDR | 0.909 | True positive rate on non-placebo units |
| PNDR | 0.080 | True negative rate on placebos (92% false positive rate) |
| SLA | 0.261 | Step Localization Accuracy (26%) |
| TIA | 0.298 | Type Identification Accuracy (30%) |
| CER | 0.897 | Critical Error Recall (90%) |
| AUC | 0.518 | Detection AUC (near random) |
| F1 | 0.784 | F1 Score |

### 1.3 Calibration Metrics (6B)

| Metric | Value | Description |
|--------|-------|-------------|
| CCorr | -0.076 | Spearman correlation with outcome_degradation |
| \|CCE\| | 0.645 | Mean absolute calibration error |
| ORR | 0.764 | Over-reaction rate (76%) |
| URR | N/A | Under-reaction rate (no critical impacts in data) |
| Failure-ECE | 0.295 | Expected calibration error for failure prediction |

### 1.4 Proxy Calibration (using expected_impact)

*Note: outcome_degradation has no variance (all baselines failed), so expected_impact is used as proxy.*

| Metric | Value | Description |
|--------|-------|-------------|
| Proxy CCorr | 0.025 | Almost no correlation with expected_impact |
| Proxy MAE | 0.321 | Mean absolute error (0-1 scale) |
| Tier Accuracy | 0.164 | Only 16% exact tier match |
| Proxy ORR | 0.474 | 47% over-reaction on low expected_impact |
| Proxy URR | 0.188 | 19% under-reaction on critical expected_impact |

### 1.5 Human-Judge Agreement (N=50)

| Metric | Value | Description |
|--------|-------|-------------|
| Detection Agreement | 0.580 | 58% agreement on error detection |
| Type Agreement | 0.000 | 0% agreement on error type |
| Impact Tier MAE | 0.857 | ~1 tier difference on average |
| Impact Bias | +0.214 | Judge rates impact higher than humans |

---

## 2. Critical Data Quality Issues

### 2.1 Outcome Degradation Has No Signal

**Problem:** All 652 baseline trajectories failed (`baseline_outcome_binary = False`).

```
Baseline success rate:     0 / 652  (0%)
Outcome degradation = 0:   635 / 652  (97%)
Outcome degradation < 0:   17 / 652  (3%, perturbed did BETTER)
```

**Impact:** CCorr metric is meaningless. There's no variance in ground truth to correlate with.

**Evidence:**
- All toolbench and swebench baselines in this experiment failed their final answer grading
- This may indicate issues with the grading function or trajectory selection

### 2.2 `_old` Suffix Artifacts in Data

**Problem:** Found 18 units with `_old` suffix in tool arguments, suggesting mutation artifacts leaked into data.

**Example:**
```json
{
  "tool_name": "Finish",
  "tool_arguments": {
    "return_type": "give_answer_old",  // <-- Should be "give_answer"
    "final_answer": "..."
  }
}
```

**Impact:** Judge may be detecting these artifacts rather than semantic errors.

### 2.3 `</function>_mutated` Tags

**Problem:** Found 24 units with `</function>_mutated` patterns, indicating XML-like mutation markers visible in trajectory.

**Example:**
```
</parameter>
</function>_mutated
```

**Impact:** These are visible "tells" that could bias the judge toward detecting mutations without understanding the semantic error.

### 2.4 Perturbations Concentrated at End of Trajectories

**Problem:** 48.9% of perturbations occur at the last step (relative position > 0.8).

```
Perturbations at last step:  317 / 652  (48.6%)
Mean relative position:      0.54
Median relative position:    0.77
```

**Impact:** 
- Early planning errors (our research focus) are underrepresented
- Late-stage errors may have less downstream impact (already near completion)
- This biases results toward errors that don't propagate

### 2.5 Skipped Prerequisite Shows Deleted Steps

**Problem:** For `skipped_prerequisite` perturbations, we delete steps from the trajectory, making the error obvious in side-by-side comparison.

**Example:**
```
Unit: eval::swebench_pydantic__pydantic...::029
  Baseline steps: 25
  Perturbed steps: 24
  Difference: 1 step removed
```

**Impact:** Structural differences in trajectory length are trivially detectable, inflating PDR.

### 2.6 Perturbation Type Distribution Imbalance

```
Perturbation Distribution:
  placebo/synonym:                201 (31%)
  fine_grained/wrong_parameter:   201 (31%)
  coarse_grained/skipped_prereq:  137 (21%)
  coarse_grained/wrong_plan:      99  (15%)
  other types:                    14  (2%)
```

**Impact:** Only 4 perturbation types dominate. Planning errors (`wrong_plan`) are 15% of data.

### 2.7 Blind Review A/B Assignment Not Randomized

**Problem:** In annotation UI, Trajectory A appears to always be the perturbed version.

```
Human detected error in A (non-placebo): 17
Human detected error in B (non-placebo): 11
```

**Impact:** Annotators may develop bias toward labeling A as the error trajectory.

---

## 3. Interpretation of Results

### What the Results Show (if valid)

1. **High Detection Rate (PDR=0.91):** Judge is very good at detecting *something* is wrong
2. **Poor Specificity (PNDR=0.08):** Judge flags 92% of placebos as errors (high false positive)
3. **No Calibration Signal:** Cannot evaluate consequence estimation due to data issues
4. **Poor Human Agreement:** Only 58% detection agreement with humans

### What the Results May Actually Reflect

1. **Artifact Detection:** Judge may be detecting `_old`, `_mutated`, or structural differences rather than semantic errors
2. **Bias Toward "Something Wrong":** 92% false positive rate suggests judge defaults to detecting errors
3. **Trivial Differences:** Deleted steps and late-stage perturbations create easy detection targets
4. **Insufficient Evaluation:** Without outcome variance, we cannot test calibration

---

## 4. Questions for Expert Review

1. **Is the main claim actually tested?** Given that CCorr has no signal, does the PDR-CCorr comparison have meaning?

2. **Are the perturbations valid?** Should we filter out units with `_old` and `_mutated` artifacts?

3. **Should we re-run with different baselines?** Need trajectories where baseline succeeds to measure outcome degradation.

4. **Is the judge detecting artifacts or errors?** The 92% false positive rate suggests something systematic.

5. **What explains the 0% type agreement with humans?** This is surprisingly low and warrants investigation.

6. **Should perturbation position be controlled?** 49% at last step may not test "early planning errors" hypothesis.

---

## Appendix: Raw Data Samples

### A.1 Example `_old` Suffix

```json
Unit: eval::toolbench_52646::006
Step 11 (final):
{
  "tool_name": "Finish",
  "tool_arguments": {
    "return_type": "give_answer_old",  // ARTIFACT
    "final_answer": "The language of the text is Romanian..."
  }
}
```

### A.2 Example `_mutated` Tag

```
Unit: eval::swebench_dbader__schedule...::027
Step content:
...grep -n "def model_validator" /testbed/pydantic/functional_validators.py</parameter>
</function>_mutated  // ARTIFACT VISIBLE IN TRAJECTORY
```

### A.3 Example Skipped Prerequisite

```
Unit: eval::swebench_pydantic__pydantic...::029
Perturbation: skipped_prerequisite

Baseline trajectory: 25 steps
Perturbed trajectory: 24 steps (step removed)

The missing step is structurally obvious in side-by-side view.
```

### A.4 Step Position Distribution

```
Relative Position of Perturbation:
  0.0-0.2 (early):   ~10%
  0.2-0.4:           ~10%
  0.4-0.6:           ~10%
  0.6-0.8:           ~20%
  0.8-1.0 (late):    ~50%  <-- Half of all perturbations
```

---

*Generated by Section 6 Analysis Pipeline*
*Requesting expert review of methodology and results validity*
