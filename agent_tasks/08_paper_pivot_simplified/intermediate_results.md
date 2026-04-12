# Intermediate Results: LLM Judge Error Detection Study

**Date:** 2026-04-08  
**Experiment ID:** `exp_trajectory_sampling_v7`  
**Analysis Script:** [ops/analyze_judge_eval.py](../../ops/analyze_judge_eval.py)

## Datasets Used

| Dataset | Path | Count |
|---------|------|-------|
| Judge Eval Outputs | [data/judge_outputs/eval_units/exp_trajectory_sampling_v7_judge_eval_outputs.json](../../data/judge_outputs/eval_units/exp_trajectory_sampling_v7_judge_eval_outputs.json) | 652 |
| Outcome Evidence | [data/outcome_evidence/exp_trajectory_sampling_v7_outcome_evidence.json](../../data/outcome_evidence/exp_trajectory_sampling_v7_outcome_evidence.json) | 652 |
| Perturbations | [data/perturbed/exp_trajectory_sampling_v7_perturbations.json](../../data/perturbed/exp_trajectory_sampling_v7_perturbations.json) | 652 |

## Ground Truth Distribution

| Expected Impact | Label | Count |
|-----------------|-------|-------|
| 0 | Placebo (synonym swaps) | 201 |
| 1 | Minor | 19 |
| 2 | Moderate | 113 |
| 3 | Severe | 319 |

---

## Key Results

### 1. Detection Performance

| Metric | Value |
|--------|-------|
| Overall Detection Rate | 91.3% |
| Precision | 0.689 |
| Recall | 0.909 |
| **Specificity** | **0.080** |
| F1 Score | 0.784 |

**Finding:** The judge flags 92% of placebos as errors (false positive rate). Specificity of 8% means the `error_detected` field is essentially uninformative.

### 2. Impact Calibration

| Severity Level | Mean Predicted Impact | Expected (normalized) |
|----------------|----------------------|----------------------|
| Placebo (0) | 0.650 | 0.00 |
| Minor (1) | 0.461 | 0.33 |
| Moderate (2) | 0.619 | 0.67 |
| Severe (3) | 0.662 | 1.00 |

| Statistic | Value |
|-----------|-------|
| Spearman ρ | 0.010 |
| Spearman p-value | 0.794 |
| T-test (placebo vs real) | p = 0.781 |
| Cohen's d | -0.02 |

**Finding:** Near-zero correlation between predicted and expected impact. The judge predicts **identical impact scores** for placebos (0.650) and severe errors (0.662).

### 3. Localization Accuracy (Real Errors Only)

| Metric | Value |
|--------|-------|
| Exact Step Match | 26.1% |
| Within ±1 Step | 31.2% |
| Mean Step Error | 5.0 steps |

**Finding:** Even when detecting errors, the judge points to the wrong step ~74% of the time.

### 4. Critical Errors

| Category | Count |
|----------|-------|
| Missed Critical (expected ≥ 2, predicted < 0.3) | 54 |
| False Alarms (expected = 0, predicted ≥ 0.7) | 144 |

### 5. Detection by Perturbation Type

| Type | Detection Rate | Mean Predicted Impact | Expected Impact |
|------|---------------|----------------------|-----------------|
| synonym (placebo) | 92.0% | 0.650 | 0.0 |
| wrong_parameter | 92.5% | 0.640 | 2.5 |
| skipped_prerequisite | 87.6% | 0.639 | 2.8 |
| wrong_plan | 90.9% | 0.667 | 2.8 |

**Finding:** Detection rates are ~90% regardless of perturbation type. The judge cannot discriminate.

---

## Research Insights for the Community

### Insight 1: The Sensitivity Problem, Not a Detection Problem

LLM judges detect errors at **91%+ rate across ALL conditions** — including placebos. This is not high recall; it's **near-zero specificity**.

**Implication:** Binary error detection (`error_detected`) is uninformative. The judge essentially labels everything as erroneous.

### Insight 2: Impact Predictions Are Uncalibrated

Placebo mean = 0.650, Severe error mean = 0.662 — statistically indistinguishable (p = 0.79).

**Implication:** Judges cannot rank trajectories by quality. A trajectory with a catastrophic tool failure receives the same impact score as one with cosmetic synonym changes.

### Insight 3: Localization Is Poor

26% exact match on error step identification, 5 steps off on average.

**Implication:** Even if you trust the judge's detection, you cannot trust its explanation of *where* the error occurred. This undermines interpretability claims.

### Insight 4: Placebos Expose the Failure Mode

Placebo perturbations (synonym swaps like "Finish" → "Complete") are semantically identical but syntactically different. Judges flag 92% of these as errors.

**Implication:** Judges may be detecting *any* deviation from expected patterns, not actual errors. They confuse "unusual" with "wrong."

### Insight 5: Contrastive Evaluation May Be Necessary

Single-trajectory evaluation invites hallucinated errors. The judge has no reference point for "normal."

**Recommendation:** Evaluate trajectory *pairs* (baseline vs. perturbed) and ask which is better, rather than asking if a single trajectory contains errors.

### Insight 6: Heuristic Grading Is Insufficient

Tier 3 heuristic grading (pattern matching for final answers) returned **all zeros** — it cannot capture semantic degradation from perturbations.

**Implication:** Ground truth for agent evaluation requires either:
- Execution-based replay (expensive)
- Human annotation (doesn't scale)
- Better LLM grading prompts (needs research)

### Insight 7: The Core Insight

> **LLM judges are critics, not diagnosticians.**
> 
> They can articulate *why something might be wrong*, but cannot reliably distinguish *whether* something is actually wrong. High fluency masks low discrimination.

This has implications for any benchmark that relies on LLM-as-a-Judge for agent evaluation (SWE-bench, ToolBench, etc.) — reported scores may reflect judge sensitivity more than agent capability.

---

## Practical Recommendations

| Problem | Recommendation |
|---------|----------------|
| High FP rate | Calibrate detection threshold using known-good trajectories |
| No severity discrimination | Use ordinal ranking, not binary detection |
| Poor localization | Don't trust step-level explanations |
| Uncalibrated impact | Report confidence intervals, not point estimates |
| Single-trajectory bias | Use contrastive/pairwise evaluation |

---

## Notes

- **Outcome Evidence Limitation:** Heuristic Tier 3 grading returned all zeros (insufficient for semantic perturbations). Analysis used `predicted_impact_score` as outcome signal.
- **Judge Model:** Claude Sonnet 4.5 (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`)
- **Prompt Template:** `single_trajectory_v1`

---

## How to Reproduce

```bash
cd /Users/amanzing/Paper-Research
python ops/analyze_judge_eval.py
```
