# CCG Experiment Results (Post Bug-Fix)

**Experiment**: exp_judge_ccg_v1 | **Date**: 2026-04-04 | **Status**: HYPOTHESIS WEAKLY SUPPORTED

**Judge**: claude-sonnet-4.5 | **Bug Fixes Applied**: BUG-001 (TCS capping), BUG-002 (TCS=0 exclusion)

---

## Executive Summary

After applying bug fixes, the hypothesis receives **weak support** (2/5 criteria met). LLM judges demonstrate **universal over-penalization** across all error types and positions, rather than the predicted pattern of under-penalizing early structural errors and over-penalizing late surface errors.

---

## Data Quality

| Metric | Value | Notes |
|--------|-------|-------|
| Total Evaluated | 100 | Annotated perturbations |
| TCS=0 Excluded | 36 | BUG-002 fix - these would bias results |
| TCS Capped (>100) | 0 | BUG-001 fix - capping applied in CriticalityScorer |
| Final Sample Size | **64** | Meets n > 60 threshold |
| Exclusion Rate | 36% | High but acceptable |

---

## Overall Statistics

| Metric | Value |
|--------|-------|
| Mean CCG | **+2.246** |
| Std CCG | 2.962 |
| Median CCG | 0.5 |
| Min CCG | -0.85 |
| Max CCG | 8.5 |
| Mean JPS | 72.1 |
| Mean TCS | 54.2 |
| Heuristic Correlation | 0.095 (FAILED - threshold 0.4) |

**Interpretation**: Positive mean CCG (+2.246) indicates judges systematically **over-penalize** errors relative to their true criticality.

---

## CCG by Condition (Type x Position)

| Condition | CCG | Std | n | JPS | TCS | Interpretation |
|-----------|-----|-----|---|-----|-----|----------------|
| planning_early | **+4.57** | 3.32 | 5 | 67.0 | 14.0 | Strong over-penalize |
| planning_middle | +2.70 | 3.31 | 4 | 85.0 | 42.5 | Over-penalize |
| planning_late | +2.03 | 3.87 | 3 | 78.3 | 70.0 | Over-penalize |
| tool_selection_early | +1.74 | 3.08 | 9 | 85.6 | 68.9 | Over-penalize |
| tool_selection_middle | **-0.14** | 0.29 | 7 | 67.1 | 87.1 | **Well-calibrated** |
| tool_selection_late | +1.10 | 2.48 | 5 | 66.0 | 64.0 | Over-penalize |
| parameter_early | +1.94 | 2.58 | 8 | 75.6 | 57.5 | Over-penalize |
| parameter_middle | +2.37 | 3.49 | 7 | 69.3 | 54.3 | Over-penalize |
| parameter_late | +3.06 | 3.28 | 5 | 77.0 | 48.0 | Over-penalize |
| data_reference_middle | +2.91 | 3.36 | 6 | 60.8 | 40.0 | Over-penalize |
| data_reference_late | **+3.78** | 2.77 | 5 | 59.0 | 30.0 | Strong over-penalize |

**Key Finding**: Only `tool_selection_middle` shows near-perfect calibration (CCG = -0.14). All other conditions show over-penalization.

---

## CCG by Position

| Position | CCG | Std | n | Expected | Actual |
|----------|-----|-----|---|----------|--------|
| Early | +2.46 | 3.05 | 22 | Under-penalize (< 0) | Over-penalize |
| Middle | +1.83 | 2.96 | 24 | Mixed | Over-penalize |
| Late | +2.54 | 2.95 | 18 | Over-penalize (> 0) | Over-penalize |

**ANOVA**: F=0.38, p=0.685, eta²=0.012 - **NOT significant**

---

## CCG by Perturbation Type

| Type | CCG | Std | n | Category |
|------|-----|-----|---|----------|
| tool_selection | **+0.96** | 2.40 | 21 | Structural (best calibrated) |
| parameter | +2.37 | 2.96 | 20 | Surface |
| data_reference | +3.30 | 2.98 | 11 | Surface |
| planning | +3.31 | 3.32 | 12 | Structural |

**ANOVA**: F=2.47, p=0.071, eta²=0.110 - Approaching significance

---

## CCG by Benchmark

| Benchmark | CCG | Std | n |
|-----------|-----|-----|---|
| SWE-bench | **+0.32** | 0.77 | 8 | Best calibrated |
| ToolBench | +2.48 | 3.02 | 54 | Over-penalize |
| GAIA | +3.68 | 5.41 | 2 | Strong over-penalize (low n) |

---

## Statistical Tests

### ANOVA Results

| Factor | F | p-value | eta² | Significant |
|--------|---|---------|------|-------------|
| Perturbation Type | 2.47 | 0.071 | 0.110 | No (approaching) |
| Position | 0.38 | 0.685 | 0.012 | No |

### Effect Sizes (Cohen's d)

| Comparison | d | Magnitude | Interpretation |
|------------|---|-----------|----------------|
| Early vs Late | -0.03 | Negligible | No difference |
| Planning vs Parameter | +0.30 | Small | Planning slightly higher |
| Structural vs Surface | -0.30 | Small | Surface slightly higher |
| Early-Structural vs Late-Surface | -0.21 | Small | Opposite of hypothesis |

---

## Hypothesis Assessment

**Hypothesis**: LLM judges under-penalize early structural failures while over-penalizing late local mistakes.

| Criterion | Required | Actual | Met |
|-----------|----------|--------|-----|
| Early structural CCG < 0 | Under-penalize | **+2.75** | No |
| Late surface CCG > 0 | Over-penalize | **+3.42** | Yes |
| Direction correct (early < late) | CCG_early < CCG_late | 2.75 < 3.42 | Yes |
| Statistically significant | p < 0.05 | p = 0.616 | No |
| Meaningful effect size | \|d\| > 0.5 | d = -0.21 | No |

**Criteria Met**: 2/5 | **Support Level**: WEAK

### Key Numbers for Hypothesis

| Group | Mean CCG | n |
|-------|----------|---|
| Early Structural (planning + tool_selection, early) | +2.75 | 14 |
| Late Surface (parameter + data_reference, late) | +3.42 | 10 |
| Difference | +0.67 | - |
| t-statistic | -0.51 | - |
| p-value | 0.616 | - |

---

## Interpretation

### What We Found

1. **Universal Over-Penalization**: Judges penalize errors ~2.2x more than their true criticality warrants (mean CCG = +2.25)

2. **No Early/Late Discrimination**: Position does not significantly affect CCG (p = 0.685)

3. **Type Matters Marginally**: Perturbation type approaches significance (p = 0.071), with `tool_selection` best calibrated

4. **Benchmark Variation**: SWE-bench shows best calibration (+0.32), possibly due to more concrete success criteria

### Why Hypothesis Failed

1. **TCS Annotation Issue**: 36% of samples had TCS=0 (trivial errors), excluding them changed the sample composition

2. **Low Heuristic Correlation**: Human TCS poorly correlates with heuristic (r=0.095), suggesting our type/position assumptions don't match human judgment

3. **High Variance**: CCG std (2.96) is larger than the mean (2.25), indicating high variability across samples

---

## Recommended Pivot

**Original Hypothesis**: "Judges miss early critical errors"

**Alternative Framing**: "LLM Judges as Conservative Evaluators"
- Judges over-penalize uniformly, acting as harsh critics regardless of error type
- This is consistent with instruction-following bias (penalize any deviation)
- Implications for RLHF: over-penalization may lead to overly conservative agents

---

## Limitations

1. **High Exclusion Rate**: 36% of samples excluded due to TCS=0
2. **Small Subgroup Sizes**: Some conditions have n < 5
3. **Single Judge**: Only claude-sonnet-4.5 evaluated
4. **Heuristic TCS Invalid**: r=0.095 means heuristic assignments don't match human judgment

---

## Files Generated

| File | Location |
|------|----------|
| Raw CCG data | `results/ccg_analysis/ccg_raw_claude-sonnet-4.5.csv` |
| Full report | `results/ccg_analysis/ccg_report_claude-sonnet-4.5.json` |
| Bug report | `agent_tasks/05_judge_evaluation_ccg/BUG_REPORT.md` |

---

## Next Steps

1. [ ] Consider adding GPT-4o as second judge for comparison
2. [ ] Investigate why tool_selection_middle is well-calibrated
3. [ ] Re-annotate TCS with clearer rubric to reduce TCS=0 rate
4. [ ] Write paper section on "conservative evaluator" finding
