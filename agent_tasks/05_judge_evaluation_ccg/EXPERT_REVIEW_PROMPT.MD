# Expert Review Request: LLM Judge Miscalibration Study

**Request**: We need a second opinion on this research study. Please review the methodology, results, and provide recommendations on whether to continue, pivot, or abandon this line of research.

**Your role**: You are an expert reviewer with experience in NeurIPS/ICLR program committees, specializing in LLM evaluation, agent systems, and AI safety.

---

## 1. Research Abstract

This research investigates whether LLM-based judges are systematically miscalibrated when evaluating agent trajectories. Specifically, we hypothesize that judges under-penalize early structural failures (e.g., flawed planning, wrong tool selection) while over-penalizing late local mistakes (e.g., parameter typos). Early errors are more critical because they propagate downstream and corrupt subsequent steps, yet judges may focus on surface-level correctness rather than causal impact. We introduce the Criticality-Calibration Gap (CCG) metric to quantify this miscalibration by comparing judge penalty scores against ground-truth criticality derived from task success degradation and downstream error propagation. Through controlled perturbation experiments on ToolBench and GAIA benchmarks, we aim to reveal systematic blind spots in LLM judgment that have implications for agent evaluation, RLHF reward modeling, and AI safety oversight.

---

## 2. Experimental Design

### 2.1 Dataset

| Source | Trajectories | Description |
|--------|--------------|-------------|
| ToolBench | 400 | API tool-calling agent trajectories |
| GAIA | 100 | General AI assistant trajectories |
| SWE-bench | 100 | Software engineering agent trajectories |
| **Total** | 600 | Multi-step agent trajectories |

### 2.2 Perturbation Design

We injected controlled errors into trajectories at different positions and of different types:

**Perturbation Types (4)**:
- `planning`: Flawed reasoning or goal decomposition
- `tool_selection`: Choosing wrong tool for the task
- `parameter`: Incorrect parameter values in tool calls
- `data_reference`: Wrong data references or hallucinated values

**Positions (3)**:
- `early`: First 33% of trajectory steps
- `middle`: Middle 33% of trajectory steps
- `late`: Final 33% of trajectory steps

**Conditions (11 total)**:
- planning_early, planning_middle, planning_late
- tool_selection_early, tool_selection_middle, tool_selection_late
- parameter_early, parameter_middle, parameter_late
- data_reference_middle, data_reference_late
- (data_reference_early excluded as conceptually invalid)

### 2.3 Core Metrics

**Judge Penalty Score (JPS)**:
```
JPS = 100 - overall_score
```
Where `overall_score` (0-100) comes from LLM judge evaluation.

**True Criticality Score (TCS)**:
Ground-truth measure of how critical an error actually is. Two approaches:

1. **Heuristic TCS** (used for full dataset):
```
planning_early: 90, planning_middle: 70, planning_late: 40
tool_selection_early: 80, tool_selection_middle: 60, tool_selection_late: 30
parameter_early: 50, parameter_middle: 40, parameter_late: 20
data_reference_middle: 30, data_reference_late: 20
```

2. **Human-annotated TCS** (for 100 sample subset):
```
TCS = (TSD * 50) + (SER * 10) + (Criticality * 8)

Where:
- TSD (0-1): Task Success Degradation - does error cause task failure?
- SER (0-3): Subsequent Error Rate - how many downstream errors?
- Criticality (1-5): Human rating of overall severity
```

**Criticality-Calibration Gap (CCG)**:
```
CCG = (JPS - TCS) / TCS
```
- CCG < 0: Under-penalization (judge missed critical error)
- CCG > 0: Over-penalization (judge overreacted)
- CCG = 0: Perfect calibration

### 2.4 Hypothesis

**H1**: LLM judges under-penalize early structural failures (CCG < 0 for planning_early, tool_selection_early)

**H2**: LLM judges over-penalize late surface errors (CCG > 0 for parameter_late, data_reference_late)

**H3**: The difference is statistically significant (p < 0.05) with meaningful effect size (Cohen's d > 0.5)

### 2.5 Judge Configuration

- **Model**: claude-sonnet-4.5 (via AWS Bedrock)
- **Mode**: Batch (single call returns all metrics)
- **Evaluated**: 100 human-annotated perturbations (subset for validation)

---

## 3. Results

### 3.1 Data Quality Issues

| Issue | Count | Action Taken |
|-------|-------|--------------|
| TCS > 100 (annotation error) | 0 | Capped at 100 |
| TCS = 0 (trivial error) | 36 | Excluded from analysis |
| **Final sample size** | 64 | After exclusions |
| **Exclusion rate** | 36% | Higher than expected |

### 3.2 Overall Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean CCG | **+2.246** | Universal over-penalization |
| Std CCG | 2.962 | High variance |
| Median CCG | 0.5 | Right-skewed distribution |
| Min CCG | -0.85 | Some under-penalization exists |
| Max CCG | 8.5 | Extreme over-penalization cases |
| Mean JPS | 72.1 | Judges penalize heavily |
| Mean TCS | 54.2 | Human-rated criticality |

### 3.3 Heuristic vs Human TCS Correlation

**CRITICAL FINDING**:
```
Pearson r = 0.095 (p > 0.05)
```

The heuristic TCS (based on type/position assumptions) does NOT correlate with human annotations. This means our fundamental assumption that "early = more critical" doesn't match human judgment.

### 3.4 CCG by Condition (Type x Position)

| Condition | Mean CCG | Std | n | Mean JPS | Mean TCS | Interpretation |
|-----------|----------|-----|---|----------|----------|----------------|
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

**Key observation**: Only `tool_selection_middle` shows near-perfect calibration. All other conditions show over-penalization.

### 3.5 CCG by Position

| Position | Mean CCG | Std | n | Expected | Actual |
|----------|----------|-----|---|----------|--------|
| Early | +2.46 | 3.05 | 22 | Under-penalize (< 0) | **Over-penalize** |
| Middle | +1.83 | 2.96 | 24 | Mixed | Over-penalize |
| Late | +2.54 | 2.95 | 18 | Over-penalize (> 0) | Over-penalize |

### 3.6 CCG by Perturbation Type

| Type | Mean CCG | Std | n | Category |
|------|----------|-----|---|----------|
| tool_selection | **+0.96** | 2.40 | 21 | Structural (best calibrated) |
| parameter | +2.37 | 2.96 | 20 | Surface |
| data_reference | +3.30 | 2.98 | 11 | Surface |
| planning | +3.31 | 3.32 | 12 | Structural |

### 3.7 CCG by Benchmark

| Benchmark | Mean CCG | Std | n | Notes |
|-----------|----------|-----|---|-------|
| SWE-bench | **+0.32** | 0.77 | 8 | Best calibrated |
| ToolBench | +2.48 | 3.02 | 54 | Over-penalize |
| GAIA | +3.68 | 5.41 | 2 | Worst (low n) |

### 3.8 Statistical Tests

**ANOVA Results**:

| Factor | F-statistic | p-value | eta² | Significant? |
|--------|-------------|---------|------|--------------|
| Perturbation Type | 2.47 | 0.071 | 0.110 | No (approaching) |
| Position | 0.38 | 0.685 | 0.012 | No |

**Effect Sizes (Cohen's d)**:

| Comparison | d | Magnitude |
|------------|---|-----------|
| Early vs Late | -0.03 | Negligible |
| Planning vs Parameter | +0.30 | Small |
| Structural vs Surface | -0.30 | Small |
| Early-Structural vs Late-Surface | -0.21 | Small |

**Hypothesis Test (Early-Structural vs Late-Surface)**:

| Metric | Value |
|--------|-------|
| Early-Structural Mean CCG | +2.75 (n=14) |
| Late-Surface Mean CCG | +3.42 (n=10) |
| Difference | +0.67 |
| t-statistic | -0.51 |
| p-value | 0.616 |

### 3.9 Hypothesis Assessment

| Criterion | Required | Actual | Met? |
|-----------|----------|--------|------|
| Early structural CCG < 0 | Under-penalize | **+2.75** | NO |
| Late surface CCG > 0 | Over-penalize | **+3.42** | YES |
| Direction correct (early < late) | CCG_early < CCG_late | 2.75 < 3.42 | YES |
| Statistically significant | p < 0.05 | p = 0.616 | NO |
| Meaningful effect size | \|d\| > 0.5 | d = -0.21 | NO |

**Criteria Met**: 2/5
**Support Level**: WEAK

---

## 4. Key Findings Summary

### 4.1 The Hypothesis Failed

We expected judges to under-penalize early errors. Instead, judges **over-penalize everything** uniformly.

### 4.2 Three Interesting Patterns Emerged

1. **Universal over-penalization** (CCG = +2.25): Judges are 2.2x harsher than human-rated criticality warrants

2. **tool_selection_middle is perfectly calibrated** (CCG = -0.14, std = 0.29): Why? Unknown.

3. **SWE-bench calibrates 7x better** than ToolBench (CCG = +0.32 vs +2.48): Possibly due to verifiable success criteria.

### 4.3 Critical Methodological Issue

The heuristic TCS (our assumed ground truth) correlates at **r = 0.095** with human annotations. This means:
- Our assumption that "early = critical" doesn't match human judgment
- The CCG metric is measuring something, but ground truth is questionable
- 36% of samples had TCS = 0 (humans rated as trivial)

---

## 5. Current Assessment

### 5.1 Strengths

- Novel metric (CCG) for measuring judge calibration
- Controlled perturbation methodology
- Multi-benchmark coverage
- Systematic experimental design

### 5.2 Weaknesses

- Hypothesis not supported
- Ground truth validity issue (r = 0.095)
- High exclusion rate (36%)
- Small cell sizes (some n < 5)
- Single judge only
- No statistical significance

### 5.3 First Reviewer's Opinion

The first reviewer (internal) concluded:
- **Not publishable** at ICLR/NeurIPS in current form
- **Salvageable** with pivot to "LLM Judges as Conservative Critics"
- Estimated **4-6 weeks** additional work needed

---

## 6. Questions for Second Reviewer

Please provide your expert opinion on:

### Q1: Validity Assessment
Is the CCG metric fundamentally sound, or does the r=0.095 correlation invalidate the entire approach? Can this be fixed, or should we abandon CCG?

### Q2: Finding Significance
The "universal over-penalization" finding (CCG = +2.25) emerged unexpectedly. Is this:
- A genuine contribution worth publishing?
- An artifact of methodology issues?
- Too weak without statistical significance?

### Q3: Pivot Recommendation
Which pivot (if any) has the best chance of producing a publishable paper?

**Option A**: "LLM Judges as Conservative Critics"
- Reframe around universal over-penalization
- Implications for RLHF reward modeling

**Option B**: "Benchmark-Dependent Judge Calibration"
- Focus on SWE-bench vs ToolBench difference
- Guidance for practitioners on when to trust LLM judges

**Option C**: "Error Detection vs Calibration"
- Abandon CCG, switch to binary detection analysis
- Did judge identify the perturbed step? (avoids TCS issues)

**Option D**: Abandon this direction entirely
- Cut losses, start fresh with different research question

### Q4: Minimum Viable Paper
What is the minimum additional work needed to publish at:
- A workshop (ARR, EMNLP workshop)
- A short paper venue
- A main track (ICLR/NeurIPS)

### Q5: tool_selection_middle Anomaly
The near-perfect calibration of tool_selection_middle (CCG = -0.14) is puzzling. Is this:
- Statistical noise (n=7)?
- A genuine insight worth investigating?
- Evidence that judges understand some error types better?

### Q6: TCS = 0 Problem
36% of human-annotated samples received TCS = 0 (trivial error). This is concerning. Should we:
- Re-annotate with clearer rubric?
- Accept this as valid signal (some perturbations really are trivial)?
- Analyze TCS=0 cases separately?

### Q7: Sample Size
With n=64 after exclusions (some cells n=3-5), is the statistical power sufficient to draw any conclusions? Should we:
- Run on more perturbations (all 600)?
- Add more human annotations?
- Accept limited conclusions?

---

## 7. Raw Data for Reference

### 7.1 Full CCG Distribution by Condition

```
Condition                  | n  | Mean   | Std    | Min    | Max   
---------------------------|----+--------+--------+--------+-------
planning_early             | 5  | +4.57  | 3.32   | +0.89  | +8.50
planning_middle            | 4  | +2.70  | 3.31   | -0.65  | +6.43
planning_late              | 3  | +2.03  | 3.87   | -0.85  | +6.29
tool_selection_early       | 9  | +1.74  | 3.08   | -0.76  | +7.00
tool_selection_middle      | 7  | -0.14  | 0.29   | -0.54  | +0.29
tool_selection_late        | 5  | +1.10  | 2.48   | -0.69  | +4.75
parameter_early            | 8  | +1.94  | 2.58   | -0.60  | +6.00
parameter_middle           | 7  | +2.37  | 3.49   | -0.71  | +8.20
parameter_late             | 5  | +3.06  | 3.28   | -0.13  | +7.25
data_reference_middle      | 6  | +2.91  | 3.36   | -0.40  | +7.50
data_reference_late        | 5  | +3.78  | 2.77   | +0.33  | +7.00
```

### 7.2 Comparison: Heuristic vs Human TCS

```
Condition                  | Heuristic TCS | Human TCS (mean) | Difference
---------------------------|---------------|------------------|------------
planning_early             | 90            | 14.0             | -76
planning_middle            | 70            | 42.5             | -27.5
planning_late              | 40            | 70.0             | +30
tool_selection_early       | 80            | 68.9             | -11.1
tool_selection_middle      | 60            | 87.1             | +27.1
tool_selection_late        | 30            | 64.0             | +34
parameter_early            | 50            | 57.5             | +7.5
parameter_middle           | 40            | 54.3             | +14.3
parameter_late             | 20            | 48.0             | +28
data_reference_middle      | 30            | 40.0             | +10
data_reference_late        | 20            | 30.0             | +10
```

**Observation**: Heuristic dramatically overestimated planning_early (90 vs 14) and underestimated tool_selection_middle (60 vs 87). The assumption that "early planning = most critical" appears wrong.

### 7.3 TCS = 0 Breakdown (Excluded Samples)

```
Condition                  | TCS=0 count | Total | Exclusion Rate
---------------------------|-------------|-------|---------------
planning_early             | 4           | 9     | 44%
planning_middle            | 5           | 9     | 56%
planning_late              | 6           | 9     | 67%
tool_selection_early       | 0           | 9     | 0%
tool_selection_middle      | 2           | 9     | 22%
tool_selection_late        | 4           | 9     | 44%
parameter_early            | 1           | 9     | 11%
parameter_middle           | 2           | 9     | 22%
parameter_late             | 4           | 9     | 44%
data_reference_middle      | 3           | 9     | 33%
data_reference_late        | 5           | 10    | 50%
```

**Observation**: Planning perturbations have highest TCS=0 rates (44-67%), suggesting humans often judged them as trivial. This contradicts our hypothesis that planning errors are critical.

---

## 8. Appendix: Experimental Artifacts

### Files Available
- `results/ccg_analysis/ccg_raw_claude-sonnet-4.5.csv` - Raw per-perturbation data
- `results/ccg_analysis/ccg_report_claude-sonnet-4.5.json` - Full statistical report
- `agent_tasks/05_judge_evaluation_ccg/BUG_REPORT.md` - Bug fixes applied

### Code Modules Implemented
- `src/metrics/ccg_calculator.py` - CCG computation
- `src/metrics/criticality_scorer.py` - TCS computation
- `src/judges/parallel_evaluator.py` - Parallel judge evaluation
- `src/annotation/stratified_sampler.py` - Sampling for human annotation

### What Was NOT Done
- GPT-4o evaluation (only Claude Sonnet 4.5)
- Full 600 perturbation evaluation (only 100 annotated subset)
- Outcome-based TCS validation (task success after perturbation)
- Qualitative error analysis

---

## 9. Your Response Format

Please structure your review as:

1. **Overall Assessment** (2-3 sentences)
2. **Answers to Q1-Q7** (numbered)
3. **Recommended Path Forward** (specific action items)
4. **Timeline Estimate** (to publishable paper)
5. **Risk Assessment** (what could still go wrong)

Thank you for your expert review.
