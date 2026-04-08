# 6B. Consequentiality + Calibration

**Status: REQUIREMENTS DEFINED**

Consequentiality metrics measure how well LLM judges estimate the downstream impact of errors. These metrics form the **calibration** side of the main research question: "Do judges detect errors better than they estimate their downstream consequence?"

---

## Design Principles

1. **Grounded in outcomes**: Calibration is measured against actual outcome degradation (OD), not just expected impact
2. **Directional**: Distinguish over-reaction (penalizing harmless errors) from under-reaction (missing critical errors)
3. **Multi-source**: Compare judge predictions to outcome evidence (5C), human impact ratings (5A), and expected impact (3)
4. **Calibration-aware**: Use proper calibration metrics (ECE) not just correlation
5. **Tied to main claim**: Metrics directly test whether judges estimate consequences poorly

---

## 6B.1 Role in the Pipeline

```
5A. Human Labels ─────┐
                      │
5B. LLM Judge Outputs ┼──► 6B. Consequentiality ──► 6C. Agreement + Claim
                      │
5C. Outcome Evidence ─┘
```

Consequentiality Metrics provide:
- Quantified judge calibration on impact prediction
- Evidence for the "consequence estimation" claim
- Measures of over-reaction and under-reaction
- Failure prediction calibration (ECE)

---

## 6B.2 Ground Truth for Consequence

### Three Sources of "True Impact"

| Source | Definition | Available For |
|--------|------------|---------------|
| **Outcome Degradation (OD)** | `Score(baseline) - Score(perturbed)` | All units with outcome evidence |
| **Human Impact Rating** | `aggregated_consequence.mean_impact_tier` | Human-labeled subset |
| **Expected Impact** | `perturbation_record.expected_impact` | All units (but is a prediction) |

**Primary ground truth:** OD (outcome evidence) is the most objective measure.

**Secondary ground truth:** Human impact ratings provide a perception-based reference.

**Validation reference:** Expected impact validates the perturbation generation process.

### True Impact Binning

For discrete comparison, bin continuous OD into impact levels:

| Impact Level | OD Range | Binary |
|--------------|----------|--------|
| 0 (None) | OD = 0 | 0 |
| 1 (Minor) | 0 < OD ≤ 0.25 | 1 |
| 2 (Moderate) | 0.25 < OD ≤ 0.5 | 1 |
| 3 (Critical) | OD > 0.5 | 1 |

---

## 6B.3 Core Consequentiality Metrics

### Consequence Correlation (CCorr)

**Definition:** Spearman correlation between judge's predicted impact and actual outcome degradation.

```
CCorr = Spearman(predicted_impact_score, OD)
```

**Computation:**
```python
from scipy.stats import spearmanr

def compute_ccorr(judge_outputs, outcome_evidence):
    predicted = []
    actual = []
    
    for unit_id in judge_outputs:
        if unit_id not in outcome_evidence:
            continue
        
        predicted.append(judge_outputs[unit_id]["impact"]["predicted_impact_score"])
        actual.append(outcome_evidence[unit_id]["metrics"]["outcome_degradation"])
    
    corr, pvalue = spearmanr(predicted, actual)
    return {"ccorr": corr, "pvalue": pvalue}
```

**Interpretation:**
| CCorr Value | Interpretation |
|-------------|---------------|
| > 0.7 | Strong calibration |
| 0.4-0.7 | Moderate calibration |
| 0.2-0.4 | Weak calibration |
| < 0.2 | Poor calibration |

### Consequence Calibration Error (CCE)

**Definition:** Signed difference between predicted and true impact.

```
CCEᵢ = predicted_impact - true_impact
```

**Aggregate forms:**
- **Mean CCE**: `(1/N) × Σ CCEᵢ` — systematic bias direction
- **|CCE|**: `(1/N) × Σ |CCEᵢ|` — mean absolute error

**Interpretation:**
| CCE Sign | Meaning |
|----------|---------|
| CCE > 0 | Over-estimation (judge thinks worse than reality) |
| CCE < 0 | Under-estimation (judge thinks better than reality) |
| CCE ≈ 0 | Unbiased (on average) |

### Over-Reaction Rate (ORR)

**Definition:** Rate at which judges assign high penalties to low-impact or placebo perturbations.

```
ORR = P(predicted_impact_score > 0.5 | true_impact ≤ 1)
```

**Computation:**
```python
def compute_orr(judge_outputs, outcome_evidence, perturbation_records):
    over_reactions = 0
    low_impact_total = 0
    
    for unit_id in judge_outputs:
        pert = perturbation_records[unit_id]
        outcome = outcome_evidence.get(unit_id)
        
        # Low-impact: placebo OR true_impact ≤ 1
        is_low_impact = (
            pert["perturbation_class"] == "placebo" or
            (outcome and derive_impact_level(outcome["metrics"]["outcome_degradation"]) <= 1)
        )
        
        if not is_low_impact:
            continue
        
        low_impact_total += 1
        judge_impact = judge_outputs[unit_id]["impact"]["predicted_impact_score"]
        
        if judge_impact > 0.5:
            over_reactions += 1
    
    return over_reactions / low_impact_total if low_impact_total > 0 else 0
```

**Target:** ORR < 0.15 — judges should rarely over-penalize harmless errors.

### Under-Reaction Rate (URR)

**Definition:** Rate at which judges assign low penalties to critical (high-impact) perturbations.

```
URR = P(predicted_impact_score < 0.5 | true_impact = 3)
```

**Computation:**
```python
def compute_urr(judge_outputs, outcome_evidence):
    under_reactions = 0
    critical_total = 0
    
    for unit_id in judge_outputs:
        outcome = outcome_evidence.get(unit_id)
        if not outcome:
            continue
        
        true_impact = derive_impact_level(outcome["metrics"]["outcome_degradation"])
        
        if true_impact != 3:  # Only count critical
            continue
        
        critical_total += 1
        judge_impact = judge_outputs[unit_id]["impact"]["predicted_impact_score"]
        
        if judge_impact < 0.5:
            under_reactions += 1
    
    return under_reactions / critical_total if critical_total > 0 else 0
```

**Target:** URR < 0.10 — judges should almost always flag critical errors as severe.

---

## 6B.4 Failure Prediction Calibration

### Failure-ECE (Expected Calibration Error)

**Definition:** Measures how well `predicted_failure_prob` matches actual failure rate.

```
Failure-ECE = Σ (|bin_size|/N) × |accuracy(bin) - confidence(bin)|
```

**Computation:**
```python
def compute_failure_ece(judge_outputs, outcome_evidence, n_bins=10):
    # Collect predictions and outcomes
    predictions = []
    actuals = []
    
    for unit_id in judge_outputs:
        outcome = outcome_evidence.get(unit_id)
        if not outcome:
            continue
        
        pred_prob = judge_outputs[unit_id]["impact"]["predicted_failure_prob"]
        actual_fail = outcome["perturbed"]["outcome_binary"] == False
        
        predictions.append(pred_prob)
        actuals.append(int(actual_fail))
    
    # Bin by predicted probability
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    
    for i in range(n_bins):
        bin_mask = (predictions >= bins[i]) & (predictions < bins[i+1])
        if bin_mask.sum() == 0:
            continue
        
        bin_acc = np.mean([actuals[j] for j in np.where(bin_mask)[0]])
        bin_conf = np.mean([predictions[j] for j in np.where(bin_mask)[0]])
        bin_size = bin_mask.sum()
        
        ece += (bin_size / len(predictions)) * abs(bin_acc - bin_conf)
    
    return ece
```

**Interpretation:**
| ECE Value | Interpretation |
|-----------|---------------|
| < 0.05 | Well calibrated |
| 0.05-0.10 | Moderately calibrated |
| 0.10-0.20 | Poorly calibrated |
| > 0.20 | Very poorly calibrated |

### Reliability Diagram

Plot for visualizing calibration:
- X-axis: Predicted failure probability (binned)
- Y-axis: Actual failure rate
- Perfect calibration: diagonal line

---

## 6B.5 Impact Tier Agreement

### Human-Judge Impact Agreement

Compare judge impact to human impact rating:

```python
from sklearn.metrics import cohen_kappa_score

def impact_agreement(judge_outputs, human_labels):
    # Convert judge continuous score to 0-3 tier
    judge_tiers = []
    human_tiers = []
    
    for unit_id in human_labels:
        if unit_id not in judge_outputs:
            continue
        
        judge_score = judge_outputs[unit_id]["impact"]["predicted_impact_score"]
        judge_tier = score_to_tier(judge_score)  # 0-0.25→0, 0.25-0.5→1, etc.
        judge_tiers.append(judge_tier)
        
        human_tier = round(human_labels[unit_id]["aggregated_consequence"]["mean_impact_tier"])
        human_tiers.append(human_tier)
    
    return cohen_kappa_score(judge_tiers, human_tiers, weights='quadratic')
```

**Expected:** Weighted kappa > 0.4 indicates moderate agreement on impact tiers.

### Judge-Outcome Impact Agreement

Compare judge impact to outcome-derived impact:

```python
def outcome_agreement(judge_outputs, outcome_evidence):
    judge_tiers = []
    outcome_tiers = []
    
    for unit_id in outcome_evidence:
        if unit_id not in judge_outputs:
            continue
        
        judge_score = judge_outputs[unit_id]["impact"]["predicted_impact_score"]
        judge_tiers.append(score_to_tier(judge_score))
        
        od = outcome_evidence[unit_id]["metrics"]["outcome_degradation"]
        outcome_tiers.append(derive_impact_level(od))
    
    return cohen_kappa_score(judge_tiers, outcome_tiers, weights='quadratic')
```

---

## 6B.6 Derived Calibration Metrics

### Impact Prediction MSE

Mean squared error between predicted and true impact:

```
MSE = (1/N) × Σ (predicted_impact_score - normalized_OD)²
```

Where `normalized_OD = OD` (already 0-1 range).

### Impact RMSE

```
RMSE = √MSE
```

### Impact MAE

```
MAE = (1/N) × Σ |predicted_impact_score - normalized_OD|
```

### Directional Accuracy

Fraction where predicted impact direction matches true impact direction:

```
DA = P(sign(predicted - 0.5) = sign(OD - 0.25))
```

---

## 6B.7 Stratified Calibration Analysis

### By True Impact Level

| True Impact | n | Mean Predicted | CCorr | ORR/URR |
|-------------|---|----------------|-------|---------|
| 0 (None) | - | - | - | ORR |
| 1 (Minor) | - | - | - | ORR |
| 2 (Moderate) | - | - | - | - |
| 3 (Critical) | - | - | - | URR |

### By Perturbation Class

| Class | n | CCorr | |CCE| | ORR | URR |
|-------|---|-------|------|-----|-----|
| Placebo | - | N/A | - | - | N/A |
| Fine-grained | - | - | - | - | - |
| Coarse-grained | - | - | - | - | - |

### By Perturbation Family

| Family | n | CCorr | |CCE| |
|--------|---|-------|------|
| data_reference | - | - | - |
| parameter | - | - | - |
| tool_selection | - | - | - |
| structural | - | - | - |
| terminal_flag | - | - | - |

### By Benchmark

| Benchmark | n | CCorr | |CCE| | ORR | URR |
|-----------|---|-------|------|-----|-----|
| ToolBench | - | - | - | - | - |
| GAIA | - | - | - | - | - |
| SWE-bench | - | - | - | - | - |

---

## 6B.8 Consequentiality Metric Schema

### Per-Unit Calibration Result

```json
{
  "evaluation_unit_id": "string",
  "judge_model": "string",
  
  // === Ground Truth ===
  "ground_truth": {
    "outcome_degradation": "float",
    "true_impact_level": "int (0-3)",
    "human_impact_rating": "float | null",
    "expected_impact": "int (0-3)"
  },
  
  // === Judge Predictions ===
  "judge": {
    "predicted_impact_score": "float (0-1)",
    "predicted_impact_level": "int (0-3)",
    "predicted_failure_prob": "float (0-1)"
  },
  
  // === Calibration Evaluation ===
  "calibration_eval": {
    "cce": "float (signed error)",
    "abs_cce": "float (absolute error)",
    "over_reaction": "bool (predicted high, true low)",
    "under_reaction": "bool (predicted low, true high)",
    "failure_correct": "bool (predicted fail matches actual)"
  }
}
```

### Aggregated Calibration Metrics

```json
{
  "judge_model": "string",
  "computed_at": "datetime",
  
  // === Overall Metrics ===
  "overall": {
    "ccorr": "float",
    "ccorr_pvalue": "float",
    "ccorr_ci": ["float", "float"],
    "mean_cce": "float",
    "abs_cce": "float",
    "orr": "float",
    "urr": "float",
    "failure_ece": "float",
    "impact_mse": "float",
    "impact_rmse": "float",
    "impact_mae": "float"
  },
  
  // === Human Comparison ===
  "human_comparison": {
    "impact_kappa": "float",
    "impact_correlation": "float",
    "n_compared": "int"
  },
  
  // === By True Impact Level ===
  "by_true_impact": {
    "0": {"n": "int", "mean_predicted": "float", "orr": "float"},
    "1": {"n": "int", "mean_predicted": "float", "orr": "float"},
    "2": {"n": "int", "mean_predicted": "float"},
    "3": {"n": "int", "mean_predicted": "float", "urr": "float"}
  },
  
  // === By Perturbation Class ===
  "by_class": {
    "placebo": {"n": "int", "mean_predicted": "float", "orr": "float"},
    "fine_grained": {"n": "int", "ccorr": "float", "abs_cce": "float"},
    "coarse_grained": {"n": "int", "ccorr": "float", "abs_cce": "float"}
  },
  
  // === By Perturbation Family ===
  "by_family": {
    "data_reference": {"n": "int", "ccorr": "float", "abs_cce": "float"},
    "parameter": {"n": "int", "ccorr": "float", "abs_cce": "float"},
    "tool_selection": {"n": "int", "ccorr": "float", "abs_cce": "float"},
    "structural": {"n": "int", "ccorr": "float", "abs_cce": "float"},
    "terminal_flag": {"n": "int", "ccorr": "float", "abs_cce": "float"}
  },
  
  // === By Benchmark ===
  "by_benchmark": {
    "toolbench": {"n": "int", "ccorr": "float", "abs_cce": "float", "orr": "float", "urr": "float"},
    "gaia": {"n": "int", "ccorr": "float", "abs_cce": "float", "orr": "float", "urr": "float"},
    "swebench": {"n": "int", "ccorr": "float", "abs_cce": "float", "orr": "float", "urr": "float"}
  }
}
```

---

## 6B.9 Statistical Analysis

### Confidence Intervals for Correlation

Fisher z-transformation for correlation CI:

```python
def correlation_ci(r, n, confidence=0.95):
    import math
    from scipy import stats
    
    # Fisher z-transformation
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1 / math.sqrt(n - 3)
    
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    
    z_lower = z - z_crit * se
    z_upper = z + z_crit * se
    
    # Back-transform
    r_lower = (math.exp(2 * z_lower) - 1) / (math.exp(2 * z_lower) + 1)
    r_upper = (math.exp(2 * z_upper) - 1) / (math.exp(2 * z_upper) + 1)
    
    return (r_lower, r_upper)
```

### Testing CCorr Against Zero

```python
from scipy.stats import spearmanr

corr, pvalue = spearmanr(predicted, actual)
# H0: correlation = 0
# Reject if pvalue < 0.05
```

### Testing ORR/URR Against Thresholds

Binomial test for rate against threshold:

```python
from scipy.stats import binom_test

def test_rate_against_threshold(rate, n, threshold, alternative='greater'):
    k = int(rate * n)
    return binom_test(k, n, threshold, alternative=alternative)
```

---

## 6B.10 Visualization Requirements

### Key Figures

1. **Reliability Diagram**: Predicted failure prob vs actual failure rate
2. **Calibration Scatter**: Predicted impact vs OD (with regression line)
3. **Error Distribution**: Histogram of CCE values
4. **ORR/URR by Judge**: Bar chart comparing judges
5. **Impact Confusion Matrix**: Predicted tier vs true tier heatmap

### Reliability Diagram

```python
def plot_reliability_diagram(predictions, actuals, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accuracies = []
    
    for i in range(n_bins):
        mask = (predictions >= bins[i]) & (predictions < bins[i+1])
        if mask.sum() > 0:
            bin_accuracies.append(np.mean(actuals[mask]))
        else:
            bin_accuracies.append(np.nan)
    
    plt.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.xlabel('Predicted Failure Probability')
    plt.ylabel('Actual Failure Rate')
    plt.title('Reliability Diagram')
```

---

## 6B.11 Data Location

```
data/analysis/
├── calibration_metrics/
│   ├── per_unit/
│   │   └── calibration_results.json
│   ├── aggregated/
│   │   ├── gpt-4o_calibration.json
│   │   ├── claude-3.5-sonnet_calibration.json
│   │   └── ...
│   ├── comparisons/
│   │   ├── judge_calibration_comparison.json
│   │   └── human_judge_calibration.json
│   └── figures/
│       ├── reliability_diagram_gpt4o.png
│       ├── calibration_scatter.png
│       ├── cce_distribution.png
│       └── ...
└── reports/
    └── calibration_metrics_report.md
```

---

## 6B.12 Implementation Plan

### Module Structure

```
src/analysis/calibration/
├── __init__.py
├── metrics.py              # CCorr, CCE, ORR, URR computation
├── failure_calibration.py  # Failure-ECE computation
├── stratified.py           # Stratified analysis
├── human_comparison.py     # Human-judge impact agreement
├── statistics.py           # CIs, significance tests
├── visualization.py        # Reliability diagrams, scatter plots
└── report.py               # Generate report
```

### Scripts

```bash
# Compute per-unit calibration results
python -m src.analysis.calibration.metrics \
  --judge-outputs data/judge_outputs/raw/ \
  --outcome-evidence data/outcome_evidence/aggregated/ \
  --perturbation-records data/perturbed/ \
  --output data/analysis/calibration_metrics/per_unit/

# Compute aggregated metrics per judge
python -m src.analysis.calibration.stratified \
  --per-unit data/analysis/calibration_metrics/per_unit/ \
  --output data/analysis/calibration_metrics/aggregated/

# Compare to human labels
python -m src.analysis.calibration.human_comparison \
  --judge-outputs data/judge_outputs/raw/ \
  --human-labels data/evaluation_units/human_labels/aggregated/ \
  --output data/analysis/calibration_metrics/comparisons/

# Generate figures
python -m src.analysis.calibration.visualization \
  --metrics data/analysis/calibration_metrics/ \
  --output data/analysis/calibration_metrics/figures/

# Generate report
python -m src.analysis.calibration.report \
  --metrics data/analysis/calibration_metrics/ \
  --output data/analysis/reports/calibration_metrics_report.md
```

---

## 6B.13 Dependencies

| Dependency | Section | What's Needed |
|------------|---------|---------------|
| LLM Judge Outputs (5B) | Input | Impact predictions, failure probabilities |
| Outcome Evidence (5C) | Ground truth | OD for calibration |
| Human Labels (5A) | Reference | Human impact ratings |
| Perturbation Records (3) | Metadata | Expected impact, perturbation type |
| Agreement Analysis (6C) | Output | Calibration metrics feed into main claim |

---

## 6B.14 Success Criteria

Section 6B is complete when:

### Metrics Computed
1. [ ] CCorr computed for all judge models
2. [ ] Mean CCE and |CCE| computed
3. [ ] ORR computed (over-reaction rate)
4. [ ] URR computed (under-reaction rate)
5. [ ] Failure-ECE computed

### Stratification
6. [ ] All metrics stratified by true_impact_level
7. [ ] All metrics stratified by perturbation_class
8. [ ] All metrics stratified by perturbation_family
9. [ ] All metrics stratified by benchmark

### Comparison
10. [ ] Human-judge impact agreement computed
11. [ ] Multi-judge calibration comparison generated
12. [ ] Significance tests run for differences

### Visualization
13. [ ] Reliability diagrams generated per judge
14. [ ] Calibration scatter plots generated
15. [ ] CCE distribution histograms generated

### Documentation
16. [ ] Calibration metrics report generated
17. [ ] All metrics have confidence intervals
18. [ ] Key findings summarized

---

## 6B.15 Expected Results (Hypotheses)

Based on the study design, expected patterns:

| Hypothesis | Expected Result | Rationale |
|------------|-----------------|-----------|
| **H1** | CCorr < 0.5 | Judges poorly calibrated on impact |
| **H2** | ORR > 0.2 on placebo | Over-penalize harmless errors |
| **H3** | URR > 0.1 on critical | Miss some critical errors |
| **H4** | Failure-ECE > 0.15 | Poorly calibrated failure prediction |
| **H5** | Mean CCE > 0 | Systematic over-estimation |
| **H6** | CCorr(human) > CCorr(judge) | Humans better calibrated |

These hypotheses inform the main claim in Section 6C.

---

## 6B.16 Connection to Main Claim

The main claim is:

> "LLM judges detect errors better than they estimate their downstream consequence."

Section 6B provides evidence via:

| Detection (6A) | Consequence (6B) | Comparison |
|----------------|------------------|------------|
| PDR | CCorr | PDR high, CCorr low → claim supported |
| SLA | |CCE| | SLA high, |CCE| high → claim supported |
| CER | URR | CER high, URR > 0 → judges detect but underestimate |

The final comparison happens in Section 6C.
