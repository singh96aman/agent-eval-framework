# 6A. Detection Metrics

**Status: REQUIREMENTS DEFINED**

Detection Metrics quantify how well LLM judges can detect, localize, and classify errors in agent trajectories. These metrics form the **detectability** side of the main research question: "Do judges detect errors better than they estimate their downstream consequence?"

---

## Design Principles

1. **Comparative**: All metrics compare judge outputs against ground truth (human labels, perturbation records)
2. **Granular**: Separate metrics for detection, localization, and type identification
3. **Stratified**: Report metrics by perturbation class, family, expected impact, and benchmark
4. **Multi-judge**: Compute for each judge model and cross-judge comparisons
5. **Statistically sound**: Report confidence intervals and significance tests

---

## 6A.1 Role in the Pipeline

```
5A. Human Labels ─────┐
                      │
5B. LLM Judge Outputs ┼──► 6A. Detection Metrics ──► 6C. Agreement + Claim
                      │
Section 3. Perturbation Records
```

Detection Metrics provide:
- Quantified judge performance on error detection
- Evidence for the "detectability" claim
- Basis for comparing detection vs consequence estimation (6C)

---

## 6A.2 Core Detection Metrics

### Perturbation Detection Rate (PDR)

**Definition:** Fraction of perturbed trajectories where the judge detects an error.

```
PDR = (1/N) × Σ 𝟙[judge detects perturbation]
```

**Computation:**
```python
def compute_pdr(judge_outputs, perturbation_records):
    correct = 0
    total = 0
    
    for unit_id, judge_output in judge_outputs.items():
        pert_record = perturbation_records[unit_id]
        
        # Skip placebo for PDR
        if pert_record["perturbation_class"] == "placebo":
            continue
        
        total += 1
        if judge_output["detection"]["error_detected"]:
            correct += 1
    
    return correct / total if total > 0 else 0
```

**Variants:**
| Variant | Description |
|---------|-------------|
| PDR_overall | Across all non-placebo perturbations |
| PDR_fine | Fine-grained perturbations only |
| PDR_coarse | Coarse-grained perturbations only |
| PDR@impact_k | Perturbations with expected_impact ≥ k |

### Placebo Non-Detection Rate (PNDR)

**Definition:** Fraction of placebo perturbations where judge correctly does NOT detect an error.

```
PNDR = (1/N_placebo) × Σ 𝟙[judge does NOT detect on placebo]
```

**Purpose:** Tests calibration — a good judge should not flag placebos.

**Target:** PNDR > 0.90 (judges should rarely flag placebos)

### Step Localization Accuracy (SLA)

**Definition:** Fraction of detected errors where the judge correctly identifies the perturbed step.

```
SLA = (1/N_detected) × Σ 𝟙[predicted_step = target_step]
```

**Computation:**
```python
def compute_sla(judge_outputs, perturbation_records):
    correct = 0
    total = 0
    
    for unit_id, judge_output in judge_outputs.items():
        pert_record = perturbation_records[unit_id]
        
        if pert_record["perturbation_class"] == "placebo":
            continue
        
        if not judge_output["detection"]["error_detected"]:
            continue  # Only count detected errors
        
        total += 1
        predicted_step = judge_output["localization"]["predicted_error_step_canonical"]
        target_step = pert_record["target_step_canonical_id"]
        
        if predicted_step == target_step:
            correct += 1
    
    return correct / total if total > 0 else 0
```

### Step Localization Accuracy with Tolerance (SLA±k)

**Definition:** Fraction where predicted step is within k steps of target (by display_step_index).

```
SLA±k = (1/N_detected) × Σ 𝟙[|predicted_step_idx - target_step_idx| ≤ k]
```

**Common variants:**
- SLA±0 = SLA (exact match)
- SLA±1 (within 1 step)
- SLA±2 (within 2 steps)

### Type Identification Accuracy (TIA)

**Definition:** Fraction of detected errors where judge correctly classifies the error type.

```
TIA = (1/N_detected) × Σ 𝟙[predicted_type = true_type]
```

**Type mapping:**
| Judge Prediction | Ground Truth (perturbation_family) |
|-----------------|-----------------------------------|
| planning | structural |
| tool_selection | tool_selection |
| parameter | parameter |
| data_reference | data_reference |
| other | any (counts as incorrect unless partial credit) |

### Critical Error Recall (CER)

**Definition:** Detection rate specifically for high-impact perturbations.

```
CER = (1/N_critical) × Σ 𝟙[judge detects AND expected_impact = 3]
```

**Purpose:** High-stakes errors are most important to detect. CER measures whether judges catch the errors that matter most.

**Target:** CER should be higher than overall PDR — judges should be *especially* good at catching critical errors.

---

## 6A.3 Derived Detection Metrics

### Detection Precision & Recall

Beyond raw rates, compute precision/recall for detection:

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Where:
- TP = Judge detects error AND perturbation is non-placebo
- FP = Judge detects error AND perturbation is placebo
- FN = Judge misses error AND perturbation is non-placebo
- TN = Judge misses error AND perturbation is placebo

### Detection ROC-AUC

Using `error_confidence` as a continuous score:

```python
from sklearn.metrics import roc_auc_score

def compute_detection_auc(judge_outputs, perturbation_records):
    y_true = []  # 1 if non-placebo, 0 if placebo
    y_score = []  # judge's error_confidence
    
    for unit_id, judge_output in judge_outputs.items():
        pert_record = perturbation_records[unit_id]
        
        y_true.append(0 if pert_record["perturbation_class"] == "placebo" else 1)
        y_score.append(judge_output["detection"]["error_confidence"])
    
    return roc_auc_score(y_true, y_score)
```

### Localization Mean Error (LME)

Average step distance for incorrect localizations:

```
LME = (1/N_incorrect) × Σ |predicted_step_idx - target_step_idx|
```

**Purpose:** Measures *how wrong* incorrect localizations are — off by 1 step is better than off by 5.

---

## 6A.4 Stratified Analysis

All metrics should be computed with stratification:

### By Perturbation Class

| Class | Expected PDR | Expected SLA |
|-------|--------------|--------------|
| Placebo | ~0 (should not detect) | N/A |
| Fine-grained | Moderate (0.4-0.7) | Variable |
| Coarse-grained | High (0.7-0.9) | Higher |

### By Perturbation Family

| Family | Description | Expected Difficulty |
|--------|-------------|---------------------|
| data_reference | Wrong value, typo | Hard to detect |
| parameter | Wrong threshold, query | Moderate |
| tool_selection | Wrong tool | Easier |
| structural | Wrong plan, skipped step | Easiest |
| terminal_flag | Premature termination | Variable |

### By Expected Detectability

| Detectability | Definition | Expected PDR |
|---------------|------------|--------------|
| 0 (invisible) | Very subtle error | Low (<0.3) |
| 1 (subtle) | Requires attention | Moderate (0.3-0.7) |
| 2 (obvious) | Clearly wrong | High (>0.7) |

### By Benchmark

| Benchmark | Domain | Notes |
|-----------|--------|-------|
| ToolBench | API tool use | Many natural errors, noisy |
| GAIA | QA with tools | Numeric extraction critical |
| SWE-bench | Code editing | Structural errors visible |

---

## 6A.5 Multi-Judge Comparison

### Per-Judge Metrics Table

| Metric | GPT-4o | Claude 3.5 | Claude 3 Opus | Llama 3.1 70B |
|--------|--------|------------|---------------|---------------|
| PDR | - | - | - | - |
| PNDR | - | - | - | - |
| SLA | - | - | - | - |
| SLA±1 | - | - | - | - |
| TIA | - | - | - | - |
| CER | - | - | - | - |
| AUC | - | - | - | - |

### Judge Agreement

Inter-judge agreement on detection:

```python
from sklearn.metrics import cohen_kappa_score

def judge_agreement(judge_a_outputs, judge_b_outputs):
    y_a = [o["detection"]["error_detected"] for o in judge_a_outputs.values()]
    y_b = [o["detection"]["error_detected"] for o in judge_b_outputs.values()]
    return cohen_kappa_score(y_a, y_b)
```

### Best Performing Judge

Identify which judge performs best on each metric:

```python
def best_judge_by_metric(all_judge_metrics):
    results = {}
    for metric in ["PDR", "SLA", "TIA", "CER"]:
        best_judge = max(all_judge_metrics, key=lambda j: j[metric])
        results[metric] = {
            "judge": best_judge["judge_model"],
            "value": best_judge[metric]
        }
    return results
```

---

## 6A.6 Comparison to Human Labels

### Human-Judge Detection Agreement

Compare judge detection to human consensus:

```python
def human_judge_agreement(judge_outputs, human_labels):
    agreements = []
    
    for unit_id in judge_outputs:
        judge_detected = judge_outputs[unit_id]["detection"]["error_detected"]
        human_detected = human_labels[unit_id]["aggregated_detectability"]["error_detected_majority"]
        
        agreements.append(judge_detected == human_detected)
    
    return sum(agreements) / len(agreements)
```

### Human as Upper Bound

Human detection rate serves as an upper bound for judge performance:

```
Human PDR vs Judge PDR → Gap indicates room for improvement
Human SLA vs Judge SLA → Localization quality gap
```

### Human-Judge Confusion Matrix

| | Human: Detected | Human: Not Detected |
|---|---|---|
| **Judge: Detected** | True Agreement | False Positive (Judge) |
| **Judge: Not Detected** | False Negative (Judge) | True Agreement (No Error) |

---

## 6A.7 Detection Metric Schema

### Per-Unit Detection Result

```json
{
  "evaluation_unit_id": "string",
  "judge_model": "string",
  
  // === Ground Truth ===
  "ground_truth": {
    "perturbation_class": "placebo | fine_grained | coarse_grained",
    "perturbation_family": "string",
    "target_step_canonical_id": "string",
    "expected_impact": "int (0-3)",
    "expected_detectability": "int (0-2)"
  },
  
  // === Judge Outputs ===
  "judge": {
    "error_detected": "bool",
    "error_confidence": "float (0-1)",
    "predicted_step_canonical_id": "string | null",
    "predicted_type": "string | null"
  },
  
  // === Detection Evaluation ===
  "detection_eval": {
    "detection_correct": "bool (true if detected non-placebo OR missed placebo)",
    "localization_correct": "bool | null",
    "localization_distance": "int | null",
    "type_correct": "bool | null"
  }
}
```

### Aggregated Detection Metrics

```json
{
  "judge_model": "string",
  "computed_at": "datetime",
  
  // === Overall Metrics ===
  "overall": {
    "pdr": "float",
    "pdr_ci": ["float", "float"],
    "pndr": "float",
    "pndr_ci": ["float", "float"],
    "sla": "float",
    "sla_ci": ["float", "float"],
    "sla_pm1": "float",
    "tia": "float",
    "cer": "float",
    "detection_auc": "float",
    "detection_precision": "float",
    "detection_recall": "float",
    "detection_f1": "float"
  },
  
  // === By Perturbation Class ===
  "by_class": {
    "fine_grained": {
      "pdr": "float",
      "sla": "float",
      "tia": "float",
      "n": "int"
    },
    "coarse_grained": {
      "pdr": "float",
      "sla": "float",
      "tia": "float",
      "n": "int"
    }
  },
  
  // === By Perturbation Family ===
  "by_family": {
    "data_reference": {"pdr": "float", "sla": "float", "n": "int"},
    "parameter": {"pdr": "float", "sla": "float", "n": "int"},
    "tool_selection": {"pdr": "float", "sla": "float", "n": "int"},
    "structural": {"pdr": "float", "sla": "float", "n": "int"},
    "terminal_flag": {"pdr": "float", "sla": "float", "n": "int"}
  },
  
  // === By Expected Detectability ===
  "by_detectability": {
    "0": {"pdr": "float", "n": "int"},
    "1": {"pdr": "float", "n": "int"},
    "2": {"pdr": "float", "n": "int"}
  },
  
  // === By Expected Impact ===
  "by_impact": {
    "0": {"pdr": "float", "n": "int"},
    "1": {"pdr": "float", "n": "int"},
    "2": {"pdr": "float", "n": "int"},
    "3": {"pdr": "float", "cer": "float", "n": "int"}
  },
  
  // === By Benchmark ===
  "by_benchmark": {
    "toolbench": {"pdr": "float", "sla": "float", "n": "int"},
    "gaia": {"pdr": "float", "sla": "float", "n": "int"},
    "swebench": {"pdr": "float", "sla": "float", "n": "int"}
  }
}
```

---

## 6A.8 Statistical Analysis

### Confidence Intervals

Compute 95% CIs using bootstrap:

```python
def bootstrap_ci(metric_values, n_bootstrap=1000, ci=0.95):
    bootstrapped = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(metric_values, size=len(metric_values), replace=True)
        bootstrapped.append(np.mean(sample))
    
    lower = np.percentile(bootstrapped, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrapped, (1 + ci) / 2 * 100)
    return (lower, upper)
```

### Significance Tests

Compare judges using McNemar's test (paired data):

```python
from statsmodels.stats.contingency_tables import mcnemar

def compare_judges(judge_a_correct, judge_b_correct):
    # Build contingency table
    # Both correct, A correct B wrong, A wrong B correct, Both wrong
    table = build_contingency(judge_a_correct, judge_b_correct)
    result = mcnemar(table, exact=True)
    return result.pvalue
```

### Effect Size

Cohen's h for proportion differences:

```python
def cohens_h(p1, p2):
    import math
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    return phi1 - phi2
```

---

## 6A.9 Visualization Requirements

### Key Figures

1. **PDR by Detectability Band**: Bar chart showing PDR for detectability 0, 1, 2
2. **SLA Distribution**: Histogram of localization distances
3. **ROC Curves**: Per-judge detection ROC curves
4. **Heatmap**: PDR by perturbation_family × expected_impact
5. **Judge Comparison**: Radar chart of all metrics per judge

### Table Formatting

| Metric | GPT-4o | Claude 3.5 | Δ | p-value |
|--------|--------|------------|---|---------|
| PDR | 0.72 ± 0.03 | 0.68 ± 0.03 | +0.04 | 0.023 |
| SLA | 0.45 ± 0.04 | 0.52 ± 0.04 | -0.07 | 0.008 |
| ... | ... | ... | ... | ... |

---

## 6A.10 Data Location

```
data/analysis/
├── detection_metrics/
│   ├── per_unit/
│   │   └── detection_results.json
│   ├── aggregated/
│   │   ├── gpt-4o_metrics.json
│   │   ├── claude-3.5-sonnet_metrics.json
│   │   └── ...
│   ├── comparisons/
│   │   ├── judge_comparison.json
│   │   └── human_judge_comparison.json
│   └── figures/
│       ├── pdr_by_detectability.png
│       ├── sla_distribution.png
│       └── ...
└── reports/
    └── detection_metrics_report.md
```

---

## 6A.11 Implementation Plan

### Module Structure

```
src/analysis/detection/
├── __init__.py
├── metrics.py           # Core metric computations
├── stratified.py        # Stratified analysis
├── comparison.py        # Multi-judge comparison
├── human_agreement.py   # Human-judge comparison
├── statistics.py        # CIs, significance tests
├── visualization.py     # Plotting functions
└── report.py            # Generate report
```

### Scripts

```bash
# Compute per-unit detection results
python -m src.analysis.detection.metrics \
  --judge-outputs data/judge_outputs/raw/ \
  --perturbation-records data/perturbed/ \
  --output data/analysis/detection_metrics/per_unit/

# Compute aggregated metrics per judge
python -m src.analysis.detection.stratified \
  --per-unit data/analysis/detection_metrics/per_unit/ \
  --output data/analysis/detection_metrics/aggregated/

# Compare judges
python -m src.analysis.detection.comparison \
  --aggregated data/analysis/detection_metrics/aggregated/ \
  --output data/analysis/detection_metrics/comparisons/

# Compare to human labels
python -m src.analysis.detection.human_agreement \
  --judge-outputs data/judge_outputs/raw/ \
  --human-labels data/evaluation_units/human_labels/aggregated/ \
  --output data/analysis/detection_metrics/comparisons/

# Generate figures
python -m src.analysis.detection.visualization \
  --metrics data/analysis/detection_metrics/ \
  --output data/analysis/detection_metrics/figures/

# Generate report
python -m src.analysis.detection.report \
  --metrics data/analysis/detection_metrics/ \
  --output data/analysis/reports/detection_metrics_report.md
```

---

## 6A.12 Dependencies

| Dependency | Section | What's Needed |
|------------|---------|---------------|
| LLM Judge Outputs (5B) | Input | Judge detection, localization, type predictions |
| Human Labels (5A) | Reference | Human detection as upper bound |
| Perturbation Records (3) | Ground truth | True perturbation location and type |
| Agreement Analysis (6C) | Output | Detection metrics feed into main claim |

---

## 6A.13 Success Criteria

Section 6A is complete when:

### Metrics Computed
1. [ ] PDR computed for all judge models
2. [ ] PNDR computed (placebo non-detection)
3. [ ] SLA and SLA±1 computed
4. [ ] TIA computed
5. [ ] CER computed for high-impact subset
6. [ ] Detection AUC computed

### Stratification
7. [ ] All metrics stratified by perturbation_class
8. [ ] All metrics stratified by perturbation_family
9. [ ] All metrics stratified by expected_detectability
10. [ ] All metrics stratified by expected_impact
11. [ ] All metrics stratified by benchmark

### Comparison
12. [ ] Multi-judge comparison table generated
13. [ ] Human-judge agreement computed
14. [ ] Significance tests run for judge differences
15. [ ] Effect sizes computed

### Visualization
16. [ ] PDR by detectability bar chart
17. [ ] SLA distribution histogram
18. [ ] ROC curves per judge
19. [ ] Heatmaps for stratified analysis

### Documentation
20. [ ] Detection metrics report generated
21. [ ] All metrics have 95% confidence intervals
22. [ ] Key findings summarized

---

## 6A.14 Expected Results (Hypotheses)

Based on the study design, expected patterns:

| Hypothesis | Expected Result | Rationale |
|------------|-----------------|-----------|
| **H1** | PDR(coarse) > PDR(fine) | Structural errors more obvious |
| **H2** | PDR(detectability=2) > PDR(detectability=0) | By design |
| **H3** | CER > PDR | Judges attend more to severe errors |
| **H4** | SLA(structural) > SLA(data_reference) | Data errors harder to pinpoint |
| **H5** | PNDR > 0.90 | Judges rarely flag placebos |
| **H6** | Human PDR > Judge PDR | Human upper bound |

These hypotheses will be tested in Section 6C (Agreement + Claim).
