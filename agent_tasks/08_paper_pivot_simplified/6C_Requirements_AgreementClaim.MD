# 6C. Agreement + Claim

**Status: REQUIREMENTS DEFINED**

This section synthesizes results from 6A (Detection) and 6B (Consequentiality) to test the main research claim and conduct inter-annotator/inter-judge agreement analysis.

---

## Main Claim

> **"LLM judges detect errors better than they estimate their downstream consequence."**

This claim has two components:
1. **Detectability**: Judges are reasonably good at detecting errors (6A metrics)
2. **Consequentiality**: Judges are poorly calibrated on impact (6B metrics)
3. **Comparison**: Detection performance exceeds consequence estimation performance

---

## Design Principles

1. **Claim-focused**: All analyses tie back to testing the main claim
2. **Multi-reference**: Compare judges to humans, outcomes, and expected values
3. **Statistically rigorous**: Formal hypothesis tests with effect sizes
4. **Transparent**: Report both supporting and contradicting evidence
5. **Actionable**: Identify when judges are trustworthy vs when they need calibration

---

## 6C.1 Role in the Pipeline

```
6A. Detection Metrics ──┐
                        ├──► 6C. Agreement + Claim ──► Final Report
6B. Consequentiality ───┘
                        │
5A. Human Labels ───────┘
```

Section 6C provides:
- Inter-annotator agreement analysis (human baseline)
- Inter-judge agreement analysis (judge consistency)
- Main claim hypothesis tests
- Synthesis of findings

---

## 6C.2 Agreement Analysis

### Human Inter-Annotator Agreement

Measure agreement among human annotators on key dimensions:

| Dimension | Type | Agreement Metric |
|-----------|------|------------------|
| error_detected | Binary | Cohen's κ, Krippendorff's α |
| error_trajectory | Categorical | Fleiss' κ, Krippendorff's α |
| error_step_id | Nominal (step matching) | Custom step-match κ |
| impact_tier | Ordinal (0-3) | Weighted κ, Krippendorff's α |
| error_type | Categorical | Krippendorff's α |

#### Agreement Interpretation

| κ / α Value | Interpretation |
|-------------|---------------|
| < 0.20 | Poor agreement |
| 0.20-0.40 | Fair agreement |
| 0.40-0.60 | Moderate agreement |
| 0.60-0.80 | Substantial agreement |
| > 0.80 | Almost perfect agreement |

#### Computing Krippendorff's Alpha

```python
import krippendorff

def compute_human_agreement(annotations_by_unit):
    """
    annotations_by_unit: {unit_id: [annotator_1_label, annotator_2_label, ...]}
    """
    # Build reliability data matrix
    reliability_data = []
    for unit_id, labels in annotations_by_unit.items():
        reliability_data.append(labels)
    
    alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='nominal')
    return alpha
```

### LLM Judge Variance

For stochastic samples (temperature=0.7, n=5), measure within-judge variance:

| Metric | Computation | Purpose |
|--------|-------------|---------|
| Detection consistency | % samples with same error_detected | Stability of detection |
| Localization stability | % samples with same predicted_step | Stability of localization |
| Impact variance | std(predicted_impact_score) | Consistency of impact estimates |

```python
def compute_judge_variance(stochastic_outputs):
    """
    stochastic_outputs: {unit_id: [sample_1, sample_2, ...]}
    """
    results = []
    
    for unit_id, samples in stochastic_outputs.items():
        detections = [s["detection"]["error_detected"] for s in samples]
        impacts = [s["impact"]["predicted_impact_score"] for s in samples]
        
        results.append({
            "unit_id": unit_id,
            "detection_consistency": max(detections.count(True), detections.count(False)) / len(detections),
            "impact_mean": np.mean(impacts),
            "impact_std": np.std(impacts)
        })
    
    return results
```

### Inter-Judge Agreement

Compare outputs across different judge models:

| Comparison | Metric | Interpretation |
|------------|--------|----------------|
| Detection agreement | Cohen's κ (pairwise) | Do judges agree on what's an error? |
| Localization agreement | Step-match rate | Do judges agree on where the error is? |
| Impact correlation | Pearson r | Do impact predictions correlate? |
| Type agreement | Krippendorff's α | Do judges agree on error type? |

```python
def compute_interjudge_agreement(judge_outputs_a, judge_outputs_b):
    # Detection agreement
    detect_a = [o["detection"]["error_detected"] for o in judge_outputs_a.values()]
    detect_b = [o["detection"]["error_detected"] for o in judge_outputs_b.values()]
    detection_kappa = cohen_kappa_score(detect_a, detect_b)
    
    # Impact correlation
    impact_a = [o["impact"]["predicted_impact_score"] for o in judge_outputs_a.values()]
    impact_b = [o["impact"]["predicted_impact_score"] for o in judge_outputs_b.values()]
    impact_corr = pearsonr(impact_a, impact_b)[0]
    
    return {
        "detection_kappa": detection_kappa,
        "impact_correlation": impact_corr
    }
```

---

## 6C.3 Main Claim Hypothesis Tests

### Hypothesis Structure

**Null Hypothesis (H0):** Judges are equally good at detection and consequence estimation.
- Detection performance ≈ Consequence calibration

**Alternative Hypothesis (H1):** Judges detect better than they estimate consequence.
- Detection performance > Consequence calibration

### Operationalization

Define a **normalized performance gap**:

```
Gap = Normalized_Detection - Normalized_Calibration
```

Where:
- `Normalized_Detection = PDR` (already 0-1)
- `Normalized_Calibration = CCorr` (already -1 to 1, but typically 0-1)

If Gap > 0 significantly, claim is supported.

### Primary Test: Detection vs Calibration

```python
from scipy import stats

def test_main_claim(detection_metrics, calibration_metrics):
    """
    Test whether PDR > CCorr for each judge.
    """
    results = []
    
    for judge in detection_metrics:
        pdr = detection_metrics[judge]["overall"]["pdr"]
        ccorr = calibration_metrics[judge]["overall"]["ccorr"]
        
        # Gap
        gap = pdr - ccorr
        
        # Bootstrap CI for gap
        gap_ci = bootstrap_gap_ci(judge_outputs, outcome_evidence)
        
        # Is gap significantly > 0?
        claim_supported = gap_ci[0] > 0  # Lower CI bound > 0
        
        results.append({
            "judge": judge,
            "pdr": pdr,
            "ccorr": ccorr,
            "gap": gap,
            "gap_ci": gap_ci,
            "claim_supported": claim_supported
        })
    
    return results
```

### Secondary Tests

#### Test 1: SLA vs |CCE|

Localization accuracy vs calibration error:
- H0: SLA = (1 - normalized_|CCE|)
- H1: SLA > (1 - normalized_|CCE|)

#### Test 2: CER vs URR

Critical error recall vs under-reaction:
- H0: CER ≈ (1 - URR)
- H1: CER > (1 - URR) — judges catch critical errors but underestimate their impact

#### Test 3: PNDR vs ORR

Placebo calibration:
- H0: PNDR ≈ (1 - ORR)
- H1: PNDR < (1 - ORR) — judges might correctly reject placebos but still over-react

---

## 6C.4 Effect Size Analysis

### Cohen's d for Detection-Calibration Gap

```python
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std
```

### Effect Size Interpretation

| d Value | Interpretation |
|---------|---------------|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |
| > 1.0 | Very large effect |

---

## 6C.5 Synthesis Table

### Main Results Summary

| Judge | PDR | CCorr | Gap | Gap CI | Claim Supported? |
|-------|-----|-------|-----|--------|------------------|
| GPT-4o | - | - | - | - | - |
| Claude 3.5 Sonnet | - | - | - | - | - |
| Claude 3 Opus | - | - | - | - | - |
| Llama 3.1 70B | - | - | - | - | - |

### Claim Support Matrix

| Evidence Type | Supports Claim | Against Claim | Notes |
|---------------|----------------|---------------|-------|
| PDR > 0.5 | ✓ | | Judges detect majority of errors |
| CCorr < 0.5 | ✓ | | Poor calibration |
| PDR > CCorr | ✓ | | Primary test |
| URR > 0 | ✓ | | Under-react to critical |
| ORR > 0 | Partial | Partial | Over-react to harmless |
| CER > PDR | ✓ | | Catch critical more than average |

---

## 6C.6 Failure Mode Analysis

### When Do Judges Fail Most?

Identify conditions where judges perform worst:

```python
def failure_mode_analysis(results):
    failure_modes = {
        "detection_failure": [],  # High miss rate
        "localization_failure": [],  # Wrong step
        "calibration_failure": [],  # Large |CCE|
        "over_reaction_failure": [],  # ORR on placebo
        "under_reaction_failure": []  # URR on critical
    }
    
    for unit in results:
        if not unit["detected"] and unit["true_impact"] == 3:
            failure_modes["detection_failure"].append(unit)
        
        if unit["detected"] and unit["localization_distance"] > 2:
            failure_modes["localization_failure"].append(unit)
        
        if abs(unit["cce"]) > 0.5:
            failure_modes["calibration_failure"].append(unit)
        
        if unit["perturbation_class"] == "placebo" and unit["predicted_impact"] > 0.5:
            failure_modes["over_reaction_failure"].append(unit)
        
        if unit["true_impact"] == 3 and unit["predicted_impact"] < 0.5:
            failure_modes["under_reaction_failure"].append(unit)
    
    return failure_modes
```

### Failure Mode Summary Table

| Failure Mode | Count | % of Total | Dominant Perturbation Family |
|--------------|-------|------------|------------------------------|
| Missed critical error | - | - | - |
| Wrong localization | - | - | - |
| Large calibration error | - | - | - |
| Over-reaction to placebo | - | - | - |
| Under-reaction to critical | - | - | - |

---

## 6C.7 Trustworthiness Guidelines

Based on results, generate guidelines for when to trust judge outputs:

### Detection Trustworthiness

| Condition | Trust Level | Rationale |
|-----------|-------------|-----------|
| Coarse-grained structural | High | High PDR |
| Obvious detectability (2) | High | By design |
| Fine-grained data reference | Low | Low PDR |
| Subtle detectability (0) | Low | Hard to detect |

### Calibration Trustworthiness

| Condition | Trust Level | Rationale |
|-----------|-------------|-----------|
| Placebo | Moderate | PNDR high, but some ORR |
| Critical errors | Low | URR indicates under-estimation |
| Moderate errors | Moderate | Better calibration in middle |
| Structural errors | Higher | More interpretable impact |

### Actionable Recommendations

1. **Use judges for detection**: Reasonably reliable for flagging errors
2. **Don't trust impact estimates**: Require human review for consequence assessment
3. **Calibrate on your domain**: CCorr varies by benchmark
4. **Ensemble for detection**: Multiple judges reduce false negatives
5. **Don't auto-reject on low impact**: Judges under-estimate critical errors

---

## 6C.8 Agreement Schema

### Human Agreement Report

```json
{
  "computed_at": "datetime",
  "n_annotators": "int",
  "n_units_with_overlap": "int",
  
  "by_dimension": {
    "error_detected": {
      "krippendorff_alpha": "float",
      "fleiss_kappa": "float",
      "pairwise_cohens_kappa": {"A-B": "float", "A-C": "float", "B-C": "float"}
    },
    "error_trajectory": {
      "krippendorff_alpha": "float",
      "pairwise_agreement_rate": "float"
    },
    "error_step_id": {
      "exact_match_rate": "float",
      "near_match_rate": "float (±1 step)"
    },
    "impact_tier": {
      "krippendorff_alpha": "float",
      "weighted_kappa": "float",
      "mean_absolute_difference": "float"
    },
    "error_type": {
      "krippendorff_alpha": "float"
    }
  },
  
  "overall_reliability": {
    "summary": "string (poor/fair/moderate/substantial/excellent)",
    "notes": "string"
  }
}
```

### Judge Variance Report

```json
{
  "computed_at": "datetime",
  "judge_model": "string",
  "n_units_sampled": "int",
  "n_samples_per_unit": "int",
  
  "variance_metrics": {
    "detection_consistency_mean": "float",
    "detection_consistency_min": "float",
    "localization_stability_rate": "float",
    "impact_std_mean": "float",
    "impact_std_max": "float"
  },
  
  "high_variance_units": [
    {
      "unit_id": "string",
      "detection_consistency": "float",
      "impact_std": "float"
    }
  ]
}
```

### Claim Test Report

```json
{
  "computed_at": "datetime",
  "claim": "LLM judges detect errors better than they estimate their downstream consequence",
  
  "primary_test": {
    "metric": "PDR - CCorr",
    "results_by_judge": [
      {
        "judge": "string",
        "pdr": "float",
        "ccorr": "float",
        "gap": "float",
        "gap_ci_lower": "float",
        "gap_ci_upper": "float",
        "claim_supported": "bool"
      }
    ],
    "overall_conclusion": "string"
  },
  
  "secondary_tests": {
    "sla_vs_cce": {
      "test_statistic": "float",
      "pvalue": "float",
      "claim_supported": "bool"
    },
    "cer_vs_urr": {
      "cer": "float",
      "urr": "float",
      "claim_supported": "bool"
    }
  },
  
  "effect_size": {
    "cohens_d": "float",
    "interpretation": "string"
  },
  
  "synthesis": {
    "evidence_for_claim": ["string list"],
    "evidence_against_claim": ["string list"],
    "overall_conclusion": "string",
    "confidence_level": "high | moderate | low"
  }
}
```

---

## 6C.9 Data Location

```
data/analysis/
├── agreement/
│   ├── human_agreement_report.json
│   ├── judge_variance/
│   │   ├── gpt-4o_variance.json
│   │   └── ...
│   └── interjudge_agreement.json
├── claim_tests/
│   ├── claim_test_report.json
│   ├── failure_mode_analysis.json
│   └── trustworthiness_guidelines.json
└── reports/
    ├── agreement_analysis_report.md
    ├── main_claim_report.md
    └── synthesis_report.md
```

---

## 6C.10 Implementation Plan

### Module Structure

```
src/analysis/agreement_claim/
├── __init__.py
├── human_agreement.py      # Inter-annotator agreement
├── judge_variance.py       # Within-judge variance
├── interjudge_agreement.py # Cross-judge agreement
├── claim_tests.py          # Hypothesis tests for main claim
├── failure_modes.py        # Failure mode analysis
├── synthesis.py            # Synthesize all findings
├── visualization.py        # Agreement heatmaps, claim figures
└── report.py               # Generate final reports
```

### Scripts

```bash
# Compute human agreement
python -m src.analysis.agreement_claim.human_agreement \
  --human-labels data/evaluation_units/human_labels/raw/ \
  --output data/analysis/agreement/human_agreement_report.json

# Compute judge variance
python -m src.analysis.agreement_claim.judge_variance \
  --stochastic-outputs data/judge_outputs/raw/single_trajectory/*/stochastic/ \
  --output data/analysis/agreement/judge_variance/

# Compute inter-judge agreement
python -m src.analysis.agreement_claim.interjudge_agreement \
  --judge-outputs data/judge_outputs/raw/single_trajectory/ \
  --output data/analysis/agreement/interjudge_agreement.json

# Run main claim tests
python -m src.analysis.agreement_claim.claim_tests \
  --detection-metrics data/analysis/detection_metrics/aggregated/ \
  --calibration-metrics data/analysis/calibration_metrics/aggregated/ \
  --output data/analysis/claim_tests/claim_test_report.json

# Failure mode analysis
python -m src.analysis.agreement_claim.failure_modes \
  --detection-results data/analysis/detection_metrics/per_unit/ \
  --calibration-results data/analysis/calibration_metrics/per_unit/ \
  --output data/analysis/claim_tests/failure_mode_analysis.json

# Generate synthesis report
python -m src.analysis.agreement_claim.synthesis \
  --all-metrics data/analysis/ \
  --output data/analysis/reports/synthesis_report.md
```

---

## 6C.11 Dependencies

| Dependency | Section | What's Needed |
|------------|---------|---------------|
| Detection Metrics (6A) | Input | PDR, SLA, TIA, CER |
| Calibration Metrics (6B) | Input | CCorr, CCE, ORR, URR |
| Human Labels (5A) | Input | Agreement analysis |
| LLM Judge Outputs (5B) | Input | Variance analysis |
| Outcome Evidence (5C) | Reference | Ground truth for validation |

---

## 6C.12 Success Criteria

Section 6C is complete when:

### Agreement Analysis
1. [ ] Human inter-annotator agreement computed for all dimensions
2. [ ] Krippendorff's α reported for binary and ordinal measures
3. [ ] Judge variance computed for stochastic samples
4. [ ] Inter-judge agreement computed for all judge pairs

### Claim Testing
5. [ ] Primary claim test (PDR vs CCorr) run for all judges
6. [ ] Bootstrap CIs computed for gap
7. [ ] Secondary tests (SLA vs CCE, CER vs URR) completed
8. [ ] Effect sizes computed

### Synthesis
9. [ ] Evidence summary table completed
10. [ ] Claim support determination for each judge
11. [ ] Overall claim conclusion written
12. [ ] Confidence level assigned

### Failure Analysis
13. [ ] Failure modes identified and quantified
14. [ ] Trustworthiness guidelines generated
15. [ ] Actionable recommendations documented

### Documentation
16. [ ] Agreement analysis report generated
17. [ ] Main claim report generated
18. [ ] Synthesis report generated

---

## 6C.13 Expected Results & Hypotheses

### Expected Findings

| Finding | Expected Value | Confidence |
|---------|----------------|------------|
| Human α (detection) | > 0.6 | High |
| Human α (impact) | 0.4-0.6 | Moderate |
| Judge variance (detection) | > 90% consistent | High |
| Judge variance (impact) | σ < 0.15 | Moderate |
| PDR - CCorr gap | > 0.2 | Main claim |
| Gap CI excludes 0 | True | For claim support |

### If Claim Is Supported

The paper concludes:
- LLM judges are useful for **flagging potential errors**
- LLM judges should **not** be trusted for **consequence assessment**
- Impact predictions require **calibration** or **human review**
- Recommend using judges as **first-pass filters**, not final arbiters

### If Claim Is Not Supported

Alternative conclusions:
- Perhaps judges are uniformly bad (poor detection AND poor calibration)
- Perhaps judges are uniformly good (good detection AND good calibration)
- Need to identify conditions where each holds

---

## 6C.14 Optional: RLAIF Pilot

If the main claim is supported, consider a follow-on study:

### Pilot Design

1. **Identify validated judge outputs**: Units where judge detection is correct
2. **Use for RL feedback**: Train agent on judge-flagged errors
3. **Measure**: Does training on judge feedback improve agent performance?

### Metrics for Pilot

| Metric | Definition |
|--------|------------|
| Agent improvement | Task success rate post-training vs pre-training |
| Error reduction | Rate of errors in trajectories post-training |
| Judge alignment | Do agent outputs align with judge preferences? |

This is explicitly optional and depends on main study results.

---

## 6C.15 Visualization Requirements

### Key Figures

1. **Agreement Heatmap**: Annotator × Annotator κ values
2. **Claim Support Bar Chart**: PDR vs CCorr per judge with CI error bars
3. **Gap Distribution**: Bootstrap distribution of PDR - CCorr
4. **Failure Mode Pie Chart**: Distribution of failure types
5. **Trustworthiness Matrix**: Condition × Trust Level heatmap

### Claim Figure Template

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Detection vs Calibration                        │
│                                                                     │
│  PDR  ████████████████████████████ 0.72                            │
│       [──────────────CI──────────────]                              │
│                                                                     │
│  CCorr ██████████████ 0.38                                          │
│       [────CI────]                                                  │
│                                                                     │
│  Gap                   ████████████ 0.34 *                          │
│                        [───CI───]                                   │
│                                                                     │
│  * Gap CI excludes 0 → Claim supported                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6C.16 Final Report Structure

### Suggested Paper Outline (from this study)

1. **Introduction**: LLM-as-judge for agent trajectories
2. **Related Work**: Evaluation methods, calibration studies
3. **Method**:
   - Trajectory sources (Section 1)
   - Typed representation (Section 2)
   - Controlled perturbations (Section 3)
   - Evaluation framework (Section 4)
4. **Experiments**:
   - Human annotation (5A)
   - Judge evaluation (5B)
   - Outcome evidence (5C)
5. **Results**:
   - Detection metrics (6A)
   - Calibration metrics (6B)
   - Main claim analysis (6C)
6. **Discussion**: When to trust judges, limitations, future work
7. **Conclusion**: Main claim, recommendations

The synthesis report from 6C feeds directly into Sections 5-6 of this outline.
