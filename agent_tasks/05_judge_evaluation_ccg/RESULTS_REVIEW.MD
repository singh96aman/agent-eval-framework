# Expert Review: CCG Research Results

**Reviewer perspective**: ICLR/NeurIPS program committee member  
**Date**: 2026-04-04  
**Verdict**: NOT READY for top venue in current form, but SALVAGEABLE with pivot

---

## 1. Summary of Findings

### Original Hypothesis
> LLM judges under-penalize early structural failures (planning, tool_selection) while over-penalizing late surface mistakes (parameter, data_reference).

### Actual Results
- **Universal over-penalization**: Mean CCG = +2.25 (judges penalize ~2.2x more than warranted)
- **No position effect**: p = 0.685, eta^2 = 0.012 (negligible)
- **Weak type effect**: p = 0.071, eta^2 = 0.110 (approaching significance but not there)
- **Hypothesis criteria met**: 2/5 (WEAK support)

---

## 2. Critical Question: Is There a Strong Paper Here?

### The Honest Answer: No, Not in Current Form

**Why the original hypothesis fails:**

| Issue | Severity | Impact |
|-------|----------|--------|
| Heuristic TCS correlation r=0.095 | **FATAL** | The entire CCG metric is built on TCS. If heuristic TCS doesn't correlate with human judgment, the metric measures noise. |
| 36% data exclusion (TCS=0) | HIGH | Excluding 36/100 samples biases the analysis. What are these "trivial" errors? |
| No statistical significance | HIGH | p=0.685 for position means we cannot claim position matters |
| Single judge only | MODERATE | No generalization claim possible with n=1 judges |
| Small cell sizes (n=3-5) | MODERATE | Cannot trust condition-level means |

### The r=0.095 Problem is Fatal

This is the core issue. The paper proposes CCG = (JPS - TCS) / TCS as a measure of miscalibration. But:

- **Heuristic TCS** was assigned based on the assumption that early errors are more critical
- **Human TCS** (the ground truth) correlates at r=0.095 with heuristic
- This means **the fundamental assumption is wrong**

In other words: the research assumed "early planning errors cascade" but human annotators didn't agree. The heuristic isn't measuring what we thought it measures.

---

## 3. What's Actually Interesting in the Data?

Despite the hypothesis failure, there are **three genuinely interesting findings**:

### Finding 1: Universal Over-Penalization (+2.25 CCG)

```
Mean JPS = 72.1  (judges penalize this much)
Mean TCS = 54.2  (human-rated criticality)
Ratio   = 1.33x  (33% harsher than warranted)
Mean CCG = +2.25 (when normalized by TCS)
```

**Interpretation**: LLM judges act as **harsh critics** - they penalize errors more than humans think warranted. This is the *opposite* of the concern that judges miss errors.

**Why this matters**: RLHF reward models trained on LLM judge feedback may be overly conservative, penalizing agents for minor deviations.

### Finding 2: tool_selection_middle is Perfectly Calibrated

```
Condition: tool_selection_middle
CCG = -0.14 (nearly perfect)
Std = 0.29 (very low variance)
n = 7
```

**Question worth exploring**: Why does the judge understand tool selection errors in the middle of trajectories so well? Is it because:
- Tool selection has clear right/wrong semantics?
- Middle position provides sufficient context?
- Something about how Claude evaluates tool calls?

### Finding 3: Benchmark-Specific Calibration

```
SWE-bench: CCG = +0.32 (well-calibrated)
ToolBench: CCG = +2.48 (over-penalize 7x more)
GAIA:      CCG = +3.68 (worst, but n=2)
```

**Hypothesis**: Judges calibrate better when tasks have concrete, verifiable success criteria (SWE-bench: does the code pass tests?) versus open-ended tasks (ToolBench: was the API call appropriate?).

---

## 4. Pivot Recommendations

### Option A: "LLM Judges as Conservative Critics" (RECOMMENDED)

**New thesis**: LLM judges don't miss critical errors - they systematically over-penalize all errors. This conservatism has implications for RLHF reward modeling.

**Required work**:
1. **Fix TCS methodology** - Either:
   - Use only human TCS (need 600 annotations, not 100)
   - Derive TCS from task outcome (did perturbed trajectory fail?)
   - Use pairwise comparison (is error A worse than B?) instead of absolute scores

2. **Add second judge** (GPT-4o) - Show over-penalization is consistent across judges

3. **Quantify implications** - What happens to RLHF when reward model over-penalizes by 2x?

**Contribution claim**: First systematic quantification of LLM judge conservatism in agent evaluation.

### Option B: "Benchmark-Dependent Judge Calibration"

**New thesis**: LLM judge reliability varies dramatically by benchmark type. Verifiable tasks (SWE-bench) yield well-calibrated judgments; open-ended tasks (ToolBench, GAIA) yield over-penalization.

**Required work**:
1. More samples from SWE-bench and GAIA (currently n=8 and n=2)
2. Characterize what makes a benchmark "verifiable"
3. Practical guidance for practitioners on when to trust LLM judges

### Option C: Abandon CCG, Focus on Error Detection

**New thesis**: Instead of calibration (do judges penalize the right amount?), focus on detection (do judges even notice errors?).

**Required work**:
1. Binary analysis: did judge identify the perturbed step?
2. Compare judge "critical_issues" to actual perturbation
3. Measure false positive / false negative rates

This sidesteps the TCS problem entirely.

---

## 5. What Would Make This ICLR-Ready?

For a top venue, you need:

| Criterion | Current | Required | Gap |
|-----------|---------|----------|-----|
| Novel contribution | Yes (CCG metric) | Yes | CCG needs validation |
| Statistical significance | p=0.685 | p<0.05 | Need stronger effect or more power |
| Effect size | d=-0.21 | d>0.5 | Need 2x larger effect |
| Ground truth validity | r=0.095 | r>0.6 | **Major redesign** |
| Generalization | 1 judge | 2+ judges | Add GPT-4o |
| Sample size | n=64 (after exclusion) | n>200 | Run on more perturbations |
| Actionable insight | Unclear | Clear | Pivot to "conservative critic" |

### Minimum viable paper (workshop/short paper)
- Fix TCS methodology (even if smaller scale)
- Show consistent over-penalization across 2 judges
- Clear "so what" for practitioners

### Strong paper (main track)
- All of the above, plus
- Formal analysis of why judges over-penalize
- Intervention study: can we calibrate judges with better prompts?
- Connection to RLHF reward modeling

---

## 6. Specific Recommendations

### Immediate (before any more experiments)

1. **Understand TCS=0 cases**: Why did 36% of samples get TCS=0? Read a sample of them. Are these truly trivial, or did annotators misunderstand the task?

2. **Understand r=0.095**: Plot heuristic TCS vs human TCS. Where do they disagree? This reveals whether the heuristic assumption (early=critical) is wrong or annotation is noisy.

3. **Read tool_selection_middle cases**: Why are these well-calibrated? What can we learn?

### Short-term (1-2 weeks)

4. **Add GPT-4o judge**: Run on same 100 samples. If it also over-penalizes, that's a robust finding.

5. **Compute detection metrics**: Instead of CCG, measure: did the judge flag the perturbed step in "critical_issues"? This is binary and avoids TCS issues.

6. **Qualitative analysis**: Sample 20 cases. For each, read the perturbation, the judge output, and human TCS. Where do disagreements come from?

### Medium-term (1-2 months)

7. **Outcome-based TCS**: For perturbations where we know task outcome (success/fail), use that as ground truth. Avoids human annotation issues.

8. **Calibration intervention**: Try adding "calibration examples" to judge prompt. Can we reduce over-penalization?

9. **RLHF simulation**: What happens to a toy reward model trained on over-penalizing judge feedback?

---

## 7. Final Verdict

### Current paper status: **NOT PUBLISHABLE** at ICLR/NeurIPS

**Reasons**:
- Hypothesis not supported
- Fundamental validity issue (TCS correlation)
- No statistical significance
- Single judge, small samples

### Salvageability: **HIGH** with pivot

**Path forward**:
1. Reframe around "conservative critic" finding
2. Fix ground truth methodology
3. Add second judge
4. Focus on actionable practitioner guidance

### Estimated effort to ICLR-ready: **4-6 weeks** of additional work

---

## 8. One-Line Takeaways

| Audience | Takeaway |
|----------|----------|
| Paper authors | Your hypothesis was wrong, but you found something more interesting: judges are harsh, not lenient |
| Practitioners | Don't worry about judges missing errors; worry about them being too harsh |
| RLHF researchers | Over-penalizing judges may lead to overly conservative agents |
| Future researchers | CCG is a useful metric but needs validated ground truth |

---

## Appendix: Key Numbers to Remember

```
Mean CCG = +2.25 (universal over-penalization)
Position effect: p=0.685 (not significant)
Type effect: p=0.071 (approaching, not there)
Heuristic-Human TCS correlation: r=0.095 (FAILED)
Best calibrated: tool_selection_middle (CCG=-0.14)
Best calibrated benchmark: SWE-bench (CCG=+0.32)
Worst calibrated: planning_early (CCG=+4.57)
Exclusion rate: 36% (TCS=0 samples)
```

---

*Review complete. The research has promise but needs significant methodological fixes before publication.*
