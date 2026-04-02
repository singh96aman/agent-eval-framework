# Literature Review Summary: Judge Blindness to Trajectory Criticality

**Date:** April 2, 2026  
**Papers Reviewed:** 25 key papers  
**Status:** Initial comprehensive search complete

---

## Executive Summary

**Main Finding:** No existing work systematically studies whether LLM judges are calibrated to error criticality in agent trajectories. While judge biases, error propagation, and position-dependent metrics have been studied separately, they have never been combined to address trajectory criticality in agent evaluation.

**Key Gap:** Current work demonstrates:
- Judges have systematic biases (position, self-preference, temperature)
- Errors propagate through trajectories with quantifiable impact
- Position matters for some metrics (consistency)
- Effort-based criticality works in autonomous driving

But NO work examines: **Do judges distinguish planning errors (high criticality) from execution errors (low criticality)?**

---

## Critical Papers by Category

### Category 1: LLM Judge Failure Modes (Most Relevant: 4/7)

1. **G-Eval** (Liu et al., 2023, arXiv:2303.16634)
   - Foundational LLM-as-judge paper
   - Identifies self-preference bias
   - **Gap:** No trajectory or criticality analysis

2. **Position Bias** (Wang et al., 2023, arXiv:2305.17926)
   - Response order affects judge rankings
   - **Implication:** Judges are position-sensitive but only studied for presentation order, not trajectory position

3. **Confidence Miscalibration** (Ghosh & Panday, 2026, arXiv:2603.09985)
   - Weaker models show higher overconfidence (ECE 0.726 at 23.3% accuracy)
   - **Implication:** If judges miscalibrate confidence, they may miscalibrate criticality too

4. **SLMEval** (Daynauth et al., 2025, arXiv:2505.16003)
   - G-Eval shows negative correlation with humans on some tasks
   - **Implication:** Judge failures are task/context-dependent

### Category 2: Agent Trajectory Evaluation (Most Relevant: 3/7)

5. **SWE-Bench Randomness** (Bjarnason et al., 2026, arXiv:2602.07150)
   - Single-run estimates vary 2.2-6.0 percentage points across 60k trajectories
   - **Implication:** High trajectory variance suggests early instability (planning errors?) affecting outcomes

6. **SWE-ABS** (Yu et al., 2026, arXiv:2603.00520)
   - 19.71% of passing patches semantically incorrect
   - **Implication:** Outcome metrics mask structural failures

7. **AdaRubric** (Ding, 2026, arXiv:2603.21362)
   - Step-by-step scoring with task-adaptive rubrics
   - Pearson r=0.79 with humans
   - **Gap:** No position-dependent weighting or criticality modeling

### Category 3: Error Propagation Theory (Most Relevant: 2/5)

8. **Information Fidelity** (Fan et al., 2026, arXiv:2602.13320) ⭐⭐⭐
   - **CRITICAL:** Cumulative distortion grows linearly, bounded by O(√T)
   - Semantic weighting reduces distortion 80%
   - Re-grounding every ~9 steps controls propagation
   - **Implication:** Theoretical framework for quantifying error propagation we can build on

9. **F-LLM** (Zhang et al., 2026, arXiv:2602.12756) ⭐⭐⭐
   - **CRITICAL:** "Minor early deviations cascade into significant trajectory drift"
   - Explicitly models early errors as high-impact
   - **Implication:** Direct support for early criticality hypothesis

### Category 4: Criticality Metrics (Most Relevant: 2/4)

10. **Position-Weighted Consistency (PWC)** (Li et al., 2025, arXiv:2503.22353) ⭐⭐⭐
    - **CRITICAL:** Only paper with explicit position-dependent scoring
    - Emphasizes early-stage stability importance
    - **Gap:** Applied to consistency, not error impact
    - **Our contribution:** Extend PWC from consistency to criticality

11. **Effort-Based Criticality in Autonomous Driving** (Kaul et al., 2026, arXiv:2603.28029) ⭐⭐⭐
    - **CRITICAL:** Criticality = downstream effort/cost, not just correctness
    - Introduces FSR (velocity loss), MDR (braking demand), LEA (steering effort)
    - 65-93% of errors non-critical by effort metrics
    - **Implication:** Perfect conceptual parallel for agent systems (re-planning cost, token waste)

### Category 5: Supporting Evidence

12. **AgentProcessBench** (Fan et al., 2026, arXiv:2603.14465)
    - Error propagation rules explicitly modeled
    - Ternary labels (correct/incorrect/exploratory)
    - **Gap:** No criticality weighting by position

13. **Stop Rewarding Hallucinated Steps** (Nie et al., 2026, arXiv:2602.05897)
    - "Outcome-based rewards inadvertently reinforce unfaithful reasoning"
    - **Implication:** Outcome metrics mask trajectory problems

14. **Consistency Amplifies** (Mehta, 2026, arXiv:2603.25764)
    - Higher consistency aligns with accuracy but amplifies incorrect patterns
    - **Implication:** Systematic trajectory patterns (from early planning) propagate consistently

---

## Novel Contribution Statement

**We are the FIRST to:**

1. Study **judge calibration to error criticality** in agent trajectories
2. Introduce **position-dependent criticality metrics** for agents (extending PWC from consistency to impact)
3. Quantitatively distinguish **structural (planning) from local (execution) errors** and measure judge sensitivity
4. Demonstrate **judge blindness** to early high-criticality failures
5. Propose **criticality-calibrated evaluation** combining error propagation theory + position weighting + effort metrics

**Closest Related Work - Why We're Different:**

| Work | What They Do | What We Add |
|------|-------------|-------------|
| **AdaRubric** | Step-wise scoring | + Position-dependent criticality weighting |
| **AgentProcessBench** | Error propagation rules | + Criticality weighting by impact |
| **PWC** | Position weighting for consistency | + Extension to error impact/cost |
| **Information Fidelity** | Error propagation theory | + Application to evaluation & judge calibration |
| **AV Effort Metrics** | Physical effort (braking, steering) | + Cognitive/computational effort (re-planning, tokens) |

---

## Evidence Supporting Our Hypothesis

### 1. Judges ARE Miscalibrated (Multiple Biases Documented)
- Position bias in presentation order (Wang 2023)
- Self-preference bias (G-Eval)
- Temperature sensitivity (Gameiro 2024)
- Confidence miscalibration (Ghosh 2026)

**Implication:** Criticality miscalibration is plausible extension.

### 2. Outcome Metrics Mislead
- 19.71% semantic failures in passing tests (SWE-ABS)
- Outcome rewards reinforce unfaithful reasoning (Nie 2026)
- 2.2-6.0 pp variance on identical tasks (Bjarnason 2026)

**Implication:** Judges using outcome-centric evaluation will miss structural issues.

### 3. Early Errors Have Compounding Impact
- "Minor early deviations cascade" (F-LLM)
- Linear distortion growth O(√T) (Information Fidelity)
- Planning failures cause cascades (multiple papers)

**Implication:** Early errors should be weighted higher; judges probably don't do this.

### 4. Position and Structure Matter
- PWC demonstrates position importance (Li 2025)
- Error propagation modeled (AgentProcessBench)
- 65-93% of errors non-critical by effort (AV paper)

**Implication:** Not all errors equal; judges need criticality awareness.

---

## Experimental Design Recommendations

### Core Experiments

**Experiment 1: Judge-Criticality Correlation**
- Create controlled trajectories with errors at different positions
- Measure actual downstream impact (tokens wasted, re-planning steps)
- Compare judge scores to impact
- **Hypothesis:** Correlation < 0.5 (miscalibrated)

**Experiment 2: Planning vs. Execution Error Discrimination**
- Matched pairs: planning error (wrong tool) vs. execution error (wrong param)
- Both cause same outcome failure
- **Hypothesis:** Judges don't distinguish criticality (similar scores despite different impact)

**Experiment 3: Criticality-Calibrated Judge**
- Train judge with criticality-weighted supervision
- Test on downstream impact prediction
- **Hypothesis:** Calibrated judge outperforms baseline (R² > 0.7 vs. < 0.4)

### Datasets

**Primary:**
- **SWE-Bench:** High variance documented, complex trajectories
- **AgentBench:** Multi-step, diverse domains
- **GAIA:** Failures on unindexed info (planning issues)

**Secondary:**
- **ToolBench:** Tool selection criticality
- **WebArena:** Long trajectories, cascading errors

### Metrics to Introduce

**Criticality-Weighted Process Reward (CWPR):**
```
CWPR = Σ w_i × PRM(step_i)
where w_i = position_weight × effort_weight
```

**Effort-Based Criticality (for agents):**
- Token waste from invalid paths
- Re-planning steps required
- Cascade depth (how many steps invalidated)

**Judge-Criticality Correlation (JCC):**
```
JCC = correlation(judge_penalty, actual_downstream_cost)
```

---

## Key Gaps Confirmed

1. **Position-dependent error criticality in agent trajectories:** NONE FOUND
2. **Judge calibration to trajectory structure:** NONE FOUND
3. **Planning vs. execution error distinction (quantitative):** NONE FOUND
4. **Effort-based criticality in agent systems:** NONE FOUND (only in AV)
5. **Judge-criticality alignment metrics:** NONE FOUND

---

## Next Steps

1. **Narrow focus to 15-18 papers** for related work section (prioritize ⭐⭐⭐ marked papers)

2. **Design synthetic trajectory benchmark** with controlled error positions and types

3. **Implement baseline metrics:**
   - Uniform PRM (AgentProcessBench style)
   - PWC adapted to errors
   - Outcome-only
   - G-Eval judge

4. **Implement proposed metrics:**
   - CWPR (Criticality-Weighted Process Reward)
   - Effort-based criticality (token waste, cascade depth)
   - JCC (Judge-Criticality Correlation)

5. **Run experiments** on SWE-Bench, AgentBench, GAIA

6. **Write paper sections:**
   - Introduction (hypothesis + gap)
   - Related work (15-18 papers, organized by our taxonomy)
   - Method (CWPR, effort metrics, JCC)
   - Experiments (3 core experiments)
   - Results (judge miscalibration demonstrated)
   - Discussion (implications for evaluation protocols)

---

## Critical Papers for Related Work (Top 15)

### Must Include (7 papers)

1. **G-Eval** - Foundational judge paper
2. **Information Fidelity** - Error propagation theory
3. **F-LLM** - Early deviation cascade
4. **PWC** - Position-dependent metrics
5. **AV Effort Metrics** - Criticality definition
6. **AgentProcessBench** - Error propagation in agents
7. **Position Bias (Wang)** - Judge position sensitivity

### Strong Support (8 papers)

8. **SWE-Bench Randomness** - Trajectory variance
9. **SWE-ABS** - Outcome metrics mislead
10. **AdaRubric** - Step-wise baseline
11. **Stop Rewarding Hallucinated** - Outcome bias problem
12. **Consistency Amplifies** - Pattern propagation
13. **AgentBench** - Standard benchmark
14. **Survey (Mohammadi)** - Comprehensive gap analysis
15. **Confidence Miscalibration** - Judge reliability issues

---

## Quick Reference: Key Concepts

**Judge Blindness:** Systematic failure of LLM judges to distinguish high-criticality from low-criticality errors

**Trajectory Criticality:** Position-dependent error impact measured by downstream cost (re-planning, wasted computation, cascade depth)

**Structural Error:** Planning or tool-selection mistakes that invalidate subsequent trajectory

**Local Error:** Execution mistakes (wrong parameters, formatting) with limited propagation

**Criticality-Calibration:** Training judges to weight errors by downstream impact rather than local correctness

**Effort-Based Criticality:** Measuring error importance by correction cost (inspired by autonomous driving metrics)

---

## Tags for Paper Organization

**#judge-bias** - Papers on LLM judge failure modes
**#error-propagation** - Papers on cascading errors in sequences
**#position-dependent** - Papers with position-aware metrics
**#agent-evaluation** - Agent benchmarks and evaluation methods
**#criticality** - Papers defining or measuring error importance
**#outcome-vs-process** - Papers comparing reward types
**#trajectory-variance** - Papers documenting trajectory instability
