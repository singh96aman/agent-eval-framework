# Paper Materials: Judge Blindness to Trajectory Criticality

**Research Project:** Investigating whether LLM judges systematically miss early structural failures in agent trajectories while over-penalizing late local errors.

**Literature Review Date:** April 2, 2026  
**Papers Reviewed:** 25 key papers across 5 categories  
**Status:** ✅ Comprehensive search complete, ready for experiments

---

## 📁 Files in This Directory

### Core Documents

1. **`literature_review.md`** (33 KB)
   - Full comprehensive review
   - 25 papers with detailed summaries
   - Gap analysis and novelty statement
   - Complete with key findings and relevance assessments
   - **Use for:** Detailed understanding, writing Related Work section

2. **`literature_review_summary.md`** (12 KB)
   - Condensed executive summary
   - Top 15 papers prioritized
   - Quick gap confirmation
   - **Use for:** Fast reference, presentations, explaining to others

3. **`paper_quick_reference.md`** (16 KB)
   - Searchable paper database
   - Organized by theme, venue, method
   - One-sentence summaries
   - Citation clusters ready for paper sections
   - **Use for:** Writing paper, finding specific citations

4. **`research_roadmap.md`** (17 KB)
   - 12-week experimental plan
   - 3 core experiments designed
   - Dataset selection and metrics
   - Success criteria and risk mitigation
   - **Use for:** Planning experiments, grant proposals

5. **`references.bib`** (15 KB)
   - Complete BibTeX entries
   - 40+ papers categorized
   - Ready for LaTeX compilation
   - **Use for:** Paper bibliography

6. **`abstract.MD`** (2.4 KB)
   - Original project abstract
   - **Use for:** Context on overall project goals

---

## 🎯 Quick Start: Where to Begin

### If you want to...

**Understand the research gap:**
→ Read `literature_review_summary.md` → Section "Novel Contribution Statement"

**Find a specific paper:**
→ Open `paper_quick_reference.md` → Use Cmd+F to search by topic

**Design experiments:**
→ Read `research_roadmap.md` → Phase 2-4

**Write the paper:**
→ Use `paper_quick_reference.md` → "Citation Clusters for Paper Sections"

**Get citations for LaTeX:**
→ Copy from `references.bib`

**Deep dive on related work:**
→ Read `literature_review.md` → Category-by-category analysis

---

## 🔍 Key Findings Summary

### The Gap We Fill

**What exists:**
- ✅ Judge biases documented (position, self-preference, confidence)
- ✅ Error propagation theory (O(√T) bounds)
- ✅ Position-dependent metrics (for consistency)
- ✅ Effort-based criticality (in autonomous vehicles)
- ✅ Agent benchmarks (outcome-focused)

**What's missing:**
- ❌ Judge calibration to trajectory criticality
- ❌ Position-dependent error impact in agent systems
- ❌ Planning vs. execution error distinction (quantitative)
- ❌ Effort-based metrics for cognitive/computational cost
- ❌ Criticality-aware evaluation protocols

**Our contribution:**
- 🎯 First systematic study of judge blindness to trajectory criticality
- 🎯 Position + effort weighted evaluation metrics
- 🎯 Quantitative framework for structural vs. local errors
- 🎯 Calibration methods for criticality-aware judges

---

## 📊 Literature Landscape

### Papers by Category (Top 15)

**🏆 Category 1: LLM Judge Methods (4 papers)**
- G-Eval (2303.16634) - Foundation
- Position Bias (2305.17926) - Bias
- Confidence Miscalibration (2603.09985) - Reliability
- SLMEval (2505.16003) - Calibration issues

**🤖 Category 2: Agent Evaluation (4 papers)**
- AgentBench (2308.03688) - Standard benchmark
- SWE-Bench Randomness (2602.07150) - Trajectory variance
- SWE-ABS (2603.00520) - Outcome misleads
- AdaRubric (2603.21362) - Step-wise baseline

**🔗 Category 3: Error Propagation (3 papers)**
- Information Fidelity (2602.13320) ⭐⭐⭐ - Theory
- F-LLM (2602.12756) ⭐⭐⭐ - Early cascade
- AgentProcessBench (2603.14465) - Agent-specific

**📏 Category 4: Criticality Metrics (2 papers)**
- PWC (2503.22353) ⭐⭐⭐ - Position weighting
- AV Effort Metrics (2603.28029) ⭐⭐⭐ - Criticality definition

**🔬 Category 5: Supporting Evidence (2 papers)**
- Stop Rewarding Hallucinated (2602.05897) - Outcome bias
- Consistency Amplifies (2603.25764) - Pattern propagation

---

## 🧭 Research Flow

```
Literature Review (DONE)
    ↓
Theoretical Framework (Week 1-2)
    → CWPR formulation
    → Effort metrics (TWR/MRD/CIS)
    → JCC metric definition
    ↓
Baseline Experiments (Week 3-4)
    → Exp 1: Judge-Criticality Correlation (expect JCC < 0.5)
    → Exp 2: Planning vs Execution Discrimination (expect < 0.4)
    ↓
Proposed Methods (Week 5-7)
    → CWPR implementation
    → Calibrated judge training
    ↓
Comprehensive Eval (Week 8-10)
    → SWE-Bench + AgentBench + GAIA
    → Ablations
    → Human validation
    ↓
Paper Writing (Week 11-12)
    → Use citation clusters from quick_reference.md
    → Results: demonstrate judge blindness
    → Discussion: implications for evaluation protocols
```

---

## 📖 How Papers Support Our Work

### Foundational Theory

**Information Fidelity (Fan et al., 2026)**
- **Provides:** O(√T) error propagation bounds
- **We extend:** Theory → evaluation metrics
- **Our addition:** Judge calibration to propagation-aware criticality

**PWC (Li et al., 2025)**
- **Provides:** Position-dependent weighting function
- **We extend:** Consistency → error impact
- **Our addition:** Criticality weighting, not just temporal decay

**AV Effort Metrics (Kaul et al., 2026)**
- **Provides:** Criticality = downstream effort (FSR/MDR/LEA)
- **We extend:** Physical effort → cognitive/computational effort
- **Our addition:** TWR (token waste), MRD (re-planning), CIS (cascade)

---

### Empirical Motivation

**SWE-Bench Randomness (Bjarnason et al., 2026)**
- **Shows:** 2.2-6.0pp variance, high trajectory instability
- **Implies:** Early planning errors cause divergent outcomes
- **We test:** Whether judges detect this instability

**SWE-ABS (Yu et al., 2026)**
- **Shows:** 19.71% semantic failures in passing tests
- **Implies:** Outcome metrics insufficient
- **We test:** Whether judges catch structural issues outcome misses

**F-LLM (Zhang et al., 2026)**
- **Shows:** "Minor early deviations cascade"
- **Implies:** Early errors more critical than late
- **We test:** Whether judges weight accordingly

---

### Methodological Baselines

**G-Eval (Liu et al., 2023)**
- **Baseline:** r=0.514 judge-human correlation
- **We compare:** Our JCC metric vs. G-Eval performance
- **Expected:** G-Eval low JCC (< 0.4) despite decent outcome correlation

**AdaRubric (Ding, 2026)**
- **Baseline:** Step-wise scoring, r=0.79
- **We compare:** AdaRubric (uniform step weights) vs. CWPR (criticality-weighted)
- **Expected:** CWPR higher JCC (> 0.7 vs. < 0.5)

**AgentProcessBench (Fan et al., 2026)**
- **Baseline:** Error propagation rules, ternary labels
- **We compare:** Rules without criticality weights vs. our effort metrics
- **Expected:** Criticality weighting improves downstream cost prediction

---

## 🎓 Key Concepts Defined

### Judge-Criticality Correlation (JCC)
```
JCC = correlation(judge_penalty, ground_truth_downstream_cost)
```
- Measures if judge scores predict actual impact
- Range: -1 to 1 (higher = better calibrated)
- **Hypothesis:** Current judges have JCC < 0.5

### Criticality-Weighted Process Reward (CWPR)
```
CWPR = Σ position_weight(i) × effort_weight(i) × PRM(step_i)
```
- Extends process rewards with criticality awareness
- Position weight: Higher for early steps (PWC-inspired)
- Effort weight: Downstream cost (AV-inspired)

### Effort-Based Criticality Metrics
1. **Token Waste Rate (TWR):** Tokens in invalidated paths / total
2. **Maximum Re-planning Depth (MRD):** Steps from error to recovery
3. **Cascade Invalidation Scope (CIS):** Dependent steps invalidated

### Structural vs. Local Errors
- **Structural:** Planning, tool selection (high criticality)
- **Local:** Parameters, formatting (low criticality)
- **Key distinction:** Same outcome failure, different downstream cost

---

## 📈 Expected Results

### Claim 1: Judge Miscalibration
**Measure:** JCC < 0.5 for G-Eval, GPT-4, AdaRubric  
**Evidence papers:** Position Bias, Confidence Miscalibration  
**Interpretation:** Judges don't predict downstream cost

### Claim 2: Position Blindness
**Measure:** Similar judge penalties for early vs. late errors despite 10x cost difference  
**Evidence papers:** F-LLM (early cascade), PWC (position matters)  
**Interpretation:** Judges weight uniformly, miss position effects

### Claim 3: Structural Blindness
**Measure:** Discrimination ratio < 0.4 (planning vs. execution)  
**Evidence papers:** APEX-EM, Environment Maps (structural issues)  
**Interpretation:** Judges conflate error types by salience, not impact

### Claim 4: Calibration Works
**Measure:** CWPR achieves JCC > 0.7 (vs. < 0.5 baseline)  
**Evidence papers:** Information Fidelity (propagation), AV Effort (calibration possible)  
**Interpretation:** Explicit criticality training improves judge-impact alignment

---

## ✅ Next Immediate Actions

1. **Implement baseline metrics** (Week 1)
   - G-Eval API wrapper
   - Uniform PRM (AgentProcessBench-style)
   - Ground truth cost computation (token waste, cascade depth)

2. **Create synthetic dataset** (Week 1-2)
   - 50 trajectories with controlled error placement
   - Error types: planning, tool, parameter, formatting
   - Positions: early (first quartile) vs. late (fourth quartile)

3. **Pilot JCC computation** (Week 2)
   - Run baseline judges on synthetic data
   - Compute ground truth costs
   - Calculate JCC for each judge
   - **Decision point:** If JCC > 0.5, hypothesis unsupported → pivot

4. **Design CWPR implementation** (Week 3)
   - Position weighting function (test linear, exponential, PWC)
   - Effort computation (TWR, MRD, CIS)
   - Integration with existing PRMs

---

## 🔗 Related Work Organization

**For paper's Related Work section, organize as:**

### 2.1 LLM-as-Judge Evaluation
*Theme: Judges have biases but criticality unexplored*

Cite: G-Eval, Position Bias, Confidence Miscal., SLMEval, Dual Optimal

### 2.2 Agent Trajectory Evaluation  
*Theme: Benchmarks outcome-focused, trajectory analysis limited*

Cite: AgentBench, SWE-Bench Random., SWE-ABS, AdaRubric, AgentProcessBench

### 2.3 Error Propagation in Sequential Systems
*Theme: Propagation real and quantifiable, motivates position-awareness*

Cite: Information Fidelity, F-LLM, CoFiCot, Doctor-RAG

### 2.4 Criticality and Position-Dependent Metrics
*Theme: Position and effort matter but not in agent evaluation*

Cite: PWC, AV Effort Metrics, Process Supervision

### 2.5 Outcome vs. Process-Based Evaluation
*Theme: Process supervision insufficient without criticality*

Cite: Stop Rewarding Halluc., Rethinking Rewards, Process-Supervised MARL

---

## 🎯 Success Metrics

**Paper accepted if we demonstrate:**

1. ✅ **Judge miscalibration** (JCC < 0.5, discrimination < 0.4)
2. ✅ **Criticality metrics work** (ρ > 0.6 with experts, R² > 0.7 for cost prediction)
3. ✅ **Calibration improves judges** (JCC > 0.7 after training)
4. ✅ **Generalizes across benchmarks** (SWE-Bench, AgentBench, GAIA)

**Paper strong if we also show:**

5. ⭐ **Ablations** (position weight + effort weight both necessary)
6. ⭐ **Real-world impact** (criticality-weighted leaderboard changes rankings)
7. ⭐ **Qualitative insights** (case studies of judge failures)

---

## 📚 Paper Template Structure

```
1. Introduction
   - Motivation: Outcome metrics mislead [SWE-ABS]
   - Problem: Judges may too [Position Bias]
   - Hypothesis: Judge blindness to criticality
   - Contributions: Study + metrics + calibration

2. Related Work (use quick_reference.md citation clusters)
   - 2.1 LLM-as-Judge
   - 2.2 Agent Evaluation
   - 2.3 Error Propagation
   - 2.4 Criticality Metrics
   - 2.5 Outcome vs Process

3. Problem Formulation
   - Trajectory representation
   - Criticality definition
   - JCC metric

4. Methods
   - CWPR (Criticality-Weighted Process Reward)
   - Effort metrics (TWR/MRD/CIS)
   - Calibrated judge training

5. Experiments
   - Exp 1: Judge-Criticality Correlation
   - Exp 2: Planning vs Execution Discrimination
   - Exp 3: Calibrated Methods Evaluation

6. Results
   - Baselines miscalibrated (JCC < 0.5)
   - Proposed methods improve (JCC > 0.7)
   - Ablations validate design choices
   - Human evaluation confirms

7. Discussion
   - Implications for benchmark design
   - Trajectory-level evaluation importance
   - Limitations and future work

8. Conclusion
   - Judge blindness demonstrated
   - Calibration methods effective
   - Call for criticality-aware protocols
```

---

## 🔖 Bookmarks

**Most important papers to read first:**
1. Information Fidelity (2602.13320) - Error propagation theory
2. PWC (2503.22353) - Position-dependent metrics
3. AV Effort Metrics (2603.28029) - Criticality definition
4. G-Eval (2303.16634) - Judge foundation
5. F-LLM (2602.12756) - Early cascade evidence

**Best evidence for our hypothesis:**
1. SWE-Bench Randomness - Trajectory variance
2. SWE-ABS - Outcome misleads
3. Position Bias - Judge position sensitivity
4. Stop Rewarding Hallucinated - Outcome bias problem

**Closest methodological competitors:**
1. AdaRubric - Step-wise scoring (no position weight)
2. AgentProcessBench - Error propagation (no criticality weight)
3. Uniform PRM - Process rewards (no effort awareness)

---

## 📞 Contact & Updates

**Project Owner:** [Your name]  
**Last Updated:** April 2, 2026  
**Next Review:** After pilot experiments (Week 2)

**Files to update after experiments:**
- Add `experimental_results.md` with findings
- Update `research_roadmap.md` with actual timelines
- Expand `literature_review.md` if new papers found

---

## 🚀 Getting Started Checklist

- [x] Literature review complete (25 papers)
- [x] Gap analysis confirmed (no prior work on judge-criticality)
- [x] Theoretical framework designed (CWPR, TWR/MRD/CIS, JCC)
- [x] Research roadmap with experiments (12 weeks)
- [x] BibTeX references ready (40+ papers)
- [ ] Synthetic dataset created (Week 1-2)
- [ ] Baseline judges implemented (Week 1)
- [ ] Pilot JCC computed (Week 2)
- [ ] Hypothesis confirmed (Week 2, decision point)
- [ ] Paper outline drafted (Week 11)

**Current Status:** ✅ Ready to start experimental phase

---

*This README serves as the navigation hub for all paper materials. Start here, then branch to specific documents based on your needs.*
