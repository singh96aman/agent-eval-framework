# Quick Reference: Key Papers by Theme

**Last Updated:** April 2, 2026

---

## 🎯 MUST-READ Papers (Top 7)

| # | Paper | Why Critical | arXiv ID | Key Metric/Finding |
|---|-------|--------------|----------|-------------------|
| 1 | **Information Fidelity** | Error propagation theory | 2602.13320 | O(√T) distortion bounds |
| 2 | **F-LLM** | Early deviation cascade | 2602.12756 | "Minor early deviations cascade" |
| 3 | **PWC** | Position-dependent metrics | 2503.22353 | Position-Weighted Consistency |
| 4 | **AV Effort Metrics** | Criticality definition | 2603.28029 | FSR/MDR/LEA, 65-93% non-critical |
| 5 | **G-Eval** | Judge foundation | 2303.16634 | r=0.514, self-preference bias |
| 6 | **Position Bias** | Judge position sensitivity | 2305.17926 | Order affects rankings |
| 7 | **AgentProcessBench** | Error propagation in agents | 2603.14465 | Ternary labels + propagation rules |

---

## 📊 By Research Question

### Q1: Are judges miscalibrated?

| Paper | Finding | arXiv |
|-------|---------|-------|
| Position Bias (Wang) | Order affects judge rankings | 2305.17926 |
| G-Eval | Self-preference bias | 2303.16634 |
| Confidence Miscal. | ECE 0.726 at 23.3% accuracy | 2603.09985 |
| SLMEval | G-Eval negative correlation | 2505.16003 |
| Rank Manipulation | Adversarial judge attacks work | 2510.06732 |

**Answer:** YES, multiple bias types documented. Criticality bias unexplored.

---

### Q2: Do errors propagate in trajectories?

| Paper | Finding | arXiv |
|-------|---------|-------|
| Information Fidelity | Linear growth O(√T) | 2602.13320 |
| F-LLM | Early deviations cascade | 2602.12756 |
| CoFiCot | Context fragmentation problem | 2603.08251 |
| Doctor-RAG | Error localization + prefix reuse | 2604.00865 |
| Environment Maps | Cascading errors documented | 2603.23610 |

**Answer:** YES, theoretically grounded (O(√T)) and empirically observed.

---

### Q3: Does position matter?

| Paper | Finding | arXiv |
|-------|---------|-------|
| PWC | Position-Weighted Consistency | 2503.22353 |
| F-LLM | EARLY deviations cascade (not late) | 2602.12756 |
| Omanic | Errors amplify in later hops | 2603.16654 |

**Answer:** YES, early steps have compounding impact (consistency + propagation).

---

### Q4: Do outcome metrics mislead?

| Paper | Finding | arXiv |
|-------|---------|-------|
| SWE-ABS | 19.71% semantic failures in passing | 2603.00520 |
| Stop Rewarding Halluc. | Outcome rewards reinforce unfaithful | 2602.05897 |
| SWE-Bench Randomness | 2.2-6.0pp variance same task | 2602.07150 |
| Consistency Amplifies | Consistency ≠ correctness | 2603.25764 |

**Answer:** YES, outcome masks trajectory problems.

---

### Q5: How should criticality be defined?

| Paper | Definition | arXiv |
|-------|-----------|-------|
| AV Effort Metrics | Downstream effort/cost | 2603.28029 |
| PWC | Position importance | 2503.22353 |
| Information Fidelity | Distortion magnitude | 2602.13320 |
| Process Supervision | Information gain | 2603.17815 |

**Answer:** Downstream impact (effort, cost, cascade) > local correctness.

---

## 🏷️ By Paper Category

### LLM Judge Methods

| Paper | Focus | Key Contribution | arXiv |
|-------|-------|------------------|-------|
| G-Eval | Foundation | CoT judge, r=0.514 | 2303.16634 |
| Position Bias | Bias | Order affects ranking | 2305.17926 |
| Dual Optimal | Calibration | Item Response Theory | 2604.00979 |
| SLMEval | Calibration | Entropy-based | 2505.16003 |
| Confidence Miscal. | Bias | Weaker = more overconfident | 2603.09985 |
| Rank Manipulation | Security | Adversarial attacks | 2510.06732 |
| Autorubric | Framework | Position/verbosity bias mitigation | 2603.00077 |

### Agent Evaluation

| Paper | Focus | Key Contribution | arXiv |
|-------|-------|------------------|-------|
| AgentBench | Benchmark | 8 environments | 2308.03688 |
| SWE-Bench Random. | Variance | 2.2-6.0pp single-run | 2602.07150 |
| SWE-ABS | Validation | 19.71% semantic failures | 2603.00520 |
| AdaRubric | Method | Step-wise, r=0.79 | 2603.21362 |
| Consistency Amplif. | Analysis | Consistency ≠ correctness | 2603.25764 |
| ToolLLM | Tool Use | 16k APIs, DFS search | 2307.16789 |
| AgentProcessBench | Process | Error propagation rules | 2603.14465 |

### Error Propagation

| Paper | Focus | Key Contribution | arXiv |
|-------|-------|------------------|-------|
| Information Fidelity | Theory | O(√T) bounds, 80% reduction | 2602.13320 |
| F-LLM | Cascade | Early deviations cascade | 2602.12756 |
| CoFiCot | Context | Stateful propagation | 2603.08251 |
| Doctor-RAG | Repair | Error localization | 2604.00865 |
| Dynamic Adjudication | Reduction | 3-stage framework | 2510.05134 |

### Criticality Metrics

| Paper | Focus | Metric | arXiv |
|-------|-------|--------|-------|
| PWC | Position | Position-Weighted Consistency | 2503.22353 |
| AV Effort | Effort | FSR/MDR/LEA | 2603.28029 |
| Process Supervision | Information | Monte Carlo info gain | 2603.17815 |
| AgentProcessBench | Propagation | Ternary + rules | 2603.14465 |

### Supporting Evidence

| Paper | Focus | Key Finding | arXiv |
|-------|-------|-------------|-------|
| Stop Rewarding Halluc. | Outcome vs Process | Outcome misleads | 2602.05897 |
| Rethinking Rewards | ORM vs PRM | Similar performance | 2510.00492 |
| APEX-EM | Structure | Re-derive structurally similar | 2603.29093 |
| Environment Maps | Planning | Cascading from planning | 2603.23610 |
| Omanic | Multi-hop | Errors amplify in hops | 2603.16654 |

---

## 🔍 By Venue/Year

### 2026 (Most Recent)

| Month | Paper | arXiv | Category |
|-------|-------|-------|----------|
| Feb | Information Fidelity | 2602.13320 | Propagation ⭐ |
| Feb | F-LLM | 2602.12756 | Propagation ⭐ |
| Feb | SWE-Bench Random. | 2602.07150 | Agent Eval |
| Mar | PWC | 2503.22353 | Criticality ⭐ |
| Mar | CoFiCot | 2603.08251 | Propagation |
| Mar | AV Effort Metrics | 2603.28029 | Criticality ⭐ |
| Mar | SWE-ABS | 2603.00520 | Agent Eval |
| Mar | AdaRubric | 2603.21362 | Agent Eval |
| Mar | Consistency Amplif. | 2603.25764 | Agent Eval |
| Mar | AgentProcessBench | 2603.14465 | Criticality |
| Mar | APEX-EM | 2603.29093 | Structure |
| Mar | Environment Maps | 2603.23610 | Structure |
| Mar | Omanic | 2603.16654 | Multi-hop |
| Apr | Dual Optimal | 2604.00979 | Judge Calib. |
| Apr | Doctor-RAG | 2604.00865 | Propagation |

### 2025

| Month | Paper | arXiv | Category |
|-------|-------|-------|----------|
| Mar | PWC (published) | 2503.22353 | Criticality ⭐ |
| May | SLMEval | 2505.16003 | Judge Calib. |
| Oct | Dynamic Adjudication | 2510.05134 | Propagation |
| Oct | Rethinking Rewards | 2510.00492 | Outcome vs Process |
| Oct | Rank Manipulation | 2510.06732 | Judge Security |

### 2023-2024

| Year | Paper | arXiv | Category |
|------|-------|-------|----------|
| 2023 | G-Eval | 2303.16634 | Judge Foundation ⭐ |
| 2023 | Position Bias | 2305.17926 | Judge Bias ⭐ |
| 2023 | AgentBench | 2308.03688 | Agent Benchmark |
| 2023 | ToolLLM | 2307.16789 | Tool Use |
| 2024 | LLM Detectors | 2409.03291 | Judge Robustness |

---

## 🎨 By Method/Contribution Type

### Theoretical Frameworks

| Paper | Framework | arXiv |
|-------|-----------|-------|
| Information Fidelity | O(√T) distortion bounds | 2602.13320 |
| Process Supervision | Information-theoretic step importance | 2603.17815 |
| Dual Optimal | Item Response Theory for judges | 2604.00979 |

### Metrics Introduced

| Paper | Metric | What It Measures | arXiv |
|-------|--------|------------------|-------|
| PWC | Position-Weighted Consistency | Early-stage stability | 2503.22353 |
| AV Effort | FSR/MDR/LEA | Physical effort (AV) | 2603.28029 |
| G-Eval | Spearman correlation | Judge-human alignment | 2303.16634 |
| AgentProcessBench | Ternary labels | Step quality + propagation | 2603.14465 |

### Benchmarks

| Paper | Benchmark | Domain | arXiv |
|-------|-----------|--------|-------|
| AgentBench | 8 environments | Multi-domain agents | 2308.03688 |
| ToolLLM | 16k+ APIs | Tool use | 2307.16789 |
| StableToolBench | Virtual API server | Tool use (stable) | 2403.07714 |
| AgentProcessBench | Step-level labels | Tool agents process | 2603.14465 |

### Methodological Papers

| Paper | Method | Purpose | arXiv |
|-------|--------|---------|-------|
| AdaRubric | Task-adaptive rubrics | Step-wise scoring | 2603.21362 |
| CoFiCot | Stateful propagation | Context preservation | 2603.08251 |
| Doctor-RAG | Failure-aware repair | Error localization | 2604.00865 |
| F-LLM | Feedback control | Bound error propagation | 2602.12756 |

---

## 🔗 Paper Relationships

### Direct Build-Ons for Our Work

```
Our Work: Judge Blindness to Trajectory Criticality
    ↓
Built on:
    → Information Fidelity (propagation theory)
    → PWC (position weighting)
    → AV Effort Metrics (criticality definition)
    → AgentProcessBench (error propagation in agents)
    
Extended to:
    → Judge calibration (not covered in above)
    → Position-dependent criticality (PWC is for consistency)
    → Effort metrics for agents (AV is for autonomous vehicles)
```

### Foundational → Our Focus

```
G-Eval (2023)
    → Position Bias (2023)
        → Confidence Miscal. (2026)
            → **OUR WORK: Criticality Miscalibration**

Information Fidelity (2026)
    → F-LLM (2026)
        → **OUR WORK: Propagation → Evaluation**

PWC (2025)
    → **OUR WORK: Consistency → Criticality**

AV Effort Metrics (2026)
    → **OUR WORK: Physical Effort → Cognitive Effort**
```

---

## 📝 Citation Clusters for Paper Sections

### Introduction
**Theme:** Judges are important, may be miscalibrated, position matters

```
"LLM-as-judge evaluation has become standard [G-Eval], but judges 
exhibit systematic biases [Position Bias, Confidence Miscal]. 
Meanwhile, errors in agent trajectories propagate [Information Fidelity, 
F-LLM], with early failures causing cascading degradation. While 
position matters for consistency [PWC] and criticality is well-defined 
in other domains [AV Effort], no work studies judge calibration to 
trajectory criticality."
```

**Cite:** G-Eval, Position Bias, Confidence Miscal., Information Fidelity, F-LLM, PWC, AV Effort

---

### Related Work - Judge Reliability
**Theme:** Judges have biases, calibration is hard

```
"G-Eval [cite] established LLM-as-judge evaluation but identified 
self-preference bias. Subsequent work documented position bias [Wang], 
confidence miscalibration [Ghosh], and adversarial vulnerabilities 
[Xing]. Calibration methods using entropy [SLMEval] and IRT [Dual Optimal] 
show promise but don't address trajectory-specific challenges."
```

**Cite:** G-Eval, Position Bias, Confidence Miscal., Rank Manipulation, SLMEval, Dual Optimal, Autorubric

---

### Related Work - Agent Evaluation
**Theme:** Benchmarks exist but are outcome-focused

```
"Agent benchmarks like AgentBench [Liu] and ToolLLM [Qin] use 
primarily outcome-based metrics. Recent work shows high variance 
[SWE-Bench Randomness] and semantic failures in passing tests [SWE-ABS], 
suggesting outcome metrics mislead. AdaRubric [Ding] introduced 
step-wise scoring but without position-dependent weighting. 
AgentProcessBench [Fan] models error propagation rules but not criticality."
```

**Cite:** AgentBench, ToolLLM, SWE-Bench Randomness, SWE-ABS, AdaRubric, AgentProcessBench

---

### Related Work - Error Propagation
**Theme:** Propagation is real, quantifiable, position-dependent

```
"Fan et al. [Information Fidelity] prove cumulative distortion grows 
linearly with O(√T) bounds in tool-using agents. Zhang et al. [F-LLM] 
demonstrate 'minor early deviations cascade into significant trajectory 
drift.' Other work addresses context fragmentation [CoFiCot] and error 
localization [Doctor-RAG]. These findings motivate position-aware 
evaluation but don't connect to judge calibration."
```

**Cite:** Information Fidelity, F-LLM, CoFiCot, Doctor-RAG, Environment Maps

---

### Related Work - Criticality Metrics
**Theme:** Position and effort matter, but not in agent evaluation

```
"Li et al. [PWC] introduce Position-Weighted Consistency emphasizing 
early-stage stability in sequential interactions. In autonomous driving, 
Kaul et al. [AV Effort] define criticality as downstream effort (FSR/MDR/LEA), 
showing 65-93% of errors are non-critical by impact. Process reward 
models use information gain [Royer] but lack position weighting. We 
extend these concepts to agent trajectory evaluation."
```

**Cite:** PWC, AV Effort, Process Supervision, AgentProcessBench

---

### Discussion - Outcome vs Process
**Theme:** Process supervision insufficient without criticality

```
"Our findings align with Nie et al. [Stop Rewarding Halluc.] showing 
outcome-based rewards reinforce unfaithful reasoning. While Lee et al. 
[Rethinking Rewards] find ORMs and PRMs perform similarly, we show 
criticality-weighted process rewards outperform both by distinguishing 
high-impact from low-impact steps."
```

**Cite:** Stop Rewarding Halluc., Rethinking Rewards, Process-Supervised MARL

---

## 🎯 Key Numbers to Remember

| Finding | Paper | Number | Use In |
|---------|-------|--------|--------|
| Judge-human correlation | G-Eval | 0.514 | Baseline performance |
| Trajectory variance | SWE-Bench Random. | 2.2-6.0pp | Instability evidence |
| Semantic failures | SWE-ABS | 19.71% | Outcome misleads |
| Distortion reduction | Information Fidelity | 80% | Propagation control |
| Non-critical errors | AV Effort | 65-93% | Criticality definition |
| Step-wise correlation | AdaRubric | r=0.79 | Baseline comparison |
| Confidence miscal. | Dunning-Kruger | ECE 0.726 @ 23.3% | Judge reliability |

---

## 🚀 Quick Search

**Need a paper about...**

- **Judge bias?** → Position Bias (2305.17926), Confidence Miscal. (2603.09985)
- **Error propagation theory?** → Information Fidelity (2602.13320), F-LLM (2602.12756)
- **Position weighting?** → PWC (2503.22353)
- **Criticality definition?** → AV Effort (2603.28029)
- **Outcome vs process?** → Stop Rewarding Halluc. (2602.05897), Rethinking Rewards (2510.00492)
- **Agent benchmarks?** → AgentBench (2308.03688), SWE-Bench Random. (2602.07150)
- **Step-wise evaluation?** → AdaRubric (2603.21362), AgentProcessBench (2603.14465)
- **Early vs late errors?** → F-LLM (2602.12756), Omanic (2603.16654)

---

## ⚡ One-Sentence Summaries

| Paper | One-Sentence Summary |
|-------|---------------------|
| Information Fidelity | Errors in tool-using agents grow linearly with O(√T) bounds, reduced 80% by semantic weighting. |
| F-LLM | Minor early deviations cascade into significant trajectory drift through autoregressive generation. |
| PWC | Position-Weighted Consistency metric captures importance of early-stage stability in sequential interactions. |
| AV Effort | Criticality = downstream effort (FSR/MDR/LEA); 65-93% of perception errors non-critical by impact. |
| G-Eval | GPT-4 as judge achieves r=0.514 with humans but exhibits self-preference bias. |
| Position Bias | Judge rankings manipulated by simply reordering responses, not content. |
| AgentProcessBench | Ternary step labels (correct/incorrect/exploratory) with error propagation rules for tool agents. |
| SWE-ABS | 19.71% of passing patches semantically incorrect due to weak test coverage. |
| SWE-Bench Random. | Single-run agent estimates vary 2.2-6.0 percentage points across 60k trajectories. |
| AdaRubric | Task-adaptive rubrics with step-wise scoring achieve r=0.79 human correlation. |
| Consistency Amplif. | Higher consistency aligns with accuracy but can amplify incorrect interpretations. |
| Stop Rewarding Halluc. | Outcome-based rewards inadvertently reinforce unfaithful reasoning when answer is correct. |
| Confidence Miscal. | Weaker LLMs show higher overconfidence (Kimi K2: ECE 0.726 at 23.3% accuracy). |
| Doctor-RAG | Coverage-gated failure taxonomy with error localization enables prefix reuse. |
| Omanic | Errors amplify in later hops when factual completeness gaps exist. |
