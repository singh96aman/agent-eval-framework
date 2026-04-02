# Literature Review: Judge Blindness to Trajectory Criticality in LLM-Evaluated Agent Systems

**Research Context:** This literature review examines the hypothesis that LLM judges are systematically miscalibrated to error criticality in agent trajectories, potentially over-penalizing visible local mistakes while under-penalizing early structural failures (planning/tool selection errors) that cause larger downstream degradation.

**Date:** April 2, 2026
**Papers Reviewed:** 25 key papers across 5 categories

---

## Category 1: LLM-as-Judge Evaluation Methods (7 papers)

### 1.1 Foundational Work: G-Eval

**G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment**
- **Citation:** Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, Chenguang Zhu
- **Venue:** arXiv:2303.16634
- **Year:** 2023

**Summary:** Proposes using GPT-4 with chain-of-thought reasoning for NLG evaluation, achieving Spearman correlation of 0.514 with human judgments on summarization tasks, significantly outperforming traditional metrics like BLEU and ROUGE.

**Key Findings:**
- Chain-of-thought prompting improves judge-human alignment
- Form-filling paradigm enables reference-free evaluation
- **Critical limitation identified:** LLM evaluators exhibit bias toward LLM-generated texts

**Metrics:** Spearman correlation with human judgments

**Relevance:** Foundational paper establishing LLM-as-judge methodology but acknowledges self-preference bias. Does not address position-dependent error importance or structural vs. local error distinction.

**My Notes**
* This is basically LLMaJ with Naturalness, Coherence, Engaginess, Groundedness as metrics and spearman correlation coefficient on it

---

### 1.2 Judge Calibration and Failure Modes

**1.2.1 Dual Optimal: Make Your LLM Peer-like with Dignity**
- **Citation:** Xiangqi Wang, Yue Huang, Haomin Zhuang, Kehan Guo, Xiangliang Zhang
- **Venue:** arXiv:2604.00979
- **Year:** 2026

**Summary:** Addresses dual failure modes in aligned language models using Item Response Theory evaluation protocol to disentangle model capability from judge biases.

**Key Findings:**
- Aligned LLMs exhibit systematic dual failure modes
- Judge biases significantly distort evaluation outcomes
- Psychometric calibration can separate genuine capability from evaluator bias

**Metrics:** Item Response Theory-based calibration metrics

**Relevance:** Directly addresses judge calibration issues but focuses on persona-level biases rather than trajectory-position effects.

---

**1.2.2 SLMEval: Entropy-Based Calibration**
- **Citation:** Roland Daynauth, Christopher Clarke, Krisztian Flautner, Lingjia Tang, Jason Mars
- **Venue:** arXiv:2505.16003
- **Year:** 2025

**Summary:** Critiques G-Eval limitations, showing negative correlation with human judgment on certain tasks. Proposes entropy-maximization approach for better calibration.

**Key Findings:**
- G-Eval exhibits task-dependent failures with negative human correlation
- Entropy-based methods achieve superior performance
- Calibration issues vary across task types

**Metrics:** Entropy-based scoring, human correlation coefficients

**Relevance:** Demonstrates systematic G-Eval failures but doesn't address sequential/trajectory-specific evaluation challenges.

---

**1.2.3 Large Language Models are not Fair Evaluators**
- **Citation:** Peiyi Wang, Lei Li, Liang Chen, et al.
- **Venue:** arXiv:2305.17926
- **Year:** 2023

**Summary:** Identifies position bias in LLM judges where quality rankings can be manipulated by altering response order.

**Key Findings:**
- Response order significantly affects judge rankings
- Quality assessment is susceptible to presentation bias
- Simple permutation can reverse judge preferences

**Metrics:** Ranking consistency across orderings

**Relevance:** **Critical for our work** - demonstrates judges are sensitive to position, but in presentation context not trajectory-step context.

---

### 1.3 Judge Security and Adversarial Robustness

**1.3.1 Are LLMs Reliable Rankers? Rank Manipulation via Two-Stage Token Optimization**
- **Citation:** Xing, Li, Du, Hu
- **Venue:** arXiv:2510.06732
- **Year:** 2025

**Summary:** Shows LLM ranking systems are susceptible to adversarial manipulation through natural-sounding prompts.

**Key Findings:**
- LLM-based reranking inherently susceptible to adversarial manipulation
- Attacks can use natural language (no gibberish/special tokens)
- Security implications for judge systems

**Relevance:** Highlights judge vulnerability but in adversarial context rather than organic trajectory evaluation.

---

**1.3.2 LLM Detectors Still Fall Short of Real World**
- **Citation:** Gameiro, Kucharavy, Dolamic
- **Venue:** arXiv:2409.03291
- **Year:** 2024

**Summary:** Detection systems highly vulnerable to sampling temperature increase, revealing brittleness in evaluation systems.

**Key Findings:**
- Temperature variation trivially defeats detection
- Evaluation systems lack robustness to simple distributional shifts

**Relevance:** Shows judge fragility to generation parameters, suggesting deeper calibration issues.

---

**1.3.3 The Dunning-Kruger Effect in Large Language Models**
- **Citation:** Sudipta Ghosh, Mrityunjoy Panday
- **Venue:** arXiv:2603.09985
- **Year:** 2026

**Summary:** Poorly performing models display markedly higher overconfidence. Example: Kimi K2 showed ECE of 0.726 despite only 23.3% accuracy.

**Key Findings:**
- Inverse relationship between accuracy and confidence calibration
- Weaker models exhibit more severe miscalibration
- Implications for self-evaluation and judge reliability

**Relevance:** Confidence miscalibration in judges may parallel miscalibration to error criticality.

---

## Category 2: Agent Trajectory Evaluation (7 papers)

### 2.1 Agent Benchmarks

**2.1.1 AgentBench: Evaluating LLMs as Agents**
- **Citation:** Xiao Liu, Hao Yu, Hanchen Zhang, et al. (22 authors)
- **Venue:** arXiv:2308.03688
- **Year:** 2023

**Summary:** Foundational agent benchmark with 8 distinct environments assessing reasoning and decision-making abilities across API-based and open-source models.

**Key Findings:**
- Multi-environment evaluation reveals capability gaps
- End-to-end success metrics dominant
- Limited step-wise trajectory analysis

**Metrics:** Task completion rates across 8 environments

**Relevance:** Establishes agent evaluation paradigm but uses outcome-based metrics without trajectory decomposition.

---

**2.1.2 MCP-AgentBench**
- **Citation:** arXiv:2509.09734
- **Year:** 2025

**Summary:** Tests 188 tools across 33 MCP servers with outcome-oriented evaluation methodology.

**Key Findings:**
- Outcome-oriented approach dominates tool-use evaluation
- Step-level analysis minimal

**Relevance:** Exemplifies outcome-bias in agent benchmarks, creating gap our work addresses.

---

### 2.2 GAIA Benchmark Papers

**2.2.1 Multiple GAIA studies reviewed:**
- UIS-Digger (arXiv:2603.08117): 70.90% to 24.55% drop on unindexed information
- Lemon Agent (arXiv:2602.07092): 91.36% accuracy
- AgentCPM-Explore (arXiv:2602.06485): 97.09% on text tasks at pass@64

**Common Pattern:**
- All report final accuracy metrics
- No decomposition of where failures occur in trajectories
- Pass@k suggests retry-until-success rather than diagnosis

**Relevance:** **Critical gap** - high variance in results and pass@k reliance suggests trajectory-level issues being masked by outcome metrics.

---

### 2.3 SWE-Bench and Code Agent Evaluation

**2.3.1 On Randomness in Agentic Evals**
- **Citation:** Bjarni Haukur Bjarnason, André Silva, Martin Monperrus
- **Venue:** arXiv:2602.07150
- **Year:** 2026

**Summary:** 60,000 trajectories on SWE-Bench-Verified reveal single-run estimates vary by 2.2-6.0 percentage points.

**Key Findings:**
- **High trajectory variance even on same task**
- Stochasticity affects outcome metrics significantly
- Multiple runs necessary for reliable estimates

**Relevance:** **Highly relevant** - suggests structural instability in trajectories that outcome metrics don't capture. Could be early planning errors causing divergent paths.

---

**2.3.2 SWE-ABS: Adversarial Benchmark Strengthening**
- **Citation:** Boxi Yu et al.
- **Venue:** arXiv:2603.00520
- **Year:** 2026

**Summary:** Two-stage adversarial testing revealed 19.71% of passing patches are semantically incorrect.

**Key Findings:**
- Outcome-based test suites miss semantic errors
- Weak test coverage inflates success rates
- Need for stronger verification

**Relevance:** **Critical** - outcome metrics mislead; aligns with our hypothesis that surface success masks structural failures.

---

**2.3.3 AdaRubric: Task-Adaptive Rubrics**
- **Citation:** Liang Ding
- **Venue:** arXiv:2603.21362
- **Year:** 2026

**Summary:** Generates task-specific rubrics achieving Pearson r=0.79 human correlation through step-by-step trajectory scoring with dimension-aware filtering.

**Key Findings:**
- Step-by-step scoring improves correlation
- Task-adaptive rubrics outperform generic
- Dimension-aware filtering enhances accuracy

**Metrics:** Pearson correlation, step-level scoring

**Relevance:** **Closest to our work** - uses step-level scoring but doesn't address position-dependent criticality or error propagation.

---

**2.3.4 Consistency Amplifies: How Behavioral Variance Shapes Agent Accuracy**
- **Citation:** Aman Mehta
- **Venue:** arXiv:2603.25764
- **Year:** 2026

**Summary:** 50 runs per model reveal higher consistency aligns with higher accuracy, yet consistency can amplify incorrect interpretations.

**Key Findings:**
- Consistency is double-edged: amplifies both correct and incorrect patterns
- Behavioral variance diagnostic of underlying capabilities
- **Consistency ≠ correctness**

**Relevance:** **Important** - suggests systematic trajectory patterns (potentially from early planning) propagate consistently, supporting error propagation hypothesis.

---

## Category 3: Error Propagation in Sequential Systems (5 papers)

### 3.1 Error Propagation Theory

**3.1.1 Information Fidelity in Tool-Using LLM Agents**
- **Citation:** Flint Xiaofeng Fan, Cheston Tan, Roger Wattenhofer, Yew-Soon Ong
- **Venue:** arXiv:2602.13320
- **Year:** 2026

**Summary:** Demonstrates cumulative distortion exhibits linear growth with high-probability deviations bounded by O(√T) in Model Context Protocol agents.

**Key Findings:**
- **Quantitative propagation framework:** errors grow linearly, not exponentially
- Semantic weighting reduces distortion by 80%
- Re-grounding every ~9 steps controls accumulation
- Theoretical bounds on error propagation

**Metrics:** Cumulative distortion, O(√T) deviation bounds

**Relevance:** **Directly relevant** - provides theoretical framework for error propagation we can build on. Linear growth suggests early errors have compounding impact.

---

**3.1.2 F-LLM: Feedback-driven LLM**
- **Citation:** Xingyu Zhang, Hanyun Du, Zeen Song, et al.
- **Venue:** arXiv:2602.12756
- **Year:** 2026

**Summary:** Control-theoretic framework with learnable residual estimator addressing how "minor early deviations cascade into significant trajectory drift."

**Key Findings:**
- **Early deviations explicitly modeled as cascading**
- Autoregressive generation amplifies early errors
- Feedback control ensures bounded error propagation

**Relevance:** **Highly relevant** - explicitly identifies early deviations as critical, supports our hypothesis about early vs. late error importance.

---

**3.1.3 CoFiCot: Confidence-Guided Fine-Grained Chain-of-Thought**
- **Citation:** Dongxu Zhang, Hongqiang Lin, Yiding Sun, et al.
- **Venue:** arXiv:2603.08251
- **Year:** 2026

**Summary:** Multi-metric classifier with stateful sequential propagation addressing context fragmentation.

**Key Findings:**
- Stateless refinement loses error context
- Sequential propagation preserves error information
- Process Reward Models insufficient without state

**Relevance:** Highlights importance of error context in trajectories.

---

**3.1.4 Dynamic Adjudication Template**
- **Citation:** Zhihao Yang, Ancheng Xu, et al.
- **Venue:** arXiv:2510.05134
- **Year:** 2025

**Summary:** Three-stage framework reducing error propagation through sequential reasoning chains.

**Key Findings:**
- Structured stages reduce propagation
- Evidence gathering critical between stages
- Adjudication requires full trajectory context

**Relevance:** Shows propagation can be controlled with architecture, but doesn't quantify criticality.

---

**3.1.5 Doctor-RAG: Failure-Aware Repair for Agentic RAG**
- **Citation:** Shuguang Jiao, Chengkai Huang, Shuhan Qi, et al.
- **Venue:** arXiv:2604.00865
- **Year:** 2026

**Summary:** Coverage-gated taxonomy for failure classification with explicit error localization and prefix reuse.

**Key Findings:**
- Error localization enables prefix reuse
- Failures classified by propagation impact
- Minimizing recomputation requires understanding error scope

**Relevance:** **Relevant** - error localization approach related to criticality; prefix reuse implies early errors invalidate more work.

---

## Category 4: Criticality Metrics (4 papers)

### 4.1 Position-Dependent Metrics

**4.1.1 Firm or Fickle? Evaluating Large Language Models Consistency in Sequential Interactions**
- **Citation:** Yubo Li, Yidi Miao, Xueying Ding, Ramayya Krishnan, Rema Padman
- **Venue:** arXiv:2503.22353
- **Year:** 2025

**Summary:** Introduces Position-Weighted Consistency (PWC), a metric capturing importance of early-stage stability in multi-turn interactions.

**Key Findings:**
- **Position-based weighting for evaluation importance**
- Early-stage stability critical for downstream performance
- Temporal position affects evaluation criteria

**Metrics:** Position-Weighted Consistency (PWC)

**Relevance:** **Highly relevant** - only paper found with explicit position-dependent scoring. PWC directly related to our trajectory criticality concept. However, applied to consistency not error impact.

---

**4.1.2 Effort-Based Criticality Metrics for Evaluating 3D Perception Errors in Autonomous Driving**
- **Citation:** Sharang Kaul, Simon Bultmann, Mario Berk, Abhinav Valada
- **Venue:** arXiv:2603.28029
- **Year:** 2026

**Summary:** Introduces effort-based metrics (FSR, MDR, LEA) distinguishing false positives from false negatives by planning/control effort required.

**Key Findings:**
- Traditional metrics conflate FP and FN consequences
- **Criticality = downstream effort/cost, not just correctness**
- 65-93% of errors are non-critical by effort metrics
- Reachability-based collision filtering identifies dynamically plausible threats

**Metrics:** False Speed Reduction (FSR), Maximum Deceleration Rate (MDR), Lateral Evasion Acceleration (LEA)

**Relevance:** **Critical cross-domain inspiration** - defines criticality as downstream impact/effort rather than local correctness. Perfect conceptual parallel: early agent planning errors require more "downstream effort" (re-planning, token waste) than late local errors.

---

### 4.2 Process Reward Models (Step-Level Scoring)

**4.2.1 Process Supervision for Chain-of-Thought Reasoning via Monte Carlo Net Information Gain**
- **Citation:** Corentin Royer, Debarun Bhattacharjya, Gaetano Rossiello, Andrea Giovannini, Mennatallah El-Assady
- **Venue:** arXiv:2603.17815
- **Year:** 2026

**Summary:** Information-theoretic approach generating step-level labels by estimating each step's effect on answer likelihood.

**Key Findings:**
- Step importance measured by information gain
- Automated step-level labeling at O(N) complexity
- Information-theoretic criticality proxy

**Metrics:** Monte Carlo Net Information Gain

**Relevance:** **Relevant** - information gain as step importance proxy, but doesn't model propagation or position effects.

---

**4.2.2 AgentProcessBench: Diagnosing Step-Level Process Quality in Tool-Using Agents**
- **Citation:** Shengda Fan, Xuyan Ye, Yupeng Huo, et al.
- **Venue:** arXiv:2603.14465
- **Year:** 2026

**Summary:** Benchmark with ternary labeling (correct/incorrect/exploratory) capturing error propagation rules for fine-grained step verification.

**Key Findings:**
- **Error propagation rules explicitly modeled**
- Ternary labels distinguish error types
- Exploration vs. error distinction critical

**Metrics:** Ternary step labels with propagation rules

**Relevance:** **Highly relevant** - explicitly models error propagation in tool-use trajectories. Could be evaluation framework for our work.

---

## Category 5: Gap Analysis and Related Work (2 papers)

### 5.1 Survey Papers

**5.1.1 Evaluation and Benchmarking of LLM Agents: A Survey**
- **Citation:** Mahmoud Mohammadi, Yipeng Li, Jane Lo, Wendy Yip
- **Venue:** arXiv:2603.07670
- **Year:** 2026

**Summary:** Comprehensive two-dimensional taxonomy organizing evaluation by objectives (behavior, capabilities, reliability, safety) and processes (interaction modes, datasets, metrics, tooling).

**Key Findings:**
- Taxonomy reveals gaps in current evaluation practices
- Multi-dimensional assessment beyond accuracy needed
- Process-oriented evaluation underexplored

**Coverage:** Comprehensive survey of agent evaluation methods

**Relevance:** **Critical for positioning** - confirms trajectory-level, position-aware criticality metrics are not systematically covered in existing work.

---

**5.1.2 A Comprehensive Survey on Benchmarks and Solutions in Software Engineering of LLM-Empowered Agentic System**
- **Citation:** Jiale Guo et al.
- **Venue:** arXiv:2510.09721
- **Year:** 2025

**Summary:** Reviews 150+ papers connecting 50+ benchmarks across code generation, translation, and repair tasks.

**Key Findings:**
- Most benchmarks use end-to-end success metrics
- Trajectory analysis limited to pass/fail logging
- Failure diagnosis remains underexplored

**Coverage:** 50+ benchmarks, 150+ papers

**Relevance:** **Gap confirmation** - extensive survey finds limited trajectory-level failure analysis, especially position-dependent criticality.

---

## Additional Key Papers (Reinforcing Themes)

### A. Outcome vs. Process Reward Models

**A.1 Rethinking Reward Models for Multi-Domain Test-Time Scaling**
- **Citation:** Dong Bok Lee, Seanie Lee, Sangwoo Park, et al.
- **Venue:** arXiv:2510.00492
- **Year:** 2025

**Key Finding:** "Discriminative ORM performs on par with discriminative PRM" across 14 domains, contradicting assumptions about PRM superiority.

**Relevance:** Questions whether process supervision alone is sufficient without criticality weighting.

---

**A.2 Stop Rewarding Hallucinated Steps: Faithfulness-Aware Step-Level Reinforcement Learning**
- **Citation:** Shuo Nie, Hexuan Deng, Chao Wang, et al.
- **Venue:** arXiv:2602.05897
- **Year:** 2026

**Key Finding:** "Outcome-based rewards can inadvertently reinforce unfaithful reasoning when the final answer is correct."

**Relevance:** **Critical** - outcome metrics mask trajectory problems, supporting need for step-level criticality analysis.

---

### B. Structural vs. Execution Errors

**B.1 APEX-EM: Memory Persistence in LLM Agents**
- **Citation:** Pratyay Banerjee, Masud Moshtaghi, Ankit Chadha
- **Venue:** arXiv:2603.29093
- **Year:** 2026

**Key Finding:** Agents "re-derive solutions from scratch even when structurally identical tasks have been solved before."

**Relevance:** Structural understanding vs. execution - planning failures are higher-level errors than execution mistakes.

---

**B.2 Environment Maps**
- **Citation:** Yenchia Feng, Chirag Sharma, Karime Maamari
- **Venue:** arXiv:2603.23610
- **Year:** 2026

**Key Finding:** "Agents frequently suffer from cascading errors" in complex workflows; persistent environmental representation doubles performance.

**Relevance:** Cascading from planning errors vs. local execution errors - supports structural criticality hypothesis.

---

## Gap Analysis: What's NOT Covered

### 1. Position-Dependent Error Criticality in Agent Trajectories
- **Existing work:** Position-Weighted Consistency (PWC) for consistency, not error impact
- **Gap:** No metrics quantifying how error position in trajectory affects downstream cost/failure
- **Our contribution:** Position-dependent criticality scoring for agent trajectories

### 2. Judge Calibration to Trajectory Structure
- **Existing work:** Judge biases (presentation order, self-preference, temperature)
- **Gap:** No analysis of whether judges distinguish planning errors from execution errors
- **Our contribution:** Systematic study of judge sensitivity to error type and position

### 3. Planning vs. Execution Error Distinction
- **Existing work:** Error propagation studied, structural vs. local errors mentioned qualitatively
- **Gap:** No quantitative framework separating structural (planning/tool-selection) from local (parameter/formatting) errors
- **Our contribution:** Taxonomy and quantitative impact analysis

### 4. Effort-Based Criticality in Agent Systems
- **Existing work:** Autonomous driving uses effort-based metrics (FSR, MDR, LEA)
- **Gap:** Not applied to agent trajectories (re-planning cost, wasted computation, token usage)
- **Our contribution:** Effort-based criticality metrics for agent evaluation

### 5. Trajectory-Level Failure Diagnosis
- **Existing work:** AgentProcessBench has error propagation rules, but no criticality weighting
- **Gap:** Error propagation modeled but not weighted by position or structural importance
- **Our contribution:** Criticality-weighted trajectory evaluation

### 6. Judge-Criticality Alignment
- **Existing work:** Judge-human alignment studied (G-Eval, etc.)
- **Gap:** No study of judge-criticality alignment (do judge scores correlate with actual impact?)
- **Our contribution:** Judge-criticality calibration metrics

---

## Novelty Statement

**Our work is the first to:**

1. **Systematically study judge calibration to error criticality** in agent trajectories, bridging gap between judge evaluation literature (Category 1) and agent evaluation literature (Category 2)

2. **Introduce position-dependent criticality metrics** for agent trajectories, extending PWC (consistency) and autonomous driving effort metrics to the agent domain

3. **Quantitatively distinguish structural (planning) from local (execution) errors** and measure judge sensitivity to this distinction

4. **Demonstrate judge blindness** to early high-criticality failures while overweighting late low-criticality errors

5. **Propose criticality-calibrated evaluation protocols** combining error propagation theory (Information Fidelity framework) with position-dependent weighting (PWC) and effort-based metrics (autonomous driving)

**Key differentiation from closest work:**
- **vs. AdaRubric:** We add position-dependent weighting to step-wise scoring
- **vs. AgentProcessBench:** We weight error propagation by criticality, not just model propagation rules
- **vs. PWC:** We extend from consistency to error impact/cost
- **vs. Information Fidelity:** We connect propagation theory to evaluation practice and judge calibration
- **vs. Effort-based Metrics (AV):** We apply effort concept to cognitive/computational cost in agent systems

---

## Supporting Evidence for Our Hypothesis

### Evidence Type 1: Judge Miscalibration Exists
- Position bias (Wang et al., 2023)
- Self-preference bias (Liu et al., G-Eval)
- Temperature sensitivity (Gameiro et al., 2024)
- Confidence miscalibration (Ghosh & Panday, 2026)

**Implication:** Judges have known systematic biases; criticality bias is plausible extension.

### Evidence Type 2: Trajectory Variance and Inconsistency
- 2.2-6.0 pp variance on SWE-Bench (Bjarnason et al., 2026)
- Consistency amplifies errors (Mehta, 2026)
- 19.71% semantic failures in passing tests (SWE-ABS, 2026)

**Implication:** Outcome metrics mask trajectory-level issues; suggests early structural instability.

### Evidence Type 3: Error Propagation is Real and Quantifiable
- Linear distortion growth O(√T) (Fan et al., 2026)
- "Minor early deviations cascade" (F-LLM, 2026)
- "Cascading errors" documented (Environment Maps, 2026)

**Implication:** Early errors have compounding impact; judges should weight them higher.

### Evidence Type 4: Outcome Metrics Mislead
- "Outcome rewards reinforce unfaithful reasoning" (Nie et al., 2026)
- ORMs and PRMs perform similarly despite theory (Lee et al., 2025)
- 65-93% of errors non-critical by effort metrics (Kaul et al., 2026)

**Implication:** Current evaluation (including judges) conflates error presence with error impact.

### Evidence Type 5: Position and Structure Matter
- PWC demonstrates position importance (Li et al., 2025)
- AgentProcessBench models error propagation rules (Fan et al., 2026)
- Planning failures cause cascades (multiple papers)

**Implication:** Position and error type are first-order factors, not secondary considerations.

---

## BibTeX Entries

```bibtex
@article{liu2023geval,
  title={G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment},
  author={Liu, Yang and Iter, Dan and Xu, Yichong and Wang, Shuohang and Xu, Ruochen and Zhu, Chenguang},
  journal={arXiv preprint arXiv:2303.16634},
  year={2023}
}

@article{wang2026dual,
  title={Dual Optimal: Make Your LLM Peer-like with Dignity},
  author={Wang, Xiangqi and Huang, Yue and Zhuang, Haomin and Guo, Kehan and Zhang, Xiangliang},
  journal={arXiv preprint arXiv:2604.00979},
  year={2026}
}

@article{daynauth2025slmeval,
  title={SLMEval: Entropy-Based Calibration},
  author={Daynauth, Roland and Clarke, Christopher and Flautner, Krisztian and Tang, Lingjia and Mars, Jason},
  journal={arXiv preprint arXiv:2505.16003},
  year={2025}
}

@article{wang2023fair,
  title={Large Language Models are not Fair Evaluators},
  author={Wang, Peiyi and Li, Lei and Chen, Liang and others},
  journal={arXiv preprint arXiv:2305.17926},
  year={2023}
}

@article{xing2025reliable,
  title={Are LLMs Reliable Rankers? Rank Manipulation via Two-Stage Token Optimization},
  author={Xing and Li and Du and Hu},
  journal={arXiv preprint arXiv:2510.06732},
  year={2025}
}

@article{gameiro2024detectors,
  title={LLM Detectors Still Fall Short of Real World},
  author={Gameiro and Kucharavy and Dolamic},
  journal={arXiv preprint arXiv:2409.03291},
  year={2024}
}

@article{ghosh2026dunning,
  title={The Dunning-Kruger Effect in Large Language Models},
  author={Ghosh, Sudipta and Panday, Mrityunjoy},
  journal={arXiv preprint arXiv:2603.09985},
  year={2026}
}

@article{liu2023agentbench,
  title={AgentBench: Evaluating LLMs as Agents},
  author={Liu, Xiao and Yu, Hao and Zhang, Hanchen and others},
  journal={arXiv preprint arXiv:2308.03688},
  year={2023}
}

@article{bjarnason2026randomness,
  title={On Randomness in Agentic Evals},
  author={Bjarnason, Bjarni Haukur and Silva, André and Monperrus, Martin},
  journal={arXiv preprint arXiv:2602.07150},
  year={2026}
}

@article{yu2026sweabs,
  title={SWE-ABS: Adversarial Benchmark Strengthening},
  author={Yu, Boxi and others},
  journal={arXiv preprint arXiv:2603.00520},
  year={2026}
}

@article{ding2026adarubric,
  title={AdaRubric: Task-Adaptive Rubrics},
  author={Ding, Liang},
  journal={arXiv preprint arXiv:2603.21362},
  year={2026}
}

@article{mehta2026consistency,
  title={Consistency Amplifies: How Behavioral Variance Shapes Agent Accuracy},
  author={Mehta, Aman},
  journal={arXiv preprint arXiv:2603.25764},
  year={2026}
}

@article{fan2026information,
  title={Information Fidelity in Tool-Using LLM Agents},
  author={Fan, Flint Xiaofeng and Tan, Cheston and Wattenhofer, Roger and Ong, Yew-Soon},
  journal={arXiv preprint arXiv:2602.13320},
  year={2026}
}

@article{zhang2026fllm,
  title={F-LLM: Feedback-driven LLM},
  author={Zhang, Xingyu and Du, Hanyun and Song, Zeen and others},
  journal={arXiv preprint arXiv:2602.12756},
  year={2026}
}

@article{zhang2026coficot,
  title={CoFiCot: Confidence-Guided Fine-Grained Chain-of-Thought},
  author={Zhang, Dongxu and Lin, Hongqiang and Sun, Yiding and others},
  journal={arXiv preprint arXiv:2603.08251},
  year={2026}
}

@article{yang2025dynamic,
  title={Dynamic Adjudication Template},
  author={Yang, Zhihao and Xu, Ancheng and others},
  journal={arXiv preprint arXiv:2510.05134},
  year={2025}
}

@article{jiao2026doctor,
  title={Doctor-RAG: Failure-Aware Repair for Agentic RAG},
  author={Jiao, Shuguang and Huang, Chengkai and Qi, Shuhan and others},
  journal={arXiv preprint arXiv:2604.00865},
  year={2026}
}

@article{li2025firm,
  title={Firm or Fickle? Evaluating Large Language Models Consistency in Sequential Interactions},
  author={Li, Yubo and Miao, Yidi and Ding, Xueying and Krishnan, Ramayya and Padman, Rema},
  journal={arXiv preprint arXiv:2503.22353},
  year={2025}
}

@article{kaul2026effort,
  title={Effort-Based Criticality Metrics for Evaluating 3D Perception Errors in Autonomous Driving},
  author={Kaul, Sharang and Bultmann, Simon and Berk, Mario and Valada, Abhinav},
  journal={arXiv preprint arXiv:2603.28029},
  year={2026}
}

@article{royer2026process,
  title={Process Supervision for Chain-of-Thought Reasoning via Monte Carlo Net Information Gain},
  author={Royer, Corentin and Bhattacharjya, Debarun and Rossiello, Gaetano and Giovannini, Andrea and El-Assady, Mennatallah},
  journal={arXiv preprint arXiv:2603.17815},
  year={2026}
}

@article{fan2026agentprocess,
  title={AgentProcessBench: Diagnosing Step-Level Process Quality in Tool-Using Agents},
  author={Fan, Shengda and Ye, Xuyan and Huo, Yupeng and others},
  journal={arXiv preprint arXiv:2603.14465},
  year={2026}
}

@article{mohammadi2026evaluation,
  title={Evaluation and Benchmarking of LLM Agents: A Survey},
  author={Mohammadi, Mahmoud and Li, Yipeng and Lo, Jane and Yip, Wendy},
  journal={arXiv preprint arXiv:2603.07670},
  year={2026}
}

@article{guo2025comprehensive,
  title={A Comprehensive Survey on Benchmarks and Solutions in Software Engineering of LLM-Empowered Agentic System},
  author={Guo, Jiale and others},
  journal={arXiv preprint arXiv:2510.09721},
  year={2025}
}

@article{lee2025rethinking,
  title={Rethinking Reward Models for Multi-Domain Test-Time Scaling},
  author={Lee, Dong Bok and Lee, Seanie and Park, Sangwoo and others},
  journal={arXiv preprint arXiv:2510.00492},
  year={2025}
}

@article{nie2026stop,
  title={Stop Rewarding Hallucinated Steps: Faithfulness-Aware Step-Level Reinforcement Learning},
  author={Nie, Shuo and Deng, Hexuan and Wang, Chao and others},
  journal={arXiv preprint arXiv:2602.05897},
  year={2026}
}

@article{banerjee2026apex,
  title={APEX-EM: Memory Persistence in LLM Agents},
  author={Banerjee, Pratyay and Moshtaghi, Masud and Chadha, Ankit},
  journal={arXiv preprint arXiv:2603.29093},
  year={2026}
}
```

---

## Research Strategy Recommendations

### 1. Experiments to Design

**Experiment 1: Judge Sensitivity to Error Position**
- Create synthetic trajectories with controlled errors at different positions
- Compare judge scores to actual downstream impact (tokens wasted, re-planning steps)
- Hypothesis: Judge penalties uncorrelated with position-dependent impact

**Experiment 2: Planning vs. Execution Error Distinction**
- Construct matched pairs: planning error (wrong tool selected) vs. execution error (wrong parameter)
- Both lead to same outcome failure
- Test if judges distinguish criticality

**Experiment 3: Criticality-Calibrated Judge Training**
- Train judges with criticality-weighted supervision
- Compare to baseline on downstream impact prediction
- Test if explicit criticality training improves calibration

### 2. Datasets to Use

**Primary:**
- SWE-Bench (high variance documented, complex trajectories)
- AgentBench environments (multi-step, diverse domains)
- GAIA (failures on unindexed information = planning issue)

**Secondary:**
- ToolBench (tool selection criticality)
- WebArena (long trajectories, cascading errors likely)

### 3. Baseline Comparisons

**Judge baselines:**
- G-Eval (foundational)
- AdaRubric (step-wise, closest to our approach)
- Generic GPT-4 judge (common practice)

**Metric baselines:**
- Outcome-only (pass/fail)
- Uniform step-level PRM (AgentProcessBench style)
- PWC adapted to errors (not just consistency)

### 4. Success Criteria

**Our work succeeds if we demonstrate:**
1. Judges systematically underweight early/structural errors (judge-criticality correlation < 0.5)
2. Criticality-weighted metrics better predict downstream cost (R² > 0.7 vs. < 0.4 for uniform weighting)
3. Effort-based criticality metrics (token waste, re-planning steps) correlate with human expert judgments (ρ > 0.6)
4. Criticality-calibrated evaluation changes rank-ordering of agents on benchmarks

---

## Key Takeaways

1. **Judge miscalibration is well-documented** (position bias, self-preference, confidence) but never studied for trajectory criticality

2. **Error propagation is theoretically grounded** (Information Fidelity paper provides O(√T) framework) but not connected to evaluation

3. **Position-dependent metrics exist** (PWC) but only for consistency, not error impact

4. **Effort-based criticality proven in autonomous driving** but not applied to agent systems

5. **Gap is clear:** No work combines judge evaluation + error propagation + position-dependent criticality + agent trajectories

6. **Closest related work:**
   - AdaRubric (step-wise scoring, no position weighting)
   - AgentProcessBench (error propagation, no criticality weighting)
   - PWC (position weighting, for consistency not errors)
   - Information Fidelity (propagation theory, not evaluation)

7. **Our contribution is novel:** First systematic study of judge blindness to trajectory criticality, with quantitative framework and calibration protocols.
