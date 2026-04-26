# Research Roadmap: LLM-as-Judge Calibration for Agent Trajectories

---

## **STOP - DO NOT EDIT WITHOUT HUMAN APPROVAL**

> **THIS FILE REQUIRES HUMAN APPROVAL BEFORE ANY MODIFICATIONS**
> 
> This document serves as the singular source of truth for the research project.
> Agents should READ this file but must NOT make updates without explicit human approval.

---

## Critical Path & Immediate Actions

### Blocking Issue
**Current experiment (dual_mode_v4) has insufficient sample size (n=88 vs target 300)**

Root cause: `baseline_verify_min_score: 98` rejects 72% of valid trajectories (score=95 with `task_would_succeed=True`)

### Immediate Action: Create dual_mode_v5
1. Copy `config/experiments/v3/poc/dual_mode_v4.json` → `dual_mode_v5.json`
2. Change `baseline_verify_min_score: 98` → `95`
3. Change perturbation distribution: `fine_grained: 0.40`, `coarse_grained: 0.40`
4. Increase trajectory targets: `toolbench: 100`, `swebench: 20`
5. Run full pipeline

### Success Metrics Summary

| RQ | Key Metric | Current | Target | Status |
|----|------------|---------|--------|--------|
| RQ-1 | Exec vs Reasoning detection gap | 62.6pp | >30pp | PASS (need larger N) |
| RQ-2 | Placebo FP rate | 20% | <15% | FAIL (investigate) |
| RQ-3 | Blinded improvement on coarse | +26pp | >10pp | PASS (need significance) |
| RQ-4 | Multi-judge agreement | N/A | κ>0.4 | NOT STARTED |
| RQ-5 | Calibration analysis | N/A | Report | NOT STARTED |

---

## Abstract

**Title:** LLM Judges Detect Execution Errors, Not Reasoning Flaws: A Controlled Study of Agent Trajectory Evaluation

LLM-as-judge methods are increasingly used to evaluate AI agent trajectories, but their reliability across different error types remains poorly understood. We present a controlled study using perturbation injection to measure judge detection capabilities. Our key finding: **judges reliably detect errors that change observable execution (72-82% for premature termination, wrong tool selection) but systematically fail to detect errors that only affect reasoning (13% for wrong planning)**. This suggests judges evaluate "did it work" rather than "was the reasoning sound," with significant implications for agent oversight and safety. We introduce a perturbation-based evaluation framework with three error classes (placebo, fine-grained, coarse-grained) and demonstrate that blinded pair comparison reduces false positives while improving structural error detection. Our results highlight when LLM-as-judge is reliable versus when human oversight remains necessary.

**Core Claims:**
1. Judges detect execution-visible errors but miss reasoning-only errors
2. Blinded comparison improves reliability over single-trajectory evaluation
3. False positive rates remain elevated without task-focused prompting
4. Findings generalize across judge models (pending multi-judge validation)

---

## Literature Review

### Category 1: LLM-as-Judge Evaluation Methods

| Paper | Contribution |
|-------|--------------|
| [G-Eval (Liu et al., 2023)](https://arxiv.org/abs/2303.16634) | Foundational LLM-as-judge with CoT; Spearman r=0.514 with humans; identifies self-preference bias |
| [Large Language Models are not Fair Evaluators (Wang et al., 2023)](https://arxiv.org/abs/2305.17926) | Demonstrates position bias: response order significantly affects judge rankings |
| [SLMEval (Daynauth et al., 2025)](https://arxiv.org/abs/2505.16003) | Shows G-Eval exhibits task-dependent failures; entropy-maximization improves calibration |
| [Dual Optimal (Wang et al., 2026)](https://arxiv.org/abs/2604.00979) | Uses Item Response Theory to disentangle model capability from judge biases |
| [Are LLMs Reliable Rankers? (Xing et al., 2025)](https://arxiv.org/abs/2510.06732) | LLM ranking systems susceptible to adversarial manipulation via natural language |
| [Dunning-Kruger in LLMs (Ghosh & Panday, 2026)](https://arxiv.org/abs/2603.09985) | Weak models display extreme overconfidence (ECE 0.726 at 23.3% accuracy) |

### Category 2: Agent Trajectory Evaluation

| Paper | Contribution |
|-------|--------------|
| [AgentBench (Liu et al., 2023)](https://arxiv.org/abs/2308.03688) | Foundational benchmark with 8 environments; outcome-based metrics dominant |
| [MCP-AgentBench (2025)](https://arxiv.org/abs/2509.09734) | Tests 188 tools across 33 MCP servers; outcome-oriented evaluation |
| [On Randomness in Agentic Evals (Bjarnason et al., 2026)](https://arxiv.org/abs/2602.07150) | 60,000 SWE-Bench trajectories show single-run estimates vary 2.2-6.0 pp |
| [SWE-ABS (Yu et al., 2026)](https://arxiv.org/abs/2603.00520) | 19.71% of passing patches are semantically incorrect due to weak test coverage |
| [AdaRubric (Ding, 2026)](https://arxiv.org/abs/2603.21362) | Step-by-step trajectory scoring achieves Pearson r=0.79 with humans |
| [ToolLLM (Qin et al., 2023)](https://arxiv.org/abs/2307.16789) | Depth-first search for multi-tool scenarios |

### Category 3: Error Propagation in Sequential Systems

| Paper | Contribution |
|-------|--------------|
| **[Information Fidelity in Tool-Using LLM Agents (Fan et al., 2026)](https://arxiv.org/abs/2602.13320)** | Cumulative distortion grows linearly with O(√T) bounds; semantic weighting reduces 80% |
| **[F-LLM (Zhang et al., 2026)](https://arxiv.org/abs/2602.12756)** | "Minor early deviations cascade into significant trajectory drift" |
| [CoFiCot (Zhang et al., 2026)](https://arxiv.org/abs/2603.08251) | Stateful sequential propagation preserves error context |
| [Doctor-RAG (Jiao et al., 2026)](https://arxiv.org/abs/2604.00865) | Coverage-gated taxonomy with error localization enabling prefix reuse |

### Category 4: Criticality Metrics

| Paper | Contribution |
|-------|--------------|
| **[Firm or Fickle? (Li et al., 2025)](https://arxiv.org/abs/2503.22353)** | Introduces Position-Weighted Consistency (PWC); captures early-stage stability importance |
| **[Effort-Based Criticality for AV (Kaul et al., 2026)](https://arxiv.org/abs/2603.28029)** | Criticality as downstream effort/cost; 65-93% of errors non-critical by impact |
| [Process Supervision via Monte Carlo (Royer et al., 2026)](https://arxiv.org/abs/2603.17815) | Information gain as criticality proxy |
| [AgentProcessBench (Fan et al., 2026)](https://arxiv.org/abs/2603.14465) | Ternary labeling with explicit error propagation rules |

### Category 5: Supporting Work

| Paper | Contribution |
|-------|--------------|
| [LLM Agent Evaluation Survey (Mohammadi et al., 2026)](https://arxiv.org/abs/2603.07670) | Confirms trajectory-level, position-aware criticality metrics underexplored |
| [Agentic Systems Benchmark Survey (Guo et al., 2025)](https://arxiv.org/abs/2510.09721) | Reviews 150+ papers; identifies limited trajectory-level failure analysis |
| [Stop Rewarding Hallucinated Steps (Nie et al., 2026)](https://arxiv.org/abs/2602.05897) | Outcome-based rewards reinforce unfaithful reasoning when final answer correct |

**Research Gap:** No prior work systematically studies judge calibration to error criticality in agent trajectories or quantitatively distinguishes structural (planning) vs. local (execution) errors.

---

## Setup

### Pipeline Architecture (Block Diagram)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        EXPERIMENT PIPELINE OVERVIEW                              │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   DATA SOURCES   │     │   DATA SOURCES   │     │   DATA SOURCES   │
│                  │     │                  │     │                  │
│    TOOLBENCH     │     │    SWE-BENCH     │     │   GAIA (future)  │
│   (API tasks)    │     │   (code edits)   │     │   (multi-step)   │
│    75 trajs      │     │    25 trajs      │     │    0 trajs       │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: LOAD                                                                   │
│  • Load trajectories from benchmarks                                             │
│  • Baseline verification with Claude (min_score=98)                             │
│  • Store in MongoDB: trajectories collection                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: TYPING                                                                 │
│  • Enrich trajectories with semantic types                                       │
│  • Identify: step_role, perturbable_slots, dependencies, critical_path          │
│  • Store in MongoDB: typed_trajectories collection                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: PERTURB                                                                │
│                                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐          │
│  │    PLACEBO      │  │  FINE-GRAINED   │  │    COARSE-GRAINED       │          │
│  │    (20%)        │  │     (50%)       │  │       (30%)             │          │
│  │                 │  │                 │  │                         │          │
│  │ • formatting    │  │ • wrong_value   │  │ • false_terminal        │          │
│  │ • synonym       │  │ • off_by_one    │  │ • premature_termination │          │
│  │ • paraphrase    │  │ • threshold     │  │ • wrong_tool_family     │          │
│  │                 │  │ • query_drift   │  │ • wrong_plan            │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘          │
│                                                                                  │
│  • LLM-based generation for: paraphrase, value_mutation, wrong_plan             │
│  • Class validation with Claude                                                  │
│  • Target: 300 perturbations                                                     │
│  • Store in MongoDB: perturbed_trajectories collection                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: EVALUATION_UNIT                                                        │
│  • Pair baseline + perturbed trajectories                                        │
│  • Deterministic hash-based blinding (A/B assignment)                           │
│  • Balance verification: 35% ≤ ratio ≤ 65%                                      │
│  • Store in MongoDB: evaluation_units collection                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 5: JUDGE                                                                  │
│                                                                                  │
│  ┌─────────────────────────┐     ┌─────────────────────────┐                   │
│  │   SINGLE_TRAJECTORY     │     │     BLINDED_PAIR        │                   │
│  │                         │     │                         │                   │
│  │  Evaluate one           │     │  Compare A vs B         │                   │
│  │  trajectory in          │     │  (order randomized)     │                   │
│  │  isolation              │     │  without knowing        │                   │
│  │                         │     │  which is perturbed     │                   │
│  └─────────────────────────┘     └─────────────────────────┘                   │
│                                                                                  │
│  • Model: claude-sonnet-4 via AWS Bedrock                                       │
│  • Temperature: 0.0 (deterministic)                                             │
│  • Prompts: V3 (task-focused, reduces false positives)                          │
│  • Store in MongoDB: judge_eval_outputs collection                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 6: COMPUTE                                                                │
│  • Aggregate judge decisions by perturbation class/type                         │
│  • Compute: PDR (detection rate), FP rate, agreement metrics                    │
│  • Per-version comparison (single vs blinded)                                   │
│  • Store in MongoDB: metrics collection                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### How to Run

```bash
# Run full pipeline
python main.py --config v3/poc/dual_mode_v5 --phase load,typing,perturb,judge,compute

# Run specific phases
python main.py --config v3/poc/dual_mode_v5 --phase judge,compute
```

### Key Configuration

**Config file:** `config/experiments/v3/poc/dual_mode_v5.json` (to be created)

**MongoDB:** `mongodb+srv://...@judging-the-agents.zzzls0n.mongodb.net/agent_judge_experiment`

---

## Experiment Design & Quality Gates

### Current Bottleneck Analysis

| Metric | dual_mode_v4 | Problem | Fix |
|--------|--------------|---------|-----|
| Trajectories loaded | 32/100 target | `baseline_verify_min_score: 98` too strict | Lower to 95 |
| Score distribution | 9×100, 23×95 | Score=95 are valid (task_would_succeed=True) | 95 is acceptable |
| Perturbations | 88/300 target | Not enough source trajectories | More trajectories |
| Evaluation units | 88 | Same | Same |

### Target Experiment: dual_mode_v5

**Sample Size Targets:**

| Stage | Target | Minimum | Rationale |
|-------|--------|---------|-----------|
| Trajectories loaded | 120 | 100 | Need ~100 for 300 perturbations at 3/traj |
| Baseline verification pass | 95% | 90% | Lower threshold to 95 |
| Perturbations generated | 300 | 250 | Statistical power for per-type analysis |
| Evaluation units | 300 | 250 | Same as perturbations |
| Judge evaluations | 600 | 500 | 2 modes × perturbations |

**Perturbation Distribution Targets:**

| Class | Target % | Target N | Min N per Type |
|-------|----------|----------|----------------|
| Placebo | 20% | 60 | 20 per type (3 types) |
| Fine-grained | 40% | 120 | 30 per type (4 types) |
| Coarse-grained | 40% | 120 | 30 per type (4 types) |

**Why 40/40 instead of 50/30?** We need more coarse-grained samples to test the execution-visibility hypothesis with per-type breakdowns.

**Type-Level Minimum Sample Sizes:**

| Type | Class | Min N | Rationale |
|------|-------|-------|-----------|
| paraphrase | placebo | 30 | FP rate estimation |
| synonym | placebo | 15 | FP rate estimation |
| formatting | placebo | 15 | FP rate estimation |
| wrong_parameter | fine | 40 | Core fine-grained type |
| off_by_one | fine | 30 | Subtle error type |
| threshold_shift | fine | 25 | Subtle error type |
| query_drift | fine | 25 | Subtle error type |
| premature_termination | coarse | 40 | Execution-changing (expect HIGH detection) |
| wrong_tool_family | coarse | 30 | Execution-changing (expect HIGH detection) |
| wrong_plan | coarse | 30 | Reasoning-only (expect LOW detection) |
| false_terminal | coarse | 20 | Reasoning-only (expect LOW detection) |

### Quality Gates (Go/No-Go Criteria)

**Phase 1: Load**
- [ ] Trajectories loaded >= 100
- [ ] Baseline verification pass rate >= 90%
- [ ] At least 2 benchmarks represented (ToolBench required, SWE-bench desired)

**Phase 2: Perturb**
- [ ] Total perturbations >= 250
- [ ] Class validation pass rate >= 85%
- [ ] Each perturbation type has N >= 15
- [ ] No single type > 25% of total

**Phase 3: Judge**
- [ ] Parse success rate >= 95%
- [ ] Both modes (single + blinded) complete
- [ ] No systematic failures by perturbation type

**Phase 4: Metrics (RQ Quality Gates)**

| Metric | Threshold | Action if Fail |
|--------|-----------|----------------|
| Placebo FP rate | < 20% | Investigate baseline quality or prompt |
| Exec-changing detection (PT+WTF) | > 60% | Good signal; proceed |
| Reasoning-only detection (WP+FT) | < 40% | Expected; confirms hypothesis |
| Statistical significance | p < 0.05 for main comparison | Need larger N or clearer effect |

### Configuration Changes for dual_mode_v5

```json
{
  "phases": {
    "load": {
      "baseline_verify_min_score": 95,  // Changed from 98
      "sampling": {
        "targets": {
          "toolbench": 100,  // Increased from 75
          "swebench": 20     // Reduced - has issues
        }
      }
    },
    "perturb": {
      "targets": {
        "total_perturbations": 300,
        "by_class": {
          "placebo": 0.20,
          "fine_grained": 0.40,  // Changed from 0.50
          "coarse_grained": 0.40  // Changed from 0.30
        }
      }
    }
  }
}
```

---

## Timelines

**Target Venue:** EMNLP 2025  
**Deadline:** May 25, 2025  
**Today:** April 16, 2026  
**Time Remaining:** ~5.5 weeks  

| Phase | Duration | Start | End | Status |
|-------|----------|-------|-----|--------|
| Phase 1: Core Experiments (RQ1-RQ3) | 1 week | Apr 16 | Apr 23 | IN PROGRESS |
| Phase 2: FP Investigation + Scale Up | 4 days | Apr 24 | Apr 27 | NOT STARTED |
| Phase 3: Multi-Judge Run (RQ4) | 3 days | Apr 28 | Apr 30 | NOT STARTED |
| Phase 4: Analysis + Figures | 4 days | May 1 | May 4 | NOT STARTED |
| Phase 5: Draft Writing | 10 days | May 5 | May 14 | NOT STARTED |
| Phase 6: Revision + Polish | 10 days | May 15 | May 24 | NOT STARTED |
| **Submission** | - | **May 25** | - | - |

---

## Research Questions

---

### RQ-1: Detectability by Execution Visibility

**Hypothesis:** LLM judges detect errors based on execution visibility rather than error severity. Errors that change observable execution (tool calls, outputs) are detected at higher rates than errors that only affect reasoning/planning text.

**Dates:**  
- Start: April 16, 2026  
- End: April 23, 2026  

**Experiment Config:**  
`config/experiments/v3/poc/dual_mode_v5.json`

**Quality Gates & Success Criteria:**

| Metric | Target | Minimum | Status |
|--------|--------|---------|--------|
| Sample size (total) | 300 | 250 | BLOCKED (n=88) |
| Exec-changing types (PT+WTF) | n≥60 | n≥40 | BLOCKED (n=15) |
| Reasoning-only types (WP+FT) | n≥50 | n≥30 | BLOCKED (n=31) |
| Detection gap (exec vs reasoning) | >30pp | >20pp | PASS (65pp gap) |
| Statistical significance | p<0.01 | p<0.05 | TBD |
| Effect size (Cramér's V) | >0.3 | >0.2 | TBD |

**Claim Criteria:**
- STRONG CLAIM: Exec-changing >70% AND Reasoning-only <30% AND gap >40pp AND p<0.01
- MODERATE CLAIM: Exec-changing >60% AND Reasoning-only <40% AND gap >20pp AND p<0.05
- WEAK/NO CLAIM: Otherwise

**Current Findings (dual_mode_v4, n=88):**
| Type | Class | N | Detection (blinded) |
|------|-------|---|---------------------|
| premature_termination | coarse/exec | 11 | 81.8% |
| wrong_tool_family | coarse/exec | 4 | 75.0% |
| wrong_plan | coarse/reason | 23 | 13.0% |
| false_terminal | coarse/reason | 8 | 25.0% |
| wrong_parameter | fine | 22 | 54.5% |

**Preliminary:** Exec-changing (78.7%) vs Reasoning-only (16.1%) = **62.6pp gap** - supports STRONG CLAIM if replicated at scale.

**Next Steps:**
- [ ] Create dual_mode_v5 config with baseline_verify_min_score=95
- [ ] Run full pipeline to get n≥250
- [ ] Ensure exec-changing types have n≥40 combined
- [ ] Run Fisher's exact test for exec vs reasoning comparison
- [ ] Compute 95% CI for detection rates
- [ ] Report Cramér's V effect size

---

### RQ-2: False Positive Characterization

**Hypothesis:** Elevated false positive rates on placebos (20-25%) are caused by one or more of: (a) pre-existing issues in baseline trajectories, (b) judge over-sensitivity to stylistic differences, (c) trajectory complexity correlation.

**Dates:**  
- Start: April 18, 2026  
- End: April 23, 2026  

**Experiment Config:**  
Same as RQ-1, with additional analysis scripts

**Quality Gates & Success Criteria:**

| Metric | Target | Minimum | Status |
|--------|--------|---------|--------|
| Placebo sample size | n≥60 | n≥40 | BLOCKED (n=20) |
| FP rate (blinded) | <10% | <15% | FAIL (20%) |
| FP rate (single) | <15% | <20% | FAIL (25%) |
| Root cause identified | Yes | Partial | TBD |

**Diagnosis Checklist:**
- [ ] H2a: Pre-existing baseline errors → Check baseline_score of FP cases
- [ ] H2b: Stylistic over-sensitivity → Manual review of judge reasoning
- [ ] H2c: Complexity correlation → Correlate FP with step count

**Remediation Options:**
| If Root Cause | Fix | Effort |
|---------------|-----|--------|
| Pre-existing errors | Filter baselines with baseline_score=100 only | Low |
| Stylistic sensitivity | Create V4 prompt with explicit "ignore style" | Medium |
| Complexity | Stratify analysis by trajectory length | Low |

**Current Findings (dual_mode_v4, n=20 placebos):**
| Type | N | FP (blinded) | FP (single) |
|------|---|--------------|-------------|
| paraphrase | 14 | 21.4% | 21.4% |
| synonym | 6 | 16.7% | 33.3% |

**Next Steps:**
- [ ] Query MongoDB for FP cases - examine judge reasoning
- [ ] Cross-reference FP cases with baseline_score (100 vs 95)
- [ ] If baseline_score=95 → FP, add filter for 100-only baselines
- [ ] If FP on score=100 baselines, examine judge explanation
- [ ] Scale up placebo sample to n≥40
- [ ] Re-measure FP rate after fixes

---

### RQ-3: Blinded vs Single-Trajectory Mode

**Hypothesis:** Blinded pair comparison improves judge reliability over single-trajectory evaluation by reducing scale drift and improving structural error detection.

**Dates:**  
- Start: April 16, 2026  
- End: April 23, 2026  

**Experiment Config:**  
`config/experiments/v3/poc/dual_mode_v5.json`  
Both modes run on same evaluation units (paired design)

**Quality Gates & Success Criteria:**

| Metric | Target | Minimum | Status |
|--------|--------|---------|--------|
| Sample size | n≥250 | n≥200 | BLOCKED (n=88) |
| FP reduction (blinded vs single) | >10pp | >5pp | MARGINAL (5pp) |
| Coarse detection improvement | >15pp | >10pp | PASS (26pp) |
| McNemar test significance | p<0.05 | p<0.10 | TBD |

**Claim Criteria:**
- STRONG CLAIM: Blinded beats Single on FP AND coarse detection AND p<0.05
- MODERATE CLAIM: Blinded beats Single on ≥1 metric with p<0.10
- NO CLAIM: No significant difference

**Current Findings (dual_mode_v4, n=88):**
| Metric | Single | Blinded | Δ |
|--------|--------|---------|---|
| Placebo FP | 25.0% | 20.0% | -5pp |
| Fine-grained | 54.5% | 54.5% | 0pp |
| Coarse (exec) | 52.3% | 78.7% | **+26pp** |
| Coarse (reason) | 17.4% | 16.1% | -1pp |

**Preliminary:** Blinded significantly improves exec-changing coarse detection (+26pp). Main benefit is for structural errors, not fine-grained.

**Next Steps:**
- [ ] Scale to n≥200 for statistical power
- [ ] Run McNemar's test (paired comparison)
- [ ] Stratify by perturbation type to identify where blinded helps most
- [ ] Compute 95% CI for mode differences
- [ ] Check for position bias in blinded mode (A vs B preference)

---

### RQ-4: Multi-Judge Robustness

**Hypothesis:** If the execution-visibility finding is fundamental (not model-specific), detection patterns should replicate across different judge models (Claude vs GPT).

**Dates:**  
- Start: April 28, 2026  
- End: April 30, 2026  

**Experiment Config:**  
`config/experiments/v3/poc/dual_mode_v5_gpt.json` (to be created)  
Same evaluation units, different judge model

**Quality Gates & Success Criteria:**

| Metric | Target | Minimum | Status |
|--------|--------|---------|--------|
| Second judge runs | Complete | Complete | NOT STARTED |
| Pattern replication | Same direction | Same direction | TBD |
| Inter-judge agreement (κ) | >0.6 | >0.4 | TBD |
| Detection correlation (r) | >0.7 | >0.5 | TBD |

**Claim Criteria:**
- STRONG CLAIM: Both judges show exec>reasoning AND κ>0.6 AND r>0.7
- MODERATE CLAIM: Both judges show same direction AND κ>0.4
- WEAK CLAIM: Same pattern but low agreement (model-specific calibration)
- NO CLAIM: Different patterns (finding is Claude-specific)

**Judge Models to Test:**
| Model | Provider | Status | Notes |
|-------|----------|--------|-------|
| Claude Sonnet 4 | Bedrock | DONE | Primary judge |
| GPT-4o | Bedrock/OpenAI | TBD | Second judge |
| Claude Opus 4 | Bedrock | OPTIONAL | If budget allows |

**Current Findings:**
- Only Claude Sonnet 4 tested
- No multi-judge data yet

**Next Steps:**
- [ ] Create dual_mode_v5_gpt config (copy v5, change judge model)
- [ ] Run judge phase only on existing evaluation units
- [ ] Compute per-type detection rates for GPT
- [ ] Compare Claude vs GPT detection patterns
- [ ] Compute Cohen's κ for binary detection agreement
- [ ] Compute Pearson r for detection rate correlation by type

---

### RQ-5: Calibration to Consequences (Stretch)

**Hypothesis:** Judges may detect errors but mis-estimate their impact. The `predicted_failure_prob` from judge output may not correlate well with actual perturbation severity.

**Dates:**  
- Start: May 1, 2026  
- End: May 4, 2026  

**Experiment Config:**  
Same as RQ-1 (analysis of existing judge outputs)

**Quality Gates & Success Criteria:**

| Metric | Target | Minimum | Status |
|--------|--------|---------|--------|
| Impact score availability | >90% | >80% | TBD |
| Correlation with class | Report | Report | TBD |
| Calibration analysis | Complete | Partial | TBD |

**Analysis Plan:**
1. Use `predicted_failure_prob` and `predicted_impact_score` from judge outputs
2. Use perturbation class as severity proxy: placebo(0) < fine(1) < coarse(2)
3. For coarse: reasoning-only(2a) vs exec-changing(2b)

**Claim Criteria:**
- FINDING: Report calibration curves regardless of result
- POSITIVE: Good correlation (r>0.5) = judges estimate impact well
- NEGATIVE: Poor correlation (r<0.3) = judges detect but don't price correctly
- INTERESTING: Detection ≠ Calibration (high detection, poor calibration)

**Current Findings:**
- Judge outputs include `predicted_failure_prob` (0-1)
- Judge outputs include `predicted_impact_score` (0-1)
- Not yet analyzed

**Next Steps:**
- [ ] Extract impact scores from judge_eval_outputs
- [ ] Compute Spearman correlation with perturbation class
- [ ] Create calibration curve: predicted vs "actual" (class-based proxy)
- [ ] Stratify by detection (detected vs missed errors)
- [ ] If time: implement Tier 3 outcome scoring for ground truth
- [ ] Report as finding regardless of direction

---

## Paper Progress

---

### Draft 2-Pager Ready

**Dates:**  
- Target: May 7, 2026  
- Actual: TBD  

**Feedback:**  
_Awaiting draft completion_

---

### Draft 6-Pager Ready

**Dates:**  
- Target: May 12, 2026  
- Actual: TBD  

**Feedback:**  
_Awaiting draft completion_

---

### Draft EMNLP Paper Ready

**Dates:**  
- Target: May 18, 2026  
- Actual: TBD  

**Feedback:**  
_Awaiting draft completion_

---

### Final Submission

**Dates:**  
- Target: **May 25, 2026**  
- Actual: TBD  

**Feedback:**  
_Awaiting submission_

---

## Appendix: Key File Paths

| Purpose | Path |
|---------|------|
| Main config | `config/experiments/v3/poc/dual_mode_v4.json` |
| Judge prompts | `src/prompts/judge_prompts.py` |
| Perturbation prompts | `src/prompts/perturbation_prompts.py` |
| Current results | `agent_tasks/10_perturbations_pdr/results.md` |
| Pivot roadmap | `agent_tasks/06_paper_pivot/Pivot roadmap...pdf` |
| Architecture | `agent_tasks/08_paper_pivot_simplified/architecture_diagram.excalidraw.json` |
| Literature review | `paper/LITERATURE_REVIEW.MD` |

---

*Last updated: April 16, 2026*
