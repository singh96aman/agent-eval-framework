# Research Roadmap: Judge Blindness to Trajectory Criticality

**Based on Literature Review - April 2, 2026**

---

## Phase 1: Foundation and Gap Confirmation (Weeks 1-2)

### 1.1 Theoretical Framework Development
**Goal:** Formalize criticality metrics building on existing theory

**Key Papers to Build On:**
- Information Fidelity (Fan et al., 2026) - O(√T) propagation bounds
- PWC (Li et al., 2025) - Position-dependent weighting
- AV Effort Metrics (Kaul et al., 2026) - Criticality definition

**Deliverables:**
- Mathematical formulation of Criticality-Weighted Process Reward (CWPR)
- Definition of effort-based criticality for agents:
  - Token waste metric
  - Re-planning depth metric
  - Cascade invalidation metric
- Position-weighting function for trajectory steps

**Implementation:**
```
CWPR(trajectory) = Σ w(i) × impact(i) × PRM(step_i)

where:
  w(i) = position_weight(i, T)  # based on PWC
  impact(i) = downstream_cost(i)  # inspired by AV effort
  PRM(step_i) = process_reward_model score
```

### 1.2 Dataset Selection and Preparation
**Goal:** Identify datasets with rich trajectory information

**Selected Datasets:**
1. **SWE-Bench** (primary)
   - Why: High variance documented (2.2-6.0pp), complex trajectories
   - Use: 500 instances with diverse trajectory lengths
   
2. **AgentBench** (secondary)
   - Why: Multi-domain, 8 environments
   - Use: Subset from OS/DB/KnowledgeGraph tasks
   
3. **GAIA** (tertiary)
   - Why: Known failures on unindexed information (planning issues)
   - Use: Level 2-3 tasks with multi-step requirements

**Annotation Protocol:**
- Collect full trajectories (not just outcomes)
- Label error types: planning, tool selection, parameter, formatting
- Measure ground truth downstream cost (tokens to recovery, steps invalidated)
- Expert human criticality ratings (3 annotators, weighted kappa > 0.7)

---

## Phase 2: Judge Baseline Analysis (Weeks 3-4)

### 2.1 Experiment 1: Judge-Criticality Correlation
**Hypothesis:** Judge scores uncorrelated with position-dependent criticality (r < 0.5)

**Method:**
1. Create synthetic trajectory variants:
   - Same outcome failure
   - Errors at different positions (early/middle/late)
   - Controlled error types (planning vs. execution)

2. Measure judge responses:
   - G-Eval (foundational baseline)
   - GPT-4 generic judge (common practice)
   - AdaRubric (step-wise baseline)

3. Measure ground truth criticality:
   - Actual token waste to recovery
   - Number of steps invalidated
   - Re-planning depth required

4. Compute Judge-Criticality Correlation (JCC):
   ```
   JCC = correlation(judge_penalty, ground_truth_cost)
   ```

**Expected Results:**
- G-Eval: JCC ≈ 0.2-0.4 (weak correlation)
- GPT-4 generic: JCC ≈ 0.1-0.3 (very weak)
- AdaRubric: JCC ≈ 0.3-0.5 (moderate, best baseline)

**Success Criteria:** Demonstrate JCC < 0.5 for all baselines

---

### 2.2 Experiment 2: Planning vs. Execution Error Discrimination
**Hypothesis:** Judges fail to distinguish criticality (similar scores despite different impact)

**Method:**
1. Construct matched pairs:
   - **Pair A:** Planning error (wrong tool selected) → 20 steps invalidated
   - **Pair B:** Execution error (wrong parameter value) → 2 steps affected
   - Both lead to same outcome failure

2. Present to judges without outcome information (trajectory-only evaluation)

3. Measure:
   - Judge penalty difference: |score(A) - score(B)|
   - Ground truth cost difference: |cost(A) - cost(B)|
   - Discrimination ratio: judge_diff / ground_truth_diff

**Expected Results:**
- Judges give similar penalties (discrimination ratio < 0.3)
- Ground truth costs differ by 10x
- Judges weight both error types similarly despite impact difference

**Success Criteria:** Discrimination ratio < 0.4 (judges miss criticality distinction)

---

### 2.3 Analysis: Error Type Sensitivity
**Goal:** Decompose judge bias by error type and position

**Method:**
- Error taxonomy: planning, tool selection, parameter, formatting, logical
- Position taxonomy: first quartile, second, third, fourth
- 2D analysis: error_type × position → judge_penalty
- Compare to: error_type × position → ground_truth_cost

**Deliverables:**
- Heatmap of judge sensitivity vs. ground truth
- Identification of systematically underweighted errors (planning + early position)
- Identification of systematically overweighted errors (formatting + late position)

---

## Phase 3: Criticality-Calibrated Methods (Weeks 5-7)

### 3.1 Method 1: CWPR (Criticality-Weighted Process Reward)
**Goal:** Extend process rewards with position and effort weighting

**Implementation:**
```python
def cwpr_score(trajectory, prm):
    T = len(trajectory)
    score = 0
    
    for i, step in enumerate(trajectory):
        # Position weight (inspired by PWC)
        w_pos = position_decay(i, T)  # Higher for early steps
        
        # Effort weight (inspired by AV metrics)
        w_effort = downstream_cost(step, trajectory[i+1:])
        
        # Process reward
        prm_score = prm.score(step, context=trajectory[:i])
        
        # Combined
        score += w_pos * w_effort * prm_score
    
    return score / T  # Normalized
```

**Position Decay Options to Test:**
- Linear: w(i) = 1 - (i/T)
- Exponential: w(i) = exp(-λ × i/T)
- PWC-inspired: w(i) from Li et al. consistency weighting

**Effort Computation:**
- Token waste: If step_i fails, count tokens in trajectory[i+1:] that must be redone
- Cascade depth: If step_i invalid, count subsequent dependent steps
- Re-planning cost: Estimate tokens needed to return to valid state

---

### 3.2 Method 2: Effort-Based Criticality Metrics
**Goal:** Define agent-specific effort metrics (analogous to AV FSR/MDR/LEA)

**Proposed Metrics:**

1. **Token Waste Rate (TWR)** - analogous to False Speed Reduction
   ```
   TWR = Σ (tokens in invalidated paths) / total_tokens
   ```

2. **Maximum Re-planning Depth (MRD)** - analogous to Maximum Deceleration Rate
   ```
   MRD = max(steps_from_error_to_recovery)
   ```

3. **Cascade Invalidation Scope (CIS)** - analogous to Lateral Evasion Acceleration
   ```
   CIS = number_of_dependent_steps_invalidated
   ```

**Validation:**
- Correlate with human expert ratings of criticality
- Test if TWR/MRD/CIS distinguish planning from execution errors
- Target: Spearman ρ > 0.6 with expert judgments

---

### 3.3 Method 3: Criticality-Calibrated Judge Training
**Goal:** Train judges with explicit criticality supervision

**Training Data Generation:**
1. Collect trajectories from Phase 2
2. Annotate with ground truth criticality (TWR, MRD, CIS)
3. Generate criticality-weighted targets:
   ```
   target_score = base_score × criticality_multiplier
   where criticality_multiplier = f(error_type, position, effort)
   ```

**Training Protocol:**
- Fine-tune judge model (e.g., Llama-3-70B) with:
  - Input: trajectory up to step i
  - Output: criticality-aware score
  - Loss: MSE to ground truth downstream cost
- Alternative: Prompt engineering with few-shot examples highlighting criticality

**Evaluation:**
- JCC (Judge-Criticality Correlation) on held-out data
- Target: JCC > 0.7 (vs. < 0.5 for baselines)

---

## Phase 4: Comprehensive Evaluation (Weeks 8-10)

### 4.1 Benchmark Comparison
**Goal:** Test all methods on real benchmarks

**Benchmarks:**
- SWE-Bench (500 instances)
- AgentBench (selected environments)
- GAIA (level 2-3)

**Metrics to Report:**
1. Judge-Criticality Correlation (JCC)
2. Rank correlation with human expert rankings
3. Discrimination ratio (planning vs. execution errors)
4. Position sensitivity (early vs. late error weighting)
5. Prediction of downstream cost (R² for TWR, MRD, CIS)

**Baselines:**
- Outcome-only (binary pass/fail)
- G-Eval (foundational judge)
- GPT-4 generic judge
- AdaRubric (step-wise)
- Uniform PRM (AgentProcessBench style)

**Proposed Methods:**
- CWPR (position + effort weighted)
- Effort metrics (TWR/MRD/CIS)
- Calibrated judge (criticality-trained)

---

### 4.2 Ablation Studies

**Ablation 1: Position Weighting**
- No position weight (w_pos = 1)
- Linear position weight
- Exponential position weight
- PWC-inspired weight

**Ablation 2: Effort Components**
- Token waste only
- Cascade depth only
- Re-planning cost only
- Combined (full model)

**Ablation 3: Error Type Sensitivity**
- Uniform weighting across types
- Planning-only boosting
- Execution-only boosting
- Learned type weights

**Analysis:** Which components contribute most to JCC improvement?

---

### 4.3 Human Evaluation
**Goal:** Validate that criticality-calibrated methods align with experts

**Protocol:**
1. Sample 100 trajectories with errors
2. Show to 5 expert evaluators (agent system researchers)
3. Collect:
   - Criticality rating (1-5 scale)
   - Error type classification
   - Estimated recovery cost
4. Compare expert ratings to:
   - Baseline judge scores
   - CWPR scores
   - Effort metrics (TWR/MRD/CIS)

**Analysis:**
- Spearman correlation with expert criticality ratings
- Agreement on error type importance ranking
- Qualitative feedback on failure cases

**Target:** ρ > 0.7 for proposed methods, > 0.4 for baselines

---

## Phase 5: Paper Writing and Artifact Release (Weeks 11-12)

### 5.1 Paper Structure

**Title Options:**
- "Judge Blindness to Trajectory Criticality in LLM-Evaluated Agent Systems"
- "Do LLM Judges Understand Error Criticality? A Study of Position-Dependent Impact in Agent Trajectories"
- "Criticality-Calibrated Evaluation: Teaching LLM Judges to Weight Agent Errors by Downstream Cost"

**Sections:**

1. **Introduction**
   - Motivation: Outcome metrics mislead, judges may too
   - Hypothesis: Judges miscalibrated to trajectory criticality
   - Contributions: First study + metrics + calibration methods

2. **Related Work** (15-18 papers)
   - LLM judges and biases (G-Eval, position bias, calibration)
   - Agent evaluation (AgentBench, SWE-Bench, AdaRubric)
   - Error propagation (Information Fidelity, F-LLM)
   - Criticality metrics (PWC, AV effort metrics)

3. **Problem Formulation**
   - Trajectory representation
   - Criticality definition
   - Judge-Criticality Correlation (JCC) metric

4. **Methods**
   - CWPR: Position + effort weighted process rewards
   - Effort metrics: TWR, MRD, CIS
   - Calibrated judge training

5. **Experiments**
   - Exp 1: Judge-criticality correlation (JCC < 0.5 for baselines)
   - Exp 2: Planning vs. execution discrimination (ratio < 0.4)
   - Exp 3: Calibrated methods improve JCC > 0.7

6. **Results**
   - Baselines miscalibrated (visualizations, heatmaps)
   - Proposed methods improve correlation
   - Ablations show position + effort both matter
   - Human evaluation validates

7. **Discussion**
   - Implications for benchmark design
   - Importance of trajectory-level evaluation
   - Limitations and future work

8. **Conclusion**
   - Judges are blind to criticality (demonstrated)
   - Calibration methods work (evidence provided)
   - Call for criticality-aware evaluation protocols

---

### 5.2 Artifact Release

**Code Repository:**
- Trajectory annotation toolkit
- CWPR implementation
- Effort metric computation (TWR/MRD/CIS)
- Evaluation scripts (JCC, discrimination ratio)
- Judge fine-tuning code

**Data:**
- Annotated trajectories from SWE-Bench/AgentBench/GAIA
- Error type labels
- Ground truth criticality scores
- Human expert ratings

**Models:**
- Criticality-calibrated judge (fine-tuned checkpoint)
- Few-shot prompts for calibration

**Benchmarks:**
- Criticality-weighted leaderboard for SWE-Bench
- Diagnostic suite for testing judge calibration

---

## Success Criteria Summary

### Empirical Claims to Demonstrate:

1. **Judge Miscalibration (Core Claim)**
   - JCC < 0.5 for G-Eval, GPT-4, AdaRubric
   - Discrimination ratio < 0.4 for planning vs. execution errors
   - Position bias: judges weight late errors similarly to early errors despite cost difference

2. **Criticality Metrics Work**
   - JCC > 0.7 for CWPR (vs. < 0.5 for baselines)
   - ρ > 0.6 between effort metrics (TWR/MRD/CIS) and human expert ratings
   - R² > 0.7 for predicting downstream cost (vs. < 0.4 for outcome metrics)

3. **Calibration Improves Judges**
   - Calibrated judge: JCC > 0.7 (vs. < 0.5 before calibration)
   - Maintains or improves outcome prediction accuracy
   - Generalizes across benchmarks (SWE-Bench, AgentBench, GAIA)

### Theoretical Contributions:

1. **Formalization of trajectory criticality**
   - Position-dependent weighting function
   - Effort-based metrics for agent systems
   - Judge-Criticality Correlation (JCC) metric

2. **Connection of existing frameworks**
   - Information Fidelity → error propagation in evaluation
   - PWC → trajectory position importance
   - AV effort metrics → agent downstream cost

### Practical Impact:

1. **Better evaluation protocols** for agent benchmarks
2. **Criticality-aware judges** for research community
3. **Diagnostic tools** for identifying trajectory weaknesses (not just outcomes)

---

## Risk Mitigation

### Risk 1: Judges actually ARE calibrated
**Mitigation:** Pilot study first (50 trajectories, compute JCC)
**Contingency:** Pivot to "when are judges calibrated?" (task/domain analysis)

### Risk 2: Criticality is too subjective
**Mitigation:** Use objective measures (token waste, cascade depth) first
**Contingency:** Focus on rank-order agreement rather than absolute scores

### Risk 3: Calibration doesn't generalize
**Mitigation:** Test on multiple benchmarks (SWE-Bench, AgentBench, GAIA)
**Contingency:** Analyze failure modes, propose domain-specific calibration

### Risk 4: Outcome metrics sufficient (reviewers argue)
**Mitigation:** Show high-outcome-overlap cases where criticality differs
**Contingency:** Emphasize diagnostic value for agent development (not just ranking)

---

## Timeline

| Week | Phase | Key Deliverables |
|------|-------|------------------|
| 1-2 | Foundation | Theoretical framework, dataset preparation, annotation protocol |
| 3-4 | Baseline Analysis | Exp 1 (JCC), Exp 2 (discrimination), judge bias decomposition |
| 5-7 | Proposed Methods | CWPR implementation, effort metrics, calibrated judge training |
| 8-10 | Evaluation | Benchmark comparison, ablations, human evaluation |
| 11-12 | Writing | Paper draft, artifact release, submission |

**Total:** 12 weeks to submission-ready paper

---

## Literature Integration Strategy

### Papers to Cite in Each Section:

**Introduction:**
- G-Eval (judge foundation)
- SWE-ABS (outcome metrics mislead)
- Information Fidelity (error propagation exists)

**Related Work - Judges:**
- G-Eval, Position Bias (Wang), Confidence Miscalibration (Ghosh)
- SLMEval, Autorubric

**Related Work - Agents:**
- AgentBench, SWE-Bench Randomness, AdaRubric
- AgentProcessBench, ToolLLM

**Related Work - Propagation:**
- Information Fidelity ⭐
- F-LLM ⭐
- Doctor-RAG, CoFiCot

**Related Work - Criticality:**
- PWC ⭐
- AV Effort Metrics ⭐
- Process Supervision (Royer)

**Methods:**
- PWC (position weighting inspiration)
- AV Effort Metrics (effort definition)
- Information Fidelity (propagation bounds)

**Discussion:**
- Survey papers (gap confirmation)
- Outcome vs. Process papers (Stop Rewarding Hallucinated)
- Structural error papers (APEX-EM, Environment Maps)

---

## Next Immediate Actions

1. **Create synthetic trajectory dataset** (Phase 1.2)
   - 50 trajectories with controlled error placement
   - Ground truth downstream cost annotations
   - Pilot JCC computation

2. **Implement baseline judges** (Phase 2.1 prep)
   - G-Eval API wrapper
   - GPT-4 generic judge prompt
   - AdaRubric replication (if code available) or approximation

3. **Define effort metrics precisely** (Phase 3.2)
   - Token waste computation algorithm
   - Cascade depth detection logic
   - Re-planning cost estimation

4. **Design annotation interface** (Phase 1.2)
   - Show trajectory step-by-step
   - Collect error type labels
   - Record expert criticality ratings

**Priority Order:** 3 → 1 → 2 → 4
(Define metrics first, then create data, then test judges, then scale annotation)

---

## Open Questions to Resolve

1. **Position weighting function:** Linear, exponential, or learned? Test all in ablations.

2. **Effort normalization:** How to compare TWR/MRD/CIS across different trajectory lengths? Need principled approach.

3. **Calibration data requirements:** How many annotated trajectories needed for fine-tuning? Test with 100/500/1000.

4. **Cross-benchmark generalization:** Can judge trained on SWE-Bench transfer to AgentBench? Critical for practical use.

5. **Error type taxonomy:** How granular? Planning, tool selection, parameter, formatting, logical - sufficient or need more?

---

## Expected Novel Findings

Based on literature analysis, we expect to find:

1. **Judges systematically underweight early planning errors** (based on F-LLM cascade finding + position bias literature)

2. **Judges overweight late formatting errors** (salient but low-impact, similar to verbosity bias)

3. **JCC varies by benchmark** (SWE-Bench > AgentBench > GAIA, due to trajectory complexity)

4. **Position weighting + effort weighting both necessary** (neither alone sufficient, ablations will show)

5. **Calibration generalizes within domains** (e.g., code tasks) but not across (code → web navigation)

These findings will structure our results section and discussion.
