# Perturbation Diversity Strategy

**Task:** 03_dataset_exploration  
**Date:** 2026-04-03  
**Status:** Revision 1 (Addressing Reviewer Feedback)

---

## 1. Objective

Define a comprehensive perturbation taxonomy that:
1. Covers realistic agent failure modes
2. Enables testing judge sensitivity to different error types
3. Provides sufficient diversity for meaningful CCG analysis
4. Maintains experimental tractability

---

## 2. CRITICAL: Revised Hypothesis (Addressing POC Findings)

### 2.1 Original Hypothesis (Partially Contradicted)

> LLM judges under-penalize early structural failures (planning, tool selection) while over-penalizing late local mistakes (parameter errors).

### 2.2 POC Results

| Type | Mean CCG | Finding |
|------|----------|---------|
| Parameter | +1.13 | Under-penalized (judges miss these) |
| Planning | -0.13 | Well-calibrated |
| Tool Selection | -0.26 | Over-penalized |

**The POC contradicts parts of the original hypothesis:**
- Parameter errors (local mistakes) are **under**-penalized, not over-penalized
- Planning errors are well-calibrated, not under-penalized
- Tool selection is over-penalized, as predicted

### 2.3 Revised Hypothesis

Based on POC evidence, we revise to:

> **Revised Hypothesis:** LLM judges systematically miscalibrate based on error **salience** rather than **criticality**. 
> - **High-salience errors** (wrong tool, flawed plan) are accurately or over-penalized because they are obvious
> - **Low-salience errors** (parameter values, data references) are under-penalized because they require tracking information flow across steps
> - **Position effects** exist: middle-position errors (steps 3-5) show worst calibration because context obscures the error's impact

### 2.4 Theoretical Grounding

This salience-based hypothesis aligns with:
1. **Attention limitations** in long-context evaluation
2. **Surface vs. deep reasoning** in LLM judgments
3. **Information tracking failures** across trajectory steps

The full study tests whether this revised hypothesis holds at scale across multiple benchmarks and judge models.

### 2.5 CRITICAL: Falsification Criteria

**What would count as evidence against the salience hypothesis?**

The hypothesis predicts:
- **H1**: Low-salience errors (parameters, data references) will have CCG > 0 (under-penalized)
- **H2**: High-salience errors (planning, tool selection) will have CCG ≤ 0 (well-calibrated or over-penalized)
- **H3**: Position effects exist, with middle positions showing worst calibration for low-salience errors

**Falsification thresholds:**

| Outcome | Interpretation |
|---------|----------------|
| All CCG ≈ 0 (within 0.3) across all types | Judges are well-calibrated; hypothesis FALSE |
| Parameter CCG < 0 AND Planning CCG > 0 | Pattern reversed; original hypothesis was closer; salience hypothesis FALSE |
| No position effects (early/middle/late CCG equivalent) | Position component FALSE; salience component may still hold |
| Cross-judge disagreement (Claude vs GPT show opposite patterns) | No systematic LLM bias; findings are model-specific artifacts |

**Pre-specified decision rule:** If >2 falsification outcomes occur, we report null/negative results rather than post-hoc reframing.

---

## 3. Current Perturbation Coverage (POC)

### 2.1 Implemented Types

| Type | Strategy | Example |
|------|----------|---------|
| **Planning (Type A)** | Goal substitution, scope reduction | "Compare X and Y" → only analyze X |
| **Tool Selection (Type B)** | Similar tool swap | `racecards` → `results` |
| **Parameter (Type C)** | Value corruption, missing params | `{"id": "123"}` → `{"id": "99999"}` |

### 2.2 Position Coverage

| Position | Step Range | Description |
|----------|------------|-------------|
| Early | Steps 1-2 | Initial planning/setup errors |
| Middle | Steps 3-5 | Mid-execution errors |
| Late | Steps 6+ | Near-completion errors |

### 2.3 POC Results Summary

| Type | CCG | Judge Behavior |
|------|-----|----------------|
| Parameter | +1.13 | **Under-penalized** |
| Planning | -0.13 | Well-calibrated |
| Tool Selection | -0.26 | Over-penalized |

**Key insight:** Judges are most miscalibrated for parameter errors, especially in middle positions (CCG = 2.788).

---

## 4. CRITICAL: Ground Truth Criticality Operationalization

### 4.1 The Core Question

**Reviewer concern:** "How do you measure task success degradation on a static perturbed trajectory that was never executed?"

This is the central methodological challenge. We cannot re-execute perturbed trajectories because:
1. APIs may have changed or require authentication
2. Perturbations may cascade unpredictably in live execution
3. Cost/time prohibitive at scale

### 4.2 Our Approach: Counterfactual Human Judgment

**Ground truth criticality is operationalized as:**

> Human expert assessment of "how much would this error degrade task success, assuming the trajectory continued as written?"

This is a **counterfactual judgment**, not a measurement of actual execution. Annotators are asked:

1. **Identify the error** in the perturbed trajectory
2. **Assess downstream impact** by reasoning about what would happen
3. **Rate criticality** on a 0-100 scale where:
   - 0 = No impact on task success
   - 50 = Partial degradation (task partially succeeds)
   - 100 = Complete task failure

### 4.3 Formal Operationalization

**True Criticality Score (TCS)** is defined as:

```
TCS = 100 - E[TaskSuccess | perturbation]

where TaskSuccess ∈ [0, 100] is judged by:
- Whether the final answer would be correct
- Whether intermediate results are corrupted
- Whether the error propagates to downstream steps
```

### 4.4 Addressing the "Never Executed" Problem

We acknowledge this limitation and mitigate via:

| Approach | How It Helps |
|----------|--------------|
| **Conservative annotation** | Annotators told to assume "reasonable continuation" - the agent doesn't magically recover |
| **Cascade heuristics** | Annotators given explicit rules for reasoning about error propagation |
| **Execution proxies** | For some perturbations (parameter errors), we can actually call the API and verify the wrong data is returned |
| **Multiple annotators** | Inter-annotator agreement reveals where judgment is reliable vs. uncertain |
| **Execution-based validation subset** | See 4.4.1 below |

### 4.4.1 CRITICAL: Execution-Based Validation Subset

**Reviewer concern:** Counterfactual human judgment is the weakest methodological link.

**Our mitigation:** For a subset of 40 perturbations, we will **actually re-execute** the perturbed trajectory and measure task success:

| Subset | Size | Approach |
|--------|------|----------|
| Parameter errors (ToolBench) | 20 | Call actual APIs, compare returned data |
| Tool selection errors (ToolBench) | 10 | Execute both tools, compare outputs |
| Planning errors (GAIA) | 10 | Re-run agent with perturbed plan, measure answer correctness |

**What this provides:**
1. **Ground truth calibration**: Compare human counterfactual judgments against actual execution outcomes
2. **Adjustment factors**: If humans systematically over/under-estimate, we can adjust CCG
3. **Methodological defense**: "40 trajectories were execution-validated; human judgments correlated r=X with actual outcomes"

**Limitations:**
- Not all perturbations can be re-executed (APIs may require auth, may have changed)
- Execution validation focuses on ToolBench parameter errors (easiest to validate)
- 40 is a small subset; execution-based CCG will be exploratory, not primary

**Decision rule:** If human-execution correlation < 0.5, we flag this as a major limitation and report execution-based CCG as a sensitivity analysis.

### 4.5 Inter-Annotator Agreement (Target)

| Phase | Annotations | Agreement Target |
|-------|-------------|------------------|
| POC | 26 | Exploratory (no target) |
| Pilot | 50 | Krippendorff's alpha >= 0.6 |
| Full study | 275 | Krippendorff's alpha >= 0.7 |

**If agreement is low:** We report it as a limitation and stratify results by "high-agreement" vs. "low-agreement" cases.

### 4.6 Why This Is Still Valid

The CCG metric measures **relative miscalibration** between judges and humans. Even if humans' criticality judgments are imperfect:

1. **Humans are the current gold standard** for agent evaluation
2. **CCG measures judge-human disagreement**, which is valuable regardless of whether humans are "correct"
3. **Systematic patterns** (e.g., judges always under-penalize parameters) reveal judge biases even if absolute scores are noisy

**Key reframe:** The contribution is not "judges are wrong," but "judges disagree with human experts in systematic ways that have implications for agent evaluation and RLHF."

### 4.7 Defending the Human = Ground Truth Assumption

**Reviewer concern:** "Maybe some errors should be penalized less if they don't cause task failure. Humans may over-weight early errors due to hindsight bias."

**Response:**

1. **We measure correlation, not absolute accuracy.** If humans have hindsight bias, and judges have the *same* bias, CCG would be zero. CCG detects *differential* biases.

2. **Hindsight bias affects both directions.** Humans might over-penalize early errors (hindsight) OR under-penalize them (outcome bias - "it worked out"). The bias direction is itself an empirical question.

3. **Alternative: Behavioral ground truth.** In future work, we could:
   - Re-execute a subset of perturbations and measure actual task success
   - Use this to calibrate human judgments
   - Report both "human-based CCG" and "execution-based CCG"

**For this paper:** We clearly state that human judgment is our ground truth, acknowledge the assumption, and discuss its limitations.

---

## 5. CRITICAL: Selection Bias Acknowledgment

### 5.1 The Bias

**Reviewer concern:** "By excluding perturbations that break execution (sequencing errors, missing required params), you're selecting for errors that allow continuation - errors that are harder to notice. This biases toward finding judges miss errors."

**This is a valid concern.** We acknowledge it explicitly.

### 5.2 Why We Accept This Bias

1. **Crash errors are trivially detected.** If an API call fails, the trajectory ends. Judges (and humans) would immediately see this. There's nothing interesting to measure about calibration for obvious failures.

2. **The interesting question is silent errors.** Real-world agent failures often involve errors that *don't* crash but still corrupt results. These are the errors that matter for:
   - RLHF reward modeling (silent errors get positive reward)
   - Production monitoring (failures go undetected)
   - Safety oversight (subtle misalignment is harder to catch)

3. **We reframe the contribution:** Our study measures "calibration for silent errors" not "calibration for all errors." This is a narrower but more interesting claim.

### 5.3 Explicit Scope Statement (for paper)

> This study focuses on **non-crashing perturbations** - errors that allow agent execution to continue but degrade task success. We exclude crash-inducing errors (API failures, missing dependencies) because these are trivially detectable and not relevant to the calibration question. Our findings apply to silent errors that propagate through trajectories, which are the errors most likely to escape detection in production.

### 5.4 Future Work: Crash Error Study

A complementary study could examine judge behavior on crash errors:
- Do judges correctly identify the failure point?
- Do they over-penalize crash errors (catastrophizing)?
- Do they attribute blame correctly?

This is out of scope for the current study but noted as future work.

---

## 6. Perturbation Realism Constraint (CRITICAL)

### 3.1 The Problem with Naive Perturbations

**Question:** When we perturb step 4, what happens to steps 5, 6, 7?

In reality, if step 4 fails or returns wrong data:
1. **Error → Recovery:** Agent detects error, re-plans, trajectory diverges
2. **Error → Propagation:** Agent doesn't detect, downstream steps use bad data
3. **Error → Failure:** Task fails at the error point

**Current approach problem:** We perturb step 4 but leave steps 5-7 unchanged. This creates an **incoherent trajectory** that would never happen in reality.

### 3.2 Realism Requirement

**For our study to be grounded in real-world problems, perturbations must satisfy:**

> The perturbed trajectory must remain **plausibly executable** - i.e., the error should not immediately crash the execution, but rather allow the trajectory to continue with incorrect information.

### 3.3 Perturbation Validity Matrix

| Perturbation Type | Downstream Effect | Validity |
|-------------------|-------------------|----------|
| Parameter: wrong value | API returns data, but wrong data | ✅ **VALID** |
| Parameter: missing required | API fails → agent would retry | ⚠️ Only if API has defaults |
| Tool selection: similar tool | Different API returns valid data | ✅ **VALID** (if tool works) |
| Planning: wrong goal | Wrong reasoning, but execution proceeds | ✅ **VALID** |
| Sequencing: swap steps | Breaks dependency chain | ❌ Only if steps independent |
| Data: hallucinated reference | Uses wrong ID from prior step | ✅ **VALID** (realistic error) |

### 3.4 Implementation Approach

For each perturbation, we must verify:

1. **Execution continuity:** The perturbed step doesn't cause immediate crash
2. **Downstream coherence:** Subsequent steps can still execute (even if with wrong data)
3. **Detectability:** A human annotator can identify the error

**Perturbations that break execution are EXCLUDED** unless we also simulate the recovery/failure path.

---

## 4. Expanded Perturbation Taxonomy (Revised)

### 4.1 Category Overview (Realism-Constrained)

Based on the realism constraint, we focus on perturbations that **allow execution to continue**:

| Category | Subtypes | Realism Status | Priority |
|----------|----------|----------------|----------|
| Planning Errors | 2 types | ✅ All valid | HIGH |
| Tool Selection Errors | 1 type | ✅ Valid (if tool works) | HIGH |
| Parameter Errors | 3 subtypes | ✅ Valid (returns wrong data) | **HIGHEST** |
| ~~Sequencing Errors~~ | ~~3 types~~ | ❌ Breaks execution | EXCLUDED |
| Data Reference Errors | 1 type | ✅ Valid (hallucination) | MEDIUM |

**Total: 7 perturbation subtypes** (down from 18 proposed, up from 3 in POC)

---

## 5. Detailed Perturbation Specifications (Realism-Constrained)

### 5.1 Planning Errors (Type A) - ✅ VALID

Planning errors modify the agent's reasoning but don't break execution. The trajectory continues with wrong intent.

#### A1: Goal Substitution (Existing)
- **Description:** Replace goal with related but incorrect goal
- **Example:** "Get latest news" → "Get historical articles"
- **Implementation:** Regex pattern matching on Thought field
- **Realism:** Agent pursues wrong goal but execution proceeds

#### A2: Scope Reduction (Existing)
- **Description:** Address only part of multi-part request
- **Example:** "Compare A and B" → "Analyze A only"
- **Implementation:** Remove second clause from planning
- **Realism:** Agent completes partial task, common real failure

### 5.2 Tool Selection Errors (Type B) - ✅ VALID (with constraint)

Tool selection errors are valid **only if the wrong tool still returns valid output**.

#### B1: Similar Tool Swap (Existing)
- **Description:** Replace tool with semantically similar wrong tool
- **Example:** `get_weather` → `get_forecast`
- **Implementation:** ToolSimilarityMatcher
- **Realism constraint:** Only swap if both tools return similar data structure
- **Real-world analogy:** Developer uses `array.find()` instead of `array.filter()`

### 5.3 Parameter Errors (Type C) - ✅ VALID (highest priority)

Parameter errors are the most realistic because APIs typically return data (even if wrong) rather than crashing. This is why POC found these most under-penalized.

#### C1: Wrong Parameter Value (Existing) - **PRIMARY**
- **Description:** Replace value with incorrect but plausible value
- **Example:** `{"id": "123"}` → `{"id": "999"}`
- **Realism:** API returns data for wrong entity, execution continues

#### C2: Format Error (NEW) - **SECONDARY**
- **Description:** Correct value, wrong format
- **Example:** `{"date": "2024-01-15"}` → `{"date": "01/15/2024"}`
- **Realism:** Some APIs are format-tolerant, return unexpected results

#### C3: Off-by-One Error (NEW) - **TERTIARY**
- **Description:** Numeric values slightly wrong
- **Example:** `{"page": 1}` → `{"page": 0}`
- **Realism:** Common programming error, subtle impact

**EXCLUDED Parameter Subtypes:**
- ~~Missing required parameter~~ → Causes API failure, not realistic continuation
- ~~Wrong parameter name~~ → Causes API failure
- ~~Wrong parameter type~~ → Often causes API failure

### 5.4 Data Reference Errors (Type D) - ✅ VALID (new)

Data reference errors occur when an agent uses incorrect data from a prior step. This is highly realistic and tests judge's ability to track information flow.

#### D1: Hallucinated Reference (NEW)
- **Description:** Reference non-existent data from prior step
- **Example:** Prior step returned `article_id: 123`, but agent uses `article_id: 999`
- **Implementation:** Replace referenced value with plausible but wrong value
- **Realism:** Very common - agents lose track of IDs, names, values across steps

### 5.5 EXCLUDED Perturbation Types

The following were considered but excluded due to realism constraints:

| Type | Reason for Exclusion |
|------|---------------------|
| **Sequencing: Step Swap** | Breaks dependency chain, trajectory becomes non-executable |
| **Sequencing: Missing Step** | Subsequent steps reference missing output, breaks execution |
| **Sequencing: Premature Action** | Uses data that doesn't exist yet, breaks execution |
| **Parameter: Missing Required** | API typically fails with error, agent would retry |
| **Parameter: Wrong Name** | API typically fails, not realistic continuation |
| **Output: Incomplete Answer** | Too easy to detect, not interesting for calibration study |

**Note:** Sequencing errors could be included in future work if we also simulate the agent's recovery/failure response.

---

## 6. Final Perturbation Matrix (Realism-Constrained)

### 6.1 Approved Perturbation Types

| Type | Subtype | Early | Middle | Late | Realism |
|------|---------|-------|--------|------|---------|
| Planning | A1: Goal Substitution | ✅ | ✅ | ✅ | Agent pursues wrong goal |
| Planning | A2: Scope Reduction | ✅ | ✅ | ✅ | Agent does partial task |
| Tool Selection | B1: Similar Tool Swap | ✅ | ✅ | ✅ | Returns different data |
| Parameter | C1: Wrong Value | ✅ | ✅ | ✅ | Returns wrong entity |
| Parameter | C2: Format Error | ✅ | ✅ | ✅ | Unexpected API behavior |
| Parameter | C3: Off-by-One | ✅ | ✅ | ✅ | Subtle numeric error |
| Data Reference | D1: Hallucinated Ref | ❌ | ✅ | ✅ | Lost track of data |

**Total: 7 perturbation subtypes × 3 positions = 21 conditions**  
(But D1 not applicable to early position → 19 effective conditions)

### 6.2 Condition Distribution for 600 Perturbations

| Condition | Count | Notes |
|-----------|-------|-------|
| Planning × 3 positions | 2 × 3 × 33 = 198 | ~33 per subtype × position |
| Tool Selection × 3 positions | 1 × 3 × 33 = 99 | |
| Parameter × 3 positions | 3 × 3 × 33 = 297 | Highest priority (POC finding) |
| Data Reference × 2 positions | 1 × 2 × 33 = 66 | Not applicable to early |
| **Total** | ~**660** | Round to 600 via sampling |

---

## 6. Intensity Levels

### 6.1 Proposed Levels

| Level | Description | Example (Parameter Error) |
|-------|-------------|---------------------------|
| **Subtle** | Easy to miss, plausible | `page: 1` → `page: 0` |
| **Moderate** | Noticeable but not obvious | `user_id: "123"` → `user_id: "456"` |
| **Severe** | Obvious, catastrophic | `user_id: "123"` → `{}` (empty) |

### 6.2 Intensity Distribution

For each perturbation type, generate:
- 40% Subtle
- 40% Moderate
- 20% Severe

**Rationale:** Subtle errors test fine-grained judge calibration; severe errors provide sanity check.

---

## 8. Experimental Design: One Perturbation Per Trajectory

### 8.1 Design Decision

**Each trajectory receives exactly ONE perturbation.**

| Approach | Trajectories | Perturbations | Pros | Cons |
|----------|--------------|---------------|------|------|
| ~~All perturbations~~ | 600 | 5,400 | Controls for trajectory effects | Confounding between perturbation types |
| **One per trajectory** | 600 | 600 | Clean design, tractable | Trajectory confounded with perturbation |

### 8.2 CRITICAL: Addressing the Trajectory-Perturbation Confound

**Reviewer concern:** "Each trajectory has different difficulty/domain/quality - now confounded with perturbation type."

**This is a valid concern.** With one perturbation per trajectory, if Planning errors show low CCG, it might be because:
1. Judges are well-calibrated for planning errors (true effect), OR
2. Planning errors happened to be applied to "easier" trajectories (confound)

### 8.3 Mitigation Strategies

**Strategy 1: Stratified Random Assignment**
- Trajectories are first classified by difficulty (simple/medium/complex) and domain
- Perturbation types are randomly assigned *within* each stratum
- This ensures each perturbation type sees similar trajectory distributions

```
For each (benchmark, domain, complexity) stratum:
    trajectories = filter(all_trajectories, stratum)
    shuffle(trajectories)
    assign_round_robin(trajectories, perturbation_types)
```

**Strategy 2: Covariate Adjustment**
- Record trajectory features: step count, domain, task complexity
- Include these as covariates in regression analysis
- Report both raw CCG and covariate-adjusted CCG

**Strategy 3: Within-Trajectory Replication (Subset)**
- For **100 trajectories** (random sample), apply ALL perturbation types
- This creates a "within-trajectory" control group
- If results differ from main study, the confound is real

| Sample | Design | Purpose |
|--------|--------|---------|
| Main (500) | One perturbation per trajectory | Primary analysis |
| Control (100) | All perturbations per trajectory | Confound check |

**Why 100 (not 50)?** Per reviewer feedback, 50 is too small for the confound check to be statistically meaningful. With 100 trajectories × 7 perturbation types = 700 within-trajectory perturbations, we have adequate power to detect confound effects.

### 8.4 Analysis Plan

1. **Primary analysis:** Compare CCG across perturbation types (one-per-trajectory data)
2. **Confound check:** Compare CCG in control group (same trajectories, all perturbations)
3. **Sensitivity analysis:** Report covariate-adjusted CCG
4. **If confound detected:** Report within-trajectory results as primary

### 8.5 Stratified Assignment

To ensure domain-agnostic results, each perturbation condition is spread across benchmarks:

| Perturbation | ToolBench | GAIA | SWE-bench | Total |
|--------------|-----------|------|-----------|-------|
| Planning (A1+A2) × 3 pos | 120 | 30 | 30 | 180 |
| Tool Selection (B1) × 3 pos | 60 | 15 | 15 | 90 |
| Parameter (C1+C2+C3) × 3 pos | 180 | 45 | 45 | 270 |
| Data Reference (D1) × 2 pos | 40 | 10 | 10 | 60 |
| **Total** | **400** | **100** | **100** | **600** |

### 7.3 Domain-Agnostic Validation

After running judges, we can verify results are not domain-specific by:
1. Comparing CCG within each condition across benchmarks
2. Testing for interaction effects (perturbation type × benchmark)
3. Reporting confidence intervals that span all benchmarks

---

## 8. Implementation Requirements

### 8.1 New Strategy Class

```python
class DataReferenceErrorStrategy(BasePerturbationStrategy):
    """Type D: Data reference errors (hallucination)."""
    
    def hallucinate_reference(self, step, trajectory):
        """
        Replace a referenced value from prior step with wrong value.
        
        Example: If step 3 uses article_id=123 from step 2's output,
        change to article_id=999 (plausible but wrong).
        """
        ...
```

### 8.2 Updated Perturbation Types

```python
PERTURBATION_TYPES = [
    "planning",        # A: Goal substitution, scope reduction
    "tool_selection",  # B: Similar tool swap
    "parameter",       # C: Wrong value, format error, off-by-one
    "data_reference",  # D: Hallucinated reference (NEW)
]
# Note: Sequencing errors EXCLUDED (break execution)
```

### 8.3 Config Schema

```json
{
  "perturbations": {
    "types": ["planning", "tool_selection", "parameter", "data_reference"],
    "positions": ["early", "middle", "late"],
    "subtypes": {
      "planning": ["goal_substitution", "scope_reduction"],
      "tool_selection": ["similar_tool_swap"],
      "parameter": ["wrong_value", "format_error", "off_by_one"],
      "data_reference": ["hallucinated_ref"]
    },
    "one_per_trajectory": true,
    "stratify_by": ["benchmark", "domain"]
  }
}
```

---

## 9. Validation Criteria

### 9.1 Perturbation Quality Checks

| Check | Criterion |
|-------|-----------|
| **Execution continuity** | Perturbed step doesn't crash API |
| **Downstream coherence** | Subsequent steps can still execute |
| **Realism** | Error could occur in real agent |
| **Detectability** | Human annotator can identify error |

### 9.2 Coverage Validation

| Metric | Target |
|--------|--------|
| Perturbation types covered | 7/7 |
| Positions covered | 3/3 (2/2 for data_ref) |
| Benchmarks covered per condition | 3/3 |
| Min samples per condition | 30 |

---

## 10. Risk Assessment

### 10.1 Implementation Risks

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Data reference detection difficult | Medium | Validate with annotator pilot |
| SWE-bench tool structure different | Medium | Map tool types explicitly |
| Insufficient samples per condition | Low | Reserve pool of trajectories |

### 10.2 Analysis Risks

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Results ARE domain-specific | Medium | Report by benchmark, still valuable |
| Parameter errors dominate | Low | Equal allocation to planning/tool |

---

## 11. Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| Sequencing errors? | **EXCLUDED** - break execution, not realistic |
| All perturbations per trajectory? | **NO** - one per trajectory, stratified |
| Intensity levels? | **DEFERRED** - focus on one intensity first |
| Combine multiple perturbations? | **NO** - one per trajectory for clean design |

---

## 12. Summary

### 12.1 Key Changes from POC

| Aspect | POC | Revised Proposal |
|--------|-----|------------------|
| Trajectories | 50 | 600 (400 TB + 100 GAIA + 100 SWE) |
| Perturbation types | 3 | 4 (add Data Reference) |
| Subtypes | 7 | 7 (realism-constrained) |
| Perturbations/trajectory | ~9 | **1** (cleaner design) |
| Total perturbations | 443 | 600 |
| Sequencing errors | Proposed | **EXCLUDED** (break execution) |

### 12.2 Design Principles

1. **Realism first:** Only include perturbations where execution can continue
2. **One per trajectory:** Avoid confounding, cleaner experimental design
3. **Domain stratification:** Each condition spans all 3 benchmarks
4. **Parameter focus:** POC showed these are most under-penalized

### 12.3 Expected Impact

1. **Cleaner statistical analysis:** No confounding between perturbation types
2. **Domain-agnostic validation:** Can verify results generalize across benchmarks
3. **Context tracking test:** Data reference errors reveal if judges track information flow
4. **Tractable annotation:** 600 samples vs 5,400

---

## 13. Reviewer Feedback Addressed

This revision addresses the following concerns from the expert reviewer:

| # | Concern | Section | Resolution |
|---|---------|---------|------------|
| 1 | POC results contradict hypothesis | 2.1-2.4 | Revised to salience-based hypothesis |
| 2 | Ground truth criticality underspecified | 4.1-4.7 | Detailed operationalization, inter-annotator agreement targets |
| 3 | Realism constraint creates selection bias | 5.1-5.4 | Acknowledged, reframed as "silent errors" study |
| 4 | One-per-trajectory confounds variables | 8.2-8.4 | Stratified assignment + within-trajectory control group |
| 9 | Human criticality = ground truth | 4.6-4.7 | Defended with CCG measures relative bias, not absolute correctness |

**Concerns addressed in DatasetDiversity.MD:**
- #5 (Single judge model) → Multi-judge design
- #6 (GAIA generation bias) → Multi-generator design
- #7 (SWE-bench semantics) → Separate analysis, native perturbations
- #8 (Statistical power) → Hierarchical analysis, revised allocation

---

**Document Status:** Revision 1 (Addressing Reviewer Feedback)  
**Next Action:** Request researcher review of revisions
