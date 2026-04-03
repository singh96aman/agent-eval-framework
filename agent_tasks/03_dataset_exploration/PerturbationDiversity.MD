# Perturbation Diversity Strategy

**Task:** 03_dataset_exploration  
**Date:** 2026-04-03  
**Status:** Draft for Review

---

## 1. Objective

Define a comprehensive perturbation taxonomy that:
1. Covers realistic agent failure modes
2. Enables testing judge sensitivity to different error types
3. Provides sufficient diversity for meaningful CCG analysis
4. Maintains experimental tractability

---

## 2. Current Perturbation Coverage (POC)

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

## 3. Perturbation Realism Constraint (CRITICAL)

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

## 7. Experimental Design: One Perturbation Per Trajectory

### 7.1 Design Decision

**Each trajectory receives exactly ONE perturbation.**

| Approach | Trajectories | Perturbations | Pros | Cons |
|----------|--------------|---------------|------|------|
| ~~All perturbations~~ | 600 | 5,400 | Controls for trajectory effects | Confounding |
| **One per trajectory** | 600 | 600 | Clean design, tractable | Need domain stratification |

### 7.2 Stratified Assignment

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

**Document Status:** Draft for Review  
**Next Action:** Request researcher review before implementation
