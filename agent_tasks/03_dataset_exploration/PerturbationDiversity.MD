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

## 3. Expanded Perturbation Taxonomy

### 3.1 Category Overview

| Category | Current | Proposed | Priority |
|----------|---------|----------|----------|
| Planning Errors | 1 type | 3 types | HIGH |
| Tool Selection Errors | 1 type | 2 types | MEDIUM |
| Parameter Errors | 4 subtypes | 6 subtypes | HIGH |
| **NEW: Sequencing Errors** | 0 | 3 types | **HIGH** |
| **NEW: Data Errors** | 0 | 2 types | **HIGH** |
| **NEW: Output Errors** | 0 | 2 types | LOW |

---

## 4. Detailed Perturbation Specifications

### 4.1 Planning Errors (Type A)

#### A1: Goal Substitution (Existing)
- **Description:** Replace goal with related but incorrect goal
- **Example:** "Get latest news" → "Get historical articles"
- **Implementation:** Regex pattern matching on Thought field

#### A2: Scope Reduction (Existing)
- **Description:** Address only part of multi-part request
- **Example:** "Compare A and B" → "Analyze A only"
- **Implementation:** Remove second clause from planning

#### A3: Constraint Violation (NEW)
- **Description:** Ignore explicit constraints in task
- **Example:** "Find hotels under $100" → Ignores price constraint
- **Implementation:** Remove constraint mentions from Thought
- **Priority:** MEDIUM

### 4.2 Tool Selection Errors (Type B)

#### B1: Similar Tool Swap (Existing)
- **Description:** Replace tool with semantically similar wrong tool
- **Example:** `get_weather` → `get_forecast`
- **Implementation:** ToolSimilarityMatcher

#### B2: Category Confusion (NEW)
- **Description:** Use tool from wrong category but similar name
- **Example:** `search_products` → `search_articles`
- **Implementation:** Match by action verb, ignore domain
- **Priority:** MEDIUM

### 4.3 Parameter Errors (Type C)

#### C1: Missing Required Parameter (Existing)
- **Description:** Remove a required parameter
- **Example:** `{"user_id": "123", "date": "2024-01-01"}` → `{"date": "2024-01-01"}`

#### C2: Wrong Parameter Value (Existing)
- **Description:** Replace value with incorrect but plausible value
- **Example:** `{"id": "123"}` → `{"id": "999"}`

#### C3: Wrong Parameter Name (Existing)
- **Description:** Rename parameter key
- **Example:** `{"user_id": "123"}` → `{"userid": "123"}`

#### C4: Wrong Parameter Type (Existing)
- **Description:** Change value type
- **Example:** `{"count": 5}` → `{"count": "5"}`

#### C5: Off-by-One Error (NEW)
- **Description:** Numeric values slightly wrong
- **Example:** `{"page": 1}` → `{"page": 0}` or `{"limit": 10}` → `{"limit": 11}`
- **Priority:** HIGH (realistic, subtle)

#### C6: Format Error (NEW)
- **Description:** Correct value, wrong format
- **Example:** `{"date": "2024-01-15"}` → `{"date": "01/15/2024"}`
- **Priority:** HIGH (common real-world error)

### 4.4 Sequencing Errors (Type D) - NEW

#### D1: Step Swap
- **Description:** Swap order of two consecutive steps
- **Example:** Steps 3-4 become Steps 4-3
- **Implementation:** Swap step objects, renumber
- **Priority:** **HIGH** (tests causal understanding)

#### D2: Premature Action
- **Description:** Execute action before prerequisite
- **Example:** Use result before fetching it
- **Implementation:** Move tool call before its dependency
- **Priority:** **HIGH**

#### D3: Missing Intermediate Step
- **Description:** Remove a required intermediate step
- **Example:** Delete step 3 in a 5-step sequence
- **Implementation:** Remove step, don't renumber (gap visible)
- **Priority:** **HIGH** (tests completeness checking)

### 4.5 Data Errors (Type E) - NEW

#### E1: Hallucinated Reference
- **Description:** Reference non-existent data from prior step
- **Example:** Use `article_id: 999` when prior step returned `article_id: 123`
- **Implementation:** Replace referenced value with random plausible value
- **Priority:** **HIGH** (tests context tracking)

#### E2: Stale Data Usage
- **Description:** Use outdated data from earlier step
- **Example:** Use step 2's result when step 4's result should be used
- **Implementation:** Reference wrong step's output
- **Priority:** MEDIUM

### 4.6 Output Errors (Type F) - NEW

#### F1: Incomplete Answer
- **Description:** Final answer missing required information
- **Example:** Asked for "name and price" → only returns name
- **Implementation:** Truncate final answer
- **Priority:** LOW (easy to detect)

#### F2: Wrong Format Answer
- **Description:** Answer correct but wrong format
- **Example:** Asked for JSON → returns plain text
- **Implementation:** Convert answer format
- **Priority:** LOW

---

## 5. Perturbation Matrix

### 5.1 Full Matrix (Types × Positions)

| Type/Subtype | Early | Middle | Late | Priority |
|--------------|-------|--------|------|----------|
| A1: Goal Substitution | ✅ | ✅ | ✅ | HIGH |
| A2: Scope Reduction | ✅ | ✅ | ✅ | HIGH |
| A3: Constraint Violation | ✅ | ✅ | ❌ | MEDIUM |
| B1: Similar Tool Swap | ✅ | ✅ | ✅ | HIGH |
| B2: Category Confusion | ✅ | ✅ | ✅ | MEDIUM |
| C1: Missing Parameter | ✅ | ✅ | ✅ | HIGH |
| C2: Wrong Value | ✅ | ✅ | ✅ | HIGH |
| C3: Wrong Name | ✅ | ✅ | ✅ | HIGH |
| C4: Wrong Type | ✅ | ✅ | ✅ | HIGH |
| C5: Off-by-One | ✅ | ✅ | ✅ | HIGH |
| C6: Format Error | ✅ | ✅ | ✅ | HIGH |
| D1: Step Swap | ❌ | ✅ | ✅ | **HIGH** |
| D2: Premature Action | ✅ | ✅ | ❌ | **HIGH** |
| D3: Missing Step | ✅ | ✅ | ✅ | **HIGH** |
| E1: Hallucinated Ref | ❌ | ✅ | ✅ | **HIGH** |
| E2: Stale Data | ❌ | ✅ | ✅ | MEDIUM |
| F1: Incomplete Answer | ❌ | ❌ | ✅ | LOW |
| F2: Wrong Format | ❌ | ❌ | ✅ | LOW |

### 5.2 Recommended Subset for Full Study

For tractability, focus on **12 perturbation types** (down from 18):

| Type | Subtype | Rationale |
|------|---------|-----------|
| Planning | Goal Substitution | POC baseline |
| Planning | Scope Reduction | POC baseline |
| Tool Selection | Similar Tool Swap | POC baseline |
| Parameter | Missing Required | POC baseline |
| Parameter | Wrong Value | POC baseline |
| Parameter | Off-by-One | NEW: subtle, realistic |
| Parameter | Format Error | NEW: common real error |
| Sequencing | Step Swap | NEW: tests causality |
| Sequencing | Missing Step | NEW: tests completeness |
| Sequencing | Premature Action | NEW: tests dependency |
| Data | Hallucinated Reference | NEW: tests context |
| Data | Stale Data | NEW: tests memory |

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

## 7. Expected Perturbation Counts

### 7.1 Per Trajectory

| Scenario | Perturbations/Trajectory |
|----------|-------------------------|
| Minimum | 3 (one per position) |
| Target | 5-6 (sampling across types) |
| Maximum | 12 (one per selected type) |

### 7.2 Total Dataset

| Dataset | Trajectories | Perturbations (3-6 each) |
|---------|--------------|--------------------------|
| ToolBench | 400 | 1,600-2,400 |
| GAIA | 100 | 400-600 |
| **Total** | **500** | **2,000-3,000** |

### 7.3 Sampling Strategy

Not every trajectory gets every perturbation type. Sample to ensure:
- Each type × position has ≥50 samples
- Each intensity level has ≥100 samples
- Total tractable for judge evaluation (~2,500 max)

---

## 8. Implementation Requirements

### 8.1 New Strategy Classes

```python
# New strategies to implement
class SequencingErrorStrategy(BasePerturbationStrategy):
    """Type D: Sequencing errors."""
    
    def swap_steps(self, trajectory, step_a, step_b): ...
    def remove_step(self, trajectory, step_idx): ...
    def move_step_early(self, trajectory, step_idx): ...

class DataErrorStrategy(BasePerturbationStrategy):
    """Type E: Data reference errors."""
    
    def hallucinate_reference(self, step, trajectory): ...
    def use_stale_data(self, step, trajectory, stale_step): ...
```

### 8.2 Generator Updates

```python
# Update PerturbationGenerator to support new types
PERTURBATION_TYPES = [
    "planning",        # A: Planning errors
    "tool_selection",  # B: Tool selection errors
    "parameter",       # C: Parameter errors
    "sequencing",      # D: Sequencing errors (NEW)
    "data",            # E: Data errors (NEW)
]
```

### 8.3 Config Schema Updates

```json
{
  "perturbations": {
    "types": ["planning", "tool_selection", "parameter", "sequencing", "data"],
    "positions": ["early", "middle", "late"],
    "subtypes": {
      "parameter": ["missing", "wrong_value", "off_by_one", "format_error"],
      "sequencing": ["step_swap", "missing_step", "premature_action"],
      "data": ["hallucinated_ref", "stale_data"]
    },
    "intensity_levels": ["subtle", "moderate", "severe"],
    "intensity_distribution": {"subtle": 0.4, "moderate": 0.4, "severe": 0.2}
  }
}
```

---

## 9. Validation Criteria

### 9.1 Perturbation Quality Checks

| Check | Criterion |
|-------|-----------|
| Syntactic validity | Perturbed trajectory is well-formed JSON |
| Semantic coherence | Error is localized, rest of trajectory unchanged |
| Realism | Error could plausibly occur in real agent |
| Detectability | Human annotator can identify the error |

### 9.2 Coverage Validation

| Metric | Target |
|--------|--------|
| Types covered | 12/12 |
| Positions covered | 3/3 per type |
| Intensity levels | 3/3 per type |
| Min samples per cell | 50 |

---

## 10. Risk Assessment

### 10.1 Implementation Risks

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Sequencing perturbations break trajectory | Medium | Validate output coherence |
| Hallucinated references not detectable | Low | Use obviously wrong values |
| Too many perturbation types | Medium | Subset to 12 for main study |

### 10.2 Annotation Risks

| Risk | Probability | Mitigation |
|------|-------------|------------|
| New types confuse annotators | Medium | Clear annotation guide |
| Intensity levels subjective | Medium | Provide calibration examples |
| Fatigue from volume | Medium | Batch annotations, breaks |

---

## 11. Questions for Researcher Review

1. **Sequencing errors:** Should step swaps preserve output references, or let them break naturally?

2. **Intensity levels:** Is subtle/moderate/severe the right granularity, or should we use a 1-5 scale?

3. **Perturbation subset:** Are the 12 selected types the right ones? Should we include/exclude any?

4. **Data errors:** How should we validate that hallucinated references are "obviously wrong" but not "absurdly wrong"?

5. **Position applicability:** Some perturbations don't make sense in all positions (e.g., step swap in early). Should we enforce this or let it be flexible?

6. **Interaction effects:** Should we test combining multiple perturbations in one trajectory (e.g., planning error + parameter error)?

---

## 12. Summary

### 12.1 Key Changes from POC

| Aspect | POC | Proposed |
|--------|-----|----------|
| Types | 3 | 5 (add Sequencing, Data) |
| Subtypes | 7 | 12 |
| Intensity | 1 (moderate) | 3 (subtle/moderate/severe) |
| Total perturbations | 443 | 2,000-3,000 |

### 12.2 Expected Impact on Study

1. **Better causality testing:** Sequencing errors directly test if judges understand step dependencies
2. **Subtlety testing:** Intensity levels reveal threshold for judge detection
3. **Context testing:** Data errors reveal if judges track information flow
4. **Statistical power:** More samples enable robust subgroup analysis

---

**Document Status:** Draft for Review  
**Next Action:** Request researcher review before implementation
