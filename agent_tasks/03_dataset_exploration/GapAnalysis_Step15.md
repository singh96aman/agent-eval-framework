# Gap Analysis: Steps 9-14 Validation Results

**Task:** 03_dataset_exploration  
**Step:** 15  
**Date:** 2026-04-03  
**Status:** Complete

---

## Executive Summary

This document analyzes the gaps between target and actual results from Steps 9-14. The validation reveals significant shortfalls in trajectory count, perturbation coverage, and metadata persistence that must be addressed before the study can proceed.

---

## 1. Quantitative Gap Summary

| Metric | Target | Actual | Gap | Severity |
|--------|--------|--------|-----|----------|
| Total Trajectories | 600 | 188 | -412 (69% short) | **CRITICAL** |
| Total Perturbations | 600 | 150 | -450 (75% short) | **CRITICAL** |
| Domains with counts | 8 categories | All "unknown" | Classification not persisted | **HIGH** |
| Complexity distribution | 20/50/30% (simple/medium/complex) | 64/36/0% (all simple/medium) | No complex trajectories | **HIGH** |
| Samples per condition | ~50 | 10-18 | Underpowered for statistical analysis | **HIGH** |

---

## 2. Root Cause Analysis

### 2.1 Trajectory Shortfall (412 missing)

**Root Causes:**
1. **ToolBench eval set smaller than expected** - Only 762 trajectories in eval set, down to 188 after quality filters
2. **GAIA deferred** - Requires API calls to generate trajectories (not just loading)
3. **SWE-bench deferred** - Dataset structure issues prevented integration
4. **Strict quality filters** - Removed trajectories with:
   - Fewer than 4 steps
   - Invalid tool calls
   - Missing ground truth

**Evidence:**
```
Trajectories loaded: 188
Step counts: min=4, max=6, avg=4.4
```

### 2.2 Domain Classification Not Persisted

**Root Cause:** The `classify_trajectory_domain()` function exists and works, but the sampling script does not save the domain field to the output JSON.

**Evidence:**
```python
# data/sampled/toolbench_400.json analysis:
Domains: unknown: 188  # All trajectories marked unknown
```

**Code gap in `scripts/sample_trajectories.py`:**
- Domain is computed for stratification but not included in the saved trajectory JSON
- Same issue with complexity and position metadata

### 2.3 Complexity Distribution Skew

**Root Cause:** ToolBench eval set contains primarily simple trajectories (4-6 steps).

**Evidence:**
```
Step counts: min=4, max=6, avg=4.4
# Target distribution: 20% simple (1-3), 50% medium (4-6), 30% complex (7+)
# Actual distribution: 0% simple, 100% medium, 0% complex
```

**Issue:** No trajectories with 7+ steps available for "complex" category.

### 2.4 Perturbation Position Not Tracked

**Root Cause:** Position metadata ("early", "middle", "late") is used during generation but not persisted to the output JSON.

**Evidence:**
```python
# data/perturbed/toolbench_perturbations.json analysis:
Positions: unknown: 150  # All perturbations missing position
```

### 2.5 Low Samples Per Condition

**Condition breakdown (target ~50 each):**

| Condition | Actual | Target | Gap |
|-----------|--------|--------|-----|
| planning | 52 | 50 | +2 (OK) |
| parameter | 40 | 50 | -10 |
| tool_selection | 37 | 50 | -13 |
| data_reference | 21 | 50 | -29 |

**Statistical power issue:** With only 10-18 samples per (type, position) cell, we cannot detect moderate effect sizes (d=0.5) with 80% power.

---

## 3. Impact Assessment

### 3.1 Scientific Validity Impact

| Issue | Impact on Study | Severity |
|-------|-----------------|----------|
| Low trajectory count | Cannot achieve statistical power for main claims | CRITICAL |
| Missing domain metadata | Cannot report domain breakdown in paper | HIGH |
| No complex trajectories | Cannot test hypothesis on complex reasoning | HIGH |
| Position not tracked | Cannot analyze position effects | HIGH |
| Low condition counts | Wide confidence intervals, potential type II errors | HIGH |

### 3.2 Approved Design vs. Current State

**From REQUIREMENTS.MD approved design:**
- ToolBench: 400 trajectories -> **Actual: 188**
- GAIA: 100 trajectories -> **Actual: 0 (deferred)**
- SWE-bench: 100 trajectories -> **Actual: 0 (deferred)**
- Control group: 100 trajectories -> **Actual: 0 (not separated)**
- Complexity: 20/50/30% -> **Actual: 0/100/0%**

---

## 4. Gaps to Close (Prioritized)

### Priority 1: Critical (Must fix)

1. **Persist metadata to JSON outputs**
   - Domain classification
   - Complexity classification
   - Perturbation position
   - Benchmark source

2. **Increase trajectory count**
   - Option A: Use ToolBench training set (larger)
   - Option B: Relax quality filters
   - Option C: Actually generate GAIA trajectories

3. **Ensure perturbation coverage**
   - Each (type, position) needs ~50 samples minimum
   - data_reference needs significant increase

### Priority 2: High (Should fix)

4. **Add complex trajectories**
   - Look for 7+ step trajectories in training set
   - Or generate longer GAIA trajectories

5. **Separate control group**
   - Reserve 100 trajectories for all-perturbation analysis

### Priority 3: Medium (Nice to have)

6. **Add SWE-bench integration**
   - Different error modalities
   - Code domain coverage

---

## 5. Data Quality Assessment

### 5.1 What's Working

- Perturbation generation success rate: 79.8%
- Planning perturbations: 100% success
- Position assignment logic: Correct (just not persisted)
- Domain classification logic: Correct (just not persisted)

### 5.2 What's Not Working

- Metadata persistence: All key fields missing from JSON
- Trajectory quantity: 69% below target
- Complexity diversity: No complex trajectories
- Benchmark diversity: Only ToolBench (no GAIA/SWE-bench)

---

## 6. Recommended Fix Strategy

### Phase 1: Fix Metadata Persistence (Quick Win)

1. Update `scripts/sample_trajectories.py` to include domain/complexity in output
2. Update `scripts/generate_perturbations.py` to include position in output
3. Re-run on existing data to verify fixes

### Phase 2: Increase Trajectory Count

Option A (Recommended): Use ToolBench training set
- Training set has ~188k trajectories
- Filter for quality, sample 400
- More likely to find complex trajectories

Option B: Generate GAIA trajectories
- Run Claude/GPT-4o on GAIA tasks
- Requires API calls and budget
- Provides multi-step reasoning diversity

### Phase 3: Re-run Full Pipeline

1. Sample 400 ToolBench trajectories (with metadata)
2. Generate 150 perturbations (one per non-control trajectory)
3. Validate all metadata persists correctly
4. Create 100-trajectory control group

---

## 7. Success Criteria for Gap Closure

| Metric | Current | Target | Acceptance |
|--------|---------|--------|------------|
| Trajectories | 188 | 400+ | >= 350 |
| Perturbations | 150 | 350+ | >= 300 |
| Domain metadata | 0% | 100% | 100% |
| Position metadata | 0% | 100% | 100% |
| Complex trajectories | 0 | 90+ | >= 60 |
| Samples per condition | 10-18 | ~35 | >= 25 |

---

## 8. Next Steps

Proceed to Step 16: Use this Gap document to define concrete code changes needed.

---

**Document Status:** Complete  
**Next Action:** Step 16 - Define concrete implementation steps
