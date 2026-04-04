# Implementation Plan: Closing Gaps (Step 16)

**Task:** 03_dataset_exploration  
**Step:** 16  
**Date:** 2026-04-03  
**Status:** Complete

---

## Overview

This document defines concrete code changes to close the gaps identified in Step 15.

---

## Code Changes Required

### Change 1: Add domain/complexity fields to Trajectory schema

**File:** `src/data/schema.py`

**Change:** Add `domain` and `complexity` fields to `Trajectory` dataclass

```python
@dataclass
class Trajectory:
    trajectory_id: str
    benchmark: str
    steps: List[Step]
    ground_truth: GroundTruth
    metadata: Dict[str, Any] = field(default_factory=dict)
    domain: Optional[str] = None  # NEW
    complexity: Optional[str] = None  # NEW
```

**Update `to_dict()` and `from_dict()` to include these fields.**

---

### Change 2: Update sampling script to populate metadata

**File:** `scripts/sample_trajectories.py`

**Change:** After sampling, set domain/complexity on each trajectory before saving.

```python
# After sampling, populate metadata
for traj in sampled:
    traj.domain = classify_trajectory_domain(traj)
    traj.complexity = classify_trajectory_complexity(traj)
```

---

### Change 3: Fix perturbation position field name in analysis

**File:** Analysis scripts

**Issue:** The JSON contains `perturbation_position` but analysis scripts look for `position`.

**Fix:** Update analysis to use correct field name or standardize on one name.

---

### Change 4: Use ToolBench training set for more trajectories

**File:** `scripts/sample_trajectories.py`

**Change:** Add option to sample from training set (larger) instead of just eval set.

```python
trajectories = load_toolbench_trajectories(
    split="train",  # Changed from "eval"
    max_trajectories=2000,  # Sample from larger pool
    ...
)
```

---

### Change 5: Add experiment_id tracking

**File:** Create `src/experiment/config.py`

**Purpose:** Generate unique experiment IDs and track configuration for reproducibility.

```python
def generate_experiment_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"exp_{timestamp}"
```

---

## Implementation Order

1. **Schema changes** (Change 1) - Foundation for other changes
2. **Sampling script** (Changes 2, 4) - Populate fields, use larger pool
3. **Experiment tracking** (Change 5) - Add experiment_id
4. **Run pipeline** - Generate new dataset
5. **Validate** - Verify metadata persists

---

## Validation Criteria

After implementation, verify:

| Check | Expected |
|-------|----------|
| Trajectories saved with domain field | 100% have non-null domain |
| Trajectories saved with complexity field | 100% have non-null complexity |
| Trajectory count | >= 350 |
| Perturbations have position in JSON | "perturbation_position" present |
| Complex trajectories (7+ steps) | > 0 (ideally 60+) |

---

## Files to Modify

1. `src/data/schema.py` - Add domain/complexity fields
2. `scripts/sample_trajectories.py` - Populate fields, use training set
3. `scripts/run_perturbations.py` - No changes needed (position already saved correctly)

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Training set has different quality | Apply same quality filters |
| Schema change breaks existing code | Fields are optional with default None |
| Complex trajectories still rare | Check distribution before full run |

---

**Document Status:** Complete  
**Next Action:** Step 17 - Implement code changes
