# Architecture Change: Add experiment_id to Trajectories - 2026-04-02

## Summary

Changed trajectory storage from "pure cache" to "experiment-scoped" by adding `experiment_id` field. This enables:
1. **Per-experiment cleanup** - Can delete all data for one experiment
2. **Forced reload after bug fixes** - Each experiment loads fresh trajectories with current loader
3. **Better isolation** - Each experiment has its own copy of trajectories

## Changes Made

### 1. Experiment Runner ([src/experiment_runner.py](../src/experiment_runner.py))

**Trajectory storage (lines 255-305):**
- ✅ Added `experiment_id` to trajectory dict before saving
- ✅ Changed cache lookup to `get_trajectory_by_experiment()`
- ✅ Updated verification to filter by experiment_id

**Perturbation loading (lines 386-398):**
- ✅ Load trajectories by experiment_id instead of benchmark
- ✅ Simpler query: Just filter by `experiment_id` and `is_perturbed=False`

**Perturbed trajectory storage (lines 485-495):**
- ✅ Added `experiment_id` to perturbed trajectories
- ✅ Updated cache lookup to use `get_trajectory_by_experiment()`

### 2. MongoDB Storage ([src/storage/mongodb.py](../src/storage/mongodb.py))

**New method (line ~223):**
```python
def get_trajectory_by_experiment(
    self,
    trajectory_id: str,
    experiment_id: str
) -> Optional[Dict[str, Any]]:
    """Get trajectory by ID for a specific experiment."""
    return self.trajectories.find_one({
        "trajectory_id": trajectory_id,
        "experiment_id": experiment_id
    })
```

**Index updates (lines 87-103):**
- ✅ Changed unique index from `trajectory_id` to `(trajectory_id, experiment_id)`
- ✅ Added `experiment_id` index
- ✅ Added compound index: `(experiment_id, is_perturbed)`

### 3. Clear Experiment Script ([ops/clear_experiment.py](../ops/clear_experiment.py))

**User already added trajectories to collections_to_clear:**
```python
collections_to_clear = {
    "experiments": {"experiment_id": experiment_id},
    "trajectories": {"experiment_id": experiment_id},  # ← Now included!
    "perturbations": {"experiment_id": experiment_id},
    "annotations": {"experiment_id": experiment_id},
    "judge_evaluations": {"experiment_id": experiment_id},
    "ccg_metrics": {"experiment_id": experiment_id},
}
```

## Migration Path

### Old trajectories (without experiment_id)

The 50 old trajectories in MongoDB **don't have** `experiment_id`, so:
- ✅ They won't be loaded by the new code
- ✅ They won't be deleted by `clear_experiment.py`
- ✅ They can be manually deleted or left as orphans

### Clean start

```bash
# 1. Clear old experiment data
python ops/clear_experiment.py exp_poc_toolbench_20260402

# 2. Re-run with fixed loader
python main.py --config poc_experiment_toolbench --runner load,perturb
```

This will:
1. Load 50 trajectories with the **fixed loader** (tool_input as dict, system_prompt, full content)
2. Tag them with `experiment_id`
3. Generate perturbations successfully (no more string errors!)

## Benefits

### ✅ Clean Experiment Lifecycle
```bash
# Run experiment
python main.py --config my_experiment --runner all

# Something went wrong? Clean it up completely
python ops/clear_experiment.py exp_my_experiment_20260402

# Re-run with fresh data
python main.py --config my_experiment --runner all
```

### ✅ Bug Fix Testing
When you fix a loader bug:
1. Old experiments keep old (broken) trajectories
2. New experiments get fresh (fixed) trajectories
3. Can compare side-by-side

### ✅ Experiment Isolation
Each experiment ID gets its own copy of trajectories, preventing:
- Accidental cache pollution
- Version conflicts
- Data corruption across experiments

## Trade-offs

**Before (pure cache):**
- ✅ Pro: Reuse trajectories across experiments
- ✅ Pro: Less storage space
- ❌ Con: Can't clean up per-experiment
- ❌ Con: Stuck with old broken data after bug fixes

**After (experiment-scoped):**
- ✅ Pro: Clean up entire experiment
- ✅ Pro: Force reload after fixes
- ✅ Pro: Better isolation
- ❌ Con: Duplicate trajectories if running same config twice

**Decision:** Experiment-scoped is better because:
1. Loader bugs are common during development
2. Clean experiments are critical for reproducibility
3. Storage cost is minimal (50 trajectories = ~500KB)

## Next Steps

1. ✅ Clear old experiment
2. ✅ Re-run load + perturb phases
3. ✅ Verify perturbations actually modify content
4. ✅ Proceed with annotation phase
