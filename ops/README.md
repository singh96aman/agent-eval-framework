# Operations Scripts

This folder contains utility scripts for one-time operations, ad-hoc analysis, and database management.

**Note:** The primary interface is `python main.py --config <config> --runner <phases>`. These scripts are for bootstrap/debugging only.

## Scripts

### One-time Bootstrap

#### `sample_trajectories.py`
Initial sampling of trajectories from raw datasets to create `data/sampled/*.json` files.

```bash
python ops/sample_trajectories.py  # Creates toolbench_400.json, gaia_100.json, etc.
```

**Note:** After JSON files exist, use `python main.py --config dataset_full_study --runner load` instead.

### Development Utilities

#### `smoke_test_loaders.py`
Quick smoke test for dataset loaders (loads small samples).

```bash
python ops/smoke_test_loaders.py
```

#### `validate_trajectory_coverage.py`
Ad-hoc analysis of trajectory coverage across domains/complexity.

```bash
python ops/validate_trajectory_coverage.py
```

### Database Management

### `clear_experiment.py`

Clear all MongoDB collections for a specific experiment.

**What it deletes:**
- ✓ Experiment metadata
- ✓ Perturbations  
- ✓ Annotations
- ✓ Judge evaluations
- ✓ CCG metrics

**What it preserves:**
- ✓ Trajectories (they are pure cache, no experiment_id)

#### Usage

```bash
# Dry run to preview what would be deleted
python ops/clear_experiment.py exp_poc_toolbench_20260402 --dry-run

# Clear specific experiment (requires confirmation)
python ops/clear_experiment.py exp_poc_toolbench_20260402

# Clear ALL experiments (DANGEROUS - requires typing 'DELETE ALL')
python ops/clear_experiment.py --all

# Dry run for all experiments
python ops/clear_experiment.py --all --dry-run
```

#### Environment Configuration

The script loads MongoDB configuration from `.env` file:

```bash
# MongoDB connection (choose one)
MONGODB_URI=mongodb://localhost:27017      # Local MongoDB
# MONGODB_URI=mongodb://user:pass@host:port  # Remote MongoDB

# Database name
MONGODB_DATABASE=agent_judge_experiment
```

#### Safety Features

1. **Dry run mode**: Use `--dry-run` to preview deletions without making changes
2. **Confirmation prompts**: Requires explicit confirmation before deleting
3. **Clear summary**: Shows exactly what will be deleted before proceeding
4. **Trajectory preservation**: Never deletes cached trajectories

#### Examples

**Clear experiment after fixing bugs:**
```bash
# 1. Preview what will be deleted
python ops/clear_experiment.py exp_poc_toolbench_20260402 --dry-run

# 2. Clear old perturbations
python ops/clear_experiment.py exp_poc_toolbench_20260402

# 3. Re-run perturbation generation
python main.py --config poc_experiment_toolbench --runner perturb
```

**Start fresh with clean database:**
```bash
# Clear everything
python ops/clear_experiment.py --all

# Re-run full experiment
python main.py --config poc_experiment_toolbench --runner all
```

#### Output Example

```
======================================================================
CLEARING EXPERIMENT: exp_poc_toolbench_20260402
======================================================================
Name: POC: ToolBench Judge Calibration
Description: POC experiment with ToolBench only
Created: 2026-04-02T10:30:00.000Z
======================================================================

📊 Documents to delete:
----------------------------------------------------------------------
  experiments         :     1 documents
  perturbations       :   208 documents
  annotations         :     0 documents
  judge_evaluations   :     0 documents
  ccg_metrics         :     0 documents
----------------------------------------------------------------------
  TOTAL               :   209 documents

ℹ️  Note: Trajectories are NOT deleted (they are pure cache)

⚠️  WARNING: This action cannot be undone!
Delete 209 documents? [y/N]: 
```
