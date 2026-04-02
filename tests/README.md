# Tests Directory

## Overview

This directory contains tests for the MongoDB schema implementation with foreign keys.

## Test Files

### Unit Tests

**`test_mongodb_schema.py`**
- Tests individual MongoDB operations
- Verifies foreign key relationships
- Tests pagination and counting
- Tests cache mechanisms
- Tests scalability (no 16MB limit)

### Integration Tests

**`test_integration_pipeline.py`**
- **End-to-end pipeline test** from config to CCG scores
- Uses stubbed data (no real API calls)
- Single test class with setup that runs full pipeline
- Tests verify schema format and foreign keys

## Test Data

**`data/test_config.json`**
- Test experiment configuration
- Defines 1 trajectory, 1 perturbation, 1 judge

**`data/sample_trajectory.json`**
- Hardcoded sample trajectory from ToolBench
- 4 steps: planning → tool execution → reasoning → final answer
- Task: "What is the population of Tokyo in 2023?"

**`data/stubbed_responses.json`**
- Stubbed perturbation (planning error)
- Stubbed annotation (TCS = 110.0)
- Stubbed judge evaluation (JPS = 25.0, under-penalized)
- Expected CCG score (-0.773, severe under-penalty)

## Running Tests

### Prerequisites

**Tests DO NOT require MongoDB!** All tests use JSON files or in-memory mocks.

- `test_integration_pipeline.py` - Uses JSON files in `tests/data/results/`
- `test_mongodb_schema.py` - Uses in-memory mock storage
- `test_storage.py` - Uses mocked MongoDB client

### Production vs Tests

**Production (experiments):**
- Uses MongoDB Atlas via `.env` configuration
- Real database with persistence
- Connection verified on startup

**Tests:**
- Use JSON files or in-memory mocks
- No MongoDB required
- Fast and reproducible

### Run Unit Tests

```bash
# All unit tests
pytest tests/test_mongodb_schema.py -v

# Specific test
pytest tests/test_mongodb_schema.py::test_foreign_key_relationships -v
```

### Run Integration Tests (No MongoDB Required!)

```bash
# Full integration test (runs pipeline in setup, saves to JSON)
pytest tests/test_integration_pipeline.py -v -s

# Specific test
pytest tests/test_integration_pipeline.py::TestIntegrationPipeline::test_experiment_schema_format -v

# Results saved to: tests/data/results/
```

### Run All Tests

```bash
pytest tests/ -v
```

## Integration Test Flow

The `IntegrationTestPipeline` class runs this flow:

```
1. Load config (test_config.json)
   ↓
2. Create experiment (no arrays, only progress counts!)
   ↓
3. Load trajectories (with experiment_id foreign key)
   ↓
4. Generate perturbations (planning error at step 1)
   ↓
5. Create annotations (stubbed: TCS = 110.0)
   ↓
6. Create judge evaluations (stubbed: JPS = 25.0)
   ↓
7. Compute CCG scores (CCG = -0.773, under-penalized)
```

### What Gets Tested

#### Schema Format Tests:
- ✅ `test_experiment_schema_format` - No arrays in experiments
- ✅ `test_trajectory_schema_format` - Foreign keys present
- ✅ `test_annotation_schema_format` - Foreign keys + TCS formula
- ✅ `test_judge_evaluation_schema_format` - Foreign keys + JPS formula
- ✅ `test_ccg_score_schema_format` - All foreign keys + CCG formula

#### Functionality Tests:
- ✅ `test_cache_check_works` - Judge evaluation caching
- ✅ `test_pagination_queries` - Skip/limit with foreign keys
- ✅ `test_cross_collection_queries` - Queries across collections

## Test Output

When running integration tests, you'll see:

```
🚀 Starting integration test pipeline...
1️⃣  Loading config...
2️⃣  Creating experiment...
3️⃣  Loading trajectories...
4️⃣  Generating perturbations...
5️⃣  Creating annotations (stubbed)...
6️⃣  Creating judge evaluations (stubbed)...
7️⃣  Computing CCG scores...
✅ Pipeline complete!

PASSED test_experiment_schema_format
✅ Experiment schema correct (no arrays, only counts)

PASSED test_trajectory_schema_format
✅ Trajectory schema correct (foreign keys present)

PASSED test_annotation_schema_format
✅ Annotation schema correct (foreign keys + TCS)

PASSED test_judge_evaluation_schema_format
✅ Judge evaluation schema correct (foreign keys + JPS)

PASSED test_ccg_score_schema_format
✅ CCG score schema correct (all foreign keys + CCG formula)

PASSED test_cache_check_works
✅ Cache check working correctly

PASSED test_pagination_queries
✅ Pagination working correctly

PASSED test_cross_collection_queries
✅ Cross-collection queries working correctly
```

## Verify Output Data

After running tests, inspect the JSON files:

```bash
# View all results
ls tests/data/results/

# View experiment (no arrays!)
cat tests/data/results/experiment.json | jq .

# View trajectories (have experiment_id foreign key)
cat tests/data/results/trajectories.json | jq '.[] | {trajectory_id, experiment_id, is_perturbed}'

# View annotations
cat tests/data/results/annotations.json | jq .

# View judge evaluations
cat tests/data/results/judge_evaluations.json | jq .

# View CCG scores (all foreign keys present)
cat tests/data/results/ccg_scores.json | jq '.[] | {experiment_id, trajectory_id, annotation_id, evaluation_id, ccg}'
```

## Expected Results

### Experiment Document
```javascript
{
  experiment_id: "test_integration_001",
  name: "Integration Test Experiment",
  config: { ... },
  progress: {
    trajectories_loaded: 1,        // Count, not array!
    perturbations_generated: 1,
    annotations_completed: 1,
    evaluations_completed: 1,
    ccg_scores_computed: 1
  },
  status: "completed"
  // ⚠️  NO trajectory_refs array!
}
```

### Trajectory Document
```javascript
{
  trajectory_id: "toolbench_sample_001",
  experiment_id: "test_integration_001",  // Foreign key!
  benchmark: "toolbench",
  is_perturbed: false,
  steps: [ ... ],
  ground_truth: { ... }
}
```

### CCG Score Document
```javascript
{
  ccg_id: "ccg_test_integration_001_001",
  experiment_id: "test_integration_001",      // FK
  trajectory_id: "toolbench_sample_001_perturbed",  // FK
  annotation_id: "ann_test_integration_001_001",    // FK
  evaluation_id: "eval_test_integration_001_claude_001",  // FK
  true_criticality_score: 110.0,
  judge_penalty_score: 25.0,
  criticality_calibration_gap: -0.773,
  calibration_status: "under_penalized"
}
```

## Troubleshooting

### MongoDB Connection Error
```
Error: Could not connect to MongoDB
```
**Solution:** Start MongoDB on port 27017

### Import Errors
```
ModuleNotFoundError: No module named 'src'
```
**Solution:** Run from repo root: `cd /Users/amanzing/Paper-Research && pytest tests/`

### Test Data Not Found
```
FileNotFoundError: test_config.json
```
**Solution:** Ensure you're in the repo root and `tests/data/` exists

## Adding New Tests

To add a new integration test:

1. Add test data to `tests/data/`
2. Update `stubbed_responses.json` if needed
3. Add test method to `TestIntegrationPipeline` class
4. Follow pattern: verify schema format + foreign keys

Example:
```python
def test_new_feature(self, storage, pipeline_results):
    """Test new feature schema format."""
    experiment_id = pipeline_results["experiment_id"]
    
    # Query using foreign key
    data = storage.get_something(experiment_id)
    
    # Verify foreign keys
    assert "experiment_id" in data
    assert data["experiment_id"] == experiment_id
    
    print("✅ New feature working correctly")
```

## References

- Schema Design: [/docs/SCHEMA_DESIGN_NOTES.md](/docs/SCHEMA_DESIGN_NOTES.md)
- Implementation Changes: [/docs/SCHEMA_IMPLEMENTATION_CHANGES.md](/docs/SCHEMA_IMPLEMENTATION_CHANGES.md)
- MongoDB Schema: [/docs/MONGODB_SCHEMA.md](/docs/MONGODB_SCHEMA.md)
