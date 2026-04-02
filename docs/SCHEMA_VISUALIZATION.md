# MongoDB Schema Structure with Content-Addressable Keys

## Overview

This design uses **content-addressable storage** where:
- Experiment configs are hashed (SHA256) to create unique experiment IDs
- Trajectory parameters are hashed to enable caching
- Only the experiment collection stores references, not duplicates

This is similar to Git's content-addressable storage model.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONFIG FOLDER (Source of Truth)              │
├─────────────────────────────────────────────────────────────────┤
│  config/experiments/                                            │
│  ├── poc_2026_04_02.json                                        │
│  ├── scaled_2026_04_10.json                                     │
│  └── production_2026_05_01.json                                 │
│                                                                  │
│  Each JSON defines:                                             │
│  - Datasets & filters                                           │
│  - Perturbation config                                          │
│  - Judge models & params                                        │
│  - Annotation requirements                                      │
│  - Execution settings                                           │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ SHA256(config.json)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EXPERIMENTS COLLECTION                       │
│                  (Metadata & References Only)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  {                                                               │
│    experiment_hash: "a3f5b2c1...",  ← SHA256 of config          │
│    config: { ...full config JSON... },                          │
│    status: "in_progress",                                       │
│    trajectory_refs: [                                           │
│      "traj_hash_1", "traj_hash_2", ...  ← References only       │
│    ],                                                            │
│    annotation_refs: ["ann_hash_1", ...],                        │
│    judge_eval_refs: ["eval_hash_1", ...],                       │
│    ccg_refs: ["ccg_hash_1", ...],                               │
│    created_at: "...",                                            │
│    progress: { ... }                                             │
│  }                                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ Points to cached trajectories
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TRAJECTORIES COLLECTION                       │
│              (Content-Addressable Cache Storage)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  {                                                               │
│    trajectory_hash: "d7e9f1a2...",  ← SHA256 of params          │
│    cache_key: {                      ← What was hashed          │
│      benchmark: "toolbench",                                     │
│      hf_index: 123,                                             │
│      filters: { min_steps: 5, ... },                            │
│      is_perturbed: false,                                       │
│      perturbation: null                                         │
│    },                                                            │
│    trajectory_data: {                ← Actual trajectory         │
│      steps: [ ... ],                                            │
│      ground_truth: { ... }                                      │
│    },                                                            │
│    referenced_by: [                  ← Which experiments use it  │
│      "a3f5b2c1...",                                             │
│      "b8c4d6e2..."                                              │
│    ],                                                            │
│    first_cached: "2026-04-02T12:00:00Z",                        │
│    last_accessed: "2026-04-05T15:30:00Z",                       │
│    access_count: 5                                              │
│  }                                                               │
│                                                                  │
│  Perturbed trajectories:                                        │
│  {                                                               │
│    trajectory_hash: "e8f2a3b4...",                              │
│    cache_key: {                                                  │
│      original_trajectory_hash: "d7e9f1a2...",                   │
│      perturbation_type: "planning",                             │
│      perturbation_position: "early",                            │
│      perturbation_params: { ... }                               │
│    },                                                            │
│    trajectory_data: { ... },                                    │
│    referenced_by: ["a3f5b2c1..."],                              │
│    ...                                                           │
│  }                                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ Evaluated by judges
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                 ANNOTATIONS COLLECTION                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  {                                                               │
│    annotation_hash: "f9a3b5c2...",                              │
│    experiment_hash: "a3f5b2c1...",                              │
│    trajectory_hash: "e8f2a3b4...",  ← Points to trajectory      │
│    annotator: "human_researcher",                               │
│    tsd: 1.0,                                                     │
│    ser: 0.67,                                                    │
│    tcs: 110.0,                                                   │
│    annotated_at: "...",                                          │
│    ...                                                           │
│  }                                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              JUDGE_EVALUATIONS COLLECTION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  {                                                               │
│    evaluation_hash: "g1b4c6d3...",                              │
│    experiment_hash: "a3f5b2c1...",                              │
│    trajectory_hash: "e8f2a3b4...",                              │
│    judge_model: "claude-3.5-sonnet",                            │
│    cache_key: {                      ← For judge call caching   │
│      trajectory_hash: "e8f2a3b4...",                            │
│      judge_config: { temp: 0.7, ... },                          │
│      prompt_version: "v1.0"                                     │
│    },                                                            │
│    overall_score: 85.0,                                          │
│    jps: 15.0,                                                    │
│    errors_identified: [ ... ],                                  │
│    evaluated_at: "...",                                          │
│    ...                                                           │
│  }                                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   CCG_SCORES COLLECTION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  {                                                               │
│    ccg_hash: "h2c5d7e4...",                                     │
│    experiment_hash: "a3f5b2c1...",                              │
│    trajectory_hash: "e8f2a3b4...",                              │
│    annotation_hash: "f9a3b5c2...",                              │
│    evaluation_hash: "g1b4c6d3...",                              │
│    tcs: 110.0,                                                   │
│    jps: 15.0,                                                    │
│    ccg: -0.86,                                                   │
│    computed_at: "...",                                           │
│    ...                                                           │
│  }                                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### 1. Content-Addressable Experiments

```python
import json
import hashlib

def compute_experiment_hash(config: dict) -> str:
    """
    Compute SHA256 hash of experiment config.
    
    This creates a deterministic, unique identifier.
    Same config = same hash = can reuse results.
    """
    # Normalize config (sort keys, remove comments)
    config_canonical = json.dumps(config, sort_keys=True)
    
    # Compute SHA256
    hash_obj = hashlib.sha256(config_canonical.encode('utf-8'))
    return hash_obj.hexdigest()

# Example
config = json.load(open("config/experiments/poc_2026_04_02.json"))
experiment_hash = compute_experiment_hash(config)
# → "a3f5b2c189d7e4f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5"
```

### 2. Content-Addressable Trajectories

```python
def compute_trajectory_hash(trajectory_params: dict) -> str:
    """
    Compute SHA256 hash of trajectory parameters.
    
    Used for caching: if we've seen these exact params before,
    we can reuse the cached trajectory instead of reloading.
    """
    params_canonical = json.dumps(trajectory_params, sort_keys=True)
    hash_obj = hashlib.sha256(params_canonical.encode('utf-8'))
    return hash_obj.hexdigest()

# Original trajectory params
original_params = {
    "benchmark": "toolbench",
    "hf_index": 123,
    "split": "train",
    "filters": {"min_steps": 5, "max_steps": 10},
    "is_perturbed": False,
    "perturbation": None
}
orig_hash = compute_trajectory_hash(original_params)

# Perturbed trajectory params
perturbed_params = {
    "original_trajectory_hash": orig_hash,
    "perturbation_type": "planning",
    "perturbation_position": "early",
    "perturbation_params": {"method": "goal_substitution"}
}
pert_hash = compute_trajectory_hash(perturbed_params)
```

### 3. Cache Lookup Logic

```python
def get_or_load_trajectory(trajectory_params: dict, storage) -> str:
    """
    Check cache first, load from HuggingFace only if needed.
    """
    # Compute hash
    traj_hash = compute_trajectory_hash(trajectory_params)
    
    # Check if cached
    cached = storage.trajectories.find_one({"trajectory_hash": traj_hash})
    
    if cached:
        print(f"✓ Cache hit: {traj_hash[:8]}...")
        # Update access tracking
        storage.trajectories.update_one(
            {"trajectory_hash": traj_hash},
            {
                "$set": {"last_accessed": datetime.utcnow()},
                "$inc": {"access_count": 1}
            }
        )
        return traj_hash
    else:
        print(f"✗ Cache miss: Loading from HuggingFace...")
        # Load trajectory
        trajectory_data = load_from_huggingface(trajectory_params)
        
        # Store in cache
        storage.trajectories.insert_one({
            "trajectory_hash": traj_hash,
            "cache_key": trajectory_params,
            "trajectory_data": trajectory_data,
            "first_cached": datetime.utcnow(),
            "last_accessed": datetime.utcnow(),
            "access_count": 1,
            "referenced_by": []
        })
        
        return traj_hash
```

---

## Benefits of This Design

### 1. **Automatic Deduplication**
- If two experiments use the same trajectory → stored once
- If you re-run an experiment with same config → reuses all cached data
- Saves storage space and API costs

### 2. **Reproducibility**
```python
# Same config always produces same experiment_hash
experiment_1 = run_experiment("config/experiments/poc.json")
experiment_2 = run_experiment("config/experiments/poc.json")

assert experiment_1.hash == experiment_2.hash
# Can compare results directly, no ambiguity
```

### 3. **Incremental Experiments**
```python
# Config v1: 50 trajectories
config_v1 = load_config("poc_v1.json")
run_experiment(config_v1)  # Loads 50 trajectories

# Config v2: Same 50 + 50 more
config_v2 = load_config("poc_v2.json")
run_experiment(config_v2)  # Reuses first 50 (cache hit), loads 50 new
```

### 4. **Cross-Experiment Analysis**
```javascript
// Find all experiments that used a specific trajectory
db.trajectories.findOne({
  "trajectory_hash": "d7e9f1a2..."
}).referenced_by
// → ["a3f5b2c1...", "b8c4d6e2...", ...]

// Find all experiments with GPT-OSS judge
db.experiments.find({
  "config.judges.models.name": "gpt-oss-120b"
})
```

### 5. **Efficient Storage**
- Trajectory used in 10 experiments? Stored once, referenced 10 times
- Judge evaluations cached (same trajectory + judge + prompt → same evaluation)
- Only experiment metadata duplicated, actual data shared

---

## Example Queries

### Check if experiment already run
```python
import hashlib
import json

config = json.load(open("config/experiments/poc.json"))
experiment_hash = hashlib.sha256(
    json.dumps(config, sort_keys=True).encode()
).hexdigest()

existing = db.experiments.find_one({"experiment_hash": experiment_hash})

if existing:
    print(f"Experiment already run: {existing['status']}")
    if existing['status'] == 'completed':
        print("Loading cached results...")
        # Load from database, no re-computation
else:
    print("New experiment, running...")
```

### Find cached trajectory
```python
trajectory_params = {
    "benchmark": "toolbench",
    "hf_index": 123,
    ...
}

traj_hash = compute_trajectory_hash(trajectory_params)
cached = db.trajectories.find_one({"trajectory_hash": traj_hash})

if cached:
    print(f"✓ Found in cache (accessed {cached['access_count']} times)")
    trajectory_data = cached['trajectory_data']
else:
    print("Not cached, loading from HuggingFace...")
```

### Cross-experiment comparison
```python
# Compare CCG for same trajectory across experiments
db.ccg_scores.aggregate([
  {
    $match: {
      trajectory_hash: "e8f2a3b4..."
    }
  },
  {
    $group: {
      _id: "$experiment_hash",
      mean_ccg: { $avg: "$ccg" },
      judges: { $addToSet: "$judge_model" }
    }
  }
])
```

---

## Updated Schema Documents

### experiments
```javascript
{
  experiment_hash: "a3f5b2c1...",  // SHA256(config)
  config: { ...full config... },   // From config/experiments/*.json
  status: "in_progress",
  
  // References (arrays of hashes)
  trajectory_refs: ["d7e9f1a2...", "e8f2a3b4...", ...],
  annotation_refs: ["f9a3b5c2...", ...],
  judge_eval_refs: ["g1b4c6d3...", ...],
  ccg_refs: ["h2c5d7e4...", ...],
  
  progress: {
    trajectories_cached: 50,
    perturbations_generated: 50,
    annotations_completed: 25,
    evaluations_completed: 150,
    ccg_scores_computed: 150
  },
  
  created_at: "...",
  updated_at: "..."
}
```

### trajectories
```javascript
{
  trajectory_hash: "d7e9f1a2...",  // SHA256(cache_key)
  cache_key: {
    // What uniquely identifies this trajectory
    benchmark: "toolbench",
    hf_index: 123,
    filters: {...},
    is_perturbed: false,
    perturbation: null
  },
  trajectory_data: {
    // Actual trajectory content
    steps: [...],
    ground_truth: {...}
  },
  
  // Cache management
  referenced_by: ["a3f5b2c1...", "b8c4d6e2..."],
  first_cached: "...",
  last_accessed: "...",
  access_count: 5
}
```

---

## Implementation Plan

1. ✅ Create `config/experiments/` directory
2. ✅ Example experiment config JSON
3. ⏳ Implement `compute_experiment_hash()` function
4. ⏳ Implement `compute_trajectory_hash()` function  
5. ⏳ Update MongoDB storage with content-addressable methods
6. ⏳ Implement cache lookup logic
7. ⏳ Update experiment runner to use config system

**Does this schema structure align with your vision?** Any changes before I implement?
