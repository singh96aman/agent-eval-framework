# Experiment Configuration System

## Overview

Experiments are defined as JSON configuration files in `config/experiments/`. Each config is hashed (SHA256) to create a unique experiment ID, enabling:
- **Reproducibility**: Same config → same results
- **Caching**: Automatic reuse of trajectories, evaluations
- **Deduplication**: Shared data stored once
- **Version control**: Track config changes in git

## Quick Start

### 1. Create Experiment Config

```bash
# Copy template
cp config/experiments/poc_2026_04_02.json config/experiments/my_experiment.json

# Edit your config
vim config/experiments/my_experiment.json
```

### 2. Run Experiment

```python
from src.experiment_runner import run_experiment

# Automatically computes experiment_hash from config
# Caches all trajectories, evaluations, etc.
results = run_experiment("config/experiments/my_experiment.json")
```

### 3. Check if Already Run

```python
# If same config was run before, reuses cached results
# No API costs, instant results
```

## Config Structure

```json
{
  "experiment": {
    "name": "...",
    "description": "...",
    "tags": ["..."]
  },
  "datasets": {
    "toolbench": { ... },
    "gaia": { ... }
  },
  "perturbations": { ... },
  "judges": { ... },
  "annotation": { ... },
  "metrics": { ... },
  "execution": { ... }
}
```

## Content-Addressable Keys

### Experiment Hash
```python
SHA256(config.json) → "a3f5b2c189d7e4f1..."
```
- Unique identifier for the experiment
- Same config always produces same hash
- Used to check if experiment already run

### Trajectory Hash
```python
SHA256({
  benchmark: "toolbench",
  hf_index: 123,
  filters: {...},
  perturbation: {...}
}) → "d7e9f1a289c4b3e2..."
```
- Unique identifier for trajectory + perturbation
- Cache key for trajectory lookup
- Enables sharing across experiments

## Benefits

### 1. Cost Savings
```
Experiment 1: Load 50 trajectories → $10 API cost
Experiment 2: Reuse 40 trajectories → $2 API cost (only 10 new)
Savings: 80%
```

### 2. Speed
```
First run: Load from HuggingFace (30 min)
Repeat run: Load from cache (30 sec)
Speedup: 60x
```

### 3. Reproducibility
```bash
# Run today
python run.py config/experiments/poc.json
# → experiment_hash: a3f5b2c1...

# Run next week (same config)
python run.py config/experiments/poc.json
# → experiment_hash: a3f5b2c1... (identical!)
# → All results match exactly
```

## Examples

See `config/experiments/poc_2026_04_02.json` for full example with:
- Dual benchmarks (ToolBench + GAIA)
- 3 perturbation types × 3 positions
- 2 judges (Claude + GPT-OSS)
- Human-in-loop annotation
- Caching enabled

## Schema Documentation

- [SCHEMA_VISUALIZATION.md](../docs/SCHEMA_VISUALIZATION.md) - Full schema design
- [MONGODB_SCHEMA.md](../docs/MONGODB_SCHEMA.md) - Collection details
- [AWS_BEDROCK_INTEGRATION.md](../docs/AWS_BEDROCK_INTEGRATION.md) - Judge setup
