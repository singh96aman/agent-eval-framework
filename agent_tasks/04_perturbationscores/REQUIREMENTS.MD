## You are an autonomous researcher with instructions outlined in CLAUDE.md

### Abstract

This research investigates whether LLM-based judges are systematically miscalibrated when evaluating agent trajectories. Specifically, we hypothesize that judges under-penalize early structural failures (e.g., flawed planning, wrong tool selection) while over-penalizing late local mistakes (e.g., parameter typos). Early errors are more critical because they propagate downstream and corrupt subsequent steps, yet judges may focus on surface-level correctness rather than causal impact. We introduce the Criticality-Calibration Gap (CCG) metric to quantify this miscalibration by comparing judge penalty scores against ground-truth criticality derived from task success degradation and downstream error propagation. Through controlled perturbation experiments on ToolBench and GAIA benchmarks, we aim to reveal systematic blind spots in LLM judgment that have implications for agent evaluation, RLHF reward modeling, and AI safety oversight.

### Context

Human Annotation and LLMaJ are long pole items. 1 judge can take up to 8 hours over a sample of 500 while human beings can probably burn through 300 samples in 4-5 hours. In `/Users/amanzing/Paper-Research/agent_tasks/02_poc_implementation` we ran LLMaJ over 600 items. Iterating on this piece would be much much slower. We can speed things up if we have good trajectories and perturbations recorded.

Additionally, we've successfully added dataset diversity and perturbation diversity in task `03_dataset_exploration`.

---

## Target Config

The sub-agent should create/update the config file at `config/experiments/perturbation_quality.json`:

```json
{
  "experiment": {
    "name": "Perturbation Quality Scoring & Primary Selection",
    "experiment_id": "exp_perturbation_quality_v1",
    "description": "Score perturbation validity and select primary sample for human annotation",
    "version": "1.0",
    "created_by": "researcher",
    "created_date": "2026-04-03"
  },

  "datasets": {
    "toolbench": {
      "enabled": true,
      "source": "json",
      "json_path": "data/sampled/toolbench_400.json",
      "num_trajectories": 400,
      "filters": {
        "min_steps": 4,
        "max_steps": 15,
        "filter_successful": true
      },
      "random_seed": 42
    },
    "gaia": {
      "enabled": true,
      "source": "json",
      "json_path": "data/sampled/gaia_100.json",
      "num_trajectories": 100,
      "filters": {
        "min_steps": 4,
        "max_steps": 100
      },
      "random_seed": 42
    },
    "swebench": {
      "enabled": true,
      "source": "json",
      "json_path": "data/sampled/swebench_100.json",
      "num_trajectories": 100,
      "filters": {
        "min_steps": 4,
        "max_steps": 30
      },
      "random_seed": 42
    }
  },

  "storage": {
    "backend": "mongodb",
    "database": "agent_judge_experiment"
  },

  "perturbations": {
    "types": ["planning", "tool_selection", "parameter", "data_reference"],
    "positions": ["early", "middle", "late"],
    "mode": "static",
    "batch_size": 50,
    "conditions": [
      {"type": "planning", "position": "early"},
      {"type": "planning", "position": "middle"},
      {"type": "planning", "position": "late"},
      {"type": "tool_selection", "position": "early"},
      {"type": "tool_selection", "position": "middle"},
      {"type": "tool_selection", "position": "late"},
      {"type": "parameter", "position": "early"},
      {"type": "parameter", "position": "middle"},
      {"type": "parameter", "position": "late"},
      {"type": "data_reference", "position": "middle"},
      {"type": "data_reference", "position": "late"}
    ],
    "assignment_strategy": "round_robin",
    "num_per_trajectory": 1
  },

  "quality_scoring": {
    "enabled": true,
    "judge_model": {
      "name": "claude-sonnet-4-5",
      "provider": "aws_bedrock",
      "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
      "config": {
        "temperature": 0.3,
        "max_tokens": 500
      }
    },
    "metrics": {
      "content_changed": {
        "description": "Did the content actually change from original?",
        "score_range": [0, 1],
        "weight": 1.0
      },
      "syntactically_valid": {
        "description": "Is the perturbed content well-formed (not broken JSON, coherent text)?",
        "score_range": [0, 1],
        "weight": 1.0
      },
      "semantically_meaningful": {
        "description": "Is the change semantically meaningful (not just whitespace)?",
        "score_range": [0, 1],
        "weight": 1.0
      },
      "type_matches_intent": {
        "description": "Does the error match the intended perturbation type?",
        "score_range": [0, 1],
        "weight": 1.0
      },
      "realistic_error": {
        "description": "Could this error plausibly occur in a real agent execution?",
        "score_range": [0, 3],
        "weight": 1.0
      }
    },
    "total_score_range": [0, 7],
    "tier_thresholds": {
      "high": 6,
      "medium": 4,
      "low": 1,
      "invalid": 0
    },
    "batch_size": 10,
    "retry_on_failure": true,
    "max_retries": 2
  },

  "primary_selection": {
    "enabled": true,
    "total_primary": 600,
    "tier_distribution": {
      "high": 0.33,
      "medium": 0.50,
      "low": 0.17
    },
    "quota_strategy": "balanced_conditions",
    "min_per_condition": 40,
    "conditions": [
      {"type": "planning", "position": "early"},
      {"type": "planning", "position": "middle"},
      {"type": "planning", "position": "late"},
      {"type": "tool_selection", "position": "early"},
      {"type": "tool_selection", "position": "middle"},
      {"type": "tool_selection", "position": "late"},
      {"type": "parameter", "position": "early"},
      {"type": "parameter", "position": "middle"},
      {"type": "parameter", "position": "late"},
      {"type": "data_reference", "position": "middle"},
      {"type": "data_reference", "position": "late"}
    ],
    "random_seed": 42
  },

  "judges": {
    "models": [
      {
        "name": "claude-sonnet-4-5",
        "provider": "aws_bedrock",
        "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "config": {
          "temperature": 0.7,
          "max_tokens": 2000
        }
      },
      {
        "name": "gpt-4o",
        "provider": "aws_bedrock",
        "model_id": "openai.gpt-4o-1:0",
        "config": {
          "temperature": 0.7,
          "max_tokens": 2000
        }
      }
    ],
    "samples_per_trajectory": 3
  },

  "execution": {
    "runner": "load,perturb",
    "dry_run": false,
    "enable_checkpointing": true,
    "checkpoint_dir": "results/checkpoints",
    "verbose": true,
    "max_retries": 3
  }
}
```

---

## Goal: Perturbation Quality Scoring & Primary Sample Selection

### Problem Statement

We have generated **all perturbations** for all trajectories (~5000+ perturbations for 600 trajectories). However:

1. **Human annotation** requires 1 perturbation per trajectory (avoid fatigue seeing same trajectory)
2. **LLMaJ** can run on all perturbations (no fatigue)
3. **Quality varies** — some perturbations are invalid, unrealistic, or low-quality (POC found template artifacts, text corruption)
4. **Balance required** — need equal distribution across perturbation types and positions for statistical power

### Solution: LLMaJ Quality Scoring + Stratified Sampling

**Approach:**
1. Run LLMaJ to score **validity/quality** of each perturbation (not type selection)
2. Sample across quality spectrum (high/medium/low) to fill quotas
3. Mark sampled perturbations as `is_primary_for_experiment = True`
4. Human annotation targets primary perturbations only; LLMaJ runs on all

**Why this works:**
- Perturbation **type** assignment remains random (no circular reasoning)
- LLMaJ only scores **validity**, not difficulty or detectability
- Sampling includes medium/low quality to avoid filtering out interesting edge cases
- Quotas ensure balanced distribution

---

## Execution Plan

| Step | Description | Status | Validation |
|------|-------------|--------|------------|
| 1 | Add `quality_scoring` and `primary_selection` sections to config | ⬜ TODO | Config validates |
| 2 | Design quality scoring prompt (store in `src/scoring/prompts.py`) | ⬜ TODO | Prompt review |
| 3 | Implement `PerturbationQualityScorer` module | ⬜ TODO | Unit tests pass |
| 4 | Implement `PrimarySelector` module | ⬜ TODO | Unit tests pass |
| 5 | Integrate both into `_phase_generate_perturbations` | ⬜ TODO | Integration test |
| 6 | Run `--runner perturb` with both flags enabled | ⬜ TODO | Perturbations + scores + primaries |
| 7 | Validate quality score distribution | ⬜ TODO | Distribution analysis |
| 8 | Validate primary sample distribution | ⬜ TODO | Quotas met, 600 primaries |
| 9 | Final review and mark complete | ⬜ TODO | Expert approval |

---

## Step 1: Add Config Sections

### 1.1 Task

Add `quality_scoring` and `primary_selection` sections to experiment config file (see "Config Schema Requirements" section above).

### 1.2 Validation Criteria

| Check | Criterion | Pass |
|-------|-----------|------|
| Config parses | No JSON syntax errors | ⬜ |
| Required fields present | All fields in schema exist | ⬜ |
| Judge model valid | Model ID is correct for provider | ⬜ |
| Metrics complete | All 5 metrics defined | ⬜ |
| Tier thresholds valid | high > medium > low > invalid | ⬜ |

---

## Step 2: Design Quality Scoring Prompt

### 2.1 Quality Criteria (Validity-Focused, NOT Difficulty-Focused)

The quality score measures **perturbation validity**, not how hard it is to detect:

| Criterion | Description | Score |
|-----------|-------------|-------|
| **content_changed** | Did the content actually change from original? | 0/1 |
| **syntactically_valid** | Is the perturbed content syntactically valid (not broken JSON, coherent text)? | 0/1 |
| **semantically_meaningful** | Is the change semantically meaningful (not just whitespace, punctuation)? | 0/1 |
| **type_matches_intent** | Does the error match the intended perturbation type? | 0/1 |
| **realistic_error** | Could this error plausibly occur in a real agent execution? | 0-3 |

**Total Score: 0-7**

### 1.2 Quality Scoring Prompt

```
You are evaluating the VALIDITY of a perturbation applied to an agent trajectory.

Your task is to score how well-formed and realistic this perturbation is, NOT how difficult it would be to detect.

## Original Step
{original_step_content}

## Perturbed Step  
{perturbed_step_content}

## Perturbation Metadata
- Type: {perturbation_type}
- Position: {perturbation_position}
- Intended error: {perturbation_description}

## Scoring Criteria

Rate each criterion:

1. **content_changed** (0 or 1): Did the content actually change?
   - 0 = Identical or trivially different (whitespace only)
   - 1 = Meaningful change occurred

2. **syntactically_valid** (0 or 1): Is the perturbed content well-formed?
   - 0 = Broken syntax, corrupted text, invalid JSON
   - 1 = Valid syntax, coherent text

3. **semantically_meaningful** (0 or 1): Is the change meaningful?
   - 0 = Change is trivial (punctuation, formatting only)
   - 1 = Change affects meaning or behavior

4. **type_matches_intent** (0 or 1): Does the error match the intended type?
   - 0 = Error is different from intended type (e.g., labeled "parameter" but changed planning)
   - 1 = Error matches the intended perturbation type

5. **realistic_error** (0-3): Could this error plausibly occur in a real agent?
   - 0 = Impossible/nonsensical error
   - 1 = Unlikely but possible
   - 2 = Plausible error
   - 3 = Very realistic, common agent failure mode

## Output Format (JSON)

{
  "content_changed": 0 or 1,
  "syntactically_valid": 0 or 1,
  "semantically_meaningful": 0 or 1,
  "type_matches_intent": 0 or 1,
  "realistic_error": 0-3,
  "total_score": sum of above (0-7),
  "reasoning": "Brief explanation of scores"
}
```

### 1.3 Schema Addition to Perturbation Document

```python
{
    "perturbation_id": "...",
    "experiment_id": "...",
    # ... existing fields ...
    
    # NEW: Quality scoring fields
    "quality_score": {
        "content_changed": 1,
        "syntactically_valid": 1,
        "semantically_meaningful": 1,
        "type_matches_intent": 1,
        "realistic_error": 2,
        "total_score": 6,
        "reasoning": "...",
        "scored_by": "claude-sonnet-4-5",
        "scored_at": "2026-04-03T12:00:00Z"
    },
    
    # NEW: Primary selection fields
    "is_primary_for_experiment": False,  # Set True after sampling
    "primary_selection_reason": null,    # "quota_fill", "quality_tier"
    "quality_tier": null                 # "high", "medium", "low"
}
```

### 2.4 Validation Criteria for Step 2

| Check | Criterion | Pass |
|-------|-----------|------|
| Prompt completeness | All 5 criteria defined | ⬜ |
| Output format | Valid JSON schema | ⬜ |
| Score range | Total 0-7 | ⬜ |
| No difficulty bias | Criteria focus on validity, not detectability | ⬜ |
| Prompt stored | File exists at `src/scoring/prompts.py` | ⬜ |

---

## Step 3: Implement Quality Scorer Module

### 2.1 Module Location

```
src/
  scoring/
    __init__.py
    quality_scorer.py      # NEW
    prompts.py             # NEW (store prompts)
```

### 2.2 Interface Design

```python
# src/scoring/quality_scorer.py

class PerturbationQualityScorer:
    """
    Scores perturbation validity using LLMaJ.
    
    Usage:
        scorer = PerturbationQualityScorer(model="claude-sonnet-4-5")
        score = scorer.score_perturbation(perturbation_doc)
        # Returns: {"total_score": 6, "content_changed": 1, ...}
    """
    
    def __init__(self, model: str, random_seed: int = 42):
        """Initialize with LLM model for scoring."""
        pass
    
    def score_perturbation(self, perturbation: dict) -> dict:
        """Score a single perturbation's validity."""
        pass
    
    def score_batch(self, perturbations: list[dict], batch_size: int = 10) -> list[dict]:
        """Score multiple perturbations with batching."""
        pass
    
    def update_perturbation_in_db(self, perturbation_id: str, score: dict) -> None:
        """Update perturbation document with quality score."""
        pass
```

### 2.3 Validation Criteria for Step 2

| Check | Criterion | Pass |
|-------|-----------|------|
| Unit test: score_perturbation | Returns valid score dict | ⬜ |
| Unit test: score range | All scores within expected range | ⬜ |
| Unit test: JSON parsing | Handles LLM output parsing errors | ⬜ |
| Unit test: batch processing | Processes batches correctly | ⬜ |
| Integration test | Scores 10 real perturbations | ⬜ |

**Test file:** `tests/test_quality_scorer.py`

---

## Step 3: Run Quality Scoring on All Perturbations

### 3.1 Execution

```bash
python abcd.py --config config/experiments/dataset_quality_check.json --runner score_quality
```

### 3.2 Expected Output

- All perturbations in `exp_dataset_quality` have `quality_score` field populated
- Progress logged every 100 perturbations
- Failures logged with perturbation_id for retry

### 3.3 Validation Criteria for Step 3

| Check | Criterion | Pass |
|-------|-----------|------|
| Coverage | 100% of perturbations scored | ⬜ |
| No nulls | All `quality_score.total_score` are non-null | ⬜ |
| Score distribution | Scores span 0-7 range (not all same value) | ⬜ |
| Timing | Average < 5 seconds per perturbation | ⬜ |

**Validation query:**
```python
# Check coverage
total = db.perturbations.count_documents({"experiment_id": EXP_ID})
scored = db.perturbations.count_documents({
    "experiment_id": EXP_ID,
    "quality_score.total_score": {"$exists": True}
})
assert scored == total, f"Coverage: {scored}/{total}"

# Check distribution
scores = [p["quality_score"]["total_score"] for p in db.perturbations.find({"experiment_id": EXP_ID})]
assert min(scores) < 3, "No low scores — suspicious"
assert max(scores) > 5, "No high scores — suspicious"
```

---

## Step 4: Implement Primary Selector Module

### 4.1 Sampling Strategy

**Quality Tiers:**
- High: score >= 6 (top tier)
- Medium: score 4-5 (middle tier)
- Low: score 1-3 (bottom tier, but still valid — score > 0)
- Invalid: score = 0 (excluded)

**Target Distribution for 600 Primary Perturbations:**
- High quality: 200 (33%)
- Medium quality: 300 (50%)
- Low quality: 100 (17%)

**Quota per Condition (11 conditions):**
- ~55 per condition (type × position)
- Enforce minimum 40 per condition

### 4.2 Sampling Algorithm

```python
def select_primary_perturbations(
    perturbations: list[dict],
    total_target: int = 600,
    tier_distribution: dict = {"high": 0.33, "medium": 0.50, "low": 0.17},
    min_per_condition: int = 40
) -> list[str]:
    """
    Select primary perturbations using stratified sampling with quotas.
    
    Returns list of perturbation_ids to mark as primary.
    """
    
    # Step 1: Assign quality tiers
    for p in perturbations:
        score = p["quality_score"]["total_score"]
        if score >= 6:
            p["quality_tier"] = "high"
        elif score >= 4:
            p["quality_tier"] = "medium"
        elif score >= 1:
            p["quality_tier"] = "low"
        else:
            p["quality_tier"] = "invalid"
    
    # Step 2: Filter out invalid (score = 0)
    valid = [p for p in perturbations if p["quality_tier"] != "invalid"]
    
    # Step 3: Group by condition (type × position)
    conditions = defaultdict(list)
    for p in valid:
        cond = (p["perturbation_type"], p["perturbation_position"])
        conditions[cond].append(p)
    
    # Step 4: For each condition, sample to meet quota
    selected = []
    assigned_trajectories = set()
    
    for cond, cond_perturbations in conditions.items():
        # Sort by quality score (descending) within tier preferences
        # Prioritize: high > medium > low, then by score within tier
        cond_perturbations.sort(
            key=lambda p: (
                {"high": 3, "medium": 2, "low": 1}[p["quality_tier"]],
                p["quality_score"]["total_score"]
            ),
            reverse=True
        )
        
        quota = max(min_per_condition, total_target // len(conditions))
        selected_for_cond = 0
        
        for p in cond_perturbations:
            if p["original_trajectory_id"] in assigned_trajectories:
                continue  # Already assigned this trajectory
            
            selected.append(p["perturbation_id"])
            assigned_trajectories.add(p["original_trajectory_id"])
            selected_for_cond += 1
            
            if selected_for_cond >= quota:
                break
    
    return selected
```

### 4.3 Module Location

```
src/
  sampling/
    __init__.py
    primary_selector.py    # NEW
```

### 4.4 Validation Criteria for Step 4

| Check | Criterion | Pass |
|-------|-----------|------|
| Unit test: tier assignment | Correct tier boundaries | ⬜ |
| Unit test: quota enforcement | Min per condition met | ⬜ |
| Unit test: no duplicate trajectories | Each trajectory appears once | ⬜ |
| Unit test: tier distribution | Matches target ±10% | ⬜ |

**Test file:** `tests/test_primary_selector.py`

---

## Step 5: Integrate Both into `_phase_generate_perturbations`

### 5.1 Integration Points

Modify `src/experiment_runner.py` `_phase_generate_perturbations()`:

```python
def _phase_generate_perturbations(self):
    # ... existing setup code ...
    
    # NEW: Initialize quality scorer if enabled
    quality_config = self.config.get("quality_scoring", {})
    scorer = None
    if quality_config.get("enabled", False):
        from src.scoring.quality_scorer import PerturbationQualityScorer
        scorer = PerturbationQualityScorer(
            model_config=quality_config["judge_model"],
            metrics=quality_config["metrics"]
        )
    
    # ... existing perturbation generation loop ...
    for i, traj in enumerate(trajectories):
        for condition in conditions:
            perturbed = generator.generate_perturbation(...)
            
            if perturbed:
                # NEW: Score quality if enabled
                if scorer:
                    quality_score = scorer.score_perturbation(perturbed)
                    perturbed.quality_score = quality_score
                
                batch.append(perturbed)
                # ... rest of existing code ...
    
    # NEW: Select primaries if enabled (after all perturbations stored)
    primary_config = self.config.get("primary_selection", {})
    if primary_config.get("enabled", False):
        from src.sampling.primary_selector import PrimarySelector
        selector = PrimarySelector(
            total_primary=primary_config["total_primary"],
            tier_distribution=primary_config["tier_distribution"],
            min_per_condition=primary_config["min_per_condition"]
        )
        
        # Load all perturbations for this experiment
        all_perturbations = self.storage.get_perturbations_by_experiment(self.experiment_id)
        
        # Select primaries
        primary_ids = selector.select(all_perturbations)
        
        # Mark primaries in DB
        self.storage.mark_primaries(primary_ids, self.experiment_id)
        
        print(f"✓ Selected {len(primary_ids)} primary perturbations")
```

### 5.2 Validation Criteria for Step 5

| Check | Criterion | Pass |
|-------|-----------|------|
| Integration test: scoring disabled | No `quality_score` when `enabled=false` | ⬜ |
| Integration test: scoring enabled | All perturbations have `quality_score` | ⬜ |
| Integration test: selection disabled | No `is_primary_for_experiment` when `enabled=false` | ⬜ |
| Integration test: selection enabled | 600 primaries marked | ⬜ |
| Integration test: both enabled | End-to-end works | ⬜ |

**Test file:** `tests/test_experiment_runner_integration.py`

---

## Step 6: Run Perturb with Both Flags Enabled

### 6.1 Execution

```bash
python main.py --config config/experiments/dataset_quality_check.json --runner perturb
```

With config:
```json
{
  "quality_scoring": { "enabled": true, ... },
  "primary_selection": { "enabled": true, ... }
}
```

### 6.2 Expected Output

1. All perturbations generated with `quality_score` field populated
2. 600 perturbations marked with `is_primary_for_experiment = True`
3. Each primary has `quality_tier` and `primary_selection_reason`
4. Progress logged; results in `results/primary_selection/`

### 6.3 Validation Criteria for Step 6

| Check | Criterion | Pass |
|-------|-----------|------|
| All scored | 100% of perturbations have `quality_score` | ⬜ |
| Total primaries | Exactly 600 `is_primary_for_experiment = True` | ⬜ |
| Unique trajectories | 600 unique `original_trajectory_id` in primaries | ⬜ |
| Condition coverage | All 11 conditions have >= 40 samples | ⬜ |
| Tier distribution | High ~200, Medium ~300, Low ~100 | ⬜ |

**Validation query:**
```python
# Check all scored
total = db.perturbations.count_documents({"experiment_id": EXP_ID})
scored = db.perturbations.count_documents({
    "experiment_id": EXP_ID,
    "quality_score.total_score": {"$exists": True}
})
assert scored == total, f"Scoring coverage: {scored}/{total}"

# Check primaries
primary = list(db.perturbations.find({
    "experiment_id": EXP_ID,
    "is_primary_for_experiment": True
}))
assert len(primary) == 600

# Check unique trajectories
traj_ids = [p["original_trajectory_id"] for p in primary]
assert len(set(traj_ids)) == 600

# Check condition distribution
from collections import Counter
conditions = Counter((p["perturbation_type"], p["perturbation_position"]) for p in primary)
for cond, count in conditions.items():
    assert count >= 40, f"Condition {cond} has only {count} samples"

# Check tier distribution
tiers = Counter(p["quality_tier"] for p in primary)
print(f"Tier distribution: {tiers}")
```

---

## Step 7: Validate Quality Score Distribution

### 7.1 Distribution Analysis

Generate report showing:

| Metric | Expected |
|--------|----------|
| Mean score | 4-5 (most perturbations are valid) |
| Std dev | > 1.0 (meaningful variance) |
| % score < 3 | 5-15% (some invalid perturbations) |
| % score >= 6 | 30-50% (good portion high quality) |

### 7.2 Score by Perturbation Type

| Type | Mean Score | Notes |
|------|------------|-------|
| planning | ? | Check for template artifacts |
| tool_selection | ? | Should be high (well-defined) |
| parameter | ? | Should be high |
| data_reference | ? | May be lower (harder to generate) |

### 7.3 Score by Benchmark

| Benchmark | Mean Score | Notes |
|-----------|------------|-------|
| ToolBench | ? | Expected highest (most mature) |
| GAIA | ? | Check for template issues |
| SWE-bench | ? | Check for template issues |

### 7.4 Validation Criteria for Step 7

| Check | Criterion | Pass |
|-------|-----------|------|
| Distribution report | Generated in `results/quality_scores/` | ⬜ |
| No systematic issues | No type/benchmark has mean < 3 | ⬜ |
| Variance exists | Std dev > 1.0 | ⬜ |
| Outliers reviewed | 10 lowest scores manually reviewed | ⬜ |

---

## Step 8: Validate Primary Sample Distribution

### 8.1 Generate Distribution Report

Create `results/primary_selection/distribution_report.md`:

```markdown
# Primary Sample Distribution Report

## Overall
- Total primary perturbations: 600
- Unique trajectories: 600

## By Perturbation Type
| Type | Count | % |
|------|-------|---|
| planning | X | Y% |
| tool_selection | X | Y% |
| parameter | X | Y% |
| data_reference | X | Y% |

## By Position
| Position | Count | % |
|----------|-------|---|
| early | X | Y% |
| middle | X | Y% |
| late | X | Y% |

## By Condition (Type × Position)
| Condition | Count | Target | Status |
|-----------|-------|--------|--------|
| (planning, early) | X | 55 | ✅/⚠️ |
| ... | | | |

## By Quality Tier
| Tier | Count | % | Target % |
|------|-------|---|----------|
| high | X | Y% | 33% |
| medium | X | Y% | 50% |
| low | X | Y% | 17% |

## By Benchmark
| Benchmark | Count | % |
|-----------|-------|---|
| ToolBench | X | Y% |
| GAIA | X | Y% |
| SWE-bench | X | Y% |
```

### 8.2 Validation Criteria for Step 8

| Check | Criterion | Pass |
|-------|-----------|------|
| Report generated | File exists at expected path | ⬜ |
| Type balance | Each type has 100-200 samples | ⬜ |
| Position balance | Each position has 180-220 samples | ⬜ |
| Condition minimums | All conditions >= 40 | ⬜ |
| Tier distribution | Within 10% of targets | ⬜ |
| Benchmark coverage | All 3 benchmarks represented | ⬜ |

---

## Step 9: Final Review and Mark Complete

### 9.1 Completion Checklist

| Item | Status |
|------|--------|
| Config updated with `quality_scoring` and `primary_selection` | ⬜ |
| Quality scorer implemented and tested | ⬜ |
| Primary selector implemented and tested | ⬜ |
| Both integrated into `_phase_generate_perturbations` | ⬜ |
| All perturbations scored | ⬜ |
| Primary sample selected (600) | ⬜ |
| Distribution meets quotas | ⬜ |
| Distribution report generated | ⬜ |
| Code merged to main | ⬜ |

### 9.2 Artifacts Produced

| Artifact | Location |
|----------|----------|
| Updated config schema | `config/experiments/*.json` |
| Quality scoring prompt | `src/scoring/prompts.py` |
| Quality scorer module | `src/scoring/quality_scorer.py` |
| Primary selector module | `src/sampling/primary_selector.py` |
| Test files | `tests/test_quality_scorer.py`, `tests/test_primary_selector.py` |
| Distribution report | `results/primary_selection/distribution_report.md` |
### 9.3 Database State

After running `--runner perturb` with both flags enabled:
- All perturbations have `quality_score` field
- 600 perturbations have `is_primary_for_experiment = True`
- All primary perturbations have `quality_tier` and `primary_selection_reason`

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| LLM quality scores unreliable | Medium | Medium | Review low-scoring samples manually |
| Quality scoring too slow | Low | Medium | Batch processing, async calls |
| Insufficient high-quality samples | Low | Medium | Include medium/low tiers |
| Quota cannot be filled for some conditions | Low | High | Relax minimum if needed, document |

---

## Pipeline Integration Requirements

### Requirement: Integrate Quality Scoring into `perturb` Runner

Quality scoring should happen **inline** during perturbation generation, not as a separate step.

**Current flow:**
```
python main.py --config config.json --runner perturb
  └── For each trajectory:
        └── For each condition:
              └── Generate perturbation
              └── Store to DB
```

**Required flow:**
```
python main.py --config config.json --runner perturb
  └── For each trajectory:
        └── For each condition:
              └── Generate perturbation
              └── IF quality_scoring.enabled:
                    └── Score perturbation quality (LLMaJ call)
                    └── Attach quality_score to perturbation
              └── Store to DB (with quality_score)
```

**Benefits:**
- Single pass (no re-querying DB)
- Quality score available immediately
- Checkpointing includes scores

---

## Config Schema Requirements

### Required Config Additions

The sub-agent must add a `quality_scoring` section to the experiment config:

```json
{
  "experiment": { ... },
  "datasets": { ... },
  "perturbations": { ... },
  
  "quality_scoring": {
    "enabled": true,
    "judge_model": {
      "name": "claude-sonnet-4-5",
      "provider": "aws_bedrock",
      "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
      "config": {
        "temperature": 0.3,
        "max_tokens": 500
      }
    },
    "metrics": {
      "content_changed": {
        "description": "Did the content actually change from original?",
        "score_range": [0, 1],
        "weight": 1.0
      },
      "syntactically_valid": {
        "description": "Is the perturbed content well-formed (not broken JSON, coherent text)?",
        "score_range": [0, 1],
        "weight": 1.0
      },
      "semantically_meaningful": {
        "description": "Is the change semantically meaningful (not just whitespace)?",
        "score_range": [0, 1],
        "weight": 1.0
      },
      "type_matches_intent": {
        "description": "Does the error match the intended perturbation type?",
        "score_range": [0, 1],
        "weight": 1.0
      },
      "realistic_error": {
        "description": "Could this error plausibly occur in a real agent execution?",
        "score_range": [0, 3],
        "weight": 1.0
      }
    },
    "total_score_range": [0, 7],
    "tier_thresholds": {
      "high": 6,
      "medium": 4,
      "low": 1,
      "invalid": 0
    },
    "batch_size": 10,
    "retry_on_failure": true,
    "max_retries": 2
  },

  "primary_selection": {
    "enabled": true,
    "total_primary": 600,
    "tier_distribution": {
      "high": 0.33,
      "medium": 0.50,
      "low": 0.17
    },
    "quota_strategy": "balanced_conditions",
    "min_per_condition": 40,
    "conditions": [
      {"type": "planning", "position": "early"},
      {"type": "planning", "position": "middle"},
      {"type": "planning", "position": "late"},
      {"type": "tool_selection", "position": "early"},
      {"type": "tool_selection", "position": "middle"},
      {"type": "tool_selection", "position": "late"},
      {"type": "parameter", "position": "early"},
      {"type": "parameter", "position": "middle"},
      {"type": "parameter", "position": "late"},
      {"type": "data_reference", "position": "middle"},
      {"type": "data_reference", "position": "late"}
    ],
    "random_seed": 42
  },

  "judges": { ... },
  "execution": { ... }
}
```

### Config Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `quality_scoring.enabled` | bool | Enable/disable quality scoring during perturbation |
| `quality_scoring.judge_model` | object | LLM model config for quality scoring |
| `quality_scoring.metrics` | object | Quality criteria with descriptions and score ranges |
| `quality_scoring.tier_thresholds` | object | Score boundaries for high/medium/low/invalid |
| `quality_scoring.batch_size` | int | Number of perturbations to score before DB write |
| `primary_selection.enabled` | bool | Enable primary sample selection after scoring |
| `primary_selection.total_primary` | int | Number of primary perturbations to select |
| `primary_selection.tier_distribution` | object | Target % for each quality tier |
| `primary_selection.min_per_condition` | int | Minimum samples per (type × position) condition |

### Runner Behavior

Single runner, config-driven behavior:

```bash
python main.py --config config.json --runner perturb
```

| Config Flag | Effect |
|-------------|--------|
| `quality_scoring.enabled = true` | Score each perturbation inline after generation |
| `primary_selection.enabled = true` | After all perturbations scored, select primaries |

**Flow:**
```
--runner perturb
  └── For each trajectory:
        └── For each condition:
              └── Generate perturbation
              └── IF quality_scoring.enabled: Score quality
              └── Store to DB
  └── IF primary_selection.enabled:
        └── Load all perturbations with scores
        └── Run stratified quota-based selection
        └── Mark selected as is_primary_for_experiment = True
```

**No separate runner needed.** All controlled by config flags.

---

## Dependencies

- Task 03 complete (perturbations generated)
- MongoDB populated with `exp_dataset_quality` perturbations
- LLM API access (model specified in `quality_scoring.judge_model`)

---

## My Help

If you need my help or if there is experiment failure and you can't fix it, trigger a notification.

**How to notify me:**
```bash
osascript -e 'display notification "Your message here" with title "Claude Code" sound name "default"'
```


