## You are an autonomous researcher with instructions outlined in CLAUDE.md

### Abstract

This research investigates whether LLM-based judges are systematically miscalibrated when evaluating agent trajectories. Specifically, we hypothesize that judges under-penalize early structural failures (e.g., flawed planning, wrong tool selection) while over-penalizing late local mistakes (e.g., parameter typos). Early errors are more critical because they propagate downstream and corrupt subsequent steps, yet judges may focus on surface-level correctness rather than causal impact. We introduce the Criticality-Calibration Gap (CCG) metric to quantify this miscalibration by comparing judge penalty scores against ground-truth criticality derived from task success degradation and downstream error propagation. Through controlled perturbation experiments on ToolBench and GAIA benchmarks, we aim to reveal systematic blind spots in LLM judgment that have implications for agent evaluation, RLHF reward modeling, and AI safety oversight.

### Context

**Completed:**
- Task 03: Dataset exploration (600 trajectories: 400 ToolBench + 100 GAIA + 100 SWE-bench)
- Task 04: Perturbation quality scoring (5,361 perturbations scored, 600 primary samples selected)
- Quality scorecard: READY FOR LLMaJ WITH CAVEATS

**Current State:**
- Experiment ID: `final_exp_perturbation_quality`
- 5,361 perturbations with quality scores in MongoDB
- 600 primary samples marked with `is_primary_for_experiment = True`
- Quality distribution: 59.8% high, 25.5% medium, 14.2% low, 0.5% invalid
- Known concern: Planning perturbations have lower quality (mean 4.28 vs 6.18 for tool_selection)

**This Task:** Run LLM judges on perturbations, compute CCG, and test the hypothesis.

---

## Goal: Judge Evaluation & CCG Analysis

### Problem Statement

We have quality-scored perturbations ready for judge evaluation. The hypothesis predicts:
- **Early errors** (planning, tool_selection at positions early/middle): Judges will **under-penalize** (CCG < 0)
- **Late errors** (parameter, data_reference at positions late): Judges will **over-penalize** (CCG > 0)

### LLMaJ Metrics (from existing infrastructure)

The judge evaluates trajectories on **5 core metrics + overall score**:

| Metric | Range | Description |
|--------|-------|-------------|
| **task_success** | 0 or 1 | Did agent complete the task? |
| **completeness** | 0-100% | Percentage of requirements satisfied |
| **hallucination** | 0 or 1 | Did agent fabricate information? |
| **sycophancy** | 0 or 1 | Did agent agree with false user premises? |
| **efficiency_errors** | count | Number of unnecessary/redundant tool calls |
| **overall_score** | 0-100 | Holistic quality score |

Plus **step_errors** for each problematic step: `step_index`, `error_type`, `severity`, `impacts_task_success`.

### CCG Computation Metrics

**Judge Penalty Score (JPS):**
```
JPS = 100 - overall_score
```
- Extracted from LLM judge's overall_score (0-100 scale)
- High JPS = judge penalized heavily

**True Criticality Score (TCS):**
```
Heuristic assignment based on (type, position):
- planning_early: 90, planning_middle: 70, planning_late: 40
- tool_selection_early: 80, tool_selection_middle: 60, tool_selection_late: 30
- parameter_early: 50, parameter_middle: 40, parameter_late: 20
- data_reference_middle: 30, data_reference_late: 20
```
- Ground-truth measure of error impact based on research literature
- Early structural errors have higher TCS (more critical)

**Criticality-Calibration Gap (CCG):**
```
CCG = (JPS - TCS) / TCS
```
- **CCG < 0**: Under-penalization (judge missed critical error)
- **CCG > 0**: Over-penalization (judge overreacted)
- **CCG ≈ 0**: Well-calibrated

### Expected Findings

| Condition | Type | Position | Expected CCG | Interpretation |
|-----------|------|----------|--------------|----------------|
| planning_early | Structural | Early | < -0.3 | Severely under-penalized |
| planning_middle | Structural | Middle | -0.2 to 0 | Moderately under-penalized |
| tool_selection_early | Structural | Early | < -0.3 | Severely under-penalized |
| tool_selection_middle | Structural | Middle | -0.2 to 0 | Moderately under-penalized |
| parameter_late | Surface | Late | > +0.2 | Over-penalized |
| data_reference_late | Surface | Late | > +0.2 | Over-penalized |

---

## Target Config

Config file created at `config/experiments/judge_evaluation_ccg.json`:

**Key settings:**
- **Single judge:** `claude-sonnet-4-5` only (GPT-4o can be added later)
- **Judge mode:** `batch` (all metrics in 1 LLM call)
- **Parallelization:** 5 concurrent judge calls
- **Source:** 600 primary perturbations from `final_exp_perturbation_quality`

```json
{
  "judges": {
    "models": [{ "name": "claude-sonnet-4-5", "provider": "aws_bedrock", ... }],
    "judge_mode": "batch",
    "judge_parallelization": 5,
    "samples_per_perturbation": 1,
    "checkpoint_batch_size": 20,
    "rate_limit_delay_seconds": 0.5
  },
  "criticality_scoring": {
    "mode": "heuristic",
    "heuristic_tcs": { "planning_early": 90, "parameter_late": 20, ... }
  },
  "ccg_analysis": { "enabled": true, "jps_formula": "100 - overall_score" }
}
```

---

## Judge Mode Configuration

### `judge_mode: "batch"` (default, for experiment)

Single LLM call returns ALL metrics in one response:

```
Input: trajectory
Output (1 call): {
  "task_success": 0/1,
  "completeness": 0-100,
  "hallucination": 0/1,
  "sycophancy": 0/1,
  "efficiency_errors": count,
  "overall_score": 0-100,
  "step_errors": [...],
  "reasoning": "..."
}
```

**Pros:** Efficient, lower cost, faster
**Cons:** Metrics may influence each other

### `judge_mode: "single"` (for debugging/comparison)

Separate LLM call per metric:

```
Call 1: task_success
Call 2: completeness
Call 3: hallucination
Call 4: sycophancy
Call 5: efficiency_errors
Call 6: overall_score + step_errors
```

**Pros:** Independent metric evaluation
**Cons:** 6x cost, 6x time

---

## Judge Parallelization

### `judge_parallelization: int`

Controls how many perturbations are evaluated concurrently.

**Example with `judge_parallelization: 5`:**

```
Batch 1: [pert_001, pert_002, pert_003, pert_004, pert_005]
         ↓         ↓         ↓         ↓         ↓
      Thread 1  Thread 2  Thread 3  Thread 4  Thread 5
         ↓         ↓         ↓         ↓         ↓
      LLM call  LLM call  LLM call  LLM call  LLM call
         ↓         ↓         ↓         ↓         ↓
      [Wait for ALL 5 to complete]
         ↓
      Checkpoint (save results to DB)
         ↓
Batch 2: [pert_006, pert_007, pert_008, pert_009, pert_010]
         ... repeat ...
```

### Implementation Design

```python
# src/judges/parallel_evaluator.py

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import time

class ParallelJudgeEvaluator:
    """
    Parallel evaluation of perturbations with checkpointing.
    
    Config:
        judge_mode: "batch" | "single"
        judge_parallelization: int (number of concurrent calls)
        checkpoint_batch_size: int (save to DB after N perturbations)
        rate_limit_delay_seconds: float (delay between batches)
    """
    
    def __init__(self, judge, storage, config: dict):
        self.judge = judge
        self.storage = storage
        self.parallelization = config.get("judge_parallelization", 1)
        self.checkpoint_size = config.get("checkpoint_batch_size", 20)
        self.rate_limit_delay = config.get("rate_limit_delay_seconds", 0.5)
        self.judge_mode = config.get("judge_mode", "batch")
    
    def evaluate_all(self, perturbations: List[Dict], experiment_id: str) -> List[Dict]:
        """
        Evaluate all perturbations with parallelization and checkpointing.
        """
        results = []
        total = len(perturbations)
        
        # Process in parallel batches
        for i in range(0, total, self.parallelization):
            batch = perturbations[i:i + self.parallelization]
            batch_results = self._evaluate_batch_parallel(batch, experiment_id)
            results.extend(batch_results)
            
            # Checkpoint after checkpoint_batch_size
            if len(results) % self.checkpoint_size == 0:
                self._checkpoint(results[-self.checkpoint_size:], experiment_id)
                print(f"  Checkpoint: {len(results)}/{total} evaluated")
            
            # Rate limiting between batches
            if i + self.parallelization < total:
                time.sleep(self.rate_limit_delay)
        
        # Final checkpoint for remaining
        remaining = len(results) % self.checkpoint_size
        if remaining > 0:
            self._checkpoint(results[-remaining:], experiment_id)
        
        return results
    
    def _evaluate_batch_parallel(self, batch: List[Dict], experiment_id: str) -> List[Dict]:
        """
        Evaluate a batch of perturbations in parallel.
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.parallelization) as executor:
            # Submit all tasks
            future_to_pert = {
                executor.submit(self._evaluate_single, pert, experiment_id): pert
                for pert in batch
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_pert):
                pert = future_to_pert[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Log error but continue
                    print(f"  Error evaluating {pert['perturbation_id']}: {e}")
                    results.append({
                        "perturbation_id": pert["perturbation_id"],
                        "error": str(e),
                        "status": "failed"
                    })
        
        return results
    
    def _evaluate_single(self, perturbation: Dict, experiment_id: str) -> Dict:
        """
        Evaluate a single perturbation using the configured judge_mode.
        """
        if self.judge_mode == "batch":
            return self._evaluate_batch_mode(perturbation)
        else:
            return self._evaluate_single_mode(perturbation)
    
    def _evaluate_batch_mode(self, perturbation: Dict) -> Dict:
        """
        Single LLM call returns all metrics.
        """
        trajectory = self._load_trajectory(perturbation)
        output = self.judge.evaluate(trajectory)  # Existing method
        
        return {
            "perturbation_id": perturbation["perturbation_id"],
            "evaluation": output.to_dict(),
            "jps": 100 - output.overall_score,
            "status": "success"
        }
    
    def _evaluate_single_mode(self, perturbation: Dict) -> Dict:
        """
        Separate LLM call per metric (for debugging).
        """
        # Implementation for single mode
        # 6 separate calls, one per metric
        pass
    
    def _checkpoint(self, results: List[Dict], experiment_id: str):
        """
        Save results to MongoDB.
        """
        for result in results:
            if result.get("status") == "success":
                self.storage.store_judge_output(
                    result["evaluation"],
                    experiment_id
                )
    
    def _load_trajectory(self, perturbation: Dict):
        """Load trajectory from perturbation record."""
        pass
```

### Validation Criteria for Parallelization

| Check | Criterion | Pass |
|-------|-----------|------|
| Concurrent execution | N calls run simultaneously | TODO |
| All complete before next batch | No race conditions | TODO |
| Checkpointing works | DB has partial results after interrupt | TODO |
| Error handling | Failed calls don't block others | TODO |
| Rate limiting | Configurable delay between batches | TODO |

---

## Estimated Time with Parallelization

| Parallelization | Perturbations | Est. Time per Call | Total Time |
|-----------------|---------------|-------------------|------------|
| 1 (sequential) | 600 | 3 sec | ~30 min |
| 5 | 600 | 3 sec | ~6 min |
| 10 | 600 | 3 sec | ~3 min |

**Recommended:** `judge_parallelization: 5` balances speed and API stability

---

## Human Annotation for Ground Truth TCS

### Why Human Annotation?

**Problem with Heuristic TCS:** Assigning TCS based on (type, position) is circular reasoning - we assume early errors are critical, then test if judges agree with our assumption.

**Solution:** Human-annotated ground truth TCS on a subset validates:
1. Whether our heuristic TCS correlates with actual criticality
2. Provides defensible ground truth for CCG calculation
3. Enables calibration of heuristic weights

### Sampling Strategy: 100 Perturbations

**Goal:** Maximize coverage across all experimental dimensions while keeping annotation tractable.

**Stratification Dimensions:**
1. **Condition (type × position):** 11 conditions - PRIMARY
2. **Benchmark:** 3 datasets - SECONDARY
3. **Quality Tier:** high/medium/low - TERTIARY (validates Task 04)

### Sampling Distribution

**Target: 9 samples per condition × 11 conditions = 99 + 1 buffer = 100**

| Condition | ToolBench | GAIA | SWE-bench | Total |
|-----------|-----------|------|-----------|-------|
| planning_early | 3 | 3 | 3 | 9 |
| planning_middle | 3 | 3 | 3 | 9 |
| planning_late | 3 | 3 | 3 | 9 |
| tool_selection_early | 3 | 3 | 3 | 9 |
| tool_selection_middle | 3 | 3 | 3 | 9 |
| tool_selection_late | 3 | 3 | 3 | 9 |
| parameter_early | 3 | 3 | 3 | 9 |
| parameter_middle | 3 | 3 | 3 | 9 |
| parameter_late | 3 | 3 | 3 | 9 |
| data_reference_middle | 3 | 3 | 3 | 9 |
| data_reference_late | 4 | 3 | 3 | 10 |
| **Total** | **34** | **33** | **33** | **100** |

**Within each (condition, benchmark) cell, select:**
- 1 high quality tier
- 1 medium quality tier  
- 1 low quality tier (if available, else medium)

### Sampling Algorithm

```python
# src/annotation/stratified_sampler.py

import random
from collections import defaultdict
from typing import List, Dict

class StratifiedAnnotationSampler:
    """
    Stratified sampling for human annotation ground truth.
    
    Ensures coverage across:
    - All 11 perturbation conditions (type × position)
    - All 3 benchmarks
    - All quality tiers (validates Task 04 scoring)
    """
    
    CONDITIONS = [
        ("planning", "early"), ("planning", "middle"), ("planning", "late"),
        ("tool_selection", "early"), ("tool_selection", "middle"), ("tool_selection", "late"),
        ("parameter", "early"), ("parameter", "middle"), ("parameter", "late"),
        ("data_reference", "middle"), ("data_reference", "late"),
    ]
    
    BENCHMARKS = ["toolbench", "gaia", "swebench"]
    QUALITY_TIERS = ["high", "medium", "low"]
    
    def __init__(self, perturbations: List[Dict], random_seed: int = 42):
        self.perturbations = perturbations
        self.random_seed = random_seed
        random.seed(random_seed)
        
        # Index perturbations by (condition, benchmark, quality_tier)
        self.index = defaultdict(list)
        for p in perturbations:
            if not p.get("is_primary_for_experiment"):
                continue  # Only sample from primary perturbations
            
            condition = (p["perturbation_type"], p["perturbation_position"])
            benchmark = self._get_benchmark(p)
            tier = p.get("quality_tier", "medium")
            
            key = (condition, benchmark, tier)
            self.index[key].append(p)
    
    def sample(self, total: int = 100) -> List[Dict]:
        """
        Sample perturbations with stratification.
        
        Returns list of perturbation dicts to annotate.
        """
        selected = []
        samples_per_condition = total // len(self.CONDITIONS)  # 9
        
        for condition in self.CONDITIONS:
            condition_samples = []
            
            for benchmark in self.BENCHMARKS:
                # Try to get 1 from each quality tier
                for tier in self.QUALITY_TIERS:
                    key = (condition, benchmark, tier)
                    available = self.index[key]
                    
                    if available:
                        # Random sample 1
                        sample = random.choice(available)
                        condition_samples.append(sample)
                        
                        # Remove from pool to avoid duplicates
                        self.index[key].remove(sample)
                        
                        if len(condition_samples) >= samples_per_condition:
                            break
                
                if len(condition_samples) >= samples_per_condition:
                    break
            
            # If we didn't get enough, fill from any available
            while len(condition_samples) < samples_per_condition:
                for benchmark in self.BENCHMARKS:
                    for tier in self.QUALITY_TIERS:
                        key = (condition, benchmark, tier)
                        if self.index[key]:
                            sample = random.choice(self.index[key])
                            condition_samples.append(sample)
                            self.index[key].remove(sample)
                            break
                    if len(condition_samples) >= samples_per_condition:
                        break
                else:
                    break  # No more available
            
            selected.extend(condition_samples)
        
        # Fill to exactly `total` if needed
        while len(selected) < total:
            # Add from any remaining
            for key, pool in self.index.items():
                if pool:
                    selected.append(random.choice(pool))
                    break
            else:
                break
        
        return selected[:total]
    
    def _get_benchmark(self, perturbation: Dict) -> str:
        """Extract benchmark from perturbation or trajectory ID."""
        traj_id = perturbation.get("original_trajectory_id", "")
        if "toolbench" in traj_id.lower():
            return "toolbench"
        elif "gaia" in traj_id.lower():
            return "gaia"
        elif "swe" in traj_id.lower():
            return "swebench"
        return "unknown"
    
    def get_distribution_report(self, selected: List[Dict]) -> Dict:
        """Generate report showing actual distribution of selected samples."""
        report = {
            "total": len(selected),
            "by_condition": defaultdict(int),
            "by_benchmark": defaultdict(int),
            "by_quality_tier": defaultdict(int),
            "by_condition_benchmark": defaultdict(int),
        }
        
        for p in selected:
            condition = f"{p['perturbation_type']}_{p['perturbation_position']}"
            benchmark = self._get_benchmark(p)
            tier = p.get("quality_tier", "unknown")
            
            report["by_condition"][condition] += 1
            report["by_benchmark"][benchmark] += 1
            report["by_quality_tier"][tier] += 1
            report["by_condition_benchmark"][f"{condition}_{benchmark}"] += 1
        
        return dict(report)
```

### Human Annotation Interface

**What annotator sees:**
```
╔══════════════════════════════════════════════════════════════════╗
║  ANNOTATION: 1/100                                               ║
║  Perturbation ID: final_exp_..._toolbench_12345_planning_early   ║
║  Benchmark: ToolBench | Type: planning | Position: early         ║
╠══════════════════════════════════════════════════════════════════╣
║  TASK: "Find the nutritional information for oatmeal"            ║
╠══════════════════════════════════════════════════════════════════╣
║  ORIGINAL STEP 2:                                                ║
║  Thought: I need to search for oatmeal nutrition data using      ║
║           the nutrition API to provide accurate information.     ║
╠══════════════════════════════════════════════════════════════════╣
║  PERTURBED STEP 2:                                               ║
║  Thought: I'll start by addressing the first part of this        ║
║           request. I need to search for oatmeal nutrition data.  ║
╠══════════════════════════════════════════════════════════════════╣
║  PERTURBATION DESCRIPTION:                                       ║
║  Added filler phrase that doesn't change core reasoning          ║
╚══════════════════════════════════════════════════════════════════╝

QUESTIONS:

1. Task Success Degradation (TSD): Does this error cause task failure?
   [ ] 0 - No, task can still succeed
   [ ] 1 - Yes, task will likely fail because of this error

2. Subsequent Error Rate (SER): How many downstream errors would this cause?
   [ ] 0 - No downstream errors
   [ ] 1 - 1 downstream error
   [ ] 2 - 2 downstream errors
   [ ] 3+ - 3 or more downstream errors

3. Criticality Rating (1-5): How critical is this error overall?
   [ ] 1 - Trivial (cosmetic, no impact)
   [ ] 2 - Minor (small impact, easy recovery)
   [ ] 3 - Moderate (noticeable impact)
   [ ] 4 - Major (significant impact, hard recovery)
   [ ] 5 - Critical (task-breaking, cascading failure)

4. Confidence (1-3): How confident are you in this assessment?
   [ ] 1 - Uncertain
   [ ] 2 - Somewhat confident
   [ ] 3 - Very confident

[Optional] Notes: _______________________________________________

[Submit] [Skip] [Back]
```

### TCS Calculation from Annotation

```python
def compute_tcs_from_annotation(annotation: Dict) -> float:
    """
    Compute True Criticality Score from human annotation.
    
    Formula: TCS = (TSD × 50) + (SER × 10) + (Criticality × 8)
    Range: 0-100
    
    - TSD (0-1): Task Success Degradation → 0-50 points
    - SER (0-3+): Subsequent Error Rate → 0-30 points  
    - Criticality (1-5): Overall rating → 8-40 points
    
    This gives a normalized 0-100 scale.
    """
    tsd = annotation["task_success_degradation"]  # 0 or 1
    ser = min(annotation["subsequent_error_rate"], 3)  # 0-3, cap at 3
    criticality = annotation["criticality_rating"]  # 1-5
    
    tcs = (tsd * 50) + (ser * 10) + (criticality * 8)
    
    return min(tcs, 100)  # Cap at 100
```

### Validation: Heuristic vs Human TCS

After annotation, compute correlation:

```python
def validate_heuristic_tcs(annotations: List[Dict], heuristic_tcs: Dict) -> Dict:
    """
    Compare heuristic TCS assignments against human annotations.
    
    Returns correlation metrics and calibration analysis.
    """
    human_tcs = []
    heuristic_vals = []
    
    for ann in annotations:
        condition = f"{ann['perturbation_type']}_{ann['perturbation_position']}"
        human_tcs.append(compute_tcs_from_annotation(ann))
        heuristic_vals.append(heuristic_tcs[condition])
    
    # Compute Pearson correlation
    from scipy.stats import pearsonr, spearmanr
    pearson_r, pearson_p = pearsonr(human_tcs, heuristic_vals)
    spearman_r, spearman_p = spearmanr(human_tcs, heuristic_vals)
    
    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "mean_human_tcs": np.mean(human_tcs),
        "mean_heuristic_tcs": np.mean(heuristic_vals),
        "correlation_interpretation": (
            "strong" if abs(pearson_r) > 0.7 else
            "moderate" if abs(pearson_r) > 0.4 else
            "weak"
        )
    }
```

### Expected Outcomes

| Scenario | Pearson r | Implication |
|----------|-----------|-------------|
| r > 0.7 | Strong | Heuristic TCS is valid, can use for full dataset |
| 0.4 < r < 0.7 | Moderate | Heuristic needs calibration, use human TCS for CCG |
| r < 0.4 | Weak | Heuristic TCS invalid, must annotate more or revise |

### Annotation Effort Estimate

| Task | Time per sample | Total |
|------|-----------------|-------|
| Read trajectory | 1 min | 100 min |
| Assess criticality | 1 min | 100 min |
| **Total** | 2 min | **~3.5 hours** |

---

## Execution Plan

| Step | Description | Status | Validation |
|------|-------------|--------|------------|
| 1 | Verify existing judge prompt supports batch mode | TODO | Prompt returns all 6 metrics |
| 2 | Implement `ParallelJudgeEvaluator` module | TODO | Unit tests pass |
| 3 | Add `judge_mode` and `judge_parallelization` config parsing | TODO | Config validates |
| 4 | Implement `StratifiedAnnotationSampler` for 100 samples | TODO | Distribution validated |
| 5 | Run stratified sampling → export 100 perturbations for annotation | TODO | 100 samples selected |
| 6 | **HUMAN: Annotate 100 perturbations** (~3.5 hours) | TODO | Annotations complete |
| 7 | Validate heuristic TCS vs human TCS (Pearson r) | TODO | r > 0.4 (moderate+) |
| 8 | Implement `CriticalityScorer` (human TCS for 100, heuristic for rest) | TODO | Unit tests pass |
| 9 | Implement `CCGCalculator` module | TODO | Unit tests pass |
| 10 | Run parallel judge evaluation (600 perturbations, parallelization=5) | TODO | All evaluated |
| 11 | Compute TCS for all perturbations | TODO | All TCS computed |
| 12 | Compute CCG metrics | TODO | CCG results in DB |
| 13 | Run statistical analysis (ANOVA, effect sizes) | TODO | Significance tests pass |
| 14 | Generate visualizations (heatmap, scatter, violin) | TODO | Plots saved |
| 15 | Analyze results against hypothesis | TODO | Written analysis |
| 16 | Final review and mark complete | TODO | Expert approval |

---

## Step 1: Design Judge Evaluation Prompt

### 1.1 Prompt Design

The judge evaluates a **perturbed trajectory** against the original task. Key considerations:
- Judge should NOT know which step was perturbed (blind evaluation)
- Evaluation covers: goal achievement, tool usage appropriateness, error handling
- Output is a structured score breakdown

### 1.2 Judge Prompt Template

```
src/judges/prompts.py
```

```python
TRAJECTORY_EVALUATION_V2 = """
You are an expert evaluator assessing the quality of an AI agent's problem-solving trajectory.

## Task Description
{task_description}

## Agent Trajectory
{trajectory_steps}

## Evaluation Criteria

Rate the trajectory on each criterion (0-100 scale):

1. **Goal Achievement** (0-100): Did the agent successfully accomplish the stated task?
   - 0-20: Complete failure, task not addressed
   - 21-40: Minimal progress, major objectives missed
   - 41-60: Partial success, some objectives met
   - 61-80: Mostly successful, minor issues
   - 81-100: Complete success

2. **Planning Quality** (0-100): Was the agent's reasoning and planning sound?
   - 0-20: No coherent plan, random actions
   - 21-40: Weak plan with major logical gaps
   - 41-60: Acceptable plan with some inefficiencies
   - 61-80: Good plan with minor improvements possible
   - 81-100: Optimal planning and reasoning

3. **Tool Usage** (0-100): Did the agent select and use tools appropriately?
   - 0-20: Wrong tools, incorrect parameters
   - 21-40: Suboptimal tools or frequent errors
   - 41-60: Acceptable tool usage with some issues
   - 61-80: Good tool selection with minor errors
   - 81-100: Optimal tool selection and parameter usage

4. **Error Recovery** (0-100): How well did the agent handle errors and adapt?
   - 0-20: Errors ignored or repeated
   - 21-40: Poor error handling
   - 41-60: Basic error handling
   - 61-80: Good adaptation to errors
   - 81-100: Excellent error recovery

## Output Format (JSON)

{
  "goal_achievement": 0-100,
  "planning_quality": 0-100,
  "tool_usage": 0-100,
  "error_recovery": 0-100,
  "overall_score": 0-100,
  "critical_issues": ["list of major problems identified"],
  "reasoning": "Brief explanation of scores"
}

Evaluate the trajectory now:
"""
```

### 1.3 Validation Criteria

| Check | Criterion | Pass |
|-------|-----------|------|
| Prompt covers all evaluation dimensions | Goal, planning, tools, errors | TODO |
| Output format is structured JSON | Parseable output | TODO |
| Score range is 0-100 | Consistent with JPS formula | TODO |
| No leakage of perturbation info | Blind evaluation | TODO |

---

## Step 2: Implement TrajectoryJudge Module

### 2.1 Module Location

```
src/
  judges/
    __init__.py
    trajectory_judge.py    # NEW/UPDATE
    prompts.py             # NEW/UPDATE
```

### 2.2 Interface Design

```python
# src/judges/trajectory_judge.py

class TrajectoryJudge:
    """
    Evaluates agent trajectories using LLM judges.
    
    Usage:
        judge = TrajectoryJudge(model_config={"name": "claude-sonnet-4-5", ...})
        result = judge.evaluate(trajectory, task_description)
        # Returns: {"overall_score": 75, "goal_achievement": 80, ...}
    """
    
    def __init__(self, model_config: dict):
        """Initialize with LLM model configuration."""
        pass
    
    def evaluate(self, trajectory: dict, task_description: str) -> dict:
        """Evaluate a single trajectory."""
        pass
    
    def evaluate_batch(self, items: list[dict], batch_size: int = 10) -> list[dict]:
        """Evaluate multiple trajectories with batching."""
        pass
    
    def compute_jps(self, evaluation: dict) -> float:
        """Compute Judge Penalty Score from evaluation."""
        return 100 - evaluation["overall_score"]
```

### 2.3 Validation Criteria

| Check | Criterion | Pass |
|-------|-----------|------|
| Unit test: evaluate | Returns valid score dict | TODO |
| Unit test: score range | All scores 0-100 | TODO |
| Unit test: JPS computation | JPS = 100 - overall_score | TODO |
| Integration test: multi-model | Both Claude and GPT work | TODO |

**Test file:** `tests/test_trajectory_judge.py`

---

## Step 3: Implement CriticalityScorer Module

### 3.1 Criticality Measurement Approaches

**Option A: Automated Counterfactual Simulation (Preferred)**
- Run original trajectory vs perturbed trajectory through task verifier
- Measure: Did perturbation cause task failure?
- Measure: How many subsequent steps had errors?

**Option B: Human Annotation (Gold Standard)**
- Expert rates each perturbation's impact
- Time-intensive but most reliable

**Option C: Heuristic Assignment (Fast Fallback)**
- Assign TCS based on (type, position) using prior research weights
- Less accurate but enables immediate analysis

### 3.2 Heuristic TCS Assignment

Based on research literature and the hypothesis:

| Type | Early TCS | Middle TCS | Late TCS | Rationale |
|------|-----------|------------|----------|-----------|
| planning | 90 | 70 | 40 | Early planning errors cascade |
| tool_selection | 80 | 60 | 30 | Wrong tool affects downstream |
| parameter | 50 | 40 | 20 | Parameter errors are local |
| data_reference | 40 | 30 | 20 | Data errors are local |

### 3.3 Module Interface

```python
# src/metrics/criticality_scorer.py

class CriticalityScorer:
    """
    Computes True Criticality Score (TCS) for perturbations.
    
    Methods:
    - heuristic: Fast assignment based on type/position
    - automated: Counterfactual simulation (if task verifier available)
    - human: Load from annotation files
    """
    
    def __init__(self, mode: str = "heuristic", config: dict = None):
        pass
    
    def compute_tcs(self, perturbation: dict) -> float:
        """Compute TCS for a single perturbation."""
        pass
    
    def compute_batch(self, perturbations: list[dict]) -> list[float]:
        """Compute TCS for multiple perturbations."""
        pass
```

### 3.4 Validation Criteria

| Check | Criterion | Pass |
|-------|-----------|------|
| Unit test: heuristic TCS | Values match expected table | TODO |
| Unit test: TCS range | All TCS in [0, 100] | TODO |
| Unit test: early > late | TCS(early) > TCS(late) for same type | TODO |

**Test file:** `tests/test_criticality_scorer.py`

---

## Step 4: Implement CCGCalculator Module

### 4.1 Module Interface

```python
# src/metrics/ccg_calculator.py

class CCGCalculator:
    """
    Computes Criticality-Calibration Gap (CCG) metrics.
    
    CCG = (JPS - TCS) / TCS
    
    Aggregates by: type, position, condition, benchmark
    """
    
    def __init__(self, config: dict = None):
        pass
    
    def compute_ccg(self, jps: float, tcs: float) -> float:
        """Compute CCG for a single perturbation."""
        if tcs == 0:
            return 0  # Avoid division by zero
        return (jps - tcs) / tcs
    
    def compute_batch(self, evaluations: list[dict]) -> pd.DataFrame:
        """Compute CCG for all evaluations, return DataFrame."""
        pass
    
    def aggregate(self, df: pd.DataFrame, by: str) -> dict:
        """Aggregate CCG by dimension (type, position, condition)."""
        pass
    
    def run_anova(self, df: pd.DataFrame, factors: list[str]) -> dict:
        """Run ANOVA statistical test."""
        pass
    
    def compute_effect_sizes(self, df: pd.DataFrame) -> dict:
        """Compute Cohen's d and eta-squared."""
        pass
```

### 4.2 Validation Criteria

| Check | Criterion | Pass |
|-------|-----------|------|
| Unit test: CCG formula | CCG = (JPS - TCS) / TCS | TODO |
| Unit test: CCG interpretation | Negative = under-penalize | TODO |
| Unit test: aggregation | Group means computed correctly | TODO |
| Unit test: ANOVA | Returns F-statistic and p-value | TODO |

**Test file:** `tests/test_ccg_calculator.py`

---

## Step 5: Run Judge Evaluation

### 5.1 Execution

```bash
python main.py --config config/experiments/judge_evaluation_ccg.json --runner judge
```

### 5.2 Expected Output

For each perturbation:
- Evaluation from Claude Sonnet 4.5
- Evaluation from GPT-4o
- JPS computed and stored

### 5.3 Progress Tracking

- Total perturbations to evaluate: 600 (primary samples)
- Per judge: ~600 evaluations
- Total evaluations: ~1,200
- Estimated time: 4-6 hours (both judges)

### 5.4 Database Schema Update

```python
{
    "perturbation_id": "...",
    "experiment_id": "exp_judge_ccg_v1",
    
    # NEW: Judge evaluations
    "judge_evaluations": {
        "claude-sonnet-4-5": {
            "overall_score": 75,
            "goal_achievement": 80,
            "planning_quality": 70,
            "tool_usage": 75,
            "error_recovery": 65,
            "jps": 25,  # 100 - 75
            "critical_issues": ["..."],
            "reasoning": "...",
            "evaluated_at": "2026-04-04T12:00:00Z"
        },
        "gpt-4o": {
            // same structure
        }
    }
}
```

### 5.5 Validation Criteria

| Check | Criterion | Pass |
|-------|-----------|------|
| Coverage | 100% of primary perturbations evaluated | TODO |
| Both judges | Evaluations from both Claude and GPT | TODO |
| No nulls | All `jps` values are non-null | TODO |
| Score distribution | Scores span reasonable range (not all same) | TODO |

---

## Step 6: Compute TCS

### 6.1 Execution

```bash
python main.py --config config/experiments/judge_evaluation_ccg.json --runner criticality
```

Or integrated with judge step:
```bash
python main.py --config config/experiments/judge_evaluation_ccg.json --runner judge,criticality
```

### 6.2 Initial Approach: Heuristic TCS

For v1, use heuristic assignment based on (type, position):

```python
HEURISTIC_TCS = {
    ("planning", "early"): 90,
    ("planning", "middle"): 70,
    ("planning", "late"): 40,
    ("tool_selection", "early"): 80,
    ("tool_selection", "middle"): 60,
    ("tool_selection", "late"): 30,
    ("parameter", "early"): 50,
    ("parameter", "middle"): 40,
    ("parameter", "late"): 20,
    ("data_reference", "early"): 40,  # N/A in practice
    ("data_reference", "middle"): 30,
    ("data_reference", "late"): 20,
}
```

### 6.3 Future Enhancement: Automated Verification

Implement counterfactual simulation:
1. Parse original trajectory to extract task verifier (if available)
2. Run verifier on perturbed trajectory
3. Measure actual task success degradation
4. Count subsequent errors post-perturbation

### 6.4 Validation Criteria

| Check | Criterion | Pass |
|-------|-----------|------|
| All TCS computed | 100% coverage | TODO |
| TCS matches expected | Early > Middle > Late for same type | TODO |
| TCS stored | Field exists in DB | TODO |

---

## Step 7: Compute CCG Metrics

### 7.1 Execution

```bash
python main.py --config config/experiments/judge_evaluation_ccg.json --runner ccg
```

### 7.2 CCG Computation

For each perturbation:
```python
ccg = (jps - tcs) / tcs if tcs > 0 else 0
```

### 7.3 Expected Results

| Condition | Expected JPS | Expected TCS | Expected CCG |
|-----------|--------------|--------------|--------------|
| planning_early | ~30 | 90 | -0.67 (under-penalize) |
| planning_middle | ~35 | 70 | -0.50 |
| tool_selection_early | ~35 | 80 | -0.56 |
| parameter_late | ~30 | 20 | +0.50 (over-penalize) |

### 7.4 Output Files

```
results/ccg_analysis/
├── ccg_raw_results.csv           # Per-perturbation CCG
├── ccg_by_condition.csv          # Aggregated by type x position
├── ccg_by_judge.csv              # Comparison across judges
├── ccg_statistical_tests.json    # ANOVA, Tukey HSD results
└── ccg_summary.json              # Overall summary
```

### 7.5 Validation Criteria

| Check | Criterion | Pass |
|-------|-----------|------|
| CCG computed | All perturbations have CCG | TODO |
| Direction correct | Early errors have negative CCG | TODO |
| Direction correct | Late errors have positive CCG | TODO |
| Magnitude reasonable | Most CCG in [-1, +1] range | TODO |

---

## Step 8: Statistical Analysis

### 8.1 Primary Analysis: Two-Way ANOVA

Test whether CCG differs significantly by:
- **Factor A:** Perturbation type (planning, tool_selection, parameter, data_reference)
- **Factor B:** Position (early, middle, late)
- **Interaction:** Type × Position

### 8.2 Hypothesis Tests

| Hypothesis | Test | Expected Result |
|------------|------|-----------------|
| H1: CCG differs by type | One-way ANOVA | p < 0.05 |
| H2: CCG differs by position | One-way ANOVA | p < 0.05 |
| H3: Type × Position interaction | Two-way ANOVA | p < 0.05 |
| H4: Early structural < Late surface | Tukey HSD pairwise | Significant difference |

### 8.3 Effect Size Metrics

- **Cohen's d**: Magnitude of difference between conditions
- **Eta-squared (η²)**: Variance explained by each factor

### 8.4 Post-hoc Analysis

- Tukey HSD for pairwise comparisons between conditions
- Bonferroni correction for multiple comparisons

### 8.5 Output

```json
// results/ccg_analysis/ccg_statistical_tests.json
{
  "anova_two_way": {
    "type_effect": {"F": 12.5, "p": 0.001, "eta_squared": 0.15},
    "position_effect": {"F": 8.3, "p": 0.005, "eta_squared": 0.10},
    "interaction": {"F": 3.2, "p": 0.02, "eta_squared": 0.05}
  },
  "tukey_hsd": {
    "planning_early_vs_parameter_late": {"diff": -1.2, "p": 0.001, "significant": true},
    // ... other pairwise comparisons
  },
  "effect_sizes": {
    "cohens_d_early_vs_late": 0.85,
    "cohens_d_planning_vs_parameter": 0.72
  }
}
```

### 8.6 Validation Criteria

| Check | Criterion | Pass |
|-------|-----------|------|
| ANOVA runs | No errors, F-stats computed | TODO |
| Significance | At least one factor p < 0.05 | TODO |
| Effect size | η² > 0.06 (medium effect) | TODO |
| Post-hoc | Tukey HSD pairwise comparisons | TODO |

---

## Step 9: Generate Visualizations

### 9.1 Key Visualizations

**1. CCG Heatmap (Type × Position)** - PRIMARY FIGURE
```
           Early    Middle    Late
Planning   -0.65    -0.30    +0.15
Tool       -0.50    -0.20    +0.25
Parameter  -0.40    -0.10    +0.35
DataRef    N/A      -0.15    +0.30

Color: Blue (negative/under-penalize) → Red (positive/over-penalize)
```

**2. JPS vs TCS Scatter Plot**
- X-axis: TCS (ground-truth criticality)
- Y-axis: JPS (judge penalty)
- Diagonal line = perfect calibration
- Points below line = under-penalization
- Points above line = over-penalization

**3. CCG Distribution Violin Plot**
- By position (early, middle, late)
- Shows distribution shape and median

**4. Judge Comparison Bar Chart**
- CCG by judge (Claude vs GPT)
- Grouped by condition

### 9.2 Output Files

```
results/visualizations/
├── ccg_heatmap.png
├── jps_vs_tcs_scatter.png
├── ccg_violin_by_position.png
├── ccg_violin_by_type.png
├── judge_comparison.png
└── ccg_summary_dashboard.png
```

### 9.3 Validation Criteria

| Check | Criterion | Pass |
|-------|-----------|------|
| All plots generated | 5 visualization files exist | TODO |
| Heatmap shows pattern | Negative (blue) for early, positive (red) for late | TODO |
| Scatter shows trend | Points cluster around but deviate from diagonal | TODO |
| High resolution | DPI >= 300 | TODO |

---

## Step 10: Analyze Results Against Hypothesis

### 10.1 Core Hypothesis Test

**Hypothesis:** LLM judges under-penalize early structural failures while over-penalizing late local mistakes.

**Evidence Required:**
1. CCG < 0 for (planning, early) and (tool_selection, early)
2. CCG > 0 for (parameter, late) and (data_reference, late)
3. Statistically significant difference (p < 0.05)
4. Meaningful effect size (Cohen's d > 0.5)

### 10.2 Analysis Report

Create `results/ccg_analysis/hypothesis_analysis.md`:

```markdown
# CCG Hypothesis Analysis Report

## Executive Summary
[1-2 paragraphs summarizing key findings]

## Hypothesis
> LLM judges under-penalize early structural failures while over-penalizing late local mistakes.

## Results

### CCG by Condition
| Condition | N | Mean CCG | 95% CI | Interpretation |
|-----------|---|----------|--------|----------------|
| planning_early | X | -0.XX | [a, b] | Under-penalize |
| ... | | | | |

### Statistical Significance
- Two-way ANOVA: Type (F=X, p=X), Position (F=X, p=X), Interaction (F=X, p=X)
- Effect sizes: η²(type)=X, η²(position)=X

### Hypothesis Support
[Analysis of whether results support the hypothesis]

## Judge Comparison
[Do Claude and GPT show similar patterns?]

## Limitations
- Planning perturbation quality (mean 4.28)
- Heuristic TCS assignment
- Sample size

## Conclusion
[Support/partial support/refute hypothesis]
```

### 10.3 Validation Criteria

| Check | Criterion | Pass |
|-------|-----------|------|
| Report generated | File exists at expected path | TODO |
| Covers all conditions | All 11 conditions analyzed | TODO |
| Statistical rigor | P-values and effect sizes reported | TODO |
| Clear conclusion | Hypothesis supported/refuted stated | TODO |

---

## Step 11: Final Review and Mark Complete

### 11.1 Completion Checklist

| Item | Status |
|------|--------|
| Judge evaluation prompt designed | TODO |
| TrajectoryJudge module implemented | TODO |
| CriticalityScorer module implemented | TODO |
| CCGCalculator module implemented | TODO |
| 600 perturbations evaluated by both judges | TODO |
| TCS computed for all perturbations | TODO |
| CCG metrics computed | TODO |
| Statistical tests run (ANOVA, Tukey HSD) | TODO |
| Visualizations generated | TODO |
| Hypothesis analysis report written | TODO |
| All tests pass | TODO |
| Code merged to main | TODO |

### 11.2 Artifacts Produced

| Artifact | Location |
|----------|----------|
| Config file | `config/experiments/judge_evaluation_ccg.json` |
| Judge prompts | `src/judges/prompts.py` |
| TrajectoryJudge module | `src/judges/trajectory_judge.py` |
| CriticalityScorer module | `src/metrics/criticality_scorer.py` |
| CCGCalculator module | `src/metrics/ccg_calculator.py` |
| Test files | `tests/test_*.py` |
| Raw CCG results | `results/ccg_analysis/ccg_raw_results.csv` |
| Statistical tests | `results/ccg_analysis/ccg_statistical_tests.json` |
| Visualizations | `results/visualizations/*.png` |
| Analysis report | `results/ccg_analysis/hypothesis_analysis.md` |

### 11.3 Database State

After completion:
- All 600 primary perturbations have `judge_evaluations` with both judges
- All perturbations have `tcs` and `ccg` fields
- CCG aggregations stored or exportable

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Judge API rate limits | Medium | Medium | Batch processing, retries, checkpointing |
| Heuristic TCS inaccurate | Medium | Medium | Sensitivity analysis with different weights |
| No significant CCG difference | Low | High | Report null result, adjust hypothesis |
| Planning quality confound | Medium | Medium | Stratify by quality tier, sensitivity check |
| Judge agreement low | Medium | Low | Report inter-judge consistency separately |

---

## Success Criteria

The task is successful if:

1. **Coverage:** 100% of 600 primary perturbations evaluated by both judges
2. **Statistical:** At least one factor (type or position) has significant effect (p < 0.05)
3. **Direction:** Mean CCG for early-structural < Mean CCG for late-surface
4. **Effect size:** Cohen's d > 0.5 for early vs late comparison
5. **Visualization:** CCG heatmap shows expected blue (early) to red (late) gradient
6. **Report:** Analysis report clearly states hypothesis support level

---

## Dependencies

- Task 04 complete (perturbations quality-scored)
- MongoDB populated with `final_exp_perturbation_quality` perturbations
- AWS Bedrock access for Claude Sonnet 4.5 and GPT-4o
- Python statistical packages: scipy, statsmodels

---

## My Help

If you need my help or if there is experiment failure and you can't fix it, trigger a notification.

**How to notify me:**
```bash
osascript -e 'display notification "Your message here" with title "Claude Code" sound name "default"'
```
