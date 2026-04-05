# Implementation Prompt: Task 05 - Judge Evaluation & CCG Analysis

## You are an implementation agent. Your goal is to implement the Judge Evaluation & CCG Analysis pipeline.

---

## Context

**Research Goal:** Test whether LLM judges are miscalibrated when evaluating agent trajectories - specifically, do they under-penalize early structural failures while over-penalizing late local mistakes?

**What exists:**
- 600 primary perturbations in MongoDB (`final_exp_perturbation_quality`)
- Each perturbation has: `perturbation_type`, `perturbation_position`, `quality_score`, `quality_tier`
- Existing judge infrastructure in `src/judges/` (claude_judge.py, prompts.py, schema.py, evaluator.py)
- Config at `config/experiments/judge_evaluation_ccg.json`

**What you need to build:**
1. Parallel judge evaluator with configurable parallelization
2. Stratified annotation sampler for 100 human annotation samples
3. Criticality scorer (hybrid: human TCS + heuristic TCS)
4. CCG calculator with statistical analysis
5. Integration into experiment runner

---

## Config Reference

Read the full config at: `config/experiments/judge_evaluation_ccg.json`

**Key config sections:**

```json
{
  "judges": {
    "judge_parallelization": 5,      // Run 5 concurrent LLM calls
    "checkpoint_batch_size": 20      // Save to DB every 20 perturbations
  },
  
  "judge_metrics": {
    "judge_mode": "batch",           // All metrics in 1 LLM call
    "metrics": {
      "task_success": { "score_range": [0, 1], "weight": 1.0 },
      "completeness": { "score_range": [0, 100], "weight": 1.0 },
      "hallucination": { "score_range": [0, 1], "weight": 1.0 },
      "sycophancy": { "score_range": [0, 1], "weight": 1.0 },
      "efficiency_errors": { "score_range": [0, null], "weight": 1.0 },
      "overall_score": { "score_range": [0, 100], "used_for_jps": true }
    },
    "jps_formula": "100 - overall_score"
  },
  
  "human_annotation": {
    "total_samples": 100,
    "sampling_strategy": "stratified"  // By condition > benchmark > quality_tier
  },
  
  "criticality_scoring": {
    "mode": "hybrid",                  // Human TCS for 100, heuristic for 500
    "metrics": {
      "task_success_degradation": { "score_range": [0, 1], "weight": 50.0 },
      "subsequent_error_rate": { "score_range": [0, 3], "weight": 10.0 },
      "criticality_rating": { "score_range": [1, 5], "weight": 8.0 }
    },
    "tcs_formula": "(TSD * 50) + (SER * 10) + (criticality * 8)"
  }
}
```

---

## Implementation Steps

### Step 1: Implement ParallelJudgeEvaluator

**File:** `src/judges/parallel_evaluator.py`

**Requirements:**
- Use `concurrent.futures.ThreadPoolExecutor` for parallelization
- Process N perturbations concurrently (N = `judge_parallelization` from config)
- Wait for all N to complete before starting next batch
- Checkpoint to MongoDB every `checkpoint_batch_size` perturbations
- Handle errors gracefully (log and continue, don't block batch)
- Support resume from checkpoint (skip already-evaluated perturbations)

**Interface:**
```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import time

class ParallelJudgeEvaluator:
    """
    Parallel evaluation of perturbations with checkpointing.
    """
    
    def __init__(
        self,
        judge,                    # Judge instance (from src/judges/)
        storage,                  # MongoDBStorage instance
        config: Dict[str, Any]    # Config dict with judge settings
    ):
        self.judge = judge
        self.storage = storage
        self.parallelization = config.get("judge_parallelization", 1)
        self.checkpoint_size = config.get("checkpoint_batch_size", 20)
        self.rate_limit_delay = config.get("rate_limit_delay_seconds", 0.5)
    
    def evaluate_all(
        self,
        perturbations: List[Dict],
        experiment_id: str,
        resume: bool = True
    ) -> List[Dict]:
        """
        Evaluate all perturbations with parallelization.
        
        Args:
            perturbations: List of perturbation dicts from MongoDB
            experiment_id: Experiment ID for storing results
            resume: If True, skip already-evaluated perturbations
        
        Returns:
            List of evaluation results
        """
        # Filter already evaluated if resuming
        if resume:
            perturbations = self._filter_evaluated(perturbations, experiment_id)
        
        results = []
        total = len(perturbations)
        
        # Process in parallel batches
        for i in range(0, total, self.parallelization):
            batch = perturbations[i:i + self.parallelization]
            batch_num = i // self.parallelization + 1
            total_batches = (total + self.parallelization - 1) // self.parallelization
            
            print(f"  Processing batch {batch_num}/{total_batches} "
                  f"({len(batch)} perturbations)...")
            
            batch_results = self._evaluate_batch_parallel(batch, experiment_id)
            results.extend(batch_results)
            
            # Checkpoint periodically
            if len(results) % self.checkpoint_size == 0:
                self._checkpoint(results[-self.checkpoint_size:], experiment_id)
                print(f"  Checkpoint: {len(results)}/{total} evaluated")
            
            # Rate limiting between batches
            if i + self.parallelization < total:
                time.sleep(self.rate_limit_delay)
        
        # Final checkpoint
        remaining = len(results) % self.checkpoint_size
        if remaining > 0:
            self._checkpoint(results[-remaining:], experiment_id)
        
        return results
    
    def _evaluate_batch_parallel(
        self,
        batch: List[Dict],
        experiment_id: str
    ) -> List[Dict]:
        """Evaluate batch using ThreadPoolExecutor."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.parallelization) as executor:
            future_to_pert = {
                executor.submit(self._evaluate_single, pert, experiment_id): pert
                for pert in batch
            }
            
            for future in as_completed(future_to_pert):
                pert = future_to_pert[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"    Error evaluating {pert['perturbation_id']}: {e}")
                    results.append({
                        "perturbation_id": pert["perturbation_id"],
                        "status": "failed",
                        "error": str(e)
                    })
        
        return results
    
    def _evaluate_single(self, perturbation: Dict, experiment_id: str) -> Dict:
        """Evaluate single perturbation."""
        # Load trajectory from perturbation
        trajectory = self._load_trajectory(perturbation)
        
        # Call judge
        output = self.judge.evaluate(trajectory)
        
        # Compute JPS
        jps = 100 - output.overall_score
        
        return {
            "perturbation_id": perturbation["perturbation_id"],
            "trajectory_id": perturbation["perturbed_trajectory_id"],
            "evaluation": output.to_dict(),
            "jps": jps,
            "status": "success"
        }
    
    def _filter_evaluated(self, perturbations: List[Dict], experiment_id: str) -> List[Dict]:
        """Filter out already-evaluated perturbations."""
        # Query MongoDB for existing evaluations
        # Return only perturbations without evaluations
        pass
    
    def _checkpoint(self, results: List[Dict], experiment_id: str):
        """Save results to MongoDB."""
        for result in results:
            if result.get("status") == "success":
                self.storage.store_judge_output(
                    result["evaluation"],
                    experiment_id
                )
    
    def _load_trajectory(self, perturbation: Dict):
        """Load trajectory object from perturbation record."""
        # Get perturbed_trajectory_id from perturbation
        # Load from MongoDB
        # Convert to Trajectory object
        pass
```

**Test file:** `tests/test_parallel_evaluator.py`

```python
def test_parallelization_runs_concurrent():
    """Verify N evaluations run concurrently."""
    pass

def test_checkpoint_saves_periodically():
    """Verify results saved every checkpoint_batch_size."""
    pass

def test_resume_skips_evaluated():
    """Verify already-evaluated perturbations are skipped."""
    pass

def test_error_handling_continues():
    """Verify failed evaluation doesn't block batch."""
    pass
```

---

### Step 2: Implement StratifiedAnnotationSampler

**File:** `src/annotation/stratified_sampler.py`

**Requirements:**
- Sample 100 perturbations from 600 primary samples
- Stratify by: condition (primary) > benchmark (secondary) > quality_tier (tertiary)
- 9 samples per condition × 11 conditions = 99 + 1 buffer
- Within each (condition, benchmark), try to get 1 high, 1 medium, 1 low quality
- Export selected samples to JSON for annotation interface

**Interface:**
```python
from collections import defaultdict
from typing import List, Dict
import random
import json

class StratifiedAnnotationSampler:
    """
    Stratified sampling for human annotation.
    
    Ensures coverage across:
    - 11 perturbation conditions (type × position)
    - 3 benchmarks (toolbench, gaia, swebench)
    - 3 quality tiers (high, medium, low)
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
        self.perturbations = [p for p in perturbations if p.get("is_primary_for_experiment")]
        self.random_seed = random_seed
        random.seed(random_seed)
        self._build_index()
    
    def _build_index(self):
        """Index perturbations by (condition, benchmark, tier)."""
        self.index = defaultdict(list)
        for p in self.perturbations:
            condition = (p["perturbation_type"], p["perturbation_position"])
            benchmark = self._get_benchmark(p)
            tier = p.get("quality_tier", "medium")
            self.index[(condition, benchmark, tier)].append(p)
    
    def sample(self, total: int = 100) -> List[Dict]:
        """
        Sample perturbations with stratification.
        
        Returns:
            List of perturbation dicts to annotate
        """
        selected = []
        samples_per_condition = total // len(self.CONDITIONS)  # 9
        
        for condition in self.CONDITIONS:
            condition_samples = self._sample_for_condition(
                condition, samples_per_condition
            )
            selected.extend(condition_samples)
        
        # Fill to exactly total if needed
        while len(selected) < total:
            extra = self._sample_any_remaining()
            if extra:
                selected.append(extra)
            else:
                break
        
        return selected[:total]
    
    def _sample_for_condition(self, condition: tuple, target: int) -> List[Dict]:
        """Sample for a single condition with benchmark/tier stratification."""
        samples = []
        
        for benchmark in self.BENCHMARKS:
            for tier in self.QUALITY_TIERS:
                key = (condition, benchmark, tier)
                if self.index[key]:
                    sample = random.choice(self.index[key])
                    samples.append(sample)
                    self.index[key].remove(sample)
                    
                    if len(samples) >= target:
                        return samples
        
        return samples
    
    def _sample_any_remaining(self) -> Dict:
        """Sample from any remaining pool."""
        for key, pool in self.index.items():
            if pool:
                return pool.pop()
        return None
    
    def _get_benchmark(self, perturbation: Dict) -> str:
        """Extract benchmark from trajectory ID."""
        traj_id = perturbation.get("original_trajectory_id", "").lower()
        if "toolbench" in traj_id:
            return "toolbench"
        elif "gaia" in traj_id:
            return "gaia"
        elif "swe" in traj_id:
            return "swebench"
        return "unknown"
    
    def export_for_annotation(self, selected: List[Dict], output_path: str):
        """Export selected samples to JSON for annotation interface."""
        export_data = []
        for i, p in enumerate(selected):
            export_data.append({
                "annotation_id": i + 1,
                "perturbation_id": p["perturbation_id"],
                "original_trajectory_id": p["original_trajectory_id"],
                "perturbed_trajectory_id": p["perturbed_trajectory_id"],
                "perturbation_type": p["perturbation_type"],
                "perturbation_position": p["perturbation_position"],
                "quality_tier": p.get("quality_tier"),
                "benchmark": self._get_benchmark(p),
                # Include step content for annotation
                "original_step": p.get("original_step_content"),
                "perturbed_step": p.get("perturbed_step_content"),
                "perturbation_description": p.get("perturbation_description"),
                # Annotation fields (to be filled)
                "annotation": {
                    "task_success_degradation": None,
                    "subsequent_error_rate": None,
                    "criticality_rating": None,
                    "confidence": None,
                    "notes": None,
                    "annotated_at": None,
                    "annotator_id": None
                }
            })
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Exported {len(export_data)} samples to {output_path}")
    
    def get_distribution_report(self, selected: List[Dict]) -> Dict:
        """Generate distribution report for validation."""
        report = {
            "total": len(selected),
            "by_condition": defaultdict(int),
            "by_benchmark": defaultdict(int),
            "by_quality_tier": defaultdict(int)
        }
        
        for p in selected:
            cond = f"{p['perturbation_type']}_{p['perturbation_position']}"
            report["by_condition"][cond] += 1
            report["by_benchmark"][self._get_benchmark(p)] += 1
            report["by_quality_tier"][p.get("quality_tier", "unknown")] += 1
        
        return dict(report)
```

**Test file:** `tests/test_stratified_sampler.py`

```python
def test_samples_100_perturbations():
    """Verify exactly 100 samples returned."""
    pass

def test_all_conditions_covered():
    """Verify all 11 conditions have samples."""
    pass

def test_all_benchmarks_represented():
    """Verify all 3 benchmarks have samples."""
    pass

def test_quality_tier_distribution():
    """Verify mix of high/medium/low quality."""
    pass
```

---

### Step 3: Implement CriticalityScorer

**File:** `src/metrics/criticality_scorer.py`

**Requirements:**
- Support hybrid mode: human TCS for annotated samples, heuristic for rest
- Load human annotations from JSON file
- Apply TCS formula: `(TSD * 50) + (SER * 10) + (criticality * 8)`
- Validate heuristic against human TCS (compute Pearson correlation)

**Interface:**
```python
from typing import Dict, List, Optional
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr

class CriticalityScorer:
    """
    Computes True Criticality Score (TCS) for perturbations.
    
    Modes:
    - heuristic: Assign TCS based on (type, position)
    - human: Load from annotation file
    - hybrid: Human for annotated, heuristic for rest
    """
    
    HEURISTIC_TCS = {
        "planning_early": 90, "planning_middle": 70, "planning_late": 40,
        "tool_selection_early": 80, "tool_selection_middle": 60, "tool_selection_late": 30,
        "parameter_early": 50, "parameter_middle": 40, "parameter_late": 20,
        "data_reference_early": 40, "data_reference_middle": 30, "data_reference_late": 20,
    }
    
    def __init__(self, config: Dict):
        self.mode = config.get("mode", "heuristic")
        self.human_annotations = {}
        
        if self.mode in ["human", "hybrid"]:
            annotation_path = config.get("human_annotation_path")
            if annotation_path:
                self._load_human_annotations(annotation_path)
    
    def _load_human_annotations(self, path: str):
        """Load human annotations from JSON."""
        with open(path, 'r') as f:
            annotations = json.load(f)
        
        for ann in annotations:
            if ann["annotation"]["task_success_degradation"] is not None:
                self.human_annotations[ann["perturbation_id"]] = ann["annotation"]
        
        print(f"Loaded {len(self.human_annotations)} human annotations")
    
    def compute_tcs(self, perturbation: Dict) -> float:
        """
        Compute TCS for a single perturbation.
        
        Args:
            perturbation: Perturbation dict with type, position, and ID
        
        Returns:
            TCS value (0-100)
        """
        pert_id = perturbation.get("perturbation_id")
        
        # Try human annotation first (if hybrid or human mode)
        if self.mode in ["human", "hybrid"] and pert_id in self.human_annotations:
            return self._compute_human_tcs(self.human_annotations[pert_id])
        
        # Fall back to heuristic
        if self.mode in ["heuristic", "hybrid"]:
            return self._compute_heuristic_tcs(perturbation)
        
        raise ValueError(f"No TCS available for {pert_id} in mode {self.mode}")
    
    def _compute_human_tcs(self, annotation: Dict) -> float:
        """Compute TCS from human annotation."""
        tsd = annotation.get("task_success_degradation", 0)
        ser = min(annotation.get("subsequent_error_rate", 0), 3)
        criticality = annotation.get("criticality_rating", 3)
        
        tcs = (tsd * 50) + (ser * 10) + (criticality * 8)
        return min(tcs, 100)
    
    def _compute_heuristic_tcs(self, perturbation: Dict) -> float:
        """Compute TCS from heuristic table."""
        ptype = perturbation.get("perturbation_type")
        pos = perturbation.get("perturbation_position")
        key = f"{ptype}_{pos}"
        
        return self.HEURISTIC_TCS.get(key, 50)  # Default to 50 if unknown
    
    def compute_batch(self, perturbations: List[Dict]) -> List[float]:
        """Compute TCS for multiple perturbations."""
        return [self.compute_tcs(p) for p in perturbations]
    
    def validate_heuristic(self) -> Dict:
        """
        Validate heuristic TCS against human annotations.
        
        Returns correlation metrics.
        """
        if not self.human_annotations:
            return {"error": "No human annotations loaded"}
        
        human_tcs = []
        heuristic_tcs = []
        
        for pert_id, annotation in self.human_annotations.items():
            # Need perturbation metadata for heuristic
            # This should be passed in or stored with annotation
            human_val = self._compute_human_tcs(annotation)
            human_tcs.append(human_val)
            
            # Get heuristic value from annotation metadata
            ptype = annotation.get("perturbation_type")
            pos = annotation.get("perturbation_position")
            key = f"{ptype}_{pos}"
            heuristic_val = self.HEURISTIC_TCS.get(key, 50)
            heuristic_tcs.append(heuristic_val)
        
        # Compute correlations
        pearson_r, pearson_p = pearsonr(human_tcs, heuristic_tcs)
        spearman_r, spearman_p = spearmanr(human_tcs, heuristic_tcs)
        
        return {
            "n_samples": len(human_tcs),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "mean_human_tcs": np.mean(human_tcs),
            "mean_heuristic_tcs": np.mean(heuristic_tcs),
            "correlation_strength": (
                "strong" if abs(pearson_r) > 0.7 else
                "moderate" if abs(pearson_r) > 0.4 else
                "weak"
            ),
            "heuristic_valid": abs(pearson_r) > 0.4
        }
```

---

### Step 4: Implement CCGCalculator

**File:** `src/metrics/ccg_calculator.py`

**Requirements:**
- Compute CCG = (JPS - TCS) / TCS for each perturbation
- Aggregate by condition, type, position, benchmark
- Run ANOVA statistical tests
- Compute effect sizes (Cohen's d, eta-squared)

**Interface:**
```python
from typing import Dict, List
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

class CCGCalculator:
    """
    Computes Criticality-Calibration Gap (CCG) metrics.
    
    CCG = (JPS - TCS) / TCS
    - CCG < 0: Under-penalization
    - CCG > 0: Over-penalization
    - CCG ≈ 0: Well-calibrated
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
    
    def compute_ccg(self, jps: float, tcs: float) -> float:
        """Compute CCG for single perturbation."""
        if tcs == 0:
            return 0.0
        return (jps - tcs) / tcs
    
    def compute_all(
        self,
        evaluations: List[Dict],
        tcs_values: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Compute CCG for all evaluations.
        
        Args:
            evaluations: List of judge evaluation dicts with perturbation_id, jps
            tcs_values: Dict mapping perturbation_id -> TCS
        
        Returns:
            DataFrame with CCG values and metadata
        """
        records = []
        for eval_dict in evaluations:
            pert_id = eval_dict["perturbation_id"]
            jps = eval_dict["jps"]
            tcs = tcs_values.get(pert_id, 50)
            ccg = self.compute_ccg(jps, tcs)
            
            records.append({
                "perturbation_id": pert_id,
                "jps": jps,
                "tcs": tcs,
                "ccg": ccg,
                "perturbation_type": eval_dict.get("perturbation_type"),
                "position": eval_dict.get("perturbation_position"),
                "benchmark": eval_dict.get("benchmark"),
                "condition": f"{eval_dict.get('perturbation_type')}_{eval_dict.get('perturbation_position')}"
            })
        
        return pd.DataFrame(records)
    
    def aggregate(self, df: pd.DataFrame, by: str) -> pd.DataFrame:
        """Aggregate CCG by dimension."""
        return df.groupby(by).agg({
            "ccg": ["mean", "std", "count"],
            "jps": "mean",
            "tcs": "mean"
        }).round(3)
    
    def run_anova(self, df: pd.DataFrame, factors: List[str]) -> Dict:
        """
        Run ANOVA tests.
        
        Args:
            df: DataFrame with CCG values
            factors: List of factor columns to test
        
        Returns:
            Dict with F-statistics, p-values, effect sizes
        """
        results = {}
        
        for factor in factors:
            groups = [group["ccg"].values for name, group in df.groupby(factor)]
            f_stat, p_value = stats.f_oneway(*groups)
            
            # Compute eta-squared (effect size)
            ss_between = sum(len(g) * (np.mean(g) - df["ccg"].mean())**2 for g in groups)
            ss_total = sum((df["ccg"] - df["ccg"].mean())**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            results[factor] = {
                "f_statistic": f_stat,
                "p_value": p_value,
                "eta_squared": eta_squared,
                "significant": p_value < 0.05
            }
        
        return results
    
    def run_tukey_hsd(self, df: pd.DataFrame, factor: str) -> pd.DataFrame:
        """Run Tukey HSD post-hoc test."""
        tukey = pairwise_tukeyhsd(df["ccg"], df[factor], alpha=0.05)
        return pd.DataFrame(data=tukey._results_table.data[1:],
                          columns=tukey._results_table.data[0])
    
    def compute_effect_sizes(self, df: pd.DataFrame) -> Dict:
        """Compute Cohen's d for key comparisons."""
        effect_sizes = {}
        
        # Early vs Late
        early = df[df["position"] == "early"]["ccg"]
        late = df[df["position"] == "late"]["ccg"]
        effect_sizes["early_vs_late"] = self._cohens_d(early, late)
        
        # Planning vs Parameter
        planning = df[df["perturbation_type"] == "planning"]["ccg"]
        parameter = df[df["perturbation_type"] == "parameter"]["ccg"]
        effect_sizes["planning_vs_parameter"] = self._cohens_d(planning, parameter)
        
        return effect_sizes
    
    def _cohens_d(self, group1, group2) -> float:
        """Compute Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
    
    def generate_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive CCG analysis report."""
        return {
            "overall": {
                "mean_ccg": df["ccg"].mean(),
                "std_ccg": df["ccg"].std(),
                "mean_jps": df["jps"].mean(),
                "mean_tcs": df["tcs"].mean()
            },
            "by_condition": self.aggregate(df, "condition").to_dict(),
            "by_position": self.aggregate(df, "position").to_dict(),
            "by_type": self.aggregate(df, "perturbation_type").to_dict(),
            "anova": self.run_anova(df, ["perturbation_type", "position"]),
            "effect_sizes": self.compute_effect_sizes(df),
            "hypothesis_support": self._assess_hypothesis(df)
        }
    
    def _assess_hypothesis(self, df: pd.DataFrame) -> Dict:
        """Assess whether results support the hypothesis."""
        early_structural = df[
            (df["position"] == "early") & 
            (df["perturbation_type"].isin(["planning", "tool_selection"]))
        ]["ccg"].mean()
        
        late_surface = df[
            (df["position"] == "late") & 
            (df["perturbation_type"].isin(["parameter", "data_reference"]))
        ]["ccg"].mean()
        
        return {
            "early_structural_mean_ccg": early_structural,
            "late_surface_mean_ccg": late_surface,
            "difference": late_surface - early_structural,
            "direction_as_expected": early_structural < late_surface,
            "early_under_penalized": early_structural < 0,
            "late_over_penalized": late_surface > 0
        }
```

---

### Step 5: Integration into Experiment Runner

**File:** Update `src/experiment_runner.py`

Add new phases for this experiment:

```python
def _phase_judge_evaluation(self):
    """Phase: Run LLM judges on perturbations."""
    print("\n" + "=" * 70)
    print("PHASE: JUDGE EVALUATION")
    print("=" * 70)
    
    # Load config
    judge_config = self.config.get("judges", {})
    source_config = self.config.get("source_experiment", {})
    
    # Load perturbations from source experiment
    perturbations = self.storage.get_primary_perturbations(
        source_config["experiment_id"]
    )
    print(f"Loaded {len(perturbations)} primary perturbations")
    
    # Initialize judge
    from src.judges import ClaudeJudge
    judge = ClaudeJudge(judge_config["models"][0])
    
    # Initialize parallel evaluator
    from src.judges.parallel_evaluator import ParallelJudgeEvaluator
    evaluator = ParallelJudgeEvaluator(judge, self.storage, judge_config)
    
    # Run evaluation
    results = evaluator.evaluate_all(
        perturbations,
        self.experiment_id,
        resume=True
    )
    
    print(f"Completed: {len([r for r in results if r['status'] == 'success'])} successful")
    print(f"Failed: {len([r for r in results if r['status'] == 'failed'])} failed")

def _phase_sample_for_annotation(self):
    """Phase: Sample 100 perturbations for human annotation."""
    print("\n" + "=" * 70)
    print("PHASE: SAMPLE FOR ANNOTATION")
    print("=" * 70)
    
    config = self.config.get("human_annotation", {})
    source_config = self.config.get("source_experiment", {})
    
    # Load perturbations
    perturbations = self.storage.get_primary_perturbations(
        source_config["experiment_id"]
    )
    
    # Sample
    from src.annotation.stratified_sampler import StratifiedAnnotationSampler
    sampler = StratifiedAnnotationSampler(
        perturbations,
        random_seed=config.get("random_seed", 42)
    )
    
    selected = sampler.sample(config.get("total_samples", 100))
    
    # Export
    output_path = f"{config['output_dir']}/annotation_samples.json"
    sampler.export_for_annotation(selected, output_path)
    
    # Print distribution
    report = sampler.get_distribution_report(selected)
    print(f"\nDistribution report:")
    print(f"  By condition: {dict(report['by_condition'])}")
    print(f"  By benchmark: {dict(report['by_benchmark'])}")
    print(f"  By quality: {dict(report['by_quality_tier'])}")

def _phase_compute_ccg(self):
    """Phase: Compute CCG metrics."""
    print("\n" + "=" * 70)
    print("PHASE: COMPUTE CCG")
    print("=" * 70)
    
    # Load evaluations
    evaluations = self.storage.get_judge_outputs(self.experiment_id)
    print(f"Loaded {len(evaluations)} judge evaluations")
    
    # Initialize criticality scorer
    from src.metrics.criticality_scorer import CriticalityScorer
    crit_config = self.config.get("criticality_scoring", {})
    scorer = CriticalityScorer(crit_config)
    
    # Compute TCS for all perturbations
    perturbations = self.storage.get_primary_perturbations(
        self.config["source_experiment"]["experiment_id"]
    )
    tcs_values = {p["perturbation_id"]: scorer.compute_tcs(p) for p in perturbations}
    
    # Validate heuristic if human annotations exist
    if scorer.human_annotations:
        validation = scorer.validate_heuristic()
        print(f"\nHeuristic validation:")
        print(f"  Pearson r: {validation['pearson_r']:.3f} (p={validation['pearson_p']:.4f})")
        print(f"  Correlation: {validation['correlation_strength']}")
    
    # Compute CCG
    from src.metrics.ccg_calculator import CCGCalculator
    calculator = CCGCalculator(self.config.get("ccg_analysis", {}))
    
    df = calculator.compute_all(evaluations, tcs_values)
    report = calculator.generate_report(df)
    
    # Save results
    output_dir = self.config["ccg_analysis"]["output_dir"]
    df.to_csv(f"{output_dir}/ccg_raw_results.csv", index=False)
    
    with open(f"{output_dir}/ccg_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print(f"\nCCG Results:")
    print(f"  Mean CCG: {report['overall']['mean_ccg']:.3f}")
    print(f"  Early structural: {report['hypothesis_support']['early_structural_mean_ccg']:.3f}")
    print(f"  Late surface: {report['hypothesis_support']['late_surface_mean_ccg']:.3f}")
    print(f"  Hypothesis supported: {report['hypothesis_support']['direction_as_expected']}")
```

---

## Validation Checklist

Before marking complete, verify:

- [ ] `ParallelJudgeEvaluator` processes 5 perturbations concurrently
- [ ] Checkpointing saves every 20 evaluations
- [ ] Resume skips already-evaluated perturbations
- [ ] `StratifiedAnnotationSampler` returns exactly 100 samples
- [ ] All 11 conditions represented in sample
- [ ] All 3 benchmarks represented in sample
- [ ] `CriticalityScorer` loads human annotations correctly
- [ ] TCS formula matches config: `(TSD * 50) + (SER * 10) + (criticality * 8)`
- [ ] `CCGCalculator` computes CCG = (JPS - TCS) / TCS
- [ ] ANOVA runs without errors
- [ ] All tests pass: `pytest tests/test_parallel_evaluator.py tests/test_stratified_sampler.py tests/test_criticality_scorer.py tests/test_ccg_calculator.py -v`

---

## File Structure After Implementation

```
src/
├── judges/
│   ├── parallel_evaluator.py    # NEW
│   └── ... (existing)
├── annotation/
│   ├── __init__.py              # NEW
│   └── stratified_sampler.py    # NEW
├── metrics/
│   ├── __init__.py              # UPDATE
│   ├── criticality_scorer.py    # NEW
│   └── ccg_calculator.py        # NEW
└── experiment_runner.py         # UPDATE

tests/
├── test_parallel_evaluator.py   # NEW
├── test_stratified_sampler.py   # NEW
├── test_criticality_scorer.py   # NEW
└── test_ccg_calculator.py       # NEW

data/
└── annotations/
    └── annotation_samples.json  # Generated by sampler

results/
└── ccg_analysis/
    ├── ccg_raw_results.csv
    └── ccg_report.json
```

---

## Run Order

```bash
# 1. Sample for annotation
python main.py --config judge_evaluation_ccg --runner sample_annotation

# 2. HUMAN: Annotate 100 samples (~3.5 hours)
# Edit data/annotations/annotation_samples.json

# 3. Validate heuristic TCS
python main.py --config judge_evaluation_ccg --runner validate_tcs

# 4. Run judge evaluation (parallel)
python main.py --config judge_evaluation_ccg --runner judge

# 5. Compute CCG
python main.py --config judge_evaluation_ccg --runner ccg

# 6. Generate visualizations
python main.py --config judge_evaluation_ccg --runner analyze
```

---

## Success Criteria

1. **Coverage:** 600 perturbations evaluated
2. **Statistical significance:** ANOVA p < 0.05 for position or type
3. **Effect size:** Cohen's d > 0.5 for early vs late
4. **Direction:** Early CCG < 0 (under-penalized), Late CCG > 0 (over-penalized)
5. **Heuristic validation:** Pearson r > 0.4 with human TCS
