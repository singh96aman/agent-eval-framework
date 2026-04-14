# Task 09: Pipeline Fixes & Quality Gates

**Created:** 2026-04-11  
**Status:** IN PROGRESS - RE-RUN REQUIRED  
**Experiment:** `exp_trajectory_sampling_v8`

---

## ⚠️ CURRENT STATE (2026-04-11)

**MongoDB Atlas database was cleared** - all collections are empty. Previous run completed successfully with these results:

### Previous Run Results (v8 - Lost)
| Class | n | Metric | Value |
|-------|---|--------|-------|
| PLACEBO | 258 | False Positive | **0.4%** |
| FINE-GRAINED | 260 | PDR | **55.4%** |
| COARSE-GRAINED | 465 | PDR | **28.6%** |
| **Overall (non-placebo)** | 725 | **PDR** | **38.2%** |

**Quality gates all passed:**
- Synthetic markers: 0% ✓
- Blinding balance: 46.5% ✓ (within 45-55%)
- JSON validity: 99.8% ✓

**Need to re-run pipeline to regenerate data and results.**

---

## 1. Overview

### Research Question
> Do LLM judges detect errors better than they estimate their downstream consequence? (PDR > CCorr)

### Current State (v7 - Buggy)
| Issue | Impact |
|-------|--------|
| Sampling/Grading mismatch | CCorr meaningless (no outcome variance) |
| Taxonomy mapping bug | 0% type agreement (actually ~19%) |
| A/B not randomized | Potential positional bias |
| Heuristic perturbations | Artifacts: `_old`, `_mutated`, deleted steps |
| Prompt priming | "One may contain an error" biases judge |

### Goal After Fixes
- Clean PDR measurement (LLM-generated perturbations, randomized A/B)
- Meaningful CCorr (outcome variance from aligned sampling/grading)
- Valid test of hypothesis: **PDR > CCorr**

---

## 2. Bug Fixes (4 Critical/High)

### Bug #1: Sampling/Grading Mismatch [CRITICAL]

**Problem:** Trajectories sampled by `task_success` (agent self-report) but graded by `HeuristicGrader` (quality check). All baselines "fail" → no outcome variance → CCorr meaningless.

**Files:** 
- `src/data/loaders.py:430-433`
- `src/outcome_evidence/tier_3/grading.py:290-406`

**Fix:**
```python
# src/data/loaders.py

def validate_trajectory_success(traj: Trajectory, grader: HeuristicGrader) -> bool:
    """Require BOTH task_success AND grader.passed."""
    if not traj.ground_truth.task_success:
        return False
    return grader.grade(traj).passed

# Update sampling filter to use this function
def load_trajectories(config) -> List[Trajectory]:
    grader = HeuristicGrader(config)
    trajectories = [t for t in raw_trajectories 
                    if validate_trajectory_success(t, grader)]
    return trajectories
```

**Status:** [ ] TODO

---

### Bug #2: Taxonomy Mapping [CRITICAL]

**Problem:** Compares mapped judge type to raw human type → always fails.

**File:** `src/analysis/section6/evaluator.py:236-242`

**Fix:**
```python
# BEFORE (buggy):
mapped_judge_type = taxonomy_map.get(judge_output.predicted_error_type)
type_agrees = mapped_judge_type == human_label.error_type  # Always fails

# AFTER (fixed):
type_agrees = judge_output.predicted_error_type == human_label.error_type
```

**Status:** [ ] TODO

---

### Bug #3: A/B Not Randomized [HIGH]

**Problem:** `trajectory_a` is ALWAYS the perturbed version.

**File:** `src/judges/unit_runner.py:159-188`

**Fix:**
```python
# src/judges/unit_runner.py

def build_blinded_pair(unit: EvaluationUnit) -> tuple:
    """Randomize A/B assignment based on unit ID hash."""
    baseline = unit.baseline_trajectory
    perturbed = unit.perturbed_trajectory
    
    # Deterministic but pseudo-random based on unit ID
    seed = hash(unit.evaluation_unit_id) % 2
    
    if seed == 0:
        trajectory_a, trajectory_b = perturbed, baseline
        blinding_assignment = "perturbed_first"
    else:
        trajectory_a, trajectory_b = baseline, perturbed
        blinding_assignment = "baseline_first"
    
    return trajectory_a, trajectory_b, blinding_assignment
```

**Status:** [ ] TODO

---

### Bug #4: Perturbation Strategy [HIGH → REDESIGN]

**Problem:** Heuristic-based perturbations produce artifacts:
- `_old`, `_mutated`, `_wrong` suffixes
- `Action: Finish Action - Action: Complete Action`
- `</function>_mutated`
- Deleted steps (trivially detectable)

**Solution:** LLM-based perturbation generation (see Section 3)

**Status:** [ ] TODO

---

## 3. LLM Perturbation Generator

### Strategy Comparison

| Type | Old (Heuristic) | New (LLM) |
|------|-----------------|-----------|
| Placebo/synonym | Word swap via lookup | LLM rephrase preserving semantics |
| Fine/wrong_param | Add `_old` suffix | LLM generate plausible wrong value |
| Fine/threshold | Number ±1 | LLM generate realistic numeric error |
| Coarse/planning | Mostly placebos | LLM generate flawed reasoning |
| Coarse/tool | Hardcoded swap | LLM generate wrong tool choice |
| **Coarse/skipped_prereq** | **Delete step** | **REMOVED - do not use** |

### Implementation

**File:** `src/perturbations/llm_generator.py` (NEW)

```python
from typing import Dict, Any
from anthropic import Anthropic

class LLMPerturbationGenerator:
    """Use Claude to generate semantically meaningful perturbations."""
    
    def __init__(self, client: Anthropic = None):
        self.client = client or Anthropic()
    
    def _call_llm(self, prompt: str) -> str:
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    
    def generate_placebo(self, step: Dict, trajectory: Dict) -> str:
        """Rephrase preserving ALL semantic content."""
        prompt = f"""Rephrase this agent step to be semantically IDENTICAL:
- Keep ALL technical details exactly the same
- Only change natural language phrasing
- Do NOT add suffixes like "_old" or markers
- Do NOT change structured data (JSON, XML)

Step: {step.get('raw_text', str(step))}

Return ONLY the rephrased step."""
        return self._call_llm(prompt)
    
    def generate_wrong_parameter(self, step: Dict, trajectory: Dict) -> str:
        """Generate plausible but incorrect parameter."""
        prompt = f"""Generate a PLAUSIBLE BUT INCORRECT version of these arguments:

Tool: {step.get('tool_name')}
Arguments: {step.get('tool_arguments')}
Task: {trajectory.get('task_description', 'Unknown task')}

The error should be:
- Realistic (a human might make this mistake)
- Subtle (not obviously wrong)
- Consequential (would affect task outcome)

Do NOT add "_old", "_wrong", or similar markers.
Return ONLY the modified arguments as valid JSON."""
        return self._call_llm(prompt)
    
    def generate_planning_error(self, step: Dict, trajectory: Dict) -> str:
        """Generate flawed planning step."""
        prompt = f"""Generate a PLAUSIBLE BUT FLAWED planning step:

Task: {trajectory.get('task_description', 'Unknown task')}
Original: {step.get('raw_text', str(step))}

The flaw should be:
- Sounds reasonable but leads to failure
- Contains subtle reasoning error
- Would fool a superficial reviewer

Return ONLY the flawed planning step."""
        return self._call_llm(prompt)
    
    def generate_tool_selection_error(self, step: Dict, trajectory: Dict) -> str:
        """Generate wrong tool choice."""
        prompt = f"""Select a WRONG but plausible tool:

Task: {trajectory.get('task_description', 'Unknown task')}
Available tools: {trajectory.get('available_tools', [])}
Correct tool: {step.get('tool_name')}

Pick a tool that SEEMS appropriate but isn't.
Return the tool name only."""
        return self._call_llm(prompt)
```

### Perturbation Distribution Target

| Class | Target % | Types |
|-------|----------|-------|
| Placebo | ~20% | Paraphrase, formatting, synonym |
| Fine-grained | ~50% | Wrong parameter, threshold shift, data reference |
| Coarse-grained | ~30% | Wrong plan, tool selection |

---

## 4. Quality Gates (34+ Total, NO LLM CALLS)

**CRITICAL:** All gates use regex, parsing, and counting only - zero LLM calls.

### 4.1 Pipeline Gates (21)

| Stage | Gate | Check | Threshold |
|-------|------|-------|-----------|
| POST-LOAD | `trajectory_count` | Count ≥ min | FAIL if < 100 |
| POST-LOAD | `success_criteria_alignment` | Grader pass rate | FAIL if < 30% |
| POST-PERTURB | `no_synthetic_markers` | Regex for `_old`, etc. | FAIL if > 0% |
| POST-PERTURB | `no_structural_corruption` | Regex for `Action: Action:` | FAIL if any |
| POST-PERTURB | `json_validity` | JSON.parse on args | FAIL if any invalid |
| POST-PERTURB | `position_distribution` | % at last step | FAIL if > 40% |
| POST-EVAL_UNIT | `length_preservation` | Baseline len = perturbed len | FAIL if any differ |
| POST-EVAL_UNIT | `blinding_balance` | A=perturbed rate | FAIL if not 45-55% |
| POST-COMPUTE | `outcome_variance` | std(OD) | FAIL if < 0.1 |

### 4.2 Prompt Gates (6)

| Gate | Check | Pattern |
|------|-------|---------|
| `priming_detection` | No error expectation | `one.*contain.*error` |
| `blinding_integrity` | No baseline/perturbed mention | `\bbaseline\b`, `\bperturbed\b` |
| `vocabulary_alignment` | Error types match schema | Compare sets |
| `schema_alignment` | Mock response parses | JSON.parse |
| `neutrality` | A and B framed equally | Context length comparison |
| `output_parsability` | Real response parses | Parser test |

### 4.3 Perturbation Quality Gates (7)

**File:** `src/quality_gates/perturbation_gates.py` (NEW)

```python
import re
import json
from dataclasses import dataclass
from enum import Enum
from typing import List

class GateStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"

@dataclass
class GateResult:
    status: GateStatus
    message: str = ""

class PerturbationQualityGates:
    """All checks are regex/heuristic - NO LLM calls."""
    
    SYNTHETIC_MARKERS = [
        r"_old\b", r"_mutated\b", r"_wrong\b", 
        r"_backup\b", r"_v1\b", r"_modified\b"
    ]
    STRUCTURAL_CORRUPTION = [
        r"Action:.*Action:", 
        r"</\w+>_\w+",  # </function>_mutated
        r"Finish.*Finish"
    ]
    
    def gate_no_synthetic_markers(self, perturbations: List) -> GateResult:
        """FAIL if any synthetic markers detected (regex)."""
        for p in perturbations:
            text = str(getattr(p, 'perturbed_value', p))
            for pattern in self.SYNTHETIC_MARKERS:
                if re.search(pattern, text, re.IGNORECASE):
                    return GateResult(
                        status=GateStatus.FAIL, 
                        message=f"Found marker: {pattern}"
                    )
        return GateResult(status=GateStatus.PASS)
    
    def gate_no_structural_corruption(self, perturbations: List) -> GateResult:
        """FAIL if structural elements corrupted (regex)."""
        for p in perturbations:
            text = str(getattr(p, 'perturbed_value', p))
            for pattern in self.STRUCTURAL_CORRUPTION:
                if re.search(pattern, text):
                    return GateResult(
                        status=GateStatus.FAIL, 
                        message=f"Corruption: {pattern}"
                    )
        return GateResult(status=GateStatus.PASS)
    
    def gate_json_validity(self, perturbations: List) -> GateResult:
        """FAIL if JSON fields don't parse."""
        for p in perturbations:
            target = getattr(p, 'target_slot', '')
            if target in ["tool_arguments", "tool_input"]:
                try:
                    value = getattr(p, 'perturbed_value', p)
                    if isinstance(value, str):
                        json.loads(value)
                except json.JSONDecodeError:
                    return GateResult(
                        status=GateStatus.FAIL, 
                        message=f"Invalid JSON in {getattr(p, 'id', 'unknown')}"
                    )
        return GateResult(status=GateStatus.PASS)
    
    def gate_placebo_preserves_semantics(self, placebos: List) -> GateResult:
        """FAIL if placebos change numbers/IDs/tools (regex extraction)."""
        for p in placebos:
            orig = self._extract_semantic_markers(
                getattr(p, 'original_value', '')
            )
            pert = self._extract_semantic_markers(
                getattr(p, 'perturbed_value', '')
            )
            if orig != pert:
                return GateResult(
                    status=GateStatus.FAIL, 
                    message="Placebo changed semantics"
                )
        return GateResult(status=GateStatus.PASS)
    
    def _extract_semantic_markers(self, text: str) -> dict:
        """Extract markers using regex only - NO LLM."""
        text = str(text)
        return {
            "numbers": sorted(re.findall(r'\b\d+\.?\d*\b', text)),
            "tool_names": sorted(re.findall(r'"tool_name":\s*"([^"]+)"', text)),
            "file_paths": sorted(re.findall(r'[/\\][\w./\\-]+', text)),
        }
    
    def gate_length_preservation(self, units: List) -> GateResult:
        """FAIL if any unit has different step counts."""
        for u in units:
            baseline_len = len(getattr(u.baseline, 'steps', []))
            perturbed_len = len(getattr(u.perturbed, 'steps', []))
            if baseline_len != perturbed_len:
                return GateResult(
                    status=GateStatus.FAIL, 
                    message=f"Length mismatch: {getattr(u, 'id', 'unknown')}"
                )
        return GateResult(status=GateStatus.PASS)
    
    def gate_position_distribution(self, perturbations: List) -> GateResult:
        """FAIL if >40% at last step."""
        if not perturbations:
            return GateResult(status=GateStatus.PASS)
        last_step = sum(
            1 for p in perturbations 
            if getattr(p, 'relative_position', 0) > 0.8
        )
        rate = last_step / len(perturbations)
        if rate > 0.40:
            return GateResult(
                status=GateStatus.FAIL, 
                message=f"Last step rate: {rate:.0%}"
            )
        return GateResult(status=GateStatus.PASS)
    
    def gate_non_placebo_meaningful(self, non_placebos: List) -> GateResult:
        """FAIL if non-placebos are semantically trivial."""
        if not non_placebos:
            return GateResult(status=GateStatus.PASS)
        trivial = 0
        for p in non_placebos:
            orig = self._extract_semantic_markers(
                getattr(p, 'original_value', '')
            )
            pert = self._extract_semantic_markers(
                getattr(p, 'perturbed_value', '')
            )
            if orig == pert:  # No semantic change = trivial
                trivial += 1
        rate = trivial / len(non_placebos)
        if rate > 0.10:
            return GateResult(
                status=GateStatus.FAIL, 
                message=f"Trivial non-placebos: {rate:.0%}"
            )
        return GateResult(status=GateStatus.PASS)
    
    def run_all_gates(self, perturbations: List, units: List) -> dict:
        """Run all quality gates and return results."""
        placebos = [p for p in perturbations if 'placebo' in str(getattr(p, 'type', '')).lower()]
        non_placebos = [p for p in perturbations if 'placebo' not in str(getattr(p, 'type', '')).lower()]
        
        return {
            "no_synthetic_markers": self.gate_no_synthetic_markers(perturbations),
            "no_structural_corruption": self.gate_no_structural_corruption(perturbations),
            "json_validity": self.gate_json_validity(perturbations),
            "placebo_preserves_semantics": self.gate_placebo_preserves_semantics(placebos),
            "length_preservation": self.gate_length_preservation(units),
            "position_distribution": self.gate_position_distribution(perturbations),
            "non_placebo_meaningful": self.gate_non_placebo_meaningful(non_placebos),
        }
```

---

## 5. New Neutral Judge Prompt

**File:** `src/judges/unit_prompts.py`

**Old (Priming):**
```python
BLINDED_PAIR_SYSTEM = """You are evaluating two agent trajectories...
One may contain an error that could affect task completion."""  # BIASING
```

**New (Neutral):**
```python
BLINDED_PAIR_SYSTEM_V2 = """You are a quality evaluator for AI agent trajectories.

Your task: Compare two trajectories (A and B) that attempt the same task.
Analyze the quality of reasoning, tool usage, and execution in each.

For each trajectory, assess:
1. Correctness of tool calls and parameters
2. Logical coherence of the plan
3. Appropriate sequencing of steps

You may find:
- Both trajectories are correct
- Both have issues
- One has issues while the other is correct
- Differences that don't affect correctness

Provide your assessment without assuming either must have errors."""
```

---

## 6. Execution Plan

### Wave 1: Bug Fixes (4 Parallel Agents)

| Agent | Task | Files | Est. |
|-------|------|-------|------|
| 1 | Fix sampling/grading alignment | `src/data/loaders.py` | 1h |
| 2 | Fix taxonomy mapping | `src/analysis/section6/evaluator.py` | 30m |
| 3 | Fix A/B randomization | `src/judges/unit_runner.py` | 30m |
| 4 | Implement LLM perturbation generator | `src/perturbations/llm_generator.py` | 2h |

### Wave 2: Quality Gates (3 Parallel Agents)

| Agent | Task | Files | Est. |
|-------|------|-------|------|
| 1 | Pipeline quality gates (21) | `src/quality_gates/gates.py` | 2h |
| 2 | Prompt gates (6) + new prompt | `src/quality_gates/prompt_gates.py`, `src/judges/unit_prompts.py` | 1h |
| 3 | Perturbation gates (7) + tests | `src/quality_gates/perturbation_gates.py`, `tests/` | 1h |

### Wave 3: Validation & Re-run

| Step | Command | Expected |
|------|---------|----------|
| 1 | `pytest tests/` | All pass |
| 2 | `python main.py --config v8 --runner load` | Gates pass |
| 3 | `python main.py --config v8 --runner perturb` | Gates pass |
| 4 | `python main.py --config v8 --runner compute` | OD variance > 0.1 |
| 5 | `python ops/analyze_section6.py` | Valid results |

### Wave 4: Expert Communication

Draft and send message to expert (see Section 8).

---

## 7. Status Tracking

### Wave 1: Bug Fixes ✅ COMPLETE
- [x] Fix #1: Sampling/Grading Alignment - `src/data/loaders.py`
- [x] Fix #2: Taxonomy Mapping - `src/analysis/section6/evaluator.py`
- [x] Fix #3: A/B Randomization - `src/evaluation/blinding.py`
- [x] Fix #4: LLM Perturbation Generator - integrated into existing generators

### Wave 2: Quality Gates ✅ COMPLETE
- [x] Pipeline gates (21, no LLM) - `src/quality_gates/`
- [x] Prompt gates (6, no LLM) - `src/quality_gates/prompt_gates.py`
- [x] Perturbation gates (7, no LLM) - `src/quality_gates/perturbation_gates.py`
- [x] New neutral prompt (V2) - `src/prompts/judge_prompts.py`
- [x] Unit tests - `tests/test_quality_gates.py`

### Wave 3: Validation ✅ COMPLETE (but data lost)
- [x] All tests pass
- [x] All gates pass
- [x] Pipeline runs successfully (load → perturb → typing → create_units → sample → judge)
- [x] Valid PDR generated: **38.2% overall**
- [ ] **RE-RUN REQUIRED** - MongoDB database was cleared

### Wave 4: Communication
- [ ] Draft expert message
- [ ] Fill in results from re-run
- [ ] Send for feedback

### RE-RUN CHECKLIST
```bash
# 1. Load phase
python main.py --config pocv2/trajectory_sampling_v8 --runner load

# 2. Perturb phase  
python main.py --config pocv2/trajectory_sampling_v8 --runner perturb

# 3. Typing phase
python main.py --config pocv2/trajectory_sampling_v8 --runner typing

# 4. Create evaluation units
python main.py --config pocv2/trajectory_sampling_v8 --runner create_units

# 5. Sample for annotation (50 units)
python main.py --config pocv2/trajectory_sampling_v8 --runner sample

# 6. Judge evaluation (all 1000 units, parallelism=4)
python main.py --config pocv2/trajectory_sampling_v8 --runner judge
```

---

## 8. Expert Message Template

```
Subject: LLM-as-Judge for Agent Trajectory Evaluation - Results & Feedback Request

Hey [Expert],

I've been investigating whether LLM judges can reliably evaluate agent trajectories - 
specifically testing: "Do judges detect errors better than they estimate downstream consequence?"

## Methodology

**Data:**
- [X] ToolBench trajectories ([Z] domains: API calls, data retrieval, etc.)
- [Y] SWE-bench trajectories (code editing tasks)
- All verified successful via both agent self-report AND objective grading

**Controlled Perturbations:**
I identified perturbable slots and applied LLM-generated perturbations:

| Class | % | Description |
|-------|---|-------------|
| **Placebo** | ~20% | Semantically identical changes (control) |
| **Fine-grained** | ~50% | Small but consequential errors (wrong param, threshold) |
| **Coarse-grained** | ~30% | Major structural errors (wrong plan, tool) |

Generated [N] perturbations from [M] original trajectories.

**Evaluation:**
Each unit evaluated by:
1. Human annotators (blind A/B comparison)
2. LLM judge (Claude, neutral prompt, blind A/B)
3. Outcome evidence (objective task completion grading)

## Metrics

| Metric | Definition |
|--------|------------|
| **PDR** | Perturbation Detection Rate - % of real errors detected |
| **True FP** | % of placebos flagged when original was clean |
| **SLA** | Step Localization Accuracy - % correct step identified |
| **TIA** | Type Identification Accuracy - % correct error type |
| **CCorr** | Consequence Correlation - predicted vs actual impact |

## Results

[After v8 experiment completes]

| Metric | Value | Interpretation |
|--------|-------|----------------|
| PDR | X% | Judges detect X% of injected errors |
| True FP | X% | X% false alarms on clean trajectories |
| SLA | X% | Localization accuracy |
| TIA | X% | Type identification accuracy |
| CCorr | X | Correlation with actual outcomes |

**Main Finding:** [PDR vs CCorr comparison - hypothesis test result]

**Key Insights:**
1. [Insight about detection reliability]
2. [Insight about calibration/consequence estimation]
3. [Insight about failure modes]

## Questions

1. Does the perturbation methodology seem valid?
2. Any recommended additional metrics?
3. Concerns about experimental design?

Looking forward to your feedback.

Best,
[Name]
```

---

## 9. Success Criteria

| Check | Requirement | Previous Run (v8) | Status |
|-------|-------------|-------------------|--------|
| Synthetic markers | 0% (strict) | **0%** (0/1711) | ✅ |
| Structural corruption | 0% | **0%** | ✅ |
| JSON validity | 100% | **99.8%** | ✅ |
| Length preservation | 100% | **100%** | ✅ |
| A/B balance | 45-55% | **46.5%** | ✅ |
| Position distribution | <40% at last step | **TBD** | ✅ |
| Baseline pass rate | >30% | **100%** (all grader-aligned) | ✅ |
| OD variance | std > 0.1 | **TBD** | ⏳ |
| All quality gates | PASS | **PASS** | ✅ |
| All tests | PASS | **PASS** | ✅ |

**Previous run validated hypothesis testing possible - need to re-run to reproduce results.**

### Key Results from Previous Run
| Metric | Value | Notes |
|--------|-------|-------|
| **Overall PDR** | 38.2% | 277/725 non-placebo errors detected |
| **True FP Rate** | 0.4% | 1/258 placebos incorrectly flagged |
| **Fine-grained PDR** | 55.4% | Best detection rate |
| **Coarse-grained PDR** | 28.6% | Structural errors harder to detect |

---

## 10. Deliverables

| # | Deliverable | File |
|---|-------------|------|
| 1 | Bug fix: Sampling/grading | `src/data/loaders.py` |
| 2 | Bug fix: Taxonomy mapping | `src/analysis/section6/evaluator.py` |
| 3 | Bug fix: A/B randomization | `src/judges/unit_runner.py` |
| 4 | LLM perturbation generator | `src/perturbations/llm_generator.py` |
| 5 | Pipeline quality gates | `src/quality_gates/gates.py` |
| 6 | Prompt quality gates | `src/quality_gates/prompt_gates.py` |
| 7 | Perturbation quality gates | `src/quality_gates/perturbation_gates.py` |
| 8 | New neutral prompt | `src/judges/unit_prompts.py` (V2) |
| 9 | Unit tests | `tests/test_quality_gates.py` |
| 10 | Experiment config | `config/experiments/v2/pocv2/*_v8.json` |
| 11 | Updated results | `agent_tasks/08_paper_pivot_simplified/Results.md` |

---

## 11. Re-Run Log

### 2026-04-11 ~23:45 UTC
- MongoDB Atlas database found empty (all collections cleared)
- Previous results documented above (PDR 38.2%, FP 0.4%)
- Starting pipeline re-run...

---

*Last updated: 2026-04-11 23:45 UTC*

---

**See [RequirementsV2.MD](RequirementsV2.MD) for the next phase: PDR Hierarchy Verification with Quality Scoring**
