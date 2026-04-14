# Task 09 V2: PDR Hierarchy Verification with Quality Scoring

**Created:** 2026-04-13  
**Status:** PENDING  
**Experiment:** `dual_mode_v2`

---

## Context

The research hypothesis is that LLM judges should detect errors with this hierarchy:
- **Placebo FP (<10%)** < **Fine-grained PDR** < **Coarse-grained PDR**

Previous v8 results showed unexpected ordering (coarse-grained 28.6% < fine-grained 55.4%), suggesting perturbation quality issues. 

**Key insight:** We use LLM to generate perturbations, so we should also use LLM to **validate** that perturbations match their intended class before running the judge.

---

## Step 0: Implement Perturbation Quality Scoring Module

The quality scoring module was designed (task 04) but implementation is missing. We need to build it.

### 0A: Create `src/scoring/quality_scorer.py`

```python
"""
LLM-based perturbation class validator.
Validates that perturbations match their intended class (simplified approach).
"""
class PerturbationClassValidator:
    def __init__(self, model_config: dict):
        self.model_config = model_config
        
    def validate_perturbation(self, perturbation: dict) -> dict:
        """Validate a perturbation's class using LLM."""
        # Returns: {
        #   "class_matches": 0 or 1,  # Does perturbation match claimed class?
        #   "reasoning": "..."        # Brief explanation
        # }
```

### 0B: Prompt is in config (not hardcoded)

The prompt is defined in the config file at `phases.perturb.class_validation.prompt`.

**No separate prompts.py needed** - the validator reads the prompt from config and substitutes template variables.

**Template substitution in class_validator.py:**
```python
def _build_prompt(self, perturbation: dict, prompt_template: str) -> str:
    """Substitute template variables in prompt."""
    return prompt_template.replace(
        "{{perturbation_class}}", perturbation["perturbation_class"]
    ).replace(
        "{{perturbation_type}}", perturbation["perturbation_type"]
    ).replace(
        "{{step_index}}", str(perturbation["target_step_index"])
    ).replace(
        "{{original_value}}", str(perturbation["original_value"])[:500]
    ).replace(
        "{{perturbed_value}}", str(perturbation["perturbed_value"])[:500]
    ).replace(
        "{{mutation_method}}", perturbation["mutation_method"]
    )
```

**Validation logic by class:**
| Class | class_matches=1 if... |
|-------|----------------------|
| PLACEBO | Change is purely cosmetic, no semantic difference |
| FINE_GRAINED | Change is subtle but would affect task outcome |
| COARSE_GRAINED | Change is major structural error |

### 0C: Integrate into perturb phase in `main.py`

After generating each perturbation, call quality scorer if enabled.

### 0D: Add quality gate

New gate: `perturbation_class_validity` 
- FAIL if >10% of perturbations have `class_matches = 0`
- This ensures placebo/fine/coarse classifications are accurate

**Calculation:**
```python
def check_class_match_rate(perturbations, min_rate=0.90):
    matches = sum(1 for p in perturbations if p["class_validation"]["class_matches"] == 1)
    rate = matches / len(perturbations)
    passed = rate >= min_rate
    return passed, rate

# Example: 100 perturbations, 92 match their class
# rate = 92/100 = 0.92 (92%) → PASS (>= 90%)
```

**On completion, notify:**
```bash
osascript -e 'display notification "Step 0 complete: Class validation module created" with title "Claude Code" sound name "default"'
```

---

## Step 1: Create dual_mode_v2 Config

**File to create:** `config/experiments/v3/poc/dual_mode_v2.json`

Copy from `dual_mode_v1.json` and add class validation:
```json
{
  "experiment": {
    "id": "dual_mode_v2",
    "output": {
      "results": { "dir": "results/dual_mode_v2" }
    }
  },
  "phases": {
    "perturb": {
      "class_validation": {
        "enabled": true,
        "model": {
          "provider": "bedrock",
          "model": "global.anthropic.claude-sonnet-4-20250514-v1:0",
          "max_tokens": 500,
          "temperature": 0.0
        },
        "prompt": "You are validating whether a perturbation matches its claimed class.\n\n## Perturbation Classes:\n- **PLACEBO**: Change should be purely cosmetic (formatting, synonyms, paraphrasing). NO semantic change - the trajectory should behave identically.\n- **FINE_GRAINED**: Subtle but real error (wrong parameter value, off-by-one, typo in ID). Small change that could cause task failure.\n- **COARSE_GRAINED**: Major structural error (wrong tool, flawed plan, premature termination). Significant change that would likely cause task failure.\n\n## Perturbation to Validate:\n- Claimed class: {{perturbation_class}}\n- Perturbation type: {{perturbation_type}}\n- Target step: {{step_index}}\n- Original value: {{original_value}}\n- Perturbed value: {{perturbed_value}}\n- Mutation method: {{mutation_method}}\n\n## Question:\nDoes this perturbation match its claimed class?\n\nRespond with JSON only:\n{\"class_matches\": 0 or 1, \"reasoning\": \"brief explanation\"}",
        "min_match_rate": 0.90
      }
    }
  }
}
```

**Template variables in prompt:**
- `{{perturbation_class}}` - placebo, fine_grained, or coarse_grained
- `{{perturbation_type}}` - specific type (e.g., formatting, wrong_value, false_terminal)
- `{{step_index}}` - which step was perturbed
- `{{original_value}}` - original content
- `{{perturbed_value}}` - modified content
- `{{mutation_method}}` - how the perturbation was generated

**On completion, notify:**
```bash
osascript -e 'display notification "Step 1 complete: dual_mode_v2.json created" with title "Claude Code" sound name "default"'
```

---

## Step 2: Run Load Phase

```bash
python main.py --config v3/poc/dual_mode_v2 --phase load
```

**Quality Gate Check:**
```bash
python ops/run_quality_gate.py --phase load --all --experiment dual_mode_v2
```

**Pass Criteria:**
- trajectory_count: min 10
- grader_pass_rate: min 30%
- Baseline verification passing at min_score 98

**On completion, notify:**
```bash
osascript -e 'display notification "Step 2 complete: Load phase passed" with title "Claude Code" sound name "default"'
```

---

## Step 3: Run Perturb Phase (with Class Validation)

```bash
python main.py --config v3/poc/dual_mode_v2 --phase perturb
```

**What happens:**
1. Generate perturbations (placebo, fine_grained, coarse_grained)
2. **NEW: Score each perturbation with LLM quality scorer**
3. Store perturbations with `quality_score` field
4. Run quality gates

**Quality Gate Checks:**
```bash
python ops/run_quality_gate.py --phase perturb --all --experiment dual_mode_v2
python ops/run_quality_gate.py --phase evaluation_unit --all --experiment dual_mode_v2
```

**Pass Criteria:**
- no_synthetic_markers: 0% (no _old, _mutated artifacts)
- no_structural_corruption: 0%
- json_validity: 100%
- position_distribution: max 40% at last step
- blinding_balance: 35-65% A-is-perturbed
- **NEW: class_matches rate >= 90%** (perturbations match their class)

**On completion, notify:**
```bash
osascript -e 'display notification "Step 3 complete: Perturb phase passed, class_matches >= 90%" with title "Claude Code" sound name "default"'
```

---

## Step 4: Run Judge Phase

```bash
python main.py --config v3/poc/dual_mode_v2 --phase judge
```

Runs both modes configured:
- `single_trajectory` mode
- `blinded_pair` mode (V3 prompts)

Monitor for completion and parse errors.

**On completion, notify:**
```bash
osascript -e 'display notification "Step 4 complete: Judge phase finished" with title "Claude Code" sound name "default"'
```

---

## Step 5: Run Compute Phase

```bash
python main.py --config v3/poc/dual_mode_v2 --phase compute
```

**Expected Output Table:**
```
Version                        Placebo FP   Fine-grained   Coarse-grained   PDR
---------------------------------------------------------------------------------
v1_single_trajectory           X.X%         Y.Y%           Z.Z%             W.W%
v1_blinded_pair                X.X%         Y.Y%           Z.Z%             W.W%
```

**On completion, notify:**
```bash
osascript -e 'display notification "Step 5 complete: Compute phase finished - check PDR table" with title "Claude Code" sound name "default"'
```

---

## Step 6: Verify PDR Hierarchy

**Success Criteria:**
| Criterion | Target |
|-----------|--------|
| Placebo FP Rate | < 10% |
| Fine-grained PDR > Placebo FP | Margin > 10 pts |
| Coarse-grained PDR > Fine-grained PDR | Margin > 0 pts |

**Decision Logic:**
```
IF placebo_fp < 10% 
   AND fine_grained_pdr > placebo_fp + 10% 
   AND coarse_grained_pdr > fine_grained_pdr:
    SUCCESS - Hierarchy verified
ELSE:
    INVESTIGATE - Run failure analysis (Step 7)
```

**On success, notify:**
```bash
osascript -e 'display notification "Step 6 complete: PDR HIERARCHY VERIFIED!" with title "Claude Code" sound name "default"'
```

**On failure, notify:**
```bash
osascript -e 'display notification "Step 6: PDR hierarchy NOT met - proceeding to failure analysis" with title "Claude Code" sound name "default"'
```

---

## Step 7: Failure Analysis (If Needed)

### 7A: If Placebo FP > 10%

**First check: Did class validation catch this?**
```python
# Check if placebo FPs had class_matches=0 (misclassified)
from pymongo import MongoClient
import os
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["agent_judge_experiment"]

# Get placebo perturbations with their class validation
placebos = list(db["perturbed_trajectories"].find({
    "experiment_id": "dual_mode_v2",
    "perturbation_record.perturbation_class": "placebo"
}))

# Check class_matches distribution
misclassified = [p for p in placebos 
    if p.get("class_validation", {}).get("class_matches", 1) == 0]
print(f"Misclassified placebos: {len(misclassified)}/{len(placebos)}")
```

**If class validation caught it (class_matches=0):**
- Filter out misclassified perturbations before judge phase
- Fix placebo generators

**If class validation missed it (class_matches=1 but still FP):**
- Judge is too sensitive
- Review V3 prompt

### 7B: If Coarse-grained PDR <= Fine-grained PDR

**Inspect detection rates by type:**
```python
cg_by_type = db["judge_eval_outputs"].aggregate([
    {"$match": {"experiment_id": "dual_mode_v2"}},
    {"$lookup": {
        "from": "evaluation_units",
        "localField": "evaluation_unit_id",
        "foreignField": "evaluation_unit_id",
        "as": "unit"
    }},
    {"$unwind": "$unit"},
    {"$match": {"unit.perturbation_class": "coarse_grained"}},
    {"$group": {
        "_id": "$unit.perturbation_type",
        "total": {"$sum": 1},
        "detected": {"$sum": {"$cond": ["$detection.error_detected", 1, 0]}}
    }}
])

for row in cg_by_type:
    rate = row['detected'] / row['total'] * 100 if row['total'] > 0 else 0
    print(f"{row['_id']}: {row['detected']}/{row['total']} ({rate:.1f}%)")
```

**Likely issues:**
- `false_terminal` creates "incomplete" not "wrong" trajectories
- `premature_termination` same issue
- `wrong_plan` not using LLM (template fallback)

**Inspect missed coarse-grained with class validation:**
```python
cg_missed = list(db["judge_eval_outputs"].aggregate([
    {"$match": {"experiment_id": "dual_mode_v2"}},
    {"$lookup": {
        "from": "evaluation_units",
        "localField": "evaluation_unit_id",
        "foreignField": "evaluation_unit_id",
        "as": "unit"
    }},
    {"$unwind": "$unit"},
    {"$match": {
        "unit.perturbation_class": "coarse_grained",
        "detection.error_detected": False
    }},
    {"$limit": 5}
]))

for s in cg_missed:
    print(f"Type: {s['unit'].get('perturbation_type')}")
    pert_id = s['unit'].get('perturbation_id')
    pert = db["perturbed_trajectories"].find_one({"perturbation_id": pert_id})
    if pert:
        cv = pert.get("class_validation", {})
        print(f"Class matches: {cv.get('class_matches')}, Reasoning: {cv.get('reasoning', '')[:100]}")
        pr = pert.get("perturbation_record", {})
        print(f"Original: {str(pr.get('original_value', ''))[:200]}")
        print(f"Perturbed: {str(pr.get('perturbed_value', ''))[:200]}")
```

---

## Critical Files for Implementation

| File | Purpose | Action |
|------|---------|--------|
| `config/experiments/v3/poc/dual_mode_v1.json` | Source config | COPY |
| `config/experiments/v3/poc/dual_mode_v2.json` | Config with class validation + prompt | CREATE |
| `main.py` | Pipeline orchestrator | MODIFY |
| `src/scoring/__init__.py` | Scoring module init | CREATE |
| `src/scoring/class_validator.py` | LLM class validator (reads prompt from config) | CREATE |
| `src/quality_gates/pipeline_gates.py` | Gate implementations | MODIFY |
| `tests/test_class_validator.py` | Unit tests | CREATE |

### Reference Files
| File | Purpose |
|------|---------|
| `src/perturbations/llm_generator.py` | Pattern for LLM calls in perturbation code |
| `src/llm/__init__.py` | Bedrock client utilities |

---

## Verification Checklist

- [ ] Class validation module created (`src/scoring/`)
- [ ] Unit tests pass (`pytest tests/test_class_validator.py`)
- [ ] dual_mode_v2.json config created with class_validation enabled
- [ ] Load phase: trajectory_count >= 10, grader_pass_rate >= 30%
- [ ] Perturb phase: class_matches rate >= 90%
- [ ] Judge phase: Both modes complete without errors
- [ ] Compute phase: PDR hierarchy verified (placebo < fine < coarse)

---

## Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| class_matches rate | >= 90% | Validates perturbation classification |
| Placebo FP | < 10% | Control group working |
| Fine-grained PDR | > Placebo FP + 10% | Subtle errors detected |
| Coarse-grained PDR | > Fine-grained PDR | Major errors most detectable |

---

*Created: 2026-04-13*
