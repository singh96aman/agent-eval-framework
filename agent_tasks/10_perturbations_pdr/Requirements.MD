# Task 10: Fix Coarse-Grained PDR Detection

## Background

**Experiment:** dual_mode_v3  
**Date:** 2026-04-13  
**Status:** PDR Hierarchy NOT Verified

### Current Results (Clean Baselines Only)

| Mode | Placebo FP | Fine-grained | Coarse-grained |
|------|------------|--------------|----------------|
| blinded_pair | 0.0% | 50.0% | **0.0%** |
| single_trajectory | 0.0% | 33.3% | **0.0%** |

### Target Hierarchy

```
Placebo FP (<10%) < Fine-grained PDR < Coarse-grained PDR
```

**Problem:** Coarse-grained PDR is 0%, lower than fine-grained. This inverts the expected hierarchy.

---

## Phase 1: Investigation

### Step 1.1: Inspect Actual Coarse-Grained Perturbations

**Goal:** Understand what wrong_plan perturbations were actually generated.

```bash
# Start MongoDB if needed
mongod --dbpath ~/data/db --fork --logpath ~/data/db/mongo.log

# Query coarse-grained perturbations
python -c "
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client['agent_judge_experiment']

perturbs = list(db['perturbed_trajectories'].find(
    {'experiment_id': 'dual_mode_v3', 'perturbation.class': 'coarse_grained'},
    {'trajectory_id': 1, 'perturbation': 1}
))

print(f'Total coarse-grained: {len(perturbs)}')
for i, p in enumerate(perturbs):
    pert = p.get('perturbation', {})
    print(f'\\n=== Perturbation {i+1} ===')
    print(f'Type: {pert.get(\"type\")}')
    print(f'Mutation method: {pert.get(\"mutation_method\")}')
    print(f'Class validated: {pert.get(\"class_validation\", {})}')
    details = pert.get('details', {})
    if 'original_plan' in details:
        print(f'\\nOriginal plan (first 300 chars):\\n{details[\"original_plan\"][:300]}')
    if 'mutated_plan' in details:
        print(f'\\nMutated plan (first 300 chars):\\n{details[\"mutated_plan\"][:300]}')
"
```

**Validation Checkpoint:**
- [ ] Can see what wrong_plan mutations were generated
- [ ] Can compare original vs mutated plans
- [ ] Note: Are the mutations task-blocking or just suboptimal?

---

### Step 1.2: Check Judge Reasoning for Coarse-Grained Units

**Goal:** See if judge noticed the errors but scored differently.

```bash
python -c "
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client['agent_judge_experiment']

# Get coarse-grained evaluation units
units = list(db['evaluation_units'].find(
    {'experiment_id': 'dual_mode_v3', 'perturbation.class': 'coarse_grained'}
))

print(f'Coarse-grained units: {len(units)}')

for unit in units[:5]:
    unit_id = unit['unit_id']
    print(f'\\n=== Unit: {unit_id} ===')
    print(f'Type: {unit[\"perturbation\"].get(\"type\")}')
    
    # Get judge outputs for this unit
    outputs = list(db['judge_outputs'].find({'unit_id': unit_id}))
    for out in outputs:
        version = out.get('version', 'unknown')
        score = out.get('score')
        reasoning = out.get('reasoning', '')[:500]
        print(f'\\n  [{version}] Score: {score}')
        print(f'  Reasoning: {reasoning}...')
"
```

**Validation Checkpoint:**
- [ ] Review judge reasoning for coarse-grained units
- [ ] Note: Does judge mention plan issues?
- [ ] Note: Is score high (>80) despite detecting issues?

---

### Step 1.3: Check Class Validation Details

**Goal:** Understand why class match rate was only 70.3%.

```bash
python -c "
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client['agent_judge_experiment']

# Get all perturbations with class validation
perturbs = list(db['perturbed_trajectories'].find(
    {'experiment_id': 'dual_mode_v3'},
    {'perturbation.class': 1, 'perturbation.type': 1, 'perturbation.class_validation': 1}
))

# Count by class and validation result
from collections import defaultdict
stats = defaultdict(lambda: {'match': 0, 'mismatch': 0})

for p in perturbs:
    pert = p.get('perturbation', {})
    cls = pert.get('class', 'unknown')
    validation = pert.get('class_validation', {})
    if validation.get('class_matches') == 1:
        stats[cls]['match'] += 1
    else:
        stats[cls]['mismatch'] += 1
        print(f'\\nMISMATCH: {cls}/{pert.get(\"type\")}')
        print(f'  Reasoning: {validation.get(\"reasoning\", \"N/A\")[:200]}')

print('\\n=== Summary ===')
for cls, counts in stats.items():
    total = counts['match'] + counts['mismatch']
    rate = counts['match'] / total * 100 if total > 0 else 0
    print(f'{cls}: {counts[\"match\"]}/{total} matched ({rate:.1f}%)')
"
```

**Validation Checkpoint:**
- [ ] Identify which perturbation types have low class match rates
- [ ] Note: Are coarse-grained specifically failing validation?
- [ ] Note: What does validator say is wrong?

---

### Step 1.4: Compare Coarse-Grained Types

**Goal:** Check if specific coarse-grained types (wrong_plan vs false_terminal vs premature_termination) perform differently.

```bash
python -c "
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client['agent_judge_experiment']

# Get coarse-grained by type
pipeline = [
    {'\\$match': {'experiment_id': 'dual_mode_v3', 'perturbation.class': 'coarse_grained'}},
    {'\\$group': {
        '_id': '\\$perturbation.type',
        'count': {'\\$sum': 1},
        'mutation_methods': {'\\$addToSet': '\\$perturbation.mutation_method'}
    }}
]
results = list(db['perturbed_trajectories'].aggregate(pipeline))

print('Coarse-grained breakdown by type:')
for r in results:
    print(f'  {r[\"_id\"]}: {r[\"count\"]} (methods: {r[\"mutation_methods\"]})')
"
```

**Validation Checkpoint:**
- [ ] See distribution of coarse-grained types
- [ ] Note: Is wrong_plan dominant or balanced?

---

## Phase 2: Hypothesis Testing

Based on Phase 1 findings, test one of these hypotheses:

### Hypothesis A: Wrong_plan perturbations are not task-blocking enough

**Test:** Review 5 wrong_plan perturbations manually. Score each:
- 0 = Looks like it would succeed
- 1 = Looks suboptimal but might succeed
- 2 = Clearly task-blocking (would fail)

If average < 1.5, the prompt needs to be more aggressive.

### Hypothesis B: Judge detects errors but doesn't lower score enough

**Test:** Check if judge mentions plan issues in reasoning but still scores >80.

If true, the issue is judge calibration, not perturbation quality.

### Hypothesis C: Sample size too small

**Test:** Check how many coarse-grained units have clean baselines.

If < 10 samples, results may not be statistically meaningful.

---

## Phase 3: Fixes

### Fix Option 1: Improve WRONG_PLAN_PROMPT_V3

If Hypothesis A is true, create a more aggressive prompt.

**File:** `src/prompts/perturbation_prompts.py`

```python
WRONG_PLAN_PROMPT_V3 = """You are generating test data for AI judge evaluation. 
Create a WRONG plan that makes the task IMPOSSIBLE to complete.

TASK: {task_description}

ORIGINAL CORRECT PLAN:
{original_plan}

Create a plan that will DEFINITELY FAIL. Choose ONE:

1. COMPLETELY WRONG GOAL (most severe)
   - Plan for a DIFFERENT task entirely
   - Example: Task is "find Italian restaurants" -> Plan searches for "Italian language courses"
   
2. CRITICAL MISSING STEP (severe)
   - Remove a step without which the task CANNOT complete
   - Example: Remove authentication before API call -> API will reject
   
3. WRONG DATA SOURCE (severe)
   - Use completely wrong tool/API for the data needed
   - Example: Use weather API when stock prices needed

REQUIREMENTS:
- The plan MUST be broken - not just slow or inefficient
- An expert reviewing this should say "this will not work"
- Keep 3-5 steps, similar format to original
- Do NOT explain why it's wrong - just output the bad plan

Output ONLY the wrong plan text, nothing else."""
```

**Update config:**
```json
"coarse_grained": {
  "prompts": {
    "wrong_plan": "WRONG_PLAN_PROMPT_V3"
  }
}
```

---

### Fix Option 2: Add More Coarse-Grained Types

If wrong_plan alone isn't working, emphasize other coarse-grained types.

**File:** `config/experiments/v3/poc/dual_mode_v4.json`

```json
"coarse_grained": {
  "types": ["wrong_tool_family", "premature_termination", "missing_critical_step"],
  "type_weights": {
    "wrong_tool_family": 0.4,
    "premature_termination": 0.3,
    "missing_critical_step": 0.3
  },
  "llm_required": []
}
```

These rule-based mutations may be more reliable than LLM-generated wrong_plan.

---

### Fix Option 3: Lower Detection Threshold

If Hypothesis B is true (judge detects but scores high), adjust scoring.

**Current:** Detection = score < 80 (baseline) or error_trajectory = perturbed  
**Option:** Detection = score < 90 for single_trajectory mode

**File:** `main.py` compute phase

```python
# For single_trajectory, use higher threshold
if version == 'v1_single_trajectory':
    detected = score < 90  # More sensitive threshold
else:
    detected = error_trajectory == perturbed_position
```

---

## Phase 4: Run New Experiment

### Step 4.1: Create dual_mode_v4.json

Apply chosen fix(es) from Phase 3.

```bash
cp config/experiments/v3/poc/dual_mode_v3.json config/experiments/v3/poc/dual_mode_v4.json
# Edit with fixes
```

### Step 4.2: Run All Phases

```bash
python main.py --config v3/poc/dual_mode_v4 --phase load
python main.py --config v3/poc/dual_mode_v4 --phase perturb
python main.py --config v3/poc/dual_mode_v4 --phase judge
python main.py --config v3/poc/dual_mode_v4 --phase compute
```

### Step 4.3: Validate Results

**Success Criteria:**

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Placebo FP (clean) | < 10% | Compute output |
| Fine-grained PDR | > Placebo + 10pts | Compute output |
| Coarse-grained PDR | > Fine-grained | Compute output |
| Class match rate | >= 90% | Perturb phase log |

---

## Phase 5: Document Findings

### Step 5.1: Create Summary Report

Save to `results/dual_mode_v4/analysis.md`:

```markdown
# PDR Hierarchy Analysis - dual_mode_v4

## Investigation Findings
- [Summary of Phase 1 findings]

## Root Cause
- [Which hypothesis was correct]

## Fix Applied
- [What was changed]

## Results
| Mode | Placebo FP | Fine | Coarse |
|------|------------|------|--------|
| ... | ... | ... | ... |

## Hierarchy Status
- [ ] Placebo < Fine
- [ ] Fine < Coarse
```

### Step 5.2: Notify Completion

```bash
osascript -e 'display notification "dual_mode_v4 complete - check results" with title "Claude Code" sound name "default"'
```

---

## Appendix: Quick Reference

### MongoDB Queries

```bash
# Count by experiment
db.perturbed_trajectories.countDocuments({experiment_id: 'dual_mode_v4'})

# Get judge outputs with low scores
db.judge_outputs.find({experiment_id: 'dual_mode_v4', score: {$lt: 80}})

# Get clean baseline units
db.evaluation_units.find({
  experiment_id: 'dual_mode_v4',
  'baseline_verification.is_clean': true
})
```

### File Locations

| Purpose | Path |
|---------|------|
| Config | `config/experiments/v3/poc/dual_mode_v4.json` |
| Prompts | `src/prompts/perturbation_prompts.py` |
| Generator | `src/perturbations/coarse_grained.py` |
| Compute | `main.py` (compute phase) |
| Logs | `logs/dual_mode_v4.log` |

### Key Metrics Definitions

- **PDR (Perturbation Detection Rate):** % of perturbations where judge correctly identified the error
- **Placebo FP:** % of placebos incorrectly flagged as errors (false positives)
- **Clean Baseline:** Baseline trajectory with no pre-existing errors (baseline_verify score >= threshold)
- **Class Match:** Perturbation matches its claimed class per LLM validator
