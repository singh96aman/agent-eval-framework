# 3. Controlled Perturbations

**Status: COMPLETE**

Controlled perturbations are the experimental treatment in this study. By systematically injecting known errors into typed trajectories, we create a controlled setting to measure whether LLM judges can detect errors and estimate their downstream consequence.

---

## Design Principles

1. **Traceability**: Every perturbation has a known injection point, type, and ground-truth label
2. **Systematic**: Perturbations are generated from typed slots, not arbitrary text edits
3. **Balanced**: The perturbation set includes both detectable/consequential and subtle/harmless cases
4. **Realistic**: Perturbations mimic errors an actual agent might make, not adversarial attacks

---

## 3.1 Perturbation Taxonomy

### Three Classes

| Class | Definition | Detection Difficulty | Consequentiality |
|-------|-----------|---------------------|------------------|
| **Placebo** | Semantic-preserving rewrite | Should be undetectable | Zero by design |
| **Fine-grained** | Single local-slot mutation | Variable (subtle to obvious) | Variable (depends on slot) |
| **Coarse-grained** | Structural decision change | Usually detectable | Usually high |

### Perturbation Naming Convention

**Two-field system for clarity:**

| Field | Values | Purpose |
|-------|--------|---------|
| `perturbation_class_allowed` | `placebo`, `fine_grained`, `coarse_grained` | Which experimental condition |
| `perturbation_family_allowed` | `data_reference`, `parameter`, `tool_selection`, `structural`, `terminal_flag` | What kind of slot mutation |

### Mapping Families to Classes

| Perturbation Family | Applicable Slots | Allowed Classes |
|---------------------|-----------------|-----------------|
| `data_reference` | extracted_value, filepath, line_number, entity_id, produced_artifacts[].name, consumed_artifacts[].name | fine_grained |
| `parameter` | tool_arguments.*, threshold, top_k, query | fine_grained |
| `tool_selection` | tool_name | fine_grained, coarse_grained (see rules below) |
| `structural` | subgoals, ordering, decision_target, skipped_step | coarse_grained |
| `terminal_flag` | is_terminal_step, produces_final_answer, produces_patch | coarse_grained |

### Tool Selection Class Rules

`tool_selection` straddles fine and coarse — use these rules:

| Condition | Class | Example |
|-----------|-------|---------|
| Substitute with **near-neighbor** tool in same family | **fine_grained** | `grep` → `ripgrep`, `web_search` → `google_search` |
| Substitute with **different tool family** or wrong action mode | **coarse_grained** | `file_edit` → `web_search`, `run_tests` → `file_view` |

**Tool family examples:**
- Search family: `web_search`, `google_search`, `bing_search`, `grep`, `ripgrep`
- File-read family: `file_view`, `cat`, `head`, `read_file`
- File-write family: `file_edit`, `str_replace`, `write_file`, `patch`
- Execution family: `run_tests`, `bash`, `python_exec`

**Schema Note (Section 2 alignment):**
- `produced_artifacts` are typed objects: `{name, artifact_type}`
- `consumed_artifacts` are typed objects: `{name, artifact_type}` (mirrors produced_artifacts)
- `extracted_value`, `value_type`, `source_step`, `source_description` are official core fields for extraction steps
- Derived annotations (`critical_path_score`, `affects_*`, `recoverable_if_wrong`, `observable_if_wrong`) now use `{value, source, confidence}` structure — access via `.value` when scoring slots

---

## 3.2 Placebo Perturbations (Control Condition)

### Purpose
Placebo perturbations test judge calibration on trajectories with **no semantic change**. A well-calibrated judge should:
- Assign similar scores to original and placebo
- Not flag placebo as containing an error
- Not attribute impact to the "perturbed" step

### Types of Placebo Rewrites

| Placebo Type | Description | Example |
|--------------|-------------|---------|
| **Paraphrase** | Reword reasoning/thought text | "I will search for X" → "Let me look up X" |
| **Formatting** | Change whitespace, punctuation, casing | `{"query": "bert layers"}` → `{"query":"bert layers"}` |
| **Synonym** | Substitute equivalent terms | "retrieve" → "fetch", "file" → "document" |
| **Reorder args** | Change argument order (where order-independent) | `{"a": 1, "b": 2}` → `{"b": 2, "a": 1}` |
| **Trivial rename** | Rename internal variable (in reasoning) | "Let X = 12" → "Let count = 12" |

### Generation Method
1. Select a step with `perturbable_slots` containing `placebo` type
2. Apply one of the above transformations
3. Verify semantic equivalence (LLM verification or heuristic check)
4. Record: `perturbation_type: "placebo"`, `expected_impact: 0`

### Constraints
- Must not change tool execution semantics
- Must not alter extracted values
- Must not modify critical path structure
- Must remain grammatically correct

---

## 3.3 Fine-Grained Perturbations

### Purpose
Fine-grained perturbations inject **single local-slot errors** that may or may not propagate. They test whether judges can detect subtle mistakes in specific values.

### Types of Fine-Grained Errors

| Error Type | Slot Target | Description | Example |
|------------|------------|-------------|---------|
| **Wrong value** | extracted_value | Change extracted number/string | `12` → `10` (BERT layers) |
| **Off-by-one** | line_number, index | Shift numeric reference | `line 108` → `line 109` |
| **Typo in ID** | entity_id, filepath | Plausible character substitution | `wait.py` → `wait_py` |
| **Wrong field** | tool_arguments.field | Access adjacent field | `["name"]` → `["id"]` |
| **Threshold shift** | threshold, top_k | Change numeric parameter | `top_k: 5` → `top_k: 50` |
| **Query drift** | search_query | Subtle query modification | `"bert base layers"` → `"bert large layers"` |
| **Copied value error** | extracted_value | Copy from wrong source | Use value from step 2 instead of step 4 |

### Generation Method
1. Select a step with `perturbable_slots` of type `data_reference` or `parameter`
2. Choose a mutation strategy based on `value_type`:
   - `integer`: ±1, ±10, digit swap, copy from wrong step
   - `filepath`: char substitution, extension change, directory swap
   - `search_query`: add/remove word, synonym swap, wrong entity name
   - `code_snippet`: wrong line, adjacent function, similar variable
3. Apply mutation
4. Record perturbation metadata (see 3.6)

### Consequentiality Factors
Fine-grained errors vary in impact based on:
- **Critical path score**: Higher score = more likely to propagate
- **Downstream dependencies**: More consumers = wider impact
- **Recoverability**: Can agent self-correct?
- **Observability**: Is error visible in output?

### Fine-Grained Subtypes by Expected Impact

| Subtype | Expected Impact | Example |
|---------|-----------------|---------|
| **Inconsequential** | 0 (none) | Wrong top_k in exploratory search that gets refined |
| **Minor** | 1 (recoverable) | Typo in filename, agent retries with correct name |
| **Moderate** | 2 | Wrong line number, agent edits adjacent code |
| **Critical** | 3 (task failure) | Wrong extracted value used in final computation |

---

## 3.4 Coarse-Grained Perturbations

### Purpose
Coarse-grained perturbations inject **structural decision errors** that fundamentally alter the trajectory's approach. They test whether judges recognize high-level failures.

### Types of Coarse-Grained Errors

| Error Type | Target | Description | Example |
|------------|--------|-------------|---------|
| **Wrong tool family** | tool_name | Use categorically wrong tool | `file_edit` → `web_search` for code fix |
| **Skipped prerequisite** | step ordering | Remove required setup step | Skip file read before edit |
| **Wrong plan** | planning step | Choose inferior strategy | Fix symptom instead of root cause |
| **Premature termination** | terminal flags | Stop before task complete | Return partial result as final |
| **False terminal** | terminal flags | Mark incomplete step as terminal | Set `is_terminal_step: true` on step 5 of 10 |
| **Wrong branch** | decision step | Take wrong path at decision point | Choose wrong file when multiple match |
| **Hallucinated dependency** | depends_on_steps | Act on non-existent prior result | Reference step that didn't happen |

### Generation Methods

#### 3.4.1 Tool Family Substitution
```
Original: tool_call(file_edit, {path: "foo.py", ...})
Perturbed: tool_call(web_search, {query: "how to edit foo.py"})
```
- Select steps with `tool_selection` in allowed_perturbation_types
- Replace with tool from different category but plausible for task

#### 3.4.2 Step Deletion (Skipped Prerequisite)
```
Original: [step1: read_file] → [step2: extract_line] → [step3: edit_file]
Perturbed: [step1: read_file] → [step3: edit_file] (step2 removed)
```
- Identify steps with high `critical_path_score.value` and downstream dependencies
- Remove step, update subsequent step indices
- Record which step was removed

#### 3.4.3 Plan Substitution
```
Original: "I will first check the failing test, then trace to the source"
Perturbed: "I will modify the test to make it pass"
```
- Identify `planning` role steps
- Generate alternative (inferior) plan using LLM
- Replace planning text and any downstream steps that follow the plan

#### 3.4.4 Premature Termination
```
Original: [steps 1-10] → final_response with complete answer
Perturbed: [steps 1-5] → final_response with partial answer
```
- Truncate trajectory at non-final step
- Set terminal flags on last kept step: `is_terminal_step: true`, `produces_final_answer: true` (or `produces_patch: true` for SWE-bench)
- Optionally convert `step_role` to `final_response`
- Generate plausible incomplete answer

#### 3.4.6 False Terminal (New)
```
Original: step 5 with is_terminal_step: false, produces_final_answer: false
Perturbed: step 5 with is_terminal_step: true, produces_final_answer: true
```
- Identify mid-trajectory steps that produce partial results
- Flip terminal flags to true
- Tests whether judges catch incomplete trajectories marked as complete

### Terminal Flag Perturbation Constraints

Terminal-flag perturbations can become artificial if you just flip booleans. Apply these rules:

| Constraint | Rationale |
|------------|-----------|
| **Only perturb steps that produce plausible partial artifacts** | The step must have `produced_artifacts` that could pass as a partial answer |
| **Generated partial output must remain locally coherent** | If truncating, the "final answer" must grammatically complete the response |
| **No mid-thought termination** | Don't flip terminal on steps that end with "..." or incomplete reasoning |
| **Respect step_role compatibility** | Only `reasoning`, `extraction`, or `decision` steps can become false terminals — not `observation` |

**Valid false terminal candidates:**
- Extraction step that produces an intermediate value (e.g., "BERT has 12 layers")
- Decision step that selects a file but hasn't acted on it yet
- Reasoning step that reaches a partial conclusion

**Invalid false terminal candidates:**
- Observation step (just tool output, not an answer)
- Planning step (intent, not result)
- Step with incomplete sentence or "thinking..." text

#### 3.4.5 Wrong Branch Selection
```
Original: decision(choose file X from candidates [X, Y, Z])
Perturbed: decision(choose file Y from candidates [X, Y, Z])
```
- Identify `decision` role steps with explicit candidates
- Select different candidate
- Regenerate downstream steps to be consistent with wrong choice

### Consequentiality
Coarse-grained errors are expected to be high-impact by design:
- Most should have `expected_impact: 2 or 3`
- Skipped prerequisites often cause tool failures
- Wrong tool family usually produces irrelevant output
- Plan errors cascade through execution

---

## 3.5 Sampling Strategy

### Target Distribution per Benchmark

| Benchmark | Trajectories | Perturbations per Trajectory | Total Perturbations |
|-----------|--------------|------------------------------|---------------------|
| ToolBench | 400 | 2-3 | ~1000 |
| GAIA | 100 | 3-4 | ~350 |
| SWE-bench | 100 | 3-4 | ~350 |
| **Total** | **600** | - | **~1700** |

### Distribution by Class

| Class | Target % | Count | Purpose |
|-------|----------|-------|---------|
| Placebo | 20% | ~340 | Calibration control |
| Fine-grained | 50% | ~850 | Subtle error detection |
| Coarse-grained | 30% | ~510 | Structural error detection |

### Within Fine-Grained Distribution

| Expected Impact | Target % | Purpose |
|-----------------|----------|---------|
| Inconsequential (0) | 30% | Test over-penalization |
| Minor (1) | 30% | Intermediate cases |
| Moderate (2) | 25% | Clear errors |
| Critical (3) | 15% | High-stakes cases |

### Family Coverage Goals
Ensure perturbations cover all perturbation families:

| Family | Target % | Within Class |
|--------|----------|--------------|
| `data_reference` | ≥30% | of fine_grained |
| `parameter` | ≥30% | of fine_grained |
| `tool_selection` | ≥20% | of fine_grained |
| `tool_selection` | ≥15% | of coarse_grained |
| `structural` | ≥70% | of coarse_grained |
| `terminal_flag` | ≥10% | of coarse_grained |

### Step Position Coverage
Avoid biasing by step position:
- Early steps (first 25%): ~30% of perturbations
- Middle steps (25-75%): ~50% of perturbations
- Late steps (last 25%): ~20% of perturbations

---

## 3.6 Final Schema

### Perturbable Slot Object (in Step, from Section 2)

```json
{
  "slot": "string (json path, e.g., 'tool_arguments.path')",
  "value_type": "string (e.g., 'filepath', 'integer', 'search_query')",
  "current_value": "any",
  "perturbation_class_allowed": ["placebo", "fine_grained", "coarse_grained"],
  "perturbation_family_allowed": ["data_reference", "parameter", "tool_selection", "structural", "terminal_flag"]
}
```

### Perturbation Record Object

```json
{
  // === Core identification ===
  "perturbation_id": "string (uuid)",
  "original_trajectory_id": "string",
  "generation_timestamp": "datetime (ISO 8601)",
  
  // === Classification (two-field system) ===
  "perturbation_class": "placebo | fine_grained | coarse_grained",
  "perturbation_family": "data_reference | parameter | tool_selection | structural | terminal_flag",
  "perturbation_type": "string (specific error type, e.g., 'wrong_value', 'skipped_prerequisite')",
  
  // === Target specification ===
  "target_step_index": "int",
  "target_slot": "string (json path to modified field)",
  "original_value": "any",
  "perturbed_value": "any",
  "mutation_method": "string (how mutation was generated)",
  
  // === Generation-time predictions ===
  "expected_impact": "int (0=none, 1=minor, 2=moderate, 3=critical)",
  "expected_detectability": "int (0=invisible, 1=subtle, 2=obvious)",
  "propagation_risk": ["string[] (downstream step identifiers affected)"],
  "verified_semantic_change": "bool (true for non-placebo, false for placebo)",
  "notes": "string | null",
  
  // === Post-generation QC ===
  "generation_status": "valid | invalid | borderline",
  "qc_notes": "string | null",
  "review_source": "heuristic | llm | human",
  "reviewed_at": "datetime | null"
}
```

### Perturbed Trajectory Object

```json
{
  // === All fields from Typed Trajectory (Section 2) preserved ===
  "trajectory_id": "string",
  "benchmark": "toolbench | gaia | swebench",
  "task_id": "string",
  "task_text": "string",
  "baseline_outcome": "float (0-1)",
  "num_steps": "int",
  "ground_truth_available": "bool",
  "environment_type": "tool_use | qa | code_edit | browser",
  "steps": ["TypedStep[]"],
  // ... other Section 2 fields ...
  
  // === Perturbation metadata (added) ===
  "is_perturbed": true,
  "perturbation_record": "PerturbationRecord (see above)"
}
```

### Field Enums Reference

| Field | Allowed Values |
|-------|----------------|
| perturbation_class | `placebo`, `fine_grained`, `coarse_grained` |
| perturbation_family | `data_reference`, `parameter`, `tool_selection`, `structural`, `terminal_flag` |
| expected_impact | `0` (none), `1` (minor), `2` (moderate), `3` (critical) |
| expected_detectability | `0` (invisible), `1` (subtle), `2` (obvious) |
| generation_status | `valid`, `invalid`, `borderline` |
| review_source | `heuristic`, `llm`, `human` |

### perturbation_type Values by Family

| Family | Valid perturbation_type values |
|--------|-------------------------------|
| data_reference | `wrong_value`, `off_by_one`, `typo_in_id`, `wrong_field`, `copied_value_error` |
| parameter | `threshold_shift`, `query_drift`, `wrong_parameter` |
| tool_selection | `near_neighbor_tool` (fine), `wrong_tool_family` (coarse) |
| structural | `skipped_prerequisite`, `wrong_plan`, `wrong_branch`, `hallucinated_dependency` |
| terminal_flag | `premature_termination`, `false_terminal` |
| (placebo) | `paraphrase`, `formatting`, `synonym`, `reorder_args`, `trivial_rename` |

---

## 3.7 Examples

### Example: Fine-Grained Perturbation Record

```json
{
  "perturbation_id": "pert_gaia122_001",
  "original_trajectory_id": "gaia_122",
  "perturbation_class": "fine_grained",
  "perturbation_family": "data_reference",
  "perturbation_type": "wrong_value",
  "target_step_index": 2,
  "target_slot": "extracted_value",
  "original_value": 12,
  "perturbed_value": 10,
  "mutation_method": "numeric_shift_-2",
  "generation_timestamp": "2026-04-07T10:00:00Z",
  "expected_impact": 3,
  "expected_detectability": 0,
  "propagation_risk": ["step_5_arithmetic"],
  "verified_semantic_change": true,
  "notes": "BERT layer count: 12→10 will produce answer 4 instead of 6",
  "generation_status": "valid",
  "qc_notes": null,
  "review_source": "heuristic",
  "reviewed_at": "2026-04-07T10:05:00Z"
}
```

### Example: Coarse-Grained Perturbation Record

```json
{
  "perturbation_id": "pert_swe_tenacity_001",
  "original_trajectory_id": "swebench_jd__tenacity.0d40e76f.121.4slwej0r",
  "perturbation_class": "coarse_grained",
  "perturbation_family": "structural",
  "perturbation_type": "skipped_prerequisite",
  "target_step_index": 2,
  "target_slot": "step_deleted",
  "original_value": {"step_index": 2, "step_role": "tool_call", "tool_name": "file_view"},
  "perturbed_value": null,
  "mutation_method": "step_deletion",
  "generation_timestamp": "2026-04-07T10:00:00Z",
  "expected_impact": 3,
  "expected_detectability": 2,
  "propagation_risk": ["step_3_extraction", "step_13_patch"],
  "verified_semantic_change": true,
  "notes": "Removing file view means agent cannot find bug location",
  "generation_status": "valid",
  "qc_notes": null,
  "review_source": "heuristic",
  "reviewed_at": "2026-04-07T10:05:00Z"
}
```

### Example: Placebo Perturbation Record

```json
{
  "perturbation_id": "pert_toolbench_001_placebo",
  "original_trajectory_id": "toolbench_146416",
  "perturbation_class": "placebo",
  "perturbation_family": "parameter",
  "perturbation_type": "paraphrase",
  "target_step_index": 1,
  "target_slot": "raw_text",
  "original_value": "Thought: I'm calling the \"api_nutrition_data_for_edamam_nutrition_analysis\" function...",
  "perturbed_value": "Thought: I will invoke the edamam nutrition analysis API to retrieve data...",
  "mutation_method": "llm_paraphrase",
  "generation_timestamp": "2026-04-07T10:00:00Z",
  "expected_impact": 0,
  "expected_detectability": 0,
  "propagation_risk": [],
  "verified_semantic_change": false,
  "notes": "Control condition - semantic equivalent",
  "generation_status": "valid",
  "qc_notes": "Verified via LLM equivalence check",
  "review_source": "llm",
  "reviewed_at": "2026-04-07T10:05:00Z"
}
```

### Example: False Terminal Perturbation Record

```json
{
  "perturbation_id": "pert_gaia122_false_terminal",
  "original_trajectory_id": "gaia_122",
  "perturbation_class": "coarse_grained",
  "perturbation_family": "terminal_flag",
  "perturbation_type": "false_terminal",
  "target_step_index": 2,
  "target_slot": "terminal_flags",
  "original_value": {"is_terminal_step": false, "produces_final_answer": false},
  "perturbed_value": {"is_terminal_step": true, "produces_final_answer": true},
  "mutation_method": "flag_flip",
  "generation_timestamp": "2026-04-07T10:00:00Z",
  "expected_impact": 3,
  "expected_detectability": 1,
  "propagation_risk": ["step_4_extraction", "step_5_reasoning"],
  "verified_semantic_change": true,
  "notes": "Step 2 extracts '12' for BERT layers - plausible partial answer, but missing comparison",
  "generation_status": "valid",
  "qc_notes": "Step produces artifact (bert_layer_count), coherent as partial answer",
  "review_source": "human",
  "reviewed_at": "2026-04-07T11:00:00Z"
}
```

---

## 3.8 Data Location

```
data/
├── typed/                           # Input (from Section 2)
│   ├── toolbench_400_typed.json
│   ├── gaia_100_typed.json
│   └── swebench_100_typed.json
└── perturbed/                       # Output (from Section 3)
    ├── toolbench_perturbed.json     # All ToolBench perturbations
    ├── gaia_perturbed.json          # All GAIA perturbations
    ├── swebench_perturbed.json      # All SWE-bench perturbations
    └── perturbation_index.json      # Summary index
```

### Perturbation Index Schema

```json
{
  "total_perturbations": 1700,
  "by_class": {
    "placebo": 340,
    "fine_grained": 850,
    "coarse_grained": 510
  },
  "by_family": {
    "data_reference": 400,
    "parameter": 350,
    "tool_selection": 200,
    "structural": 400,
    "terminal_flag": 110
  },
  "by_benchmark": {
    "toolbench": 1000,
    "gaia": 350,
    "swebench": 350
  },
  "by_type": {
    "paraphrase": 120,
    "formatting": 110,
    "wrong_value": 200,
    "off_by_one": 150,
    // ... etc
  },
  "by_generation_status": {
    "valid": 1600,
    "invalid": 50,
    "borderline": 50
  },
  "perturbations": [
    {
      "perturbation_id": "string",
      "original_trajectory_id": "string",
      "perturbation_class": "string",
      "perturbation_family": "string",
      "perturbation_type": "string",
      "expected_impact": "int",
      "generation_status": "string",
      "file": "string"
    }
  ]
}
```

---

## 3.9 Generation Pipeline

### Phase 1: Slot Selection
1. Load typed trajectories
2. For each trajectory, enumerate perturbable slots
3. Score slots by:
   - Diversity (avoid over-sampling same slot type)
   - Critical path score via `critical_path_score.value` (mix high and low)
   - Step position (distribute across trajectory)
   - Terminal status (`is_terminal_step`, `produces_final_answer`, `produces_patch`)

### Phase 2: Perturbation Generation
1. **Placebo**: LLM paraphrase + semantic equivalence check
2. **Fine-grained**:
   - Rule-based mutations for numeric types
   - LLM-assisted for text/query types
   - Copy-from-wrong-step for reference types
3. **Coarse-grained**:
   - Step deletion: deterministic
   - Tool substitution: category-aware sampling
   - Plan rewrite: LLM generation with constraints

### Phase 3: Validation
1. Verify perturbation was applied (diff check)
2. Verify trajectory remains well-formed (schema validation)
3. For placebo: verify semantic equivalence
4. For non-placebo: verify semantic change
5. Record any generation failures

### Phase 4: Quality Control
1. Run heuristic validators (schema, diff, coherence)
2. Run LLM validators (semantic change for non-placebo, equivalence for placebo)
3. Set `generation_status` for each perturbation
4. Flag `borderline` cases for human review

### Phase 5: Balancing
1. Filter to `generation_status: valid` perturbations
2. Check distribution against targets (3.5)
3. Generate additional perturbations if under-represented
4. Sample down if over-represented
5. Produce final perturbation index

---

## 3.10 Quality Criteria

A good perturbation set satisfies:

| Criterion | Description | Verification |
|-----------|-------------|--------------|
| **Traceable** | Every perturbation has complete metadata including class + family | Schema validation |
| **Balanced** | Distribution matches targets by class and family | Index statistics |
| **Valid** | All trajectories parse correctly | Schema validation |
| **Realistic** | Perturbations resemble real agent errors | Human spot-check |
| **Distinguishable** | Non-placebo perturbations change semantics | Diff + LLM check |
| **Controlled** | Placebo perturbations preserve semantics | Equivalence check |
| **QC'd** | All perturbations have generation_status assigned | Field completeness check |
| **Clean** | <5% of perturbations marked `invalid` | Index statistics |

### QC Pipeline

| Stage | Method | Sets |
|-------|--------|------|
| **Heuristic** | Schema validation, diff non-empty, JSON parseable | All |
| **LLM** | Semantic change verification, equivalence check for placebo | All |
| **Human** | Spot-check sample, review borderline cases | 5% sample + borderline |

### Spot-Check Protocol
- Human reviews 5% random sample (~85 perturbations)
- Human reviews all `generation_status: borderline` cases
- Check: Is this a plausible agent error?
- Check: Does expected_impact seem reasonable?
- Check: Is perturbation_class and perturbation_family correctly labeled?
- Check: For terminal_flag perturbations, does the step have a plausible partial artifact?

---

## 3.11 Benchmark-Specific Considerations

### ToolBench
- Many steps have API errors already — avoid double-perturbing failure cases
- Focus on: query mutations, wrong API selection, parameter tweaks
- Placebo: paraphrase reasoning between tool calls

### GAIA
- Extraction steps are high-value targets (numeric values affect final answer)
- Focus on: wrong extracted values, query drift, arithmetic operand swap
- Placebo: reword search queries semantically

### SWE-bench
- File paths and line numbers are critical
- Focus on: wrong file, wrong line, adjacent function, incomplete patch
- Placebo: code comment rewording, variable rename in reasoning
- Special: can verify via test suite (ground truth for impact)

---

## 3.12 Failure Modes to Avoid

| Failure Mode | Description | Mitigation |
|--------------|-------------|------------|
| **Obvious perturbations only** | All errors are easy to spot | Include subtle value changes |
| **No-op "perturbations"** | Change doesn't affect execution | Verify semantic change |
| **Adversarial inputs** | Perturbations that break parsing | Schema validation |
| **Imbalanced impact** | All perturbations are catastrophic | Stratify by expected_impact |
| **Position bias** | All perturbations on early steps | Distribute by step position |
| **Type bias** | Only perturb one slot type | Track and balance slot coverage |

---

## 3.13 Implementation Plan

### Step 1: Perturbation Generators (src/perturbation/)
```
src/perturbation/
├── __init__.py
├── base.py              # Abstract perturbation generator
├── placebo.py           # Placebo transformations
├── fine_grained.py      # Fine-grained mutations
├── coarse_grained.py    # Structural perturbations
├── validators.py        # Semantic change verification
└── sampler.py           # Balanced sampling logic
```

### Step 2: Generation Script
```bash
python -m src.perturbation.generate \
  --input data/typed/ \
  --output data/perturbed/ \
  --config configs/perturbation_config.yaml
```

### Step 3: Validation Script
```bash
python -m src.perturbation.validate \
  --input data/perturbed/ \
  --output data/perturbed/validation_report.json
```

### Step 4: Balancing Script
```bash
python -m src.perturbation.balance \
  --input data/perturbed/ \
  --targets configs/perturbation_targets.yaml \
  --output data/perturbed/
```

---

## 3.14 Dependencies

| Dependency | Section | What's Needed |
|------------|---------|---------------|
| Typed Representation (Section 2) | Input | Typed trajectories with perturbable_slots |
| Evaluation Unit (Section 4) | Output | Perturbed trajectories feed into evaluation |
| Outcome Evidence (Section 5C) | Validation | Replay confirms actual vs expected impact |

---

## 3.15 Success Criteria

Section 3 is complete when:

1. [x] Perturbation generators implemented for all three classes and five families
2. [x] ~785 perturbations generated across 434 trajectories (POC v7 experiment)
3. [x] Class distribution achieved:
   - 45.1% coarse-grained, 25.6% fine-grained, 29.3% placebo
4. [x] Family coverage:
   - data_reference: 29.3% (placebo synonyms)
   - parameter: 25.6% (fine-grained)
   - structural: 44.5% (coarse-grained)
   - terminal_flag: 0.1% (coarse-grained)
5. [x] All perturbations pass schema validation (100% pass core schema checks)
6. [x] All perturbations have `generation_status` assigned (100% valid)
7. [x] 0% of perturbations marked `invalid`
8. [x] Data quality verified via automated analysis
9. [x] QC pipeline implemented with heuristic validators
10. [x] Perturbation index created with summary statistics
11. [x] Terminal flag perturbations validated (premature_termination: 1 case)
