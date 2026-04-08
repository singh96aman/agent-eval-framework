# 2. Typed Representation

**Status: REQUIREMENTS DEFINED**

The typed representation is the foundation for controlled perturbation. It structures each trajectory step so the system knows what kind of step it is, what objects it touches, and what later steps depend on it.

---

## Storage Decision

**Approach: Enrich existing trajectories** (not a separate collection)

- Typed fields are added directly to the trajectory JSON
- Output location: `data/typed/{benchmark}_{count}_typed.json`
- Raw fields preserved, enrichment fields added alongside

```
data/
├── sampled/                    # Raw trajectories (current)
│   ├── toolbench_400.json
│   ├── gaia_100.json
│   └── swebench_100.json
└── typed/                      # Enriched trajectories (output)
    ├── toolbench_400_typed.json
    ├── gaia_100_typed.json
    └── swebench_100_typed.json
```

---

## Final Schema

### Trajectory Object

```json
{
  // === Identity (from raw) ===
  "trajectory_id": "string",
  "benchmark": "toolbench|gaia|swebench",
  
  // === Task (from raw, normalized) ===
  "task_id": "string",
  "task_text": "string",
  "expected_answer": "string|null",
  
  // === Metadata (from raw) ===
  "domain": "string|null",
  "difficulty": "string|null",
  
  // === Enrichment (added by typing pass) ===
  "num_steps": "int",
  "environment_type": "tool_use|qa|code_edit|browser",
  "ground_truth_available": "bool",
  "baseline_outcome": "float (0-1)",
  "has_objective_verifier": "bool",
  "can_replay": "bool",
  "can_regenerate_downstream": "bool",
  
  // === Steps ===
  "steps": ["Step[]"]
}
```

### Step Object — Core Fields (Pass 1)

Core fields are structural: they describe what the step is and what it references. These can be filled by parsing or simple heuristics.

```json
{
  // === Identity (from raw, renamed) ===
  "step_index": "int",
  "raw_text": "string",
  
  // === Core Typing ===
  "step_role": "planning|tool_call|observation|reasoning|extraction|decision|final_response",
  
  // === Terminal Flags ===
  "is_terminal_step": "bool",
  "produces_final_answer": "bool",
  "produces_patch": "bool",
  
  // === Tool Fields (from raw + normalized) ===
  "tool_name": "string|null",
  "tool_arguments": "object|null",
  "observation": "string|null",
  
  // === Extraction Fields (for extraction steps) ===
  "extracted_value": "any|null",
  "value_type": "string|null",
  "source_step": "int|null",
  "source_description": "string|null",
  
  // === Dependency Graph ===
  "depends_on_steps": ["int[]"],
  "dependency_edges": [
    {
      "type": "uses_observation_from|uses_extracted_value_from|follows_plan_from|acts_on_entity_from",
      "source_step": "int",
      "reason": "string"
    }
  ],
  
  // === Artifacts (typed) ===
  "entities": ["string[]"],
  "produced_artifacts": [
    {
      "name": "string",
      "artifact_type": "search_results|filepath|patch|line_number|numeric_answer|code_snippet|diagnosis|api_response|text"
    }
  ],
  "consumed_artifacts": ["string[]"],
  
  // === Perturbation Support ===
  "perturbable_slots": [
    {
      "slot": "string (json path)",
      "value_type": "string",
      "current_value": "any",
      "allowed_perturbation_types": ["placebo|data_reference|parameter|tool_selection|structural"]
    }
  ]
}
```

### Step Object — Derived Annotations (Pass 2/3)

Derived fields are interpretive: they estimate impact, recoverability, and criticality. These are filled by heuristics or LLM inference and should carry provenance.

```json
{
  // === Criticality (Pass 2/3) ===
  "critical_path_score": {
    "value": "float (0-1)",
    "source": "heuristic|llm|human",
    "confidence": "float (0-1)|null"
  },
  
  // === Outcome Linkage (Pass 2/3) ===
  "affects_final_answer": {
    "value": "bool",
    "source": "heuristic|llm|human",
    "confidence": "float (0-1)|null"
  },
  "affects_patch": {
    "value": "bool",
    "source": "heuristic|llm|human",
    "confidence": "float (0-1)|null"
  },
  "affects_tool_execution": {
    "value": "bool",
    "source": "heuristic|llm|human",
    "confidence": "float (0-1)|null"
  },
  
  // === Recoverability (Pass 2/3) ===
  "recoverable_if_wrong": {
    "value": "bool",
    "source": "heuristic|llm|human",
    "confidence": "float (0-1)|null"
  },
  "observable_if_wrong": {
    "value": "bool",
    "source": "heuristic|llm|human",
    "confidence": "float (0-1)|null"
  }
}
```

**Provenance Rule:** Any field derived by heuristic or LLM must carry `source` and `confidence`. This lets downstream analysis distinguish high-confidence parser output from low-confidence inference.

### Field Sources Summary

| Category | Fields | Source | Pass |
|----------|--------|--------|------|
| Identity | trajectory_id, benchmark, task_id, task_text | Raw (normalized) | 1 |
| Step Core | step_index, raw_text, tool_name, tool_arguments, observation | Raw (renamed) | 1 |
| Typing | step_role, is_terminal_step, produces_final_answer, produces_patch | Enrichment (parser/heuristic) | 1 |
| Extraction | extracted_value, value_type, source_step, source_description | Enrichment (parser) | 1 |
| Dependencies | depends_on_steps, dependency_edges | Enrichment (analysis) | 1-2 |
| Artifacts | entities, produced_artifacts (typed), consumed_artifacts | Enrichment (extraction) | 1-2 |
| Perturbation | perturbable_slots | Enrichment (derivation) | 2 |
| **Derived** | critical_path_score | Enrichment (heuristic/LLM) | 2-3 |
| **Derived** | affects_*, recoverable_if_wrong, observable_if_wrong | Enrichment (heuristic/LLM) | 2-3 |

---

## 2.1 Trajectory-Level Fields

| Field | Required | Description |
|-------|----------|-------------|
| trajectory_id | Yes | Unique identifier |
| benchmark | Yes | toolbench, gaia, swebench |
| task_id | Yes | Original task identifier |
| task_text | Yes | Task description |
| baseline_outcome | Yes | Baseline outcome score (float 0-1) |
| num_steps | Yes | Step count |
| ground_truth_available | Yes | Boolean flag |
| environment_type | Yes | tool_use, qa, code_edit, browser |
| domain | No | Task domain |
| difficulty | No | Difficulty level |
| has_objective_verifier | No | Whether objective verification is possible |
| can_replay | No | Whether replay is supported |
| can_regenerate_downstream | No | Whether downstream regeneration is possible |

## 2.2 Step-Level Fields

### Core Fields (Pass 1)

| Field | Required | Description |
|-------|----------|-------------|
| step_index | Yes | Position in trajectory |
| raw_text | Yes | Original text |
| step_role | Yes | One of: planning, tool_call, observation, reasoning, extraction, decision, final_response |
| is_terminal_step | Yes | Whether this is the last step in the trajectory |
| produces_final_answer | Yes | Whether this step produces the final answer |
| produces_patch | Yes | Whether this step produces a code patch (SWE-bench) |
| tool_name | Conditional | Required for tool_call steps |
| tool_arguments | Conditional | Required for tool_call steps |
| observation | Conditional | Optional for tool_call steps (if trace includes inline output); required for observation steps |
| extracted_value | Conditional | Required for extraction steps |
| value_type | Conditional | Required for extraction steps |
| source_step | Conditional | Optional for extraction steps (which step the value came from) |
| source_description | Conditional | Optional for extraction steps |
| entities | Yes | Referenced entities |
| depends_on_steps | Yes | List of step indices this step depends on |
| dependency_edges | Yes | Typed causal dependencies |
| produced_artifacts | Yes | What this step produces (typed) |
| consumed_artifacts | Yes | What this step uses |
| perturbable_slots | Yes | What can be mutated and how |

### Derived Annotations (Pass 2-3)

| Field | Required | Description |
|-------|----------|-------------|
| critical_path_score | Pass 2-3 | Likelihood of affecting task success if corrupted (0-1), with provenance |
| affects_final_answer | Pass 2-3 | Whether error impacts final answer, with provenance |
| affects_patch | Pass 2-3 | Whether error impacts code patch, with provenance |
| affects_tool_execution | Pass 2-3 | Whether error impacts tool execution, with provenance |
| recoverable_if_wrong | Pass 2-3 | Whether error can be recovered by later steps, with provenance |
| observable_if_wrong | Pass 2-3 | Whether error would be visible in output, with provenance |

## 2.3 Step Role Definitions

| Role | Description | Perturbation Use |
|------|-------------|------------------|
| **planning** | High-level decomposition or next-step intent | Inject structural perturbations: missing subgoals, wrong plan choice |
| **tool_call** | External action (search, open file, run tests, API call) | Mutate tool selection and parameters |
| **observation** | Result returned by tool or environment | Model stale/incorrect/ignored evidence |
| **reasoning** | Intermediate interpretation (not a tool call) | Study planning/reasoning corruption separately from execution |
| **extraction** | Value copied from evidence into working memory | Fine-grained data-reference errors |
| **decision** | Concrete branch choice (file selection, tool choice, stop) | Coarse perturbations |
| **final_response** | Produced answer or patch | Ties trajectory to outcome grading |

## 2.4 Slot Typing for Perturbation

Each step needs typed slots for systematic perturbation generation.

**Tool Call Slots:**
- tool_name
- arg_name → arg_value
- resource_id, filepath, query, line_number, top_k, threshold

**Extraction Slots** (now part of core schema):
- `extracted_value`: the actual value extracted
- `value_type`: date, float, integer, filepath, identifier, entity_name, code_snippet
- `source_step`: which step the value was extracted from
- `source_description`: human-readable description of where value came from

**Decision Slots:**
- decision_target
- candidate_set
- chosen_option

**Planning Slots:**
- subgoals
- ordering
- termination_condition

**Artifact Types** (for produced_artifacts):
- search_results
- filepath
- patch
- line_number
- numeric_answer
- code_snippet
- diagnosis
- api_response
- text

## 2.5 Dependency Structure

Each step must record dependencies on previous steps.

**Minimum:** `depends_on_steps: [indices]`

**Better (typed edges):**
- uses_observation_from
- uses_extracted_value_from
- follows_plan_from
- acts_on_entity_from

Example: Step 5 depends on Step 2 for selected file path; Step 6 depends on Step 4 for test failure interpretation.

## 2.6 Critical Path Scoring (Derived, Pass 2-3)

Approximate signal for how likely this step is to affect task success if corrupted. This is a **derived annotation** — not available in Pass 1, filled by heuristics or LLM in Pass 2-3.

**Derivation heuristics:**
- Step type (planning > decision > tool_call > extraction > reasoning > observation)
- Dependency fan-out (more downstream consumers = higher score)
- Whether later steps consume its output
- Direct tie to final answer/patch/test execution

**Storage format:**
```json
{
  "critical_path_score": {
    "value": 0.85,
    "source": "heuristic",
    "confidence": null
  }
}
```

## 2.7 Perturbation Metadata

Each step exposes perturbable_slots with allowed_perturbation_types.

```json
{
  "step_index": 4,
  "step_role": "tool_call",
  "tool_name": "open_file",
  "tool_arguments": {"path": "qrcode/image/base.py"},
  "perturbable_slots": [
    {
      "slot": "tool_arguments.path",
      "value_type": "filepath",
      "allowed_perturbation_types": ["data_reference", "parameter"]
    },
    {
      "slot": "tool_name",
      "value_type": "tool",
      "allowed_perturbation_types": ["tool_selection"]
    }
  ]
}
```

## 2.8 Outcome-Linkage Fields (Derived)

| Field | Type | Description |
|-------|------|-------------|
| affects_final_answer | {value: bool, source, confidence} | Whether error impacts answer |
| affects_patch | {value: bool, source, confidence} | Whether error impacts code patch |
| affects_tool_execution | {value: bool, source, confidence} | Whether error impacts tool use |
| recoverable_if_wrong | {value: bool, source, confidence} | Whether error can be recovered |
| observable_if_wrong | {value: bool, source, confidence} | Whether error is visible |

**Provenance fields:** Each derived annotation includes:
- `source`: "heuristic" | "llm" | "human"
- `confidence`: float (0-1) or null if not applicable

These enable analysis categories:
- Visible but harmless
- Invisible but catastrophic
- Recoverable local error
- Irrecoverable structural error

## 2.9 Quality Criteria

A good typed representation satisfies:

1. **Perturbable**: Clearly identifies what can be mutated
2. **Composable**: Works across ToolBench, GAIA, SWE-bench (some fields may be null)
3. **Causal**: Captures enough dependencies for propagation reasoning
4. **Replay-aware**: Stores enough structure for downstream replay/regeneration
5. **Auditable**: Human-inspectable with verifiable step typing

## 2.10 Common Failure Modes to Avoid

- **Too shallow**: Only labeling steps as "thought" or "action" is insufficient
- **Too benchmark-specific**: Schema must work cross-benchmark
- **Too free-form**: All fields as text blobs makes perturbations noisy
- **No dependencies**: Cannot justify claims about downstream effect
- **No typed slots**: Fine-grained vs coarse-grained becomes subjective

## 2.11 Implementation Plan

**Pass 1: Universal Schema**
Build one schema that works for all datasets.

**Pass 2: Benchmark Adapters**
Add dataset-specific parsers:
- ToolBench adapter
- GAIA adapter
- SWE-bench adapter

**Pass 3: Enrichment**
Add:
- Dependency edges
- Critical-path score
- Perturbable slots
- Recoverability/observability tags

---

# Concrete Examples: Raw → Typed Representation

## Example A: ToolBench (toolbench_146416) — Steps 1-2 Typed

**Trajectory-level:**
```json
{
  "trajectory_id": "toolbench_146416",
  "benchmark": "toolbench",
  "task_id": "146416",
  "task_text": "I want to create a personalized diet plan... provide nutritional analysis... guidelines for healthy lifestyle.",
  "baseline_outcome": 0.0,
  "num_steps": 4,
  "ground_truth_available": false,
  "environment_type": "tool_use",
  "domain": "data_information",
  "has_objective_verifier": false
}
```

**Step 1 — Typed (Core Fields):**
```json
{
  "step_index": 1,
  "raw_text": "Thought: I'm calling the \"api_nutrition_data_for_edamam_nutrition_analysis\" function with the argument \"{'ingr': 'oatmeal'}\"...",
  "step_role": "tool_call",
  "is_terminal_step": false,
  "produces_final_answer": false,
  "produces_patch": false,
  "tool_name": "api_nutrition_data_for_edamam_nutrition_analysis",
  "tool_arguments": {"ingr": "oatmeal"},
  "observation": "{\"error\": \"request invalid, data error. status_code=500\"}",
  "entities": ["oatmeal", "edamam_nutrition_analysis"],
  "depends_on_steps": [],
  "dependency_edges": [],
  "produced_artifacts": [
    {"name": "nutrition_query_result_1", "artifact_type": "api_response"}
  ],
  "consumed_artifacts": [],
  "perturbable_slots": [
    {
      "slot": "tool_arguments.ingr",
      "value_type": "food_item",
      "current_value": "oatmeal",
      "allowed_perturbation_types": ["parameter", "data_reference"]
    },
    {
      "slot": "tool_name",
      "value_type": "api_endpoint",
      "current_value": "api_nutrition_data_for_edamam_nutrition_analysis",
      "allowed_perturbation_types": ["tool_selection"]
    }
  ],
  
  // === Derived Annotations (Pass 2-3) ===
  "critical_path_score": {"value": 0.7, "source": "heuristic", "confidence": null},
  "affects_final_answer": {"value": true, "source": "heuristic", "confidence": null},
  "recoverable_if_wrong": {"value": true, "source": "heuristic", "confidence": null},
  "observable_if_wrong": {"value": true, "source": "heuristic", "confidence": null}
}
```

**Step 2 — Typed (Core Fields):**
```json
{
  "step_index": 2,
  "raw_text": "Thought: Based on the previous action and its outcome, it seems that there was an issue with the API request...",
  "step_role": "tool_call",
  "is_terminal_step": false,
  "produces_final_answer": false,
  "produces_patch": false,
  "tool_name": "api_nutrition_data_for_edamam_nutrition_analysis",
  "tool_arguments": {"ingr": "chicken breast"},
  "observation": "{\"error\": \"request invalid, data error. status_code=500\"}",
  "entities": ["chicken breast", "edamam_nutrition_analysis"],
  "depends_on_steps": [1],
  "dependency_edges": [
    {"type": "uses_observation_from", "source_step": 1, "reason": "error triggered retry with different ingredient"}
  ],
  "produced_artifacts": [
    {"name": "nutrition_query_result_2", "artifact_type": "api_response"}
  ],
  "consumed_artifacts": ["nutrition_query_result_1"],
  "perturbable_slots": [
    {
      "slot": "tool_arguments.ingr",
      "value_type": "food_item",
      "current_value": "chicken breast",
      "allowed_perturbation_types": ["parameter", "data_reference"]
    }
  ],
  
  // === Derived Annotations (Pass 2-3) ===
  "critical_path_score": {"value": 0.6, "source": "heuristic", "confidence": null},
  "affects_final_answer": {"value": true, "source": "heuristic", "confidence": null},
  "recoverable_if_wrong": {"value": true, "source": "heuristic", "confidence": null},
  "observable_if_wrong": {"value": true, "source": "heuristic", "confidence": null}
}
```

---

## Example B: GAIA (gaia_122) — Full Trajectory Typed

**Trajectory-level:**
```json
{
  "trajectory_id": "gaia_122",
  "benchmark": "gaia",
  "task_id": "122",
  "task_text": "How many more blocks (also denoted as layers) in BERT base encoder than the encoder from the architecture proposed in Attention is All You Need?",
  "baseline_outcome": 1.0,
  "num_steps": 5,
  "ground_truth_available": true,
  "environment_type": "qa",
  "domain": "qa",
  "difficulty": "1",
  "has_objective_verifier": true
}
```

**Step 1 — tool_call:**
```json
{
  "step_index": 1,
  "raw_text": "Search the internet for \"blocks in bert base\"",
  "step_role": "tool_call",
  "is_terminal_step": false,
  "produces_final_answer": false,
  "produces_patch": false,
  "tool_name": "web_search",
  "tool_arguments": {"query": "blocks in bert base"},
  "entities": ["BERT base", "blocks", "layers"],
  "depends_on_steps": [],
  "dependency_edges": [],
  "produced_artifacts": [
    {"name": "search_results_bert", "artifact_type": "search_results"}
  ],
  "consumed_artifacts": [],
  "perturbable_slots": [
    {
      "slot": "tool_arguments.query",
      "value_type": "search_query",
      "current_value": "blocks in bert base",
      "allowed_perturbation_types": ["parameter", "data_reference"]
    }
  ],
  
  // === Derived Annotations (Pass 2-3) ===
  "critical_path_score": {"value": 0.9, "source": "heuristic", "confidence": null},
  "affects_final_answer": {"value": true, "source": "heuristic", "confidence": null},
  "recoverable_if_wrong": {"value": true, "source": "heuristic", "confidence": null},
  "observable_if_wrong": {"value": true, "source": "heuristic", "confidence": null}
}
```

**Step 2 — extraction:**
```json
{
  "step_index": 2,
  "raw_text": "Examine the search results page to locate the answer (12)",
  "step_role": "extraction",
  "is_terminal_step": false,
  "produces_final_answer": false,
  "produces_patch": false,
  
  // === Extraction Fields (core) ===
  "extracted_value": 12,
  "value_type": "integer",
  "source_step": 1,
  "source_description": "BERT base layer count from search results",
  
  "entities": ["BERT base", "12 layers"],
  "depends_on_steps": [1],
  "dependency_edges": [
    {"type": "uses_observation_from", "source_step": 1, "reason": "extracts layer count from search results"}
  ],
  "produced_artifacts": [
    {"name": "bert_layer_count", "artifact_type": "numeric_answer"}
  ],
  "consumed_artifacts": ["search_results_bert"],
  "perturbable_slots": [
    {
      "slot": "extracted_value",
      "value_type": "integer",
      "current_value": 12,
      "allowed_perturbation_types": ["data_reference"]
    }
  ],
  
  // === Derived Annotations (Pass 2-3) ===
  "critical_path_score": {"value": 0.95, "source": "heuristic", "confidence": null},
  "affects_final_answer": {"value": true, "source": "heuristic", "confidence": null},
  "recoverable_if_wrong": {"value": false, "source": "heuristic", "confidence": null},
  "observable_if_wrong": {"value": false, "source": "heuristic", "confidence": null}
}
```

**Step 3 — tool_call:**
```json
{
  "step_index": 3,
  "raw_text": "Search the internet for \"attention is all you need layers\"...",
  "step_role": "tool_call",
  "is_terminal_step": false,
  "produces_final_answer": false,
  "produces_patch": false,
  "tool_name": "web_search",
  "tool_arguments": {"query": "attention is all you need layers"},
  "entities": ["Attention is All You Need", "Transformer", "layers"],
  "depends_on_steps": [],
  "dependency_edges": [],
  "produced_artifacts": [
    {"name": "search_results_transformer", "artifact_type": "search_results"}
  ],
  "consumed_artifacts": [],
  "perturbable_slots": [
    {
      "slot": "tool_arguments.query",
      "value_type": "search_query",
      "current_value": "attention is all you need layers",
      "allowed_perturbation_types": ["parameter"]
    }
  ],
  
  // === Derived Annotations (Pass 2-3) ===
  "critical_path_score": {"value": 0.9, "source": "heuristic", "confidence": null},
  "affects_final_answer": {"value": true, "source": "heuristic", "confidence": null},
  "recoverable_if_wrong": {"value": true, "source": "heuristic", "confidence": null},
  "observable_if_wrong": {"value": true, "source": "heuristic", "confidence": null}
}
```

**Step 4 — extraction:**
```json
{
  "step_index": 4,
  "raw_text": "Examine the architecture section of the PDF to locate the answer (6)",
  "step_role": "extraction",
  "is_terminal_step": false,
  "produces_final_answer": false,
  "produces_patch": false,
  
  // === Extraction Fields (core) ===
  "extracted_value": 6,
  "value_type": "integer",
  "source_step": 3,
  "source_description": "Transformer encoder layer count from paper",
  
  "entities": ["Transformer", "6 layers", "encoder"],
  "depends_on_steps": [3],
  "dependency_edges": [
    {"type": "uses_observation_from", "source_step": 3, "reason": "extracts layer count from PDF"}
  ],
  "produced_artifacts": [
    {"name": "transformer_layer_count", "artifact_type": "numeric_answer"}
  ],
  "consumed_artifacts": ["search_results_transformer"],
  "perturbable_slots": [
    {
      "slot": "extracted_value",
      "value_type": "integer",
      "current_value": 6,
      "allowed_perturbation_types": ["data_reference"]
    }
  ],
  
  // === Derived Annotations (Pass 2-3) ===
  "critical_path_score": {"value": 0.95, "source": "heuristic", "confidence": null},
  "affects_final_answer": {"value": true, "source": "heuristic", "confidence": null},
  "recoverable_if_wrong": {"value": false, "source": "heuristic", "confidence": null},
  "observable_if_wrong": {"value": false, "source": "heuristic", "confidence": null}
}
```

**Step 5 — reasoning (terminal):**
```json
{
  "step_index": 5,
  "raw_text": "Calculate the difference between the two numbers",
  "step_role": "reasoning",
  "is_terminal_step": true,
  "produces_final_answer": true,
  "produces_patch": false,
  
  // === Reasoning-specific fields ===
  "reasoning_type": "arithmetic",
  "operation": "subtraction",
  "operands": [12, 6],
  "result": 6,
  
  "entities": ["difference", "12", "6"],
  "depends_on_steps": [2, 4],
  "dependency_edges": [
    {"type": "uses_extracted_value_from", "source_step": 2, "value": 12},
    {"type": "uses_extracted_value_from", "source_step": 4, "value": 6}
  ],
  "produced_artifacts": [
    {"name": "final_answer", "artifact_type": "numeric_answer"}
  ],
  "consumed_artifacts": ["bert_layer_count", "transformer_layer_count"],
  "perturbable_slots": [
    {
      "slot": "operands[0]",
      "value_type": "integer",
      "current_value": 12,
      "allowed_perturbation_types": ["data_reference"]
    },
    {
      "slot": "operands[1]",
      "value_type": "integer",
      "current_value": 6,
      "allowed_perturbation_types": ["data_reference"]
    }
  ],
  
  // === Derived Annotations (Pass 2-3) ===
  "critical_path_score": {"value": 1.0, "source": "heuristic", "confidence": null},
  "affects_final_answer": {"value": true, "source": "heuristic", "confidence": null},
  "recoverable_if_wrong": {"value": false, "source": "heuristic", "confidence": null},
  "observable_if_wrong": {"value": true, "source": "heuristic", "confidence": null}
}
```

**Note:** This step has `step_role: "reasoning"` with `is_terminal_step: true` and `produces_final_answer: true`. We use single-valued step_role + boolean terminal flags rather than multi-label roles for cleaner analysis.

**Key insight:** Steps 2 and 4 are extractions with `recoverable_if_wrong: false` — if the agent extracts the wrong number (e.g., 10 instead of 12), downstream computation is silently wrong. This is a high-value perturbation target.

---

## Example C: SWE-bench (swebench_jd__tenacity) — Key Steps Typed

**Trajectory-level:**
```json
{
  "trajectory_id": "swebench_jd__tenacity.0d40e76f.121.4slwej0r",
  "benchmark": "swebench",
  "task_id": "jd__tenacity.0d40e76f",
  "task_text": "Fix the wait_chain class where strategies are applied in reverse order",
  "baseline_outcome": 1.0,
  "num_steps": 14,
  "ground_truth_available": true,
  "environment_type": "code_edit",
  "has_objective_verifier": true
}
```

**Step 2 — tool_call:**
```json
{
  "step_index": 2,
  "raw_text": "Based on the PR description, we need to look at the `wait_chain` implementation. Since it's a wait strategy, it's likely in `wait.py`",
  "step_role": "tool_call",
  "is_terminal_step": false,
  "produces_final_answer": false,
  "produces_patch": false,
  "tool_name": "file_view",
  "tool_arguments": {"path": "/testbed/tenacity/wait.py"},
  "entities": ["wait_chain", "wait.py", "tenacity"],
  "depends_on_steps": [1],
  "dependency_edges": [
    {"type": "follows_plan_from", "source_step": 1, "reason": "directory listing informed file choice"}
  ],
  "produced_artifacts": [
    {"name": "wait_py_contents", "artifact_type": "code_snippet"}
  ],
  "consumed_artifacts": ["testbed_listing"],
  "perturbable_slots": [
    {
      "slot": "tool_arguments.path",
      "value_type": "filepath",
      "current_value": "/testbed/tenacity/wait.py",
      "allowed_perturbation_types": ["data_reference", "parameter"]
    }
  ],
  
  // === Derived Annotations (Pass 2-3) ===
  "critical_path_score": {"value": 0.9, "source": "heuristic", "confidence": null},
  "affects_patch": {"value": true, "source": "heuristic", "confidence": null},
  "recoverable_if_wrong": {"value": true, "source": "heuristic", "confidence": null},
  "observable_if_wrong": {"value": true, "source": "heuristic", "confidence": null}
}
```

**Step 3 — extraction:**
```json
{
  "step_index": 3,
  "raw_text": "I found the issue! In the `__init__` method on line 108, the strategies are being reversed: self.strategies = list(reversed(strategies))",
  "step_role": "extraction",
  "is_terminal_step": false,
  "produces_final_answer": false,
  "produces_patch": false,
  
  // === Extraction Fields (core) ===
  "extracted_value": {
    "file": "/testbed/tenacity/wait.py",
    "line": 108,
    "code": "self.strategies = list(reversed(strategies))",
    "diagnosis": "strategies reversed incorrectly"
  },
  "value_type": "code_location",
  "source_step": 2,
  "source_description": "Bug location identified from file_view output",
  
  "entities": ["wait_chain", "__init__", "line 108", "reversed"],
  "depends_on_steps": [2],
  "dependency_edges": [
    {"type": "uses_observation_from", "source_step": 2, "reason": "identified bug location from file contents"}
  ],
  "produced_artifacts": [
    {"name": "bug_location", "artifact_type": "line_number"},
    {"name": "bug_diagnosis", "artifact_type": "diagnosis"}
  ],
  "consumed_artifacts": ["wait_py_contents"],
  "perturbable_slots": [
    {
      "slot": "extracted_value.line",
      "value_type": "line_number",
      "current_value": 108,
      "allowed_perturbation_types": ["data_reference"]
    },
    {
      "slot": "extracted_value.file",
      "value_type": "filepath",
      "current_value": "/testbed/tenacity/wait.py",
      "allowed_perturbation_types": ["data_reference"]
    }
  ],
  
  // === Derived Annotations (Pass 2-3) ===
  "critical_path_score": {"value": 0.95, "source": "heuristic", "confidence": null},
  "affects_patch": {"value": true, "source": "heuristic", "confidence": null},
  "recoverable_if_wrong": {"value": false, "source": "heuristic", "confidence": null},
  "observable_if_wrong": {"value": false, "source": "heuristic", "confidence": null}
}
```

**Step 13 — tool_call (terminal patch):**
```json
{
  "step_index": 13,
  "raw_text": "Let's fix this by removing the `reversed()` call...",
  "step_role": "tool_call",
  "is_terminal_step": true,
  "produces_final_answer": false,
  "produces_patch": true,
  "tool_name": "str_replace_editor",
  "tool_arguments": {
    "path": "/testbed/tenacity/wait.py",
    "old_str": "self.strategies = list(reversed(strategies))",
    "new_str": "self.strategies = list(strategies)"
  },
  "entities": ["wait.py", "reversed", "strategies"],
  "depends_on_steps": [3, 12],
  "dependency_edges": [
    {"type": "uses_extracted_value_from", "source_step": 3, "value": "bug_location"},
    {"type": "uses_observation_from", "source_step": 12, "reason": "confirmed reproduction"}
  ],
  "produced_artifacts": [
    {"name": "code_patch", "artifact_type": "patch"}
  ],
  "consumed_artifacts": ["bug_location", "reproduction_result"],
  "perturbable_slots": [
    {
      "slot": "tool_arguments.path",
      "value_type": "filepath",
      "current_value": "/testbed/tenacity/wait.py",
      "allowed_perturbation_types": ["data_reference"]
    },
    {
      "slot": "tool_arguments.old_str",
      "value_type": "code_snippet",
      "current_value": "self.strategies = list(reversed(strategies))",
      "allowed_perturbation_types": ["data_reference"]
    },
    {
      "slot": "tool_arguments.new_str",
      "value_type": "code_snippet",
      "current_value": "self.strategies = list(strategies)",
      "allowed_perturbation_types": ["parameter"]
    }
  ],
  
  // === Derived Annotations (Pass 2-3) ===
  "critical_path_score": {"value": 1.0, "source": "heuristic", "confidence": null},
  "affects_patch": {"value": true, "source": "heuristic", "confidence": null},
  "recoverable_if_wrong": {"value": false, "source": "heuristic", "confidence": null},
  "observable_if_wrong": {"value": true, "source": "heuristic", "confidence": null}
}
```

**Key insight:** Step 3's extraction (`line: 108`) is critical. If the agent extracts the wrong line number, the patch in Step 13 will target the wrong code. This is `recoverable_if_wrong: false` because downstream steps consume this value directly.
