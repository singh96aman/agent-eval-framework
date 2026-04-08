# 5B. LLM Judge Outputs

**Status: REQUIREMENTS DEFINED**

LLM Judge Outputs are the primary experimental measurements. Multiple LLM judges evaluate each evaluation unit, producing detection signals, localization predictions, and impact estimates that are compared against human labels and outcome evidence.

---

## Design Principles

1. **Mode-consistent**: Judge outputs respect the view mode (single, blinded_pair, labeled_pair)
2. **Structured**: All judge outputs follow a fixed schema for automated analysis
3. **Reproducible**: Deterministic sampling (temperature=0) or recorded seeds for stochastic runs
4. **Multi-judge**: Run multiple judge models for comparison
5. **Variance-aware**: Collect multiple samples for variance estimation

---

## 5B.1 Role in the Pipeline

```
Section 4: Evaluation Unit
         |
         v
    LLM Judge Views (from 4.6.2)
         |
         v
    5B. LLM Judge Outputs
         |
    +----+----+----+
    |    |    |    |
    v    v    v    v
6A. Detection   6B. Consequentiality   6C. Agreement
    Metrics         Metrics                Analysis
```

LLM Judge outputs provide:
- Detection signals for PDR, SLA, TIA (6A)
- Impact predictions for CCorr, CCE, ORR, URR (6B)
- Multi-sample variance for judge reliability (6C)

---

## 5B.2 Judge Configuration

### Judge Models

| Model | Provider | Primary Role |
|-------|----------|--------------|
| GPT-4o | OpenAI | Primary baseline judge |
| Claude 3.5 Sonnet | Anthropic | Alternative judge |
| Claude 3 Opus | Anthropic | High-capability judge |
| Llama 3.1 70B | Meta | Open-weight comparison |

### Judge Modes (from Section 4)

| Mode | Input | Use Case |
|------|-------|----------|
| **single_trajectory** | One trajectory (usually perturbed) | Standard evaluation |
| **blinded_pair** | Two trajectories without labels | Comparative detection |
| **labeled_pair** | Two trajectories with baseline/perturbed labels | Calibration studies |

### Sampling Configuration

| Setting | Value | Purpose |
|---------|-------|---------|
| temperature | 0 | Deterministic baseline |
| temperature | 0.7 | Variance estimation (n=5 samples) |
| max_tokens | 2000 | Sufficient for structured output |
| response_format | JSON | Enforce structured output |

---

## 5B.3 Judge Output Schema

### Core Output Object

```json
{
  // === Identity ===
  "judge_output_id": "string (uuid)",
  "evaluation_unit_id": "string",
  "judge_model": "string (e.g., gpt-4o-2024-08-06)",
  "judge_mode": "single_trajectory | blinded_pair | labeled_pair",
  "created_at": "datetime (ISO 8601)",
  
  // === Configuration ===
  "config": {
    "prompt_template_id": "string",
    "temperature": "float",
    "seed": "int | null",
    "max_tokens": "int",
    "sample_index": "int (0 for deterministic, 1-n for variance samples)"
  },
  
  // === View Information ===
  "input_view": {
    "view_file": "string (path)",
    "trajectory_variant_ids": ["string[]"]
  },
  
  // === Core Outputs (All Modes) ===
  "detection": {
    "overall_score": "float (0-100)",
    "error_detected": "bool",
    "error_confidence": "float (0-1)"
  },
  
  // === Localization Outputs (If Error Detected) ===
  "localization": {
    "predicted_error_step": "int (display_step_index) | null",
    "predicted_error_step_canonical": "string (canonical_step_id) | null",
    "localization_confidence": "float (0-1) | null",
    "predicted_error_type": "planning | tool_selection | parameter | data_reference | other | null"
  },
  
  // === Impact Outputs ===
  "impact": {
    "predicted_impact_score": "float (0-1)",
    "predicted_failure_prob": "float (0-1)",
    "impact_explanation": "string | null"
  },
  
  // === Pair-Mode Outputs (blinded_pair, labeled_pair only) ===
  "pair_comparison": {
    "error_trajectory": "A | B | neither | both | null",
    "preference": "A | B | tie | null",
    "comparison_explanation": "string | null"
  },
  
  // === Raw Response ===
  "raw_response": "string (full model output)",
  "parse_status": "success | partial | failed",
  "parse_errors": ["string[] | null"]
}
```

### Mode-Specific Required Fields

| Field | single_trajectory | blinded_pair | labeled_pair |
|-------|-------------------|--------------|--------------|
| detection.* | Required | Required | Required |
| localization.* | If error_detected | If error_detected | If error_detected |
| impact.* | Required | Required | Required |
| pair_comparison.* | null | Required | Required |

---

## 5B.4 Prompt Templates

### Template Structure

```json
{
  "template_id": "string",
  "template_version": "string",
  "judge_mode": "single_trajectory | blinded_pair | labeled_pair",
  "system_prompt": "string",
  "user_prompt_template": "string (with {{placeholders}})",
  "output_schema": "JSON Schema for expected output",
  "created_at": "datetime",
  "notes": "string"
}
```

### Template: Single Trajectory Evaluation

```
template_id: single_trajectory_v1
judge_mode: single_trajectory

system_prompt: |
  You are an expert evaluator of AI agent trajectories. Your task is to evaluate
  a trajectory where an AI agent attempts to complete a task using tools.
  
  Analyze the trajectory for:
  1. Overall quality (score 0-100)
  2. Whether any errors or mistakes exist
  3. If errors exist, identify the specific step and type
  4. Estimate the impact of any errors on task success
  
  Respond in JSON format only.

user_prompt_template: |
  Task: {{task_text}}
  
  Trajectory:
  {{formatted_trajectory}}
  
  Evaluate this trajectory. Respond with JSON:
  {
    "overall_score": <0-100>,
    "error_detected": <true/false>,
    "error_confidence": <0-1>,
    "predicted_error_step": <step number or null>,
    "predicted_error_type": <"planning"|"tool_selection"|"parameter"|"data_reference"|"other"|null>,
    "localization_confidence": <0-1 or null>,
    "predicted_impact_score": <0-1>,
    "predicted_failure_prob": <0-1>,
    "impact_explanation": "<brief explanation>"
  }
```

### Template: Blinded Pair Comparison

```
template_id: blinded_pair_v1
judge_mode: blinded_pair

system_prompt: |
  You are an expert evaluator comparing two AI agent trajectories.
  Both trajectories attempt the same task. One may contain an error.
  
  Your job is to:
  1. Determine if either trajectory contains an error
  2. If so, identify which trajectory and which step
  3. Compare overall quality
  
  You do NOT know which trajectory is "baseline" or "perturbed" - evaluate purely on quality.
  Respond in JSON format only.

user_prompt_template: |
  Task: {{task_text}}
  
  Trajectory A:
  {{formatted_trajectory_a}}
  
  Trajectory B:
  {{formatted_trajectory_b}}
  
  Compare these trajectories. Respond with JSON:
  {
    "overall_score_a": <0-100>,
    "overall_score_b": <0-100>,
    "error_detected": <true/false>,
    "error_trajectory": <"A"|"B"|"neither"|"both">,
    "error_confidence": <0-1>,
    "predicted_error_step": <step number in error trajectory, or null>,
    "predicted_error_type": <type or null>,
    "preference": <"A"|"B"|"tie">,
    "predicted_impact_score": <0-1>,
    "predicted_failure_prob": <0-1>,
    "comparison_explanation": "<brief explanation>"
  }
```

### Template: Labeled Pair (Calibration)

```
template_id: labeled_pair_v1
judge_mode: labeled_pair

system_prompt: |
  You are evaluating a controlled perturbation study. You will see:
  - A baseline trajectory (original, no errors)
  - A perturbed trajectory (with a known injected error)
  
  Your task is to estimate the impact of the perturbation.
  Respond in JSON format only.

user_prompt_template: |
  Task: {{task_text}}
  
  Baseline Trajectory (no error):
  {{formatted_baseline}}
  
  Perturbed Trajectory (contains injected error at step {{target_step}}):
  {{formatted_perturbed}}
  
  Perturbation: {{perturbation_type}} at step {{target_step}}
  
  Estimate the impact. Respond with JSON:
  {
    "impact_estimate": <0-3, where 0=none, 1=minor, 2=moderate, 3=critical>,
    "predicted_failure_prob": <0-1>,
    "detectability_estimate": <0-2, where 0=invisible, 1=subtle, 2=obvious>,
    "propagation_estimate": <number of downstream steps affected>,
    "explanation": "<brief explanation>"
  }
```

---

## 5B.5 Trajectory Formatting

### Format Options

| Format | Description | Use Case |
|--------|-------------|----------|
| **minimal** | Step index + action + result summary | Fast evaluation |
| **standard** | Full step details, truncated observations | Default |
| **verbose** | Complete step content including full observations | Detailed analysis |

### Standard Format Template

```
Step {{display_step_index}}: [{{step_role}}]
{{#if tool_name}}Tool: {{tool_name}}
Arguments: {{tool_arguments}}{{/if}}
{{#if extracted_value}}Extracted: {{extracted_value}} ({{value_type}}){{/if}}
{{#if observation}}Result: {{truncate observation 500}}{{/if}}
---
```

### Example Formatted Trajectory

```
Step 1: [tool_call]
Tool: web_search
Arguments: {"query": "blocks in bert base"}
Result: [Search results showing BERT architecture information...]
---
Step 2: [extraction]
Extracted: 10 (integer)
---
Step 3: [tool_call]
Tool: web_search
Arguments: {"query": "attention is all you need layers"}
Result: [Search results showing Transformer paper...]
---
Step 4: [extraction]
Extracted: 6 (integer)
---
Step 5: [reasoning]
Calculate: 10 - 6 = 4
Final Answer: 4
---
```

---

## 5B.6 Judge Run Configuration

### Full Evaluation Matrix

| Mode | Models | Samples | Temperature | Total Runs per Unit |
|------|--------|---------|-------------|---------------------|
| single_trajectory | 4 judges | 1 | 0 | 4 |
| single_trajectory | 2 judges | 5 | 0.7 | 10 (variance) |
| blinded_pair | 2 judges | 1 | 0 | 2 |

### Per-Unit Run Summary

| Run Type | Count | Purpose |
|----------|-------|---------|
| single_trajectory (deterministic) | 4 | Multi-judge comparison |
| single_trajectory (stochastic) | 10 | Variance estimation |
| blinded_pair | 2 | Comparative evaluation |
| **Total per unit** | **16** | |

### Full Dataset Scope

| Scope | Units | Runs per Unit | Total Runs |
|-------|-------|---------------|------------|
| All units (single, deterministic) | 1700 | 4 | 6800 |
| Variance subset (single, stochastic) | 500 | 10 | 5000 |
| Pair evaluation subset | 500 | 2 | 1000 |
| **Total** | - | - | **12800** |

---

## 5B.7 Output Processing

### Parse Pipeline

1. **Extract JSON**: Parse structured output from raw response
2. **Validate schema**: Check all required fields present
3. **Normalize values**: Clamp scores to valid ranges
4. **Map step indices**: Convert display_step_index to canonical_step_id
5. **Record parse status**: success, partial (some fields missing), failed

### Canonical Step ID Mapping

Judge outputs use `display_step_index` (what the judge sees). Post-processing maps to `canonical_step_id`:

```python
def map_to_canonical(judge_output, evaluation_unit, trajectory_variant):
    display_idx = judge_output["localization"]["predicted_error_step"]
    if display_idx is None:
        return None
    
    # Find step with matching display_step_index in the variant
    trajectory = get_trajectory_variant(evaluation_unit, trajectory_variant)
    for step in trajectory["steps"]:
        if step["display_step_index"] == display_idx:
            return step["canonical_step_id"]
    
    return None  # Step not found (invalid prediction)
```

### Error Type Mapping

| Judge Output | Maps to Perturbation Family |
|--------------|----------------------------|
| planning | structural |
| tool_selection | tool_selection |
| parameter | parameter |
| data_reference | data_reference |
| other | unclear |

---

## 5B.8 Aggregation

### Per-Unit Aggregation (Across Samples)

For stochastic runs (n=5 samples per judge):

```json
{
  "evaluation_unit_id": "string",
  "judge_model": "string",
  
  "aggregated_detection": {
    "mean_overall_score": "float",
    "std_overall_score": "float",
    "error_detected_rate": "float (fraction of samples detecting error)",
    "mean_error_confidence": "float",
    "std_error_confidence": "float"
  },
  
  "aggregated_localization": {
    "modal_predicted_step": "string (canonical_step_id) | null",
    "step_agreement_rate": "float",
    "modal_predicted_type": "string",
    "type_agreement_rate": "float"
  },
  
  "aggregated_impact": {
    "mean_impact_score": "float",
    "std_impact_score": "float",
    "mean_failure_prob": "float",
    "std_failure_prob": "float"
  }
}
```

### Cross-Judge Aggregation

For multi-judge comparison (deterministic runs):

```json
{
  "evaluation_unit_id": "string",
  
  "cross_judge": {
    "judges": ["gpt-4o", "claude-3.5-sonnet", "claude-3-opus", "llama-3.1-70b"],
    "error_detected_agreement": "float (0-1)",
    "step_localization_agreement": "float (0-1)",
    "impact_correlation": "float (Spearman)",
    "mean_overall_score": "float",
    "std_overall_score": "float"
  }
}
```

---

## 5B.9 Data Location

```
data/judge_outputs/
├── raw/
│   ├── single_trajectory/
│   │   ├── gpt-4o/
│   │   │   ├── deterministic/
│   │   │   │   └── outputs.jsonl
│   │   │   └── stochastic/
│   │   │       └── outputs.jsonl
│   │   ├── claude-3.5-sonnet/
│   │   └── ...
│   ├── blinded_pair/
│   │   └── ...
│   └── labeled_pair/
│       └── ...
├── aggregated/
│   ├── per_unit_aggregated.json
│   └── cross_judge_aggregated.json
├── templates/
│   ├── single_trajectory_v1.json
│   ├── blinded_pair_v1.json
│   └── labeled_pair_v1.json
└── metadata/
    ├── run_config.json
    └── run_log.json
```

---

## 5B.10 Cost Estimation

### Token Estimates per Unit

| Component | Input Tokens | Output Tokens |
|-----------|--------------|---------------|
| System prompt | ~200 | - |
| Task + trajectory (single) | ~1000 | - |
| Task + trajectories (pair) | ~2000 | - |
| Judge response | - | ~300 |

### Cost per Model (Approximate)

| Model | Input $/1M | Output $/1M | Cost per Run |
|-------|------------|-------------|--------------|
| GPT-4o | $2.50 | $10.00 | ~$0.006 |
| Claude 3.5 Sonnet | $3.00 | $15.00 | ~$0.008 |
| Claude 3 Opus | $15.00 | $75.00 | ~$0.040 |
| Llama 3.1 70B | $0.50 | $0.50 | ~$0.001 |

### Total Estimated Cost

| Run Type | Runs | Avg Cost/Run | Total |
|----------|------|--------------|-------|
| Deterministic (4 models) | 6800 | ~$0.015 | ~$100 |
| Stochastic (2 models) | 5000 | ~$0.007 | ~$35 |
| Pair evaluation (2 models) | 1000 | ~$0.012 | ~$12 |
| **Total** | **12800** | - | **~$150** |

*Note: Excludes Claude 3 Opus for variance runs due to cost.*

---

## 5B.11 Implementation Plan

### Module Structure

```
src/judge_outputs/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── openai.py           # GPT-4o client
│   ├── anthropic.py        # Claude clients
│   └── together.py         # Llama via Together API
├── templates/
│   ├── __init__.py
│   ├── loader.py           # Load and render templates
│   └── formatter.py        # Trajectory formatting
├── runner.py               # Main judge execution
├── parser.py               # Parse judge responses
├── aggregation.py          # Aggregate across samples/judges
├── validator.py            # Validate outputs
└── export.py               # Export for analysis
```

### Scripts

```bash
# Run judges on all units (single_trajectory, deterministic)
python -m src.judge_outputs.runner \
  --mode single_trajectory \
  --models gpt-4o,claude-3.5-sonnet,claude-3-opus,llama-3.1-70b \
  --temperature 0 \
  --input data/evaluation_units/views/llm_judge/single_trajectory/ \
  --output data/judge_outputs/raw/single_trajectory/

# Run variance estimation
python -m src.judge_outputs.runner \
  --mode single_trajectory \
  --models gpt-4o,claude-3.5-sonnet \
  --temperature 0.7 \
  --samples 5 \
  --sample-units 500 \
  --input data/evaluation_units/views/llm_judge/single_trajectory/ \
  --output data/judge_outputs/raw/single_trajectory/

# Run blinded pair evaluation
python -m src.judge_outputs.runner \
  --mode blinded_pair \
  --models gpt-4o,claude-3.5-sonnet \
  --temperature 0 \
  --sample-units 500 \
  --input data/evaluation_units/views/llm_judge/blinded_pair/ \
  --output data/judge_outputs/raw/blinded_pair/

# Aggregate outputs
python -m src.judge_outputs.aggregation \
  --input data/judge_outputs/raw/ \
  --output data/judge_outputs/aggregated/

# Validate all outputs
python -m src.judge_outputs.validator \
  --input data/judge_outputs/raw/ \
  --output data/judge_outputs/metadata/validation_report.json
```

---

## 5B.12 Dependencies

| Dependency | Section | What's Needed |
|------------|---------|---------------|
| Evaluation Unit (Section 4) | Input | LLM judge views |
| Human Labels (Section 5A) | Comparison | Reference for detection accuracy |
| Outcome Evidence (Section 5C) | Comparison | Reference for impact calibration |
| Detection Metrics (Section 6A) | Output | Judge outputs for PDR, SLA, TIA |
| Consequentiality (Section 6B) | Output | Judge outputs for CCorr, CCE |
| Agreement Analysis (Section 6C) | Output | Multi-judge variance |

---

## 5B.13 Success Criteria

Section 5B is complete when:

### Coverage
1. [ ] All 1700 units evaluated in single_trajectory mode (deterministic)
2. [ ] ≥4 judge models run on all units
3. [ ] 500 units evaluated with variance estimation (5 samples each)
4. [ ] 500 units evaluated in blinded_pair mode

### Quality
5. [ ] Parse success rate >95% across all runs
6. [ ] All required output fields populated
7. [ ] Canonical step ID mapping successful for all localization predictions
8. [ ] No systematic parsing failures by model

### Outputs
9. [ ] Raw outputs stored in JSONL format
10. [ ] Per-unit aggregation computed
11. [ ] Cross-judge aggregation computed
12. [ ] Outputs linked to evaluation_unit_id

### Documentation
13. [ ] Prompt templates versioned and stored
14. [ ] Run configuration documented
15. [ ] Cost tracking log maintained
16. [ ] Validation report generated

---

## 5B.14 Example Judge Outputs

### Single Trajectory Output (GAIA Example)

```json
{
  "judge_output_id": "judge_001_gpt4o",
  "evaluation_unit_id": "eval::gaia_122::001",
  "judge_model": "gpt-4o-2024-08-06",
  "judge_mode": "single_trajectory",
  "created_at": "2026-04-11T10:00:00Z",
  
  "config": {
    "prompt_template_id": "single_trajectory_v1",
    "temperature": 0,
    "seed": null,
    "max_tokens": 2000,
    "sample_index": 0
  },
  
  "input_view": {
    "view_file": "views/llm_judge/single_trajectory/eval_gaia_122_001.json",
    "trajectory_variant_ids": ["gaia_122::pert::001"]
  },
  
  "detection": {
    "overall_score": 65,
    "error_detected": true,
    "error_confidence": 0.7
  },
  
  "localization": {
    "predicted_error_step": 2,
    "predicted_error_step_canonical": "gaia_122::step::2",
    "localization_confidence": 0.6,
    "predicted_error_type": "data_reference"
  },
  
  "impact": {
    "predicted_impact_score": 0.8,
    "predicted_failure_prob": 0.75,
    "impact_explanation": "The extracted value of 10 for BERT layers appears incorrect. BERT base typically has 12 transformer blocks. This error propagates to the final calculation."
  },
  
  "pair_comparison": null,
  
  "raw_response": "{\"overall_score\": 65, \"error_detected\": true, ...}",
  "parse_status": "success",
  "parse_errors": null
}
```

### Blinded Pair Output (Same Unit)

```json
{
  "judge_output_id": "judge_002_gpt4o_pair",
  "evaluation_unit_id": "eval::gaia_122::001",
  "judge_model": "gpt-4o-2024-08-06",
  "judge_mode": "blinded_pair",
  "created_at": "2026-04-11T10:05:00Z",
  
  "config": {
    "prompt_template_id": "blinded_pair_v1",
    "temperature": 0,
    "seed": null,
    "max_tokens": 2000,
    "sample_index": 0
  },
  
  "input_view": {
    "view_file": "views/llm_judge/blinded_pair/eval_gaia_122_001.json",
    "trajectory_variant_ids": ["gaia_122::pert::001", "gaia_122::base"]
  },
  
  "detection": {
    "overall_score": 70,
    "error_detected": true,
    "error_confidence": 0.75
  },
  
  "localization": {
    "predicted_error_step": 2,
    "predicted_error_step_canonical": "gaia_122::step::2",
    "localization_confidence": 0.65,
    "predicted_error_type": "data_reference"
  },
  
  "impact": {
    "predicted_impact_score": 0.85,
    "predicted_failure_prob": 0.8,
    "impact_explanation": "Trajectory A extracts an incorrect layer count, leading to wrong final answer."
  },
  
  "pair_comparison": {
    "error_trajectory": "A",
    "preference": "B",
    "comparison_explanation": "Trajectory B correctly extracts 12 layers and produces the correct answer of 6. Trajectory A extracts 10, resulting in answer 4."
  },
  
  "raw_response": "{\"overall_score_a\": 60, \"overall_score_b\": 95, ...}",
  "parse_status": "success",
  "parse_errors": null
}
```

### Aggregated Output (Variance Samples)

```json
{
  "evaluation_unit_id": "eval::gaia_122::001",
  "judge_model": "gpt-4o-2024-08-06",
  
  "aggregated_detection": {
    "mean_overall_score": 63.4,
    "std_overall_score": 5.2,
    "error_detected_rate": 1.0,
    "mean_error_confidence": 0.72,
    "std_error_confidence": 0.08
  },
  
  "aggregated_localization": {
    "modal_predicted_step": "gaia_122::step::2",
    "step_agreement_rate": 1.0,
    "modal_predicted_type": "data_reference",
    "type_agreement_rate": 0.8
  },
  
  "aggregated_impact": {
    "mean_impact_score": 0.78,
    "std_impact_score": 0.06,
    "mean_failure_prob": 0.73,
    "std_failure_prob": 0.09
  }
}
```
