# Bugs and Issues - Section 1: Trajectory Sources

## Implementation Date: 2026-04-07 (v1-v6)

## Status: IMPLEMENTED (v6 - expert strict filter, GAIA dropped)

---

## Summary

Implemented stratified sampling for trajectory sources with:
- **Quality filters**: Remove trajectories with HTTP errors, placeholder answers, empty content
- **Domain diversity caps**: Prevent any domain from dominating (e.g., "other" capped at 25%)
- **Full provenance tracking**: Each trajectory records how/when it was sampled

---

## Files Changed

| File | Change |
|------|--------|
| `src/data/schema.py` | Added `SamplingProvenance` and `SamplingManifest` dataclasses |
| `src/data/loaders.py` | Added quality filters, domain diversity, stratified sampling |
| `src/experiment_runner.py` | Updated `_phase_load()` to use new stratified sampling |
| `config/experiments/v2/trajectory_sampling.json` | Config with quality filters and domain caps |
| `tests/test_loaders.py` | Added 10 new tests for provenance and sampling |

---

## Quality Filters Added (v2)

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| `max_http_error_rate` | 50% | Reject trajectories where >50% of steps have 403/500/503 errors |
| `max_thought_none` | 2 | Reject trajectories with >=2 "Thought: None" steps |
| `max_empty_content_rate` | 20% (30% for SWE-bench) | Reject trajectories with too many empty steps |
| `max_missing_tool_input_rate` | 50% (GAIA only) | Reject GAIA browser actions without inputs |
| `reject_placeholder_answers` | true | Reject "Country 1", "N/A" style answers |

**Benchmark-specific overrides:**
- ToolBench: `max_missing_tool_input_rate: 1.0` (disabled - many APIs have no required params)
- SWE-bench: `max_empty_content_rate: 0.3` (relaxed - some reasoning steps are short)

---

## Domain Diversity Caps (v2)

| Benchmark | Domain | Cap | Rationale |
|-----------|--------|-----|-----------|
| ToolBench | "other" | 25% | Was 46%, too vague for analysis |
| GAIA | "general_qa" | 50% | Was 73%, need more document_analysis, web_search |

**Result:** Domain distribution is now more balanced across categories.

---

## Dry Run Results (v2)

```
Total trajectories: 400 (ToolBench only test)
Quality rejections: 178

By domain:
  other: 88 (22.0%) — capped at 25% ✓
  media_entertainment: 50 (12.5%)
  finance_business: 49 (12.2%)
  ecommerce_shopping: 48 (12.0%)
  data_information: 47 (11.8%)
  sports_gaming: 43 (10.8%)
  travel_logistics: 29 (7.2%)
  social_communication: 27 (6.8%)
  utilities_tools: 19 (4.8%)

By complexity:
  simple: 200 (50.0%)
  medium: 200 (50.0%)
```

---

## Test Results

- **All sampling tests:** 10/10 passing
- **pyflakes:** 0 errors

---

## How to Run

```bash
# Dry run to verify sampling
python main.py --config trajectory_sampling --runner load --dry-run

# Run tests
python -m pytest tests/test_loaders.py -v -k "Provenance or Manifest or Stratified"
```

---

## Config Structure (v2)

```json
{
  "quality_filters": {
    "enabled": true,
    "thresholds": {
      "max_http_error_rate": 0.5,
      "max_thought_none": 2,
      "max_empty_content_rate": 0.2,
      "reject_placeholder_answers": true
    },
    "benchmark_overrides": {
      "toolbench": {"max_missing_tool_input_rate": 1.0},
      "swebench": {"max_empty_content_rate": 0.3}
    }
  },
  "sampling": {
    "stratify_by": {
      "domain": {
        "enabled": true,
        "caps": {
          "toolbench": {"other": 0.25},
          "gaia": {"general_qa": 0.50}
        }
      }
    }
  }
}
```

---

---

## Expert Feedback (v3) - 2026-04-07

Expert identified remaining issues after v2:

1. **Bad ToolBench trajectories still present** (toolbench_19354, toolbench_23203):
   - HTTP errors (403/503), "Thought: None" steps
   - Missing Finish step
   - `expected_answer=null`, `task_success=null`

2. **SWE-bench duplicates**: Duplicate task descriptions for some PRs

3. **GAIA over-filtered**: Dropped from 100 to 29 trajectories

---

## v3 Changes (Expert Feedback Fixes)

### New Quality Filters Added

| Filter | Applied To | Rationale |
|--------|------------|-----------|
| `reject_null_outcomes` | ToolBench | Reject trajectories where expected_answer or task_success is null (toolbench_23203) |
| `require_finish_step` | ToolBench | Reject trajectories without a proper Finish step (toolbench_23203) |

### Deduplication Added

- **SWE-bench**: Task description deduplication to remove PRs with identical task descriptions
- Normalizes whitespace before comparison
- Logs duplicates to manifest for traceability

### GAIA Filter Tuning

| Filter | Old Value | New Value | Rationale |
|--------|-----------|-----------|-----------|
| `max_missing_tool_input_rate` | 0.5 | 1.0 (disabled) | GAIA browser actions often don't have explicit inputs |
| `reject_null_outcomes` | true | false | Many GAIA trajectories have null expected_answer by design |

### SWE-bench Filter Tuning

| Filter | Old Value | New Value | Rationale |
|--------|-----------|-----------|-----------|
| `reject_null_outcomes` | true | false | task_success is often null (evaluated by test pass/fail) |

---

## v3 Config Changes

```json
{
  "quality_filters": {
    "thresholds": {
      "reject_null_outcomes": true,
      "require_finish_step": true
    },
    "benchmark_overrides": {
      "gaia": {
        "max_missing_tool_input_rate": 1.0,
        "reject_null_outcomes": false
      },
      "swebench": {
        "reject_null_outcomes": false
      }
    }
  },
  "deduplication": {
    "enabled": true,
    "benchmarks": ["swebench"]
  }
}
```

---

## v3 Test Results

- **All v3 quality filter tests:** 12/12 passing
- **All sampling tests:** 22/22 passing (1 pre-existing mock issue)
- **pyflakes:** 0 errors

---

---

## Expert Feedback (v4) - 2026-04-07

Expert rated v3 as **PROCEED_WITH_FIXES** (not full PROCEED). Remaining blockers:

1. **toolbench_19354 still present**: Has 403 + 503 + "Thought: None" - each alone passes thresholds
2. **30 "Thought: None" steps** remaining in ToolBench
3. **57 empty SWE-bench steps**
4. **53 GAIA tool_execution steps with null tool_name**

---

## v4 Changes (Surgical Fixes)

### Combined Error Filter (Trajectory-Level)

New filter: `reject_combined_errors: true`
- Rejects trajectories with **BOTH** HTTP errors **AND** "Thought: None"
- Catches toolbench_19354 which has:
  - 403 error (under 50% threshold alone)
  - 503 error
  - "Thought: None" (under >=2 threshold alone)
  - But combination = fundamentally broken trajectory

### Step-Level Cleaning

New `step_cleaning` config section:
```json
{
  "step_cleaning": {
    "enabled": true,
    "remove_empty_steps": true,
    "remove_thought_none_steps": true,
    "normalize_null_tool_steps": true
  }
}
```

| Action | Target | Rationale |
|--------|--------|-----------|
| `remove_empty_steps` | SWE-bench | Expert: 57 empty steps |
| `remove_thought_none_steps` | ToolBench | Expert: 30 "Thought: None" steps |
| `normalize_null_tool_steps` | GAIA | Expert: 53 tool_execution with null tool_name → retype as REASONING |

### Step Renumbering

After cleaning, steps are renumbered sequentially (1, 2, 3...) to maintain consistency.

---

## v4 Test Results

- **All v4 quality filter tests:** 16/16 passing
- **All loaders tests:** 47/48 passing (1 pre-existing mock issue)
- **pyflakes:** 0 errors

---

---

## Expert Feedback (v5) - 2026-04-07

Expert identified two more quality issues:

1. **GAIA grounding quality**: Steps are label-only with null payloads
   - gaia_122: `tool_name: "web_search"` but `tool_input` and `tool_output` are null
   - gaia_131: entirely reasoning-only, no grounded tool evidence

2. **ToolBench "graceful failures"**: Apology answers with task_success=true
   - toolbench_87039: retrieves partial data, fails, finishes with "please try again later"
   - Valid under exact-match but weak baseline for Step 2

3. **Need 100 GAIA trajectories** (was getting 89)

---

## v5 Changes (Grounding & Graceful Failure Filters)

### New Quality Metrics

| Metric | Description |
|--------|-------------|
| `grounding_score` | Fraction of tool steps with actual payloads (tool_input OR tool_output) |
| `is_graceful_failure` | True if task_success=true but answer contains apology phrases |

### New Filters

| Filter | Applied To | Threshold | Rationale |
|--------|------------|-----------|-----------|
| `min_grounding_score` | GAIA | 0.3 (30%) | Reject trajectories where <30% of tool steps have payloads |
| `reject_graceful_failures` | All | true | Reject apology answers (toolbench_87039 style) |

### Apology Phrases Detected

```python
["please try again", "try again later", "unable to", "could not",
 "sorry", "apologize", "cannot provide", "no results", "failed to",
 "error occurred", "not available", "unavailable"]
```

### GAIA Limit Increased

- **Old limit**: 300
- **New limit**: 500
- **Rationale**: Compensate for grounding filter to hit 100 target

---

## v5 Test Results

- **All quality filter tests:** 20/20 passing
- **pyflakes:** 0 errors

---

## Additional Fixes (v5.1) - GAIA Coverage

### Issue: GAIA only loading 87/100

**Root cause 1**: `min_grounding_score: 0.3` was rejecting ALL GAIA trajectories because GAIA's data structure doesn't have structured `tool_input`/`tool_output` fields.
- **Fix**: Added `min_grounding_score: 0` to GAIA benchmark_overrides

**Root cause 2**: Domain cap `general_qa: 0.50` was too restrictive. GAIA is 73% `general_qa` but only has 162 total trajectories.
- **Fix**: Changed `general_qa` cap to `1.0` (disabled) for GAIA

### Final Results
```
Total: 294 (with reduced test limits)
  gaia: 100 ✓
  swebench: 100 ✓
  toolbench: 94 (would be 400 with full limit)
```

---

## v6 Changes - Expert Strict Filter & GAIA Dropped

### GAIA Dropped

Expert recommended dropping GAIA due to structural limitations:
- No structured `tool_input`/`tool_output` fields
- Steps are free-text annotations, not executable tool calls
- Weak for perturbation realism and LLM-judge evaluation

**New target: 500 trajectories (400 ToolBench, 100 SWE-bench)**

### Expert Strict Filter for ToolBench

New `check_toolbench_strict()` function implementing expert-provided filter:

```python
def keep_toolbench_main(traj):
    # 1. task_success must be True
    # 2. Must have steps
    # 3. Final step must be "Finish" tool
    # 4. tool_input.return_type must be "give_answer"
    # 5. Must have non-empty final_answer
    # 6. Must have at least one real tool output
    # 7. Final answer must not contain bad patterns
```

**Bad patterns detected:**
- Placeholders: `"country 1"`, `"capital 1"`, `"[link]"`
- Errors: `"please try again later"`, `"unable to retrieve"`
- Failures: `"couldn't find"`, `"no data associated"`

### v6 Config Changes

- `limit: 3000` for ToolBench (was 1500) - compensate for 31.7% pass rate
- `use_strict_filter: true` in ToolBench benchmark_overrides
- GAIA dataset disabled

### v6 Test Results

```
Total: 500
  toolbench: 400 ✓
  swebench: 100 ✓

Rejections: 9,469
  task_success_not_true: 7,180
  bad_pattern: 1,038
  apology_answer_with_success: 901
  return_type_not_give_answer: 78
```

---

## Next Steps

1. Proceed to Section 2: Typed Representation
2. All trajectory quality filters are now complete
