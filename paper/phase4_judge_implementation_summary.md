# Phase 4 Implementation Summary: Judge API Integration

**Date:** 2026-04-02  
**Status:** COMPLETE ✅  
**Commits:** 0c9e5cd → 3577e1a

---

## Overview

Successfully implemented complete LLM judge evaluation system with AWS Bedrock integration. The system evaluates agent trajectories across 5 key metrics and supports multiple judges with automatic retry, rate limiting, and progress tracking.

---

## Components Implemented

### 1. Core Schemas (`src/judges/schema.py`)

**JudgeOutput** - Structured evaluation result
- 5 core metrics: task_success, completeness, hallucination, sycophancy, efficiency_errors
- Overall score (0-100)
- Step-level error annotations
- Metadata (tokens used, evaluation time)

**StepError** - Individual error annotations
- Step index, error type, severity
- Impact on task success

**EvaluationResults** - Batch evaluation summary
- Success/failure counts
- Token usage, timing
- Error logs

### 2. Judge Base Interface (`src/judges/__init__.py`)

**Judge (Abstract Base Class)**
- Unified interface for all judge implementations
- Automatic retry with exponential backoff
- Statistics tracking (tokens, time, success rate)
- JSON response parsing helper

**Key Methods:**
- `evaluate(trajectory)` - Main evaluation method with retry logic
- `_call_llm()` - Abstract method for API calls
- `_parse_response()` - Abstract method for response parsing
- `get_stats()` - Usage statistics

### 3. Prompt Templates (`src/judges/prompts.py`)

**Evaluation Prompts:**
- Trajectory formatting for judges
- Clear metric definitions
- JSON output format specification
- Task: Extract from ground_truth.task_description
- Steps: Show content, tool name/input/output

**System Prompt:**
- Expert evaluator persona
- 5-dimension evaluation framework
- Objective, critical assessment instructions

### 4. Claude Sonnet 4.5 Judge (`src/judges/claude_judge.py`)

**Implementation:**
- AWS Bedrock integration (us.anthropic.claude-sonnet-4-5-20250929-v1:0)
- boto3 client configuration
- Anthropic API format (system + messages)
- Token usage tracking from response

**Factory Function:**
- `create_claude_judge(config)` - Create from experiment config

### 5. GPT-OSS 120B Judge (`src/judges/gpt_oss_judge.py`)

**Implementation:**
- AWS Bedrock integration (openai.gpt-oss-120b-1:0)
- OpenAI-compatible API format
- Same interface as Claude judge

**Factory Function:**
- `create_gpt_oss_judge(config)` - Create from experiment config

### 6. Batch Evaluator (`src/judges/evaluator.py`)

**JudgeEvaluator Class:**
- Orchestrates batch evaluation
- Rate limiting between API calls
- Progress tracking with periodic updates
- Resume capability (skips already-evaluated)
- Multiple samples per trajectory

**Key Features:**
- Configurable batch size
- Rate limit delay (default: 1s between calls)
- Dry run mode for testing
- Error handling and logging
- Summary statistics

**Main Method:**
- `evaluate_experiment(experiment_id)` - Evaluate all perturbations

### 7. MongoDB Integration (`src/storage/mongodb.py`)

**New Methods:**
- `store_judge_output(judge_output, experiment_id)` - Store evaluation result
- `get_judge_outputs(experiment_id, judge_name, trajectory_id)` - Retrieve outputs
- `count_judge_outputs(experiment_id, trajectory_id, judge_name)` - Count for resume logic

### 8. Experiment Runner Integration (`src/experiment_runner.py`)

**Updated:**
- `_phase_evaluate_judges()` - Fully implemented
- Reads judge config from experiment JSON
- Creates judge instances (Claude + GPT-OSS)
- Runs batch evaluation
- Prints summary statistics

**CLI Usage:**
```bash
python main.py --config poc_experiment_toolbench --runner judge
```

### 9. Test Suite (`tests/test_judges.py`)

**10 Tests Implemented:**
1. `test_format_trajectory_for_judge` - Trajectory formatting
2. `test_build_evaluation_prompt` - Full prompt construction
3. `test_parse_json_response_valid` - JSON parsing (with markdown)
4. `test_parse_json_response_no_markdown` - JSON parsing (plain)
5. `test_parse_json_response_invalid` - Error handling
6. `test_judge_evaluate_success` - Successful evaluation
7. `test_judge_evaluate_with_retry` - Retry logic
8. `test_judge_output_to_dict` - Serialization
9. `test_judge_output_from_dict` - Deserialization
10. `test_evaluation_results_to_dict` - Results serialization

**All tests passing ✅** (10/10)

---

## Architecture Highlights

### Unified Interface

All judges implement the same `Judge` abstract base class, enabling:
- Easy addition of new judge models
- Consistent error handling
- Unified statistics tracking

### Retry Logic

Built-in exponential backoff for API failures:
- Configurable max retries (default: 3)
- Exponential delay: 1s, 2s, 4s
- Graceful degradation (returns None after max retries)

### Resume Capability

Evaluator checks existing evaluations before making API calls:
- Counts samples per trajectory
- Skips if >= `samples_per_trajectory` already exist
- Enables interrupted run resumption
- Saves costs on re-runs

### Rate Limiting

Configurable delay between API calls:
- Default: 1 second
- Prevents API throttling
- Can be adjusted per experiment

### Progress Tracking

Periodic updates during batch processing:
- Every N trajectories (configurable batch_size)
- Shows completed/total, success/failure counts
- Estimated completion info

### Statistics

Per-judge tracking:
- Total API calls (successful + failed)
- Total tokens consumed
- Total time (milliseconds)
- Success rate
- Average time per call

---

## Configuration

Judges are configured in experiment JSON:

```json
{
  "judges": {
    "models": [
      {
        "name": "claude-sonnet-4.5",
        "provider": "aws_bedrock",
        "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "config": {
          "temperature": 0.7,
          "max_tokens": 2000
        }
      },
      {
        "name": "gpt-oss-120b",
        "provider": "aws_bedrock",
        "model_id": "openai.gpt-oss-120b-1:0",
        "config": {
          "temperature": 0.7,
          "max_tokens": 2000
        }
      }
    ],
    "samples_per_trajectory": 3
  }
}
```

---

## Evaluation Metrics

The judges evaluate on **5 dimensions**:

1. **Task Success** (binary: 0 or 1)
   - Did the agent complete the user's request?

2. **Completeness** (0-100%)
   - What percentage of requirements were satisfied?

3. **Hallucination** (binary: 0 or 1)
   - Did the agent fabricate information not in tool outputs?

4. **Sycophancy** (binary: 0 or 1)
   - Did the agent incorrectly validate false user assumptions?

5. **Efficiency Errors** (count)
   - Number of unnecessary/redundant tool calls

Plus:
- **Overall Score** (0-100): Holistic quality assessment
- **Step Errors**: List of specific errors with severity

---

## Next Steps: Execute Evaluation

Now that the system is implemented, the next step is to **run the actual evaluation**:

```bash
# Dry run first (no API calls)
python main.py --config poc_experiment_toolbench --runner judge --dry-run

# Full evaluation
python main.py --config poc_experiment_toolbench --runner judge
```

### Expected Workload

- **Perturbations**: 443
- **Judges**: 2 (Claude + GPT-OSS)
- **Samples per trajectory**: 3
- **Total API calls**: 443 × 2 × 3 = **2,658 calls**

### Cost Estimation

Need to estimate before running:
- Claude Sonnet 4.5 pricing
- GPT-OSS 120B pricing
- Expected tokens per evaluation (~1,500 avg?)
- Total cost projection

### Execution Time

Depends on rate limiting:
- With 1s delay: ~45 minutes minimum
- With API latency: ~1-2 hours estimated

### Monitoring

During execution:
- Watch for API errors
- Monitor token usage
- Check success rate
- Validate output quality (sample check)

---

## Testing

All tests passing:

```bash
$ python -m pytest tests/test_judges.py -v
======================== 10 passed, 7 warnings in 1.03s ========================
```

Key validations:
- Trajectory formatting works correctly
- JSON parsing handles markdown and plain formats
- Retry logic functions as expected
- Serialization/deserialization works
- Schema validation passes

---

## Files Changed

### New Files (9):
- `src/judges/__init__.py` (226 lines)
- `src/judges/schema.py` (163 lines)
- `src/judges/prompts.py` (197 lines)
- `src/judges/claude_judge.py` (175 lines)
- `src/judges/gpt_oss_judge.py` (165 lines)
- `src/judges/evaluator.py` (266 lines)
- `tests/test_judges.py` (333 lines)

### Modified Files (2):
- `src/experiment_runner.py` (+69 lines)
- `src/storage/mongodb.py` (+66 lines)

**Total**: +1,824 insertions, -2 deletions

---

## Success Criteria

✅ **Judge implementations**: Claude + GPT-OSS via Bedrock  
✅ **Unified interface**: Abstract base class with retry logic  
✅ **Evaluation prompts**: Clear, structured templates  
✅ **Batch evaluator**: Rate limiting, progress tracking, resume  
✅ **MongoDB storage**: Judge output persistence  
✅ **ExperimentRunner integration**: Judge phase handler  
✅ **Test suite**: 10 tests passing  
✅ **Documentation**: Code, docstrings, this summary  

**Phase 4 Implementation**: COMPLETE ✅

---

## Ready for Execution

The system is now ready to evaluate all 443 perturbations. Before running:

1. **Cost estimation**: Calculate expected API costs
2. **Dry run**: Test with `--dry-run` flag
3. **Sample validation**: Run on 5-10 perturbations first
4. **Full execution**: Run complete evaluation
5. **Validation**: Check output quality and completeness

After completion:
- **Phase 5**: Annotation tools + CCG computation
- **Phase 6**: Visualization and analysis
- **Phase 7**: Paper writing

---

## Technical Debt / Future Enhancements

- [ ] Add support for local LLM judges (non-Bedrock)
- [ ] Implement adaptive rate limiting (based on API response headers)
- [ ] Add prompt versioning for reproducibility
- [ ] Support for structured output mode (when available)
- [ ] Caching of identical trajectory evaluations across experiments
- [ ] Parallel evaluation with multiple API workers
- [ ] Real-time cost tracking during evaluation
- [ ] Evaluation quality metrics (inter-judge agreement)

---

## Notes for Paper

### Methods Section

- Judge evaluation system description
- 5-metric framework justification
- Multiple samples for reliability (n=3)
- Temperature setting (0.7 for balanced creativity/consistency)

### Reproducibility

- Exact model versions specified
- Prompt templates provided in appendix
- Random seed control (if applicable)
- Retry logic documented

### Limitations

- Judge evaluation is stochastic (temperature > 0)
- Multiple samples help but don't eliminate variance
- Judges may have systematic biases (that's what we're testing!)
- Cost constraints limit full factorial design
