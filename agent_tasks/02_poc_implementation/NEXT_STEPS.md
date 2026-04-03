# Next Steps: Phase 4 - Judge API Integration

**Status**: Phases 1-3 Complete ✅  
**Current Phase**: Phase 4 - Judge Evaluation  
**Date**: 2026-04-02

---

## What We've Completed

✅ **Phase 1**: Infrastructure setup  
✅ **Phase 2**: Loaded 50 ToolBench trajectories  
✅ **Phase 3**: Generated 443 perturbations (9 conditions)

**Database Status:**
- Experiment ID: `exp_poc_toolbench_20260402`
- 50 original trajectories in MongoDB
- 443 perturbed trajectories ready for evaluation
- 98.4% perturbation success rate (7 failures, all tool_selection type)

---

## Phase 4: Judge API Integration

### Goal
Implement and execute LLM judge evaluation system to score all 443 perturbations.

### Components to Build

#### 1. Judge Base Interface
**File**: `src/judges/__init__.py`

Define abstract base class:
```python
class Judge(ABC):
    @abstractmethod
    def evaluate(self, trajectory: Trajectory) -> JudgeOutput:
        """Evaluate a trajectory and return scores."""
        pass
```

#### 2. Claude Sonnet 4.5 Judge
**File**: `src/judges/claude_judge.py`

**Requirements:**
- Connect to AWS Bedrock
- Model: `us.anthropic.claude-sonnet-4-5-20250929-v1:0`
- Handle authentication via boto3
- Implement rate limiting and retry logic
- Parse structured judge outputs

**Config from experiment:**
```json
{
  "temperature": 0.7,
  "max_tokens": 2000
}
```

#### 3. GPT-OSS 120B Judge
**File**: `src/judges/gpt_oss_judge.py`

**Requirements:**
- Connect to AWS Bedrock (GPT-OSS endpoint)
- Model: `openai.gpt-oss-120b-1:0`
- Same config as Claude judge
- Unified output format

#### 4. Judge Output Schema
**File**: `src/judges/schema.py`

Define structured output format:
```python
@dataclass
class JudgeOutput:
    trajectory_id: str
    judge_name: str
    overall_score: float  # 0-100
    task_success: int     # 0 or 1
    completeness: float   # 0-100
    hallucination: int    # 0 or 1
    efficiency_errors: int
    reasoning: str
    timestamp: datetime
```

#### 5. Judge Evaluation Runner
**File**: `src/judges/evaluator.py`

**Responsibilities:**
- Load perturbations from MongoDB
- Batch process with rate limiting
- Run multiple samples per trajectory (3x per config)
- Store outputs to MongoDB
- Handle errors and retries
- Track costs and progress

**Key functions:**
```python
def evaluate_experiment(
    experiment_id: str,
    judges: List[Judge],
    samples_per_trajectory: int = 3,
    batch_size: int = 10
) -> EvaluationResults:
    """Run judge evaluation for all perturbations."""
    pass
```

#### 6. Prompt Templates
**File**: `src/judges/prompts.py`

Design evaluation prompts based on POC requirements:
- Clear task description
- Trajectory presentation format
- Scoring criteria (5 metrics from config)
- Output format specification

### Integration with ExperimentRunner

**Update**: `src/experiment_runner.py`

Add new phase handler:
```python
def _run_judge_evaluation(self):
    """Phase: Evaluate perturbations with LLM judges."""
    # Load judges from config
    # Run evaluation
    # Store results
    pass
```

---

## Execution Plan

### Step 1: Implement Judge Infrastructure (2-3 hours)
1. Create base judge interface (`src/judges/__init__.py`)
2. Implement Claude judge (`src/judges/claude_judge.py`)
3. Implement GPT-OSS judge (`src/judges/gpt_oss_judge.py`)
4. Define output schema (`src/judges/schema.py`)
5. Create prompt templates (`src/judges/prompts.py`)

### Step 2: Build Evaluation Runner (1-2 hours)
1. Implement `src/judges/evaluator.py`
2. Add MongoDB storage for judge outputs
3. Implement batching and rate limiting
4. Add progress tracking and logging

### Step 3: Integrate with Experiment Runner (1 hour)
1. Add judge phase to `src/experiment_runner.py`
2. Update config handling for judge parameters
3. Add CLI support: `python main.py --config poc_experiment_toolbench --runner judge`

### Step 4: Testing (1-2 hours)
1. Unit tests for judges (with mocked API calls)
2. Integration test with sample trajectory
3. Validate output format and storage
4. Test rate limiting and error handling

### Step 5: Execute Judge Evaluation (2-4 hours)
1. Dry run with 5 perturbations to verify
2. Full run: 443 perturbations × 2 judges × 3 samples = ~2,658 calls
3. Monitor costs and performance
4. Verify all outputs stored correctly

---

## Expected Outputs

After Phase 4 completion:

1. **Code artifacts:**
   - `src/judges/__init__.py`
   - `src/judges/claude_judge.py`
   - `src/judges/gpt_oss_judge.py`
   - `src/judges/schema.py`
   - `src/judges/prompts.py`
   - `src/judges/evaluator.py`
   - `tests/test_judges.py`

2. **Database:**
   - New collection: `judge_outputs`
   - ~2,658 judge evaluation records

3. **Logs:**
   - Judge evaluation progress
   - API call counts and costs
   - Error reports (if any)

4. **CLI command:**
   ```bash
   python main.py --config poc_experiment_toolbench --runner judge
   ```

---

## Success Criteria

Phase 4 is complete when:

✅ Both judges (Claude + GPT-OSS) implemented and tested  
✅ Judge outputs stored in MongoDB with correct schema  
✅ All 443 perturbations evaluated (3 samples each)  
✅ Tests pass for judge infrastructure  
✅ Cost and performance metrics documented  
✅ Ready to proceed to Phase 5 (Annotation + Metrics)

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| **API rate limits** | Implement exponential backoff, batch processing |
| **High API costs** | Start with dry run, monitor costs closely |
| **Judge output parsing failures** | Strict prompt templates, validation, fallback handling |
| **AWS Bedrock auth issues** | Test connection before full run, document setup |
| **GPT-OSS endpoint unavailable** | Check prereqs, have fallback plan |

---

## Questions for Human Researcher

1. **Annotation first vs. Judge first?**
   - Current plan: Judge evaluation first (Phase 4), then annotation (Phase 5)
   - Alternative: Could annotate subset first to validate perturbations
   - Recommendation: Proceed with judges first, then annotate sample for validation

2. **Sample size for judges?**
   - Config specifies 3 samples per trajectory
   - With 443 perturbations → ~2,658 judge calls
   - Is this acceptable or should we reduce to 1 sample for POC?

3. **Cost estimates?**
   - Need to calculate expected costs for 2,658 API calls
   - Should we run cost estimation first?

4. **Perturbation failures?**
   - 7 perturbations failed (all tool_selection type)
   - Should we investigate and fix, or proceed with 443?
   - Recommendation: Proceed with 443 (98.4% is sufficient)

---

## Commands to Run

### Current status check:
```bash
python main.py --config poc_experiment_toolbench --runner load,perturb
# Already complete! ✅
```

### Next step (after implementing Phase 4):
```bash
# Test with dry run first
python main.py --config poc_experiment_toolbench --runner judge --dry-run

# Full judge evaluation
python main.py --config poc_experiment_toolbench --runner judge
```

### Full pipeline (after all phases):
```bash
python main.py --config poc_experiment_toolbench --runner all
```

---

## Timeline Estimate

- **Implementation**: 5-8 hours
- **Testing**: 1-2 hours
- **Execution**: 2-4 hours (depends on rate limits)
- **Total**: 8-14 hours for Phase 4 complete

**Next milestone**: Phase 5 (Annotation + CCG Metrics)
