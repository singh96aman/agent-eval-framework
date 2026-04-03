# Experiment Run Summary: Phase 2-3 (Load + Perturb)

**Date:** 2026-04-02  
**Experiment ID:** exp_poc_toolbench_20260402  
**Config:** poc_experiment_toolbench  
**Phases:** load_trajectories, generate_perturbations

---

## Overview

Successfully completed the first two phases of the POC experiment:
1. **Load Phase**: Loaded 50 ToolBench trajectories into MongoDB
2. **Perturb Phase**: Generated 443 perturbations across 9 conditions

---

## Phase 1: Load Trajectories

### Input
- **Dataset**: ToolBench (local file: toolllama_G123_dfs_train.json)
- **Total examples in dataset**: 187,542
- **Target**: 50 trajectories

### Output
- **Loaded**: 50 ToolBench trajectories
- **GAIA**: 0 trajectories (not included in this POC run)
- **Storage**: MongoDB Atlas (database: agent_judge_experiment)
- **Cache status**: 50 new trajectories, 0 cache hits

### Sample Trajectory Details
- **Example ID**: toolbench_96212
- **Benchmark**: toolbench
- **Steps**: 4
- **Task**: "I am planning a trip to Paris with my family next month..."
- **First step type**: TOOL_EXECUTION
- **Tool used**: social_media_news_in_french_for_onelike

### Verification
✅ All 50 trajectories successfully stored  
✅ Verified count in database matches expected  
✅ Trajectory schema validated  

---

## Phase 2: Generate Perturbations

### Configuration
- **Mode**: static (rule-based perturbations)
- **Conditions**: 9 total
  - 3 perturbation types: planning, tool_selection, parameter
  - 3 positions: early, middle, late
- **Batch size**: 10 (memory-efficient storage)

### Results
- **Total perturbations generated**: 443
- **Failed perturbations**: 7
- **Success rate**: 98.4% (443/450 expected)

### Perturbation Distribution
Each of the 50 trajectories received perturbations for:
- Planning errors (early/middle/late)
- Tool selection errors (early/middle/late)
- Parameter errors (early/middle/late)

Expected: 50 trajectories × 9 conditions = 450 perturbations  
Actual: 443 perturbations (7 failures)

### Failed Perturbations
The following perturbations failed to apply:

1. **tool_selection/early** on trajectory toolbench_165970
2. **tool_selection/early** on trajectory toolbench_178173
3. **tool_selection/early** on trajectory toolbench_155364
4. **tool_selection/early** on trajectory toolbench_154649
5. **tool_selection/early** on trajectory toolbench_59263
6. **tool_selection/late** on trajectory toolbench_59263
7. **tool_selection/early** on trajectory toolbench_24526

**Pattern observed**: All failures are tool_selection perturbations (6 early, 1 late). This suggests:
- Some trajectories may not have suitable tool choices at those positions
- Tool selection perturbation logic may need refinement for edge cases
- This is acceptable for POC (98.4% success rate is sufficient)

### Storage
✅ All 443 perturbations stored in MongoDB  
✅ Batch storage verified (10 perturbations per batch)  
✅ Perturbation metadata preserved (type, position, original/perturbed steps)

---

## Database State

**Collections populated:**
- `experiments`: 1 entry (exp_poc_toolbench_20260402)
- `trajectories`: 50 original trajectories
- `perturbations`: 443 perturbed trajectories

**Storage verification:**
- Experiment record created: ✅
- Trajectories cached: ✅ (50/50)
- Perturbations stored: ✅ (443/443)

---

## Next Steps

### Phase 4: Judge Evaluation
**Goal**: Evaluate all perturbations with two LLM judges

**Judges to configure:**
1. Claude Sonnet 4.5 (AWS Bedrock)
   - Model ID: us.anthropic.claude-sonnet-4-5-20250929-v1:0
   - Temperature: 0.7
   - Max tokens: 2000
2. GPT-OSS 120B (AWS Bedrock)
   - Model ID: openai.gpt-oss-120b-1:0
   - Temperature: 0.7
   - Max tokens: 2000

**Evaluation plan:**
- 443 perturbations × 2 judges × 3 samples = ~2,658 judge calls
- Estimated cost: TBD (calculate based on token usage)
- Estimated time: TBD (depends on rate limits)

**Implementation needs:**
- [ ] Judge API integration (src/judges/)
- [ ] Prompt templates for trajectory evaluation
- [ ] Judge output parsing and storage
- [ ] Batch processing with rate limiting
- [ ] Error handling and retry logic
- [ ] Progress tracking and checkpointing

### Phase 5: Annotation + Metrics
**Goal**: Human annotation of criticality + CCG computation

**Annotation requirements:**
- Review perturbations (sample or all?)
- Score criticality based on:
  - Task success degradation
  - Completeness impact
  - Hallucination introduced
  - Efficiency impact
- Store ground truth annotations

**CCG computation:**
- Compare judge penalties vs. ground truth criticality
- Calculate Criticality-Calibration Gap
- Statistical analysis (ANOVA, effect sizes)
- Generate visualizations (heatmaps, scatter plots)

---

## Log Files

**Main log**: `/Users/amanzing/Paper-Research/logs/exp_poc_toolbench_20260402.log`

Contains full execution trace including:
- Configuration details
- Loading progress
- Perturbation generation details
- All warnings and errors
- Timing information

---

## Success Criteria Met

✅ **Infrastructure**: All systems operational  
✅ **Data loading**: 50 trajectories loaded successfully  
✅ **Perturbations**: 443/450 (98.4%) perturbations generated  
✅ **Storage**: All data persisted to MongoDB  
✅ **Logging**: Complete execution trace captured  
✅ **Checkpointing**: Can resume from this point  

**Status**: Ready for Phase 4 (Judge Evaluation)

---

## Notes for Paper

### Dataset Selection
- Chose ToolBench-only for POC to minimize complexity
- GAIA integration deferred (can add in scaling phase)
- 50 trajectories provides sufficient statistical power for POC

### Perturbation Quality
- 98.4% success rate indicates robust perturbation system
- Failed cases all tool_selection type → suggests specific edge case
- Manual review of sample perturbations recommended before judge eval

### Experimental Validity
- 443 perturbations across 9 conditions gives ~49 per condition
- Sufficient for detecting large effect sizes (if present)
- Can scale to 150+ trajectories if POC shows promise

### Technical Debt
- Tool selection perturbations may need edge case handling
- Consider adding validation step before judge evaluation
- Document perturbation failure patterns for paper methods section
