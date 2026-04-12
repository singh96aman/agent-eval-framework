# Experiment Visual Guide: LLM-as-Judge for Agent Trajectories

**Experiment ID:** `exp_trajectory_sampling_v7`  
**Date:** 2026-04-11  
**Main Question:** Do LLM judges detect errors better than they estimate their downstream consequence?

---

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                              PHASE 1: DATA PREPARATION                                        │
└──────────────────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
    │  1. TRAJECTORY  │      │  2. TYPED       │      │  3. CONTROLLED  │      │  4. EVALUATION  │
    │     SOURCES     │ ───► │  REPRESENTATION │ ───► │  PERTURBATIONS  │ ───► │     UNIT        │
    └─────────────────┘      └─────────────────┘      └─────────────────┘      └─────────────────┘
           │                        │                        │                        │
           ▼                        ▼                        ▼                        ▼
    600 trajectories         TypedTrajectory           652 perturbed            652 eval units
    from 3 benchmarks        with step roles           trajectories             (baseline + perturbed)


┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                              PHASE 2: DATA COLLECTION                                         │
└──────────────────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
    │  5A. HUMAN      │      │  5B. LLM JUDGE  │      │  5C. OUTCOME    │
    │     LABELS      │      │     OUTPUTS     │      │     EVIDENCE    │
    └─────────────────┘      └─────────────────┘      └─────────────────┘
           │                        │                        │
           ▼                        ▼                        ▼
    50 annotations            652 judge evals          652 grading results
    (target was 350)          per unit                 (ALL BASELINES FAILED)


┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                              PHASE 3: ANALYSIS                                                │
└──────────────────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
    │  6A. DETECTION  │      │  6B. CALIBRATION│      │  6C. AGREEMENT  │
    │     METRICS     │ ───► │     METRICS     │ ◄─── │     + CLAIM     │
    └─────────────────┘      └─────────────────┘      └─────────────────┘
           │                        │                        │
           ▼                        ▼                        ▼
    PDR, PNDR, SLA, etc.    CCorr (BROKEN!)          Human-Judge agreement
```

---

## Stage 1: Trajectory Sources

### What It Is
Raw agent trajectories from 3 benchmarks - traces of agents trying to complete tasks.

### Numbers
```
┌─────────────────────────────────────────────────────────────┐
│                    600 TOTAL TRAJECTORIES                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│   │  TOOLBENCH  │  │    GAIA     │  │  SWE-BENCH  │        │
│   │    400      │  │    100      │  │    100      │        │
│   │   (67%)     │  │   (17%)     │  │   (17%)     │        │
│   └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│   Tool-use tasks   Multi-step       Code editing           │
│   with APIs        reasoning        in repos               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Code Reference
```
File: src/data/loaders.py
Functions:
  - load_toolbench_trajectories()     # Lines ~200-400
  - load_gaia_trajectories()          # Lines ~400-600
  - load_swebench_trajectories()      # Lines ~600-800
  
Filter: check_toolbench_strict()      # Lines 1434-1506
  - Requires task_success=True (agent self-reported completion)
  - Requires return_type="give_answer"
```

### Example Trajectory Structure
```json
{
  "trajectory_id": "toolbench_52646",
  "benchmark": "toolbench",
  "task_description": "Detect the language of: 'Bună ziua'",
  "steps": [
    {"step_number": 1, "tool_name": "detect_language", "tool_input": {"text": "Bună ziua"}, "tool_output": "Romanian"},
    {"step_number": 2, "tool_name": "Finish", "tool_arguments": {"return_type": "give_answer", "final_answer": "..."}}
  ],
  "task_success": true  // <-- AGENT SELF-REPORTED (not objective!)
}
```

---

## Stage 2: Typed Representation

### What It Is
Enriches raw trajectories with semantic information about each step.

### Schema: TypedStep
```
┌────────────────────────────────────────────────────────────────────────┐
│                           TypedStep                                     │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  step_index: int              # Position in trajectory (0-based)        │
│  step_role: Enum              # What this step does                     │
│      ├── "planning"           # Agent reasoning/planning                │
│      ├── "tool_call"          # Calling an external tool                │
│      ├── "observation"        # Tool output/result                      │
│      └── "terminal"           # Final answer                            │
│                                                                         │
│  tool_name: str               # e.g., "detect_language", "Finish"       │
│  tool_arguments: dict         # Input parameters                        │
│  extracted_value: Any         # Key data extracted from this step       │
│  dependencies: List[int]      # Which earlier steps this depends on     │
│                                                                         │
│  perturbable_slots: List[Slot]  # What can be perturbed                 │
│      └── Slot(name, value_type, criticality)                            │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Code Reference
```
File: src/typing/schema.py        # TypedStep, TypedTrajectory classes
File: src/typing/typer.py         # Trajectory typing logic
```

---

## Stage 3: Controlled Perturbations

### What It Is
Inject known errors into trajectories to test if judges can detect them.

### Distribution (Actual from exp_trajectory_sampling_v7)
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         652 EVALUATION UNITS                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐               │
│  │     PLACEBO      │  │   FINE-GRAINED   │  │  COARSE-GRAINED  │               │
│  │      201         │  │      215         │  │      236         │               │
│  │     (31%)        │  │     (33%)        │  │     (36%)        │               │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘               │
│                                                                                  │
│  Expected: 20%          Expected: 50%          Expected: 30%                     │
│  (Control group -       (Small targeted        (Major structural                 │
│   no real error)         parameter changes)     changes)                         │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  BREAKDOWN BY TYPE:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  placebo/synonym:           201 (31%) ████████████████                  │    │
│  │  fine_grained/wrong_param:  201 (31%) ████████████████                  │    │
│  │  coarse/skipped_prereq:     137 (21%) ███████████                       │    │
│  │  coarse/wrong_plan:          99 (15%) ████████                          │    │
│  │  other:                      14  (2%) █                                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Perturbation Types Explained
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ TYPE                    │ WHAT CHANGES                │ EXAMPLE                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│ placebo/synonym         │ Word swaps, no semantic     │ "retrieve" → "fetch"    │
│                         │ change (control group)      │ "search" → "look up"    │
├─────────────────────────────────────────────────────────────────────────────────┤
│ fine_grained/           │ Wrong parameter value       │ limit=10 → limit=1      │
│ wrong_parameter         │                             │ lang="en" → lang="es"   │
├─────────────────────────────────────────────────────────────────────────────────┤
│ coarse/skipped_prereq   │ Delete a required step      │ Remove authentication   │
│                         │                             │ step before API call    │
├─────────────────────────────────────────────────────────────────────────────────┤
│ coarse/wrong_plan       │ Replace step with wrong     │ Use wrong tool entirely │
│                         │ approach                    │                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Position Distribution (PROBLEM!)
```
WHERE perturbations occur in trajectories:

Position 0.0-0.2 (early):  ~10%  ██
Position 0.2-0.4:          ~10%  ██
Position 0.4-0.6:          ~10%  ██
Position 0.6-0.8:          ~20%  ████
Position 0.8-1.0 (late):   ~50%  ██████████  <-- HALF AT THE END!

⚠️  ISSUE: 48.9% of perturbations are at the LAST step
    This means we're not testing "early planning errors that propagate"
```

### Artifact Contamination (PROBLEM!)
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ARTIFACTS LEAKED INTO DATA                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Units with "_old" suffix:        18 / 652 (2.8%)                               │
│  Units with "_mutated" tag:       24 / 652 (3.7%)                               │
│                                                                                  │
│  BY PERTURBATION TYPE:                                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐      │
│  │  wrong_parameter:    184/201 (91.5%) have artifacts  ⚠️               │      │
│  │  threshold_shift:      2/5   (40.0%) have artifacts                   │      │
│  │  skipped_prereq:       1/137 (0.7%)  have artifacts                   │      │
│  │  wrong_plan:           1/99  (1.0%)  have artifacts                   │      │
│  │  placebo:              0/201 (0.0%)  have artifacts  ✓                │      │
│  └───────────────────────────────────────────────────────────────────────┘      │
│                                                                                  │
│  Example artifact:                                                               │
│  {"return_type": "give_answer_old"}  <-- Should be "give_answer"                │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Code Reference
```
File: src/perturbations/generator_v2.py     # Main orchestrator
File: src/perturbations/placebo.py          # Placebo generators
  - PlaceboSynonymGenerator                 # Lines 381-507
  - SYNONYM_PAIRS                           # Lines 389-421 (word mappings)
File: src/perturbations/fine_grained.py     # Fine-grained generators
  - Line 1004-1005: adds "_old" suffix      # <-- ARTIFACT SOURCE
  - Line 125: adds "_mutated" fallback      # <-- ARTIFACT SOURCE
```

---

## Stage 4: Evaluation Unit

### What It Is
Pairs a baseline trajectory with its perturbed version for side-by-side comparison.

### Schema
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           EVALUATION UNIT                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  evaluation_unit_id: "eval::toolbench_52646::006"                               │
│                       ├── source: eval (evaluation)                              │
│                       ├── trajectory: toolbench_52646                            │
│                       └── variant: 006 (6th perturbation of this trajectory)    │
│                                                                                  │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐               │
│  │    BASELINE TRAJECTORY      │  │   PERTURBED TRAJECTORY      │               │
│  │    (original, no changes)   │  │   (with injected error)     │               │
│  │                             │  │                             │               │
│  │  Steps: [s1, s2, s3, ...]   │  │  Steps: [s1, s2', s3, ...]  │               │
│  │                             │  │              ↑               │               │
│  │                             │  │         changed step         │               │
│  └─────────────────────────────┘  └─────────────────────────────┘               │
│                                                                                  │
│  METADATA:                                                                       │
│  ├── perturbation_class: "fine_grained"                                         │
│  ├── perturbation_type: "wrong_parameter"                                       │
│  ├── target_step_index: 2                                                       │
│  ├── original_value: "limit=10"                                                 │
│  ├── perturbed_value: "limit=1"                                                 │
│  └── expected_impact: 2 (moderate)                                              │
│                                                                                  │
│  BLINDING (for judge evaluation):                                                │
│  ├── trajectory_a: perturbed  ⚠️ ALWAYS perturbed (not randomized!)            │
│  └── trajectory_b: baseline                                                      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Code Reference
```
File: src/evaluation/unit_assembler.py    # Creates evaluation units
File: src/evaluation/blinding.py          # A/B assignment (broken - not randomized)
```

---

## Stage 5A: Human Labels

### What It Is
Human annotators evaluate evaluation units for ground truth.

### Numbers
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           HUMAN ANNOTATIONS                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Target: 350 units                                                               │
│  Actual: 50 units  ⚠️  (only 14% of target)                                     │
│                                                                                  │
│  ANNOTATION SCHEMA:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  error_detected: bool           # Did you find an error?                │    │
│  │  error_trajectory: "A" | "B"    # Which trajectory has the error?       │    │
│  │  error_step: int                # Which step?                           │    │
│  │  error_type: Enum               # What kind of error?                   │    │
│  │      ├── "planning"                                                     │    │
│  │      ├── "tool_selection"                                               │    │
│  │      ├── "parameter"                                                    │    │
│  │      ├── "data_reference"                                               │    │
│  │      └── "unclear"                                                      │    │
│  │  impact_tier: 0-3               # How bad? (none/minor/moderate/major)  │    │
│  │  confidence: 1-5                # How sure are you?                     │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Code Reference
```
File: src/human_labels/schema.py          # HumanAnnotation dataclass
File: src/annotation/unit_annotator.py    # Annotation UI
File: ops/annotation_ui.py                # CLI for annotation
```

---

## Stage 5B: LLM Judge Outputs

### What It Is
LLM judges evaluate the same evaluation units.

### Judge Configuration
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           JUDGE SETUP                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  MODELS USED:                                                                    │
│  ├── claude-sonnet-4.5 (primary)                                                │
│  └── gpt-oss-120b                                                               │
│                                                                                  │
│  EVALUATION MODES:                                                               │
│  ├── single_trajectory    # Evaluate one trajectory in isolation                │
│  ├── blinded_pair         # Compare A vs B without knowing which is perturbed   │
│  └── labeled_pair         # Know which is baseline/perturbed (for calibration)  │
│                                                                                  │
│  THIS EXPERIMENT USED: blinded_pair                                              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### The Judge Prompt (CRITICAL!)
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PROMPT VARIABLE: BLINDED_PAIR_SYSTEM                                            │
│  FILE: src/judges/unit_prompts.py (Lines 51-59)                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  """You are an expert evaluator comparing two AI agent trajectories.            │
│  Both trajectories attempt the same task. One may contain an error.  ← PRIMING! │
│                                                                                  │
│  Your job is to:                                                                 │
│  1. Determine if either trajectory contains an error  ← FRAMES AS ERROR FINDING │
│  2. If so, identify which trajectory and which step                              │
│  3. Compare overall quality                                                      │
│                                                                                  │
│  You do NOT know which trajectory is "baseline" or "perturbed" -                │
│  evaluate purely on quality.                                                     │
│  Respond in JSON format only."""                                                 │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  PROMPT VARIABLE: BLINDED_PAIR_USER                                              │
│  FILE: src/judges/unit_prompts.py (Lines 62-83)                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  """Task: {{task_text}}                                                          │
│                                                                                  │
│  Trajectory A:                                                                   │
│  {{formatted_trajectory_a}}                                                      │
│                                                                                  │
│  Trajectory B:                                                                   │
│  {{formatted_trajectory_b}}                                                      │
│                                                                                  │
│  Compare these trajectories. Respond with JSON:                                  │
│  {                                                                               │
│    "overall_score_a": <0-100>,                                                   │
│    "overall_score_b": <0-100>,                                                   │
│    "error_detected": <true/false>,                                               │
│    "error_trajectory": <"A"|"B"|"neither"|"both">,                               │
│    "error_confidence": <0-1>,                                                    │
│    "predicted_error_step": <step number in error trajectory, or null>,           │
│    "predicted_error_type": <type or null>,                                       │
│    "preference": <"A"|"B"|"tie">,                                                │
│    "predicted_impact_score": <0-1>,                                              │
│    "predicted_failure_prob": <0-1>,                                              │
│    "comparison_explanation": "<brief explanation>"                               │
│  }"""                                                                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Judge Output Schema
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  DATACLASS: Section5JudgeOutput                                                  │
│  FILE: src/judges/schema.py                                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  judge_output_id: str                                                            │
│  evaluation_unit_id: str                                                         │
│  judge_model: str                # "claude-sonnet-4.5"                          │
│  judge_mode: str                 # "blinded_pair"                               │
│                                                                                  │
│  # Detection outputs                                                             │
│  error_detected: bool            # Did judge find an error?                     │
│  error_confidence: float         # 0-1 confidence                               │
│  predicted_error_step: int       # Which step?                                  │
│  predicted_error_type: str       # "planning" | "tool_selection" | etc.         │
│                                                                                  │
│  # Impact outputs                                                                │
│  predicted_impact_score: float   # 0-1 estimated severity                       │
│  predicted_failure_prob: float   # 0-1 will this fail?                          │
│                                                                                  │
│  # Pair comparison (blinded_pair mode only)                                      │
│  overall_score_a: int            # 0-100                                        │
│  overall_score_b: int            # 0-100                                        │
│  error_trajectory: str           # "A" | "B" | "neither" | "both"               │
│  preference: str                 # "A" | "B" | "tie"                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Code Reference
```
File: src/judges/unit_prompts.py      # BLINDED_PAIR_SYSTEM, BLINDED_PAIR_USER
File: src/judges/unit_runner.py       # UnitJudgeRunner.run_blinded_pair()
File: src/judges/schema.py            # Section5JudgeOutput
File: src/judges/claude_judge.py      # ClaudeJudge class
File: src/judges/parser.py            # parse_judge_response()
```

---

## Stage 5C: Outcome Evidence

### What It Is
Objective measurement of whether the perturbation actually degraded task outcome.

### The Metric: Outcome Degradation (OD)
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      OUTCOME DEGRADATION FORMULA                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  outcome_degradation = baseline_score - perturbed_score                          │
│                                                                                  │
│  WHERE:                                                                          │
│    baseline_score  = Did ORIGINAL trajectory pass? (0 or 1)                      │
│    perturbed_score = Did PERTURBED trajectory pass? (0 or 1)                     │
│                                                                                  │
│  INTERPRETATION:                                                                 │
│    OD = +1  →  Baseline passed, perturbed failed (perturbation hurt)            │
│    OD =  0  →  Both same outcome (no effect)                                    │
│    OD = -1  →  Baseline failed, perturbed passed (perturbation helped?!)        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### THE CRITICAL BUG
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ⚠️⚠️⚠️  ALL 652 BASELINES "FAILED"  ⚠️⚠️⚠️                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  WHAT HAPPENED:                                                                  │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  STAGE 1 (Sampling)          │  STAGE 5C (Grading)                     │    │
│  │  ─────────────────────────── │  ─────────────────────────────────────  │    │
│  │  Uses: task_success field    │  Uses: HeuristicGrader                  │    │
│  │                              │                                         │    │
│  │  Meaning: Agent SAID it      │  Meaning: Is the answer ACTUALLY        │    │
│  │  completed ("give_answer")   │  good? (checks for failure patterns)    │    │
│  │                              │                                         │    │
│  │  Filter: return_type =       │  Checks: Does answer contain            │    │
│  │          "give_answer"       │          "unable to", "error",          │    │
│  │                              │          "sorry", etc.?                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  RESULT:                                                                         │
│    Trajectories were selected because agent CLAIMED success                      │
│    But grading found the answers were actually BAD                               │
│    → All baselines "failed" → outcome_degradation = 0 for everything            │
│    → NO VARIANCE IN GROUND TRUTH → CCorr MEANINGLESS                            │
│                                                                                  │
│  NUMBERS:                                                                        │
│    Baseline pass rate:        0 / 652  (0%)                                     │
│    outcome_degradation = 0:   635 / 652  (97%)                                  │
│    outcome_degradation < 0:   17 / 652  (3%) - perturbed did BETTER             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Code Reference
```
File: src/outcome_evidence/metrics.py         # outcome_degradation calculation (Lines 19-31)
File: src/outcome_evidence/tier_3/grading.py  # HeuristicGrader (Lines 290-405)
File: src/data/loaders.py                     # task_success parsing (Lines 605-615)
```

---

## Stage 6A: Detection Metrics

### What They Measure
How well does the judge detect that perturbations exist?

### Metric Definitions
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DETECTION METRICS                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  PDR = Perturbation Detection Rate                                               │
│      = (# non-placebos where judge said error_detected=true)                    │
│        ─────────────────────────────────────────────────────                    │
│                    (# total non-placebos)                                        │
│                                                                                  │
│      "When there IS an error, how often does judge find it?"                     │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  PNDR = Placebo Non-Detection Rate                                               │
│       = (# placebos where judge said error_detected=false)                       │
│         ─────────────────────────────────────────────────────                   │
│                     (# total placebos)                                           │
│                                                                                  │
│       "When there is NO error, how often does judge correctly say 'no error'?"  │
│                                                                                  │
│       ⚠️ AS REPORTED: 8% (interpreted as 92% false positive rate)               │
│       ⚠️ ACTUAL: ~92% (see correction below)                                    │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  SLA = Step Localization Accuracy                                                │
│      = (# units where predicted_step == actual_step)                            │
│        ─────────────────────────────────────────────                            │
│             (# units where judge detected error)                                 │
│                                                                                  │
│      "When judge finds an error, does it point to the RIGHT step?"              │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  TIA = Type Identification Accuracy                                              │
│      = (# units where predicted_type == actual_type)                            │
│        ─────────────────────────────────────────────                            │
│             (# units where judge detected error)                                 │
│                                                                                  │
│      "When judge finds an error, does it label it correctly?"                   │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  CER = Critical Error Recall                                                     │
│      = (# critical errors detected) / (# total critical errors)                 │
│                                                                                  │
│      "For the WORST errors, how often does judge catch them?"                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Results (As Reported)
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DETECTION RESULTS                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  METRIC   │ VALUE   │ INTERPRETATION                                            │
│  ─────────┼─────────┼─────────────────────────────────────────────────────────  │
│  PDR      │ 0.909   │ Judge detects 91% of real perturbations        ✓ Good    │
│  PNDR     │ 0.080   │ Judge correctly ignores only 8% of placebos    ⚠️ Bad?   │
│  SLA      │ 0.261   │ Judge points to correct step 26% of time       ⚠️ Low    │
│  TIA      │ 0.298   │ Judge labels correct type 30% of time          ⚠️ Low    │
│  CER      │ 0.897   │ Judge catches 90% of critical errors           ✓ Good    │
│  AUC      │ 0.518   │ Detection AUC near random (0.5)                ⚠️ Bad?   │
│  F1       │ 0.784   │ F1 score 78%                                   ~ OK      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### CORRECTION: The 92% "False Positive" Was Wrong
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ⚠️  PNDR = 8% WAS MISINTERPRETED                                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  WHAT WAS COMPUTED:                                                              │
│    "How often does judge say NO error on placebos?"                             │
│    Answer: 8% → Interpreted as "92% false positive rate"                        │
│                                                                                  │
│  THE PROBLEM:                                                                    │
│    Placebos are created from ORIGINAL trajectories that ALREADY HAVE ERRORS!    │
│    A synonym swap doesn't FIX the underlying problems.                           │
│                                                                                  │
│  WHAT ACTUALLY HAPPENED (checked against original trajectories):                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Judge flagged BOTH placebo AND original:  132 / 144 (92%)             │    │
│  │    → Judge is CONSISTENT - finds same errors in both                   │    │
│  │                                                                         │    │
│  │  Judge correctly said NO error in both:     11 / 144 (8%)              │    │
│  │    → Judge is CORRECT when trajectory is clean                         │    │
│  │                                                                         │    │
│  │  TRUE FALSE POSITIVE (placebo flagged,                                  │    │
│  │                       original NOT flagged):  1 / 144 (0.7%)           │    │
│  │    → ACTUAL FALSE POSITIVE RATE IS ~1%                                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  CONCLUSION: Judge is actually performing WELL. The 92% reflects that           │
│              92% of original trajectories have real errors (which is            │
│              consistent with finding that all baselines "failed").              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Code Reference
```
File: src/analysis/section6/detection_metrics.py    # PDR, PNDR, SLA, TIA, CER
File: ops/analyze_section6.py                       # Analysis script
```

---

## Stage 6B: Calibration Metrics

### What They Measure
How well does the judge's predicted_impact match actual outcome_degradation?

### Metric Definitions
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CALIBRATION METRICS                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  CCorr = Consequence Correlation                                                 │
│        = Spearman(predicted_impact_score, outcome_degradation)                  │
│                                                                                  │
│        "Does judge's predicted severity correlate with actual outcome?"         │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  |CCE| = Mean Absolute Calibration Error                                         │
│        = mean(|predicted_impact - actual_impact|)                               │
│                                                                                  │
│        "On average, how far off is the judge's prediction?"                     │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  ORR = Over-Reaction Rate                                                        │
│      = (# placebos where judge predicted high impact)                           │
│        ───────────────────────────────────────────────                          │
│                    (# total placebos)                                            │
│                                                                                  │
│      "How often does judge overestimate impact on harmless changes?"            │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  URR = Under-Reaction Rate                                                       │
│      = (# critical errors where judge predicted low impact)                     │
│        ─────────────────────────────────────────────────────                    │
│                    (# total critical errors)                                     │
│                                                                                  │
│      "How often does judge underestimate impact on severe errors?"              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Results
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CALIBRATION RESULTS                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  METRIC       │ VALUE   │ INTERPRETATION                                        │
│  ─────────────┼─────────┼───────────────────────────────────────────────────── │
│  CCorr        │ -0.076  │ Almost no correlation with outcome       ⚠️ BAD      │
│  |CCE|        │  0.645  │ Average error of 0.65 on 0-1 scale       ⚠️ BAD      │
│  ORR          │  0.764  │ 76% over-reaction on placebos            ⚠️ BAD      │
│  URR          │  N/A    │ No critical impacts in data              ⚠️ CAN'T    │
│  Failure-ECE  │  0.295  │ Expected calibration error 30%           ⚠️ BAD      │
│                                                                                  │
│  ⚠️⚠️⚠️  BUT WAIT: These metrics are MEANINGLESS  ⚠️⚠️⚠️                        │
│                                                                                  │
│  Remember: outcome_degradation = 0 for 97% of data                              │
│  You can't measure correlation with a constant!                                  │
│                                                                                  │
│  CCorr = Spearman(predicted, [0, 0, 0, 0, 0, ...])                              │
│        = undefined / meaningless                                                 │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Code Reference
```
File: src/analysis/section6/calibration_metrics.py    # CCorr, CCE, ORR, URR
```

---

## Stage 6C: Agreement + Main Claim

### Human-Judge Agreement
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    HUMAN-JUDGE AGREEMENT (n=50)                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  METRIC              │ VALUE   │ INTERPRETATION                                 │
│  ────────────────────┼─────────┼──────────────────────────────────────────────  │
│  Detection Agreement │  0.580  │ 58% agree on whether error exists              │
│  Type Agreement      │  0.000  │ 0% agree on error type        ⚠️ BUG!         │
│  Impact Tier MAE     │  0.857  │ ~1 tier difference on average                  │
│  Impact Bias         │ +0.214  │ Judge rates impact higher than humans          │
│                                                                                  │
│  ⚠️  0% TYPE AGREEMENT IS A BUG, NOT A FINDING                                  │
│                                                                                  │
│  The code compares:                                                              │
│    mapped_judge_type (e.g., "structural")                                       │
│    vs.                                                                           │
│    raw_human_type (e.g., "planning")                                            │
│                                                                                  │
│  But "planning" gets mapped to "structural" → comparison fails                  │
│                                                                                  │
│  ACTUAL TYPE AGREEMENT (if fixed): ~19.2% (5/26)                                │
│                                                                                  │
│  BUG LOCATION: src/analysis/section6/schema.py:map_error_type_to_family()       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### The Main Claim
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MAIN CLAIM TEST                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  HYPOTHESIS:                                                                     │
│  "LLM judges detect errors better than they estimate downstream consequence"    │
│                                                                                  │
│  OPERATIONALIZED AS:                                                             │
│    PDR > CCorr                                                                   │
│    (detection rate) > (consequence correlation)                                 │
│                                                                                  │
│  RESULTS:                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  PDR         = 0.909                                                    │    │
│  │  CCorr       = -0.076                                                   │    │
│  │  Gap         = 0.985                                                    │    │
│  │  95% CI      = [0.908, 1.062]                                           │    │
│  │  p-value     = <0.001                                                   │    │
│  │                                                                         │    │
│  │  CONCLUSION: Claim "supported"                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ⚠️⚠️⚠️  BUT THIS IS VACUOUS  ⚠️⚠️⚠️                                            │
│                                                                                  │
│  CCorr is meaningless because outcome_degradation has no variance.              │
│  Comparing PDR to a meaningless number proves nothing.                           │
│                                                                                  │
│  It's like saying:                                                               │
│    "I can detect rain better than I can predict a constant"                     │
│    → Mathematically true, scientifically meaningless                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Code Reference
```
File: src/analysis/section6/agreement_metrics.py    # Human-judge agreement
File: src/analysis/section6/schema.py               # map_error_type_to_family() BUG
File: ops/analyze_section6.py                       # Main claim test
```

---

## Summary: What's Broken vs. What Works

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              STATUS SUMMARY                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  STAGE    │ STATUS        │ ISSUE                        │ FIX                  │
│  ─────────┼───────────────┼──────────────────────────────┼───────────────────── │
│  1        │ ✓ OK          │ -                            │ -                    │
│  2        │ ✓ OK          │ -                            │ -                    │
│  3        │ ⚠️ Issues     │ 49% perturbations at end     │ Control position     │
│           │               │ 42% artifacts in non-placebo │ Clean code           │
│  4        │ ⚠️ Issue      │ A/B not randomized           │ Randomize            │
│  5A       │ ⚠️ Small n    │ Only 50/350 annotations      │ Get more             │
│  5B       │ ✓ GOOD        │ Judge actually performs well │ -                    │
│  5C       │ ❌ BROKEN     │ Sampling/grading mismatch    │ Re-sample with       │
│           │               │ → no outcome variance        │ aligned definitions  │
│  6A       │ ⚠️ Misreported│ PNDR calculation wrong       │ Compare to original  │
│  6B       │ ❌ UNTESTABLE │ CCorr meaningless            │ Fix 5C first         │
│  6C       │ ⚠️ Bug + Void │ Type agreement bug           │ Fix mapping          │
│           │               │ Main claim untestable        │ Fix 5C first         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

### What You CAN Claim (with current data)
1. LLM judges have **high detection sensitivity** (PDR = 91%)
2. LLM judges have **low false positive rate** (true FP ~1%, not 92%)
3. Judges are **consistent** across placebo/original pairs
4. Judges are **poor at localization** (SLA = 26%) and **type identification** (TIA = 30%)

### What You CANNOT Claim (yet)
1. Anything about **calibration** (CCorr) - need outcome variance
2. Anything about the **main hypothesis** (detection vs. calibration) - need 5C fixed
3. Reliable **human-judge type agreement** - need to fix taxonomy bug

### Priority Fixes
1. **Fix 5C**: Re-sample trajectories where baseline PASSES the same grader
2. **Fix taxonomy bug**: Compare same type taxonomy for humans and judges
3. **Clean artifacts**: Remove `_old` suffix from perturbation code
4. **Improve position distribution**: Control for early vs. late perturbations
