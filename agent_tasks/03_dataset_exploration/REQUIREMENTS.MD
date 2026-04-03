## You are an autonomous researcher with instructions outlined in CLAUDE.md

### Abstract:

This research investigates whether LLM-based judges are systematically miscalibrated when evaluating agent trajectories. Specifically, we hypothesize that judges under-penalize early structural failures (e.g., flawed planning, wrong tool selection) while over-penalizing late local mistakes (e.g., parameter typos). Early errors are more critical because they propagate downstream and corrupt subsequent steps, yet judges may focus on surface-level correctness rather than causal impact. We introduce the Criticality-Calibration Gap (CCG) metric to quantify this miscalibration by comparing judge penalty scores against ground-truth criticality derived from task success degradation and downstream error propagation. Through controlled perturbation experiments on ToolBench and GAIA benchmarks, we aim to reveal systematic blind spots in LLM judgment that have implications for agent evaluation, RLHF reward modeling, and AI safety oversight.

### Context : 
Human Annotation and LLMaJ are long pole items. 1 judge can take upto 8 hours over a sample of 500 while human beings can probably burn through 300 samples in 4-5 hours. In /Users/amanzing/Paper-Research/agent_tasks/02_poc_implementation we ran LLMaJ over 600 items. Iterating on this piece would be much much slower. We can speed things up we've good trajectories and perturbations recorded.

### Goal : 
The goal of this tasks is as follows -

### Execution Plan

| Step | Description | Status | Notes |
|------|-------------|--------|-------|
| 1 | Review 02_poc_implementation findings and create HighQualityTrajectoryAndPerturbationsGapAnalysis.MD | ✅ DONE | Gap analysis complete |
| 2 | Explore datasets (ToolBench, GAIA, SWE-bench) using gap analysis | ✅ DONE | All 3 benchmarks explored |
| 3 | Create DatasetDiversity.MD and PerturbationDiversity.MD (~600 samples) | ✅ DONE | Revision 2 complete |
| 4 | STOP - Get documents reviewed | ✅ DONE | First review received |
| 5 | Iterate on feedback until approved | ✅ DONE | 2 iterations, expert approved |
| 6 | Make code changes to ingest datasets and add perturbation types | ✅ DONE | SWE-bench loader, Type D, sampling |
| 7 | Run trajectories for 10 samples per diversity domain | ✅ DONE | 100 samples tested |
| 8 | Validate trajectory results | ✅ DONE | Coverage validated |
| 9 | Run trajectories for all samples | ⏳ PENDING | Full 600 trajectories |
| 10 | Analyze coverage for all trajectories (show numbers) | ⏳ PENDING | |
| 11 | Run perturbations | ⏳ PENDING | |
| 12 | Validate perturbation results | ⏳ PENDING | Quality check |
| 13 | Analyze coverage for all perturbations (show numbers) | ⏳ PENDING | |
| 14 | Task completion | ⏳ PENDING | |

### Approved Design (from Steps 1-5)

**Dataset:**
- ToolBench: 400 trajectories
- GAIA: 100 trajectories (50 Claude + 50 GPT-4o generated)
- SWE-bench: 100 trajectories
- **Total: 600 trajectories**

**Experimental Design:**
- Main sample: 500 trajectories (1 perturbation each)
- Control group: 100 trajectories (all perturbations, for confound check)
- Execution validation: 40 trajectories (actual API re-execution)

**Perturbation Types (7 subtypes):**
- Planning: goal substitution, scope reduction
- Tool Selection: similar tool swap
- Parameter: wrong value, format error, off-by-one
- Data Reference: hallucinated reference

**Judges:** Claude Sonnet 4.5, GPT-4o, Llama 3.1 70B

**Pre-registration:** OSF before full study run

### My help
If you need my help or if there is experiment failure and you can't fix it, trigger a notification.

**How to notify me:**
The notification hook is configured in `~/.claude/settings.json`. When Claude Code needs attention (permission prompts, blockers, etc.), a native macOS notification will appear automatically.

For manual notification in urgent cases, you can run:
```bash
osascript -e 'display notification "Your message here" with title "Claude Code" sound name "default"'
```

Good luck!