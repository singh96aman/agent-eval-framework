## You are an autonomous researcher with instructions outlined in CLAUDE.md

### Abstract:

This research investigates whether LLM-based judges are systematically miscalibrated when evaluating agent trajectories. Specifically, we hypothesize that judges under-penalize early structural failures (e.g., flawed planning, wrong tool selection) while over-penalizing late local mistakes (e.g., parameter typos). Early errors are more critical because they propagate downstream and corrupt subsequent steps, yet judges may focus on surface-level correctness rather than causal impact. We introduce the Criticality-Calibration Gap (CCG) metric to quantify this miscalibration by comparing judge penalty scores against ground-truth criticality derived from task success degradation and downstream error propagation. Through controlled perturbation experiments on ToolBench and GAIA benchmarks, we aim to reveal systematic blind spots in LLM judgment that have implications for agent evaluation, RLHF reward modeling, and AI safety oversight.

### Context : 
Human Annotation and LLMaJ are long pole items. 1 judge can take upto 8 hours over a sample of 500 while human beings can probably burn through 300 samples in 4-5 hours. In /Users/amanzing/Paper-Research/agent_tasks/02_poc_implementation we ran LLMaJ over 600 items. Iterating on this piece would be much much slower. We can speed things up we've good trajectories and perturbations recorded.

### Goal : 
The goal of this tasks is as follows -
1. Review /Users/amanzing/Paper-Research/agent_tasks/02_poc_implementation and particularly /Users/amanzing/Paper-Research/agent_tasks/02_poc_implementation/Findings&NextSteps.MD. See the LLMaJ results about the trajectories and perturbations. Come up with a HighQualityTrajectoryAndPerturbationsGapAnalysis.MD. Think about different domains where agent would be asked to work and do we have enough coverage over them.
2. Use HighQualityTrajectoryAndPerturbationsGapAnalysis.MD to explore dataset with these trajectories and where we pertub and actually check LLMaJ performance. Toolbench is a great start - /Users/amanzing/Paper-Research/data/toolbench/data/. There is GAIA dataset as well.
3. Use #2 to define the DatasetDiversity.MD and PerturbationDiversity.MD. We want to have ~500 samples for few domains and then we explore what types of perturbations we can make to them. We then analyze how LLMaJ would be analyzing them.
4. STOP. Get DatasetDiversity.MD and PerturbationDiversity.MD reviewed by me
5. Iterate on my feedback in these docs until I approve do not move to Step #6.
6. Make code changes to account to ingest these datasets and add domains, perturbation types in config.
7. Run Trajectories and Perturbations for these.
8. Analyze methodically if we have good coverage over trajectories and perturbations. Show numbers
9. Task is completed

### My help
If you need my help or if there is experiment failure and you can't fix it. Please send me a slack message at #claude-communications. Instructions are in Claude.md

Good luck!