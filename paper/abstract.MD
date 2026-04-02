# Abstract

Large language model (LLM) judges are increasingly used to evaluate agentic systems, yet most current evaluation methods implicitly treat trajectory errors as if they were uniformly important. In practice, failures in multi-step agent trajectories are highly asymmetric: some errors are recoverable, while others occur at structurally critical positions—such as early planning or tool selection—and propagate through the remainder of the trajectory. This creates a central evaluation risk: an LLM judge may appear accurate overall while systematically under-penalizing the errors that matter most for downstream task success.

This project studies **judge blindness to trajectory criticality**. We propose a framework for measuring how error location and type affect downstream propagation, and whether LLM judges calibrate their scores in proportion to that true criticality. Using agent trajectories from benchmarks such as SWE-bench, GAIA, and ToolBench, we construct controlled perturbations at different trajectory positions and measure both downstream task degradation and judge-assigned penalties. We then quantify the mismatch between true error criticality and judge scoring behavior with a new metric, the **Criticality-Calibration Gap (CCG)**.

We expect to find that error propagation is strongly position-dependent, with early planning and tool-selection mistakes producing the largest downstream harm. We further expect that LLM judges will be systematically miscalibrated to this structure: they will often over-penalize visible or verbose local mistakes while under-penalizing early structural failures that cause larger end-to-end degradation. As a result, the positions where agent errors are most consequential may also be the positions where judge reliability is weakest.

The expected contribution of this work is threefold:  
1. a benchmarked framework for studying criticality in agent trajectories,  
2. empirical evidence that current LLM judges are not reliably calibrated to propagation-sensitive failures, and  
3. a metric and analysis protocol for more reliable evaluation of agentic systems.  

More broadly, this work argues that agent evaluation should move beyond aggregate judge agreement and toward **criticality-aware evaluation**, where errors are assessed not only by whether they are observable, but by how much they affect the trajectory’s final outcome.