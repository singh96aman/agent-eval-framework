# Task: Research Foundation & MVP Scoping

## Abstract (Project Context)

Large language model (LLM) judges are increasingly used to evaluate agentic systems, yet most current evaluation methods implicitly treat trajectory errors as if they were uniformly important. In practice, failures in multi-step agent trajectories are highly asymmetric: some errors are recoverable, while others occur at structurally critical positions—such as early planning or tool selection—and propagate through the remainder of the trajectory. This creates a central evaluation risk: an LLM judge may appear accurate overall while systematically under-penalizing the errors that matter most for downstream task success.

This project studies **judge blindness to trajectory criticality**. We propose a framework for measuring how error location and type affect downstream propagation, and whether LLM judges calibrate their scores in proportion to that true criticality. Using agent trajectories from benchmarks such as SWE-bench, GAIA, and ToolBench, we construct controlled perturbations at different trajectory positions and measure both downstream task degradation and judge-assigned penalties. We then quantify the mismatch between true error criticality and judge scoring behavior with a new metric, the **Criticality-Calibration Gap (CCG)**.

We expect to find that error propagation is strongly position-dependent, with early planning and tool-selection mistakes producing the largest downstream harm. We further expect that LLM judges will be systematically miscalibrated to this structure: they will often over-penalize visible or verbose local mistakes while under-penalizing early structural failures that cause larger end-to-end degradation. As a result, the positions where agent errors are most consequential may also be the positions where judge reliability is weakest.

The expected contribution of this work is threefold:  
1. a benchmarked framework for studying criticality in agent trajectories,  
2. empirical evidence that current LLM judges are not reliably calibrated to propagation-sensitive failures, and  
3. a metric and analysis protocol for more reliable evaluation of agentic systems.  

More broadly, this work argues that agent evaluation should move beyond aggregate judge agreement and toward **criticality-aware evaluation**, where errors are assessed not only by whether they are observable, but by how much they affect the trajectory's final outcome.

---

## Objective

Conduct foundational research to:
1. **Scope a minimal viable proof-of-concept (POC)** that can validate the core hypothesis
2. **Survey existing literature** to confirm novelty and identify gaps
3. **Define concrete MVP requirements** that are achievable and scientifically rigorous

This task provides the research foundation needed before implementation begins.

---

## Tasks

### 1. Literature Review
Conduct a comprehensive survey of related work covering:

#### Core Areas:
- **LLM-as-judge evaluation methods** (e.g., G-Eval, PRD, LLM-eval, judge calibration studies)
- **Agent trajectory evaluation** (benchmarks like SWE-bench, GAIA, ToolBench, AgentBench)
- **Error propagation in sequential systems** (cascading failures, error taxonomy in agents)
- **Criticality metrics** (position-dependent scoring, structural vs. local errors)
- **Evaluation reliability** (inter-annotator agreement, judge-human alignment)

#### Key Questions:
- Has anyone studied position-dependent error importance in agent trajectories?
- Are there existing metrics for "criticality" in multi-step reasoning?
- What do we know about LLM judge failure modes?
- How do current benchmarks handle error location vs. error type?
- What related work exists in software testing, fault localization, or causal tracing?

#### Deliverable:
`paper/LITERATURE_REVIEW.MD` containing:
- Summary of 15-25 key papers
- Identified gaps in current research
- Positioning statement: what makes this work novel
- Citations in BibTeX format (can be appended or separate file)

---

### 2. MVP/POC Definition

Design a minimal proof-of-concept that:
- **Validates the core hypothesis**: LLM judges are miscalibrated to trajectory criticality
- **Is achievable** with limited resources (small dataset, simple perturbations)
- **Produces interpretable results** that guide full implementation

#### POC Scope Decisions:
Answer and document:
1. **Which benchmark(s)** to use? (Start with 1-2, e.g., subset of ToolBench or GAIA)
2. **How many trajectories?** (e.g., 50-100 for POC)
3. **What perturbation types?** (e.g., early planning error, tool selection error, parameter error)
4. **How many positions?** (e.g., early, middle, late)
5. **Which judge model(s)?** (e.g., GPT-4, Claude-3.5-Sonnet)
6. **How to measure downstream impact?** (task success degradation, subsequent error rate)
7. **Simplified CCG metric**: What's the minimal version for POC?
8. **Success criteria**: What result would confirm we're on the right track?

#### Deliverable:
`paper/POC_REQUIREMENTS.MD` containing:
- Clear scope definition (dataset, sample size, perturbation types)
- Experimental design for POC
- Measurement protocol (what to track, how to score)
- Expected timeline (hours/days of compute and human effort)
- Success/failure criteria
- Risks and mitigations
- What we'll learn even if results are negative

---

## Exit Criteria

This task is **complete** when:

1. ✅ `paper/LITERATURE_REVIEW.MD` exists and contains:
   - At least 15 relevant papers surveyed
   - Clear novelty/gap statement
   - Positioning of our work

2. ✅ `paper/POC_REQUIREMENTS.MD` exists and contains:
   - Concrete, scoped POC design
   - Clear experimental protocol
   - Success criteria and timeline
   - Risk assessment

3. ✅ Both documents are **reviewed and approved** by the human researcher

4. ✅ The human researcher confirms:
   - The POC is achievable
   - The work is sufficiently novel
   - We're ready to proceed to implementation

---

## Constraints

- Use AI heavily for literature search, summarization, and initial drafting
- Human researcher retains final authority on scientific interpretation
- Prioritize clarity and actionability over comprehensiveness
- If novelty is questionable, surface that early
- If POC seems infeasible, propose alternatives

---

## Notes

- This is a **research and planning task**, not an implementation task
- No code needs to be written for this task
- Focus on building a strong foundation for the work ahead
- It's okay to iterate on scope based on what literature reveals
- Document key decisions and reasoning, not just final outputs

---

## Inputs
- This `Requirements.MD` (task definition)
- `paper/abstract.MD` (project context)
- Access to academic search (arXiv, Semantic Scholar, Google Scholar)
- Human researcher's domain knowledge and feedback

## Outputs (in paper/)
- `paper/LITERATURE_REVIEW.MD` - Survey of related work
- `paper/POC_REQUIREMENTS.MD` - MVP scope and design
- Optional: `paper/references.bib` - BibTeX citations
- Optional: `paper/notes/` - Rejected scope options and reasoning
