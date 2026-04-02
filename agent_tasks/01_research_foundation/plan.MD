# Implementation Plan: Research Foundation & MVP Scoping

## Context

This task establishes the research foundation for a novel study on **judge blindness to trajectory criticality** in LLM-evaluated agent systems. The core hypothesis: LLM judges are systematically miscalibrated to error criticality - over-penalizing visible local mistakes while under-penalizing early structural failures that cause larger downstream degradation.

**Why this task matters:**
- Validates novelty of the research direction
- Defines a minimal, achievable POC to test the core hypothesis
- Prevents wasted effort on infeasible or non-novel work
- Provides clear roadmap for implementation

**Current state:**
- Task status: not_started
- Repo structure exists with paper/, src/, tests/, data/, results/ directories
- Abstract exists at `paper/abstract.MD`
- No literature review or POC design yet

---

## Deliverables

### 1. Literature Review Document
**Location:** `paper/LITERATURE_REVIEW.MD`

**Contents:**
- Survey of 15-25 key papers across 5 categories:
  1. LLM-as-judge evaluation methods (5-7 papers)
  2. Agent trajectory evaluation (5-7 papers)
  3. Error propagation in sequential systems (3-5 papers)
  4. Criticality metrics (2-4 papers)
  5. Gap analysis and positioning (2-3 papers)
- Clear novelty statement and research positioning
- BibTeX citations (in document or separate `paper/references.bib`)

### 2. POC Requirements Document
**Location:** `paper/POC_REQUIREMENTS.MD`

**Contents:**
- Concrete experimental design with specific choices:
  * Benchmark: ToolBench (50-100 trajectories)
  * Perturbations: 3 types × 3 positions = 9 conditions
  * Judges: GPT-4 and Claude-3.5-Sonnet
  * Impact measurement: Task success degradation + subsequent error rate
  * Simplified CCG metric for POC
- Success criteria (statistical thresholds)
- Timeline (2-3 weeks) and resource estimates ($50-200)
- Risk assessment and mitigation strategies
- Alternative POC designs if primary approach is infeasible

---

## Implementation Approach

### Phase 1: Literature Review (7-10 hours, 1 week)

**Search Strategy:**
Use web search tools to find papers from arXiv, Semantic Scholar, and Google Scholar.

**Category 1: LLM-as-Judge (5-7 papers)**
- Core: G-Eval (arXiv:2303.16634)
- Recent: Judge security, temperature effects, rubric evaluation (2024-2026)
- Focus: Judge failure modes, biases, calibration techniques

**Category 2: Agent Trajectory Evaluation (5-7 papers)**
- TRAJEVAL, HippoCamp, "Signals" framework
- SWE-bench evaluation methodologies
- Focus: Step-wise vs. end-to-end evaluation, failure diagnosis

**Category 3: Error Propagation (3-5 papers)**
- "Information Fidelity in Tool-Using LLM Agents"
- Cross-domain: autonomous systems, software testing, fault localization
- Focus: Quantitative propagation frameworks

**Category 4: Criticality Metrics (2-4 papers)**
- Autonomous driving risk assessment
- "Effort-Based Criticality Metrics for 3D Perception"
- Focus: Location-aware and position-dependent metrics

**Category 5: Gap Analysis (2-3 papers)**
- Identify what's missing in current research
- Position this work's novelty

**Key Questions to Answer:**
- Has anyone studied position-dependent error importance in agent trajectories? (Likely no)
- Are there existing criticality metrics for multi-step reasoning?
- What are known LLM judge failure modes?
- How do current benchmarks handle error location vs. type?

**Execution:**
1. Use WebSearch for academic papers with targeted queries
2. Read abstracts and key sections
3. Extract key findings, metrics, and gaps
4. Organize by category in LITERATURE_REVIEW.MD
5. Write novelty statement based on identified gaps

### Phase 2: POC Design (3-5 hours, concurrent with lit review)

**Core Design Decisions:**

**1. Benchmark: ToolBench (50-100 trajectories)**
- Rationale: Explicit tool selection, clear multi-step structure
- Alternative: GAIA if ToolBench environment unavailable

**2. Perturbation Design: 3 types × 3 positions**

Perturbation Types:
- Planning Error (incorrect goal decomposition)
- Tool Selection Error (wrong tool for subtask)
- Parameter Error (correct tool, wrong arguments)

Positions:
- Early (steps 1-2): Planning phase
- Middle (steps 3-5): Execution phase
- Late (steps 6+): Validation phase

Total: 9 experimental conditions

**3. Judge Models: 2 judges**
- GPT-4 (widely-used baseline)
- Claude-3.5-Sonnet (strong reasoning)
- Tests if miscalibration is universal vs. model-specific

**4. Impact Measurement:**
- Task Success Degradation (binary: task still solves?)
- Subsequent Error Rate (propagation: contained/propagated/cascaded)

**5. Simplified CCG Metric:**
```
True Criticality Score (TCS):
  TCS = (task_success_degradation × 100) + (avg_subsequent_errors × 10)

Judge Penalty Score (JPS):
  JPS = average judge-assigned penalty (0-100 scale)

Criticality-Calibration Gap (CCG):
  CCG = |TCS - JPS| / TCS
```

**Success Criteria:**
- Early position CCG > 0.3 (judges under-penalize)
- Late position CCG < -0.2 (judges over-penalize)
- Statistical significance p < 0.05
- Clear visual pattern in heatmaps

**Timeline:**
- Data preparation: 3-5 days
- Perturbation generation: 2-3 days
- Manual annotation: 3-5 days
- Judge evaluation: 1-2 days (1800-3600 API calls)
- Analysis: 2-3 days
- **Total: 11-18 days (2-3 weeks)**

**Resources:**
- API costs: $50-200
- Manual annotation: ~20-30 hours

**Risk Mitigation:**
- Perturbations don't degrade: Pilot test, adjust strength
- Manual annotation too slow: Start with 30 trajectories
- Judge ratings noisy: Prompt engineering, increase samples
- No position effect: Pivot to type-based hypothesis
- Environment unavailable: Switch to GAIA or static analysis

**Alternative POC Designs:**
1. Static trajectory analysis (no environment)
2. Synthetic trajectories (full control)
3. Single perturbation type deep dive (150-200 samples)

### Phase 3: Write Documents (5-7 hours, 3-4 days)

**Document 1: LITERATURE_REVIEW.MD**
Structure:
- Introduction (research motivation)
- Section per category (1-5 above)
- Each paper: summary, key findings, relevance
- Gap analysis section
- Novelty statement and positioning
- References (inline or separate BibTeX)

**Document 2: POC_REQUIREMENTS.MD**
Structure:
- Executive summary
- Experimental design (benchmark, perturbations, judges)
- Measurement protocol (metrics, CCG calculation)
- Success criteria
- Timeline and resource estimates
- Risk assessment
- Alternative designs
- What we learn even if results are negative

### Phase 4: State Tracking

Update `agent_tasks/01_research_foundation/state.json` throughout:
- Mark completed steps
- Track artifacts created
- Note blockers or decisions
- Keep status current

---

## Contingency Plans

### If Work Isn't Novel
**Action:** Assess degree of overlap
- <30% overlap: Proceed with reframed positioning
- 30-70% overlap: Identify novel aspect, extend existing work
- >70% overlap: Pivot to related question or abandon

### If POC Seems Infeasible
**Simplification options:**
- Reduce to 30 trajectories minimum
- Focus on 1-2 perturbation types
- Use synthetic trajectories
- Static analysis instead of environment evaluation

### If Literature Suggests Better Alternative
**Example:** Judges miscalibrated to error type (not position)
**Action:** Evaluate alternative, consider hybrid approach testing both

---

## Critical Files

**To be created:**
- `paper/LITERATURE_REVIEW.MD` - Main deliverable 1
- `paper/POC_REQUIREMENTS.MD` - Main deliverable 2
- `paper/references.bib` - Optional BibTeX file
- `paper/notes/` - Optional directory for rejected scope options

**To be updated:**
- `agent_tasks/01_research_foundation/state.json` - Progress tracking

**Existing references:**
- `agent_tasks/01_research_foundation/Requirements.MD` - Task definition
- `paper/abstract.MD` - Project context

---

## Verification

This task is complete when:

1. ✅ `paper/LITERATURE_REVIEW.MD` exists with:
   - 15-25 papers surveyed across 5 categories
   - Clear gap analysis
   - Novelty statement
   - Citations

2. ✅ `paper/POC_REQUIREMENTS.MD` exists with:
   - Concrete experimental design (all 8 design decisions specified)
   - Success criteria and timeline
   - Risk assessment
   - Alternative designs

3. ✅ `state.json` updated with:
   - Status: "completed"
   - All artifacts listed
   - Completion date

4. ✅ Human researcher approval confirms:
   - Work is sufficiently novel
   - POC is achievable
   - Ready to proceed to implementation

---

## Work Sequencing

**Week 1:**
- Days 1-2: Literature searches (Categories 1-2)
- Days 2-4: Literature searches (Categories 3-5)
- Days 4-5: Organize findings, start POC design
- Days 5-7: Write LITERATURE_REVIEW.MD

**Week 2:**
- Days 1-3: Finalize POC experimental protocol
- Days 3-4: Define metrics and success criteria
- Days 4-5: Risk assessment and alternatives
- Days 5-7: Write POC_REQUIREMENTS.MD
- Days 8-9: Human validation and approval

**Note:** Phases can overlap; literature findings inform POC design

---

## Key Insights from Planning

1. **Novelty confirmed:** Initial search shows no existing work on position-dependent error criticality in agent trajectories
2. **Scope is manageable:** 50-100 trajectories, 2-3 weeks, $50-200 budget
3. **Hypothesis is falsifiable:** Clear quantitative CCG metric with statistical testing
4. **Built-in flexibility:** Multiple contingency plans and alternative designs
5. **Valuable even if negative:** Will still advance understanding of judge behavior

This plan prioritizes **actionability, falsifiability, and resource-consciousness** while maintaining scientific rigor.
