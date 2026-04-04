# High-Quality Trajectory and Perturbations Gap Analysis

**Task:** 03_dataset_exploration  
**Date:** 2026-04-03  
**Status:** Revision 1 (Addressing Reviewer Feedback)

---

## IMPORTANT: Reviewer Feedback Incorporated

This document has been revised based on expert reviewer feedback. Key methodological concerns addressed:

| Concern | Resolution | Document |
|---------|------------|----------|
| POC contradicts hypothesis | Revised to salience-based hypothesis | PerturbationDiversity.MD |
| Ground truth underspecified | Counterfactual human judgment + IAA targets | PerturbationDiversity.MD |
| Selection bias (realism constraint) | Explicit scope: "silent errors" study | PerturbationDiversity.MD |
| One-per-trajectory confound | **100-trajectory** control group + stratified assignment | PerturbationDiversity.MD |
| Single judge model | Multi-judge (Claude, GPT-4o, Llama 70B) | DatasetDiversity.MD |
| GAIA generation bias | Multi-generator (Claude + GPT-4o) | DatasetDiversity.MD |
| SWE-bench semantics | Separate analysis, native perturbations | DatasetDiversity.MD |
| Statistical power | Hierarchical analysis, power calculation | DatasetDiversity.MD |
| Human = ground truth | CCG measures relative bias, not absolute | PerturbationDiversity.MD |
| **Execution-based validation** | **40-trajectory subset with actual re-execution** | PerturbationDiversity.MD |
| **Pre-registration** | **OSF pre-registration before full study** | DatasetDiversity.MD |
| **Falsification criteria** | **Explicit thresholds for null results** | PerturbationDiversity.MD |
| **Contribution clarity** | **Empirical finding + CCG metric as diagnostic** | DatasetDiversity.MD |

**See DatasetDiversity.MD and PerturbationDiversity.MD for detailed resolutions.**

---

## Revision 2 Summary: Expert Review Response

This revision addresses 6 additional concerns from the latest expert review:

| # | Concern | Resolution |
|---|---------|------------|
| 1 | Ground truth is weakest link | Added 40-trajectory execution-based validation subset |
| 2 | Statistical power marginal (32 < 42 needed) | Acknowledged; hierarchical analysis mitigates |
| 3 | Control group too small (50) | Increased to 100 trajectories |
| 4 | SWE-bench integration hedging | Made decisive commitment: KEEP, analyze separately |
| 5 | Missing pre-registration | Added OSF pre-registration plan |
| 6 | Hypothesis may be unfalsifiable | Added explicit falsification criteria with thresholds |

**Reviewer verdict:** "Close to but not quite at NeurIPS/ICLR main track bar as currently designed. Borderline / Weak Accept, contingent on results strength."

**Our response:** Address all actionable feedback while acknowledging inherent limitations (marginal power for fine-grained interactions, counterfactual ground truth).

---

## Executive Summary

This document analyzes gaps in the current trajectory and perturbation coverage from the POC experiment (Task 02). The goal is to identify areas where we need to expand dataset diversity and perturbation types to produce a robust, generalizable study on LLM judge calibration.

---

## 1. Current State (POC Results)

### 1.1 Dataset Coverage

| Metric | Value | Notes |
|--------|-------|-------|
| Trajectories loaded | 50 | ToolBench only |
| Perturbations generated | 443 | 9 conditions (3 types × 3 positions) |
| Judge evaluations | 852 | Claude Sonnet 4.5 |
| Human annotations | 26 | Stratified sample |

### 1.2 Perturbation Matrix

| Type / Position | Early (1-2) | Middle (3-5) | Late (6+) |
|-----------------|-------------|--------------|-----------|
| Planning | ✅ | ✅ | ✅ |
| Tool Selection | ✅ | ✅ | ✅ |
| Parameter | ✅ | ✅ | ✅ |

### 1.3 Key POC Findings

| Perturbation Type | Mean CCG | Interpretation |
|-------------------|----------|----------------|
| Parameter | +1.13 | Under-penalized (judges miss these) |
| Planning | -0.13 | Well-calibrated |
| Tool Selection | -0.26 | Over-penalized (judges too harsh) |

**Critical finding:** Parameter errors in middle positions (CCG = 2.788) are severely under-penalized by judges.

---

## 2. Gap Analysis: Dataset Coverage

### 2.1 Domain Representation

**ToolBench domains observed in test data:**
- Logistics (package tracking, shipping)
- Movies/Entertainment (OTT platforms, movie search)
- Video/Images (media processing)
- Cryptography (encoding/decoding)
- General Tools (utilities)
- Sports (greyhound racing)
- Travel (postal codes, transit)
- E-commerce (product catalogs)
- Music (Deezer search)
- Humor/Content (Chuck Norris jokes)

**Domains NOT well-represented:**
1. **Finance/Banking** - High-stakes decisions, regulatory sensitivity
2. **Healthcare/Medical** - Safety-critical, complex reasoning
3. **Legal/Compliance** - Document analysis, regulatory queries
4. **Code/Development** - Programming tasks (SWE-bench territory)
5. **Data Analysis** - Complex multi-step analytics
6. **Calendar/Scheduling** - Temporal reasoning, constraints
7. **Email/Communication** - Multi-party coordination
8. **File Management** - CRUD operations, organization

### 2.2 Task Complexity Distribution

| Complexity | Current Coverage | Gap |
|------------|------------------|-----|
| Simple (1-3 tools) | High | None |
| Medium (4-6 tools) | Medium | Need more |
| Complex (7+ tools) | Low | **Significant gap** |

**Issue:** Most ToolBench trajectories are simple API chains. Complex multi-step reasoning with branching is underrepresented.

### 2.3 Error Propagation Patterns

| Pattern | Current Coverage | Priority |
|---------|------------------|----------|
| Single-point failure | Good | - |
| Cascading errors | Limited | **HIGH** |
| Recovery attempts | Very limited | **HIGH** |
| Partial success | Limited | Medium |

---

## 3. Gap Analysis: Perturbation Types

### 3.1 Current Perturbation Strategies

**Planning Errors (Type A):**
- Goal substitution (confuse future/past)
- Requirement omission (compare X and Y → only analyze X)
- Scope reduction (full details → subset)

**Tool Selection Errors (Type B):**
- Similar tool swap (racecards → results)
- Category confusion (search → list)
- Semantic confusion (latest → historical)

**Parameter Errors (Type C):**
- Missing required parameter
- Wrong parameter value
- Wrong parameter name
- Wrong parameter type

### 3.2 Missing Perturbation Types

| Missing Type | Description | Priority |
|--------------|-------------|----------|
| **Sequencing errors** | Steps in wrong order | **HIGH** |
| **Redundant steps** | Unnecessary repeated calls | Medium |
| **Missing steps** | Skipping required intermediate | **HIGH** |
| **Hallucinated data** | Using non-existent entities | **HIGH** |
| **Context errors** | Losing track of prior results | Medium |
| **Format errors** | Wrong output structure | Low |

### 3.3 Perturbation Intensity Levels

**Current:** Only "moderate" intensity implemented.

**Missing levels:**
- **Subtle:** Small errors that might go unnoticed
- **Severe:** Obvious catastrophic failures

This matters because judges may be calibrated differently for subtle vs. obvious errors.

---

## 4. Gap Analysis: Position Coverage

### 4.1 Current Position Definitions

| Position | Step Range | Current Logic |
|----------|------------|---------------|
| Early | Steps 1-2 | First 1/3, min 2 steps |
| Middle | Steps 3-5 | Middle 1/3 |
| Late | Steps 6+ | Last 1/3, min last 2 |

### 4.2 Position-Related Gaps

1. **Short trajectories (3-4 steps):** Position boundaries overlap, making distinctions meaningless
2. **Long trajectories (10+ steps):** "Early" position (step 1-2) is very early; may need finer granularity
3. **Non-linear trajectories:** Current model assumes linear progression; doesn't handle branches or loops

---

## 5. Gap Analysis: Benchmark Diversity

### 5.1 Current: ToolBench Only

**Strengths:**
- Large scale (188k+ trajectories available)
- Real API calls with documentation
- Multi-tool scenarios

**Weaknesses:**
- Synthetic queries (not real user tasks)
- API-centric (tool-use focused)
- Limited reasoning depth

### 5.2 Recommended Additions

| Benchmark | Type | Value Add | Access |
|-----------|------|-----------|--------|
| **GAIA** | Multi-step QA | Reasoning depth, diverse tasks | Gated (approved) |
| **SWE-bench** | Code agent | Code understanding, complex fixes | Public |
| **WebArena** | Web navigation | Sequential multi-page tasks | Public |
| **AgentBench** | Multi-domain | Broad coverage | Public |

### 5.3 GAIA-Specific Value

GAIA provides:
- **Real multi-step reasoning** (not just API chaining)
- **Diverse task types** (documents, images, web search)
- **Clear ground truth** (unambiguous answers)
- **Difficulty levels** (Level 1-3)

However, GAIA trajectories may need to be **generated** by running agents on the tasks, since the dataset only contains questions and answers.

---

## 6. Recommendations

### 6.1 Dataset Expansion (Revised)

1. **ToolBench:** 400 trajectories (up from 50)
   - Stratified across 8 domain categories
   - Filter for complexity (4-10 steps)
   
2. **GAIA:** 100 trajectories
   - Generate by running Claude/GPT on GAIA tasks
   - Focus on Level 1-2 (tractable complexity)
   
3. **SWE-bench:** 100 trajectories
   - Code domain coverage (missing from other benchmarks)
   - Popular benchmark (adds credibility)
   - Different error modes (syntax, logic, API misuse)

**Total: 600 trajectories**

### 6.2 Perturbation Strategy (Revised)

**Key design decisions:**
1. **One perturbation per trajectory** - cleaner experimental design, no confounding
2. **Realism constraint** - only perturbations that allow execution to continue
3. **Sequencing errors EXCLUDED** - they break execution (trajectory becomes incoherent)

**Approved perturbation types (7 subtypes):**
- Planning: goal substitution, scope reduction
- Tool Selection: similar tool swap
- Parameter: wrong value, format error, off-by-one
- Data Reference: hallucinated reference (NEW)

**Total: 600 trajectories × 1 perturbation = 600 perturbations**

### 6.3 Annotation Strategy

| Priority | Sample Size | Focus |
|----------|-------------|-------|
| P0 | 150 | ToolBench stratified (50 × 3 conditions) |
| P1 | 75 | GAIA stratified (25 × 3 conditions) |
| P2 | 50 | New perturbation types |

**Total target:** 275 annotations for full study

---

## 7. Risk Assessment

### 7.1 Data Quality Risks

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Perturbations not realistic | Medium | Human review of samples |
| GAIA access denied | Low | Use WebArena as fallback |
| Tool diversity insufficient | Medium | Pre-filter trajectories |

### 7.2 Statistical Power

| Sample Size | Statistical Power | Recommendation |
|-------------|-------------------|----------------|
| n=26 (current) | ~30% | Insufficient |
| n=100 | ~60% | Marginal |
| n=150 | ~80% | Recommended |
| n=225 | ~90% | Ideal |

---

## 8. Next Steps

1. **Create DatasetDiversity.MD:** Specify exact sampling strategy for ToolBench + GAIA
2. **Create PerturbationDiversity.MD:** Detail new perturbation types to implement
3. **Review with researcher:** Get approval before implementation
4. **Implement code changes:** Add new loaders, perturbation strategies
5. **Run expanded experiment:** Generate new perturbations, run judges

---

## Appendix A: ToolBench Categories (Observed)

From test instruction files:
- Logistics
- Movies
- Video_Images
- Cryptography
- Tools
- Sports
- Travel
- E-commerce
- Music
- Entertainment

## Appendix B: GAIA Task Types

From dataset documentation:
- Web browsing and search
- Document analysis (PDF, Word)
- Image understanding
- Multi-source synthesis
- Temporal reasoning
- Numerical computation

---

**Document Status:** Ready for human researcher review  
**Next Action:** Create DatasetDiversity.MD and PerturbationDiversity.MD
