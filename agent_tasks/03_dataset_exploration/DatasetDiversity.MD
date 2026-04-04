# Dataset Diversity Strategy

**Task:** 03_dataset_exploration  
**Date:** 2026-04-03  
**Status:** Revision 1 (Addressing Reviewer Feedback)

---

## 1. Objective

Define a sampling strategy to achieve ~600 high-quality trajectories across multiple domains, ensuring:
1. Domain diversity (agents work in many contexts)
2. Complexity diversity (simple to complex tasks)
3. Tool diversity (enables all perturbation types)
4. Statistical power (sufficient samples per condition)
5. **Multi-judge generalizability** (findings not model-specific)
6. **Cross-benchmark validity** (perturbation semantics comparable)

### 1.1 CRITICAL: Contribution Clarity

**Primary contribution:** Empirical finding that LLM judges are systematically miscalibrated for agent trajectories, with miscalibration predicted by error salience rather than criticality.

**Secondary contribution:** CCG (Criticality-Calibration Gap) metric as a diagnostic tool for evaluating judge reliability.

**What this paper is NOT:**
- A new benchmark (we use existing benchmarks)
- A new judge model (we evaluate existing judges)
- A method to fix calibration (that's future work)

**Sharpened claim:** "We demonstrate that all tested LLM judges (Claude, GPT-4o, Llama 70B) systematically under-penalize low-salience errors in agent trajectories, with implications for RLHF reward modeling and production monitoring."

---

## 2. Target Dataset Composition

### 2.1 Overall Targets

| Source | Trajectories | Rationale |
|--------|--------------|-----------|
| ToolBench | 400 | Primary source, rich tool calls, API diversity |
| GAIA | 100 | Reasoning depth, multi-step QA tasks |
| SWE-bench | 100 | Code domain coverage, popular benchmark |
| **Total** | **600** | Statistical power + benchmark diversity |

**Why SWE-bench?** Adds code/development domain which is:
- Missing from ToolBench (API-focused) and GAIA (QA-focused)
- Highly popular benchmark (adds credibility to paper)
- Different error modes (syntax, logic, API misuse)

---

## 2.5 CRITICAL: Multi-Judge Design

### The Problem (Reviewer Concern #5)

**"Only Claude Sonnet 4.5 used as judge. Different models may have different calibration patterns. Without this, findings may be Claude-specific artifacts."**

### Our Solution: Multi-Judge Evaluation

Each perturbation will be evaluated by **3 different judge models**:

| Judge Model | Provider | Rationale |
|-------------|----------|-----------|
| Claude Sonnet 4.5 | AWS Bedrock | Primary (already implemented) |
| GPT-4o | OpenAI | Different training, architecture |
| Llama 3.1 70B | Together/Bedrock | Open-source, different paradigm |

### Analysis Implications

| Finding | Interpretation |
|---------|----------------|
| All judges show same CCG pattern | **Strong finding** - model-independent bias |
| Claude-specific CCG pattern | Model-specific artifact, report as limitation |
| All judges disagree with humans similarly | Systematic LLM limitation |
| Judges disagree with each other | Judge-specific biases (still interesting) |

### Cost Estimate (3 judges × 600 perturbations)

| Judge | API Cost/1K tokens | Estimated Total |
|-------|-------------------|-----------------|
| Claude Sonnet 4.5 | $0.003 | ~$50 |
| GPT-4o | $0.005 | ~$80 |
| Llama 3.1 70B | $0.001 | ~$15 |
| **Total** | | **~$145** |

### Implementation

The judge framework already supports multiple models. We run each perturbation through all 3 judges and compute CCG for each (judge, perturbation_type, position) combination.

```python
JUDGE_MODELS = [
    "claude-sonnet-4.5",
    "gpt-4o", 
    "llama-3.1-70b"
]

for perturbation in perturbations:
    for judge in JUDGE_MODELS:
        evaluate(perturbation, judge)
```

### 2.2 Perturbation Strategy: One Per Trajectory

**Design decision:** Each trajectory receives exactly ONE perturbation.

| Approach | Trajectories | Perturbations/Traj | Total |
|----------|--------------|-------------------|-------|
| ~~All perturbations~~ | 600 | 9 | 5,400 |
| **One perturbation (chosen)** | 600 | 1 | 600 |

**Rationale:**
- **Cleaner experimental design** - no confounding between perturbation types on same trajectory
- **Domain-agnostic validation** - can verify results hold across domains within each condition
- **Tractable annotation** - 600 perturbations vs 5,400

### 2.3 Stratified Sampling for Domain Coverage

To ensure results are not domain-specific, we stratify by both **perturbation condition** and **domain**:

| Condition | ToolBench | GAIA | SWE-bench | Total |
|-----------|-----------|------|-----------|-------|
| Planning × Early | 25 | 8 | 8 | 41 |
| Planning × Middle | 25 | 8 | 8 | 41 |
| Planning × Late | 25 | 8 | 8 | 41 |
| Tool Sel × Early | 25 | 8 | 8 | 41 |
| Tool Sel × Middle | 25 | 8 | 8 | 41 |
| Tool Sel × Late | 25 | 8 | 8 | 41 |
| Parameter × Early | 50 | 10 | 10 | 70 |
| Parameter × Middle | 50 | 10 | 10 | 70 |
| Parameter × Late | 50 | 10 | 10 | 70 |
| **Reserve pool** | 100 | 22 | 22 | 144 |
| **Total** | **400** | **100** | **100** | **600** |

**Analysis approach:**
1. Primary: Compare CCG across 9 conditions (type × position)
2. Secondary: Within each condition, verify results hold across benchmarks
3. Tertiary: Within ToolBench, verify results hold across domain categories

---

## 3. ToolBench Sampling Strategy

### 3.1 Domain Distribution

Sample trajectories across diverse API categories to ensure generalizability:

| Domain Category | Target Count | Examples from ToolBench |
|-----------------|--------------|-------------------------|
| **Data/Information** | 80 | Weather, news, stock quotes |
| **Media/Entertainment** | 60 | Movies, music, videos, images |
| **E-commerce/Shopping** | 60 | Product search, coupons, prices |
| **Travel/Logistics** | 50 | Flights, hotels, package tracking |
| **Finance/Business** | 50 | Currency, stocks, business data |
| **Social/Communication** | 40 | Social media, messaging, reviews |
| **Utilities/Tools** | 40 | Conversion, calculation, validation |
| **Sports/Gaming** | 20 | Scores, schedules, statistics |
| **Total** | **400** | |

### 3.2 Complexity Stratification

| Complexity Level | Steps | Target % | Count |
|------------------|-------|----------|-------|
| Simple | 3-4 | 20% | 80 |
| Medium | 5-6 | 50% | 200 |
| Complex | 7-10 | 30% | 120 |

**Rationale:** Medium complexity (5-6 steps) gives best coverage of early/middle/late positions while remaining tractable for annotation.

### 3.3 Quality Filters

Apply these filters when loading ToolBench:

```python
filters = {
    "min_steps": 4,           # At least 4 steps for position coverage
    "max_steps": 10,          # Manageable for annotation
    "filter_successful": True, # Only successful trajectories
    "require_parameters_all_positions": True,  # Enables parameter perturbations
    "require_tool_diversity": True,            # Enables tool selection perturbations
}
```

### 3.4 ToolBench Data Sources

| Source File | Content | Usage |
|-------------|---------|-------|
| `toolllama_G123_dfs_eval.json` | 7.9MB, evaluation set | Primary source (higher quality) |
| `toolllama_G123_dfs_train.json` | 2GB, training set | Supplement if needed |
| `G1_category.json` | Single-tool tasks | Simple trajectories |
| `G2_category.json` | Multi-tool same-category | Medium complexity |
| `G3_instruction.json` | Multi-tool multi-category | Complex trajectories |

---

## 4. GAIA Sampling Strategy

### 4.1 Trajectory Generation

GAIA contains questions and answers but NOT agent trajectories. We need to **generate trajectories** by running an agent on GAIA tasks.

### 4.2 CRITICAL: Addressing the Claude-Judging-Claude Bias

**Reviewer Concern #6:** "GAIA doesn't have trajectories - you're generating them by running Claude on GAIA tasks, then using Claude to judge Claude-generated trajectories. This creates potential for model-specific biases."

**This is a valid concern.** If Claude generates the trajectory and Claude judges it, we may see:
- Inflated agreement (Claude "understands" its own reasoning)
- Deflated agreement (Claude is harsh on its own style)
- Either way, results may not generalize to other agent-judge pairs

### 4.3 Mitigation Strategies for GAIA

**Strategy 1: Multi-Generator Design**
Generate GAIA trajectories using MULTIPLE agent models:

| Generator Model | Trajectories | Rationale |
|-----------------|--------------|-----------|
| Claude Sonnet 4.5 | 50 | Same-model baseline |
| GPT-4o | 50 | Cross-model trajectories |

This allows us to compare:
- Claude-judging-Claude vs Claude-judging-GPT
- GPT-judging-Claude vs GPT-judging-GPT
- If CCG patterns differ, we report the cross-model results as primary

**Strategy 2: Use Existing SWE-bench Trajectories**
SWE-bench trajectories are **pre-generated** by SWE-Smith (a different agent system). These are truly external trajectories, not Claude-generated.

| Benchmark | Generator | Judge Independence |
|-----------|-----------|-------------------|
| ToolBench | External (pre-existing) | Full independence |
| SWE-bench | SWE-Smith (external) | Full independence |
| GAIA | Claude/GPT (generated) | Partial - mitigated by multi-generator |

**Strategy 3: Report by Generator-Judge Pair**
For GAIA results, we report CCG separately for each (generator, judge) combination and flag any significant differences.

### 4.4 Revised GAIA Generation Approach

1. Select 100 GAIA questions (stratified by difficulty)
2. Run **50 with Claude Sonnet 4.5**, **50 with GPT-4o**
3. Record the full trajectory (thoughts, tool calls, outputs)
4. Validate trajectory quality
5. In analysis, compare CCG for Claude-generated vs GPT-generated subsets

### 4.2 Difficulty Distribution

| Level | Description | Target Count | Rationale |
|-------|-------------|--------------|-----------|
| Level 1 | Simple, 1-2 steps | 30 | Baseline, clear correct answer |
| Level 2 | Medium, 3-5 steps | 50 | Main analysis focus |
| Level 3 | Hard, 5+ steps | 20 | Complex reasoning, edge cases |

### 4.3 Task Type Distribution

| Task Type | Target Count | Examples |
|-----------|--------------|----------|
| Web Search | 30 | Find information online |
| Document Analysis | 25 | Parse PDF, extract data |
| Multi-source Synthesis | 25 | Combine info from multiple sources |
| Calculation/Reasoning | 20 | Math, logic, temporal |

### 4.4 GAIA Access Requirements

- Dataset: `gaia-benchmark/GAIA` on HuggingFace
- Access: **Gated** - requires approval
- Status: Need to verify/request access

**Fallback if GAIA unavailable:**
- Use `AgentBench` or `WebArena` trajectories
- Generate synthetic GAIA-like trajectories from ToolBench

---

## 5. SWE-bench Sampling Strategy

### 5.1 Why SWE-bench?

SWE-bench adds the **code/development domain** which is:
- Completely missing from ToolBench (API-focused) and GAIA (QA-focused)
- A highly popular benchmark (adds credibility to paper)
- Features different error modes: syntax errors, logic errors, API misuse, test failures

### 5.2 Data Source

**Primary:** `SWE-bench/SWE-smith-trajectories` on HuggingFace
- 76k trajectories with full execution traces
- Includes thoughts, tool calls (file edits, test runs), and outputs
- Public access, no gating

### 5.3 Difficulty Distribution

| Difficulty | Target Count | Description |
|------------|--------------|-------------|
| Easy fixes | 40 | Simple bug fixes, single-file changes |
| Medium fixes | 40 | Multi-file changes, moderate complexity |
| Hard fixes | 20 | Complex refactors, multiple components |

### 5.4 Quality Filters for SWE-bench

```python
filters = {
    "min_steps": 4,           # Minimum tool calls
    "max_steps": 10,          # Manageable for annotation
    "filter_successful": True, # Only successful fixes
    "diverse_repos": True,     # Not all from same project
}
```

### 5.5 CRITICAL: SWE-bench Perturbation Semantics

**Reviewer Concern #7:** "Mapping ToolBench perturbations to SWE-bench is problematic. 'Wrong API tool' → 'Wrong file edit target' are qualitatively different errors. Comparing CCG across fundamentally different perturbation semantics undermines cross-benchmark validity."

### 5.5.1 The Problem

SWE-bench trajectories have different tool calls than ToolBench:
- `file_edit` instead of API calls
- `run_tests` instead of data retrieval
- `search_code` instead of web search

A naive mapping creates **semantic mismatch**:

| ToolBench Perturbation | Naive SWE-bench Equivalent | Problem |
|------------------------|----------------------------|---------|
| Wrong API tool | Wrong file edit target | Different cognitive demand |
| Wrong parameter value | Wrong code change | Code is more complex than params |
| Planning error (wrong goal) | Wrong bug diagnosis | Diagnosis requires domain expertise |

### 5.5.2 Options for SWE-bench

**Option A: Remove SWE-bench (Conservative)**

Drop SWE-bench entirely. Focus on ToolBench + GAIA only.

| Pro | Con |
|-----|-----|
| Clean perturbation semantics | Lose code domain |
| Simpler analysis | Less benchmark diversity |
| No mapping ambiguity | Miss popular benchmark |

**Option B: Keep SWE-bench but Analyze Separately (Recommended)**

Include SWE-bench but do NOT pool CCG across benchmarks for cross-benchmark comparisons.

| Pro | Con |
|-----|-----|
| Keep code domain coverage | More complex analysis |
| Can still report SWE-bench-specific findings | Cannot make "universal" claims |
| Valuable as separate study | Slightly weaker paper story |

**Option C: Create SWE-bench-Native Perturbations**

Design perturbations that are semantically appropriate for code:
- **Wrong file:** Edit wrong file (analogous to wrong tool)
- **Wrong hunk:** Edit wrong location in file (analogous to wrong params)
- **Wrong fix:** Apply incorrect code change (unique to SWE-bench)
- **Planning error:** Diagnose wrong root cause (analogous to planning)

These are NOT mappings from ToolBench - they are SWE-bench-native error types that test similar cognitive skills.

### 5.5.3 Our Decision: KEEP SWE-bench, Analyze Separately (FINAL)

**Decisive commitment:** We KEEP SWE-bench and commit to analyzing it separately. This is NOT hedging—it's the scientifically correct approach for benchmarks with different perturbation semantics.

**Rationale for keeping (not dropping):**
1. **Code domain matters**: Missing code would be a glaring omission for agent evaluation research
2. **Different error modes are valuable**: If patterns hold across semantically different perturbations, that's a stronger finding
3. **If patterns diverge, that's also interesting**: Domain-specificity of judge biases is itself a contribution

**What we commit to:**
1. Keep SWE-bench (100 trajectories)
2. Create SWE-bench-native perturbation types (conceptually analogous, not mechanically mapped)
3. Report SWE-bench CCG in a **dedicated section** (not pooled with ToolBench/GAIA in main results)
4. Compare **qualitative patterns** across benchmarks in discussion
5. **Pre-register** both the separate analysis plan AND what pattern consistency would mean

### 5.5.4 Revised SWE-bench Perturbation Types

| Perturbation | SWE-bench Implementation | Conceptual Category |
|--------------|--------------------------|---------------------|
| **Wrong file** | Edit a different file than needed | Tool selection analog |
| **Wrong location** | Edit wrong function/class in correct file | Parameter analog (spatial) |
| **Wrong value** | Change wrong variable/value in code | Parameter analog (value) |
| **Wrong diagnosis** | Planning thinks bug is in wrong component | Planning analog |
| **Data reference** | Use wrong variable name from earlier search | Data reference analog |

### 5.5.5 Cross-Benchmark Validity Statement

> We do NOT claim that CCG values are directly comparable across ToolBench and SWE-bench. The perturbation types are conceptually analogous but not identical. Instead, we test whether the *qualitative patterns* (e.g., parameter-like errors under-penalized, planning errors well-calibrated) hold across both domains. If patterns diverge, this is itself an interesting finding about domain-specificity of judge biases.

---

## 6. CRITICAL: Statistical Power Analysis

### 6.1 Reviewer Concern #8

**"With 19 conditions and 30 samples each, some cells will be thin. Data reference errors (only 2 positions) get only 60 samples total."**

### 6.2 Current Design Power Analysis

| Design Element | Value | Notes |
|----------------|-------|-------|
| Total trajectories | 600 | |
| Conditions | 19 | 7 subtypes × 3 positions - 2 (D1 no early) |
| Samples per condition | ~32 | 600 / 19 |
| Judge models | 3 | Claude, GPT-4o, Llama 70B |
| Human annotations | 275 | Target (up from 26) |

### 6.3 Power Calculation for CCG Detection

**Effect size from POC:**
- Parameter CCG = +1.13, Planning CCG = -0.13
- Difference = 1.26 points
- SD = 2.08 (from POC)
- Cohen's d = 1.26 / 2.08 = 0.61 (medium-large effect)

**Required n for 80% power (two-sample t-test, alpha=0.05):**
```
n_per_group = 2 * (1.96 + 0.84)^2 / d^2
n_per_group = 2 * (2.8)^2 / (0.61)^2
n_per_group = 2 * 7.84 / 0.37
n_per_group = 42 per condition
```

**Our design: 32 per condition is UNDERPOWERED for individual comparisons.**

### 6.4 Mitigation: Hierarchical Analysis

Instead of 19 individual conditions, we analyze at three levels:

| Level | Conditions | Samples | Power |
|-------|------------|---------|-------|
| **Type level** | 4 (planning, tool, param, data_ref) | ~150 each | Adequate |
| **Position level** | 3 (early, middle, late) | ~200 each | Good |
| **Type × Position** | 12 (excl. thin cells) | ~50 each | Marginal |
| **Subtype level** | 7 | ~85 each | Adequate |

**Primary analysis:** Type level (4 conditions) and Position level (3 conditions)
**Secondary analysis:** Type × Position interaction (powered for large effects only)
**Exploratory:** Subtype comparisons (report with appropriate uncertainty)

### 6.5 Sample Size Adjustments

**Option A: Increase to 900 trajectories (50 per condition)**
- More statistical power
- Higher annotation burden (400+ annotations)
- Higher API cost (~$220)

**Option B: Reduce conditions (current recommendation)**
- Collapse parameter subtypes (C1, C2, C3 → C)
- Focus on 4 types × 3 positions = 12 conditions
- 50 samples per condition
- Adequate power for primary comparisons

**Option C: Mixed design**
- 550 trajectories: one perturbation each (primary)
- 50 trajectories: all perturbations (within-subject control)
- Use mixed-effects models to pool information

### 6.6 Revised Sample Allocation (Option B Recommended)

| Condition | Samples | Rationale |
|-----------|---------|-----------|
| Planning × Early | 50 | Primary analysis |
| Planning × Middle | 50 | |
| Planning × Late | 50 | |
| Tool Selection × Early | 50 | |
| Tool Selection × Middle | 50 | |
| Tool Selection × Late | 50 | |
| Parameter × Early | 50 | Collapsed subtypes |
| Parameter × Middle | 50 | |
| Parameter × Late | 50 | |
| Data Reference × Middle | 50 | No early position |
| Data Reference × Late | 50 | |
| **Reserve pool** | 50 | For failed perturbations |
| **Total** | **600** | |

### 6.8 CRITICAL: Pre-Registration Plan

**Why pre-register?** For a study explicitly about systematic biases, pre-registration adds significant credibility and prevents post-hoc hypothesis revision.

**Pre-registration platform:** OSF (Open Science Framework)

**Pre-registered elements:**
1. **Hypotheses**: Salience-based miscalibration (Section 2.3, PerturbationDiversity.MD)
2. **Falsification criteria**: Specified thresholds for null results (Section 2.5, PerturbationDiversity.MD)
3. **Analysis plan**: Hierarchical CCG analysis, covariate adjustment
4. **Sample sizes**: 600 trajectories, 275 annotations, 100-trajectory control group
5. **SWE-bench decision**: Separate analysis, not pooled
6. **Multi-judge design**: 3 models, analysis matrix specified

**Timeline:** Pre-register BEFORE running judges on full dataset. POC data can inform hypothesis but full study must be pre-registered.

**What we will NOT pre-register** (exploratory):
- Exact API cost
- Specific tool mappings
- Annotation interface design details

### 6.7 Annotation Strategy

With 600 perturbations, we cannot annotate all. Strategy:

| Subset | Annotations | Purpose |
|--------|-------------|---------|
| Stratified sample | 220 | 20 per condition × 11 conditions |
| Control trajectories | 100 | Within-trajectory replication group |
| Execution validation | 40 | Subset with actual API re-execution |
| **Total annotations** | **275** | (some overlap with control/validation) |

**Inter-annotator reliability:** 50 samples double-annotated to compute Krippendorff's alpha.

---

## 7. Sampling Implementation

### 7.1 ToolBench Sampling Code

```python
def sample_toolbench_trajectories(
    target_count: int = 400,
    domain_distribution: Dict[str, int] = None,
    complexity_distribution: Dict[str, float] = None,
    random_seed: int = 42
) -> List[Trajectory]:
    """
    Sample ToolBench trajectories with stratification.
    """
    # Load with quality filters
    trajectories = load_toolbench_trajectories(
        max_trajectories=None,  # Load all, then sample
        min_steps=4,
        max_steps=10,
        filter_successful=True,
        require_parameters_all_positions=True,
        require_tool_diversity=True,
        random_seed=random_seed,
        split="eval"
    )
    
    # Classify by domain and complexity
    classified = classify_trajectories(trajectories)
    
    # Stratified sampling
    sampled = stratified_sample(
        classified,
        target_count,
        domain_distribution,
        complexity_distribution
    )
    
    return sampled
```

### 5.2 Domain Classification Heuristics

Map ToolBench tool names to domain categories:

| Pattern | Domain |
|---------|--------|
| `*weather*`, `*forecast*`, `*news*` | Data/Information |
| `*movie*`, `*music*`, `*video*`, `*image*` | Media/Entertainment |
| `*product*`, `*shop*`, `*coupon*`, `*price*` | E-commerce/Shopping |
| `*flight*`, `*hotel*`, `*travel*`, `*colis*` | Travel/Logistics |
| `*stock*`, `*currency*`, `*finance*`, `*bank*` | Finance/Business |
| `*social*`, `*tweet*`, `*message*`, `*review*` | Social/Communication |
| `*convert*`, `*calculate*`, `*validate*` | Utilities/Tools |
| `*sport*`, `*game*`, `*score*`, `*race*` | Sports/Gaming |

### 5.3 Complexity Classification

```python
def classify_complexity(trajectory: Trajectory) -> str:
    """Classify trajectory complexity by step count."""
    n = len(trajectory.steps)
    if n <= 4:
        return "simple"
    elif n <= 6:
        return "medium"
    else:
        return "complex"
```

---

## 6. Validation Criteria

### 6.1 Trajectory Quality Checks

Each trajectory must pass:

| Check | Criterion | Rationale |
|-------|-----------|-----------|
| Step count | 4-10 steps | Position coverage |
| Tool calls | ≥2 distinct tools | Tool selection perturbations |
| Parameters | Non-empty in each position | Parameter perturbations |
| Success | Marked successful | Clear ground truth |
| Coherence | Logical step sequence | Annotation clarity |

### 6.2 Dataset-Level Validation

| Metric | Target | Minimum |
|--------|--------|---------|
| Domain coverage | 8 domains | 6 domains |
| Complexity balance | 20/50/30 split | Within ±10% |
| Tool diversity | ≥100 unique tools | ≥50 tools |
| Mean steps | 5.5 | 5.0 |

---

## 7. Expected Outputs

### 7.1 Dataset Artifacts

| Artifact | Format | Location |
|----------|--------|----------|
| Sampled trajectories | JSON | `data/sampled/toolbench_400.json` |
| GAIA trajectories | JSON | `data/sampled/gaia_100.json` |
| Combined dataset | JSON | `data/sampled/combined_500.json` |
| Domain mapping | CSV | `data/sampled/domain_mapping.csv` |

### 7.2 Statistics Report

Generate after sampling:
- Domain distribution histogram
- Complexity distribution histogram
- Tool usage frequency
- Step count distribution
- Parameter coverage matrix

---

## 8. Risk Mitigation

### 8.1 Insufficient ToolBench Diversity

**Risk:** Not enough trajectories meet quality filters.

**Mitigation:**
- Relax `require_tool_diversity` if needed
- Draw from training set (larger pool)
- Accept some domain imbalance

### 8.2 GAIA Access Issues

**Risk:** Cannot access gated dataset.

**Mitigation:**
- Submit access request immediately
- Prepare WebArena fallback
- Generate synthetic trajectories from ToolBench

### 8.3 Trajectory Generation Quality

**Risk:** Generated GAIA trajectories are poor quality.

**Mitigation:**
- Use Claude Sonnet 4.5 (best available)
- Add temperature/sampling variation
- Manual review of 10% sample

---

## 9. Implementation Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| ToolBench sampling | 2 hours | 400 trajectories |
| GAIA access + generation | 4-8 hours | 100 trajectories |
| Validation & QA | 2 hours | Quality report |
| **Total** | **8-12 hours** | 500 validated trajectories |

---

## 10. Questions for Researcher Review

1. **Domain priorities:** Are there specific domains we should prioritize (e.g., finance for high-stakes)?

2. **GAIA vs. alternatives:** If GAIA access is difficult, should we prioritize WebArena or SWE-bench?

3. **Complexity balance:** Is 20/50/30 (simple/medium/complex) the right split, or should we focus more on medium complexity?

4. **Tool diversity requirement:** Should we strictly require tool diversity, or accept some trajectories that only support parameter perturbations?

5. **Sample size trade-offs:** Would 300 high-quality trajectories be better than 500 with some quality compromises?

---

## 11. Reviewer Feedback Addressed

This revision addresses the following concerns from the expert reviewer:

| # | Concern | Section | Resolution |
|---|---------|---------|------------|
| 5 | Single judge model | 2.5 | Multi-judge design (Claude, GPT-4o, Llama 70B) |
| 6 | GAIA trajectory generation bias | 4.2-4.4 | Multi-generator design (Claude + GPT-4o) |
| 7 | SWE-bench perturbation semantics | 5.5 | Separate analysis, SWE-bench-native perturbations |
| 8 | Statistical power | 6.1-6.7 | Hierarchical analysis, revised sample allocation |

**Concerns addressed in PerturbationDiversity.MD:**
- #1 (POC contradicts hypothesis) → Revised hypothesis based on salience
- #2 (Ground truth criticality) → Detailed operationalization + inter-annotator agreement
- #3 (Selection bias) → Acknowledged with explicit scope statement
- #4 (One-per-trajectory confound) → Stratified assignment + control group
- #9 (Human = ground truth assumption) → Defended with reframe

---

**Document Status:** Revision 1 (Addressing Reviewer Feedback)  
**Next Action:** Request researcher review of revisions
