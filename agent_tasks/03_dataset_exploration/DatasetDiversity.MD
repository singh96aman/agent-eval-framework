# Dataset Diversity Strategy

**Task:** 03_dataset_exploration  
**Date:** 2026-04-03  
**Status:** Draft for Review

---

## 1. Objective

Define a sampling strategy to achieve ~500 high-quality trajectories across multiple domains, ensuring:
1. Domain diversity (agents work in many contexts)
2. Complexity diversity (simple to complex tasks)
3. Tool diversity (enables all perturbation types)
4. Statistical power (sufficient samples per condition)

---

## 2. Target Dataset Composition

### 2.1 Overall Targets

| Source | Trajectories | Rationale |
|--------|--------------|-----------|
| ToolBench | 400 | Primary source, rich tool calls |
| GAIA | 100 | Reasoning depth, different task types |
| **Total** | **500** | Statistical power for 9-condition analysis |

### 2.2 Perturbations per Trajectory

- Target: 3-5 perturbations per trajectory
- Coverage: All 9 conditions (3 types × 3 positions)
- Estimated total: **1,500-2,500 perturbations**

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

**Generation Approach:**
1. Select 100 GAIA questions (stratified by difficulty)
2. Run Claude Sonnet 4.5 with tool access on each
3. Record the full trajectory (thoughts, tool calls, outputs)
4. Validate trajectory quality

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

## 5. Sampling Implementation

### 5.1 ToolBench Sampling Code

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

**Document Status:** Draft for Review  
**Next Action:** Create PerturbationDiversity.MD, then request researcher review
