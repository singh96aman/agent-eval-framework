# RQ1: Consequentiality Calibration

## Research Question

**Do judge penalty scores correlate with actual outcome damage?**

```
Correlation(JPS, OD) = ?
```

Where:
- **JPS** (Judge Penalty Score) = 100 - overall_score (what the judge thinks)
- **OD** (Outcome Degradation) = measured impact on task outcome (what actually happened)

---

## Approach

**Extend existing experiment** `final_exp_perturbation_quality` with OD computation and calibration analysis.

| Data | Status | Action |
|------|--------|--------|
| Trajectories (600) | Exists | Reuse |
| Perturbations (600) | Exists | Reuse |
| Quality Scores | Exists | Reuse |
| Judge Evaluations (852) | Exists | Reuse |
| **OD Values** | Missing | **Compute** |
| **Calibration Metrics** | Missing | **Compute** |

---

## Execution

```bash
python main.py --config rq1_calibration --runner od,od_validate,calibration
```

---

## Phases

### Phase 1: `od` - Compute Outcome Degradation

**What it does:** For each perturbation, grade baseline vs perturbed final answers.

```
For each perturbation:
  1. Load baseline_trajectory and perturbed_trajectory
  2. Extract final answers from both
  3. Call LLM grader to score both (0-100)
  4. OD = (baseline_score - perturbed_score) / 100
  5. Update perturbation document with od field
```

**Output:** Each perturbation document in MongoDB gets an `od` field:
```json
{
  "od": {
    "tier": 3,
    "value": 0.35,
    "baseline_outcome": 85,
    "perturbed_outcome": 50,
    "method": "static_grading",
    "grader_model": "claude-sonnet-4.5",
    "computed_at": "2026-04-05T..."
  }
}
```

**Cost:** ~600 API calls, ~$4

---

### Phase 2: `od_validate` - Validate Ground Truth

**What it does:** Check OD distribution before proceeding.

| Check | Pass Criteria |
|-------|---------------|
| Coverage | >= 70% perturbations have valid OD |
| Variance | > 0.1 (not all 0s or 1s) |
| OD=0 cases | Spot-check 20 samples for validity |

**Output:** `results/rq1/od_validation.json`

**Decision Point:**
- Pass all checks: Continue to calibration
- Fail: Stop and investigate

**Cost:** FREE (local computation)

---

### Phase 3: `calibration` - Calibration Analysis

**What it does:** Compute correlation between JPS and OD.

**Metrics:**
| Metric | Description |
|--------|-------------|
| Spearman r | Rank correlation (primary) |
| Pearson r | Linear correlation |
| ECE | Expected Calibration Error |
| MCE | Maximum Calibration Error |

**Outputs:**
| File | Description |
|------|-------------|
| `results/rq1/calibration_curve.png` | Main figure for paper |
| `results/rq1/jps_vs_od_scatter.png` | Supporting scatter plot |
| `results/rq1/statistics.json` | All metrics + bootstrap CIs |
| `results/rq1/findings.md` | Written interpretation |

**Cost:** FREE (local computation)

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/replay/__init__.py` | Package init |
| `src/replay/od_scorer.py` | OD computation logic |
| `src/analysis/__init__.py` | Package init |
| `src/analysis/calibration.py` | Calibration metrics + plots |
| `config/experiments/rq1_calibration.json` | Experiment config |

## Files to Modify

| File | Change |
|------|--------|
| `src/experiment_runner.py` | Add `od`, `od_validate`, `calibration` phases |
| `src/judges/prompts.py` | Add outcome grader prompt |

---

## Config

`config/experiments/rq1_calibration.json`:

```json
{
  "experiment": {
    "name": "RQ1: Consequentiality Calibration",
    "experiment_id": "final_exp_perturbation_quality"
  },

  "od": {
    "enabled": true,
    "tier": 3,
    "grader_model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "temperature": 0.1,
    "max_tokens": 1000,
    "batch_size": 20
  },

  "od_validation": {
    "min_coverage": 0.7,
    "min_variance": 0.1,
    "spot_check_count": 20
  },

  "calibration": {
    "metrics": ["spearman_r", "pearson_r", "ece", "mce"],
    "n_bins": 10,
    "bootstrap_iterations": 1000,
    "stratify_by": ["benchmark", "perturbation_type", "position"],
    "output_dir": "results/rq1"
  },

  "judge_source": {
    "experiment_id": "exp_judge_ccg_v1",
    "judge_name": "claude-sonnet-4.5"
  },

  "execution": {
    "runner": "od,od_validate,calibration",
    "verbose": true
  }
}
```

---

## Success Criteria

| Outcome | Spearman r | ECE | Interpretation |
|---------|------------|-----|----------------|
| Judges calibrated | > 0.5 | < 0.1 | Judges work (unlikely) |
| **Judges detect but don't size** | 0.2 - 0.5 | 0.1 - 0.3 | **Expected** |
| Judges random | < 0.15 | > 0.3 | Judges useless |

---

## Artifacts

| Artifact | Location | For Paper |
|----------|----------|-----------|
| Calibration curve | `results/rq1/calibration_curve.png` | Figure 2 |
| Statistics | `results/rq1/statistics.json` | Section 4.1 |
| Findings | `results/rq1/findings.md` | Discussion |

---

## Dependencies

- Experiment `final_exp_perturbation_quality` exists with perturbations
- Judge evaluations exist in `exp_judge_ccg_v1`
- AWS Bedrock access for OD grader calls
