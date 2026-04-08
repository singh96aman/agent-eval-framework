# 5A. Human Labels

**Status: REQUIREMENTS DEFINED**

Human labels provide the ground-truth reference for evaluating LLM judge performance. Trained annotators assess detectability, localization, error type, impact, and propagation across a stratified sample of evaluation units.

---

## Design Principles

1. **Mode-aware**: Different annotation tasks receive different views (detectability vs consequence vs preference)
2. **Blinded**: Annotators don't know which trajectory is perturbed during detection tasks
3. **Calibrated**: Inter-annotator agreement is measured and reported
4. **Efficient**: Focus annotation effort on dimensions that matter for main claims
5. **Traceable**: Every label links back to evaluation_unit_id and annotator_id

---

## 5A.1 Role in the Pipeline

```
Section 4: Evaluation Unit
         |
         v
    Human Evaluator Views (from 4.6.1)
         |
         v
    5A. Human Labels
         |
    +----+----+
    |         |
    v         v
6A. Detection   6B. Consequentiality
    Metrics         Metrics
```

Human labels serve as:
- Ground truth for judge detectability metrics (6A)
- Reference for impact calibration analysis (6B)
- Inter-rater reliability baseline for judge variance comparison (6C)

---

## 5A.2 Annotation Dimensions

### Primary Dimensions (All Units)

| Dimension | Scale | Description |
|-----------|-------|-------------|
| **error_detected** | bool | Does the annotator believe an error exists? |
| **error_trajectory** | A \| B \| neither \| both | Which trajectory contains the error? |
| **confidence** | 1-5 | How confident is the annotator? |

### Localization Dimensions (If Error Detected)

| Dimension | Scale | Description |
|-----------|-------|-------------|
| **error_step_id** | canonical_step_id \| null | Which step contains the error? |
| **localization_accuracy** | exact \| near \| wrong | How precise is the localization? |

### Type & Impact Dimensions (Consequence View Only)

| Dimension | Scale | Description |
|-----------|-------|-------------|
| **error_type** | planning \| tool_selection \| parameter \| data_reference \| unclear | What kind of error? |
| **impact_tier** | 0-3 | 0=none, 1=minor, 2=moderate, 3=critical |
| **propagation_depth** | 0-3 | How many downstream steps affected? |

### Correctness Dimensions (Consequence View Only)

| Dimension | Scale | Description |
|-----------|-------|-------------|
| **correctness_a** | correct \| partially_correct \| incorrect | Is trajectory A's answer correct? |
| **correctness_b** | correct \| partially_correct \| incorrect | Is trajectory B's answer correct? |

### Preference Dimension (Preference View Only)

| Dimension | Scale | Description |
|-----------|-------|-------------|
| **preference** | A \| B \| tie | Which trajectory is better overall? |
| **preference_reason** | string | Brief justification |

---

## 5A.3 Annotation Modes

Human annotation uses three distinct modes corresponding to Section 4's view types:

### Mode A: Detectability Annotation

**Purpose:** Can the human detect and locate the error without knowing the answer?
**View:** Detectability view (no expected_answer shown)
**Sample:** All 350 human-sampled units

**Collected dimensions:**
- error_detected
- error_trajectory
- error_step_id (if detected)
- confidence

**Key constraint:** Annotator NEVER sees expected_answer in this mode.

### Mode B: Consequence Annotation

**Purpose:** How severe is the error's impact on correctness?
**View:** Consequence view (expected_answer shown)
**Sample:** Subset where error_detected=true OR stratified sample (~200 units)

**Collected dimensions:**
- error_type
- impact_tier
- propagation_depth
- correctness_a
- correctness_b

**Key constraint:** Only run after Mode A to prevent detection bias.

### Mode C: Preference Annotation

**Purpose:** Overall trajectory quality comparison
**View:** Preference view
**Sample:** Subset for preference-based evaluation (~150 units)

**Collected dimensions:**
- preference
- preference_reason

---

## 5A.4 Human Label Schema

### Annotation Record Object

```json
{
  // === Identity ===
  "annotation_id": "string (uuid)",
  "evaluation_unit_id": "string",
  "annotator_id": "string",
  "annotation_mode": "detectability | consequence | preference",
  "created_at": "datetime (ISO 8601)",
  
  // === View Information ===
  "view_file": "string (path to view JSON used)",
  "trajectory_a_variant_id": "string",
  "trajectory_b_variant_id": "string",
  
  // === Mode A: Detectability Labels ===
  "detectability": {
    "error_detected": "bool",
    "error_trajectory": "A | B | neither | both",
    "error_step_id": "string (canonical_step_id) | null",
    "confidence": "int (1-5)"
  },
  
  // === Mode B: Consequence Labels (optional) ===
  "consequence": {
    "error_type": "planning | tool_selection | parameter | data_reference | unclear | null",
    "impact_tier": "int (0-3) | null",
    "propagation_depth": "int (0-3) | null",
    "correctness_a": "correct | partially_correct | incorrect | null",
    "correctness_b": "correct | partially_correct | incorrect | null"
  },
  
  // === Mode C: Preference Labels (optional) ===
  "preference": {
    "preference": "A | B | tie | null",
    "preference_reason": "string | null"
  },
  
  // === Metadata ===
  "time_spent_seconds": "int",
  "notes": "string | null",
  "flagged_for_review": "bool"
}
```

### Aggregated Label Object (Post-Aggregation)

```json
{
  // === Identity ===
  "evaluation_unit_id": "string",
  
  // === Source Annotations ===
  "annotation_ids": ["string[]"],
  "num_annotators": "int",
  
  // === Aggregated Detectability ===
  "aggregated_detectability": {
    "error_detected_majority": "bool",
    "error_detected_agreement": "float (0-1)",
    "error_trajectory_majority": "A | B | neither | both",
    "error_trajectory_agreement": "float (0-1)",
    "error_step_id_majority": "string | null",
    "localization_agreement": "exact | near | mixed",
    "mean_confidence": "float"
  },
  
  // === Aggregated Consequence ===
  "aggregated_consequence": {
    "error_type_majority": "string | null",
    "error_type_agreement": "float (0-1)",
    "mean_impact_tier": "float",
    "impact_tier_std": "float",
    "mean_propagation_depth": "float",
    "correctness_a_majority": "string",
    "correctness_b_majority": "string"
  },
  
  // === Aggregated Preference ===
  "aggregated_preference": {
    "preference_majority": "A | B | tie",
    "preference_agreement": "float (0-1)"
  },
  
  // === Quality Flags ===
  "low_agreement_flag": "bool",
  "needs_adjudication": "bool"
}
```

---

## 5A.5 Mapping Human Labels to Perturbation Ground Truth

After unblinding, human labels are compared against perturbation records:

| Human Label | Ground Truth Source | Comparison |
|-------------|---------------------|------------|
| error_trajectory | blinding_key.is_a_perturbed | Did human identify the perturbed trajectory? |
| error_step_id | perturbation_record.target_step_canonical_id | Did human localize correctly? |
| error_type | perturbation_record.perturbation_family | Does human type match perturbation family? |
| impact_tier | perturbation_record.expected_impact | Compare human impact to expected |
| correctness_* | outcome evidence (5C) | Compare to objective outcome |

### Error Type Mapping

| Human Label | Maps to Perturbation Family |
|-------------|----------------------------|
| planning | structural |
| tool_selection | tool_selection |
| parameter | parameter |
| data_reference | data_reference |
| unclear | any (annotator uncertainty) |

### Localization Accuracy Definition

| Label | Definition |
|-------|------------|
| **exact** | error_step_id matches target_step_canonical_id exactly |
| **near** | error_step_id is within 1 step of target (by display_step_index) |
| **wrong** | error_step_id is more than 1 step away or null when error exists |

---

## 5A.6 Annotator Requirements

### Annotator Profile

| Requirement | Description |
|-------------|-------------|
| **Technical background** | CS/ML background, familiar with agent workflows |
| **Training** | Complete training module on trajectory evaluation |
| **Calibration** | Pass calibration test (>80% accuracy on gold set) |
| **Availability** | Commit to annotating assigned batch within timeframe |

### Training Protocol

1. **Introduction** (30 min): Study objectives, what constitutes an error
2. **Schema review** (30 min): All dimensions and their definitions
3. **Practice set** (60 min): 10 annotated examples with feedback
4. **Calibration test** (30 min): 10 units, must achieve >80% agreement with gold
5. **Edge cases** (30 min): Difficult examples and resolution strategies

### Calibration Gold Set

- 30 evaluation units with expert-adjudicated labels
- Mix of: obvious errors, subtle errors, placebos, clean trajectories
- Used for: training, calibration testing, drift monitoring

---

## 5A.7 Annotation Interface Requirements

### Mode A Interface (Detectability)

```
┌─────────────────────────────────────────────────────────────────────┐
│ Task: [task_text]                                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Trajectory A                    │  Trajectory B                    │
│  ─────────────                   │  ─────────────                   │
│  Step 1: [description]           │  Step 1: [description]           │
│  Step 2: [description]           │  Step 2: [description]           │
│  Step 3: [description]           │  Step 3: [description]           │
│  ...                             │  ...                             │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ Does one trajectory contain an error?                               │
│ ○ Yes, Trajectory A  ○ Yes, Trajectory B  ○ Neither  ○ Both         │
│                                                                     │
│ [If yes] Which step? [dropdown: Step 1, Step 2, ...]                │
│                                                                     │
│ Confidence: ○ 1  ○ 2  ○ 3  ○ 4  ○ 5                                 │
│                                                                     │
│ Notes: [text field]                                                 │
│                                                                     │
│ [Flag for Review]                            [Submit]               │
└─────────────────────────────────────────────────────────────────────┘
```

### Mode B Interface (Consequence)

```
┌─────────────────────────────────────────────────────────────────────┐
│ Task: [task_text]                                                   │
│ Expected Answer: [expected_answer]                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Trajectory A (Final: [answer])  │  Trajectory B (Final: [answer])  │
│  ─────────────                   │  ─────────────                   │
│  [full steps with answers]       │  [full steps with answers]       │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ Error Type: ○ Planning  ○ Tool Selection  ○ Parameter               │
│             ○ Data Reference  ○ Unclear                             │
│                                                                     │
│ Impact: ○ 0 (None)  ○ 1 (Minor)  ○ 2 (Moderate)  ○ 3 (Critical)     │
│                                                                     │
│ Propagation: ○ 0  ○ 1  ○ 2  ○ 3  downstream steps affected          │
│                                                                     │
│ Correctness A: ○ Correct  ○ Partially Correct  ○ Incorrect          │
│ Correctness B: ○ Correct  ○ Partially Correct  ○ Incorrect          │
│                                                                     │
│ [Flag for Review]                            [Submit]               │
└─────────────────────────────────────────────────────────────────────┘
```

### Interface Requirements

| Requirement | Description |
|-------------|-------------|
| **Side-by-side display** | Both trajectories visible simultaneously |
| **Step navigation** | Click to expand/collapse steps |
| **Highlighting** | Selected step highlighted in both views |
| **Progress tracking** | Show completed/remaining annotations |
| **Time tracking** | Automatic time-on-task logging |
| **Skip & flag** | Allow skipping unclear cases with flag |

---

## 5A.8 Sampling Strategy

### Target Sample Size

| Mode | Target Count | Per-Annotator |
|------|--------------|---------------|
| Mode A (Detectability) | 350 units | ~120 per annotator (3 annotators overlap) |
| Mode B (Consequence) | 200 units | ~70 per annotator |
| Mode C (Preference) | 150 units | ~50 per annotator |

### Overlap for Agreement

| Overlap Type | Count | Purpose |
|--------------|-------|---------|
| **Full overlap** | 50 units | All annotators label; compute inter-rater |
| **Pairwise overlap** | 100 units | Each pair overlaps on additional 50 |
| **Single annotator** | 200 units | Remaining units, one annotator each |

### Stratification (Same as Section 4.8)

| Stratum | Target in Human Sample |
|---------|------------------------|
| High-impact (expected_impact=3) | 100 units |
| Placebo | 75 units |
| Fine-grained subtle | 100 units |
| Coarse-grained | 75 units |

Coverage minimums:
- ≥50 per benchmark
- ≥50 per perturbation class
- ≥20 per perturbation family
- ≥75 per detectability band

---

## 5A.9 Quality Control

### Real-Time QC

| Check | Trigger | Action |
|-------|---------|--------|
| **Time too fast** | <30s per unit | Flag for review |
| **Time too slow** | >10min per unit | Check for issues |
| **Consecutive same** | 5+ identical responses | Alert annotator |
| **Gold set drift** | <70% on embedded gold | Pause for recalibration |

### Post-Hoc QC

| Check | Method | Threshold |
|-------|--------|-----------|
| **Inter-annotator agreement** | Krippendorff's alpha | α > 0.6 for binary, α > 0.4 for ordinal |
| **Systematic bias** | Compare to gold set | Deviation < 15% from gold |
| **Annotator consistency** | Self-agreement on duplicates | >85% self-agreement |

### Adjudication Protocol

For units with low agreement (disagreement on error_detected or error_trajectory):

1. Expert reviewer examines all annotations
2. Expert provides adjudicated label
3. Adjudicated label becomes ground truth
4. Original annotations preserved for agreement analysis

---

## 5A.10 Data Location

```
data/evaluation_units/
├── views/
│   └── human/
│       ├── detectability/           # Mode A views
│       ├── consequence/             # Mode B views
│       └── preference/              # Mode C views
└── human_labels/
    ├── raw/
    │   ├── annotations_batch_001.json
    │   ├── annotations_batch_002.json
    │   └── ...
    ├── aggregated/
    │   └── aggregated_labels.json
    ├── gold/
    │   └── calibration_gold_set.json
    └── metadata/
        ├── annotator_assignments.json
        ├── agreement_report.json
        └── qc_report.json
```

---

## 5A.11 Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| **Preparation** | 1 week | Create views, build interface, prepare training |
| **Training** | 3 days | Train annotators, calibration test |
| **Pilot** | 3 days | 50 units, check agreement, iterate |
| **Main annotation** | 2 weeks | Full 350-unit sample |
| **QC & Adjudication** | 3 days | Review low-agreement, adjudicate |
| **Aggregation** | 1 day | Compute aggregated labels |

---

## 5A.12 Implementation Plan

### Module Structure

```
src/human_labels/
├── __init__.py
├── interface/
│   ├── __init__.py
│   ├── detectability.py      # Mode A interface logic
│   ├── consequence.py        # Mode B interface logic
│   └── preference.py         # Mode C interface logic
├── sampling.py               # Annotator assignment, stratification
├── aggregation.py            # Compute aggregated labels
├── agreement.py              # Inter-rater reliability metrics
├── qc.py                     # Quality control checks
├── adjudication.py           # Expert adjudication workflow
└── export.py                 # Export for downstream analysis
```

### Scripts

```bash
# Generate annotation assignments
python -m src.human_labels.sampling \
  --evaluation-units data/evaluation_units/canonical/ \
  --sample-config configs/human_sample_config.yaml \
  --output data/evaluation_units/human_labels/metadata/

# Validate raw annotations
python -m src.human_labels.qc \
  --annotations data/evaluation_units/human_labels/raw/ \
  --gold data/evaluation_units/human_labels/gold/ \
  --output data/evaluation_units/human_labels/metadata/qc_report.json

# Compute agreement
python -m src.human_labels.agreement \
  --annotations data/evaluation_units/human_labels/raw/ \
  --output data/evaluation_units/human_labels/metadata/agreement_report.json

# Aggregate labels
python -m src.human_labels.aggregation \
  --annotations data/evaluation_units/human_labels/raw/ \
  --output data/evaluation_units/human_labels/aggregated/
```

---

## 5A.13 Dependencies

| Dependency | Section | What's Needed |
|------------|---------|---------------|
| Evaluation Unit (Section 4) | Input | Human views for annotation |
| Blinding Key (Section 4) | Unblinding | Maps A/B to baseline/perturbed |
| Perturbation Record (Section 3) | Ground truth | True error location and type |
| Detection Metrics (Section 6A) | Output | Human labels as judge reference |
| Consequentiality (Section 6B) | Output | Human impact ratings as reference |

---

## 5A.14 Success Criteria

Section 5A is complete when:

### Data Collection
1. [ ] 350 evaluation units annotated in Mode A (detectability)
2. [ ] ≥200 units annotated in Mode B (consequence)
3. [ ] ≥150 units annotated in Mode C (preference)
4. [ ] All annotations have evaluation_unit_id linkage

### Coverage
5. [ ] ≥50 units per benchmark in human sample
6. [ ] ≥50 units per perturbation class
7. [ ] ≥20 units per perturbation family
8. [ ] ≥75 units per detectability band (0, 1, 2)

### Agreement
9. [ ] Inter-annotator agreement computed for all dimensions
10. [ ] Krippendorff's α > 0.6 for error_detected
11. [ ] Krippendorff's α > 0.4 for impact_tier (ordinal)
12. [ ] Agreement report generated

### Quality
13. [ ] <5% of annotations flagged for review
14. [ ] All low-agreement units adjudicated
15. [ ] QC report shows no systematic annotator bias
16. [ ] Gold set accuracy maintained >80% throughout

### Output
17. [ ] Aggregated labels computed for all annotated units
18. [ ] Labels ready for analysis scripts (6A, 6B, 6C)

---

## 5A.15 Example Annotation Records

### Mode A: Detectability Annotation (GAIA Example)

```json
{
  "annotation_id": "ann_001_detect",
  "evaluation_unit_id": "eval::gaia_122::001",
  "annotator_id": "annotator_A",
  "annotation_mode": "detectability",
  "created_at": "2026-04-10T14:30:00Z",
  
  "view_file": "views/human/detectability/eval_gaia_122_001.json",
  "trajectory_a_variant_id": "gaia_122::pert::001",
  "trajectory_b_variant_id": "gaia_122::base",
  
  "detectability": {
    "error_detected": true,
    "error_trajectory": "A",
    "error_step_id": "gaia_122::step::2",
    "confidence": 3
  },
  
  "consequence": null,
  "preference": null,
  
  "time_spent_seconds": 145,
  "notes": "Step 2 extracts 10 layers but I recall BERT base has 12",
  "flagged_for_review": false
}
```

### Mode B: Consequence Annotation (Same Unit)

```json
{
  "annotation_id": "ann_001_conseq",
  "evaluation_unit_id": "eval::gaia_122::001",
  "annotator_id": "annotator_A",
  "annotation_mode": "consequence",
  "created_at": "2026-04-10T14:35:00Z",
  
  "view_file": "views/human/consequence/eval_gaia_122_001.json",
  "trajectory_a_variant_id": "gaia_122::pert::001",
  "trajectory_b_variant_id": "gaia_122::base",
  
  "detectability": null,
  
  "consequence": {
    "error_type": "data_reference",
    "impact_tier": 3,
    "propagation_depth": 1,
    "correctness_a": "incorrect",
    "correctness_b": "correct"
  },
  
  "preference": null,
  
  "time_spent_seconds": 95,
  "notes": "Wrong layer count propagates to final subtraction, answer is 4 instead of 6",
  "flagged_for_review": false
}
```

### Aggregated Label (After Multi-Annotator)

```json
{
  "evaluation_unit_id": "eval::gaia_122::001",
  
  "annotation_ids": ["ann_001_detect", "ann_002_detect", "ann_003_detect"],
  "num_annotators": 3,
  
  "aggregated_detectability": {
    "error_detected_majority": true,
    "error_detected_agreement": 1.0,
    "error_trajectory_majority": "A",
    "error_trajectory_agreement": 1.0,
    "error_step_id_majority": "gaia_122::step::2",
    "localization_agreement": "exact",
    "mean_confidence": 3.67
  },
  
  "aggregated_consequence": {
    "error_type_majority": "data_reference",
    "error_type_agreement": 1.0,
    "mean_impact_tier": 3.0,
    "impact_tier_std": 0.0,
    "mean_propagation_depth": 1.0,
    "correctness_a_majority": "incorrect",
    "correctness_b_majority": "correct"
  },
  
  "aggregated_preference": {
    "preference_majority": "B",
    "preference_agreement": 1.0
  },
  
  "low_agreement_flag": false,
  "needs_adjudication": false
}
```
