# Quality Gates Design for LLM-as-Judge Pipeline

This document specifies comprehensive quality gates for each stage transition in the pipeline, designed to catch the issues discovered during experimentation.

---

## Overview

### Pipeline Stages
1. **load** - Load trajectories from benchmarks
2. **typing** - Add typed representation
3. **perturb** - Generate controlled perturbations
4. **evaluation_unit** - Create baseline/perturbed pairs
5. **sample** - Select units for annotation
6. **annotate** - Human labels (5A)
7. **judge** - LLM judge evaluation (5B)
8. **compute** - Outcome evidence (5C)
9. **analysis** - Section 6 metrics

### Issues to Prevent
| Issue | Discovery | Impact |
|-------|-----------|--------|
| Sampling used different success criteria than grading | All baselines "failed" | Invalid ground truth |
| 42% perturbations had artifact contamination | `_old`, `_mutated` markers leaked | Invalid perturbations |
| 49% perturbations at last step | Position bias | Unbalanced coverage |
| A/B assignment not randomized | Predictable blinding | Bias in human labels |
| Outcome degradation had no variance | All same value | Cannot compute CCorr |
| Taxonomy mismatch | Human vs judge labels | Metric comparison failed |

---

## Gate 1: POST-LOAD (load -> typing)

### Gate 1.1: Trajectory Count Gate
- **When**: After load phase completes
- **Checks**: 
  - Total trajectories >= config.min_trajectories (default: 100)
  - Per-benchmark counts >= config.min_per_benchmark (default: 20)
- **Thresholds**:
  - FAIL: total < 50 OR any benchmark < 10
  - WARN: total < 100 OR any benchmark < 20
- **Implementation**:
```python
# src/gates/post_load.py
def gate_trajectory_count(trajectories, config):
    """Check sufficient trajectories were loaded."""
    counts = Counter(t.get("benchmark") for t in trajectories)
    min_total = config.get("min_trajectories", 100)
    min_per_bench = config.get("min_per_benchmark", 20)
    
    errors = []
    warnings = []
    
    if len(trajectories) < 50:
        errors.append(f"FAIL: Only {len(trajectories)} trajectories (min 50)")
    elif len(trajectories) < min_total:
        warnings.append(f"WARN: Only {len(trajectories)} trajectories (target {min_total})")
    
    for bench, count in counts.items():
        if count < 10:
            errors.append(f"FAIL: {bench} has only {count} trajectories (min 10)")
        elif count < min_per_bench:
            warnings.append(f"WARN: {bench} has only {count} trajectories (target {min_per_bench})")
    
    return GateResult(passed=len(errors)==0, errors=errors, warnings=warnings)
```

### Gate 1.2: Success Criteria Consistency Gate
- **When**: After load phase completes
- **Checks**:
  - Success field exists on all trajectories
  - Success rate is plausible (not 0% or 100%)
  - Success criteria matches grading criteria used later
- **Thresholds**:
  - FAIL: success_rate == 0% OR success_rate == 100%
  - FAIL: success field missing on >5% of trajectories
  - WARN: success_rate < 20% OR success_rate > 80%
- **Implementation**:
```python
def gate_success_criteria(trajectories, grading_config):
    """Verify success field consistency with downstream grading."""
    missing_success = sum(1 for t in trajectories if "success" not in t)
    success_count = sum(1 for t in trajectories if t.get("success"))
    total = len(trajectories)
    
    errors = []
    warnings = []
    
    if missing_success / total > 0.05:
        errors.append(f"FAIL: {missing_success} trajectories missing success field")
    
    success_rate = success_count / total if total > 0 else 0
    if success_rate == 0:
        errors.append("FAIL: 0% success rate - check success criteria definition")
    elif success_rate == 1.0:
        errors.append("FAIL: 100% success rate - check success criteria definition")
    elif success_rate < 0.2:
        warnings.append(f"WARN: Low success rate ({success_rate:.1%})")
    elif success_rate > 0.8:
        warnings.append(f"WARN: High success rate ({success_rate:.1%})")
    
    # Cross-check: sample and verify grading matches
    sample = random.sample(trajectories, min(10, len(trajectories)))
    mismatches = 0
    for t in sample:
        graded = grade_trajectory(t, grading_config)
        if graded != t.get("success"):
            mismatches += 1
    
    if mismatches > 0:
        errors.append(f"FAIL: {mismatches}/10 sampled trajectories have success/grading mismatch")
    
    return GateResult(passed=len(errors)==0, errors=errors, warnings=warnings)
```

---

## Gate 2: POST-TYPING (typing -> perturb)

### Gate 2.1: Typing Coverage Gate
- **When**: After typing phase completes
- **Checks**:
  - All trajectories have typed representation
  - All steps have step_role assigned
  - Perturbable slots identified
- **Thresholds**:
  - FAIL: typing_coverage < 95%
  - FAIL: steps_without_role > 2%
  - WARN: avg_slots_per_trajectory < 3
- **Implementation**:
```python
def gate_typing_coverage(typed_trajectories, original_count):
    """Check typing phase produced complete typed representations."""
    errors = []
    warnings = []
    
    coverage = len(typed_trajectories) / original_count if original_count > 0 else 0
    if coverage < 0.95:
        errors.append(f"FAIL: Only {coverage:.1%} of trajectories typed")
    
    steps_without_role = 0
    total_steps = 0
    total_slots = 0
    
    for tt in typed_trajectories:
        for step in tt.steps:
            total_steps += 1
            if not step.step_role:
                steps_without_role += 1
            total_slots += len(getattr(step, 'perturbable_slots', []))
    
    if total_steps > 0 and steps_without_role / total_steps > 0.02:
        errors.append(f"FAIL: {steps_without_role}/{total_steps} steps without role")
    
    avg_slots = total_slots / len(typed_trajectories) if typed_trajectories else 0
    if avg_slots < 3:
        warnings.append(f"WARN: Low slot density ({avg_slots:.1f} avg slots per trajectory)")
    
    return GateResult(passed=len(errors)==0, errors=errors, warnings=warnings)
```

### Gate 2.2: Step Role Distribution Gate
- **When**: After typing phase completes
- **Checks**:
  - Reasonable distribution of step roles
  - No single role dominates (>80%)
- **Thresholds**:
  - WARN: any role < 5% (may indicate classification issues)
  - WARN: any role > 80%
- **Implementation**:
```python
def gate_step_role_distribution(typed_trajectories):
    """Check step role distribution is reasonable."""
    role_counts = Counter()
    for tt in typed_trajectories:
        for step in tt.steps:
            role_counts[step.step_role] += 1
    
    total = sum(role_counts.values())
    warnings = []
    
    for role, count in role_counts.items():
        pct = count / total if total > 0 else 0
        if pct < 0.05:
            warnings.append(f"WARN: Role '{role}' underrepresented ({pct:.1%})")
        if pct > 0.8:
            warnings.append(f"WARN: Role '{role}' dominates ({pct:.1%})")
    
    return GateResult(passed=True, errors=[], warnings=warnings)
```

---

## Gate 3: POST-PERTURB (perturb -> evaluation_unit)

### Gate 3.1: Artifact Contamination Gate
- **When**: After perturb phase completes
- **Checks**:
  - No `_old`, `_mutated`, `_original`, `_backup` strings in perturbed content
  - No debug markers leaked into trajectories
- **Thresholds**:
  - FAIL: contamination_rate > 5%
  - WARN: contamination_rate > 0%
- **Implementation**:
```python
CONTAMINATION_PATTERNS = [
    r"_old\b", r"_mutated\b", r"_original\b", r"_backup\b",
    r"PERTURBATION:", r"DEBUG:", r"\[INJECTED\]"
]

def gate_artifact_contamination(perturbations):
    """Check for leaked generation artifacts in perturbations."""
    contaminated = 0
    contamination_details = []
    
    for p in perturbations:
        perturbed_value = str(p.get("perturbed_value", ""))
        perturbed_traj = json.dumps(p.get("perturbed_trajectory", {}))
        combined = perturbed_value + perturbed_traj
        
        for pattern in CONTAMINATION_PATTERNS:
            if re.search(pattern, combined, re.IGNORECASE):
                contaminated += 1
                contamination_details.append({
                    "perturbation_id": p.get("perturbation_id"),
                    "pattern": pattern
                })
                break
    
    rate = contaminated / len(perturbations) if perturbations else 0
    errors = []
    warnings = []
    
    if rate > 0.05:
        errors.append(f"FAIL: {rate:.1%} contamination rate ({contaminated} perturbations)")
    elif rate > 0:
        warnings.append(f"WARN: {contaminated} perturbations have artifacts")
    
    return GateResult(
        passed=len(errors)==0, 
        errors=errors, 
        warnings=warnings,
        details={"contamination_details": contamination_details[:10]}
    )
```

### Gate 3.2: Position Distribution Gate
- **When**: After perturb phase completes
- **Checks**:
  - Perturbations are distributed across step positions
  - Not concentrated at last step
- **Thresholds**:
  - FAIL: last_step_rate > 40%
  - WARN: last_step_rate > 25%
  - WARN: early_rate < 15% (early = first 25% of steps)
- **Implementation**:
```python
def gate_position_distribution(perturbations, typed_trajectories):
    """Check perturbation position distribution."""
    traj_lengths = {
        t.trajectory_id: len(t.steps) 
        for t in typed_trajectories
    }
    
    position_bins = {"early": 0, "middle": 0, "late": 0, "last": 0}
    
    for p in perturbations:
        traj_id = p.get("original_trajectory_id")
        step_idx = p.get("target_step_index", 0)
        traj_len = traj_lengths.get(traj_id, 10)
        
        relative_pos = step_idx / traj_len if traj_len > 0 else 0
        
        if step_idx == traj_len - 1:
            position_bins["last"] += 1
        elif relative_pos < 0.25:
            position_bins["early"] += 1
        elif relative_pos < 0.75:
            position_bins["middle"] += 1
        else:
            position_bins["late"] += 1
    
    total = len(perturbations)
    errors = []
    warnings = []
    
    last_rate = position_bins["last"] / total if total > 0 else 0
    early_rate = position_bins["early"] / total if total > 0 else 0
    
    if last_rate > 0.4:
        errors.append(f"FAIL: {last_rate:.1%} of perturbations at last step")
    elif last_rate > 0.25:
        warnings.append(f"WARN: {last_rate:.1%} of perturbations at last step (target <25%)")
    
    if early_rate < 0.15:
        warnings.append(f"WARN: Only {early_rate:.1%} early-step perturbations (target >15%)")
    
    return GateResult(
        passed=len(errors)==0, 
        errors=errors, 
        warnings=warnings,
        details={"position_distribution": position_bins}
    )
```

### Gate 3.3: QC Pass Rate Gate
- **When**: After perturb phase completes
- **Checks**:
  - Perturbation QC pass rate is acceptable
  - Class/family/type taxonomy is consistent
- **Thresholds**:
  - FAIL: qc_pass_rate < 80%
  - WARN: qc_pass_rate < 90%
  - WARN: invalid_taxonomy > 2%
- **Implementation**:
```python
def gate_qc_pass_rate(perturbations):
    """Check perturbation QC statistics."""
    valid = sum(1 for p in perturbations if p.get("generation_status") == "valid")
    invalid = sum(1 for p in perturbations if p.get("generation_status") == "invalid")
    borderline = sum(1 for p in perturbations if p.get("generation_status") == "borderline")
    
    total = len(perturbations)
    pass_rate = (valid + borderline) / total if total > 0 else 0
    invalid_rate = invalid / total if total > 0 else 0
    
    errors = []
    warnings = []
    
    if pass_rate < 0.80:
        errors.append(f"FAIL: QC pass rate {pass_rate:.1%} (min 80%)")
    elif pass_rate < 0.90:
        warnings.append(f"WARN: QC pass rate {pass_rate:.1%} (target 90%)")
    
    # Check taxonomy validity
    taxonomy_errors = 0
    for p in perturbations:
        if p.get("qc_checks_failed"):
            if any("class_family" in c or "family_type" in c for c in p["qc_checks_failed"]):
                taxonomy_errors += 1
    
    tax_rate = taxonomy_errors / total if total > 0 else 0
    if tax_rate > 0.02:
        warnings.append(f"WARN: {taxonomy_errors} perturbations have taxonomy issues")
    
    return GateResult(passed=len(errors)==0, errors=errors, warnings=warnings)
```

### Gate 3.4: Class Distribution Gate
- **When**: After perturb phase completes
- **Checks**:
  - Perturbation class distribution matches targets
  - All classes represented
- **Thresholds**:
  - FAIL: any class missing entirely
  - WARN: deviation > 10pp from target distribution
- **Implementation**:
```python
def gate_class_distribution(perturbations, target_distribution):
    """Check perturbation class distribution matches targets."""
    class_counts = Counter(p.get("perturbation_class") for p in perturbations)
    total = len(perturbations)
    
    errors = []
    warnings = []
    
    required_classes = ["placebo", "fine_grained", "coarse_grained"]
    for cls in required_classes:
        if class_counts.get(cls, 0) == 0:
            errors.append(f"FAIL: No perturbations of class '{cls}'")
    
    if target_distribution:
        for cls, target_pct in target_distribution.items():
            actual_pct = class_counts.get(cls, 0) / total if total > 0 else 0
            if abs(actual_pct - target_pct) > 0.10:
                warnings.append(
                    f"WARN: Class '{cls}' is {actual_pct:.1%} (target {target_pct:.1%})"
                )
    
    return GateResult(passed=len(errors)==0, errors=errors, warnings=warnings)
```

---

## Gate 4: POST-EVALUATION_UNIT (evaluation_unit -> sample)

### Gate 4.1: Blinding Balance Gate
- **When**: After evaluation_unit phase completes
- **Checks**:
  - A/B assignment is approximately balanced (45-55%)
  - Assignment is truly random (not predictable)
- **Thresholds**:
  - FAIL: balance_ratio < 0.40 OR balance_ratio > 0.60
  - WARN: balance_ratio < 0.45 OR balance_ratio > 0.55
- **Implementation**:
```python
def gate_blinding_balance(evaluation_units):
    """Check A/B assignment is balanced and random."""
    a_is_baseline = sum(
        1 for u in evaluation_units 
        if u.get("blinding", {}).get("is_a_baseline")
    )
    total = len(evaluation_units)
    ratio = a_is_baseline / total if total > 0 else 0
    
    errors = []
    warnings = []
    
    if ratio < 0.40 or ratio > 0.60:
        errors.append(f"FAIL: Blinding balance is {ratio:.1%} (must be 40-60%)")
    elif ratio < 0.45 or ratio > 0.55:
        warnings.append(f"WARN: Blinding balance is {ratio:.1%} (target 45-55%)")
    
    # Check randomness: by trajectory source
    by_source = defaultdict(list)
    for u in evaluation_units:
        source = u.get("source_trajectory_id", "")[:10]  # First part of ID
        is_a_baseline = u.get("blinding", {}).get("is_a_baseline")
        by_source[source].append(is_a_baseline)
    
    # If same source always gets same assignment, that's suspicious
    suspicious_sources = []
    for source, assignments in by_source.items():
        if len(set(assignments)) == 1 and len(assignments) > 3:
            suspicious_sources.append(source)
    
    if suspicious_sources:
        warnings.append(f"WARN: {len(suspicious_sources)} sources have identical A/B assignments")
    
    return GateResult(
        passed=len(errors)==0, 
        errors=errors, 
        warnings=warnings,
        details={"balance_ratio": ratio}
    )
```

### Gate 4.2: Unit Validation Gate
- **When**: After evaluation_unit phase completes
- **Checks**:
  - All units pass structure validation
  - All units pass ID format validation
- **Thresholds**:
  - FAIL: validation_pass_rate < 95%
  - WARN: any validation warnings
- **Implementation**:
```python
def gate_unit_validation(evaluation_units):
    """Run full validation on all evaluation units."""
    from src.evaluation.validators import validate_all
    
    errors = []
    warnings = []
    failed_units = []
    
    for unit in evaluation_units:
        result = validate_all(unit)
        if not result["passed"]:
            failed_units.append({
                "unit_id": result["unit_id"],
                "error_count": result["total_errors"]
            })
        if result["total_warnings"] > 0:
            warnings.append(f"Unit {result['unit_id']}: {result['total_warnings']} warnings")
    
    pass_rate = (len(evaluation_units) - len(failed_units)) / len(evaluation_units)
    
    if pass_rate < 0.95:
        errors.append(f"FAIL: Only {pass_rate:.1%} of units pass validation")
        errors.append(f"  Failed units: {[u['unit_id'] for u in failed_units[:5]]}...")
    
    return GateResult(
        passed=len(errors)==0, 
        errors=errors, 
        warnings=warnings[:10],  # Limit warnings
        details={"failed_units": failed_units[:20]}
    )
```

---

## Gate 5: POST-SAMPLE (sample -> annotate)

### Gate 5.1: Sample Stratification Gate
- **When**: After sample phase completes
- **Checks**:
  - Sample covers all perturbation types
  - Sample covers all benchmarks
  - Sample covers all position bins
- **Thresholds**:
  - FAIL: any condition has 0 samples
  - WARN: any condition has < 5% of expected
- **Implementation**:
```python
def gate_sample_stratification(samples, config):
    """Check sample stratification coverage."""
    total = len(samples)
    
    by_type = Counter(s.get("perturbation_type") for s in samples)
    by_benchmark = Counter(s.get("benchmark") for s in samples)
    by_position = Counter(s.get("perturbation_position") for s in samples)
    
    errors = []
    warnings = []
    
    # Required conditions
    required_types = config.get("required_types", [
        "planning", "tool_selection", "parameter", "data_reference"
    ])
    required_benchmarks = config.get("required_benchmarks", [
        "toolbench", "gaia", "swebench"
    ])
    required_positions = config.get("required_positions", [
        "early", "middle", "late"
    ])
    
    for t in required_types:
        if by_type.get(t, 0) == 0:
            errors.append(f"FAIL: No samples of type '{t}'")
        elif by_type[t] / total < 0.05:
            warnings.append(f"WARN: Only {by_type[t]} samples of type '{t}'")
    
    for b in required_benchmarks:
        if by_benchmark.get(b, 0) == 0:
            errors.append(f"FAIL: No samples from benchmark '{b}'")
        elif by_benchmark[b] / total < 0.15:
            warnings.append(f"WARN: Only {by_benchmark[b]} samples from '{b}'")
    
    for p in required_positions:
        if by_position.get(p, 0) == 0:
            errors.append(f"FAIL: No samples at position '{p}'")
    
    return GateResult(
        passed=len(errors)==0, 
        errors=errors, 
        warnings=warnings,
        details={
            "by_type": dict(by_type),
            "by_benchmark": dict(by_benchmark),
            "by_position": dict(by_position)
        }
    )
```

### Gate 5.2: Baseline Success Rate Gate
- **When**: After sample phase completes
- **Checks**:
  - Sampled baseline trajectories have mixed success/fail
  - Not all baselines are failures (would indicate criteria mismatch)
- **Thresholds**:
  - FAIL: baseline_success_rate == 0% OR baseline_success_rate == 100%
  - WARN: baseline_success_rate < 30% OR > 90%
- **Implementation**:
```python
def gate_baseline_success_rate(samples, evaluation_units):
    """Check baseline success rates in sample match expectations."""
    unit_lookup = {u["evaluation_unit_id"]: u for u in evaluation_units}
    
    baseline_success = 0
    total = 0
    
    for s in samples:
        unit_id = s.get("evaluation_unit_id")
        if unit_id in unit_lookup:
            unit = unit_lookup[unit_id]
            baseline = unit.get("baseline", {})
            if baseline.get("outcome_passed") or baseline.get("is_successful"):
                baseline_success += 1
            total += 1
    
    rate = baseline_success / total if total > 0 else 0
    
    errors = []
    warnings = []
    
    if rate == 0:
        errors.append("FAIL: All sampled baselines are failures - check success criteria!")
    elif rate == 1.0:
        errors.append("FAIL: All sampled baselines are successes - check success criteria!")
    elif rate < 0.30:
        warnings.append(f"WARN: Low baseline success rate in sample ({rate:.1%})")
    elif rate > 0.90:
        warnings.append(f"WARN: High baseline success rate in sample ({rate:.1%})")
    
    return GateResult(
        passed=len(errors)==0, 
        errors=errors, 
        warnings=warnings,
        details={"baseline_success_rate": rate}
    )
```

---

## Gate 6: POST-ANNOTATE (annotate -> judge)

### Gate 6.1: Annotation Completeness Gate
- **When**: After annotate phase completes
- **Checks**:
  - All sampled units have annotations
  - Required fields filled
- **Thresholds**:
  - FAIL: completion_rate < 90%
  - WARN: completion_rate < 95%
- **Implementation**:
```python
def gate_annotation_completeness(annotations, sample_ids):
    """Check annotation completion rate."""
    annotated_ids = {a.evaluation_unit_id for a in annotations}
    missing = set(sample_ids) - annotated_ids
    
    completion = len(annotated_ids) / len(sample_ids) if sample_ids else 0
    
    errors = []
    warnings = []
    
    if completion < 0.90:
        errors.append(f"FAIL: Only {completion:.1%} annotation completion")
        errors.append(f"  Missing: {list(missing)[:5]}...")
    elif completion < 0.95:
        warnings.append(f"WARN: {completion:.1%} annotation completion (target 95%)")
    
    return GateResult(
        passed=len(errors)==0, 
        errors=errors, 
        warnings=warnings,
        details={"missing_ids": list(missing)[:20]}
    )
```

### Gate 6.2: Annotation Quality Gate
- **When**: After annotate phase completes
- **Checks**:
  - Annotation time within bounds (30s-600s)
  - No consecutive same responses (inattention)
  - Gold set accuracy (if available)
- **Thresholds**:
  - FAIL: too_fast_rate > 20%
  - FAIL: gold_accuracy < 70%
  - WARN: consecutive_same_flag
- **Implementation**:
```python
def gate_annotation_quality(annotations, gold_set=None):
    """Run QC checks on annotations."""
    from src.human_labels.qc import run_all_qc_checks
    
    qc_report = run_all_qc_checks(annotations, gold_set)
    
    errors = []
    warnings = []
    
    total = len(annotations)
    too_fast_rate = qc_report["time_bounds"]["too_fast_count"] / total if total > 0 else 0
    
    if too_fast_rate > 0.20:
        errors.append(f"FAIL: {too_fast_rate:.1%} of annotations are too fast (<30s)")
    elif too_fast_rate > 0.10:
        warnings.append(f"WARN: {too_fast_rate:.1%} of annotations are too fast")
    
    if qc_report["consecutive_same"]["overall_flagged"]:
        for ann_id, data in qc_report["consecutive_same"]["by_annotator"].items():
            if data["flagged"]:
                warnings.append(f"WARN: Annotator {ann_id} has consecutive same pattern")
    
    if gold_set and qc_report.get("gold_accuracy"):
        for ann_id, gold_data in qc_report["gold_accuracy"].items():
            if gold_data["accuracy"] is not None and gold_data["accuracy"] < 0.70:
                errors.append(f"FAIL: Annotator {ann_id} gold accuracy {gold_data['accuracy']:.1%}")
    
    return GateResult(
        passed=len(errors)==0, 
        errors=errors, 
        warnings=warnings,
        details={"qc_report_summary": qc_report["summary"]}
    )
```

### Gate 6.3: Taxonomy Alignment Gate
- **When**: After annotate phase completes
- **Checks**:
  - Human label taxonomy matches judge taxonomy
  - Error types are in valid enum
- **Thresholds**:
  - FAIL: taxonomy_mismatch_rate > 10%
  - WARN: any unknown error types
- **Implementation**:
```python
VALID_ERROR_TYPES = {"planning", "tool_selection", "parameter", "data_reference", "unclear"}
JUDGE_TO_HUMAN_MAP = {
    "planning": "planning",
    "tool_selection": "tool_selection", 
    "parameter": "parameter",
    "data_reference": "data_reference",
    "other": "unclear"
}

def gate_taxonomy_alignment(annotations):
    """Check human labels use valid taxonomy."""
    unknown_types = Counter()
    
    for ann in annotations:
        if ann.consequence and ann.consequence.error_type:
            error_type = ann.consequence.error_type
            if error_type not in VALID_ERROR_TYPES:
                unknown_types[error_type] += 1
    
    errors = []
    warnings = []
    
    total_typed = sum(
        1 for a in annotations 
        if a.consequence and a.consequence.error_type
    )
    unknown_count = sum(unknown_types.values())
    
    if total_typed > 0 and unknown_count / total_typed > 0.10:
        errors.append(f"FAIL: {unknown_count}/{total_typed} annotations have unknown error types")
    elif unknown_count > 0:
        warnings.append(f"WARN: {unknown_count} annotations have unknown error types: {dict(unknown_types)}")
    
    return GateResult(passed=len(errors)==0, errors=errors, warnings=warnings)
```

---

## Gate 7: POST-JUDGE (judge -> compute)

### Gate 7.1: Judge Coverage Gate
- **When**: After judge phase completes
- **Checks**:
  - All sampled units have judge outputs
  - All configured judge models ran
- **Thresholds**:
  - FAIL: coverage < 90%
  - WARN: coverage < 95%
- **Implementation**:
```python
def gate_judge_coverage(judge_outputs, sample_ids, expected_models):
    """Check judge evaluation coverage."""
    outputs_by_unit = defaultdict(set)
    for jo in judge_outputs:
        outputs_by_unit[jo.evaluation_unit_id].add(jo.judge_model)
    
    missing_units = set(sample_ids) - set(outputs_by_unit.keys())
    
    coverage = len(outputs_by_unit) / len(sample_ids) if sample_ids else 0
    
    errors = []
    warnings = []
    
    if coverage < 0.90:
        errors.append(f"FAIL: Only {coverage:.1%} judge coverage")
    elif coverage < 0.95:
        warnings.append(f"WARN: {coverage:.1%} judge coverage (target 95%)")
    
    # Check all models ran
    models_seen = set()
    for models in outputs_by_unit.values():
        models_seen.update(models)
    
    missing_models = set(expected_models) - models_seen
    if missing_models:
        errors.append(f"FAIL: Missing judge models: {missing_models}")
    
    return GateResult(
        passed=len(errors)==0, 
        errors=errors, 
        warnings=warnings,
        details={"missing_units": list(missing_units)[:20]}
    )
```

### Gate 7.2: Judge Response Validity Gate
- **When**: After judge phase completes
- **Checks**:
  - Judge outputs parse correctly
  - Required fields present
  - Values in valid ranges
- **Thresholds**:
  - FAIL: parse_error_rate > 10%
  - WARN: parse_error_rate > 2%
- **Implementation**:
```python
def gate_judge_response_validity(judge_outputs):
    """Check judge response quality."""
    parse_errors = 0
    range_errors = 0
    
    for jo in judge_outputs:
        # Check parsing
        if jo.parse_error or jo.raw_response is None:
            parse_errors += 1
            continue
        
        # Check required fields and ranges
        if jo.detectability:
            if not isinstance(jo.detectability.error_detected, bool):
                range_errors += 1
            if jo.detectability.error_confidence is not None:
                if not (0 <= jo.detectability.error_confidence <= 1):
                    range_errors += 1
        
        if jo.impact_prediction:
            if jo.impact_prediction.predicted_impact_score is not None:
                if not (0 <= jo.impact_prediction.predicted_impact_score <= 1):
                    range_errors += 1
    
    total = len(judge_outputs)
    parse_rate = parse_errors / total if total > 0 else 0
    range_rate = range_errors / total if total > 0 else 0
    
    errors = []
    warnings = []
    
    if parse_rate > 0.10:
        errors.append(f"FAIL: {parse_rate:.1%} judge responses had parse errors")
    elif parse_rate > 0.02:
        warnings.append(f"WARN: {parse_rate:.1%} judge responses had parse errors")
    
    if range_rate > 0.05:
        warnings.append(f"WARN: {range_rate:.1%} judge responses had out-of-range values")
    
    return GateResult(passed=len(errors)==0, errors=errors, warnings=warnings)
```

---

## Gate 8: POST-COMPUTE (compute -> analysis)

### Gate 8.1: Outcome Degradation Variance Gate
- **When**: After compute phase completes
- **Checks**:
  - Outcome degradation values have variance
  - Not all same value (would prevent correlation)
- **Thresholds**:
  - FAIL: od_variance == 0 (all same value)
  - FAIL: unique_od_values < 3
  - WARN: od_variance < 0.01
- **Implementation**:
```python
def gate_outcome_degradation_variance(outcome_records):
    """Check outcome degradation has sufficient variance."""
    od_values = [
        or_.metrics.outcome_degradation 
        for or_ in outcome_records 
        if or_.metrics and or_.metrics.outcome_degradation is not None
    ]
    
    if not od_values:
        return GateResult(
            passed=False,
            errors=["FAIL: No outcome degradation values computed"],
            warnings=[]
        )
    
    unique_values = set(od_values)
    variance = np.var(od_values) if len(od_values) > 1 else 0
    
    errors = []
    warnings = []
    
    if variance == 0:
        errors.append(f"FAIL: Outcome degradation has zero variance (all values = {od_values[0]})")
        errors.append("  Cannot compute correlation metrics!")
    elif len(unique_values) < 3:
        errors.append(f"FAIL: Only {len(unique_values)} unique OD values")
    elif variance < 0.01:
        warnings.append(f"WARN: Low OD variance ({variance:.4f})")
    
    # Check for suspicious patterns
    zero_count = sum(1 for v in od_values if v == 0)
    zero_rate = zero_count / len(od_values)
    if zero_rate > 0.8:
        warnings.append(f"WARN: {zero_rate:.1%} of OD values are 0 - check grading")
    
    return GateResult(
        passed=len(errors)==0, 
        errors=errors, 
        warnings=warnings,
        details={
            "variance": variance,
            "unique_values": len(unique_values),
            "mean": np.mean(od_values),
            "min": min(od_values),
            "max": max(od_values)
        }
    )
```

### Gate 8.2: Baseline vs Perturbed Outcome Gate
- **When**: After compute phase completes
- **Checks**:
  - Baseline outcomes differ from perturbed (not all tied)
  - Perturbations actually affected outcomes
- **Thresholds**:
  - FAIL: degradation_rate < 10% (perturbations had no effect)
  - WARN: degradation_rate < 30%
  - WARN: improvement_rate > 10% (perturbed better than baseline)
- **Implementation**:
```python
def gate_baseline_perturbed_outcomes(outcome_records):
    """Check perturbations had measurable effect on outcomes."""
    degradation = 0  # baseline > perturbed
    improvement = 0  # perturbed > baseline
    tie = 0  # same score
    
    for or_ in outcome_records:
        if or_.metrics and or_.metrics.outcome_degradation is not None:
            od = or_.metrics.outcome_degradation
            if od > 0:
                degradation += 1
            elif od < 0:
                improvement += 1
            else:
                tie += 1
    
    total = degradation + improvement + tie
    if total == 0:
        return GateResult(passed=False, errors=["FAIL: No outcome comparisons"], warnings=[])
    
    deg_rate = degradation / total
    imp_rate = improvement / total
    tie_rate = tie / total
    
    errors = []
    warnings = []
    
    if deg_rate < 0.10:
        errors.append(f"FAIL: Only {deg_rate:.1%} of perturbations caused degradation")
        errors.append("  Perturbations may not be having expected effect!")
    elif deg_rate < 0.30:
        warnings.append(f"WARN: Only {deg_rate:.1%} degradation rate (expected >30%)")
    
    if imp_rate > 0.10:
        warnings.append(f"WARN: {imp_rate:.1%} improvements (perturbed > baseline) - investigate!")
    
    if tie_rate > 0.50:
        warnings.append(f"WARN: {tie_rate:.1%} ties - grading may lack sensitivity")
    
    return GateResult(
        passed=len(errors)==0, 
        errors=errors, 
        warnings=warnings,
        details={
            "degradation_rate": deg_rate,
            "improvement_rate": imp_rate,
            "tie_rate": tie_rate
        }
    )
```

---

## Gate 9: PRE-ANALYSIS (before analysis phase)

### Gate 9.1: Data Completeness Gate
- **When**: Before analysis phase starts
- **Checks**:
  - All required data sources present
  - Sufficient overlap between sources
- **Thresholds**:
  - FAIL: any required collection empty
  - FAIL: join coverage < 80%
  - WARN: join coverage < 90%
- **Implementation**:
```python
def gate_data_completeness(storage, experiment_id):
    """Check all required data available for analysis."""
    # Count documents in each collection
    counts = {
        "evaluation_units": storage.db["evaluation_units"].count_documents(
            {"experiment_id": experiment_id}
        ),
        "judge_outputs": storage.db["judge_eval_outputs"].count_documents(
            {"experiment_id": experiment_id}
        ),
        "outcome_evidence": storage.db["outcome_evidence"].count_documents(
            {"experiment_id": experiment_id}
        ),
        "human_labels": storage.db["aggregated_human_labels"].count_documents(
            {"experiment_id": experiment_id}
        )
    }
    
    errors = []
    warnings = []
    
    # Check required collections have data
    required = ["evaluation_units", "judge_outputs"]
    for coll in required:
        if counts[coll] == 0:
            errors.append(f"FAIL: {coll} collection is empty")
    
    # At least one of outcome_evidence or human_labels needed
    if counts["outcome_evidence"] == 0 and counts["human_labels"] == 0:
        errors.append("FAIL: Neither outcome_evidence nor human_labels available")
    
    # Check join coverage
    eval_unit_ids = set(
        doc["evaluation_unit_id"] 
        for doc in storage.db["evaluation_units"].find(
            {"experiment_id": experiment_id},
            {"evaluation_unit_id": 1}
        )
    )
    
    judge_unit_ids = set(
        doc["evaluation_unit_id"]
        for doc in storage.db["judge_eval_outputs"].find(
            {"experiment_id": experiment_id},
            {"evaluation_unit_id": 1}
        )
    )
    
    overlap = eval_unit_ids & judge_unit_ids
    coverage = len(overlap) / len(eval_unit_ids) if eval_unit_ids else 0
    
    if coverage < 0.80:
        errors.append(f"FAIL: Only {coverage:.1%} units have judge outputs")
    elif coverage < 0.90:
        warnings.append(f"WARN: {coverage:.1%} units have judge outputs")
    
    return GateResult(
        passed=len(errors)==0, 
        errors=errors, 
        warnings=warnings,
        details={"collection_counts": counts, "join_coverage": coverage}
    )
```

### Gate 9.2: Metric Computability Gate
- **When**: Before analysis phase starts
- **Checks**:
  - Sufficient data for each metric
  - Required fields present for correlation
- **Thresholds**:
  - FAIL: cannot compute any correlation metric
  - WARN: sample size < 50 for any metric
- **Implementation**:
```python
def gate_metric_computability(judge_outputs, outcome_records, human_labels):
    """Check metrics can be computed with available data."""
    errors = []
    warnings = []
    
    # Check detection metrics feasibility
    judge_with_detection = sum(
        1 for jo in judge_outputs 
        if jo.detectability and jo.detectability.error_detected is not None
    )
    human_with_detection = sum(
        1 for hl in human_labels
        if hl.aggregated_detectability and hl.aggregated_detectability.error_detected_majority is not None
    )
    
    if judge_with_detection == 0:
        errors.append("FAIL: No judge detection predictions available")
    if human_with_detection == 0:
        warnings.append("WARN: No human detection labels - cannot compute detection accuracy")
    
    # Check calibration metrics feasibility
    od_count = sum(
        1 for or_ in outcome_records
        if or_.metrics and or_.metrics.outcome_degradation is not None
    )
    judge_impact_count = sum(
        1 for jo in judge_outputs
        if jo.impact_prediction and jo.impact_prediction.predicted_impact_score is not None
    )
    
    if od_count < 30:
        errors.append(f"FAIL: Only {od_count} outcome records - need >= 30 for correlation")
    if judge_impact_count < 30:
        errors.append(f"FAIL: Only {judge_impact_count} judge impact predictions")
    
    # Check overlap for correlation
    od_unit_ids = {or_.evaluation_unit_id for or_ in outcome_records if or_.metrics}
    judge_unit_ids = {jo.evaluation_unit_id for jo in judge_outputs if jo.impact_prediction}
    overlap = od_unit_ids & judge_unit_ids
    
    if len(overlap) < 30:
        errors.append(f"FAIL: Only {len(overlap)} units with both OD and judge impact")
    elif len(overlap) < 50:
        warnings.append(f"WARN: Only {len(overlap)} units for correlation (target 50+)")
    
    return GateResult(
        passed=len(errors)==0, 
        errors=errors, 
        warnings=warnings,
        details={
            "judge_detection_count": judge_with_detection,
            "human_detection_count": human_with_detection,
            "od_count": od_count,
            "correlation_sample_size": len(overlap)
        }
    )
```

---

## Implementation Approach

### 1. Gate Registry Module
Create `src/gates/__init__.py`:
```python
"""
Quality gates for pipeline stage transitions.

Each gate returns a GateResult with:
- passed: bool
- errors: List[str] (FAILs)
- warnings: List[str] (WARNs)
- details: Dict[str, Any] (optional diagnostic info)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class GateResult:
    """Result of a quality gate check."""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Optional[Dict[str, Any]] = None

# Gate registry
GATES = {
    "post_load": [
        "trajectory_count",
        "success_criteria",
    ],
    "post_typing": [
        "typing_coverage",
        "step_role_distribution",
    ],
    "post_perturb": [
        "artifact_contamination",
        "position_distribution",
        "qc_pass_rate",
        "class_distribution",
    ],
    "post_evaluation_unit": [
        "blinding_balance",
        "unit_validation",
    ],
    "post_sample": [
        "sample_stratification",
        "baseline_success_rate",
    ],
    "post_annotate": [
        "annotation_completeness",
        "annotation_quality",
        "taxonomy_alignment",
    ],
    "post_judge": [
        "judge_coverage",
        "judge_response_validity",
    ],
    "post_compute": [
        "outcome_degradation_variance",
        "baseline_perturbed_outcomes",
    ],
    "pre_analysis": [
        "data_completeness",
        "metric_computability",
    ],
}
```

### 2. Integration with ExperimentRunner

Modify `src/experiment_runner.py` to run gates after each phase:

```python
def run(self):
    """Run the experiment with quality gates."""
    # ... existing code ...
    
    for idx, phase in enumerate(self.phases_to_run, 1):
        # Run phase
        handler = phase_handlers[phase]
        handler()
        
        # Run post-phase gates
        gate_stage = f"post_{phase}"
        if gate_stage in GATES:
            print(f"\n  Running quality gates for {gate_stage}...")
            gate_results = self._run_gates(gate_stage)
            
            # Check for failures
            failures = [r for r in gate_results if not r.passed]
            if failures and not self.config.get("ignore_gate_failures"):
                print(f"\n  GATE FAILURES - stopping pipeline")
                for result in failures:
                    for err in result.errors:
                        print(f"    {err}")
                raise GateFailure(f"{len(failures)} gates failed at {gate_stage}")
            
            # Print warnings
            for result in gate_results:
                for warn in result.warnings:
                    print(f"    {warn}")
```

### 3. Configuration

Add gate configuration to experiment config:
```json
{
  "gates": {
    "enabled": true,
    "fail_fast": true,
    "ignore_warnings": false,
    "thresholds": {
      "post_perturb.artifact_contamination": {
        "fail": 0.05,
        "warn": 0.01
      },
      "post_perturb.position_distribution.last_step_rate": {
        "fail": 0.40,
        "warn": 0.25
      }
    }
  }
}
```

---

## Summary Table

| Gate | Stage Transition | Key Check | FAIL Threshold | WARN Threshold |
|------|------------------|-----------|----------------|----------------|
| trajectory_count | load->typing | Min trajectories | <50 total | <100 total |
| success_criteria | load->typing | Success field valid | 0% or 100% | <20% or >80% |
| typing_coverage | typing->perturb | All typed | <95% | - |
| artifact_contamination | perturb->eval_unit | No leaked markers | >5% | >0% |
| position_distribution | perturb->eval_unit | Not all last step | >40% last | >25% last |
| qc_pass_rate | perturb->eval_unit | QC passes | <80% | <90% |
| blinding_balance | eval_unit->sample | A/B balanced | <40% or >60% | <45% or >55% |
| sample_stratification | sample->annotate | All conditions covered | Any condition=0 | <5% expected |
| baseline_success_rate | sample->annotate | Mixed success/fail | 0% or 100% | <30% or >90% |
| annotation_completeness | annotate->judge | All annotated | <90% | <95% |
| annotation_quality | annotate->judge | QC checks pass | too_fast>20%, gold<70% | consecutive_same |
| taxonomy_alignment | annotate->judge | Valid error types | >10% unknown | Any unknown |
| judge_coverage | judge->compute | All units judged | <90% | <95% |
| od_variance | compute->analysis | OD has variance | variance=0 | variance<0.01 |
| data_completeness | pre-analysis | All data present | Empty collections | <90% coverage |
| metric_computability | pre-analysis | Can compute metrics | <30 samples | <50 samples |

This design provides early detection of the specific issues encountered while remaining practical to implement.
