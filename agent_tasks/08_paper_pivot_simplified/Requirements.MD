# Task 08: Paper Pivot - LLM-as-Judge for Agent Trajectories

## Overview

Revised study design focusing on **Detectability vs Consequentiality**: Do LLM judges detect errors better than they estimate their downstream consequence?

## Study Flow

```
1. Trajectory Sources → 2. Typed Representation → 3. Controlled Perturbations → 4. Evaluation Unit
                                                                                       ↓
                        6C. Agreement + Claim ← 6B. Consequentiality ← 6A. Detection Metrics
                                                         ↑
                              5A. Human Labels    5B. Judge Outputs    5C. Outcome Evidence
```

---

## Section Index

| # | Section | Status | File |
|---|---------|--------|------|
| 1 | Trajectory Sources | COMPLETED | [1_Requirements_TrajectorySources.MD](1_Requirements_TrajectorySources.MD) |
| 2 | Typed Representation | REQUIREMENTS DEFINED | [2_Requirements_TypedRepresentation.MD](2_Requirements_TypedRepresentation.MD) |
| 3 | Controlled Perturbations | REQUIREMENTS DEFINED | [3_Requirements_ControlledPerturbations.MD](3_Requirements_ControlledPerturbations.MD) |
| 4 | Evaluation Unit | REQUIREMENTS DEFINED | [4_Requirements_EvaluationUnit.MD](4_Requirements_EvaluationUnit.MD) |
| 5A | Human Labels | REQUIREMENTS DEFINED | [5A_Requirements_HumanLabels.MD](5A_Requirements_HumanLabels.MD) |
| 5B | LLM Judge Outputs | REQUIREMENTS DEFINED | [5B_Requirements_LLMJudgeOutputs.MD](5B_Requirements_LLMJudgeOutputs.MD) |
| 5C | Outcome Evidence | REQUIREMENTS DEFINED | [5C_Requirements_OutcomeEvidence.MD](5C_Requirements_OutcomeEvidence.MD) |
| 6A | Detection Metrics | REQUIREMENTS DEFINED | [6A_Requirements_DetectionMetrics.MD](6A_Requirements_DetectionMetrics.MD) |
| 6B | Consequentiality + Calibration | REQUIREMENTS DEFINED | [6B_Requirements_Consequentiality.MD](6B_Requirements_Consequentiality.MD) |
| 6C | Agreement + Claim | REQUIREMENTS DEFINED | [6C_Requirements_AgreementClaim.MD](6C_Requirements_AgreementClaim.MD) |

---

## [1. Trajectory Sources](1_Requirements_TrajectorySources.MD)

**Status: COMPLETED** — 600 trajectories sampled (ToolBench 400, GAIA 100, SWE-bench 100)

---

## [2. Typed Representation](2_Requirements_TypedRepresentation.MD)

**Status: REQUIREMENTS DEFINED** — Schema for step roles, dependencies, perturbable slots

---

## [3. Controlled Perturbations](3_Requirements_ControlledPerturbations.MD)

**Status: REQUIREMENTS DEFINED** — Placebo, fine-grained, coarse-grained perturbation taxonomy, ~1700 total perturbations, validation pipeline

---

## [4. Evaluation Unit](4_Requirements_EvaluationUnit.MD)

**Status: REQUIREMENTS DEFINED** — Three-tier ID hierarchy, canonical/derived data separation, blinding protocol, human/judge/outcome views

---

## [5A. Human Labels](5A_Requirements_HumanLabels.MD)

**Status: REQUIREMENTS DEFINED** — Three annotation modes (detectability, consequence, preference), blinded evaluation, inter-annotator agreement (κ/α), aggregation pipeline, 350 units target

---

## [5B. LLM Judge Outputs](5B_Requirements_LLMJudgeOutputs.MD)

**Status: REQUIREMENTS DEFINED** — 4 judge models, 3 evaluation modes, structured output schema, prompt templates, variance estimation (n=5), ~12,800 total runs

---

## [5C. Outcome Evidence](5C_Requirements_OutcomeEvidence.MD)

**Status: REQUIREMENTS DEFINED** — 3 replay tiers (full replay, regeneration, grading), OD/PD/RC metrics, benchmark-specific verifiers, true impact derivation

---

## [6A. Detection Metrics](6A_Requirements_DetectionMetrics.MD)

**Status: REQUIREMENTS DEFINED** — PDR, PNDR, SLA, SLA±1, TIA, CER, ROC-AUC, stratified analysis by class/family/detectability/impact/benchmark

---

## [6B. Consequentiality + Calibration](6B_Requirements_Consequentiality.MD)

**Status: REQUIREMENTS DEFINED** — CCorr, CCE, ORR, URR, Failure-ECE, human-judge impact agreement, reliability diagrams

---

## [6C. Agreement + Claim](6C_Requirements_AgreementClaim.MD)

**Status: REQUIREMENTS DEFINED** — Human inter-annotator agreement, judge variance, inter-judge agreement, main claim hypothesis tests (PDR > CCorr), failure mode analysis, trustworthiness guidelines

---

## References

- Study diagram: [llmaj_agent_trajectory_study_diagram.svg](llmaj_agent_trajectory_study_diagram.svg)
