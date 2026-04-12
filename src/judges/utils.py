"""
Judge utilities module.

Consolidated utilities for LLM judge evaluation:
- Judge base class and implementations (ClaudeJudge)
- Response parsing
- Aggregation functions
- Storage functions
- Unit runner
"""

import json
import re
import time
import statistics
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

from src.judges.schema import (
    JudgeOutput,
    StepError,
    ErrorSeverity,
    Section5JudgeOutput,
    JudgeConfig,
    InputView,
    DetectionOutput,
    LocalizationOutput,
    ImpactOutput,
    PairComparisonOutput,
    ParseStatus,
    ErrorType,
    AggregatedJudgeOutput,
    AggregatedDetection,
    AggregatedLocalization,
    AggregatedImpact,
)
from src.judges.prompts import (
    JUDGE_SYSTEM_PROMPT,
    build_evaluation_prompt,
    build_unit_prompt,
    build_view_for_single_trajectory,
)

# =============================================================================
# Judge Base Class
# =============================================================================


class Judge(ABC):
    """
    Abstract base class for LLM judges.

    Subclasses must implement:
    - _call_llm(): Make the actual API call
    - _parse_response(): Parse LLM response into JudgeOutput
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 60,
    ):
        self.name = name
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Stats tracking
        self.total_calls = 0
        self.total_tokens = 0
        self.total_time_ms = 0
        self.failed_calls = 0

    @abstractmethod
    def _call_llm(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Make the actual LLM API call."""
        pass

    @abstractmethod
    def _parse_response(self, response_text: str, trajectory_id: str) -> JudgeOutput:
        """Parse LLM response into structured JudgeOutput."""
        pass

    def evaluate(
        self, trajectory, retry_on_failure: bool = True, max_retries: int = 3
    ) -> Optional[JudgeOutput]:
        """Evaluate a trajectory and return structured output."""
        system_prompt = JUDGE_SYSTEM_PROMPT
        user_prompt = build_evaluation_prompt(trajectory)

        for attempt in range(max_retries):
            try:
                llm_result = self._call_llm(system_prompt, user_prompt)
                output = self._parse_response(
                    llm_result["response"], trajectory.trajectory_id
                )
                output.evaluation_time_ms = llm_result.get("time_ms")
                output.tokens_used = llm_result.get("tokens_used")

                self.total_calls += 1
                self.total_tokens += output.tokens_used or 0
                self.total_time_ms += output.evaluation_time_ms or 0

                return output

            except Exception:
                self.failed_calls += 1
                if attempt < max_retries - 1 and retry_on_failure:
                    wait_time = 2**attempt
                    time.sleep(wait_time)
                else:
                    return None

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this judge."""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "success_rate": (
                (self.total_calls - self.failed_calls) / self.total_calls
                if self.total_calls > 0
                else 0
            ),
            "total_tokens": self.total_tokens,
            "total_time_ms": self.total_time_ms,
            "avg_time_per_call_ms": (
                self.total_time_ms / self.total_calls if self.total_calls > 0 else 0
            ),
        }


# =============================================================================
# Claude Judge Implementation
# =============================================================================


class ClaudeJudge(Judge):
    """Judge implementation using Claude via AWS Bedrock."""

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 60,
        region_name: str = "us-east-1",
    ):
        super().__init__(
            name="claude-sonnet-4.5",
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self.bedrock = boto3.client(
            service_name="bedrock-runtime", region_name=region_name
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        start_time = time.time()

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())
            content = response_body.get("content", [])
            if not content:
                raise ValueError("Empty response from Claude")

            response_text = content[0].get("text", "")
            usage = response_body.get("usage", {})
            tokens_used = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            elapsed_ms = int((time.time() - start_time) * 1000)

            return {
                "response": response_text,
                "tokens_used": tokens_used,
                "time_ms": elapsed_ms,
            }

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            raise Exception(f"Bedrock API error ({error_code}): {error_message}")

    def _parse_response(self, response_text: str, trajectory_id: str) -> JudgeOutput:
        return parse_json_response(
            response_text=response_text,
            trajectory_id=trajectory_id,
            judge_name=self.name,
            model_id=self.model_id,
        )


def create_claude_judge(config: Dict[str, Any]) -> ClaudeJudge:
    """Factory function to create Claude judge from config."""
    model_id = config.get("model_id")
    judge_config = config.get("config", {})
    region_name = config.get("region_name", "us-east-1")

    return ClaudeJudge(
        model_id=model_id,
        temperature=judge_config.get("temperature", 0.7),
        max_tokens=judge_config.get("max_tokens", 2000),
        timeout=judge_config.get("timeout", 60),
        region_name=region_name,
    )


# =============================================================================
# Response Parsing
# =============================================================================


def parse_json_response(
    response_text: str, trajectory_id: str, judge_name: str, model_id: str
) -> JudgeOutput:
    """Parse a JSON-formatted judge response into JudgeOutput."""
    try:
        json_str = response_text.strip()

        if "```json" in json_str:
            start = json_str.index("```json") + 7
            end = json_str.index("```", start)
            json_str = json_str[start:end].strip()
        elif "```" in json_str:
            start = json_str.index("```") + 3
            end = json_str.index("```", start)
            json_str = json_str[start:end].strip()

        data = json.loads(json_str)

    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Failed to parse JSON from response: {e}")

    step_errors = []
    for err in data.get("step_errors", []):
        try:
            step_errors.append(
                StepError(
                    step_index=err["step_index"],
                    error_type=err["error_type"],
                    severity=ErrorSeverity(err["severity"]),
                    description=err["description"],
                    impacts_task_success=err.get("impacts_task_success", False),
                )
            )
        except (KeyError, ValueError):
            continue

    return JudgeOutput(
        trajectory_id=trajectory_id,
        judge_name=judge_name,
        model_id=model_id,
        task_success=int(data["task_success"]),
        completeness=float(data["completeness"]),
        efficiency_errors=int(data.get("efficiency_errors", 0)),
        hallucination=int(data.get("hallucination", 0)),
        sycophancy=int(data.get("sycophancy", 0)),
        overall_score=float(data["overall_score"]),
        step_errors=step_errors,
        reasoning=data.get("reasoning", ""),
        timestamp=datetime.utcnow(),
    )


def parse_judge_response(
    raw_text: str,
    mode: str,
    evaluation_unit_id: str,
    judge_model: str,
    config: JudgeConfig,
    input_view: InputView,
    judge_output_id: str,
) -> Section5JudgeOutput:
    """Parse a judge response into structured output for evaluation units."""
    parse_errors = []
    parse_status = ParseStatus.SUCCESS.value

    try:
        parsed = _extract_json(raw_text)
    except ValueError as e:
        parse_errors.append(f"JSON extraction failed: {e}")
        parse_status = ParseStatus.FAILED.value
        parsed = {}

    detection = None
    try:
        detection = _parse_detection(parsed, mode)
    except Exception as e:
        parse_errors.append(f"Detection parsing failed: {e}")
        if parse_status != ParseStatus.FAILED.value:
            parse_status = ParseStatus.PARTIAL.value

    localization = None
    if detection and detection.error_detected:
        try:
            localization = _parse_localization(parsed)
        except Exception as e:
            parse_errors.append(f"Localization parsing failed: {e}")
            if parse_status != ParseStatus.FAILED.value:
                parse_status = ParseStatus.PARTIAL.value

    impact = None
    try:
        impact = _parse_impact(parsed, mode)
    except Exception as e:
        parse_errors.append(f"Impact parsing failed: {e}")
        if parse_status != ParseStatus.FAILED.value:
            parse_status = ParseStatus.PARTIAL.value

    pair_comparison = None
    if mode in ["blinded_pair", "labeled_pair"]:
        try:
            pair_comparison = _parse_pair_comparison(parsed)
        except Exception as e:
            parse_errors.append(f"Pair comparison parsing failed: {e}")
            if parse_status != ParseStatus.FAILED.value:
                parse_status = ParseStatus.PARTIAL.value

    return Section5JudgeOutput(
        judge_output_id=judge_output_id,
        evaluation_unit_id=evaluation_unit_id,
        judge_model=judge_model,
        judge_mode=mode,
        config=config,
        input_view=input_view,
        detection=detection,
        localization=localization,
        impact=impact,
        pair_comparison=pair_comparison,
        raw_response=raw_text,
        parse_status=parse_status,
        parse_errors=parse_errors if parse_errors else None,
    )


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from raw text, handling markdown code blocks."""
    text = text.strip()

    if "```json" in text:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    elif "```" in text:
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")


def _parse_detection(parsed: Dict[str, Any], mode: str) -> DetectionOutput:
    """Parse detection fields from parsed JSON."""
    if mode == "blinded_pair":
        score_a = float(parsed.get("overall_score_a", 0))
        score_b = float(parsed.get("overall_score_b", 0))
        overall_score = (score_a + score_b) / 2
    else:
        overall_score = float(parsed.get("overall_score", 0))

    overall_score = max(0, min(100, overall_score))

    return DetectionOutput(
        overall_score=overall_score,
        error_detected=bool(parsed.get("error_detected", False)),
        error_confidence=_clamp(float(parsed.get("error_confidence", 0)), 0, 1),
    )


def _parse_localization(parsed: Dict[str, Any]) -> LocalizationOutput:
    """Parse localization fields from parsed JSON."""
    predicted_step = parsed.get("predicted_error_step")
    if predicted_step is not None:
        predicted_step = int(predicted_step)

    error_type = parsed.get("predicted_error_type")
    if error_type:
        valid_types = [e.value for e in ErrorType]
        if error_type not in valid_types:
            error_type = "other"

    localization_conf = parsed.get("localization_confidence")
    if localization_conf is not None:
        localization_conf = _clamp(float(localization_conf), 0, 1)

    return LocalizationOutput(
        predicted_error_step=predicted_step,
        predicted_error_step_canonical=None,
        localization_confidence=localization_conf,
        predicted_error_type=error_type,
    )


def _parse_impact(parsed: Dict[str, Any], mode: str) -> ImpactOutput:
    """Parse impact fields from parsed JSON."""
    if mode == "labeled_pair":
        impact_estimate = float(parsed.get("impact_estimate", 0))
        impact_score = impact_estimate / 3.0
    else:
        impact_score = float(parsed.get("predicted_impact_score", 0))

    return ImpactOutput(
        predicted_impact_score=_clamp(impact_score, 0, 1),
        predicted_failure_prob=_clamp(
            float(parsed.get("predicted_failure_prob", 0)), 0, 1
        ),
        impact_explanation=parsed.get("impact_explanation")
        or parsed.get("explanation"),
    )


def _parse_pair_comparison(parsed: Dict[str, Any]) -> PairComparisonOutput:
    """Parse pair comparison fields from parsed JSON."""
    error_traj = parsed.get("error_trajectory")
    if error_traj:
        error_traj = error_traj.upper() if error_traj in ["a", "b"] else error_traj
        if error_traj not in ["A", "B", "neither", "both"]:
            error_traj = None

    preference = parsed.get("preference")
    if preference:
        preference = preference.upper() if preference in ["a", "b"] else preference
        if preference not in ["A", "B", "tie"]:
            preference = None

    return PairComparisonOutput(
        error_trajectory=error_traj,
        preference=preference,
        comparison_explanation=parsed.get("comparison_explanation")
        or parsed.get("explanation"),
    )


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value to a range."""
    return max(min_val, min(max_val, value))


def map_step_to_canonical(
    display_idx: int, trajectory: Dict[str, Any]
) -> Optional[str]:
    """Map a display step index to a canonical step ID."""
    steps = trajectory.get("steps", [])

    for step in steps:
        step_display_idx = step.get("display_step_index")
        if step_display_idx is not None and step_display_idx == display_idx:
            return step.get("canonical_step_id") or step.get("step_id")

    if 1 <= display_idx <= len(steps):
        step = steps[display_idx - 1]
        return step.get("canonical_step_id") or step.get("step_id")

    return None


def validate_judge_output(
    output: Section5JudgeOutput, mode: str
) -> Tuple[bool, List[str]]:
    """Validate a judge output for required fields."""
    errors = []

    if not output.judge_output_id:
        errors.append("Missing judge_output_id")
    if not output.evaluation_unit_id:
        errors.append("Missing evaluation_unit_id")
    if not output.judge_model:
        errors.append("Missing judge_model")
    if output.judge_mode != mode:
        errors.append(f"Mode mismatch: expected {mode}, got {output.judge_mode}")

    if not output.detection:
        errors.append("Missing detection output")
    else:
        if output.detection.overall_score < 0 or output.detection.overall_score > 100:
            errors.append("Detection overall_score out of range [0-100]")
        if (
            output.detection.error_confidence < 0
            or output.detection.error_confidence > 1
        ):
            errors.append("Detection error_confidence out of range [0-1]")

    if output.detection and output.detection.error_detected:
        if not output.localization:
            errors.append("Missing localization output when error_detected=True")

    if not output.impact:
        errors.append("Missing impact output")
    else:
        if (
            output.impact.predicted_impact_score < 0
            or output.impact.predicted_impact_score > 1
        ):
            errors.append("Impact predicted_impact_score out of range [0-1]")
        if (
            output.impact.predicted_failure_prob < 0
            or output.impact.predicted_failure_prob > 1
        ):
            errors.append("Impact predicted_failure_prob out of range [0-1]")

    if mode in ["blinded_pair", "labeled_pair"]:
        if not output.pair_comparison:
            errors.append(f"Missing pair_comparison output for mode {mode}")

    return len(errors) == 0, errors


# =============================================================================
# Aggregation Functions
# =============================================================================


def aggregate_across_samples(
    outputs: List[Section5JudgeOutput],
) -> AggregatedJudgeOutput:
    """Aggregate judge outputs across multiple samples for variance estimation."""
    if not outputs:
        raise ValueError("Cannot aggregate empty outputs list")

    first = outputs[0]
    for output in outputs[1:]:
        if output.evaluation_unit_id != first.evaluation_unit_id:
            raise ValueError("All outputs must be from same evaluation unit")
        if output.judge_model != first.judge_model:
            raise ValueError("All outputs must be from same judge model")

    return AggregatedJudgeOutput(
        evaluation_unit_id=first.evaluation_unit_id,
        judge_model=first.judge_model,
        aggregated_detection=_aggregate_detection(outputs),
        aggregated_localization=_aggregate_localization(outputs),
        aggregated_impact=_aggregate_impact(outputs),
    )


def _aggregate_detection(outputs: List[Section5JudgeOutput]) -> AggregatedDetection:
    """Aggregate detection metrics across samples."""
    overall_scores = []
    error_detected_values = []
    error_confidences = []

    for output in outputs:
        if output.detection:
            overall_scores.append(output.detection.overall_score)
            error_detected_values.append(1 if output.detection.error_detected else 0)
            error_confidences.append(output.detection.error_confidence)

    n = len(overall_scores)
    if n == 0:
        return AggregatedDetection(
            mean_overall_score=0.0,
            std_overall_score=0.0,
            error_detected_rate=0.0,
            mean_error_confidence=0.0,
            std_error_confidence=0.0,
        )

    return AggregatedDetection(
        mean_overall_score=statistics.mean(overall_scores),
        std_overall_score=statistics.stdev(overall_scores) if n > 1 else 0.0,
        error_detected_rate=sum(error_detected_values) / n,
        mean_error_confidence=statistics.mean(error_confidences),
        std_error_confidence=statistics.stdev(error_confidences) if n > 1 else 0.0,
    )


def _aggregate_localization(
    outputs: List[Section5JudgeOutput],
) -> AggregatedLocalization:
    """Aggregate localization metrics across samples."""
    predicted_steps = []
    predicted_types = []

    for output in outputs:
        if output.detection and output.detection.error_detected and output.localization:
            if output.localization.predicted_error_step_canonical:
                predicted_steps.append(
                    output.localization.predicted_error_step_canonical
                )
            if output.localization.predicted_error_type:
                predicted_types.append(output.localization.predicted_error_type)

    modal_step = None
    step_agreement = 0.0
    if predicted_steps:
        step_counter = Counter(predicted_steps)
        modal_step, modal_count = step_counter.most_common(1)[0]
        step_agreement = modal_count / len(predicted_steps)

    modal_type = None
    type_agreement = 0.0
    if predicted_types:
        type_counter = Counter(predicted_types)
        modal_type, modal_count = type_counter.most_common(1)[0]
        type_agreement = modal_count / len(predicted_types)

    return AggregatedLocalization(
        modal_predicted_step=modal_step,
        step_agreement_rate=step_agreement,
        modal_predicted_type=modal_type,
        type_agreement_rate=type_agreement,
    )


def _aggregate_impact(outputs: List[Section5JudgeOutput]) -> AggregatedImpact:
    """Aggregate impact metrics across samples."""
    impact_scores = []
    failure_probs = []

    for output in outputs:
        if output.impact:
            impact_scores.append(output.impact.predicted_impact_score)
            failure_probs.append(output.impact.predicted_failure_prob)

    n = len(impact_scores)
    if n == 0:
        return AggregatedImpact(
            mean_impact_score=0.0,
            std_impact_score=0.0,
            mean_failure_prob=0.0,
            std_failure_prob=0.0,
        )

    return AggregatedImpact(
        mean_impact_score=statistics.mean(impact_scores),
        std_impact_score=statistics.stdev(impact_scores) if n > 1 else 0.0,
        mean_failure_prob=statistics.mean(failure_probs),
        std_failure_prob=statistics.stdev(failure_probs) if n > 1 else 0.0,
    )


def batch_aggregate_across_samples(
    outputs: List[Section5JudgeOutput], group_by_unit: bool = True
) -> List[AggregatedJudgeOutput]:
    """Batch aggregate outputs, grouping by evaluation unit."""
    if not group_by_unit:
        return [aggregate_across_samples(outputs)]

    grouped = {}
    for output in outputs:
        key = (output.evaluation_unit_id, output.judge_model)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(output)

    aggregated = []
    for key, group_outputs in grouped.items():
        try:
            agg = aggregate_across_samples(group_outputs)
            aggregated.append(agg)
        except ValueError as e:
            print(f"Warning: Could not aggregate {key}: {e}")

    return aggregated


# =============================================================================
# Storage Functions
# =============================================================================

DEFAULT_BASE_DIR = "data/judge_outputs"


def save_judge_output(
    output: Section5JudgeOutput, output_dir: str, append: bool = True
) -> str:
    """Save a judge output to JSONL file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    deterministic = output.config.temperature == 0 if output.config else True
    sampling = "deterministic" if deterministic else "stochastic"
    filename = f"outputs_{sampling}.jsonl"
    file_path = Path(output_dir) / filename

    mode = "a" if append else "w"
    with open(file_path, mode) as f:
        json.dump(output.to_dict(), f)
        f.write("\n")

    return str(file_path)


def load_judge_outputs(
    path: str, limit: Optional[int] = None
) -> List[Section5JudgeOutput]:
    """Load judge outputs from JSONL file."""
    outputs = []
    count = 0

    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                output = Section5JudgeOutput.from_dict(data)
                outputs.append(output)
                count += 1

                if limit and count >= limit:
                    break

    return outputs


def save_checkpoint(
    output_dir: str, judge_name: str, mode: str, completed_ids: List[str]
) -> str:
    """Save a checkpoint file for batch processing."""
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = checkpoint_dir / f"checkpoint_{judge_name}_{mode}.json"

    checkpoint = {
        "judge_name": judge_name,
        "mode": mode,
        "completed_ids": completed_ids,
        "updated_at": datetime.utcnow().isoformat(),
    }

    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint, f, indent=2)

    return str(checkpoint_file)


def load_checkpoint(output_dir: str, judge_name: str, mode: str) -> Dict[str, Any]:
    """Load a checkpoint file for batch processing."""
    checkpoint_file = (
        Path(output_dir) / "checkpoints" / f"checkpoint_{judge_name}_{mode}.json"
    )

    if not checkpoint_file.exists():
        return {"completed_ids": []}

    with open(checkpoint_file, "r") as f:
        return json.load(f)


def save_judge_outputs_to_mongodb(
    outputs: List[Section5JudgeOutput],
    storage,
    experiment_id: str,
) -> int:
    """Save judge outputs to MongoDB."""
    collection = storage.db["judge_eval_outputs"]

    collection.create_index("evaluation_unit_id")
    collection.create_index("experiment_id")
    collection.create_index("judge_output_id")

    saved = 0
    for output in outputs:
        doc = output.to_dict()
        doc["experiment_id"] = experiment_id

        collection.update_one(
            {"judge_output_id": output.judge_output_id},
            {"$set": doc},
            upsert=True,
        )
        saved += 1

    return saved


def load_judge_outputs_from_mongodb(
    experiment_id: str,
    storage,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Section5JudgeOutput]:
    """Load judge outputs from MongoDB."""
    collection = storage.db["judge_eval_outputs"]

    query = {"experiment_id": experiment_id}
    if filters:
        query.update(filters)

    docs = list(collection.find(query))

    outputs = []
    for doc in docs:
        doc.pop("_id", None)
        doc.pop("experiment_id", None)
        outputs.append(Section5JudgeOutput.from_dict(doc))

    return outputs


# =============================================================================
# Unit Judge Runner
# =============================================================================


class UnitJudgeRunner:
    """Runner for judge evaluations on evaluation units."""

    def __init__(self, judge: Judge, config: Optional[Dict[str, Any]] = None):
        self.judge = judge
        self.config = config or {}

        self.prompt_template_id = self.config.get(
            "prompt_template_id", "single_trajectory_v1"
        )
        self.temperature = self.config.get("temperature", 0)
        self.seed = self.config.get("seed")
        self.max_tokens = self.config.get("max_tokens", 2000)
        self.checkpoint_interval = self.config.get("checkpoint_interval", 50)
        self.rate_limit_delay = self.config.get("rate_limit_delay", 0.5)

    def run_single_trajectory(
        self,
        unit: Dict[str, Any],
        trajectory: Optional[Dict[str, Any]] = None,
        sample_index: int = 0,
    ) -> Section5JudgeOutput:
        """Evaluate a single trajectory for an evaluation unit."""
        if trajectory is None:
            trajectory = (
                unit.get("trajectory_variant")
                or unit.get("perturbed_trajectory")
                or (unit.get("perturbed", {}).get("trajectory"))
            )
            if trajectory is None:
                raise ValueError(
                    "No trajectory provided and unit has no trajectory data"
                )

        view = build_view_for_single_trajectory(unit, trajectory)
        system_prompt, user_prompt = build_unit_prompt(view, "single_trajectory")

        judge_config = JudgeConfig(
            prompt_template_id=self.prompt_template_id,
            temperature=self.temperature,
            seed=self.seed,
            max_tokens=self.max_tokens,
            sample_index=sample_index,
        )

        input_view = InputView(
            view_file=unit.get("view_file", ""),
            trajectory_variant_ids=[trajectory.get("trajectory_id", "")],
        )

        judge_output_id = f"judge_{unit.get('evaluation_unit_id', 'unknown')}_{self.judge.name}_{sample_index}"

        raw_response = self._call_judge(system_prompt, user_prompt)

        output = parse_judge_response(
            raw_text=raw_response,
            mode="single_trajectory",
            evaluation_unit_id=unit.get("evaluation_unit_id", ""),
            judge_model=self.judge.model_id,
            config=judge_config,
            input_view=input_view,
            judge_output_id=judge_output_id,
        )

        if output.localization and output.localization.predicted_error_step:
            canonical_id = map_step_to_canonical(
                output.localization.predicted_error_step, trajectory
            )
            output.localization.predicted_error_step_canonical = canonical_id

        return output

    def run_batch(
        self,
        units: List[Dict[str, Any]],
        mode: str,
        output_dir: str = None,
        resume: bool = True,
        samples_per_unit: int = 1,
        storage=None,
        experiment_id: str = None,
        parallelization: int = 1,
    ) -> List[Section5JudgeOutput]:
        """Process a batch of evaluation units with checkpointing."""
        use_mongodb = storage is not None and experiment_id is not None

        print(f"\n{'=' * 70}")
        print("BATCH JUDGE EVALUATION")
        print(f"{'=' * 70}")
        print(f"Judge: {self.judge.name}")
        print(f"Mode: {mode}")
        print(f"Units: {len(units)}")
        print(f"Samples per unit: {samples_per_unit}")
        print(f"Storage: {'MongoDB' if use_mongodb else 'Disk'}")
        print(f"{'=' * 70}\n")

        completed_ids = set()
        if resume:
            if use_mongodb:
                existing = load_judge_outputs_from_mongodb(experiment_id, storage)
                completed_ids = set(o.evaluation_unit_id for o in existing)
            else:
                checkpoint = load_checkpoint(output_dir, self.judge.name, mode)
                completed_ids = set(checkpoint.get("completed_ids", []))
            print(f"Resuming: {len(completed_ids)} units already completed")

        remaining_units = [
            u for u in units if u.get("evaluation_unit_id") not in completed_ids
        ]
        print(f"Remaining: {len(remaining_units)} units to process")

        if not remaining_units:
            print("All units already processed.")
            return []

        outputs = []
        processed_count = 0

        for i, unit in enumerate(remaining_units, 1):
            unit_id = unit.get("evaluation_unit_id", f"unit_{i}")

            for sample_idx in range(samples_per_unit):
                try:
                    if mode == "single_trajectory":
                        output = self.run_single_trajectory(
                            unit, sample_index=sample_idx
                        )
                    else:
                        raise ValueError(f"Invalid mode: {mode}")

                    is_valid, errors = validate_judge_output(output, mode)
                    if not is_valid:
                        print(f"  Warning: Validation errors for {unit_id}: {errors}")

                    if use_mongodb:
                        save_judge_outputs_to_mongodb([output], storage, experiment_id)
                    else:
                        save_judge_output(output, output_dir)
                    outputs.append(output)

                    if self.rate_limit_delay > 0:
                        time.sleep(self.rate_limit_delay)

                except Exception as e:
                    print(f"  Error processing {unit_id} sample {sample_idx}: {e}")
                    continue

            completed_ids.add(unit_id)
            processed_count += 1

            if not use_mongodb and processed_count % self.checkpoint_interval == 0:
                save_checkpoint(output_dir, self.judge.name, mode, list(completed_ids))

            if i % 10 == 0 or i == len(remaining_units):
                print(f"Progress: {i}/{len(remaining_units)} units")

        if not use_mongodb and output_dir:
            save_checkpoint(output_dir, self.judge.name, mode, list(completed_ids))

        print(f"\nBatch complete: {len(outputs)} outputs")
        return outputs

    def _call_judge(self, system_prompt: str, user_prompt: str) -> str:
        """Call the judge LLM and return raw response."""
        result = self.judge._call_llm(system_prompt, user_prompt)
        return result.get("response", "")


def create_unit_runner(
    judge_name: str, config: Optional[Dict[str, Any]] = None, **kwargs
) -> UnitJudgeRunner:
    """Factory function to create a unit runner with specified judge."""
    if judge_name in ["claude-sonnet-4.5", "claude"]:
        judge = ClaudeJudge(**kwargs)
    else:
        raise ValueError(f"Unknown judge: {judge_name}")

    return UnitJudgeRunner(judge, config)
