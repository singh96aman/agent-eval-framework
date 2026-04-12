"""
Streamlit Annotation UI for Section 5A Human Labels.

Run with:
    python main.py --config pocv2/perturbation_generation --runner annotate --ui

Or directly:
    streamlit run ops/annotation_ui.py

Features:
- Side-by-side trajectory comparison
- Diff highlighting between A and B
- Progress tracking with resume support
- MongoDB persistence after each annotation
- Keyboard shortcuts for faster annotation
"""

import difflib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Load environment variables from .env
from dotenv import load_dotenv

load_dotenv()

import streamlit as st

# Must be first Streamlit command
st.set_page_config(
    page_title="Trajectory Annotation",
    page_icon="🔍",
    layout="wide",
)

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.mongodb import MongoDBStorage
from src.human_labels.schema import (
    AnnotationRecord,
    DetectabilityLabels,
    ConsequenceLabels,
)
from src.human_labels.storage import (
    save_annotations_to_mongodb,
    get_completed_annotation_unit_ids,
)

# === Diff Highlighting ===
# Colors that work in dark mode (no red to avoid bias)
COLOR_A_ONLY = "color: #58a6ff; font-weight: bold;"  # Blue text for A-only
COLOR_B_ONLY = "color: #f0c000; font-weight: bold;"  # Yellow text for B-only
COLOR_SAME = "color: #8b949e;"  # Gray for identical content


def compute_diff_html(text_a: str, text_b: str) -> Tuple[str, str]:
    """
    Compute side-by-side diff with color highlighting.

    Returns (html_a, html_b) where:
    - Same text: gray
    - A-only (deleted from B): blue text
    - B-only (added in B): red/pink text
    """
    import html

    # Split into lines for comparison
    lines_a = text_a.split("\n")
    lines_b = text_b.split("\n")

    # Use SequenceMatcher for line-level diff
    matcher = difflib.SequenceMatcher(None, lines_a, lines_b)

    html_a_lines = []
    html_b_lines = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # Same content - gray text
            for line in lines_a[i1:i2]:
                html_a_lines.append(
                    f'<span style="{COLOR_SAME}">{html.escape(line)}</span>'
                )
            for line in lines_b[j1:j2]:
                html_b_lines.append(
                    f'<span style="{COLOR_SAME}">{html.escape(line)}</span>'
                )
        elif tag == "replace":
            # Different content - highlight both
            for line in lines_a[i1:i2]:
                html_a_lines.append(
                    f'<span style="{COLOR_A_ONLY}">{html.escape(line)}</span>'
                )
            for line in lines_b[j1:j2]:
                html_b_lines.append(
                    f'<span style="{COLOR_B_ONLY}">{html.escape(line)}</span>'
                )
        elif tag == "delete":
            # Only in A - blue
            for line in lines_a[i1:i2]:
                html_a_lines.append(
                    f'<span style="{COLOR_A_ONLY}">{html.escape(line)}</span>'
                )
        elif tag == "insert":
            # Only in B - red/pink
            for line in lines_b[j1:j2]:
                html_b_lines.append(
                    f'<span style="{COLOR_B_ONLY}">{html.escape(line)}</span>'
                )

    # Join with line breaks
    html_a = (
        '<pre style="white-space: pre-wrap; font-family: monospace; font-size: 13px; line-height: 1.4;">'
        + "\n".join(html_a_lines)
        + "</pre>"
    )
    html_b = (
        '<pre style="white-space: pre-wrap; font-family: monospace; font-size: 13px; line-height: 1.4;">'
        + "\n".join(html_b_lines)
        + "</pre>"
    )

    return html_a, html_b


def get_step_content(step: Dict[str, Any]) -> str:
    """Extract displayable content from a step, including tool_arguments."""
    content = step.get("raw_text", step.get("content", ""))
    tool = step.get("tool_name", "")
    role = step.get("step_role", "unknown")
    tool_args = step.get("tool_arguments", {})

    result = f"[{role}]"
    if tool:
        result += f" Tool: {tool}"
    result += f"\n{content}"

    # Add tool_arguments if present (this is where perturbations often occur!)
    # Show full values - no truncation - so diff can catch changes anywhere
    if tool_args:
        result += "\n\n--- Tool Arguments (parsed) ---"
        for key, val in tool_args.items():
            result += f"\n  {key}: {val}"

    return result


# === Configuration ===
CONFIG_PATH = "config/experiments/v2/pocv2/perturbation_generation.json"
EXPERIMENT_ID = "exp_trajectory_sampling_v7"
ANNOTATOR_ID = "researcher"


@st.cache_resource
def get_storage():
    """Get MongoDB storage (cached)."""
    return MongoDBStorage()


@st.cache_data
def load_evaluation_units():
    """Load evaluation units from JSON."""
    eval_units_path = f"data/evaluation_units/{EXPERIMENT_ID}_evaluation_units.json"
    with open(eval_units_path) as f:
        data = json.load(f)
    return data.get("evaluation_units", [])


@st.cache_data
def load_config():
    """Load experiment config."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_sampled_units(
    units: List[Dict], sample_size: int = 130, seed: int = 42
) -> List[Dict]:
    """Sample units with fixed seed for reproducibility."""
    import random

    random.seed(seed)
    return random.sample(units, min(sample_size, len(units)))


def get_completed_ids() -> set:
    """Get IDs of already-annotated units."""
    storage = get_storage()
    return get_completed_annotation_unit_ids(EXPERIMENT_ID, storage, ANNOTATOR_ID)


def format_step(step: Dict[str, Any], step_num: int) -> str:
    """Format a single step for display."""
    role = step.get("step_role", "unknown")
    content = step.get("raw_text", step.get("content", ""))
    tool = step.get("tool_name", "")

    # Truncate very long content but keep more than CLI
    if len(content) > 1000:
        content = content[:1000] + "... (truncated)"

    lines = [f"**Step {step_num}** [{role}]"]
    if content:
        lines.append(f"```\n{content}\n```")
    if tool:
        lines.append(f"🔧 Tool: `{tool}`")

    return "\n".join(lines)


def save_annotation(unit: Dict, responses: Dict) -> bool:
    """Save annotation to MongoDB."""
    storage = get_storage()

    # Build DetectabilityLabels
    detectability = DetectabilityLabels(
        error_detected=responses["error_detected"],
        error_trajectory=responses.get("error_trajectory"),
        error_step_id=(
            responses.get("error_step") if responses["error_detected"] else None
        ),
        confidence=responses["confidence"],
    )

    # Build ConsequenceLabels if error detected
    consequence = None
    if responses["error_detected"]:
        consequence = ConsequenceLabels(
            error_type=responses.get("error_type"),
            impact_tier=responses.get("impact_severity", 2),
            propagation_depth=None,  # Not collected in UI
            correctness_a=(
                responses.get("task_failed")
                if responses.get("error_trajectory") == "A"
                else "correct"
            ),
            correctness_b=(
                responses.get("task_failed")
                if responses.get("error_trajectory") == "B"
                else "correct"
            ),
        )

    # Create annotation record
    record = AnnotationRecord(
        annotation_id=f"ann_{unit['evaluation_unit_id']}_{ANNOTATOR_ID}_{int(time.time())}",
        evaluation_unit_id=unit["evaluation_unit_id"],
        annotator_id=ANNOTATOR_ID,
        annotation_mode="detectability",
        created_at=datetime.utcnow().isoformat() + "Z",
        view_file="streamlit_ui",
        trajectory_a_variant_id=unit.get("baseline", {}).get("trajectory_id", "a"),
        trajectory_b_variant_id=unit.get("perturbed", {}).get("trajectory_id", "b"),
        detectability=detectability,
        consequence=consequence,
        preference=None,
        time_spent_seconds=responses.get("time_spent", 0),
        notes=responses.get("notes"),
        flagged_for_review=responses.get("flagged", False),
    )

    # Save to MongoDB
    saved = save_annotations_to_mongodb([record], storage, EXPERIMENT_ID)
    return saved > 0


def main():
    st.title("🔍 Trajectory Annotation")
    st.markdown("Compare trajectories A and B. Identify if either contains an error.")

    # Guidelines on main page
    with st.expander("📖 Annotation Guidelines", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
**What to Look For:**
- Wrong parameters (`_old` suffix, typos)
- Synonym substitutions
- Missing/deleted steps
- Inferior planning approach
- Wrong IDs or data references
            """)
        with col2:
            st.markdown("""
**Where Errors Hide:**
- **Tool Arguments (parsed)** section
- Tool names
- Planning steps
- Final answers
            """)
        with col3:
            st.markdown("""
**Color Legend:**
- 🔵 Blue = Only in A
- 🟡 Yellow = Only in B
- ⚫ Gray = Same in both

**Tips:** A or B can have error (50/50). If unsure, flag for review.
            """)

    # Load data
    all_units = load_evaluation_units()
    config = load_config()
    sample_size = config.get("phases", {}).get("annotate", {}).get("sample_size", 130)

    # Sample units
    sampled_units = get_sampled_units(all_units, sample_size)

    # Get completed IDs
    completed_ids = get_completed_ids()

    # Filter to remaining
    remaining_units = [
        u for u in sampled_units if u["evaluation_unit_id"] not in completed_ids
    ]

    # Progress
    total = len(sampled_units)
    done = len(completed_ids)

    st.sidebar.header("Progress")
    st.sidebar.progress(done / total if total > 0 else 0)
    st.sidebar.markdown(f"**{done} / {total}** completed")
    st.sidebar.markdown(f"**{len(remaining_units)}** remaining")

    if not remaining_units:
        st.success("🎉 All annotations complete!")
        st.balloons()
        return

    # Current unit
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()

    idx = st.session_state.current_idx
    if idx >= len(remaining_units):
        idx = 0
        st.session_state.current_idx = 0

    unit = remaining_units[idx]

    # Unit info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Unit:** `{unit['evaluation_unit_id']}`")
    st.sidebar.markdown(f"**Benchmark:** {unit.get('benchmark', 'unknown')}")

    # === Metadata Section (for AI agent investigation) ===
    # Only show IDs - no hints about perturbation type/location
    eval_unit_id = unit.get("evaluation_unit_id", "unknown")
    blinding = unit.get("blinding", {})
    traj_a_id = blinding.get("trajectory_a_variant_id", "unknown")
    traj_b_id = blinding.get("trajectory_b_variant_id", "unknown")
    source_traj = unit.get("source_trajectory_id", "unknown")

    with st.expander("📋 Metadata (for AI investigation)", expanded=False):
        metadata_text = f"""evaluation_unit_id: {eval_unit_id}
source_trajectory_id: {source_traj}
trajectory_a_id: {traj_a_id}
trajectory_b_id: {traj_b_id}
experiment_id: {EXPERIMENT_ID}"""
        st.code(metadata_text, language="yaml")

    # Task description
    st.header("Task")
    task_text = unit.get("task_text", "No task description")
    st.info(task_text)

    # Get trajectories
    baseline = unit.get("baseline", {}).get("trajectory", {})
    perturbed = unit.get("perturbed", {}).get("trajectory", {})

    # Trajectories side by side with diff highlighting
    st.header("Trajectories")

    # Legend and column headers
    header_col_a, header_col_b = st.columns(2)
    with header_col_a:
        st.subheader("Trajectory A")
    with header_col_b:
        st.subheader("Trajectory B")

    st.markdown(
        """
    <div style="font-size: 14px; margin-bottom: 15px; padding: 12px;
                background: #21262d; border-radius: 8px; border: 1px solid #30363d;">
        <strong style="color: white;">Legend:</strong>&nbsp;&nbsp;&nbsp;
        <span style="color: #58a6ff; font-weight: bold;">■ Blue = Only in A</span>
        &nbsp;&nbsp;&nbsp;
        <span style="color: #f0c000; font-weight: bold;">■ Yellow = Only in B</span>
        &nbsp;&nbsp;&nbsp;
        <span style="color: #8b949e;">■ Gray = Same in both</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    steps_a = baseline.get("steps", [])
    steps_b = perturbed.get("steps", [])
    max_steps = max(len(steps_a), len(steps_b))

    # Display steps side by side with diff
    for i in range(max_steps):
        step_a = steps_a[i] if i < len(steps_a) else None
        step_b = steps_b[i] if i < len(steps_b) else None

        col_a, col_b = st.columns(2)

        # Get content for diff comparison
        content_a = get_step_content(step_a) if step_a else "(no step)"
        content_b = get_step_content(step_b) if step_b else "(no step)"

        # Compute diff HTML
        html_a, html_b = compute_diff_html(content_a, content_b)

        with col_a:
            role_a = step_a.get("step_role", "?") if step_a else "-"
            with st.expander(f"Step {i+1} [{role_a}]", expanded=True):
                if step_a:
                    st.markdown(html_a, unsafe_allow_html=True)
                else:
                    st.text("(no corresponding step)")

        with col_b:
            role_b = step_b.get("step_role", "?") if step_b else "-"
            with st.expander(f"Step {i+1} [{role_b}]", expanded=True):
                if step_b:
                    st.markdown(html_b, unsafe_allow_html=True)
                else:
                    st.text("(no corresponding step)")

    # Annotation form
    st.header("Annotation")

    with st.form("annotation_form"):
        col1, col2 = st.columns(2)

        with col1:
            error_detected = st.radio(
                "Is there an error in either trajectory?",
                options=[True, False],
                format_func=lambda x: (
                    "Yes, I see an error" if x else "No error detected"
                ),
                index=1,
                horizontal=True,
            )

            if error_detected:
                error_trajectory = st.radio(
                    "Which trajectory has the error?",
                    options=["A", "B", "both", "neither"],
                    horizontal=True,
                )

                max_steps = max(len(steps_a), len(steps_b))
                error_step = st.selectbox(
                    "Which step contains the error?",
                    options=[None] + list(range(1, max_steps + 1)),
                    format_func=lambda x: (
                        "Select step..." if x is None else f"Step {x}"
                    ),
                )

                error_type = st.selectbox(
                    "Error type",
                    options=[
                        "planning",
                        "tool_selection",
                        "parameter",
                        "data_reference",
                        "other",
                    ],
                )

                impact_severity = st.slider(
                    "Impact severity",
                    min_value=0,
                    max_value=3,
                    value=2,
                    help="0=None, 1=Minor, 2=Moderate, 3=Critical",
                )

                task_failed = st.radio(
                    "Does this error cause task failure?",
                    options=["correct", "partial", "incorrect"],
                    format_func=lambda x: {
                        "correct": "No, task still succeeds",
                        "partial": "Partial success",
                        "incorrect": "Yes, task fails",
                    }[x],
                    horizontal=True,
                )
            else:
                error_trajectory = None
                error_step = None
                error_type = None
                impact_severity = 0
                task_failed = "correct"

        with col2:
            confidence = st.slider(
                "Confidence in your assessment",
                min_value=1,
                max_value=5,
                value=5,
                help="1=Very unsure, 5=Very confident",
            )

            notes = st.text_area(
                "Notes (optional)",
                placeholder="Any observations about this unit...",
                height=100,
            )

            flagged = st.checkbox("Flag for review")

        col_submit, col_skip = st.columns([1, 1])

        with col_submit:
            submitted = st.form_submit_button(
                "✅ Submit & Next", use_container_width=True, type="primary"
            )

        with col_skip:
            skipped = st.form_submit_button("⏭️ Skip", use_container_width=True)

    if submitted:
        time_spent = time.time() - st.session_state.start_time

        responses = {
            "error_detected": error_detected,
            "error_trajectory": error_trajectory if error_detected else None,
            "error_step": f"step_{error_step}" if error_step else None,
            "error_type": error_type if error_detected else None,
            "impact_severity": impact_severity if error_detected else 0,
            "task_failed": task_failed,
            "confidence": confidence,
            "notes": notes if notes else None,
            "flagged": flagged,
            "time_spent": int(time_spent),
        }

        if save_annotation(unit, responses):
            st.success(f"✅ Saved! ({time_spent:.1f}s)")
            st.session_state.current_idx += 1
            st.session_state.start_time = time.time()
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("Failed to save annotation")

    if skipped:
        st.session_state.current_idx += 1
        st.session_state.start_time = time.time()
        st.rerun()

    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    if st.sidebar.button("⏮️ Previous"):
        if st.session_state.current_idx > 0:
            st.session_state.current_idx -= 1
            st.session_state.start_time = time.time()
            st.rerun()

    # Stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Session Stats")
    st.sidebar.markdown(f"Current: {idx + 1} / {len(remaining_units)}")


if __name__ == "__main__":
    main()
