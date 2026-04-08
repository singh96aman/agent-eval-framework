# 1. Trajectory Sources

**Status: COMPLETED (2026-04-07)**

## Implementation Notes

- Added `SamplingProvenance` and `SamplingManifest` dataclasses to `src/data/schema.py`
- Added `load_stratified_sample()` function to `src/data/loaders.py`
- Updated `_phase_load()` in `src/experiment_runner.py` to use stratified sampling
- Created config: `config/experiments/v2/trajectory_sampling.json`
- Added 10 tests to `tests/test_loaders.py`
- See `bugs_1_trajectory_sources.md` for full details

---

## Dataset Summary

| Benchmark | Count | Domains/Levels | Step Range | Median Steps |
|-----------|-------|----------------|------------|--------------|
| ToolBench | 400 | 9 domains | 4-6 | 4 |
| GAIA | 100 | Multi-level reasoning | 4-87 | 7 |
| SWE-bench | 100 | Software engineering | 6-15 | 11 |
| **Total** | **600** | - | - | - |

## ToolBench Domains (400 trajectories)
- data_information
- ecommerce_shopping
- finance_business
- media_entertainment
- social_communication
- sports_gaming
- travel_logistics
- utilities_tools
- other

## GAIA (100 trajectories)
- Multi-step reasoning tasks requiring tool use
- Variable complexity (4-87 steps)

## SWE-bench (100 trajectories)
- Real-world software engineering tasks
- Code editing and debugging workflows

## Data Location
```
data/sampled/
├── toolbench_400.json
├── gaia_100.json
└── swebench_100.json
```

---

## Example: ToolBench Trajectory (toolbench_146416)

**Task:** "I want to create a personalized diet plan for myself to achieve my fitness goals. Can you provide me with the nutritional analysis for the foods I typically consume? Additionally, I need some guidelines for maintaining a healthy lifestyle."

```json
{
  "trajectory_id": "toolbench_146416",
  "benchmark": "toolbench",
  "domain": "data_information",
  "steps": [
    {
      "step_number": 1,
      "step_type": "tool_execution",
      "content": "Thought: I'm calling the \"api_nutrition_data_for_edamam_nutrition_analysis\" function with the argument \"{'ingr': 'oatmeal'}\" to retrieve the nutritional analysis for the specific ingredient \"oatmeal\".\nAction: api_nutrition_data_for_edamam_nutrition_analysis\nAction Input: {\"ingr\": \"oatmeal\"}",
      "tool_name": "api_nutrition_data_for_edamam_nutrition_analysis",
      "tool_input": {"ingr": "oatmeal"},
      "tool_output": "{\"error\": \"request invalid, data error. status_code=500\", \"response\": \"\"}"
    },
    {
      "step_number": 2,
      "step_type": "tool_execution",
      "content": "Thought: Based on the previous action and its outcome, it seems that there was an issue with the API request for the nutritional analysis of oatmeal. Therefore, to generate a different action, I will try a different food ingredient.\nAction: api_nutrition_data_for_edamam_nutrition_analysis\nAction Input: {\"ingr\": \"chicken breast\"}",
      "tool_name": "api_nutrition_data_for_edamam_nutrition_analysis",
      "tool_input": {"ingr": "chicken breast"},
      "tool_output": "{\"error\": \"request invalid, data error. status_code=500\", \"response\": \"\"}"
    },
    {
      "step_number": 3,
      "step_type": "tool_execution",
      "content": "Thought: I will try using the \"all_for_getguidelines\" function to obtain general guidelines for maintaining a healthy lifestyle.\nAction: all_for_getguidelines\nAction Input: {}",
      "tool_name": "all_for_getguidelines",
      "tool_input": {},
      "tool_output": "{\"error\": \"Message error...\", \"response\": \"Error code 500...\"}"
    },
    {
      "step_number": 4,
      "step_type": "tool_execution",
      "content": "Thought: I will try using the \"vac_for_getguidelines\" function to obtain guidelines specifically related to vaccinations.\nAction: vac_for_getguidelines\nAction Input: {}",
      "tool_name": "vac_for_getguidelines",
      "tool_input": {},
      "tool_output": null
    }
  ]
}
```

**Observations:** 4 steps, all tool_execution, repeated API failures, agent pivots to unrelated tool (vaccination guidelines) when nutrition API fails.

---

## Example: GAIA Trajectory (gaia_122)

**Task:** "How many more blocks (also denoted as layers) in BERT base encoder than the encoder from the architecture proposed in Attention is All You Need?"

**Expected Answer:** "6"

```json
{
  "trajectory_id": "gaia_122",
  "benchmark": "gaia",
  "domain": "qa",
  "difficulty": "1",
  "steps": [
    {
      "step_number": 1,
      "step_type": "tool_execution",
      "content": "Search the internet for \"blocks in bert base\"",
      "tool_name": "web_search",
      "tool_input": null,
      "tool_output": null
    },
    {
      "step_number": 2,
      "step_type": "tool_execution",
      "content": "Examine the search results page to locate the answer (12)",
      "tool_name": "web_search",
      "tool_input": null,
      "tool_output": null
    },
    {
      "step_number": 3,
      "step_type": "tool_execution",
      "content": "Search the internet for \"attention is all you need layers\"\nNavigate to https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf",
      "tool_name": "web_search",
      "tool_input": null,
      "tool_output": null
    },
    {
      "step_number": 4,
      "step_type": "reasoning",
      "content": "Examine the architecture section of the PDF to locate the answer (6)",
      "tool_name": null,
      "tool_input": null,
      "tool_output": null
    },
    {
      "step_number": 5,
      "step_type": "reasoning",
      "content": "Calculate the difference between the two numbers",
      "tool_name": null,
      "tool_input": null,
      "tool_output": null
    }
  ]
}
```

**Observations:** 5 steps, mix of tool_execution and reasoning, extracts two values (12, 6) then computes difference. Tool inputs/outputs not recorded in raw data.

---

## Example: SWE-bench Trajectory (swebench_jd__tenacity)

**Task:** Fix the `wait_chain` class in tenacity library where strategies are applied in reverse order.

```json
{
  "trajectory_id": "swebench_jd__tenacity.0d40e76f.121.4slwej0r",
  "benchmark": "swebench",
  "steps": [
    {
      "step_number": 1,
      "step_type": "tool_execution",
      "content": "First, let's find and read the relevant code:\n<function=str_replace_editor>\n<parameter=command>view</parameter>\n<parameter=path>/testbed</parameter>",
      "tool_name": "file_edit"
    },
    {
      "step_number": 2,
      "step_type": "tool_execution",
      "content": "Based on the PR description, we need to look at the `wait_chain` implementation. Since it's a wait strategy, it's likely in `wait.py`:\n<function=str_replace_editor>\n<parameter=command>view</parameter>\n<parameter=path>/testbed/tenacity/wait.py</parameter>",
      "tool_name": "file_edit"
    },
    {
      "step_number": 3,
      "step_type": "tool_execution",
      "content": "I found the issue! In the `__init__` method on line 108, the strategies are being reversed:\n```python\ndef __init__(self, *strategies: wait_base) -> None:\n    self.strategies = list(reversed(strategies))\n```\nLet's create a script to reproduce the issue...",
      "tool_name": "file_edit"
    },
    {
      "step_number": 7,
      "step_type": "tool_execution",
      "content": "Let me search for RetryCallState:\n<function=bash>\n<parameter=command>cd /testbed && grep -n RetryCallState tenacity/__init__.py</parameter>",
      "tool_name": "search_code"
    },
    {
      "step_number": 13,
      "step_type": "tool_execution",
      "content": "Let's fix this by removing the `reversed()` call:\n<function=str_replace_editor>\n<parameter=command>str_replace</parameter>\n<parameter=path>/testbed/tenacity/wait.py</parameter>\n<parameter=old_str>self.strategies = list(reversed(strategies))</parameter>\n<parameter=new_str>self.strategies = list(strategies)</parameter>",
      "tool_name": "file_edit"
    },
    {
      "step_number": 14,
      "step_type": "tool_execution",
      "content": "Now let's run the script again to verify the fix:\n<function=bash>\n<parameter=command>cd /testbed && python reproduce_wait_chain.py</parameter>",
      "tool_name": "run_tests"
    }
  ]
}
```

**Observations:** 14 steps, code exploration → hypothesis → reproduction → fix → verification pattern. Heavy file_edit and search_code usage. Agent navigates codebase iteratively.

---

## Trajectory Schema (current)

```json
{
  "trajectory_id": "string",
  "benchmark": "toolbench|gaia|swebench",
  "steps": [
    {
      "step_id": "string",
      "step_number": int,
      "step_type": "string",
      "content": "string",
      "tool_name": "string",
      "tool_input": {},
      "tool_output": "string",
      "metadata": {}
    }
  ],
  "ground_truth": {
    "task_description": "string",
    "expected_answer": "string|null",
    "task_success": "bool|null"
  },
  "metadata": {},
  "domain": "string",
  "complexity": "string"
}
```
