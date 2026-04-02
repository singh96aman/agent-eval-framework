# Dataset Setup Instructions

This directory contains the datasets used for the POC experiment.

## Directory Structure

```
data/
├── toolbench/       # ToolBench dataset
├── gaia/            # GAIA benchmark dataset
├── perturbed/       # Generated perturbed trajectories (created by experiment)
└── annotations/     # Human annotations (created during experiment)
```

## Setting Up ToolBench Dataset

### Option 1: Use Official ToolBench Dataset

1. Download ToolBench dataset from the official repository:
   ```bash
   # Clone or download from https://github.com/OpenBMB/ToolBench
   ```

2. Place trajectory files in `data/toolbench/`:
   - Files should be in JSON or JSONL format
   - Each file should contain trajectory data with:
     - `task` or `question`: Task description
     - `steps` or `trajectory`: List of steps
     - `final_answer`: Expected answer
     - `success` (optional): Whether trajectory succeeded

3. Example format:
   ```json
   {
     "task": "Find the population of Tokyo in 2023",
     "steps": [
       {
         "thought": "I need to search for this",
         "action": "Search",
         "action_input": {"query": "Tokyo population 2023"},
         "observation": "14.09 million"
       }
     ],
     "final_answer": "14.09 million",
     "success": true
   }
   ```

### Option 2: Use Custom Trajectory Data

If using custom agent trajectories:

1. Convert your trajectories to the expected format (see example above)
2. Save as JSON or JSONL files in `data/toolbench/`
3. Ensure trajectories have 5-10 steps for optimal POC results

## Setting Up GAIA Dataset

### Option 1: Use Official GAIA Benchmark

1. Download GAIA validation set from Hugging Face:
   ```bash
   # Download from https://huggingface.co/datasets/gaia-benchmark/GAIA
   ```

2. Place files in `data/gaia/`:
   - Files should be in JSON or JSONL format
   - Each file should contain:
     - `question` or `Question`: The question
     - `final_answer` or `Final answer`: Expected answer
     - `Level` or `level`: Difficulty level
     - `trajectory` or `steps` (optional): Reasoning steps

3. Example format:
   ```json
   {
     "question": "What is the capital of France?",
     "final_answer": "Paris",
     "Level": "Level 1",
     "trajectory": [
       {
         "type": "search",
         "query": "capital of France",
         "result": "Paris"
       }
     ]
   }
   ```

### Option 2: Create Minimal Test Dataset

For testing or development:

1. Create a few sample trajectories manually
2. Save in `data/gaia/sample.json`
3. Verify with pre-requisite checker

## Verifying Dataset Setup

After placing datasets, run the pre-requisite check:

```bash
python src/prereq_check.py
```

This will verify:
- ✓ Directories exist
- ✓ Files are present and parseable
- ✓ Sample trajectories load successfully
- ✓ File formats are correct

## Dataset Requirements for POC

For the initial POC (50 trajectories):
- **ToolBench**: 30 successful trajectories (allows 5 for pilot + 25 for main)
  - Length: 5-10 steps
  - Success rate: >60%
  - Domain: Mixed (search, computation, APIs)

- **GAIA**: 30 trajectories from validation set (allows 5 for pilot + 25 for main)
  - Length: 4-8 steps
  - Difficulty: Level 1-2 preferred
  - Clear answer correctness criteria

## Troubleshooting

### "No JSON/JSONL files found"

- Check that files have `.json` or `.jsonl` extension
- Verify files are in correct directory (`data/toolbench/` or `data/gaia/`)
- Ensure directories exist

### "No valid trajectories loaded"

- Check file format matches examples above
- Verify JSON is valid (use `json.tool` or online validator)
- Check that required fields are present (`task`, `steps`, etc.)
- Review loader code in `src/data/loaders.py` for expected format

### "Trajectory too short/long"

- Adjust `min_steps` and `max_steps` parameters in loader
- Or filter your dataset to meet length requirements (5-10 steps for ToolBench, 4-8 for GAIA)

## Contact

If you have questions about dataset setup or need assistance with format conversion, contact the research team or refer to `paper/POC_REQUIREMENTS.MD` for detailed specifications.
