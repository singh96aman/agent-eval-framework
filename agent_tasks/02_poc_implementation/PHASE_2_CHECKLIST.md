# Phase 2 Checklist: Load Real Trajectories

## Goal
Load 50 real trajectories (25 ToolBench + 25 GAIA) from HuggingFace and store in MongoDB.

---

## Pre-Requisites (6 checks must pass)

### ✅ Already Passing
- [x] Directory Structure
- [x] Python Dependencies

### ❌ Need Configuration

#### 1. MongoDB Connection
```bash
# In .env file:
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/
MONGODB_DATABASE=agent_judge_experiment
```

**Get MongoDB Atlas URI:**
- Go to: https://cloud.mongodb.com/
- Create free cluster (or use existing)
- Click "Connect" → "Drivers"
- Copy connection string
- Replace `<password>` with your actual password

---

#### 2. HuggingFace Token
```bash
# In .env file:
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

**Get token:**
- Go to: https://huggingface.co/settings/tokens
- Create new token (read permissions)
- Copy and paste into .env

---

#### 3. Claude 3.5 Sonnet (Bedrock)
```bash
# In .env file - UPDATE THIS LINE:
AWS_BEDROCK_CLAUDE_3_5_SONNET=anthropic.claude-3-5-sonnet-20250320-v2:0
```

**Current issue:** Model version `20241022-v2` has reached end of life.

**Latest model IDs:**
- `anthropic.claude-3-5-sonnet-20250320-v2:0` (Sonnet 3.5 v2 - March 2025)
- `anthropic.claude-3-7-sonnet-20250219-v1:0` (Sonnet 3.7 - Feb 2025)
- Check AWS Bedrock console for latest available

---

#### 4. GPT-OSS Model (Optional for Phase 2)
```bash
# In .env file:
AWS_BEDROCK_GPT_OSS=meta.llama3-70b-instruct-v1:0
```

**Note:** Not required for Phase 2 (trajectory loading), but needed for Phase 4 (judge evaluation).

Can use any available open-source model on Bedrock:
- Meta Llama models
- Mistral models
- Cohere models

---

## Steps to Complete Phase 2

### Step 1: Configure Environment
```bash
# Edit .env file with your credentials
nano .env
```

### Step 2: Verify Pre-Requisites
```bash
# Should show 6/6 passing (or 5/6 if skipping GPT-OSS for now)
python src/prereq_check.py
```

### Step 3: Test Load (Dry Run)
```bash
# Test loading without saving to MongoDB
python src/load_trajectories.py --toolbench 5 --gaia 5 --dry-run
```

### Step 4: Load Full Dataset
```bash
# Load 25 ToolBench + 25 GAIA trajectories
python src/load_trajectories.py --toolbench 25 --gaia 25
```

### Step 5: Verify Data in MongoDB
```bash
# Check that trajectories were stored
python -c "
from src.storage.mongodb import MongoDBStorage
storage = MongoDBStorage()
experiments = list(storage.db.experiments.find())
print(f'Experiments: {len(experiments)}')
trajectories = list(storage.db.trajectories.find())
print(f'Trajectories: {len(trajectories)}')
storage.close()
"
```

---

## Expected Output

When successful, you should see:
```
======================================================================
PHASE 2: LOAD TRAJECTORIES FROM HUGGINGFACE
======================================================================

📥 Loading 25 ToolBench trajectories...
   ✓ Loaded 25 ToolBench trajectories

📥 Loading 25 GAIA trajectories...
   ✓ Loaded 25 GAIA trajectories

📊 Total trajectories loaded: 50

Sample trajectory:
  ID: toolbench_0
  Benchmark: toolbench
  Steps: 8
  Task: Find the current weather in San Francisco...

💾 Storing trajectories in MongoDB...
   ✓ Created experiment: exp_abc123
   ... stored 10/50
   ... stored 20/50
   ... stored 30/50
   ... stored 40/50
   ... stored 50/50
   ✓ Stored 50 trajectories

======================================================================
✅ PHASE 2 COMPLETE
======================================================================
Experiment ID: exp_abc123
Trajectories: 50
Database: agent_judge_experiment

Next: Phase 3 - Perturbation generation
======================================================================
```

---

## Troubleshooting

### HuggingFace: "Dataset not found"
- Verify token has read permissions
- Check dataset names in `src/data/loaders.py`:
  - ToolBench: `OpenBMB/ToolBench`
  - GAIA: `gaia-benchmark/GAIA`

### MongoDB: "Connection refused"
- Check URI format includes `mongodb+srv://` for Atlas
- Verify password doesn't have special characters (URL encode if needed)
- Check IP whitelist in MongoDB Atlas (allow your IP)

### Bedrock: "Model not found"
- Check available models in AWS Bedrock console
- Verify region is correct (default: us-east-1)
- Update model ID in .env to latest version

---

## After Phase 2

Once trajectories are loaded:
- ✅ 50 baseline trajectories in MongoDB
- Ready for **Phase 3: Perturbation Generation**
  - Generate 50 perturbed versions (9 conditions)
  - Store perturbation metadata
  - Create annotation interface for human review
