# Dataset Setup Instructions

## Overview

This project uses **Hugging Face datasets** for loading ToolBench and GAIA benchmarks. You don't need to manually download JSON files - datasets are loaded programmatically via the `datasets` library.

## Setup Instructions

### 1. Get HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is sufficient)
3. Add to your `.env` file:
   ```bash
   HUGGINGFACE_TOKEN=your_token_here
   ```

### 2. Datasets Used

**ToolBench:**
- HuggingFace: `OpenBMB/ToolBench`
- Used for: Multi-step tool-using agent trajectories
- Access: May require authentication

**GAIA:**
- HuggingFace: `gaia-benchmark/GAIA`
- Used for: Multi-hop question-answering trajectories
- Access: Public dataset

### 3. Loading Datasets

Datasets are loaded automatically by the experiment code:

```python
from src.data.loaders import load_toolbench_trajectories, load_gaia_trajectories

# Load ToolBench (automatically fetches from HuggingFace)
toolbench_trajs = load_toolbench_trajectories(
    max_trajectories=25,
    min_steps=5,
    max_steps=10,
    filter_successful=True
)

# Load GAIA
gaia_trajs = load_gaia_trajectories(
    max_trajectories=25,
    min_steps=4,
    max_steps=8,
    difficulty="Level 1"
)
```

### 4. Data Storage

**Trajectories**: Stored in MongoDB (not in this directory)
- Original trajectories
- Perturbed trajectories
- Metadata

**Results**: Stored in MongoDB
- Annotations
- Judge evaluations
- CCG scores
- Experiment metadata

See `src/storage/mongodb.py` for storage implementation.

## MongoDB Setup

### Option 1: Local MongoDB

```bash
# Install MongoDB (macOS)
brew install mongodb-community

# Start MongoDB
brew services start mongodb-community

# Set in .env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=agent_judge_experiment
```

### Option 2: MongoDB Atlas (Cloud)

1. Create free cluster at https://www.mongodb.com/cloud/atlas
2. Get connection string
3. Add to `.env`:
   ```bash
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
   MONGODB_DATABASE=agent_judge_experiment
   ```

## Verifying Setup

Run the pre-requisite checker:

```bash
python src/prereq_check.py
```

This verifies:
- ✓ MongoDB connection
- ✓ HuggingFace token and dataset access
- ✓ AWS Bedrock (Claude API)
- ✓ GPT-OSS endpoint
- ✓ Python dependencies

## Troubleshooting

### "HuggingFace token not set"

Make sure you:
1. Created token at https://huggingface.co/settings/tokens
2. Added `HUGGINGFACE_TOKEN=...` to `.env`
3. `.env` file is in project root

### "Could not connect to MongoDB"

- Check MongoDB is running: `brew services list`
- Verify URI in `.env` is correct
- For Atlas: check network access and credentials

### "Dataset not accessible"

Some datasets may require:
- Accepting terms of use on HuggingFace
- Requesting access from dataset owners
- Different authentication

## Directory Structure

```
data/
├── README.md           # This file
├── (no JSON files)     # Loaded from HuggingFace
└── (no local storage)  # Results stored in MongoDB
```

All experiment data is stored in MongoDB, not in local files.
