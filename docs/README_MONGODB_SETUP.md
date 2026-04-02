# MongoDB Atlas Setup Guide

## Quick Start

### 1. Install MongoDB Driver

```bash
python -m pip install "pymongo[srv]"
```

### 2. Configure Environment

Copy `.env.example` to `.env` and update with your MongoDB Atlas credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```bash
# MongoDB Atlas connection
MONGODB_URI=CLASSIFIED
appName=CLASSIFIED
MONGODB_DATABASE=CLASSIFIED
```

### 3. Verify Connection

```bash
python verify_atlas_connection.py
```

Expected output:
```
======================================================================
MongoDB Atlas Connection Verification
======================================================================

✅ MongoDB Atlas URI detected
   Host: ...judging-the-agents.zzzls0n.mongodb.net/?appName=Judging-the-Agents

Attempting connection...
🔌 Connecting to MongoDB Atlas...
   Database: agent_judge_experiment
✅ Connected to MongoDB Atlas

======================================================================
Connection Test Results
======================================================================
✅ Connected to: agent_judge_experiment
✅ Type: MongoDB Atlas
✅ Connection: Working

Existing collections (0):

Testing write operation...
✅ Write/Read test successful

======================================================================
✅ All checks passed - MongoDB Atlas is ready!
======================================================================
```

---

## Production vs Tests

### Production Usage (MongoDB Atlas)

When running experiments:

```python
from src.storage.mongodb import MongoDBStorage

# Automatically uses MongoDB Atlas from .env
storage = MongoDBStorage()  # test_connection=True by default

# Create experiment
storage.create_experiment({
    "experiment_id": "poc_001",
    "name": "My Experiment",
    "config": {...}
})

# Store trajectories with foreign keys
storage.save_trajectory({
    "trajectory_id": "traj_001",
    "experiment_id": "poc_001",  # Foreign key!
    "benchmark": "toolbench",
    ...
})
```

**Connection verified on startup:**
```
🔌 Connecting to MongoDB Atlas...
   Database: agent_judge_experiment
✅ Connected to MongoDB Atlas
```

### Test Usage (No MongoDB Required)

When running tests:

```bash
# Integration tests use JSON files
pytest tests/test_integration_pipeline.py -v

# Schema tests use in-memory mocks
pytest tests/test_mongodb_schema.py -v

# Storage tests use mocked connections
pytest tests/test_storage.py -v
```

**No MongoDB connection needed for tests!**

---

## MongoDB Atlas Connection Details

### Connection String Format

```
mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?appName=<app-name>
```

**Current Configuration:**
- **Username:** `singh96aman_db_user`
- **Password:** `rzVqnZkWrDrsryRy` (stored in .env, not committed)
- **Cluster:** `judging-the-agents.zzzls0n.mongodb.net`
- **App Name:** `Judging-the-Agents`
- **Database:** `agent_judge_experiment`

### Security Notes

⚠️ **IMPORTANT:**
- Never commit `.env` file to git (it's in `.gitignore`)
- Only `.env.example` is committed (without real credentials)
- Rotate credentials if accidentally exposed

---

## Collections and Indexes

### Collections Created

When you initialize `MongoDBStorage`, it creates these collections:

1. **experiments** - Experiment metadata
2. **trajectories** - Original and perturbed trajectories
3. **annotations** - Human annotations
4. **judge_evaluations** - LLM judge ratings
5. **ccg_scores** - Computed CCG scores

### Indexes Created Automatically

Foreign key indexes for O(1) queries:

```javascript
// experiments
db.experiments.createIndex({ experiment_id: 1 }, { unique: true })
db.experiments.createIndex({ status: 1 })

// trajectories
db.trajectories.createIndex({ trajectory_id: 1 }, { unique: true })
db.trajectories.createIndex({ experiment_id: 1 })  // ← Foreign key!
db.trajectories.createIndex({ benchmark: 1 })

// annotations
db.annotations.createIndex({ annotation_id: 1 }, { unique: true })
db.annotations.createIndex({ experiment_id: 1 })  // ← Foreign key!
db.annotations.createIndex({ trajectory_id: 1 })  // ← Foreign key!

// judge_evaluations
db.judge_evaluations.createIndex({ evaluation_id: 1 }, { unique: true })
db.judge_evaluations.createIndex({ experiment_id: 1 })  // ← Foreign key!
db.judge_evaluations.createIndex({ trajectory_id: 1, judge_model: 1 })

// ccg_scores
db.ccg_scores.createIndex({ ccg_id: 1 }, { unique: true })
db.ccg_scores.createIndex({ experiment_id: 1 })  // ← Foreign key!
db.ccg_scores.createIndex({ trajectory_id: 1 })
db.ccg_scores.createIndex({ perturbation_type: 1, perturbation_position: 1 })
```

**Why indexes matter:**
- Without: O(n) collection scan
- With: O(1) indexed lookup
- Critical for large experiments with 100K+ trajectories

---

## Troubleshooting

### Connection Failed

**Error:** `Failed to connect to MongoDB`

**Solutions:**
1. Check `.env` file exists and has correct URI
2. Verify MongoDB Atlas credentials are correct
3. Check network connectivity
4. Ensure your IP is whitelisted in MongoDB Atlas:
   - Go to Atlas dashboard
   - Network Access → Add IP Address
   - Add your current IP or use `0.0.0.0/0` (allow all - less secure)

### DNS Resolution Failed

**Error:** `DNSServerDoesNotSupportFeatures`

**Solution:** Install DNS resolver:
```bash
python -m pip install dnspython
```

### Authentication Failed

**Error:** `Authentication failed`

**Solution:**
1. Check username/password in MONGODB_URI
2. Verify user has read/write permissions on database
3. In Atlas: Database Access → Edit user → Grant permissions

### Timeout

**Error:** `ServerSelectionTimeoutError`

**Solutions:**
1. Check internet connection
2. Verify MongoDB Atlas cluster is running
3. Check if IP is whitelisted
4. Try increasing timeout:
   ```python
   storage = MongoDBStorage(
       uri=os.getenv("MONGODB_URI") + "&serverSelectionTimeoutMS=10000"
   )
   ```

---

## Testing Atlas Connection

### Quick Test

```bash
python verify_atlas_connection.py
```

### Full Test Suite

```bash
# Test Atlas connection and operations
pytest tests/test_mongodb_atlas.py -v -s
```

This test will:
- ✅ Verify connection to Atlas
- ✅ Test write and read operations
- ✅ Validate foreign key relationships
- ✅ Check indexes are created

**Note:** Only runs if MongoDB Atlas is configured in `.env`

---

## Monitoring Usage

### Check Database Size

```python
from src.storage.mongodb import MongoDBStorage

storage = MongoDBStorage()

# Get database stats
stats = storage.db.command("dbStats")
print(f"Database size: {stats['dataSize'] / 1024 / 1024:.2f} MB")
print(f"Collections: {stats['collections']}")
print(f"Documents: {stats['objects']}")
```

### Check Collection Sizes

```python
for coll_name in storage.db.list_collection_names():
    count = storage.db[coll_name].count_documents({})
    print(f"{coll_name}: {count} documents")
```

### Monitor Atlas Dashboard

1. Go to [MongoDB Atlas Dashboard](https://cloud.mongodb.com/)
2. Select your cluster
3. View metrics:
   - CPU usage
   - Memory usage
   - Network traffic
   - Operations per second

---

## Migration from Local MongoDB

If you have existing data in local MongoDB:

```python
from pymongo import MongoClient

# Connect to local
local_client = MongoClient("mongodb://localhost:27017")
local_db = local_client["agent_judge_experiment_local"]

# Connect to Atlas
from src.storage.mongodb import MongoDBStorage
atlas = MongoDBStorage()

# Migrate collections
for coll_name in ["experiments", "trajectories", "annotations", 
                  "judge_evaluations", "ccg_scores"]:
    docs = list(local_db[coll_name].find())
    if docs:
        atlas.db[coll_name].insert_many(docs)
        print(f"Migrated {len(docs)} documents from {coll_name}")
```

---

## References

- [MongoDB Atlas Documentation](https://docs.atlas.mongodb.com/)
- [PyMongo Documentation](https://pymongo.readthedocs.io/)
- [MongoDB Connection String URI Format](https://www.mongodb.com/docs/manual/reference/connection-string/)
- Our Schema Documentation: [docs/MONGODB_SCHEMA.md](docs/MONGODB_SCHEMA.md)
