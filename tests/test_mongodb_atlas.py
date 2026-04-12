"""
Test MongoDB Atlas connection and verify production usage.

This test verifies that:
1. MongoDB Atlas connection works with credentials
2. We can write and read data from Atlas
3. Foreign key schema works in production

Run manually when you want to test Atlas connection:
    pytest tests/test_mongodb_atlas.py -v -s

NOTE: This test requires MongoDB Atlas credentials in .env file
"""

import pytest
import os
from datetime import datetime
from src.storage.mongodb import MongoDBStorage


@pytest.mark.skipif(
    not os.getenv("MONGODB_URI") or "mongodb+srv" not in os.getenv("MONGODB_URI", ""),
    reason="MongoDB Atlas credentials not configured",
)
class TestMongoDBAtlas:
    """
    Test MongoDB Atlas connection.

    These tests run ONLY if MONGODB_URI is configured for Atlas.
    """

    @pytest.fixture(scope="class")
    def storage(self):
        """Create MongoDB Atlas storage connection."""
        print("\n" + "=" * 60)
        print("Testing MongoDB Atlas Connection")
        print("=" * 60)

        storage = MongoDBStorage(test_connection=True)

        # Clean up test data from previous runs
        storage.db.drop_collection("test_experiments")
        storage.db.drop_collection("test_trajectories")

        yield storage

        # Cleanup
        storage.db.drop_collection("test_experiments")
        storage.db.drop_collection("test_trajectories")
        storage.close()

    def test_atlas_connection(self, storage):
        """Test that we can connect to MongoDB Atlas."""
        assert storage.is_atlas, "Should be connected to MongoDB Atlas"
        assert storage.test_connection(), "Connection should be successful"

        print("✅ Successfully connected to MongoDB Atlas")

    def test_write_and_read_experiment(self, storage):
        """Test writing and reading from Atlas."""
        experiment_id = "test_atlas_experiment_001"

        # Create test experiment
        storage.db["test_experiments"].insert_one(
            {
                "experiment_id": experiment_id,
                "name": "Atlas Connection Test",
                "progress": {"trajectories_loaded": 0},
                "created_at": datetime.utcnow(),
            }
        )

        # Read it back
        experiment = storage.db["test_experiments"].find_one(
            {"experiment_id": experiment_id}
        )

        assert experiment is not None
        assert experiment["experiment_id"] == experiment_id
        assert "progress" in experiment
        assert "trajectories_loaded" in experiment["progress"]

        print(f"✅ Successfully wrote and read experiment from Atlas")
        print(f"   Experiment ID: {experiment_id}")

    def test_foreign_key_relationships(self, storage):
        """Test that foreign key schema works on Atlas."""
        experiment_id = "test_fk_001"

        # Create experiment
        storage.db["test_experiments"].insert_one(
            {
                "experiment_id": experiment_id,
                "name": "Foreign Key Test",
                "progress": {"trajectories_loaded": 0},
            }
        )

        # Create trajectories with foreign key
        for i in range(3):
            storage.db["test_trajectories"].insert_one(
                {
                    "trajectory_id": f"traj_{i}",
                    "experiment_id": experiment_id,  # Foreign key!
                    "benchmark": "toolbench",
                    "is_perturbed": False,
                }
            )

        # Query using foreign key
        trajectories = list(
            storage.db["test_trajectories"].find({"experiment_id": experiment_id})
        )

        assert len(trajectories) == 3
        for traj in trajectories:
            assert traj["experiment_id"] == experiment_id

        print(f"✅ Foreign key relationships work on Atlas")
        print(f"   Created 3 trajectories with experiment_id FK")

    def test_indexes_exist(self, storage):
        """Test that indexes are created properly."""
        # Get indexes on trajectories collection
        indexes = storage.trajectories.index_information()

        # Verify experiment_id index exists
        has_experiment_index = any(
            "experiment_id" in idx.get("key", [{}])[0] for idx in indexes.values()
        )

        assert has_experiment_index, "experiment_id index should exist"

        print("✅ Indexes created successfully on Atlas")
        print(f"   Total indexes on trajectories: {len(indexes)}")


def test_environment_setup():
    """
    Test that environment is configured correctly.

    This test always runs to verify .env setup.
    """
    mongodb_uri = os.getenv("MONGODB_URI")

    if not mongodb_uri:
        pytest.skip("MONGODB_URI not set in environment")

    # Check if it's Atlas
    is_atlas = "mongodb+srv://" in mongodb_uri

    if is_atlas:
        print("\n✅ MongoDB Atlas configured in environment")
        print(f"   URI starts with: mongodb+srv://...")
        assert "mongodb.net" in mongodb_uri
    else:
        print("\n⚠️  Using local MongoDB (not Atlas)")
        print(f"   URI: {mongodb_uri}")


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_mongodb_atlas.py -v -s
    pytest.main([__file__, "-v", "-s"])
