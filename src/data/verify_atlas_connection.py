#!/usr/bin/env python3
"""
Quick script to verify MongoDB Atlas connection.

Usage:
    python verify_atlas_connection.py
"""

import os
import sys
from dotenv import load_dotenv
from src.storage.mongodb import MongoDBStorage


def main():
    print("\n" + "="*70)
    print("MongoDB Atlas Connection Verification")
    print("="*70 + "\n")

    # Load environment variables
    load_dotenv()

    mongodb_uri = os.getenv("MONGODB_URI")

    if not mongodb_uri:
        print("❌ MONGODB_URI not found in environment")
        print("\nPlease set MONGODB_URI in .env file:")
        print("  MONGODB_URI=mongodb+srv://...")
        sys.exit(1)

    # Check if Atlas
    is_atlas = "mongodb+srv://" in mongodb_uri

    if is_atlas:
        print("✅ MongoDB Atlas URI detected")
        # Don't print full URI (contains password)
        print(f"   Host: ...{mongodb_uri.split('@')[1] if '@' in mongodb_uri else 'unknown'}")
    else:
        print("⚠️  Using local MongoDB")
        print(f"   URI: {mongodb_uri}")

    print("\nAttempting connection...")

    try:
        # Initialize storage (will test connection)
        storage = MongoDBStorage(test_connection=True)

        print("\n" + "="*70)
        print("Connection Test Results")
        print("="*70)
        print(f"✅ Connected to: {storage.database_name}")
        print(f"✅ Type: {'MongoDB Atlas' if storage.is_atlas else 'Local MongoDB'}")
        print(f"✅ Connection: Working")

        # List collections
        collections = storage.db.list_collection_names()
        print(f"\nExisting collections ({len(collections)}):")
        for coll in collections:
            count = storage.db[coll].count_documents({})
            print(f"  - {coll}: {count} documents")

        # Test write
        print("\nTesting write operation...")
        test_doc = {
            "test_id": "verify_script_test",
            "message": "Connection verification successful"
        }
        storage.db["_connection_test"].insert_one(test_doc)

        # Test read
        read_doc = storage.db["_connection_test"].find_one({
            "test_id": "verify_script_test"
        })

        if read_doc:
            print("✅ Write/Read test successful")

        # Cleanup
        storage.db["_connection_test"].delete_one({"test_id": "verify_script_test"})

        storage.close()

        print("\n" + "="*70)
        print("✅ All checks passed - MongoDB Atlas is ready!")
        print("="*70 + "\n")

        return 0

    except Exception as e:
        print("\n" + "="*70)
        print("❌ Connection Failed")
        print("="*70)
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check MONGODB_URI in .env file")
        print("2. Verify MongoDB Atlas credentials")
        print("3. Check network connectivity")
        print("4. Ensure IP is whitelisted in Atlas")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
