import os
from pymongo import MongoClient

# 🔑 Required
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable is missing")

# ✅ Optional (safe defaults)
DB_NAME = os.getenv("MONGODB_DB", "customer_intelligence")
PREDICTIONS_COLLECTION = os.getenv("PREDICTIONS_COLLECTION", "predictions")
EXPLANATIONS_COLLECTION = os.getenv("EXPLANATIONS_COLLECTION", "explanations")

# MongoDB client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Collections
predictions_col = db[PREDICTIONS_COLLECTION]
explanations_col = db[EXPLANATIONS_COLLECTION]
