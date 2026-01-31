from dotenv import load_dotenv
import os
from pymongo import MongoClient

# Load environment variables from .env
load_dotenv()

# 🔑 Read env vars (IMPORTANT: MONGO_URI, not MONGODB_URI)
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGODB_DB")
PREDICTIONS_COLLECTION = os.getenv("PREDICTIONS_COLLECTION")
EXPLANATIONS_COLLECTION = os.getenv("EXPLANATIONS_COLLECTION")

# Fail fast if anything missing
if not all([MONGO_URI, DB_NAME, PREDICTIONS_COLLECTION, EXPLANATIONS_COLLECTION]):
    raise RuntimeError("One or more MongoDB environment variables are missing")

# MongoDB client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Collections
predictions_col = db[PREDICTIONS_COLLECTION]
explanations_col = db[EXPLANATIONS_COLLECTION]
