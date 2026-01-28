from pymongo import MongoClient
import os

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb://localhost:27017"
)

client = MongoClient(MONGO_URI)

db = client["customer_categorizer_db"]
predictions_collection = db["predictions"]
