import pandas as pd
from datetime import datetime

from src.features import build_features
from src.model_loader import load_model
from src.db import predictions_col   # ✅ ADD THIS


# ✅ EXACT INPUT FEATURES USED DURING TRAINING (RAW)
REQUIRED_COLUMNS = [
    "Year_Birth",
    "Education",
    "Marital_Status",
    "Income",
    "Kidhome",
    "Teenhome",
    "Recency",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "Complain",
    "Response"
]


def predict_customer(input_data: dict):
    model = load_model()

    # 1️⃣ Create dataframe
    df = pd.DataFrame([input_data])

    # 2️⃣ Ensure ALL required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # 3️⃣ Drop any unexpected columns
    df = df[REQUIRED_COLUMNS]

    # 4️⃣ Apply SAME feature engineering as training
    df = build_features(df)

    # 5️⃣ Predict
    prediction = model.predict(df)[0] # cast for Mongo safety

    # 6️⃣ Save to MongoDB ✅
    predictions_col.insert_one({
        "input": input_data,
        "prediction": prediction,
        "model": "customer_categorizer_v1",
        "created_at": datetime.utcnow()
    })

    return prediction
