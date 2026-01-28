import pandas as pd
from src.features import build_features
from src.model_loader import load_model


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

    # 3️⃣ Drop any unexpected columns (VERY IMPORTANT)
    df = df[REQUIRED_COLUMNS]

    # 4️⃣ Apply SAME feature engineering as training
    df = build_features(df)

    # 5️⃣ Predict
    prediction = model.predict(df)[0]

    return prediction
