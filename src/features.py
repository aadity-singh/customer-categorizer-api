# src/features.py
import pandas as pd
from datetime import datetime

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Age
    current_year = datetime.now().year
    df["Age"] = current_year - df["Year_Birth"]

    # Children
    df["Children_Count"] = df["Kidhome"] + df["Teenhome"]

    # Total Spending
    spending_cols = [
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]
    df["Total_Spending"] = df[spending_cols].sum(axis=1)

    # Total Purchases
    purchase_cols = [
        "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"
    ]
    df["Total_Purchases"] = df[purchase_cols].sum(axis=1)

    # Engagement Score
    df["Engagement_Score"] = (
        df["Total_Purchases"]
        + df["NumWebVisitsMonth"]
        - df["Recency"]
    )

    # Tenure
    df["Customer_Tenure_Days"] = 365

    return df
