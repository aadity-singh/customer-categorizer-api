# src/model_training.py
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src.features import build_features


# -----------------------------
# Load data
# -----------------------------
def load_labeled_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# -----------------------------
# Create target (SEGMENTATION ONLY)
# -----------------------------
def create_customer_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Customer segmentation based on CURRENT spending.
    IMPORTANT: Total_Spending must NOT be used as a model feature.
    """

    spending_cols = [
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]

    df = df.copy()
    total_spending = df[spending_cols].sum(axis=1)

    df["Customer_Category"] = pd.cut(
        total_spending,
        bins=[-1, 800, 2000, float("inf")],
        labels=["Low Value", "Medium Value", "High Value"]
    )

    return df


# -----------------------------
# Train model
# -----------------------------
def train_model(df: pd.DataFrame):
    TARGET = "Customer_Category"

    # ❌ DROP OFFLINE / LEAKY / NON-API COLUMNS
    DROP_COLS = [
        TARGET,
        "ID",
        "Dt_Customer",
        "Total_Spending",
        "Total_Purchases",
        "Engagement_Score"
    ]

    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df[TARGET]

    print("\n📊 Target distribution:")
    print(y.value_counts())

    # ✅ SHARED FEATURE ENGINEERING
    X = build_features(X)

    categorical_features = ["Education", "Marital_Status"]
    numerical_features = X.drop(columns=categorical_features).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, pipeline.predict(X_test)))

    return pipeline

# -----------------------------
# Save model
# -----------------------------
def save_model(model, path: str):
    joblib.dump(model, path)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    input_path = "data/labeled_marketing_campaign.csv"
    model_path = "models/customer_categorizer.pkl"  # ✅ SAME NAME AS API

    df = load_labeled_data(input_path)
    df = create_customer_category(df)

    trained_model = train_model(df)
    save_model(trained_model, model_path)

    print("\n✅ Model training completed")
    print(f"📦 Model saved at: {model_path}")
