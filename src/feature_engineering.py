# feature_engineering.py
import pandas as pd
from src.features import build_features


def load_clean_data(path: str) -> pd.DataFrame:
    """
    Load cleaned dataset
    """
    return pd.read_csv(path)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering
    (SINGLE SOURCE OF TRUTH → src.features.build_features)
    """
    return build_features(df)


def save_featured_data(df: pd.DataFrame, output_path: str):
    """
    Save feature-engineered dataset
    """
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    input_path = "data/clean_marketing_campaign.csv"
    output_path = "data/featured_marketing_campaign.csv"

    df = load_clean_data(input_path)
    df_featured = create_features(df)
    save_featured_data(df_featured, output_path)

    print("✅ Feature engineering completed")
    print("Final shape:", df_featured.shape)
