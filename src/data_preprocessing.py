import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """
    Load marketing campaign dataset
    """
    df = pd.read_csv(path, sep="\t")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data cleaning
    """
    # Remove duplicate customers
    df = df.drop_duplicates()

    # Convert date column
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")

    # Fix invalid income values
    df["Income"] = df["Income"].replace(0, np.nan)
    df["Income"] = df["Income"].fillna(df["Income"].median())

    # Remove customers with unrealistic age
    current_year = 2026
    df["Age"] = current_year - df["Year_Birth"]
    df = df[df["Age"] <= 90]

    # Drop columns with no analytical value
    df.drop(columns=["Z_CostContact", "Z_Revenue"], inplace=True)

    return df


def save_clean_data(df: pd.DataFrame, output_path: str):
    """
    Save cleaned dataset
    """
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    raw_path = "data/marketing_campaign.csv"
    output_path = "data/clean_marketing_campaign.csv"

    df = load_data(raw_path)
    df_clean = clean_data(df)
    save_clean_data(df_clean, output_path)

    print("✅ Data preprocessing completed")
    print("Final shape:", df_clean.shape)
