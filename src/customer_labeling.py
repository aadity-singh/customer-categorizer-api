import pandas as pd


def load_featured_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def create_customer_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer value categories based on spending
    """

    # Create quantile thresholds
    low_threshold = df["Total_Spending"].quantile(0.33)
    high_threshold = df["Total_Spending"].quantile(0.66)

    def label_customer(spending):
        if spending >= high_threshold:
            return "High Value"
        elif spending >= low_threshold:
            return "Medium Value"
        else:
            return "Low Value"

    df["Customer_Category"] = df["Total_Spending"].apply(label_customer)

    return df


def save_labeled_data(df: pd.DataFrame, output_path: str):
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    input_path = "data/featured_marketing_campaign.csv"
    output_path = "data/labeled_marketing_campaign.csv"

    df = load_featured_data(input_path)
    df_labeled = create_customer_category(df)
    save_labeled_data(df_labeled, output_path)

    print("✅ Customer categorization completed")
    print(df_labeled["Customer_Category"].value_counts())
