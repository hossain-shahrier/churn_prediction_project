# src/feature_engineering.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ID_COLUMNS = {"CustomerID", "customerID", "Count"}
HIGH_CARDINALITY_COLUMNS = {"City", "Zip Code", "Lat Long", "Churn Reason"}
LEAKAGE_COLUMNS = {"Churn Value", "Churn Score"}
ONE_HOT_MAX_CATEGORIES = 10

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}

    def clean_data(self, df):
        """Handles missing values and incorrect data types."""
        print("Cleaning data...")
        df_clean = df.copy()

        total_charges_column = next(
            (column for column in ("Total Charges", "TotalCharges") if column in df_clean.columns),
            None,
        )
        if total_charges_column:
            df_clean[total_charges_column] = pd.to_numeric(
                df_clean[total_charges_column], errors="coerce"
            )
            df_clean[total_charges_column] = df_clean[total_charges_column].fillna(0)

        columns_to_drop = [
            column
            for column in df_clean.columns
            if column in ID_COLUMNS | HIGH_CARDINALITY_COLUMNS | LEAKAGE_COLUMNS
        ]
        if columns_to_drop:
            df_clean = df_clean.drop(columns=columns_to_drop)

        return df_clean

    def add_derived_features(self, df):
        """Adds domain-specific features that often improve churn prediction."""
        df_features = df.copy()

        tenure_column = next(
            (column for column in ("Tenure Months", "tenure") if column in df_features.columns),
            None,
        )
        monthly_charges_column = next(
            (
                column
                for column in ("Monthly Charges", "MonthlyCharges")
                if column in df_features.columns
            ),
            None,
        )
        total_charges_column = next(
            (column for column in ("Total Charges", "TotalCharges") if column in df_features.columns),
            None,
        )

        if tenure_column:
            tenure = df_features[tenure_column].clip(lower=0)
            df_features["Tenure Group"] = pd.cut(
                tenure,
                bins=[-1, 12, 24, 48, np.inf],
                labels=["0-12", "13-24", "25-48", "49+"],
            )

        if tenure_column and total_charges_column:
            tenure_safe = df_features[tenure_column].clip(lower=1)
            df_features["Avg Monthly Charge"] = (
                df_features[total_charges_column] / tenure_safe
            )

        if monthly_charges_column and "Avg Monthly Charge" in df_features.columns:
            avg_charge = df_features["Avg Monthly Charge"].replace(0, np.nan)
            df_features["Charge Ratio"] = (
                df_features[monthly_charges_column] / avg_charge
            ).fillna(1.0)

        if monthly_charges_column and tenure_column:
            df_features["Lifetime Spend Proxy"] = (
                df_features[monthly_charges_column] * df_features[tenure_column]
            )

        if monthly_charges_column:
            df_features["High Monthly Charge"] = (
                df_features[monthly_charges_column]
                > df_features[monthly_charges_column].median()
            ).astype(int)

        return df_features

    def encode_features(self, df):
        """Encodes categorical variables into numerical formats."""
        print("Encoding categorical features...")
        df_encoded = df.copy()

        categorical_cols = df_encoded.select_dtypes(include=["object", "category"]).columns

        for col in categorical_cols:
            unique_count = df_encoded[col].nunique(dropna=False)
            if unique_count <= 2:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
            elif unique_count <= ONE_HOT_MAX_CATEGORIES:
                df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True)
            else:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le

        df_encoded.columns = df_encoded.columns.astype(str)

        return df_encoded

    def run_pipeline(self, raw_df):
        """Executes the full engineering pipeline."""
        df_cleaned = self.clean_data(raw_df)
        df_enriched = self.add_derived_features(df_cleaned)
        df_final = self.encode_features(df_enriched)
        print(f"Feature engineering complete. Final shape: {df_final.shape}")
        return df_final