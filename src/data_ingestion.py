import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

class DataPipeline:
    def __init__(self, raw_data_path, db_connection_string):
        self.raw_data_path = raw_data_path
        self.client = MongoClient(db_connection_string)
        self.db = self.client['churn_database']
        self.collection = self.db['customer_features']

    def load_raw_data(self):
        """Loads data from a local CSV or Excel file."""
        print(f"Loading raw data from {self.raw_data_path}...")
        try:
            if self.raw_data_path.suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(self.raw_data_path)
            else:
                df = pd.read_csv(self.raw_data_path)
            print(f"Successfully loaded {len(df)} records.")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def upload_to_feature_store(self, df, batch_size=500):
        """Uploads processed dataframe to MongoDB in batches."""
        print("Uploading to MongoDB Atlas...")
        records = df.to_dict(orient="records")

        self.collection.delete_many({})

        total = len(records)
        for start in range(0, total, batch_size):
            batch = records[start : start + batch_size]
            self.collection.insert_many(batch)
            print(f"Uploaded {min(start + batch_size, total)}/{total} records...")

        print("Upload complete.")

if __name__ == "__main__":
    from feature_engineering import FeatureEngineer

    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise ValueError("MONGO_URI is not set. Add it to the .env file in the project root.")
    
    raw_data_path = PROJECT_ROOT / "data" / "raw" / "Telco_customer_churn.xlsx"
    processed_data_path = PROJECT_ROOT / "data" / "processed" / "engineered_features.csv"

    pipeline = DataPipeline(
        raw_data_path=raw_data_path,
        db_connection_string=MONGO_URI
    )
    
    # 1. Load Raw Data
    raw_df = pipeline.load_raw_data()
    
    if raw_df is not None:
        # 2. Engineer Features
        engineer = FeatureEngineer()
        processed_df = engineer.run_pipeline(raw_df)
        
        # Save a local copy for quick model training without needing to query the DB every time
        processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(processed_data_path, index=False)
        print(f"Saved engineered features to {processed_data_path}")
        
        # 3. Upload to MongoDB Atlas
        pipeline.upload_to_feature_store(processed_df)