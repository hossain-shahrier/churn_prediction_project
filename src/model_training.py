# src/model_training.py
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "engineered_features.csv"
TARGET_COLUMN = "Churn Label"
LEAKAGE_COLUMNS = {
    "Churn Value",
    "Churn Score",
    "Churn Reason",
}
RANDOM_STATE = 42


class ModelTrainer:
    def __init__(self, db_connection_string):
        self.client = MongoClient(db_connection_string)
        self.db = self.client["churn_database"]
        self.collection = self.db["customer_features"]
        self.model = None
        self.threshold = 0.5
        self.feature_columns = []

    def fetch_data_from_db(self):
        """Pulls processed features from MongoDB Atlas into a Pandas DataFrame."""
        print("Fetching engineered features from MongoDB Atlas...")
        cursor = self.collection.find({})
        df = pd.DataFrame(list(cursor))

        if "_id" in df.columns:
            df = df.drop("_id", axis=1)

        print(f"Retrieved {len(df)} records for training.")
        return df

    def get_feature_columns(self, df, target_column=TARGET_COLUMN):
        return [
            column
            for column in df.columns
            if column != target_column
            and column not in LEAKAGE_COLUMNS
            and not column.startswith("Churn Reason_")
        ]

    def prepare_datasets(self, df, target_column=TARGET_COLUMN):
        """Splits data into train/validation/test sets."""
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found. "
                f"Available columns include: {list(df.columns[:10])}..."
            )

        self.feature_columns = self.get_feature_columns(df, target_column)
        X = df[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )

        num_neg = np.sum(y_train == 0)
        num_pos = np.sum(y_train == 1)
        scale_pos_weight = num_neg / num_pos

        return X_train, X_val, X_test, y_train, y_val, y_test, scale_pos_weight

    def tune_hyperparameters(self, X_train, y_train, scale_pos_weight):
        """Finds stronger XGBoost settings using cross-validated search."""
        print("Tuning hyperparameters with cross-validation...")

        base_model = xgb.XGBClassifier(
            objective="binary:logistic",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        )

        param_distributions = {
            "n_estimators": [150, 200, 300, 400],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.03, 0.05, 0.08, 0.1],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.1, 0.2],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [1, 2, 5],
        }

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=24,
            scoring="average_precision",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)

        print(f"Best CV PR-AUC: {search.best_score_:.4f}")
        print(f"Best params: {search.best_params_}")
        return search.best_estimator_

    def train_xgboost(self, X_train, y_train, X_val, y_val, scale_pos_weight, tune=True):
        """Trains an XGBoost classifier with optional tuning and early stopping."""
        if tune:
            model = self.tune_hyperparameters(X_train, y_train, scale_pos_weight)
            model.set_params(early_stopping_rounds=25)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            print("Training XGBoost Classifier...")
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                early_stopping_rounds=25,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

        self.model = model
        return model

    def find_optimal_threshold(self, X_val, y_val):
        """Picks the probability threshold that maximizes F1 on validation data."""
        probabilities = self.model.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, probabilities)

        f1_scores = (2 * precision * recall) / np.clip(precision + recall, a_min=1e-9, a_max=None)
        best_index = np.argmax(f1_scores[:-1])
        self.threshold = thresholds[best_index]

        print(
            f"Optimal threshold: {self.threshold:.3f} "
            f"(validation F1: {f1_scores[best_index]:.4f})"
        )
        return self.threshold

    def run_cross_validation(self, X_train, y_train, scale_pos_weight):
        """Reports stable out-of-fold metrics before final test evaluation."""
        print("Running 5-fold cross-validation on training data...")
        cv_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        roc_scores = cross_val_score(
            cv_model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        pr_scores = cross_val_score(
            cv_model, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1
        )

        print(f"CV ROC-AUC: {roc_scores.mean():.4f} (+/- {roc_scores.std():.4f})")
        print(f"CV PR-AUC:  {pr_scores.mean():.4f} (+/- {pr_scores.std():.4f})")

    def evaluate_model(self, X_test, y_test, threshold=None):
        """Evaluates the model using default and tuned decision thresholds."""
        threshold = self.threshold if threshold is None else threshold
        probabilities = self.model.predict_proba(X_test)[:, 1]
        default_predictions = (probabilities >= 0.5).astype(int)
        tuned_predictions = (probabilities >= threshold).astype(int)

        print("\n=== Model Evaluation Report (default threshold = 0.5) ===")
        print(classification_report(y_test, default_predictions))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, probabilities):.4f}")
        print(f"PR-AUC Score:  {average_precision_score(y_test, probabilities):.4f}")

        print(f"\n=== Model Evaluation Report (tuned threshold = {threshold:.3f}) ===")
        print(classification_report(y_test, tuned_predictions))
        print(f"F1 (churn):    {f1_score(y_test, tuned_predictions):.4f}")
        print("============================================================\n")

    def generate_explainability(self, X_test):
        """Applies SHAP framework to explain model predictions."""
        print("Generating SHAP explainability values...")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(X_test)
        print("SHAP values calculated successfully.")
        return explainer, shap_values

    def save_artifact(self, filepath=DEFAULT_MODEL_PATH):
        """Saves the trained model and inference metadata to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model": self.model,
            "threshold": self.threshold,
            "feature_columns": self.feature_columns,
        }

        with open(filepath, "wb") as f:
            pickle.dump(artifact, f)
        print(f"Model artifact saved to {filepath}")


if __name__ == "__main__":
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise ValueError("MONGO_URI is not set. Add it to the .env file in the project root.")

    trainer = ModelTrainer(db_connection_string=MONGO_URI)

    try:
        df = trainer.fetch_data_from_db()
        if df.empty:
            print("MongoDB collection is empty. Using local processed CSV instead.")
            df = pd.read_csv(PROCESSED_DATA_PATH)
    except Exception as e:
        print(f"Could not fetch from MongoDB, checking local fallback. Error: {e}")
        df = pd.read_csv(PROCESSED_DATA_PATH)

    if not df.empty:
        X_train, X_val, X_test, y_train, y_val, y_test, class_weight = trainer.prepare_datasets(
            df, target_column=TARGET_COLUMN
        )

        trainer.run_cross_validation(X_train, y_train, class_weight)
        trainer.train_xgboost(X_train, y_train, X_val, y_val, class_weight, tune=True)
        trainer.find_optimal_threshold(X_val, y_val)
        trainer.evaluate_model(X_test, y_test)
        trainer.generate_explainability(X_test)
        trainer.save_artifact()
