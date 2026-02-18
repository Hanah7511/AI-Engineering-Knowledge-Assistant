import pandas as pd
import numpy as np
import logging
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Optional

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler



# LOGGING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("MLDetector")



# CONFIG

@dataclass
class MLDetectorConfig:
    contamination: float = 0.03
    n_estimators: int = 150
    random_state: int = 42



# DETECTOR CLASS

class MLAnomalyDetector:

    def __init__(self, config: Optional[MLDetectorConfig] = None):
        self.config = config or MLDetectorConfig()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = None
        self.is_trained = False

        logger.info("ML Detector initialized")
        logger.info(f"Config: {self.config}")

    
    # FEATURE PREPARATION
    
    def _prepare_features(self, df: pd.DataFrame, fit: bool = True):

        df = df.copy()

        # Basic numeric selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target / flags if exist
        exclude = ["ml_anomaly_flag", "ml_anomaly_score"]
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        if fit:
            self.feature_columns = numeric_cols

        X = df[self.feature_columns].copy()

        # Handle missing values
        X = X.fillna(X.mean())

        # Scaling
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled, df

    
    # TRAIN MODEL
    
    def fit(self, df: pd.DataFrame):

        logger.info("Training ML anomaly model...")

        X_scaled, _ = self._prepare_features(df, fit=True)

        self.model = IsolationForest(
            contamination=self.config.contamination,
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
            n_jobs=-1
        )

        self.model.fit(X_scaled)

        self.is_trained = True
        logger.info("Model training complete")

    
    # PREDICT
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:

        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        logger.info("Running anomaly prediction...")

        X_scaled, df_out = self._prepare_features(df, fit=False)

        preds = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)

        df_out["ml_anomaly_flag"] = (preds == -1).astype(int)

        # Normalize score to 0–100 anomaly scale
        df_out["ml_anomaly_score"] = 100 * (
            1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        )

        anomaly_count = df_out["ml_anomaly_flag"].sum()

        logger.info(f"Detected {anomaly_count} anomalies ({anomaly_count/len(df_out):.2%})")

        return df_out

    
    # TRAIN + PREDICT PIPELINE
    
    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("=" * 60)
        logger.info("RUNNING ML ANOMALY PIPELINE")
        logger.info("=" * 60)

        self.fit(df)
        result = self.predict(df)

        logger.info("Pipeline complete")

        return result

    
    # SAVE MODEL
    
    def save_model(self, path="models/ml_detector.pkl"):

        package = {
            "model": self.model,
            "scaler": self.scaler,
            "features": self.feature_columns,
            "config": self.config
        }

        with open(path, "wb") as f:
            pickle.dump(package, f)

        logger.info(f"Model saved → {path}")

    
    # LOAD MODEL
    
    def load_model(self, path="models/ml_detector.pkl"):

        with open(path, "rb") as f:
            package = pickle.load(f)

        self.model = package["model"]
        self.scaler = package["scaler"]
        self.feature_columns = package["features"]
        self.config = package["config"]
        self.is_trained = True

        logger.info(f"Model loaded ← {path}")



# TEST MODE

if __name__ == "__main__":

    logger.info("Test Mode Running")

    np.random.seed(42)

    dates = pd.date_range("2024-01-01", periods=120)

    revenue = 30000 + np.random.normal(0, 3000, 120)
    revenue[40] = 60000
    revenue[90] = 8000

    df = pd.DataFrame({
        "metric_date": dates,
        "daily_revenue": revenue,
        "transaction_count": revenue / 150 + np.random.normal(0, 20, 120)
    })

    detector = MLAnomalyDetector()

    result = detector.run(df)

    print(result[["daily_revenue", "ml_anomaly_flag", "ml_anomaly_score"]].head())


# ==========================================
# PIPELINE COMPATIBILITY ADAPTER
# ==========================================

from sklearn.ensemble import IsolationForest


class MLDetector:
    """
    Pipeline Adapter → Uses Isolation Forest internally.
    Moderate complexity, portfolio-ready.
    """

    def __init__(self, contamination=0.03, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=150
        )
        self.trained = False

    def train(self, df):
        """
        Train ML anomaly model.
        """

        feature_cols = df.select_dtypes(include=["number"]).columns.tolist()

        # Remove target / label columns if exist
        drop_cols = [
            "baseline_anomaly",
            "ml_anomaly_flag",
            "ml_anomaly_score"
        ]

        feature_cols = [c for c in feature_cols if c not in drop_cols]

        X = df[feature_cols].fillna(0)

        self.model.fit(X)
        self.trained = True

    def predict(self, df):
        """
        Predict anomalies.
        Returns dataframe with:
        - ml_anomaly_flag
        - ml_anomaly_score
        """

        if not self.trained:
            raise RuntimeError("ML model must be trained before prediction")

        feature_cols = df.select_dtypes(include=["number"]).columns.tolist()

        drop_cols = [
            "baseline_anomaly",
            "ml_anomaly_flag",
            "ml_anomaly_score"
        ]

        feature_cols = [c for c in feature_cols if c not in drop_cols]

        X = df[feature_cols].fillna(0)

        preds = self.model.predict(X)
        scores = self.model.decision_function(X)

        df = df.copy()
        df["ml_anomaly_flag"] = (preds == -1).astype(int)
        df["ml_anomaly_score"] = -scores  # Higher = more anomalous

        return df
