"""
End-to-End Business Anomaly Detection Pipeline
==============================================

Runs full pipeline:
Extraction → Validation → Features → Baseline → ML → Evaluation → Monitoring

Industry-Style (Moderate Complexity)
- Structured logging
- Clear stage separation
- CI/CD ready

"""
import os
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

# ===============================
# Import Your Project Modules
# ===============================

def stage_extract(self):
    if os.getenv("CI") == "true":
        return self._generate_sample_data()
    from src.data.extract_metrics import extract_metrics_data
    return extract_metrics_data() 
from src.data.data_validation import DataValidator
from src.features.build_features import FeatureBuilder
from src.models.baseline_detector import BaselineDetector
from src.models.ml_detector import MLDetector
from src.evaluation.model_evaluator import BusinessAnomalyEvaluator
from src.monitoring.data_monitor import BusinessDataMonitor
from src.monitoring.model_monitor import BusinessModelMonitor


# ===============================
# Logging Setup
# ===============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | PIPELINE | %(message)s"
)

logger = logging.getLogger("PipelineRunner")


# ===============================
# Output Paths
# ===============================

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ===============================
# Pipeline Runner Class
# ===============================

class BusinessAnomalyPipeline:

    def __init__(self):
        logger.info("Initializing Business Anomaly Pipeline")

        self.validator = DataValidator()
        self.feature_builder = FeatureBuilder()
        self.baseline = BaselineDetector()
        self.ml_model = MLDetector()
        self.evaluator = BusinessAnomalyEvaluator()

        self.data_monitor = BusinessDataMonitor(
            expected_schema=[
                "metric_date",
                "daily_revenue",
                "transaction_count"
            ]
        )

        self.model_monitor = BusinessModelMonitor()

    # ===============================
    # Stage 1 — Extract
    # ===============================
    def stage_extract(self):
        logger.info("STAGE 1 — DATA EXTRACTION")

    # CI Mode → use synthetic data
        if os.getenv("CI") == "true":
            logger.info("Running in CI mode — using synthetic data")
            df = self._generate_sample_data()
        else:
            df = extract_metrics_data()

        logger.info(f"Extracted rows: {len(df)}")
        return df
    
    def _generate_sample_data(self):
        import pandas as pd
        import numpy as np

        dates = pd.date_range("2024-01-01",periods=60)

        df = pd.DataFrame({
            "metric_date": dates,
            "daily_revenue": 20000 + np.random.normal(0, 2000, 60),
            "successful_payments": np.random.randint(200, 400, 60),
            "failed_payments": np.random.randint(10, 40, 60),
        })
        return df

    # ===============================
    # Stage 2 — Validate
    # ===============================
    def stage_validate(self, df):
        logger.info("STAGE 2 — DATA VALIDATION")
        report = self.validator.run_validation(df)

        if not report["passed"]:
            raise ValueError("Data validation failed")

        return df

    # ===============================
    # Stage 3 — Feature Engineering
    # ===============================
    def stage_features(self, df):
        logger.info("STAGE 3 — FEATURE ENGINEERING")
        df_features = self.feature_builder.build_features(df)
        return df_features

    # ===============================
    # Stage 4 — Baseline Detection
    # ===============================
    def stage_baseline(self, df):
        logger.info("STAGE 4 — BASELINE DETECTION")
        baseline_df = self.baseline.run_detection(df)
        return baseline_df

    # ===============================
    # Stage 5 — ML Detection
    # ===============================
    def stage_ml(self, df):
        logger.info("STAGE 5 — ML DETECTION")

        train_df = df.iloc[: int(len(df) * 0.7)]
        test_df = df.iloc[int(len(df) * 0.7):]

        self.ml_model.train(train_df)
        predictions = self.ml_model.predict(test_df)

        return predictions

    # ===============================
    # Stage 6 — Evaluation
    # ===============================
    def stage_evaluation(self, df):
        logger.info("STAGE 6 — MODEL EVALUATION")
        results = self.evaluator.run_full_evaluation(df, create_plots=False)
        return results

    # ===============================
    # Stage 7 — Monitoring
    # ===============================
    def stage_monitoring(self, df):
        logger.info("STAGE 7 — MONITORING")

        data_report = self.data_monitor.run_comprehensive_monitor(df)

        self.model_monitor.set_reference_data(df)
        model_report = self.model_monitor.run_comprehensive_monitor(df)

        return data_report, model_report

    # ===============================
    # Save Outputs
    # ===============================
    def save_outputs(self, df, eval_results, data_report, model_report):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        df.to_csv(OUTPUT_DIR / f"pipeline_predictions_{timestamp}.csv", index=False)

        pd.DataFrame([eval_results]).to_json(
            OUTPUT_DIR / f"evaluation_{timestamp}.json",
            orient="records"
        )

        pd.DataFrame([data_report]).to_json(
            OUTPUT_DIR / f"data_monitor_{timestamp}.json",
            orient="records"
        )

        pd.DataFrame([model_report]).to_json(
            OUTPUT_DIR / f"model_monitor_{timestamp}.json",
            orient="records"
        )

        logger.info("Outputs saved successfully")

    # ===============================
    # Run Full Pipeline
    # ===============================
    def run(self):

        logger.info("=" * 60)
        logger.info("STARTING FULL BUSINESS PIPELINE")
        logger.info("=" * 60)

        df = self.stage_extract()
        df = self.stage_validate(df)
        df = self.stage_features(df)
        df = self.stage_baseline(df)
        df = self.stage_ml(df)

        eval_results = self.stage_evaluation(df)
        data_report, model_report = self.stage_monitoring(df)

        self.save_outputs(df, eval_results, data_report, model_report)

        logger.info("PIPELINE COMPLETED SUCCESSFULLY")

        return df


# ===============================
# Main Entry Point
# ===============================

def main():
    pipeline = BusinessAnomalyPipeline()
    results = pipeline.run()

    print("\nPipeline finished successfully")
    print(results.head())

    return results


if __name__ == "__main__":
    main()
