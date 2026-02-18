"""
End-to-End Business Anomaly Intelligence Pipeline
=================================================

Production-ready pipeline with:
1. Full ML lifecycle management
2. Comprehensive monitoring
3. CI/CD integration hooks
4. Error handling & retry logic
5. Business reporting

Perfect for portfolio: Shows full-stack ML Ops understanding.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys
import traceback
from typing import Dict, Optional, Tuple, Any

# ===============================
# Import Your Project Modules
# ===============================

# Note: These imports assume your module structure
# You can comment them out for testing
try:
    from src.data.extract_metrics import extract_metrics_data
    from src.data.data_validation import DataValidator
    from src.features.build_features import FeatureBuilder
    from src.models.baseline_detector import BaselineDetector
    from src.models.ml_detector import MLDetector
    from src.evaluation.model_evaluator import BusinessAnomalyEvaluator
    from src.monitoring.data_monitor import BusinessDataMonitor
    from src.monitoring.model_monitor import BusinessModelMonitor
except ImportError:
    # Mock implementations for demonstration
    print("⚠️  Using mock implementations for demonstration")
    
    class MockExtractor:
        def __call__(self):
            dates = pd.date_range("2024-01-01", periods=100, freq='D')
            return pd.DataFrame({
                "metric_date": dates,
                "daily_revenue": 50000 + np.random.normal(0, 5000, 100),
                "transaction_count": np.random.randint(100, 1000, 100),
                "active_users": np.random.randint(500, 2000, 100)
            })
    
    extract_metrics_data = MockExtractor()
    
    # Mock other classes with simple implementations
    class DataValidator: 
        def run_validation(self, df): return {"passed": True}
    class FeatureBuilder: 
        def build_features(self, df): return df
    class BaselineDetector: 
        def run_detection(self, df): 
            df['baseline_anomaly'] = np.random.choice([0,1], len(df), p=[0.95,0.05])
            return df
    class MLDetector:
        def train(self, df): pass
        def predict(self, df): 
            df['ml_anomaly_flag'] = np.random.choice([0,1], len(df), p=[0.93,0.07])
            df['ml_anomaly_score'] = np.random.uniform(0, 100, len(df))
            return df
    class BusinessAnomalyEvaluator:
        def run_full_evaluation(self, df, **kwargs): return {"summary": "mock"}
    class BusinessDataMonitor:
        def __init__(self, **kwargs): pass
        def run_comprehensive_monitor(self, df): return {"status": "mock"}
    class BusinessModelMonitor:
        def __init__(self, **kwargs): pass
        def set_reference_data(self, df): pass
        def run_comprehensive_monitor(self, df): return {"status": "mock"}


# ===============================
# Enhanced Logging Setup
# ===============================

class PipelineLogger:
    """Custom logger for pipeline stages"""
    
    def __init__(self):
        self.logger = logging.getLogger("BusinessPipeline")
        
        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(stage)-15s | %(message)s",
            datefmt="%H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)
    
    def log_stage(self, stage_name: str, message: str, level: str = "info"):
        """Log with stage context"""
        extra = {'stage': stage_name}
        if level.lower() == "info":
            self.logger.info(message, extra=extra)
        elif level.lower() == "warning":
            self.logger.warning(message, extra=extra)
        elif level.lower() == "error":
            self.logger.error(message, extra=extra)
        elif level.lower() == "critical":
            self.logger.critical(message, extra=extra)


# ===============================
# Configuration
# ===============================

class PipelineConfig:
    """Centralized pipeline configuration"""
    
    def __init__(self):
        # Paths
        self.OUTPUT_DIR = Path("outputs")
        self.REPORTS_DIR = Path("reports")
        self.MODELS_DIR = Path("models")
        
        # Create directories
        for dir_path in [self.OUTPUT_DIR, self.REPORTS_DIR, self.MODELS_DIR]:
            dir_path.mkdir(exist_ok=True)
        
        # Pipeline settings
        self.TRAIN_TEST_SPLIT = 0.7
        self.ENABLE_MONITORING = True
        self.SAVE_OUTPUTS = True
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 5  # seconds
        
        # Business settings
        self.BUSINESS_METRICS = ["daily_revenue", "transaction_count", "active_users"]
        self.CRITICAL_THRESHOLDS = {
            "min_data_points": 50,
            "max_missing_pct": 0.1,
            "min_anomaly_rate": 0.01,
            "max_anomaly_rate": 0.1
        }


# ===============================
# Pipeline Status Tracker
# ===============================

class PipelineStatus:
    """Tracks pipeline execution status"""
    
    def __init__(self):
        self.stages = {}
        self.start_time = None
        self.end_time = None
        self.success = False
        self.errors = []
    
    def start_stage(self, stage_name: str):
        """Record stage start"""
        self.stages[stage_name] = {
            "start": datetime.now(),
            "end": None,
            "success": False,
            "duration": None
        }
    
    def end_stage(self, stage_name: str, success: bool = True, error: str = None):
        """Record stage completion"""
        if stage_name in self.stages:
            self.stages[stage_name]["end"] = datetime.now()
            self.stages[stage_name]["success"] = success
            self.stages[stage_name]["duration"] = (
                self.stages[stage_name]["end"] - self.stages[stage_name]["start"]
            ).total_seconds()
            
            if error:
                self.errors.append({"stage": stage_name, "error": error})
    
    def get_summary(self) -> Dict:
        """Generate pipeline summary"""
        successful_stages = sum(1 for s in self.stages.values() if s["success"])
        total_stages = len(self.stages)
        
        return {
            "pipeline_success": self.success,
            "stages_completed": f"{successful_stages}/{total_stages}",
            "total_duration_seconds": (
                (self.end_time - self.start_time).total_seconds() 
                if self.end_time and self.start_time else 0
            ),
            "stage_details": self.stages,
            "errors": self.errors
        }


# ===============================
# Pipeline Runner Class
# ===============================

class BusinessAnomalyIntelligencePipeline:
    """
    Complete business anomaly detection pipeline.
    
    Demonstrates:
    - Production ML pipeline design
    - Error handling and retry logic
    - Comprehensive monitoring
    - Business reporting
    - CI/CD integration hooks
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.logger = PipelineLogger()
        self.status = PipelineStatus()
        
        # Initialize components
        self._initialize_components()
        
        self.logger.log_stage("INIT", "Business Anomaly Intelligence Pipeline initialized")
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        try:
            self.validator = DataValidator()
            self.feature_builder = FeatureBuilder()
            self.baseline = BaselineDetector()
            self.ml_model = MLDetector()
            self.evaluator = BusinessAnomalyEvaluator()
            
            self.data_monitor = BusinessDataMonitor(
                expected_schema=["metric_date"] + self.config.BUSINESS_METRICS
            )
            
            self.model_monitor = BusinessModelMonitor()
            
            self.logger.log_stage("INIT", "All components initialized successfully")
            
        except Exception as e:
            self.logger.log_stage("INIT", f"Component initialization failed: {e}", "error")
            raise
    
    # ===============================
    # Stage 1 — Data Extraction
    # ===============================
    def stage_extract(self) -> pd.DataFrame:
        """Extract business metrics data"""
        stage_name = "EXTRACTION"
        self.status.start_stage(stage_name)
        
        try:
            self.logger.log_stage(stage_name, "Starting data extraction...")
            
            # Extract data
            df = extract_metrics_data()
            
            # Basic validation
            if df.empty:
                raise ValueError("No data extracted")
            
            self.logger.log_stage(stage_name, f"Extracted {len(df)} rows, {len(df.columns)} columns")
            self.logger.log_stage(stage_name, f"Date range: {df['metric_date'].min()} to {df['metric_date'].max()}")
            
            self.status.end_stage(stage_name, success=True)
            return df
            
        except Exception as e:
            error_msg = f"Extraction failed: {str(e)}"
            self.logger.log_stage(stage_name, error_msg, "error")
            self.status.end_stage(stage_name, success=False, error=error_msg)
            raise
    
    # ===============================
    # Stage 2 — Data Validation
    # ===============================
    def stage_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality and business rules"""
        stage_name = "VALIDATION"
        self.status.start_stage(stage_name)
        
        try:
            self.logger.log_stage(stage_name, "Starting data validation...")
            
            # Run validation
            report = self.validator.run_validation(df)
            
            # Check validation results
            if not report.get("passed", False):
                issues = report.get("issues", [])
                error_msg = f"Validation failed: {issues}"
                raise ValueError(error_msg)
            
            # Check business-critical metrics
            missing_business = [
                m for m in self.config.BUSINESS_METRICS 
                if m not in df.columns
            ]
            if missing_business:
                self.logger.log_stage(stage_name, 
                    f"Warning: Missing business metrics: {missing_business}", "warning")
            
            self.logger.log_stage(stage_name, "Data validation passed")
            self.status.end_stage(stage_name, success=True)
            return df
            
        except Exception as e:
            error_msg = f"Validation failed: {str(e)}"
            self.logger.log_stage(stage_name, error_msg, "error")
            self.status.end_stage(stage_name, success=False, error=error_msg)
            raise
    
    # ===============================
    # Stage 3 — Feature Engineering
    # ===============================
    def stage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for anomaly detection"""
        stage_name = "FEATURES"
        self.status.start_stage(stage_name)
        
        try:
            self.logger.log_stage(stage_name, "Starting feature engineering...")
            
            # Build features
            df_features = self.feature_builder.build_features(df)
            
            # Check feature creation
            original_cols = set(df.columns)
            new_cols = set(df_features.columns) - original_cols
            
            self.logger.log_stage(stage_name, 
                f"Created {len(new_cols)} new features, total features: {len(df_features.columns)}")
            
            self.status.end_stage(stage_name, success=True)
            return df_features
            
        except Exception as e:
            error_msg = f"Feature engineering failed: {str(e)}"
            self.logger.log_stage(stage_name, error_msg, "error")
            self.status.end_stage(stage_name, success=False, error=error_msg)
            raise
    
    # ===============================
    # Stage 4 — Baseline Detection
    # ===============================
    def stage_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run baseline statistical anomaly detection"""
        stage_name = "BASELINE"
        self.status.start_stage(stage_name)
        
        try:
            self.logger.log_stage(stage_name, "Starting baseline anomaly detection...")
            
            # Run baseline detection
            baseline_df = self.baseline.run_detection(df)
            
            # Check results
            if 'baseline_anomaly' not in baseline_df.columns:
                raise ValueError("Baseline detection failed to produce anomaly column")
            
            anomaly_rate = baseline_df['baseline_anomaly'].mean()
            self.logger.log_stage(stage_name, 
                f"Baseline anomaly rate: {anomaly_rate:.2%} ({baseline_df['baseline_anomaly'].sum()} anomalies)")
            
            self.status.end_stage(stage_name, success=True)
            return baseline_df
            
        except Exception as e:
            error_msg = f"Baseline detection failed: {str(e)}"
            self.logger.log_stage(stage_name, error_msg, "error")
            self.status.end_stage(stage_name, success=False, error=error_msg)
            raise
    
    # ===============================
    # Stage 5 — ML Detection
    # ===============================
    def stage_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """Train and run ML anomaly detection"""
        stage_name = "ML_MODEL"
        self.status.start_stage(stage_name)
        
        try:
            self.logger.log_stage(stage_name, "Starting ML anomaly detection...")
            
            # Train-test split
            split_idx = int(len(df) * self.config.TRAIN_TEST_SPLIT)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            
            self.logger.log_stage(stage_name, 
                f"Split: {len(train_df)} training, {len(test_df)} testing rows")
            
            # Train model
            self.logger.log_stage(stage_name, "Training ML model...")
            self.ml_model.train(train_df)
            
            # Make predictions
            self.logger.log_stage(stage_name, "Making ML predictions...")
            predictions = self.ml_model.predict(test_df)
            
            # Check ML results
            if 'ml_anomaly_flag' not in predictions.columns:
                raise ValueError("ML model failed to produce anomaly predictions")
            
            ml_anomaly_rate = predictions['ml_anomaly_flag'].mean()
            self.logger.log_stage(stage_name, 
                f"ML anomaly rate: {ml_anomaly_rate:.2%} ({predictions['ml_anomaly_flag'].sum()} anomalies)")
            
            self.status.end_stage(stage_name, success=True)
            return predictions
            
        except Exception as e:
            error_msg = f"ML detection failed: {str(e)}"
            self.logger.log_stage(stage_name, error_msg, "error")
            self.status.end_stage(stage_name, success=False, error=error_msg)
            raise
    
    # ===============================
    # Stage 6 — Evaluation
    # ===============================
    def stage_evaluation(self, df: pd.DataFrame) -> Dict:
        """Evaluate model performance"""
        stage_name = "EVALUATION"
        self.status.start_stage(stage_name)
        
        try:
            self.logger.log_stage(stage_name, "Starting model evaluation...")
            
            # Run evaluation
            eval_results = self.evaluator.run_full_evaluation(
                df, 
                create_plots=True,
                plot_save_path=str(self.config.REPORTS_DIR / "evaluation_plots.png")
            )
            
            # Log key metrics
            summary = eval_results.get("summary", {})
            self.logger.log_stage(stage_name, 
                f"Evaluation complete: {summary.get('overall_status', 'UNKNOWN')}")
            
            self.status.end_stage(stage_name, success=True)
            return eval_results
            
        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            self.logger.log_stage(stage_name, error_msg, "error")
            self.status.end_stage(stage_name, success=False, error=error_msg)
            raise
    
    # ===============================
    # Stage 7 — Monitoring
    # ===============================
    def stage_monitoring(self, df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Run comprehensive monitoring"""
        stage_name = "MONITORING"
        self.status.start_stage(stage_name)
        
        if not self.config.ENABLE_MONITORING:
            self.logger.log_stage(stage_name, "Monitoring disabled by config", "warning")
            self.status.end_stage(stage_name, success=True)
            return {}, {}
        
        try:
            self.logger.log_stage(stage_name, "Starting comprehensive monitoring...")
            
            # Data quality monitoring
            self.logger.log_stage(stage_name, "Running data quality monitoring...")
            data_report = self.data_monitor.run_comprehensive_monitor(df)
            
            # Model performance monitoring
            self.logger.log_stage(stage_name, "Running model performance monitoring...")
            self.model_monitor.set_reference_data(df)
            model_report = self.model_monitor.run_comprehensive_monitor(df)
            
            self.logger.log_stage(stage_name, "Monitoring complete")
            self.status.end_stage(stage_name, success=True)
            return data_report, model_report
            
        except Exception as e:
            error_msg = f"Monitoring failed: {str(e)}"
            self.logger.log_stage(stage_name, error_msg, "error")
            self.status.end_stage(stage_name, success=False, error=error_msg)
            raise
    
    # ===============================
    # CI/CD Integration Hooks
    # ===============================
    
    def _run_ci_checks(self, df: pd.DataFrame, eval_results: Dict) -> bool:
        """Run CI/CD quality checks"""
        self.logger.log_stage("CI", "Running CI/CD quality checks...")
        
        checks_passed = True
        checks = []
        
        # Check 1: Minimum data points
        if len(df) < self.config.CRITICAL_THRESHOLDS["min_data_points"]:
            checks.append({"check": "min_data_points", "passed": False})
            checks_passed = False
        else:
            checks.append({"check": "min_data_points", "passed": True})
        
        # Check 2: Anomaly rate within bounds
        if 'ml_anomaly_flag' in df.columns:
            anomaly_rate = df['ml_anomaly_flag'].mean()
            if (anomaly_rate < self.config.CRITICAL_THRESHOLDS["min_anomaly_rate"] or 
                anomaly_rate > self.config.CRITICAL_THRESHOLDS["max_anomaly_rate"]):
                checks.append({"check": "anomaly_rate", "passed": False, "rate": anomaly_rate})
                checks_passed = False
            else:
                checks.append({"check": "anomaly_rate", "passed": True, "rate": anomaly_rate})
        
        # Check 3: Evaluation metrics
        if eval_results and "summary" in eval_results:
            summary = eval_results["summary"]
            if summary.get("overall_status") == "CRITICAL":
                checks.append({"check": "evaluation_status", "passed": False})
                checks_passed = False
            else:
                checks.append({"check": "evaluation_status", "passed": True})
        
        self.logger.log_stage("CI", f"CI Checks: {sum(c['passed'] for c in checks)}/{len(checks)} passed")
        return checks_passed
    
    # ===============================
    # Save Outputs
    # ===============================
    def save_outputs(self, df: pd.DataFrame, eval_results: Dict, 
                    data_report: Dict, model_report: Dict):
        """Save all pipeline outputs"""
        if not self.config.SAVE_OUTPUTS:
            self.logger.log_stage("SAVE", "Output saving disabled by config")
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{timestamp}"
            
            self.logger.log_stage("SAVE", f"Saving outputs for run: {run_id}")
            
            # Save predictions
            predictions_path = self.config.OUTPUT_DIR / f"{run_id}_predictions.csv"
            df.to_csv(predictions_path, index=False)
            
            # Save evaluation report
            eval_path = self.config.REPORTS_DIR / f"{run_id}_evaluation.json"
            with open(eval_path, 'w') as f:
                json.dump(eval_results, f, indent=2, default=str)
            
            # Save monitoring reports
            if data_report:
                data_path = self.config.REPORTS_DIR / f"{run_id}_data_monitor.json"
                with open(data_path, 'w') as f:
                    json.dump(data_report, f, indent=2, default=str)
            
            if model_report:
                model_path = self.config.REPORTS_DIR / f"{run_id}_model_monitor.json"
                with open(model_path, 'w') as f:
                    json.dump(model_report, f, indent=2, default=str)
            
            # Save pipeline status
            status_path = self.config.REPORTS_DIR / f"{run_id}_pipeline_status.json"
            with open(status_path, 'w') as f:
                json.dump(self.status.get_summary(), f, indent=2, default=str)
            
            self.logger.log_stage("SAVE", f"Outputs saved to: {self.config.OUTPUT_DIR}")
            
        except Exception as e:
            self.logger.log_stage("SAVE", f"Failed to save outputs: {e}", "warning")
    
    # ===============================
    # Generate Business Report
    # ===============================
    def generate_business_report(self, df: pd.DataFrame, eval_results: Dict) -> str:
        """Generate business-friendly summary report"""
        
        report_lines = [
            "=" * 70,
            "BUSINESS ANOMALY INTELLIGENCE - EXECUTIVE SUMMARY",
            "=" * 70,
            f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ""
        ]
        
        # Data Summary
        report_lines.extend([
            "📊 DATA SUMMARY",
            "-" * 40,
            f"• Total Days Analyzed: {len(df)}",
            f"• Date Range: {df['metric_date'].min()} to {df['metric_date'].max()}",
        ])
        
        # Anomaly Detection Summary
        if 'ml_anomaly_flag' in df.columns:
            ml_anomalies = df['ml_anomaly_flag'].sum()
            ml_rate = df['ml_anomaly_flag'].mean() * 100
            
            report_lines.extend([
                "",
                "🎯 ANOMALY DETECTION RESULTS",
                "-" * 40,
                f"• ML Model Anomalies: {ml_anomalies} ({ml_rate:.1f}%)",
            ])
        
        # Business Impact (if available)
        if 'daily_revenue' in df.columns and 'ml_anomaly_flag' in df.columns:
            anomaly_rev = df[df['ml_anomaly_flag'] == 1]['daily_revenue'].mean()
            normal_rev = df[df['ml_anomaly_flag'] == 0]['daily_revenue'].mean()
            
            if pd.notna(anomaly_rev) and pd.notna(normal_rev):
                impact_pct = ((anomaly_rev - normal_rev) / normal_rev) * 100
                report_lines.extend([
                    "",
                    "💰 BUSINESS IMPACT",
                    "-" * 40,
                    f"• Avg Revenue (Normal Days): ${normal_rev:,.0f}",
                    f"• Avg Revenue (Anomaly Days): ${anomaly_rev:,.0f}",
                    f"• Revenue Impact: {impact_pct:+.1f}%",
                ])
        
        # Evaluation Summary
        if eval_results and "summary" in eval_results:
            summary = eval_results["summary"]
            report_lines.extend([
                "",
                "📈 MODEL PERFORMANCE",
                "-" * 40,
                f"• Overall Status: {summary.get('overall_status', 'N/A')}",
                f"• Checks Passed: {summary.get('checks_passed', 'N/A')}/{summary.get('total_checks', 'N/A')}",
            ])
        
        report_lines.extend([
            "",
            "=" * 70,
            "END OF REPORT",
            "=" * 70,
        ])
        
        return "\n".join(report_lines)
    
    # ===============================
    # Run Full Pipeline
    # ===============================
    def run(self, max_retries: int = None) -> Dict:
        """
        Execute complete pipeline with error handling and retries.
        
        Returns:
            Dict with pipeline results and status
        """
        max_retries = max_retries or self.config.MAX_RETRIES
        
        self.status.start_time = datetime.now()
        self.logger.log_stage("PIPELINE", "=" * 60, "info")
        self.logger.log_stage("PIPELINE", "STARTING BUSINESS ANOMALY INTELLIGENCE PIPELINE", "info")
        self.logger.log_stage("PIPELINE", "=" * 60, "info")
        
        retry_count = 0
        while retry_count <= max_retries:
            try:
                # Execute pipeline stages
                df = self.stage_extract()
                df = self.stage_validate(df)
                df = self.stage_features(df)
                df = self.stage_baseline(df)
                df = self.stage_ml(df)
                
                eval_results = self.stage_evaluation(df)
                data_report, model_report = self.stage_monitoring(df)
                
                # Run CI/CD checks
                ci_passed = self._run_ci_checks(df, eval_results)
                
                if not ci_passed:
                    self.logger.log_stage("PIPELINE", "CI/CD checks failed", "warning")
                
                # Save outputs
                self.save_outputs(df, eval_results, data_report, model_report)
                
                # Update status
                self.status.end_time = datetime.now()
                self.status.success = True
                
                # Generate business report
                business_report = self.generate_business_report(df, eval_results)
                
                self.logger.log_stage("PIPELINE", "=" * 60, "info")
                self.logger.log_stage("PIPELINE", "PIPELINE COMPLETED SUCCESSFULLY", "info")
                self.logger.log_stage("PIPELINE", "=" * 60, "info")
                
                return {
                    "success": True,
                    "predictions": df,
                    "evaluation": eval_results,
                    "data_monitor": data_report,
                    "model_monitor": model_report,
                    "pipeline_status": self.status.get_summary(),
                    "business_report": business_report,
                    "ci_passed": ci_passed
                }
                
            except Exception as e:
                retry_count += 1
                error_trace = traceback.format_exc()
                
                self.logger.log_stage("PIPELINE", 
                    f"Pipeline failed (attempt {retry_count}/{max_retries}): {str(e)}", 
                    "error")
                
                if retry_count <= max_retries:
                    self.logger.log_stage("PIPELINE", 
                        f"Retrying in {self.config.RETRY_DELAY} seconds...", 
                        "warning")
                    import time
                    time.sleep(self.config.RETRY_DELAY)
                else:
                    self.status.end_time = datetime.now()
                    self.status.success = False
                    
                    self.logger.log_stage("PIPELINE", 
                        f"Pipeline failed after {max_retries} retries", 
                        "critical")
                    
                    return {
                        "success": False,
                        "error": str(e),
                        "error_trace": error_trace,
                        "pipeline_status": self.status.get_summary()
                    }
        
        # Should never reach here
        return {"success": False, "error": "Maximum retries exceeded"}


# ===============================
# Command Line Interface
# ===============================

def main():
    """Main entry point with command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Business Anomaly Intelligence Pipeline"
    )
    parser.add_argument("--skip-monitoring", action="store_true",
                       help="Skip monitoring stages")
    parser.add_argument("--skip-save", action="store_true",
                       help="Skip saving outputs")
    parser.add_argument("--quick", action="store_true",
                       help="Quick run with minimal processing")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory path")
    
    args = parser.parse_args()
    
    # Configure pipeline
    config = PipelineConfig()
    config.ENABLE_MONITORING = not args.skip_monitoring
    config.SAVE_OUTPUTS = not args.skip_save
    config.OUTPUT_DIR = Path(args.output_dir)
    
    if args.quick:
        config.TRAIN_TEST_SPLIT = 0.5
        config.MAX_RETRIES = 1
    
    # Run pipeline
    pipeline = BusinessAnomalyIntelligencePipeline(config)
    results = pipeline.run()
    
    # Display results
    if results["success"]:
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION SUCCESSFUL")
        print("=" * 70)
        
        # Print business report
        print(results.get("business_report", ""))
        
        # Print summary
        status = results.get("pipeline_status", {})
        print(f"\n📊 Pipeline Summary:")
        print(f"   Status: {'✅ SUCCESS' if status.get('pipeline_success') else '❌ FAILED'}")
        print(f"   Stages: {status.get('stages_completed', 'N/A')}")
        print(f"   Duration: {status.get('total_duration_seconds', 0):.1f} seconds")
        print(f"   CI Checks: {'✅ PASSED' if results.get('ci_passed') else '❌ FAILED'}")
        
        # Show sample predictions
        df = results.get("predictions")
        if df is not None:
            print(f"\n🔍 Sample Predictions (first 5):")
            print(df[['metric_date', 'ml_anomaly_flag', 'ml_anomaly_score']].head())
            
        return 0  # Success exit code
    else:
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION FAILED")
        print("=" * 70)
        print(f"Error: {results.get('error', 'Unknown error')}")
        return 1  # Error exit code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)