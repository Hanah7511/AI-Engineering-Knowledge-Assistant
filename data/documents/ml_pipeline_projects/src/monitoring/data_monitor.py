"""
Production Data Health Monitor for Business Intelligence
========================================================

"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("BusinessDataMonitor")


# ----------------------------
# Configuration with Business Context
# ----------------------------
class DataMonitorConfig:
    """Business-aware monitoring thresholds"""
    
    def __init__(
        self,
        missing_threshold: float = 0.05,      # 5% missing allowed
        volume_drop_threshold: float = 0.2,   # 20% drop triggers alert
        drift_zscore_threshold: float = 3.0,  # 3 std deviations = drift
        freshness_hours_threshold: int = 48,  # 2 days old data
        business_metric_priority: List[str] = None
    ):
        self.missing_threshold = missing_threshold
        self.volume_drop_threshold = volume_drop_threshold
        self.drift_zscore_threshold = drift_zscore_threshold
        self.freshness_hours_threshold = freshness_hours_threshold
        
        # Business-critical metrics to monitor closely
        self.business_metric_priority = business_metric_priority or [
            'daily_revenue', 'transaction_count', 'active_users'
        ]
        
        logger.info(f"Monitor config: Missing <{missing_threshold*100}%, Volume drop <{volume_drop_threshold*100}%")
    
    def get_severity_levels(self) -> Dict:
        """Define severity for different issues"""
        return {
            'CRITICAL': ['schema_invalid', 'key_metric_missing', 'volume_drop_50pct'],
            'HIGH': ['drift_detected', 'freshness_breach', 'missing_above_threshold'],
            'MEDIUM': ['schema_warning', 'volume_drop_20pct'],
            'LOW': ['extra_columns', 'minor_missing']
        }


# ----------------------------
# Business Data Monitor
# ----------------------------
class BusinessDataMonitor:
    """
    Production data quality monitor with business context.
    
    Monitors:
    1. Schema & Structure - Is data in right format?
    2. Completeness - Are values missing?
    3. Volume & Freshness - Is data arriving on time?
    4. Statistical Health - Has data distribution changed?
    5. Business Rules - Do metrics make business sense?
    """
    
    def __init__(
        self,
        expected_schema: List[str],
        config: Optional[DataMonitorConfig] = None,
        monitor_id: str = "business_monitor"
    ):
        self.expected_schema = expected_schema
        self.config = config or DataMonitorConfig()
        self.monitor_id = monitor_id
        self.history = []  # Store monitoring history
        
        # Initialize reference statistics (empty until trained)
        self.reference_stats = None
        self.reference_volume = None
        
        logger.info(f"Business Data Monitor '{monitor_id}' initialized")
        logger.info(f"Priority metrics: {self.config.business_metric_priority}")
    
    # ----------------------------
    #  SCHEMA VALIDATION 
    # ----------------------------

    def check_schema(self, df: pd.DataFrame) -> Dict:
        """
        Validate data schema with business context.
        
        Returns:
            Dict with validation results and severity
        """
        logger.info(" Validating data schema...")
        
        missing_cols = list(set(self.expected_schema) - set(df.columns))
        extra_cols = list(set(df.columns) - set(self.expected_schema))
        
        # Check for business-critical columns
        critical_missing = [
            col for col in missing_cols 
            if col in self.config.business_metric_priority
        ]
        
        is_valid = len(missing_cols) == 0
        severity = "CRITICAL" if critical_missing else "HIGH" if missing_cols else "PASS"
        
        return {
            "check": "schema_validation",
            "timestamp": datetime.now().isoformat(),
            "valid": is_valid,
            "severity": severity,
            "missing_columns": missing_cols,
            "critical_missing": critical_missing,
            "extra_columns": extra_cols,
            "expected_columns": len(self.expected_schema),
            "actual_columns": len(df.columns)
        }
    
    # ----------------------------
    # DATA COMPLETENESS
    # ----------------------------

    def check_completeness(self, df: pd.DataFrame) -> Dict:
        """
        Check for missing values with business priority.
        """
        logger.info(" Checking data completeness...")
        
        missing_counts = df.isna().sum()
        missing_pct = (missing_counts / len(df)).to_dict()
        
        # Categorize missingness
        critical_missing = {}
        warning_missing = {}
        
        for col, pct in missing_pct.items():
            if pct > self.config.missing_threshold:
                if col in self.config.business_metric_priority:
                    critical_missing[col] = round(pct * 100, 2)
                else:
                    warning_missing[col] = round(pct * 100, 2)
        
        total_missing_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
        
        severity = "CRITICAL" if critical_missing else "HIGH" if warning_missing else "PASS"
        
        return {
            "check": "data_completeness",
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "total_missing_percent": round(total_missing_pct, 2),
            "critical_missing": critical_missing,      # Business metrics > threshold
            "warning_missing": warning_missing,        # Other metrics > threshold
            "all_missing_percent": {k: round(v*100, 2) for k, v in missing_pct.items()}
        }
    
    # ----------------------------
    #  VOLUME & FRESHNESS 
    # ----------------------------

    def check_volume_and_freshness(self, df: pd.DataFrame, date_col: str = "metric_date") -> Dict:
        """
        Monitor data volume and freshness for business continuity.
        """
        logger.info(" Checking data volume and freshness...")
        
        results = {
            "check": "volume_freshness",
            "timestamp": datetime.now().isoformat()
        }
        
        # Volume check
        current_rows = len(df)
        
        if self.reference_volume:
            volume_drop_pct = max(0, (self.reference_volume - current_rows) / self.reference_volume)
            results["volume_drop_percent"] = round(volume_drop_pct * 100, 2)
            results["volume_alert"] = volume_drop_pct > self.config.volume_drop_threshold
            results["reference_volume"] = self.reference_volume
        else:
            results["volume_alert"] = False
            results["note"] = "No reference volume set (first run)"
        
        results["current_rows"] = current_rows
        
        # Freshness check
        if date_col in df.columns:
            try:
                latest_date = pd.to_datetime(df[date_col]).max()
                now = pd.Timestamp.now()
                
                age_hours = (now - latest_date).total_seconds() / 3600
                is_stale = age_hours > self.config.freshness_hours_threshold
                
                results.update({
                    "latest_timestamp": latest_date.isoformat(),
                    "data_age_hours": round(age_hours, 2),
                    "is_stale": is_stale,
                    "freshness_threshold_hours": self.config.freshness_hours_threshold
                })
                
                severity = "HIGH" if is_stale else "PASS"
            except Exception as e:
                results["freshness_error"] = str(e)
                severity = "MEDIUM"
        else:
            results["freshness_check"] = "skipped_no_date_column"
            severity = "MEDIUM"
        
        results["severity"] = severity
        return results
    
    # ----------------------------
    #  STATISTICAL DRIFT 
    # ----------------------------

    def detect_statistical_drift(self, df: pd.DataFrame) -> Dict:
        """
        Detect statistical drift in key business metrics.
        """
        logger.info(" Detecting statistical drift...")
        
        if self.reference_stats is None:
            return {
                "check": "statistical_drift",
                "timestamp": datetime.now().isoformat(),
                "status": "no_reference_data",
                "severity": "INFO"
            }
        
        drift_results = {}
        detected_drifts = []
        
        # Focus on business-critical numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        priority_cols = [col for col in numeric_cols 
                        if col in self.config.business_metric_priority]
        
        for col in priority_cols:
            if col not in self.reference_stats:
                continue
            
            current_mean = df[col].mean()
            ref_mean = self.reference_stats[col]["mean"]
            ref_std = self.reference_stats[col]["std"] + 1e-8
            
            z_score = abs((current_mean - ref_mean) / ref_std)
            drift_detected = z_score > self.config.drift_zscore_threshold
            
            drift_results[col] = {
                "current_mean": float(round(current_mean, 2)),
                "reference_mean": float(round(ref_mean, 2)),
                "percent_change": float(round(((current_mean - ref_mean) / ref_mean * 100), 2)),
                "z_score": float(round(z_score, 2)),
                "drift_detected": drift_detected,
                "threshold": self.config.drift_zscore_threshold
            }
            
            if drift_detected:
                detected_drifts.append(col)
        
        severity = "HIGH" if detected_drifts else "PASS"
        
        return {
            "check": "statistical_drift",
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "drift_detected": detected_drifts,
            "metrics_checked": priority_cols,
            "detailed_results": drift_results
        }
    
    # ----------------------------
    #  BUSINESS RULES 
    # ----------------------------

    def check_business_rules(self, df: pd.DataFrame) -> Dict:
        """
        Apply business logic rules to detect illogical data.
        """
        logger.info(" Applying business rules...")
        
        violations = []
        
        # Rule 1: Revenue should be positive
        if 'daily_revenue' in df.columns:
            negative_revenue = df[df['daily_revenue'] < 0]
            if len(negative_revenue) > 0:
                violations.append({
                    "rule": "revenue_positive",
                    "violations": len(negative_revenue),
                    "description": "Revenue should not be negative"
                })
        
        # Rule 2: Success rate between 0 and 1
        if 'success_rate' in df.columns:
            invalid_success = df[(df['success_rate'] < 0) | (df['success_rate'] > 1)]
            if len(invalid_success) > 0:
                violations.append({
                    "rule": "success_rate_range",
                    "violations": len(invalid_success),
                    "description": "Success rate must be between 0 and 1"
                })
        
        # Rule 3: Transaction count >= successful transactions
        if all(col in df.columns for col in ['transaction_count', 'successful_transactions']):
            invalid_pairs = df[df['transaction_count'] < df['successful_transactions']]
            if len(invalid_pairs) > 0:
                violations.append({
                    "rule": "transaction_logic",
                    "violations": len(invalid_pairs),
                    "description": "Successful transactions cannot exceed total transactions"
                })
        
        severity = "HIGH" if violations else "PASS"
        
        return {
            "check": "business_rules",
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "violations": violations,
            "rules_applied": [
                "revenue_positive",
                "success_rate_range", 
                "transaction_logic"
            ]
        }
    
    # ----------------------------
    #  REFERENCE DATA SETUP 
    # ----------------------------

    def set_reference_data(self, df: pd.DataFrame):
        """
        Set current data as reference for future comparisons.
        """
        logger.info(" Setting reference data...")
        
        # Store reference volume
        self.reference_volume = len(df)
        
        # Calculate reference statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.reference_stats = {}
        
        for col in numeric_cols:
            if col in self.config.business_metric_priority:
                self.reference_stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "median": float(df[col].median())
                }
        
        logger.info(f"Reference set: {self.reference_volume} rows, {len(self.reference_stats)} metrics")

    # ----------------------------
    #  COMPREHENSIVE MONITORING 
    # ----------------------------

    def run_comprehensive_monitor(self, df: pd.DataFrame) -> Dict:
        """
        Run all monitoring checks and generate business report.
        """
        logger.info("=" * 70)
        logger.info("RUNNING BUSINESS DATA MONITOR")
        logger.info("=" * 70)
        
        # Run all checks
        checks = {
            "schema": self.check_schema(df),
            "completeness": self.check_completeness(df),
            "volume_freshness": self.check_volume_and_freshness(df),
            "statistical_drift": self.detect_statistical_drift(df),
            "business_rules": self.check_business_rules(df)
        }
        
        # Generate summary
        summary = self._generate_summary(checks)
        
        # Store in history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "checks": checks
        })
        
        # Log results
        self._log_monitoring_results(summary, checks)
        
        logger.info("=" * 70)
        logger.info("MONITORING COMPLETE")
        logger.info("=" * 70)
        
        return {
            "monitor_id": self.monitor_id,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "detailed_checks": checks,
            "all_passed": summary["overall_status"] == "HEALTHY"
        }
    
    def _generate_summary(self, checks: Dict) -> Dict:
        """Generate business-friendly summary"""
        failed_checks = []
        warning_checks = []
        
        for check_name, result in checks.items():
            severity = result.get("severity", "INFO")
            if severity == "CRITICAL":
                failed_checks.append(check_name)
            elif severity == "HIGH":
                warning_checks.append(check_name)
        
        overall_status = "HEALTHY"
        if failed_checks:
            overall_status = "CRITICAL"
        elif warning_checks:
            overall_status = "WARNING"
        
        return {
            "overall_status": overall_status,
            "failed_checks": failed_checks,
            "warning_checks": warning_checks,
            "total_checks": len(checks),
            "passed_checks": len(checks) - len(failed_checks) - len(warning_checks)
        }
    
    def _log_monitoring_results(self, summary: Dict, checks: Dict):
        """Log monitoring results in business-friendly format"""
        logger.info("\n MONITORING SUMMARY:")
        logger.info(f"  Overall Status: {summary['overall_status']}")
        logger.info(f"  Checks Passed: {summary['passed_checks']}/{summary['total_checks']}")
        
        if summary['failed_checks']:
            logger.warning(f"   Failed: {summary['failed_checks']}")
        if summary['warning_checks']:
            logger.warning(f"    Warnings: {summary['warning_checks']}")
    
    #  REPORTING & PERSISTENCE 
    
    def save_report(self, report: Dict, path: str = "reports/data_monitor_report.json"):
        """Save monitoring report with timestamps"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f" Report saved → {path}")
    
    def save_history(self, path: str = "reports/monitor_history.json"):
        """Save monitoring history for trend analysis"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        
        logger.info(f"History saved → {path}")

# ----------------------------
#  DEMONSTRATION 
# ----------------------------

def create_test_data():
    """Create test data with intentional issues"""
    dates = pd.date_range("2024-01-01", periods=50, freq='D')
    
    # Good data
    revenue = 50000 + np.random.normal(0, 5000, 50)
    transactions = np.random.randint(100, 1000, 50)
    
    # Add some issues
    revenue[10] = np.nan  # Missing value
    revenue[20] = -1000   # Negative revenue (business rule violation)
    revenue[30] = 200000  # Extreme value (drift)
    
    df = pd.DataFrame({
        "metric_date": dates,
        "daily_revenue": revenue,
        "transaction_count": transactions,
        "success_rate": np.random.uniform(0.8, 0.99, 50)
    })
    
    return df

def main():
    """Demonstrate the business data monitor"""
    print("=" * 70)
    print("BUSINESS DATA MONITOR DEMONSTRATION")
    print("=" * 70)
    
    # Create test data
    print("\n Creating test data...")
    data = create_test_data()
    print(f"   Data shape: {data.shape}")
    
    # Initialize monitor
    print("\n Initializing monitor...")
    expected_schema = ["metric_date", "daily_revenue", "transaction_count", "success_rate"]
    
    config = DataMonitorConfig(
        missing_threshold=0.05,      # 5%
        volume_drop_threshold=0.2,   # 20%
        drift_zscore_threshold=3.0,  # 3 std
        freshness_hours_threshold=48 # 2 days
    )
    
    monitor = BusinessDataMonitor(
        expected_schema=expected_schema,
        config=config,
        monitor_id="business_kpi_monitor"
    )
    
    # Set reference data (first run)
    print("\n Setting reference data...")
    monitor.set_reference_data(data)
    
    # Run comprehensive monitoring
    print("\n Running comprehensive monitoring...")
    report = monitor.run_comprehensive_monitor(data)
    
    # Save reports
    print("\n Saving reports...")
    monitor.save_report(report, "reports/data_quality_report.json")
    monitor.save_history("reports/monitoring_history.json")
    
    # Display results
    print("\n" + "=" * 70)
    print("MONITORING RESULTS:")
    print("=" * 70)
    
    summary = report["summary"]
    checks = report["detailed_checks"]
    
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Passed Checks: {summary['passed_checks']}/{summary['total_checks']}")
    
    if summary['failed_checks']:
        print(f"\n Critical Issues:")
        for check in summary['failed_checks']:
            result = checks[check]
            print(f"  • {check}: {result.get('severity')}")
    
    if summary['warning_checks']:
        print(f"\n Warnings:")
        for check in summary['warning_checks']:
            result = checks[check]
            print(f"  • {check}: {result.get('severity')}")
    
    print("\n" + "=" * 70)
    print(" Monitoring Complete")
    print("=" * 70)
    
    return report

if __name__ == "__main__":
    report = main()