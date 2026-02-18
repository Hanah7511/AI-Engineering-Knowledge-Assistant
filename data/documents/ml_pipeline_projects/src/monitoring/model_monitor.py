"""
Business ML Model Monitor for Anomaly Detection
===============================================

Key Features:
- Alert rate health monitoring
- Score distribution tracking
- Prediction drift detection
- Business confidence analysis
- Performance degradation alerts
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Any
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Enhanced Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("BusinessModelMonitor")


# ----------------------------
# Business-Aware Configuration
# ----------------------------
class BusinessModelMonitorConfig:
    """
    Business-aware monitoring thresholds based on industry best practices.
    
    Alert Rate Guidelines:
    - <1%: Model may be missing important anomalies
    - 1-5%: Optimal range for business anomaly detection
    - >8%: Too many false alerts, risk of alert fatigue
    """
    
    def __init__(
        self,
        min_alert_rate: float = 0.01,      # 1% minimum alerts
        max_alert_rate: float = 0.05,      # 5% maximum alerts (conservative)
        drift_z_threshold: float = 2.5,    # 2.5 std deviations for drift
        score_shift_threshold: float = 0.25, # Significant score shift
        confidence_threshold: float = 70.0, # Minimum confidence for action
        window_size: int = 30,             # Rolling window for trends
        business_impact_metric: str = "daily_revenue"
    ):
        self.min_alert_rate = min_alert_rate
        self.max_alert_rate = max_alert_rate
        self.drift_z_threshold = drift_z_threshold
        self.score_shift_threshold = score_shift_threshold
        self.confidence_threshold = confidence_threshold
        self.window_size = window_size
        self.business_impact_metric = business_impact_metric
        
        logger.info(f"Business Model Monitor Config:")
        logger.info(f"  Alert Rate Range: {min_alert_rate*100:.1f}% - {max_alert_rate*100:.1f}%")
        logger.info(f"  Drift Threshold: {drift_z_threshold} std devs")
        logger.info(f"  Business Metric: {business_impact_metric}")


# ----------------------------
# Business Model Monitor
# ----------------------------
class BusinessModelMonitor:
    """
    Production model monitor with business context.
    
    Monitors:
    1. Alert Rate Health - Are we getting useful alerts?
    2. Score Stability - Is model confidence consistent?
    3. Prediction Drift - Has model behavior changed?
    4. Business Confidence - Are high-confidence anomalies impactful?
    5. Performance Trends - Is model degrading over time?
    """
    
    def __init__(
        self,
        config: Optional[BusinessModelMonitorConfig] = None,
        monitor_id: str = "business_ml_monitor"
    ):
        self.config = config or BusinessModelMonitorConfig()
        self.monitor_id = monitor_id
        
        # Reference statistics from training/validation
        self.reference_stats = None
        
        # Monitoring history for trend analysis
        self.history = []
        self.alert_rate_history = deque(maxlen=100)
        self.confidence_history = deque(maxlen=100)
        
        logger.info(f"Business Model Monitor '{monitor_id}' initialized")
    
    # ==================== REFERENCE DATA SETUP ====================
    
    def set_reference_data(self, df: pd.DataFrame):
        """
        Store baseline model behavior from training/validation phase.
        
        This establishes what 'normal' model behavior looks like.
        """
        logger.info(" Setting reference model statistics...")
        
        required_cols = ["ml_anomaly_flag", "ml_anomaly_score"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Store comprehensive reference statistics
        self.reference_stats = {
            "anomaly_rate": float(df["ml_anomaly_flag"].mean()),
            "score_mean": float(df["ml_anomaly_score"].mean()),
            "score_std": float(df["ml_anomaly_score"].std()),
            "score_q25": float(df["ml_anomaly_score"].quantile(0.25)),
            "score_q75": float(df["ml_anomaly_score"].quantile(0.75)),
            "timestamp": datetime.now().isoformat(),
            "sample_size": len(df)
        }
        
        # If business metrics available, store them too
        if self.config.business_impact_metric in df.columns:
            anomalies = df[df["ml_anomaly_flag"] == 1]
            if len(anomalies) > 0:
                self.reference_stats["business_impact_mean"] = float(
                    anomalies[self.config.business_impact_metric].mean()
                )
        
        logger.info(f"Reference stats saved (n={len(df)}):")
        logger.info(f"  • Anomaly Rate: {self.reference_stats['anomaly_rate']*100:.2f}%")
        logger.info(f"  • Score Mean: {self.reference_stats['score_mean']:.2f}")
    
    # ==================== ALERT RATE HEALTH ====================
    
    def check_alert_rate_health(self, df: pd.DataFrame) -> Dict:
        """
        Monitor alert rate for business usefulness.
        
        Returns health status and recommendations.
        """
        logger.info(" Checking alert rate health...")
        
        rate = df["ml_anomaly_flag"].mean()
        self.alert_rate_history.append(rate)
        
        # Determine health status
        if rate < self.config.min_alert_rate:
            status = "TOO_LOW"
            recommendation = "Model may be missing anomalies. Consider lowering threshold."
        elif rate > self.config.max_alert_rate:
            status = "TOO_HIGH"
            recommendation = "Too many alerts - risk of alert fatigue. Consider raising threshold."
        else:
            status = "HEALTHY"
            recommendation = "Alert rate within optimal range."
        
        # Calculate trend
        trend = "STABLE"
        if len(self.alert_rate_history) >= 10:
            recent = list(self.alert_rate_history)[-10:]
            if np.polyfit(range(len(recent)), recent, 1)[0] > 0.001:
                trend = "INCREASING"
            elif np.polyfit(range(len(recent)), recent, 1)[0] < -0.001:
                trend = "DECREASING"
        
        return {
            "check": "alert_rate_health",
            "timestamp": datetime.now().isoformat(),
            "alert_rate_percent": round(rate * 100, 2),
            "health_status": status,
            "trend": trend,
            "recommendation": recommendation,
            "thresholds": {
                "min": self.config.min_alert_rate * 100,
                "max": self.config.max_alert_rate * 100
            }
        }
    
    # ==================== SCORE DISTRIBUTION ====================
    
    def check_score_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Monitor anomaly score distribution for consistency.
        """
        logger.info(" Checking score distribution...")
        
        scores = df["ml_anomaly_score"]
        
        # Calculate distribution metrics
        distribution = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "skewness": float(scores.skew()),
            "kurtosis": float(scores.kurtosis()),
            "q1": float(scores.quantile(0.25)),
            "median": float(scores.median()),
            "q3": float(scores.quantile(0.75))
        }
        
        # Check for score compression (all scores similar)
        score_range = scores.max() - scores.min()
        if score_range < 20:  # Arbitrary threshold
            compression_status = "HIGH_COMPRESSION"
        elif score_range < 50:
            compression_status = "MODERATE_COMPRESSION"
        else:
            compression_status = "GOOD_SPREAD"
        
        return {
            "check": "score_distribution",
            "timestamp": datetime.now().isoformat(),
            "distribution": distribution,
            "compression_status": compression_status,
            "score_range": float(score_range)
        }
    
    # ==================== PREDICTION DRIFT DETECTION ====================
    
    def detect_prediction_drift(self, df: pd.DataFrame) -> Dict:
        """
        Detect if model predictions have drifted from reference behavior.
        """
        logger.info(" Detecting prediction drift...")
        
        if self.reference_stats is None:
            return {
                "check": "prediction_drift",
                "timestamp": datetime.now().isoformat(),
                "status": "NO_REFERENCE_DATA",
                "severity": "INFO"
            }
        
        current_rate = df["ml_anomaly_flag"].mean()
        ref_rate = self.reference_stats["anomaly_rate"]
        
        current_score_mean = df["ml_anomaly_score"].mean()
        ref_score_mean = self.reference_stats["score_mean"]
        ref_score_std = self.reference_stats["score_std"] + 1e-8
        
        # Calculate drift metrics
        rate_change_pct = ((current_rate - ref_rate) / (ref_rate + 1e-8)) * 100
        score_shift_z = abs(current_score_mean - ref_score_mean) / ref_score_std
        
        # Determine drift status
        drift_detected = score_shift_z > self.config.drift_z_threshold
        
        if drift_detected:
            drift_status = "DRIFT_DETECTED"
            severity = "HIGH"
            recommendation = "Model behavior has changed significantly. Consider retraining."
        elif score_shift_z > self.config.drift_z_threshold * 0.7:
            drift_status = "DRIFT_SUSPECTED"
            severity = "MEDIUM"
            recommendation = "Monitor closely - model may be drifting."
        else:
            drift_status = "NO_DRIFT"
            severity = "LOW"
            recommendation = "Model behavior stable."
        
        return {
            "check": "prediction_drift",
            "timestamp": datetime.now().isoformat(),
            "drift_detected": drift_detected,
            "drift_status": drift_status,
            "severity": severity,
            "score_shift_z": round(score_shift_z, 3),
            "anomaly_rate_change_pct": round(rate_change_pct, 2),
            "recommendation": recommendation,
            "threshold": self.config.drift_z_threshold
        }
    
    # ==================== BUSINESS CONFIDENCE ANALYSIS ====================
    
    def analyze_business_confidence(self, df: pd.DataFrame) -> Dict:
        """
        Analyze if high-confidence anomalies correspond to business impact.
        """
        logger.info(" Analyzing business confidence...")
        
        required_cols = ["ml_anomaly_flag", "ml_anomaly_score"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return {"check": "business_confidence", "error": f"Missing columns: {missing}"}
        
        # Separate anomalies and normal predictions
        anomalies = df[df["ml_anomaly_flag"] == 1]
        normal = df[df["ml_anomaly_flag"] == 0]
        
        if len(anomalies) == 0:
            return {
                "check": "business_confidence",
                "timestamp": datetime.now().isoformat(),
                "status": "NO_ANOMALIES_DETECTED",
                "severity": "INFO"
            }
        
        # Confidence metrics
        avg_anomaly_score = anomalies["ml_anomaly_score"].mean()
        avg_normal_score = normal["ml_anomaly_score"].mean() if len(normal) > 0 else 0
        
        score_separation = abs(avg_anomaly_score - avg_normal_score)
        
        # Confidence distribution
        high_conf_anomalies = anomalies[anomalies["ml_anomaly_score"] > self.config.confidence_threshold]
        low_conf_anomalies = anomalies[anomalies["ml_anomaly_score"] <= self.config.confidence_threshold]
        
        # Business impact correlation (if business metric available)
        business_correlation = None
        if self.config.business_impact_metric in df.columns:
            # Check if high-confidence anomalies have higher business impact
            if len(high_conf_anomalies) > 1 and len(low_conf_anomalies) > 1:
                high_impact = high_conf_anomalies[self.config.business_impact_metric].mean()
                low_impact = low_conf_anomalies[self.config.business_impact_metric].mean()
                business_correlation = {
                    "high_confidence_impact": float(high_impact),
                    "low_confidence_impact": float(low_impact),
                    "impact_ratio": float(high_impact / (low_impact + 1e-8))
                }
        
        confidence_status = "GOOD" if score_separation > 20 else "POOR"
        
        return {
            "check": "business_confidence",
            "timestamp": datetime.now().isoformat(),
            "avg_anomaly_score": round(avg_anomaly_score, 2),
            "avg_normal_score": round(avg_normal_score, 2),
            "score_separation": round(score_separation, 2),
            "confidence_status": confidence_status,
            "high_confidence_anomalies": len(high_conf_anomalies),
            "low_confidence_anomalies": len(low_conf_anomalies),
            "business_correlation": business_correlation,
            "confidence_threshold": self.config.confidence_threshold
        }
    
    # ==================== PERFORMANCE TREND ANALYSIS ====================
    
    def analyze_performance_trends(self) -> Dict:
        """
        Analyze historical performance trends.
        """
        if len(self.history) < 5:
            return {
                "check": "performance_trends",
                "timestamp": datetime.now().isoformat(),
                "status": "INSUFFICIENT_HISTORY",
                "history_size": len(self.history)
            }
        
        # Extract historical alert rates
        alert_rates = []
        confidence_scores = []
        
        for entry in self.history[-20:]:  # Last 20 entries
            if "alert_rate" in entry:
                alert_rates.append(entry["alert_rate"]["alert_rate_percent"])
            if "confidence" in entry:
                confidence_scores.append(entry["confidence"]["avg_anomaly_score"])
        
        # Calculate trends
        if len(alert_rates) >= 3:
            alert_trend_coef = np.polyfit(range(len(alert_rates)), alert_rates, 1)[0]
            alert_trend = "INCREASING" if alert_trend_coef > 0.1 else "DECREASING" if alert_trend_coef < -0.1 else "STABLE"
        else:
            alert_trend = "UNKNOWN"
        
        return {
            "check": "performance_trends",
            "timestamp": datetime.now().isoformat(),
            "alert_rate_trend": alert_trend,
            "history_size": len(self.history),
            "days_monitored": len(set([h["timestamp"][:10] for h in self.history]))
        }
    
    # ==================== COMPREHENSIVE MONITORING ====================
    
    def run_comprehensive_monitor(self, df: pd.DataFrame) -> Dict:
        """
        Run all monitoring checks and generate business report.
        """
        logger.info("=" * 70)
        logger.info("RUNNING BUSINESS MODEL MONITOR")
        logger.info("=" * 70)
        
        # Validate required columns
        required_cols = ["ml_anomaly_flag", "ml_anomaly_score"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Cannot run monitor. Missing columns: {missing}")
        
        # Run all monitoring checks
        checks = {
            "alert_rate_health": self.check_alert_rate_health(df),
            "score_distribution": self.check_score_distribution(df),
            "prediction_drift": self.detect_prediction_drift(df),
            "business_confidence": self.analyze_business_confidence(df),
            "performance_trends": self.analyze_performance_trends()
        }
        
        # Generate summary
        summary = self._generate_monitoring_summary(checks)
        
        # Store in history
        monitor_result = {
            "timestamp": datetime.now().isoformat(),
            "monitor_id": self.monitor_id,
            "summary": summary,
            "detailed_checks": checks
        }
        
        self.history.append(monitor_result)
        
        # Log results
        self._log_monitoring_results(summary, checks)
        
        logger.info("=" * 70)
        logger.info("MODEL MONITORING COMPLETE")
        logger.info("=" * 70)
        
        return monitor_result
    
    def _generate_monitoring_summary(self, checks: Dict) -> Dict:
        """Generate business-friendly monitoring summary"""
        
        # Count issues
        critical_issues = []
        warnings = []
        
        for check_name, result in checks.items():
            if "severity" in result:
                if result["severity"] == "HIGH":
                    critical_issues.append(check_name)
                elif result["severity"] == "MEDIUM":
                    warnings.append(check_name)
            elif "health_status" in result and result["health_status"] != "HEALTHY":
                warnings.append(check_name)
            elif "drift_detected" in result and result["drift_detected"]:
                critical_issues.append(check_name)
        
        # Determine overall status
        if critical_issues:
            overall_status = "CRITICAL"
            status_color = "🔴"
        elif warnings:
            overall_status = "WARNING"
            status_color = "🟡"
        else:
            overall_status = "HEALTHY"
            status_color = "🟢"
        
        # Get key metrics
        alert_rate = checks.get("alert_rate_health", {}).get("alert_rate_percent", 0)
        drift_status = checks.get("prediction_drift", {}).get("drift_status", "UNKNOWN")
        
        return {
            "overall_status": overall_status,
            "status_color": status_color,
            "alert_rate_percent": alert_rate,
            "drift_status": drift_status,
            "critical_issues": critical_issues,
            "warnings": warnings,
            "total_checks": len(checks),
            "checks_passed": len(checks) - len(critical_issues) - len(warnings)
        }
    
    def _log_monitoring_results(self, summary: Dict, checks: Dict):
        """Log monitoring results in business-friendly format"""
        logger.info("\n MODEL MONITORING SUMMARY:")
        logger.info(f"  Overall Status: {summary['status_color']} {summary['overall_status']}")
        logger.info(f"  Alert Rate: {summary['alert_rate_percent']:.1f}%")
        logger.info(f"  Checks Passed: {summary['checks_passed']}/{summary['total_checks']}")
        
        if summary["critical_issues"]:
            logger.warning(f"  🔴 Critical Issues: {summary['critical_issues']}")
        if summary["warnings"]:
            logger.warning(f"  🟡 Warnings: {summary['warnings']}")
    
    # ==================== REPORTING & VISUALIZATION ====================
    
    def save_report(self, report: Dict, path: str = "reports/model_monitor_report.json"):
        """Save comprehensive monitoring report"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f" Model monitoring report saved → {path}")
    
    def save_history(self, path: str = "reports/model_monitor_history.json"):
        """Save monitoring history for trend analysis"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        
        logger.info(f" Monitoring history saved → {path}")
    
    def generate_health_dashboard(self) -> str:
        """Generate simple health dashboard text"""
        if not self.history:
            return "No monitoring history available."
        
        latest = self.history[-1]
        summary = latest.get("summary", {})
        
        dashboard = f"""
        ---BUSINESS ML MODEL HEALTH DASHBOARD---
    ---Status:       {summary.get('status_color', '⚫')} {summary.get('overall_status', 'UNKNOWN')}
    ---Status:       {summary.get('status_color', '⚫')} {summary.get('overall_status', 'UNKNOWN')}
    ---Alert Rate:   {summary.get('alert_rate_percent', 0):.1f}%
    ---Checks:       {summary.get('checks_passed', 0)}/{summary.get('total_checks', 0)} passed
    ---Last Check:   {latest.get('timestamp', 'N/A')[:19]}

        """
        return dashboard


# ==================== DEMONSTRATION ====================

def create_sample_predictions():
    """Create sample ML predictions for demonstration"""
    np.random.seed(42)
    
    # Simulate 30 days of predictions
    dates = pd.date_range("2024-01-01", periods=30, freq='D')
    
    # Simulate anomaly flags and scores
    anomaly_flags = np.random.choice([0, 1], 30, p=[0.93, 0.07])
    anomaly_scores = np.random.uniform(0, 100, 30)
    
    # Make some anomalies high confidence
    for i in range(len(anomaly_flags)):
        if anomaly_flags[i] == 1:
            anomaly_scores[i] = np.random.uniform(70, 95)
        else:
            anomaly_scores[i] = np.random.uniform(10, 40)
    
    # Add business metric
    daily_revenue = 50000 + np.random.normal(0, 5000, 30)
    
    df = pd.DataFrame({
        "metric_date": dates,
        "ml_anomaly_flag": anomaly_flags,
        "ml_anomaly_score": anomaly_scores,
        "daily_revenue": daily_revenue
    })
    
    return df

def main():
    """Demonstrate the business model monitor"""
    print("=" * 70)
    print("BUSINESS MODEL MONITOR DEMONSTRATION")
    print("=" * 70)
    
    # Create sample predictions
    print("\n Creating sample ML predictions...")
    predictions = create_sample_predictions()
    print(f"   Generated {len(predictions)} days of predictions")
    print(f"   Anomaly rate: {predictions['ml_anomaly_flag'].mean()*100:.1f}%")
    
    # Initialize monitor
    print("\n  Initializing business model monitor...")
    config = BusinessModelMonitorConfig(
        min_alert_rate=0.01,
        max_alert_rate=0.05,
        drift_z_threshold=2.5,
        business_impact_metric="daily_revenue"
    )
    
    monitor = BusinessModelMonitor(
        config=config,
        monitor_id="business_anomaly_monitor"
    )
    
    # Set reference data (first batch)
    print("\n Setting reference data...")
    monitor.set_reference_data(predictions.iloc[:15])  # First half as reference
    
    # Run comprehensive monitoring
    print("\n Running comprehensive monitoring...")
    report = monitor.run_comprehensive_monitor(predictions.iloc[15:])  # Second half as current
    
    # Save reports
    print("\n Saving reports...")
    monitor.save_report(report, "reports/model_health_report.json")
    monitor.save_history("reports/model_monitoring_history.json")
    
    # Display dashboard
    print("\n" + "=" * 70)
    print("MODEL HEALTH DASHBOARD")
    print("=" * 70)
    
    dashboard = monitor.generate_health_dashboard()
    print(dashboard)
    
    # Show key findings
    summary = report.get("summary", {})
    checks = report.get("detailed_checks", {})
    
    if summary.get("critical_issues"):
        print("\n🔴 CRITICAL ISSUES FOUND:")
        for issue in summary["critical_issues"]:
            print(f"  • {issue}: {checks.get(issue, {}).get('recommendation', 'Check details')}")
    
    if summary.get("warnings"):
        print("\n🟡 WARNINGS:")
        for warning in summary["warnings"]:
            print(f"  • {warning}")
    
    print("\n" + "=" * 70)
    print(" MODEL MONITORING COMPLETE")
    print("=" * 70)
    
    return report

if __name__ == "__main__":
    report = main()