"""
Baseline Anomaly Detection Engine - Enhanced
=============================================

Production-grade statistical anomaly detection with advanced features.

Enhancements:
- Multi-metric support (revenue, transactions, conversion rate, etc.)
- Seasonality-aware detection (day-of-week, monthly patterns)
- Adaptive thresholds based on data distribution
- Anomaly severity scoring (mild, moderate, severe)
- Rich metadata and explainability
- Performance optimizations
- Comprehensive error handling
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# --------------------------------------------------
# LOGGING CONFIG
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("BaselineDetector")


# ==================================================
# ENUMS FOR TYPE SAFETY
# ==================================================
class AnomalySeverity(Enum):
    """Anomaly severity levels"""
    NORMAL = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3


class DetectionMethod(Enum):
    """Available detection methods"""
    ZSCORE = "zscore"
    ROLLING_ZSCORE = "rolling_zscore"
    IQR = "iqr"
    MAD = "mad"  # Median Absolute Deviation
    PERCENTILE = "percentile"


# ==================================================
# ENHANCED CONFIGURATION
# ==================================================
@dataclass
class BaselineConfig:
    """Enhanced configuration with adaptive capabilities"""
    
    # Thresholds
    zscore_threshold: float = 3.0
    zscore_moderate: float = 2.5
    zscore_mild: float = 2.0
    iqr_multiplier: float = 1.5
    mad_threshold: float = 3.5
    percentile_lower: float = 1.0  # 1st percentile
    percentile_upper: float = 99.0  # 99th percentile
    
    # Rolling windows
    rolling_window: int = 7
    seasonal_window: int = 7  # For day-of-week patterns
    
    # Safety guards
    min_std_floor: float = 1e-8
    min_required_rows: int = 10
    
    # Feature flags
    enable_seasonality: bool = True
    enable_trend_removal: bool = True
    enable_adaptive_threshold: bool = True
    
    # Ensemble
    ensemble_threshold: int = 2  # Min methods agreeing
    
    # Supported metrics
    supported_metrics: List[str] = field(default_factory=lambda: [
        'daily_revenue', 'transaction_count', 'avg_order_value',
        'conversion_rate', 'customer_count', 'return_rate'
    ])


# ==================================================
# ANOMALY RESULT OBJECT
# ==================================================
@dataclass
class AnomalyResult:
    """Rich anomaly detection result"""
    index: int
    date: datetime
    metric_name: str
    actual_value: float
    expected_value: float
    deviation: float
    severity: AnomalySeverity
    methods_triggered: List[str]
    confidence_score: float
    explanation: str


# ==================================================
# ENHANCED DETECTOR ENGINE
# ==================================================
class BaselineAnomalyDetector:
    """Enhanced baseline anomaly detector with advanced features"""

    def __init__(self, config: Optional[BaselineConfig] = None):
        self.config = config or BaselineConfig()
        self.detection_stats = {}
        logger.info("Enhanced Baseline Detector Initialized")
        logger.info(f"Config: {self.config}")

    # --------------------------------------------------
    # ENHANCED VALIDATION
    # --------------------------------------------------
    def _validate_input(self, df: pd.DataFrame, column: str):
        """Comprehensive input validation"""
        
        if df is None or len(df) == 0:
            raise ValueError("Input DataFrame is empty")

        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found. Available: {list(df.columns)}"
            )

        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(f"Column '{column}' must be numeric")

        if df[column].isna().all():
            raise ValueError(f"Column '{column}' contains only NaN values")

        if len(df) < self.config.min_required_rows:
            logger.warning(
                f"Low row count ({len(df)}). Detection may be unstable. "
                f"Minimum recommended: {self.config.min_required_rows}"
            )

        # Check for date column
        if 'metric_date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['metric_date']):
                logger.info("Converting metric_date to datetime")
                df['metric_date'] = pd.to_datetime(df['metric_date'])

    # --------------------------------------------------
    # SAFE STATISTICS
    # --------------------------------------------------
    def _safe_std(self, series: pd.Series) -> float:
        """Safe standard deviation calculation"""
        std = series.std()
        return max(std, self.config.min_std_floor) if not np.isnan(std) else self.config.min_std_floor

    def _safe_mad(self, series: pd.Series) -> float:
        """Median Absolute Deviation - robust to outliers"""
        median = series.median()
        mad = np.median(np.abs(series - median))
        return max(mad, self.config.min_std_floor)

    # --------------------------------------------------
    # SEASONALITY HANDLING
    # --------------------------------------------------
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features for seasonality detection"""
        
        if 'metric_date' not in df.columns:
            logger.warning("No metric_date column found. Skipping temporal features.")
            return df
        
        df['day_of_week'] = df['metric_date'].dt.dayofweek
        df['day_of_month'] = df['metric_date'].dt.day
        df['month'] = df['metric_date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        
        logger.info("Temporal features added")
        return df

    def _deseasonalize(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Remove day-of-week seasonality"""
        
        if not self.config.enable_seasonality or 'day_of_week' not in df.columns:
            return df[column]
        
        # Calculate day-of-week averages
        dow_means = df.groupby('day_of_week')[column].transform('mean')
        overall_mean = df[column].mean()
        
        # Deseasonalized = actual - (dow_mean - overall_mean)
        deseasonalized = df[column] - (dow_means - overall_mean)
        
        return deseasonalized

    def _detrend(self, series: pd.Series) -> pd.Series:
        """Remove linear trend"""
        
        if not self.config.enable_trend_removal or len(series) < 20:
            return series
        
        x = np.arange(len(series))
        # Handle NaN values
        mask = ~series.isna()
        if mask.sum() < 2:
            return series
            
        coeffs = np.polyfit(x[mask], series[mask], 1)
        trend = np.polyval(coeffs, x)
        
        return series - trend + series.mean()

    # --------------------------------------------------
    # DETECTION METHODS
    # --------------------------------------------------
    def detect_global_zscore(
        self, 
        df: pd.DataFrame, 
        column: str = "daily_revenue"
    ) -> pd.DataFrame:
        """Enhanced global Z-score with severity levels"""
        
        logger.info(f"Global Z-Score Detection on {column}")

        # Optionally deseasonalize and detrend
        series = self._deseasonalize(df, column)
        series = self._detrend(series)

        mean = series.mean()
        std = self._safe_std(series)

        df[f"{column}_zscore"] = (df[column] - mean) / std
        
        # Multi-level severity
        abs_zscore = np.abs(df[f"{column}_zscore"])
        df[f"{column}_zscore_severity"] = pd.cut(
            abs_zscore,
            bins=[0, self.config.zscore_mild, self.config.zscore_moderate, 
                  self.config.zscore_threshold, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        df[f"{column}_zscore_flag"] = abs_zscore > self.config.zscore_threshold

        return df

    def detect_rolling_zscore(
        self, 
        df: pd.DataFrame, 
        column: str = "daily_revenue"
    ) -> pd.DataFrame:
        """Enhanced rolling Z-score with adaptive window"""
        
        logger.info(f"Rolling Z-Score Detection on {column}")

        window = self.config.rolling_window

        rolling_mean = df[column].rolling(window, min_periods=1).mean()
        rolling_std = (
            df[column]
            .rolling(window, min_periods=1)
            .std()
            .clip(lower=self.config.min_std_floor)
        )

        df[f"{column}_rolling_zscore"] = (df[column] - rolling_mean) / rolling_std
        
        abs_rolling = np.abs(df[f"{column}_rolling_zscore"])
        df[f"{column}_rolling_flag"] = abs_rolling > self.config.zscore_threshold

        return df

    def detect_iqr(
        self, 
        df: pd.DataFrame, 
        column: str = "daily_revenue"
    ) -> pd.DataFrame:
        """Enhanced IQR detection with dynamic bounds"""
        
        logger.info(f"IQR Detection on {column}")

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Adaptive multiplier based on data spread
        multiplier = self.config.iqr_multiplier
        if self.config.enable_adaptive_threshold:
            cv = df[column].std() / df[column].mean() if df[column].mean() != 0 else 0
            if cv > 0.5:  # High variability
                multiplier *= 1.2

        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR

        df[f"{column}_iqr_lower"] = lower
        df[f"{column}_iqr_upper"] = upper
        df[f"{column}_iqr_flag"] = (df[column] < lower) | (df[column] > upper)

        return df

    def detect_mad(
        self, 
        df: pd.DataFrame, 
        column: str = "daily_revenue"
    ) -> pd.DataFrame:
        """Median Absolute Deviation - robust to outliers"""
        
        logger.info(f"MAD Detection on {column}")

        median = df[column].median()
        mad = self._safe_mad(df[column])

        # Modified Z-score using MAD
        df[f"{column}_mad_score"] = 0.6745 * (df[column] - median) / mad
        df[f"{column}_mad_flag"] = (
            np.abs(df[f"{column}_mad_score"]) > self.config.mad_threshold
        )

        return df

    def detect_percentile(
        self, 
        df: pd.DataFrame, 
        column: str = "daily_revenue"
    ) -> pd.DataFrame:
        """Percentile-based detection"""
        
        logger.info(f"Percentile Detection on {column}")

        lower = df[column].quantile(self.config.percentile_lower / 100)
        upper = df[column].quantile(self.config.percentile_upper / 100)

        df[f"{column}_percentile_flag"] = (
            (df[column] < lower) | (df[column] > upper)
        )

        return df

    # --------------------------------------------------
    # ENSEMBLE & SCORING
    # --------------------------------------------------
    def build_ensemble_score(
        self, 
        df: pd.DataFrame, 
        column: str = "daily_revenue"
    ) -> pd.DataFrame:
        """Enhanced ensemble with confidence scoring"""
        
        logger.info(f"Building Ensemble Score for {column}")

        flag_cols = [
            f"{column}_zscore_flag",
            f"{column}_rolling_flag",
            f"{column}_iqr_flag",
            f"{column}_mad_flag",
            f"{column}_percentile_flag",
        ]
        
        # Only use flags that exist
        existing_flags = [col for col in flag_cols if col in df.columns]

        df[f"{column}_ensemble_score"] = df[existing_flags].sum(axis=1)
        df[f"{column}_is_anomaly"] = (
            df[f"{column}_ensemble_score"] >= self.config.ensemble_threshold
        )
        
        # Confidence: % of methods that agreed
        df[f"{column}_confidence"] = (
            df[f"{column}_ensemble_score"] / len(existing_flags) * 100
        )

        # Severity based on ensemble score
        df[f"{column}_severity"] = pd.cut(
            df[f"{column}_ensemble_score"],
            bins=[-1, 0, 1, 2, len(existing_flags)],
            labels=[0, 1, 2, 3]
        ).astype(int)

        return df

    # --------------------------------------------------
    # ANOMALY EXTRACTION & EXPLAINABILITY
    # --------------------------------------------------
    def extract_anomalies(
        self, 
        df: pd.DataFrame, 
        column: str = "daily_revenue"
    ) -> List[AnomalyResult]:
        """Extract rich anomaly results with explanations"""
        
        anomalies = []
        anomaly_rows = df[df[f"{column}_is_anomaly"] == True]
        
        for idx, row in anomaly_rows.iterrows():
            # Determine which methods triggered
            methods = []
            if row.get(f"{column}_zscore_flag", False):
                methods.append("Z-Score")
            if row.get(f"{column}_rolling_flag", False):
                methods.append("Rolling Z-Score")
            if row.get(f"{column}_iqr_flag", False):
                methods.append("IQR")
            if row.get(f"{column}_mad_flag", False):
                methods.append("MAD")
            if row.get(f"{column}_percentile_flag", False):
                methods.append("Percentile")
            
            # Calculate expected value (rolling mean if available)
            expected = df[column].mean()
            if f"{column}_rolling_zscore" in df.columns:
                window = self.config.rolling_window
                expected = df[column].iloc[max(0, idx-window):idx].mean()
            
            actual = row[column]
            deviation = ((actual - expected) / expected * 100) if expected != 0 else 0
            
            # Generate explanation
            direction = "above" if actual > expected else "below"
            explanation = (
                f"{column} is {abs(deviation):.1f}% {direction} expected. "
                f"Detected by: {', '.join(methods)}"
            )
            
            anomaly = AnomalyResult(
                index=idx,
                date=row.get('metric_date', None),
                metric_name=column,
                actual_value=actual,
                expected_value=expected,
                deviation=deviation,
                severity=AnomalySeverity(row[f"{column}_severity"]),
                methods_triggered=methods,
                confidence_score=row[f"{column}_confidence"],
                explanation=explanation
            )
            
            anomalies.append(anomaly)
        
        return anomalies

    # --------------------------------------------------
    # STATISTICS & SUMMARY
    # --------------------------------------------------
    def generate_detection_summary(
        self, 
        df: pd.DataFrame, 
        column: str = "daily_revenue"
    ) -> Dict:
        """Generate detection statistics summary"""
        
        total_rows = len(df)
        anomaly_count = df[f"{column}_is_anomaly"].sum()
        anomaly_rate = (anomaly_count / total_rows * 100) if total_rows > 0 else 0
        
        severity_counts = df[f"{column}_severity"].value_counts().to_dict()
        
        summary = {
            'metric': column,
            'total_observations': total_rows,
            'anomalies_detected': int(anomaly_count),
            'anomaly_rate_pct': round(anomaly_rate, 2),
            'severity_breakdown': {
                'normal': severity_counts.get(0, 0),
                'mild': severity_counts.get(1, 0),
                'moderate': severity_counts.get(2, 0),
                'severe': severity_counts.get(3, 0)
            },
            'avg_confidence': round(df[f"{column}_confidence"].mean(), 2),
            'detection_date': datetime.now().isoformat()
        }
        
        return summary

    # --------------------------------------------------
    # MASTER PIPELINE
    # --------------------------------------------------
    def run(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Enhanced master pipeline with multi-metric support
        
        Returns:
            Tuple of (enhanced_df, summary_stats)
        """
        
        logger.info("=" * 70)
        logger.info("RUNNING ENHANCED BASELINE DETECTION PIPELINE")
        logger.info("=" * 70)

        # Default to single column if not specified
        if columns is None:
            columns = ['daily_revenue'] if 'daily_revenue' in df.columns else [df.select_dtypes(include=[np.number]).columns[0]]
        
        # Ensure list
        if isinstance(columns, str):
            columns = [columns]

        # Validate all columns
        for column in columns:
            self._validate_input(df, column)

        # Deterministic ordering (CI safety)
        if 'metric_date' in df.columns:
            df = df.sort_values("metric_date").reset_index(drop=True)
        
        # Add temporal features
        df = self._add_temporal_features(df)

        # Run detection for each metric
        all_summaries = {}
        all_anomalies = {}
        
        for column in columns:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing Metric: {column}")
            logger.info(f"{'='*60}")
            
            # Run all detection methods
            df = self.detect_global_zscore(df, column)
            df = self.detect_rolling_zscore(df, column)
            df = self.detect_iqr(df, column)
            df = self.detect_mad(df, column)
            df = self.detect_percentile(df, column)
            
            # Build ensemble
            df = self.build_ensemble_score(df, column)
            
            # Extract anomalies
            anomalies = self.extract_anomalies(df, column)
            all_anomalies[column] = anomalies
            
            # Generate summary
            summary = self.generate_detection_summary(df, column)
            all_summaries[column] = summary
            
            logger.info(f"✓ Detected {summary['anomalies_detected']} anomalies "
                       f"({summary['anomaly_rate_pct']}%)")

        logger.info("\n" + "=" * 70)
        logger.info("BASELINE DETECTION COMPLETED")
        logger.info("=" * 70)

        return df, {'summaries': all_summaries, 'anomalies': all_anomalies}


# ==================================================
# STANDALONE TEST MODE
# ==================================================
if __name__ == "__main__":

    logger.info("Enhanced Baseline Detector Test Mode")
    logger.info("=" * 70)

    # Enhanced synthetic data with realistic patterns
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=120)
    
    # Base revenue with weekly seasonality
    base_revenue = 30000
    weekly_pattern = np.tile([0.9, 0.95, 1.0, 1.05, 1.1, 1.3, 1.2], 18)[:120]
    revenue = base_revenue * weekly_pattern + np.random.normal(0, 2000, 120)
    
    # Inject various anomaly types
    revenue[40] = 65000  # Severe spike
    revenue[41] = 62000  # Moderate spike
    revenue[90] = 8000   # Severe drop
    revenue[70:73] = revenue[70:73] * 1.4  # Multi-day anomaly
    
    # Additional metrics
    transactions = revenue / (200 + np.random.normal(0, 20, 120))
    aov = revenue / transactions
    
    df = pd.DataFrame({
        "metric_date": dates,
        "daily_revenue": revenue,
        "transaction_count": transactions,
        "avg_order_value": aov
    })

    # Initialize with custom config
    config = BaselineConfig(
        zscore_threshold=3.0,
        rolling_window=7,
        enable_seasonality=True,
        enable_trend_removal=True,
        ensemble_threshold=2
    )
    
    detector = BaselineAnomalyDetector(config=config)

    # Run detection on multiple metrics
    result, stats = detector.run(
        df, 
        columns=['daily_revenue', 'transaction_count']
    )

    # Display results
    print("\n" + "=" * 70)
    print("DETECTION SUMMARY")
    print("=" * 70)
    
    for metric, summary in stats['summaries'].items():
        print(f"\n{metric.upper()}:")
        print(f"  Total observations: {summary['total_observations']}")
        print(f"  Anomalies detected: {summary['anomalies_detected']} ({summary['anomaly_rate_pct']}%)")
        print(f"  Severity breakdown: {summary['severity_breakdown']}")
        print(f"  Avg confidence: {summary['avg_confidence']}%")
    
    # Show anomaly details
    print("\n" + "=" * 70)
    print("DETECTED ANOMALIES")
    print("=" * 70)
    
    for metric, anomalies in stats['anomalies'].items():
        if anomalies:
            print(f"\n{metric.upper()}:")
            for anom in anomalies[:5]:  # Show first 5
                print(f"  • {anom.date.strftime('%Y-%m-%d') if anom.date else 'N/A'}: "
                      f"{anom.explanation} (Severity: {anom.severity.name})")

    # Save results
    output_file = "baseline_detection_enhanced_output.csv"
    result.to_csv(output_file, index=False)
    logger.info(f"\n✓ Full results saved → {output_file}")
    
    # Save summary stats
    import json
    summary_file = "detection_summary.json"
    with open(summary_file, 'w') as f:
        # Convert non-serializable objects
        serializable_stats = {
            'summaries': stats['summaries'],
            'anomaly_count_by_metric': {
                k: len(v) for k, v in stats['anomalies'].items()
            }
        }
        json.dump(serializable_stats, f, indent=2)
    logger.info(f"✓ Summary stats saved → {summary_file}")

# ==========================================
# PIPELINE COMPATIBILITY ADAPTER
# ==========================================

class BaselineDetector:
    """
    Pipeline Adapter → Uses BaselineAnomalyDetector internally.
    Keeps pipeline clean and stable.
    """

    def __init__(self, config=None):
        if config is None:
            config = BaselineConfig()

        self.detector = BaselineAnomalyDetector(config=config)

    def run_detection(self, df):
        """
        Method expected by pipeline.
        Returns dataframe with baseline anomaly flag.
        """

        result_df, stats = self.detector.run(
            df,
            columns=["daily_revenue"] if "daily_revenue" in df.columns else None
        )

        # Standardize output column for pipeline compatibility
        if "daily_revenue_is_anomaly" in result_df.columns:
            result_df["baseline_anomaly"] = result_df["daily_revenue_is_anomaly"].astype(int)
        else:
            result_df["baseline_anomaly"] = 0

        return result_df
