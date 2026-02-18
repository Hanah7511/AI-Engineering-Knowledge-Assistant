import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for business anomaly detection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer with configuration
        
        Args:
            config: Feature engineering parameters
        """
        self.config = config or self._default_config()
        self.feature_info = {
            'version': 'v1.0',
            'created_features': [],
            'feature_groups': {},
            'statistics': {}
        }
        
    def _default_config(self) -> Dict:
        """Default feature engineering configuration"""
        return {
            'rolling_windows': [7, 14, 30],      # Rolling window sizes
            'lag_periods': [1, 7, 14],           # Lag features
            'ewm_spans': [7, 14],                # Exponential weighted moving average spans
            'statistical_features': True,         # Enable statistical features
            'cyclical_encoding': True,            # Encode cyclical features (day, month)
            'interaction_features': True,         # Create interaction features
            'handle_missing': 'forward_fill',    # forward_fill, interpolate, drop
        }
    
    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all feature groups
        
        Args:
            df: Input dataframe with validated data
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("="*60)
        logger.info("STARTING FEATURE ENGINEERING")
        logger.info("="*60)
        
        initial_features = len(df.columns)
        
        # Ensure data is sorted by date
        df = df.sort_values('metric_date').reset_index(drop=True)
        
        # Build feature groups
        df = self.build_time_features(df)
        df = self.build_payment_features(df)
        df = self.build_rolling_features(df)
        df = self.build_lag_features(df)
        df = self.build_ewm_features(df)
        df = self.build_trend_features(df)
        
        if self.config['statistical_features']:
            df = self.build_statistical_features(df)
        
        if self.config['cyclical_encoding']:
            df = self.build_cyclical_features(df)
        
        if self.config['interaction_features']:
            df = self.build_interaction_features(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Summary
        final_features = len(df.columns)
        new_features = final_features - initial_features
        
        logger.info("\n" + "="*60)
        logger.info("FEATURE ENGINEERING SUMMARY")
        logger.info("="*60)
        logger.info(f"Initial Features: {initial_features}")
        logger.info(f"Final Features: {final_features}")
        logger.info(f"New Features Created: {new_features}")
        logger.info(f"Total Rows: {len(df)}")
        logger.info("="*60)
        
        self.feature_info['statistics'] = {
            'initial_features': initial_features,
            'final_features': final_features,
            'new_features_created': new_features,
            'total_rows': len(df)
        }
        # SAFETY GUARD — Ensure deterministic row order (CI stability)
        df = df.sort_values("metric_date").reset_index(drop=True)

        return df
    
    def build_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with time features
        """
        logger.info("\n[1/9] Building Time-Based Features...")
        
        features_created = []
        
        # Basic time features
        df['day_of_week'] = df['metric_date'].dt.dayofweek
        df['day_of_month'] = df['metric_date'].dt.day
        df['month'] = df['metric_date'].dt.month
        df['quarter'] = df['metric_date'].dt.quarter
        df['week_of_year'] = df['metric_date'].dt.isocalendar().week
        
        # Weekend indicator
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Month start/end indicators
        df['is_month_start'] = df['metric_date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['metric_date'].dt.is_month_end.astype(int)
        
        # Quarter start/end indicators
        df['is_quarter_start'] = df['metric_date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['metric_date'].dt.is_quarter_end.astype(int)
        
        # Days since start
        df['days_since_start'] = (df['metric_date'] - df['metric_date'].min()).dt.days
        
        features_created = [
            'day_of_week', 'day_of_month', 'month', 'quarter', 'week_of_year',
            'is_weekend', 'is_month_start', 'is_month_end', 
            'is_quarter_start', 'is_quarter_end', 'days_since_start'
        ]
        
        logger.info(f"  ✓ Created {len(features_created)} time features")
        self.feature_info['created_features'].extend(features_created)
        self.feature_info['feature_groups']['time'] = features_created
        
        return df
    
    def build_payment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create payment-related features
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with payment features
        """
        logger.info("\n[2/9] Building Payment Features...")
        
        features_created = []
        
        # Total payments
        df['total_payments'] = df['successful_payments'] + df['failed_payments']
        
        df['failure_rate'] = np.where(
            df['total_payments'] > 0,
            df['failed_payments'] / df['total_payments'],
            0
        )
        df['success_rate'] = np.where(

            df['total_payments'] > 0,
            df['successful_payments'] / df['total_payments'],
            0
        )
        df['revenue_per_payment'] = np.where(
            df['total_payments'] > 0,
            df['daily_revenue'] / df['total_payments'],
            0
        )
        df['revenue_per_success'] = np.where(
            df['successful_payments'] > 0,
            df['daily_revenue'] / df['successful_payments'],
            0
        )
   
        # Payment ratio (failed to successful)
        df['payment_fail_success_ratio'] = df['failed_payments'] / (df['successful_payments'] + 1)
        
        features_created = [
            'total_payments', 'failure_rate', 'success_rate',
            'revenue_per_payment', 'revenue_per_success', 'payment_fail_success_ratio'
        ]
        
        logger.info(f"  ✓ Created {len(features_created)} payment features")
        self.feature_info['created_features'].extend(features_created)
        self.feature_info['feature_groups']['payment'] = features_created
        
        return df
    
    def build_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with rolling features
        """
        logger.info("\n[3/9] Building Rolling Window Features...")
        
        features_created = []
        
        for window in self.config['rolling_windows']:
            # Revenue rolling features
            df[f'revenue_{window}d_mean'] = df['daily_revenue'].rolling(window, min_periods=1).mean()
            df[f'revenue_{window}d_std'] = df['daily_revenue'].rolling(window, min_periods=1).std()
            df[f'revenue_{window}d_min'] = df['daily_revenue'].rolling(window, min_periods=1).min()
            df[f'revenue_{window}d_max'] = df['daily_revenue'].rolling(window, min_periods=1).max()
            
            # Payments rolling features
            df[f'payments_{window}d_mean'] = df['total_payments'].rolling(window, min_periods=1).mean()
            df[f'failure_rate_{window}d_mean'] = df['failure_rate'].rolling(window, min_periods=1).mean()
            
            # Z-score (deviation from rolling mean)
            df[f'revenue_{window}d_zscore'] = (
                (df['daily_revenue'] - df[f'revenue_{window}d_mean']) / 
                (df[f'revenue_{window}d_std'] + 1e-8)
            )
            
            features_created.extend([
                f'revenue_{window}d_mean', f'revenue_{window}d_std',
                f'revenue_{window}d_min', f'revenue_{window}d_max',
                f'payments_{window}d_mean', f'failure_rate_{window}d_mean',
                f'revenue_{window}d_zscore'
            ])
        
        logger.info(f"  ✓ Created {len(features_created)} rolling window features")
        self.feature_info['created_features'].extend(features_created)
        self.feature_info['feature_groups']['rolling'] = features_created
        
        return df
    
    def build_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features (previous day values)
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with lag features
        """
        logger.info("\n[4/9] Building Lag Features...")
        
        features_created = []
        
        for lag in self.config['lag_periods']:
            # Revenue lags
            df[f'revenue_lag_{lag}d'] = df['daily_revenue'].shift(lag)
            
            # Payment lags
            df[f'total_payments_lag_{lag}d'] = df['total_payments'].shift(lag)
            df[f'failure_rate_lag_{lag}d'] = df['failure_rate'].shift(lag)
            
            features_created.extend([
                f'revenue_lag_{lag}d',
                f'total_payments_lag_{lag}d',
                f'failure_rate_lag_{lag}d'
            ])
        
        logger.info(f"  ✓ Created {len(features_created)} lag features")
        self.feature_info['created_features'].extend(features_created)
        self.feature_info['feature_groups']['lag'] = features_created
        
        return df
    
    def build_ewm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create exponential weighted moving average features
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with EWM features
        """
        logger.info("\n[5/9] Building Exponential Weighted Moving Average Features...")
        
        features_created = []
        
        for span in self.config['ewm_spans']:
            # Revenue EWM
            df[f'revenue_ewm_{span}d'] = df['daily_revenue'].ewm(span=span, adjust=False).mean()
            
            # Payment EWM
            df[f'failure_rate_ewm_{span}d'] = df['failure_rate'].ewm(span=span, adjust=False).mean()
            
            features_created.extend([
                f'revenue_ewm_{span}d',
                f'failure_rate_ewm_{span}d'
            ])
        
        logger.info(f"  ✓ Created {len(features_created)} EWM features")
        self.feature_info['created_features'].extend(features_created)
        self.feature_info['feature_groups']['ewm'] = features_created
        
        return df
    
    def build_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create trend and change features
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with trend features
        """
        logger.info("\n[6/9] Building Trend Features...")
        
        features_created = []
        
        # Percentage changes
        df['revenue_pct_change'] = df['daily_revenue'].pct_change()
        df['revenue_pct_change_7d'] = df['daily_revenue'].pct_change(periods=7)
        df['failure_rate_change'] = df['failure_rate'].pct_change()
        
        # Absolute changes
        df['revenue_diff'] = df['daily_revenue'].diff()
        df['revenue_diff_7d'] = df['daily_revenue'].diff(periods=7)
        
        # Acceleration (second derivative)
        df['revenue_acceleration'] = df['revenue_pct_change'].diff()
        
        # Momentum (rate of change of percentage change)
        df['revenue_momentum'] = df['revenue_pct_change'].rolling(7, min_periods=1).mean()
        
        # Volatility (rolling std of returns)
        df['revenue_volatility_7d'] = df['revenue_pct_change'].rolling(7, min_periods=1).std()
        df['revenue_volatility_14d'] = df['revenue_pct_change'].rolling(14, min_periods=1).std()
        
        # Replace inf values
        trend_cols = [col for col in df.columns if 'pct_change' in col or 'diff' in col or 
                     'acceleration' in col or 'momentum' in col or 'volatility' in col]
        for col in trend_cols:
            df[col] = df[col].replace([np.inf, -np.inf], 0)
        
        features_created = [
            'revenue_pct_change', 'revenue_pct_change_7d', 'failure_rate_change',
            'revenue_diff', 'revenue_diff_7d', 'revenue_acceleration',
            'revenue_momentum', 'revenue_volatility_7d', 'revenue_volatility_14d'
        ]
        
        logger.info(f"  ✓ Created {len(features_created)} trend features")
        self.feature_info['created_features'].extend(features_created)
        self.feature_info['feature_groups']['trend'] = features_created
        
        return df
    
    def build_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with statistical features
        """
        logger.info("\n[7/9] Building Statistical Features...")
        
        features_created = []
        
        # Distance from mean (normalized by std)
        for window in [7, 14, 30]:
            rolling_mean = df['daily_revenue'].rolling(window, min_periods=1).mean()
            rolling_std = df['daily_revenue'].rolling(window, min_periods=1).std()
            df[f'revenue_distance_from_mean_{window}d'] = (
                (df['daily_revenue'] - rolling_mean) / (rolling_std + 1e-8)
            )
            features_created.append(f'revenue_distance_from_mean_{window}d')
        
        # Coefficient of variation (std / mean)
        for window in [7, 14]:
            rolling_mean = df['daily_revenue'].rolling(window, min_periods=1).mean()
            rolling_std = df['daily_revenue'].rolling(window, min_periods=1).std()
            df[f'revenue_cv_{window}d'] = rolling_std / (rolling_mean + 1e-8)
            features_created.append(f'revenue_cv_{window}d')
        
        # Skewness and Kurtosis
        df['revenue_skew_14d'] = df['daily_revenue'].rolling(14, min_periods=1).skew()
        df['revenue_kurt_14d'] = df['daily_revenue'].rolling(14, min_periods=1).kurt()
        
        # Replace inf/nan values
        for col in features_created + ['revenue_skew_14d', 'revenue_kurt_14d']:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
        
        features_created.extend(['revenue_skew_14d', 'revenue_kurt_14d'])
        
        logger.info(f"  ✓ Created {len(features_created)} statistical features")
        self.feature_info['created_features'].extend(features_created)
        self.feature_info['feature_groups']['statistical'] = features_created
        
        return df
    
    def build_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode cyclical features using sin/cos transformation
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with cyclical features
        """
        logger.info("\n[8/9] Building Cyclical Encoding Features...")
        
        features_created = []
        
        # Day of week (0-6)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Day of month (1-31)
        df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        
        # Month (1-12)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        features_created = [
            'day_of_week_sin', 'day_of_week_cos',
            'day_of_month_sin', 'day_of_month_cos',
            'month_sin', 'month_cos'
        ]
        
        logger.info(f"  ✓ Created {len(features_created)} cyclical encoding features")
        self.feature_info['created_features'].extend(features_created)
        self.feature_info['feature_groups']['cyclical'] = features_created
        
        return df
    
    def build_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("\n[9/9] Building Interaction Features...")
        
        features_created = []
        
        # Weekend * Revenue
        df['weekend_revenue_interaction'] = df['is_weekend'] * df['daily_revenue']
        
        # Failure rate * Revenue
        df['failure_revenue_interaction'] = df['failure_rate'] * df['daily_revenue']
        
        # Month end * Revenue
        df['month_end_revenue_interaction'] = df['is_month_end'] * df['daily_revenue']
        
        # Total payments * Success rate
        df['payments_success_interaction'] = df['total_payments'] * df['success_rate']
        
        features_created = [
            'weekend_revenue_interaction', 'failure_revenue_interaction',
            'month_end_revenue_interaction', 'payments_success_interaction'
        ]
        
        logger.info(f"  ✓ Created {len(features_created)} interaction features")
        self.feature_info['created_features'].extend(features_created)
        self.feature_info['feature_groups']['interaction'] = features_created
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values created by feature engineering
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with handled missing values
        """
        logger.info("\nHandling Missing Values...")
        
        missing_before = df.isnull().sum().sum()
        
        if self.config['handle_missing'] == 'forward_fill':
            df = df.ffill().bfill()
        elif self.config['handle_missing'] == 'interpolate':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
            df = df.bfill()  # For any remaining NaNs at the start
        elif self.config['handle_missing'] == 'drop':
            df = df.dropna()
        
        # Fill any remaining with 0
        df = df.fillna(0)
        
        missing_after = df.isnull().sum().sum()
        
        logger.info(f"  Missing values before: {missing_before}")
        logger.info(f"  Missing values after: {missing_after}")
        logger.info(f"  ✓ Missing values handled using: {self.config['handle_missing']}")
        
        return df
    
    def get_feature_importance_proxy(self, df: pd.DataFrame, target_col: str = 'daily_revenue') -> pd.DataFrame:
        """
        Calculate correlation-based feature importance
        
        Args:
            df: DataFrame with features
            target_col: Target column for correlation
            
        Returns:
            DataFrame with feature correlations sorted by absolute value
        """
        logger.info("\nCalculating Feature Importance Proxy (Correlations)...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col and col != 'metric_date']
        
        correlations = []
        for col in numeric_cols:
            corr = df[col].corr(df[target_col])
            if not np.isnan(corr):
                correlations.append({
                    'feature': col,
                    'correlation': corr,
                    'abs_correlation': abs(corr)
                })
        
        corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
        
        logger.info(f"  Top 5 correlated features with {target_col}:")
        for idx, row in corr_df.head(5).iterrows():
            logger.info(f"    {row['feature']}: {row['correlation']:.4f}")
        
        return corr_df
    
    def save_feature_info(self, filepath: str = 'feature_engineering_report.json') -> None:
        """Save feature engineering report to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.feature_info, f, indent=2)
        logger.info(f"\nFeature engineering report saved to: {filepath}")


# Simple function wrapper (backward compatibility with your original code)
def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick feature engineering function
    (Wrapper around FeatureEngineer class)
    
    Args:
        df: DataFrame with validated data
        
    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer()
    return engineer.build_all_features(df)


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
    
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'metric_date': dates,
        'daily_revenue': np.random.uniform(20000, 50000, len(dates)) + \
                        np.sin(np.arange(len(dates)) * 2 * np.pi / 7) * 5000,  # Weekly pattern
        'successful_payments': np.random.randint(200, 600, len(dates)),
        'failed_payments': np.random.randint(20, 80, len(dates)),
    })
    
    # Test feature engineering
    print("Testing Feature Engineering Module\n")
    
    engineer = FeatureEngineer()
    features_df = engineer.build_all_features(sample_data.copy())
    
    print(f"\nFinal shape: {features_df.shape}")
    print(f"\nSample features:")
    print(features_df.head())
    
    # Get feature importance
    importance = engineer.get_feature_importance_proxy(features_df)
    
    # Save report
    engineer.save_feature_info('feature_engineering_report.json')
    
    # Save features
    features_df.to_csv('features_output.csv', index=False)
    print("\n✓ Features saved to: features_output.csv")



# =================================================
# PIPELINE COMPATIBILITY ADAPTER
# =================================================

class FeatureBuilder:
    """
    Adapter so pipeline can use FeatureEngineer internally.
    Keeps pipeline interface clean.
    """

    def __init__(self, config=None):
        self.engineer = FeatureEngineer(config=config)

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.engineer.build_all_features(df)
