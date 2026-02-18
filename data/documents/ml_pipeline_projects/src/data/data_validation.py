import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Comprehensive data validation for business metrics
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize validator with configuration
        
        Args:
            config: Validation rules and thresholds
        """
        self.config = config or self._default_config()
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'warnings': [],
            'errors': [],
            'passed': True
        }
        
    def _default_config(self) -> Dict:
        """Default validation configuration"""
        return {
            'missing_threshold': 0.05,  # 5% missing values max
            'outlier_threshold': 3.0,   # IQR multiplier for outliers
            'min_revenue': 0,
            'max_revenue': 1e9,  # 1 billion
            'required_columns': [
                'metric_date', 'daily_revenue', 
                'failed_payments', 'successful_payments'
            ],
        }
    
    def validate_all(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Run all validation checks
        
        Args:
            df: Input dataframe with business metrics
            
        Returns:
            (is_valid, validation_report)
        """
        logger.info("="*60)
        logger.info("STARTING DATA VALIDATION")
        logger.info("="*60)

        # SAFETY GUARD 1 — Empty DataFrame Protection
        if df is None or df.empty:
            self._log_error("Input dataframe is empty or None")
            return False, self.validation_results
        
        # Run all validation checks
        self._validate_schema(df)
        self._validate_missing_dates(df)
        self._validate_revenue(df)
        self._validate_payment_counts(df)
        self._validate_missing_values(df)
        self._validate_data_types(df)
        self._validate_duplicates(df)
        self._validate_outliers(df)
        self._validate_business_logic(df)
        
        # Summary
        self._print_summary()
        
        return self.validation_results['passed'], self.validation_results
    
    # =====================================
    # PIPELINE COMPATIBILITY ADAPTER
    # =====================================
    def run_validation(self, df):

        """
    
        Adapter so pipeline can call validation safely.
        Pipeline expects:
           validator.run_validation(df) -> dict

         """

        is_valid, report = self.validate_all(df)

    # Pipeline expects dict with "passed"
        return {
        "passed": is_valid,
        "details": report
    }

    
    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Check if required columns exist"""
        logger.info("\n[1/9] Validating Schema...")
        
        missing_cols = set(self.config['required_columns']) - set(df.columns)
        
        if missing_cols:
            self._log_error(f"Missing required columns: {missing_cols}")
            self.validation_results['checks']['schema'] = {
                'passed': False,
                'missing_columns': list(missing_cols)
            }
        else:
            logger.info(f"✓ All {len(self.config['required_columns'])} required columns present")
            self.validation_results['checks']['schema'] = {
                'passed': True,
                'total_columns': len(df.columns)
            }
    
    def _validate_missing_dates(self, df: pd.DataFrame) -> None:
        """Check for missing dates in time series"""
        logger.info("\n[2/9] Checking Missing Dates...")
        
        if 'metric_date' not in df.columns:
            self._log_error("'metric_date' column not found")
            return
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['metric_date']):
            df['metric_date'] = pd.to_datetime(df['metric_date'])
        
        expected_dates = pd.date_range(
            df['metric_date'].min(), 
            df['metric_date'].max(),
            freq='D'
        )
        actual_dates = pd.to_datetime(df['metric_date'].unique())
        missing_dates = expected_dates.difference(actual_dates)
        
        if len(missing_dates) > 0:
            self._log_warning(f"Found {len(missing_dates)} missing dates")
            logger.info(f"  Date range: {df['metric_date'].min()} to {df['metric_date'].max()}")
            logger.info(f"  First missing: {missing_dates[0]}")
            
            self.validation_results['checks']['missing_dates'] = {
                'passed': True,  # Warning only
                'count': len(missing_dates),
                'percentage': len(missing_dates) / len(expected_dates) * 100,
                'first_missing': str(missing_dates[0]) if len(missing_dates) > 0 else None
            }
        else:
            logger.info("✓ No missing dates - complete time series")
            self.validation_results['checks']['missing_dates'] = {
                'passed': True,
                'count': 0
            }
    
    def _validate_revenue(self, df: pd.DataFrame) -> None:
        """Validate revenue values"""
        logger.info("\n[3/9] Validating Revenue...")
        
        if 'daily_revenue' not in df.columns:
            self._log_error("'daily_revenue' column not found")
            return
        
        issues = []
        
        # Check for negative values
        negative_count = (df['daily_revenue'] < 0).sum()
        if negative_count > 0:
            self._log_error(f"Found {negative_count} negative revenue values")
            issues.append(f"negative_values: {negative_count}")
        
        # Check for null values
        null_count = df['daily_revenue'].isnull().sum()
        if null_count > 0:
            self._log_warning(f"Found {null_count} null revenue values")
            issues.append(f"null_values: {null_count}")
        
        # Check for unrealistic values
        max_revenue = df['daily_revenue'].max()
        if max_revenue > self.config['max_revenue']:
            self._log_warning(f"Revenue exceeds expected max: ${max_revenue:,.2f}")
            issues.append(f"exceeds_max: {max_revenue}")
        
        # Check for zeros
        zero_count = (df['daily_revenue'] == 0).sum()
        if zero_count > 0:
            logger.info(f"  ⚠ Found {zero_count} days with zero revenue")
        
        if len(issues) == 0:
            logger.info("✓ All revenue values are valid")
        
        self.validation_results['checks']['revenue'] = {
            'passed': negative_count == 0,
            'negative_count': int(negative_count),
            'null_count': int(null_count),
            'zero_count': int(zero_count),
            'min': float(df['daily_revenue'].min()),
            'max': float(df['daily_revenue'].max()),
            'mean': float(df['daily_revenue'].mean()),
            'issues': issues
        }
    
    def _validate_payment_counts(self, df: pd.DataFrame) -> None:
        """Validate payment count consistency"""
        logger.info("\n[4/9] Validating Payment Counts...")
        
        required_cols = ['failed_payments', 'successful_payments']
        if not all(col in df.columns for col in required_cols):
            self._log_error(f"Missing payment columns")
            return
        
        issues = []
        
        # Check for negative values
        for col in required_cols:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                self._log_error(f"Found {negative_count} negative values in '{col}'")
                issues.append(f"{col}_negative: {negative_count}")
        
        # Check total payment consistency
        total_payments = df['failed_payments'] + df['successful_payments']
        
        # Payments should be non-negative
        if (total_payments < 0).any():
            invalid_count = (total_payments < 0).sum()
            self._log_error(f"Found {invalid_count} invalid total payment counts")
            issues.append(f"invalid_totals: {invalid_count}")
        
        # Check for days with zero payments but positive revenue
        zero_payments = (total_payments == 0) & (df['daily_revenue'] > 0)
        if zero_payments.any():
            self._log_warning(f"Found {int(zero_payments.sum())} days with revenue but no payments")
            issues.append(f"revenue_without_payments: {int(zero_payments.sum())}")
        
        # Calculate success rate
        success_rate_series = df['successful_payments'] / total_payments
        success_rate_series = success_rate_series.replace([np.inf, -np.inf], np.nan)
        success_rate = success_rate_series.mean() * 100

        
        if len(issues) == 0:
            logger.info("✓ Payment counts are valid")
        
        logger.info(f"  Average payment success rate: {success_rate:.2f}%")
        
        self.validation_results['checks']['payment_counts'] = {
            'passed': len([i for i in issues if 'negative' in i or 'invalid' in i]) == 0,
            'issues': issues,
            'avg_success_rate': float(success_rate),
            'total_failed': int(df['failed_payments'].sum()),
            'total_successful': int(df['successful_payments'].sum())
        }
    
    def _validate_missing_values(self, df: pd.DataFrame) -> None:
        """Check for missing values across all columns"""
        logger.info("\n[5/9] Checking Missing Values...")
        
        missing_summary = {}
        total_cells = len(df) * len(df.columns)
        total_missing = 0
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_pct = missing_count / len(df) * 100
                missing_summary[col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_pct)
                }
                total_missing += missing_count
                
                if missing_pct > self.config['missing_threshold'] * 100:
                    self._log_warning(f"Column '{col}': {missing_pct:.2f}% missing (threshold: {self.config['missing_threshold']*100}%)")
        
        if len(missing_summary) == 0:
            logger.info("✓ No missing values found")
        else:
            logger.info(f"  Found missing values in {len(missing_summary)} columns")
        
        self.validation_results['checks']['missing_values'] = {
            'passed': True,  # Just a warning
            'total_missing': int(total_missing),
            'columns_affected': len(missing_summary),
            'details': missing_summary
        }
    
    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """Validate expected data types"""
        logger.info("\n[6/9] Validating Data Types...")
        
        type_issues = []
        
        # Check metric_date is datetime
        if 'metric_date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['metric_date']):
                type_issues.append("metric_date should be datetime")
        
        # Check numeric columns
        numeric_cols = ['daily_revenue', 'failed_payments', 'successful_payments']
        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    type_issues.append(f"'{col}' should be numeric")
        
        if len(type_issues) > 0:
            for issue in type_issues:
                self._log_warning(issue)
        else:
            logger.info("✓ All data types are correct")
        
        self.validation_results['checks']['data_types'] = {
            'passed': len(type_issues) == 0,
            'issues': type_issues
        }
    
    def _validate_duplicates(self, df: pd.DataFrame) -> None:
        """Check for duplicate records"""
        logger.info("\n[7/9] Checking for Duplicates...")
        
        # Check for duplicate dates
        if 'metric_date' in df.columns:
            duplicate_dates = df['metric_date'].duplicated().sum()
            
            if duplicate_dates > 0:
                self._log_warning(f"Found {duplicate_dates} duplicate dates")
                self.validation_results['checks']['duplicates'] = {
                    'passed': False,
                    'duplicate_dates': int(duplicate_dates)
                }
            else:
                logger.info("✓ No duplicate dates found")
                self.validation_results['checks']['duplicates'] = {
                    'passed': True,
                    'duplicate_dates': 0
                }
    
    def _validate_outliers(self, df: pd.DataFrame) -> None:
        """Detect outliers using IQR method"""
        logger.info("\n[8/9] Detecting Outliers...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_summary = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - self.config['outlier_threshold'] * IQR
            upper_bound = Q3 + self.config['outlier_threshold'] * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outliers > 0:
                outlier_pct = outliers / len(df) * 100
                outlier_summary[col] = {
                    'count': int(outliers),
                    'percentage': float(outlier_pct),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
                logger.info(f"  {col}: {outliers} outliers ({outlier_pct:.2f}%)")
        
        if len(outlier_summary) == 0:
            logger.info("✓ No significant outliers detected")
        
        self.validation_results['checks']['outliers'] = {
            'passed': True,  # Outliers are expected in anomaly detection
            'method': 'IQR',
            'threshold': self.config['outlier_threshold'],
            'details': outlier_summary
        }
    
    def _validate_business_logic(self, df: pd.DataFrame) -> None:
        """Custom business rule validations"""
        logger.info("\n[9/9] Validating Business Logic...")
        
        issues = []
        
        # Rule 1: Revenue should generally correlate with successful payments
        if all(col in df.columns for col in ['daily_revenue', 'successful_payments']):
            correlation = df['daily_revenue'].corr(df['successful_payments'])
            
            if pd.isna(correlation):
                correlation = 0
                self._log_warning("Correlation undefined (constant data or insufficient variation)")

            if correlation < 0.3:  # Low correlation threshold
                self._log_warning(f"Low correlation between revenue and successful payments: {correlation:.3f}")
                issues.append(f"low_correlation: {correlation:.3f}")
            else:
                logger.info(f"  Revenue-Payment correlation: {correlation:.3f}")
        
        # Rule 2: Failed payment rate should be reasonable
        if all(col in df.columns for col in ['failed_payments', 'successful_payments']):
            total_payments = df['failed_payments'] + df['successful_payments']
            failure_rate = (df['failed_payments'] / total_payments).mean() * 100
            
            if failure_rate > 20:  # More than 20% failure rate
                self._log_warning(f"High payment failure rate: {failure_rate:.2f}%")
                issues.append(f"high_failure_rate: {failure_rate:.2f}%")
            else:
                logger.info(f"  Average payment failure rate: {failure_rate:.2f}%")
        
        if len(issues) == 0:
            logger.info("✓ Business logic checks passed")
        
        self.validation_results['checks']['business_logic'] = {
            'passed': len(issues) == 0,
            'issues': issues
        }
    
    def _log_error(self, message: str) -> None:
        """Log an error and mark validation as failed"""
        logger.error(f"✗ ERROR: {message}")
        self.validation_results['errors'].append(message)
        self.validation_results['passed'] = False
    
    def _log_warning(self, message: str) -> None:
        """Log a warning"""
        logger.warning(f"⚠ WARNING: {message}")
        self.validation_results['warnings'].append(message)
    
    def _print_summary(self) -> None:
        """Print validation summary"""
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)
        
        total_checks = len(self.validation_results['checks'])
        passed_checks = sum(1 for check in self.validation_results['checks'].values() if check.get('passed', True))
        
        logger.info(f"Total Checks: {total_checks}")
        logger.info(f"Passed: {passed_checks}")
        logger.info(f"Failed: {total_checks - passed_checks}")
        logger.info(f"Errors: {len(self.validation_results['errors'])}")
        logger.info(f"Warnings: {len(self.validation_results['warnings'])}")
        
        if self.validation_results['passed']:
            logger.info("\n✓ VALIDATION PASSED - Data is ready for feature engineering")
        else:
            logger.error("\n✗ VALIDATION FAILED - Please fix errors before proceeding")
        
        logger.info("="*60)
    
    def save_report(self, filepath: str = 'validation_report.json') -> None:
        """Save validation report to JSON file"""
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        serializable_results = convert_numpy(self.validation_results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"\nValidation report saved to: {filepath}")


# Simple validation function (your original approach, enhanced)
def validate_daily_metrics(df: pd.DataFrame) -> bool:
    """
    Quick validation function for basic checks
    (Wrapper around DataValidator for backward compatibility)
    
    Args:
        df: DataFrame with daily metrics
        
    Returns:
        Boolean indicating if validation passed
    """
    validator = DataValidator()
    is_valid, report = validator.validate_all(df)
    return is_valid


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
    
    sample_data = pd.DataFrame({
        'metric_date': dates,
        'daily_revenue': np.random.uniform(10000, 50000, len(dates)),
        'successful_payments': np.random.randint(100, 500, len(dates)),
        'failed_payments': np.random.randint(10, 50, len(dates)),
    })
    
    # Add some intentional issues for testing
    sample_data.loc[5, 'daily_revenue'] = -1000  # Negative revenue
    sample_data.loc[10, 'successful_payments'] = np.nan  # Missing value
    
    # Run validation
    print("Testing Validation Module\n")
    validator = DataValidator()
    is_valid, report = validator.validate_all(sample_data)
    
    # Save report
    validator.save_report('validation_report.json')
    
    print(f"\nValidation Status: {'PASSED' if is_valid else 'FAILED'}")
