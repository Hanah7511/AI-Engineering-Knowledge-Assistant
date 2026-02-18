
import pandas as pd
import os
from typing import Optional
from contextlib import contextmanager


class DatabaseConfig:
    """Database configuration management"""
    
    def __init__(self):
        # Try to get from environment variables first, fall back to defaults
        self.driver = os.getenv('DB_DRIVER', 'SQL Server')
        self.server = os.getenv('DB_SERVER', 'localhost\\SQLEXPRESS')
        self.database = os.getenv('DB_NAME', 'BusinessAnalyticsDB')
        self.trusted_connection = os.getenv('DB_TRUSTED_CONNECTION', 'yes')
        
        # Optional username/password for non-Windows authentication
        self.username = os.getenv('DB_USERNAME')
        self.password = os.getenv('DB_PASSWORD')
    
    def get_connection_string(self) -> str:
        """Build connection string based on configuration"""
        if self.username and self.password:
            # SQL Server authentication
            return (
                f"DRIVER={{{self.driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.username};"
                f"PWD={self.password};"
            )
        else:
            # Windows authentication
            return (
                f"DRIVER={{{self.driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"Trusted_Connection={self.trusted_connection};"
            )


@contextmanager
def get_sql_connection(config: Optional[DatabaseConfig] = None):
    """
    Create a database connection with proper context management.
    
    Args:
        config: DatabaseConfig instance. If None, uses default configuration.
        
    Yields:
        pyodbc.Connection: Database connection object
        
    Raises:
        pyodbc.Error: If connection fails
    """
    if config is None:
        config = DatabaseConfig()
    
    conn = None
    try:
        conn = pyodbc.connect(config.get_connection_string())
        yield conn
    except pyodbc.Error as e:
        print(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def load_daily_metrics(config: Optional[DatabaseConfig] = None, 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
    import pyodbc
    """
    Load daily business metrics from the database.
    
    Args:
        config: DatabaseConfig instance. If None, uses default configuration.
        start_date: Optional start date filter (YYYY-MM-DD format)
        end_date: Optional end date filter (YYYY-MM-DD format)
        
    Returns:
        pd.DataFrame: DataFrame containing the metrics data
        
    Raises:
        pyodbc.Error: If database query fails
        ValueError: If date format is invalid
    """
    try:
        # Build the base query
        query = """
        SELECT *
        FROM daily_business_metrics
        WHERE 1=1
        """
        
        # Add date filters if provided
        params = []
        if start_date:
            query += " AND metric_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND metric_date <= ?"
            params.append(end_date)
            
        query += "\nORDER BY metric_date"
        
        # Execute query with context manager
        with get_sql_connection(config) as conn:
            if params:
                df = pd.read_sql(query, conn, params=params, parse_dates=['metric_date'])
            else:
                df = pd.read_sql(query, conn, parse_dates=['metric_date'])
        
        print(f"Successfully loaded {len(df)} records")
        return df
        
    except pyodbc.Error as e:
        print(f"Database query error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error while loading data: {e}")
        raise


def get_data_summary(df: pd.DataFrame) -> None:
    """
    Print a comprehensive summary of the loaded data.
    
    Args:
        df: DataFrame to summarize
    """
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print("\nColumn Information:")
    print(df.dtypes)
    
    print("\nData Preview (First 5 rows):")
    print(df.head())
    
    if 'metric_date' in df.columns:
        print(f"\nDate Range:")
        print(f"  Start: {df['metric_date'].min()}")
        print(f"  End:   {df['metric_date'].max()}")
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    if missing.any():
        print(missing[missing > 0])
    else:
        print("  No missing values found")
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    print("\n" + "="*60)

# =================================================
# PIPELINE ADAPTER FUNCTION
# =================================================
def extract_metrics_data():
    config = DatabaseConfig()
    df = load_daily_metrics(config)
    return df

def main():
    """Main execution function"""
    try:
        # Load configuration
        config = DatabaseConfig()
        
        print("Connecting to database...")
        print(f"Server: {config.server}")
        print(f"Database: {config.database}")
        
        # Load data
        # Example with date filters:
        # df = load_daily_metrics(config, start_date='2024-01-01', end_date='2024-12-31')
        df = load_daily_metrics(config)
        
        # Display summary
        get_data_summary(df)
        
        # Optional: Save to CSV for backup/analysis
        output_file = 'daily_metrics_export.csv'
        df.to_csv(output_file, index=False)
        print(f"\nData exported to: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"\nScript failed with error: {e}")
        raise


if __name__ == "__main__":
    df = main()
