import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BusinessAnomalyEvaluator")


class BusinessAnomalyEvaluator:
    """
    Comprehensive evaluator for business anomaly detection systems.
    
    Key Features:
    1. Agreement Analysis - Baseline vs ML consensus
    2. Business Impact - Anomaly characteristics
    3. Temporal Patterns - When anomalies occur
    4. Confidence Metrics - Model certainty
    5. Visualization - Easy-to-understand charts
    """
    
    def __init__(self, business_impact_threshold: float = 0.1):
        """
        Args:
            business_impact_threshold: Revenue change considered significant (default 10%)
        """
        self.business_impact_threshold = business_impact_threshold
        logger.info(f"Business Anomaly Evaluator initialized (impact threshold: {business_impact_threshold*100}%)")
    
    #  CORE METRICS 
    
    def baseline_vs_ml_agreement(self, df: pd.DataFrame) -> Dict:
        """
        Compare baseline (statistical) vs ML anomaly decisions.
        
        Returns agreement metrics that show model consensus.
        """
        logger.info("Analyzing Baseline vs ML agreement...")
        
        # Validation
        required_cols = {'daily_revenue_is_anomaly', 'ml_anomaly_flag'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        baseline = df["daily_revenue_is_anomaly"].fillna(0)
        ml = df["ml_anomaly_flag"].fillna(0)
        
        # Core agreement metrics
        total = len(df)
        agreement = (baseline == ml).mean() * 100
        
        # Confusion matrix style breakdown
        true_positives = ((baseline == 1) & (ml == 1)).sum()  # Both agree: anomaly
        false_positives = ((baseline == 0) & (ml == 1)).sum() # ML only says anomaly
        false_negatives = ((baseline == 1) & (ml == 0)).sum() # Baseline only says anomaly
        true_negatives = ((baseline == 0) & (ml == 0)).sum()  # Both agree: normal
        
        # Cohen's Kappa (agreement beyond chance)
        po = agreement / 100
        pe = ((baseline.sum()/total) * (ml.sum()/total) + 
              ((total - baseline.sum())/total) * ((total - ml.sum())/total))
        kappa = (po - pe) / (1 - pe) if pe != 1 else 1.0
        
        results = {
            "agreement_percent": round(agreement, 2),
            "cohens_kappa": round(kappa, 3),
            "true_positives": int(true_positives),
            "false_positives": int(false_positives),
            "false_negatives": int(false_negatives),
            "true_negatives": int(true_negatives),
            "precision": round(true_positives / (true_positives + false_positives + 1e-8), 3),
            "recall": round(true_positives / (true_positives + false_negatives + 1e-8), 3)
        }
        
        logger.info(f"Agreement: {agreement:.1f}% | Kappa: {kappa:.3f}")
        return results
    
    # ==================== BUSINESS IMPACT ====================
    
    def analyze_anomaly_impact(self, df: pd.DataFrame) -> Dict:
        """
        Analyze business impact of detected anomalies.
        
        For each anomaly, calculate revenue impact and other metrics.
        """
        logger.info("Analyzing business impact of anomalies...")
        
        if 'daily_revenue' not in df.columns:
            logger.warning("Revenue data missing - skipping impact analysis")
            return {}
        
        # Separate anomalies and normal days
        if 'ml_anomaly_flag' in df.columns:
            anomalies = df[df['ml_anomaly_flag'] == 1]
            normal_days = df[df['ml_anomaly_flag'] == 0]
        elif 'daily_revenue_is_anomaly' in df.columns:
            anomalies = df[df['daily_revenue_is_anomaly'] == 1]
            normal_days = df[df['daily_revenue_is_anomaly'] == 0]
        else:
            return {}
        
        if len(anomalies) == 0:
            return {"no_anomalies_detected": True}
        
        # Calculate impact metrics
        avg_normal_revenue = normal_days['daily_revenue'].mean()
        avg_anomaly_revenue = anomalies['daily_revenue'].mean()
        
        # Percentage change
        revenue_change_pct = ((avg_anomaly_revenue - avg_normal_revenue) / 
                              (avg_normal_revenue + 1e-8)) * 100
        
        # Categorize anomalies
        high_revenue_anoms = anomalies[anomalies['daily_revenue'] > avg_normal_revenue * 1.2]
        low_revenue_anoms = anomalies[anomalies['daily_revenue'] < avg_normal_revenue * 0.8]
        
        results = {
            "total_anomalies": len(anomalies),
            "avg_normal_revenue": round(avg_normal_revenue, 2),
            "avg_anomaly_revenue": round(avg_anomaly_revenue, 2),
            "revenue_change_percent": round(revenue_change_pct, 2),
            "high_revenue_anomalies": len(high_revenue_anoms),
            "low_revenue_anomalies": len(low_revenue_anoms),
            "business_impact": "POSITIVE" if revenue_change_pct > self.business_impact_threshold*100 else
                              "NEGATIVE" if revenue_change_pct < -self.business_impact_threshold*100 else
                              "NEUTRAL"
        }
        
        logger.info(f"Anomaly impact: {revenue_change_pct:+.1f}% revenue change")
        return results
    
    # ==================== TEMPORAL ANALYSIS ====================
    
    def temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze when anomalies occur (day of week, time patterns).
        """
        logger.info("Analyzing temporal patterns...")
        
        if 'metric_date' not in df.columns:
            return {}
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['metric_date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['date'].dt.month
        
        # Anomaly rates by time period
        if 'ml_anomaly_flag' in df.columns:
            anomaly_col = 'ml_anomaly_flag'
        elif 'daily_revenue_is_anomaly' in df.columns:
            anomaly_col = 'daily_revenue_is_anomaly'
        else:
            return {}
        
        weekday_rate = df[df['is_weekend'] == 0][anomaly_col].mean() * 100
        weekend_rate = df[df['is_weekend'] == 1][anomaly_col].mean() * 100
        
        # Most common anomaly days
        anomaly_days = df[df[anomaly_col] == 1]['day_of_week'].value_counts()
        
        results = {
            "weekday_anomaly_rate_percent": round(weekday_rate, 2),
            "weekend_anomaly_rate_percent": round(weekend_rate, 2),
            "weekend_effect_ratio": round(weekend_rate / (weekday_rate + 1e-8), 2),
            "most_common_anomaly_day": int(anomaly_days.index[0]) if len(anomaly_days) > 0 else -1
        }
        
        logger.info(f"Weekday: {weekday_rate:.1f}% | Weekend: {weekend_rate:.1f}% anomalies")
        return results
    
    # ==================== ALERT RATE ANALYSIS ====================
    
    def alert_rate_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Monitor alert rates for stability and drift.
        """
        logger.info("Analyzing alert rates...")
        
        metrics = {}
        
        if 'daily_revenue_is_anomaly' in df.columns:
            baseline_rate = df['daily_revenue_is_anomaly'].mean() * 100
            metrics['baseline_alert_rate_percent'] = round(baseline_rate, 2)
        
        if 'ml_anomaly_flag' in df.columns:
            ml_rate = df['ml_anomaly_flag'].mean() * 100
            metrics['ml_alert_rate_percent'] = round(ml_rate, 2)
            
            # Alert rate stability (rolling window)
            if len(df) > 30:
                rolling_rate = df['ml_anomaly_flag'].rolling(window=30).mean() * 100
                metrics['alert_rate_std'] = round(rolling_rate.std(), 2)
                metrics['max_alert_rate'] = round(rolling_rate.max(), 2)
                metrics['min_alert_rate'] = round(rolling_rate.min(), 2)
        
        # Ideal alert rate (industry benchmark: 2-5% for business metrics)
        if 'ml_alert_rate_percent' in metrics:
            rate = metrics['ml_alert_rate_percent']
            if rate < 2:
                metrics['alert_rate_assessment'] = "TOO_LOW"
            elif rate > 5:
                metrics['alert_rate_assessment'] = "TOO_HIGH"
            else:
                metrics['alert_rate_assessment'] = "OPTIMAL"
        
        return metrics
    
    # ==================== CONFIDENCE ANALYSIS ====================
    
    def confidence_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Analyze model confidence scores.
        """
        if 'ml_anomaly_score' not in df.columns:
            return {}
        
        scores = df['ml_anomaly_score']
        anomalies = df[df['ml_anomaly_flag'] == 1]['ml_anomaly_score'] if 'ml_anomaly_flag' in df.columns else pd.Series()
        normal = df[df['ml_anomaly_flag'] == 0]['ml_anomaly_score'] if 'ml_anomaly_flag' in df.columns else pd.Series()
        
        results = {
            "avg_confidence_score": round(scores.mean(), 2),
            "confidence_std": round(scores.std(), 2),
            "anomaly_avg_confidence": round(anomalies.mean(), 2) if len(anomalies) > 0 else 0,
            "normal_avg_confidence": round(normal.mean(), 2) if len(normal) > 0 else 0,
            "confidence_separation": round(abs(anomalies.mean() - normal.mean()), 2) if len(anomalies) > 0 and len(normal) > 0 else 0
        }
        
        # Confidence threshold analysis
        high_conf_anomalies = (scores > 80).sum() if scores.max() > 1 else (scores > 0.8).sum()
        results['high_confidence_anomalies'] = int(high_conf_anomalies)
        
        return results
    
    # ==================== VISUALIZATION ====================
    
    def create_visualizations(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create key visualizations for business presentation.
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Business Anomaly Detection Analysis', fontsize=16)
            
            # 1. Anomaly Timeline
            if 'metric_date' in df.columns and 'ml_anomaly_flag' in df.columns:
                ax = axes[0, 0]
                dates = pd.to_datetime(df['metric_date'])
                ax.plot(dates, df['daily_revenue'] if 'daily_revenue' in df.columns else range(len(df)), 
                       alpha=0.6, label='Revenue')
                anomalies = df[df['ml_anomaly_flag'] == 1]
                if len(anomalies) > 0:
                    anomaly_dates = pd.to_datetime(anomalies['metric_date'])
                    ax.scatter(anomaly_dates, anomalies['daily_revenue'] if 'daily_revenue' in anomalies.columns else [0]*len(anomalies),
                             color='red', s=50, zorder=5, label='Anomalies')
                ax.set_title('Anomaly Timeline')
                ax.legend()
                ax.tick_params(axis='x', rotation=45)
            
            # 2. Agreement Matrix
            if all(col in df.columns for col in ['daily_revenue_is_anomaly', 'ml_anomaly_flag']):
                ax = axes[0, 1]
                agreement = (df['daily_revenue_is_anomaly'] == df['ml_anomaly_flag']).mean() * 100
                labels = ['Disagree', 'Agree']
                sizes = [100 - agreement, agreement]
                colors = ['lightcoral', 'lightgreen']
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.set_title(f'Baseline vs ML Agreement\n({agreement:.1f}% agreement)')
            
            # 3. Alert Rate Over Time
            if 'metric_date' in df.columns and 'ml_anomaly_flag' in df.columns:
                ax = axes[1, 0]
                df['date'] = pd.to_datetime(df['metric_date'])
                df.set_index('date', inplace=True)
                rolling_rate = df['ml_anomaly_flag'].rolling('7D').mean() * 100
                ax.plot(rolling_rate, color='blue', linewidth=2)
                ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Upper Limit (5%)')
                ax.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Lower Limit (2%)')
                ax.set_title('7-Day Rolling Alert Rate')
                ax.set_ylabel('Alert Rate (%)')
                ax.legend()
                df.reset_index(inplace=True)
            
            # 4. Confidence Distribution
            if 'ml_anomaly_score' in df.columns and 'ml_anomaly_flag' in df.columns:
                ax = axes[1, 1]
                anomalies = df[df['ml_anomaly_flag'] == 1]['ml_anomaly_score']
                normal = df[df['ml_anomaly_flag'] == 0]['ml_anomaly_score']
                ax.hist([normal, anomalies], bins=20, label=['Normal', 'Anomaly'], 
                       alpha=0.7, color=['blue', 'red'])
                ax.set_title('Confidence Score Distribution')
                ax.set_xlabel('Anomaly Score')
                ax.set_ylabel('Count')
                ax.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Visualizations saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.warning(f"Could not create visualizations: {e}")
    
    # ==================== COMPREHENSIVE EVALUATION ====================
    
    def run_full_evaluation(self, df: pd.DataFrame, 
                           create_plots: bool = True,
                           plot_save_path: Optional[str] = None) -> Dict:
        """
        Run complete evaluation pipeline.
        
        Args:
            df: DataFrame with anomaly detection results
            create_plots: Whether to generate visualizations
            plot_save_path: Where to save plots (optional)
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("=" * 60)
        logger.info("RUNNING COMPREHENSIVE EVALUATION")
        logger.info("=" * 60)
        
        results = {
            "model_agreement": self.baseline_vs_ml_agreement(df),
            "business_impact": self.analyze_anomaly_impact(df),
            "temporal_patterns": self.temporal_patterns(df),
            "alert_rates": self.alert_rate_analysis(df),
            "confidence_analysis": self.confidence_analysis(df)
        }
        
        # Generate summary statistics
        total_days = len(df)
        results["summary"] = {
            "total_days_analyzed": total_days,
            "evaluation_timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Create visualizations if requested
        if create_plots:
            self.create_visualizations(df, plot_save_path)
        
        # Log key findings
        self._log_evaluation_summary(results)
        
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        
        return results
    
    def _log_evaluation_summary(self, results: Dict):
        """Log key evaluation findings."""
        agreement = results.get("model_agreement", {})
        impact = results.get("business_impact", {})
        alerts = results.get("alert_rates", {})
        
        logger.info("\n📊 EVALUATION SUMMARY:")
        logger.info(f"  • Model Agreement: {agreement.get('agreement_percent', 'N/A')}%")
        logger.info(f"  • Total Anomalies: {impact.get('total_anomalies', 'N/A')}")
        logger.info(f"  • ML Alert Rate: {alerts.get('ml_alert_rate_percent', 'N/A')}%")
        logger.info(f"  • Business Impact: {impact.get('business_impact', 'N/A')}")


# ==================== DEMONSTRATION ====================

def create_demo_data():
    """Create sample data for demonstration."""
    dates = pd.date_range("2024-01-01", periods=100, freq='D')
    
    # Generate realistic data with anomalies
    np.random.seed(42)
    revenue = 50000 + np.random.normal(0, 5000, 100)
    
    # Add weekly pattern
    revenue += np.sin(np.arange(100) * 2 * np.pi / 7) * 3000
    
    # Add anomalies
    revenue[30] = 120000  # Spike
    revenue[65] = 20000   # Drop
    revenue[85] = 85000   # Spike
    
    # Create anomaly flags (simulated detection)
    df = pd.DataFrame({
        'metric_date': dates,
        'daily_revenue': revenue,
        'daily_revenue_is_anomaly': np.random.choice([0, 1], 100, p=[0.95, 0.05]),
        'ml_anomaly_flag': np.random.choice([0, 1], 100, p=[0.93, 0.07]),
        'ml_anomaly_score': np.random.uniform(0, 100, 100)
    })
    
    # Make some anomalies overlap
    df.loc[[30, 65], 'daily_revenue_is_anomaly'] = 1
    df.loc[[30, 85], 'ml_anomaly_flag'] = 1
    
    return df

def main():
    """Demonstrate the evaluator."""
    print("=" * 70)
    print("BUSINESS ANOMALY DETECTION EVALUATOR DEMO")
    print("=" * 70)
    
    # Create sample data
    print("\n📁 Loading demo data...")
    data = create_demo_data()
    print(f"   Loaded {len(data)} days of business metrics")
    
    # Initialize evaluator
    print("\n📊 Initializing evaluator...")
    evaluator = BusinessAnomalyEvaluator(business_impact_threshold=0.1)
    
    # Run comprehensive evaluation
    print("\n🔍 Running evaluation...")
    results = evaluator.run_full_evaluation(
        data, 
        create_plots=True,
        plot_save_path="evaluation_report.png"
    )
    
    # Print key results
    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    print("=" * 70)
    
    agreement = results["model_agreement"]
    print(f"Model Agreement: {agreement['agreement_percent']}%")
    print(f"True Positives: {agreement['true_positives']}")
    print(f"False Positives: {agreement['false_positives']}")
    
    impact = results["business_impact"]
    if "revenue_change_percent" in impact:
        print(f"\nBusiness Impact: {impact['revenue_change_percent']:+.1f}% revenue change")
        print(f"Anomaly Assessment: {impact.get('business_impact', 'N/A')}")
    
    alerts = results["alert_rates"]
    print(f"\nAlert Rate: {alerts.get('ml_alert_rate_percent', 'N/A')}%")
    print(f"Alert Stability: {'OPTIMAL' if alerts.get('alert_rate_assessment') == 'OPTIMAL' else 'NEEDS REVIEW'}")
    
    print("\n" + "=" * 70)
    print("✅ Evaluation Complete!")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    results = main()