###=============Business Anomaly Intelligence==============###
ML Pipeline with Baseline + Isolation Forest + Lite MLOps (CI/CD)


###====================== Overview ======================###

Business Anomaly Intelligence is a production-style machine learning pipeline designed to detect anomalies in business metrics such as:
--Daily Revenue
--Payment Success/Failure Counts
--Transaction Patterns

The system combines:
--Statistical Baseline Detection
--Isolation Forest (ML-based anomaly detection)
--Business Impact Evaluation
--Monitoring & Drift Checks
--CI-enabled automated pipeline execution

This project demonstrates end-to-end ML pipeline design with Lite MLOps practices.


###====================== System Architecture ======================###
                ┌────────────────────────────┐
                │      Data Source Layer     │
                │  - SQL Database (Prod)     │
                │  - Synthetic Data (CI)     │
                └─────────────┬──────────────┘
                              ↓
                ┌────────────────────────────┐
                │     Data Validation Layer  │
                │  - Schema checks           │
                │  - Missing values          │
                │  - Business logic rules    │
                └─────────────┬──────────────┘
                              ↓
                ┌────────────────────────────┐
                │   Feature Engineering      │
                │  - Time features           │
                │  - Rolling statistics      │
                │  - Lag features            │
                │  - Trend features          │
                │  - Interaction features    │
                └─────────────┬──────────────┘
                              ↓
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ↓                                           ↓
┌────────────────────┐                   ┌────────────────────┐
│ Baseline Detector  │                   │   ML Detector      │
│ (Statistical)      │                   │ (Isolation Forest) │
│ - Z-score          │                   │ - Scaled features  │
│ - IQR              │                   │ - Contamination    │
│ - MAD              │                   │ - Anomaly scores   │
└────────────┬───────┘                   └────────────┬───────┘
             ↓                                        ↓
              └──────────────┬────────────────────────┘
                             ↓
                ┌────────────────────────────┐
                │     Evaluation Layer       │
                │  - Model agreement         │
                │  - Business impact         │
                │  - Alert rate analysis     │
                └─────────────┬──────────────┘
                              ↓
                ┌────────────────────────────┐
                │      Monitoring Layer      │
                │  - Data health             │
                │  - Drift detection         │
                │  - Alert monitoring        │
                └─────────────┬──────────────┘
                              ↓
                ┌────────────────────────────┐
                │  Outputs & Artifacts       │
                │  - Anomaly flags           │
                │  - Scores                  │
                │  - Logs                    │
                │  - CI validation           │
                └────────────────────────────┘


###======================= Pipeline Stages =======================###

#------------------------------
Stage 1 — Data Extraction
#------------------------------
--SQL-based extraction (production mode)
--Synthetic data generation (CI mode)
--Supports automated testing without database dependency


#------------------------------
Stage 2 — Data Validation
#------------------------------
Comprehensive validation including:
--Schema checks
--Missing dates detection
--Revenue sanity checks
--Payment consistency validation
--Outlier detection (IQR)
--Business logic validation
Outputs structured validation report.


#------------------------------
Stage 3 — Feature Engineering
#------------------------------
Creates 80+ features including:
--Time-based features (day, month, cyclical encoding)
--Rolling statistics
--Lag features
--Trend features
--Exponential moving averages
--Interaction features
--Statistical summaries
Missing values handled safely.


#------------------------------
Stage 4 — Baseline Detection (Statistical)
#------------------------------
Multi-method anomaly detection:
--Global Z-score
--Rolling Z-score
--IQR method
--MAD method
--Percentile detection
--Ensemble anomaly scoring


#------------------------------
Stage 5 — ML Detection
#------------------------------
Model: Isolation Forest
--Feature scaling
--Contamination-based anomaly threshold
--Anomaly scoring
--Flag generation
--Model persistence support


#------------------------------
Stage 6 — Evaluation
#------------------------------
--Baseline vs ML agreement
--Cohen’s Kappa score
--Business impact analysis
--Alert rate health
--Temporal anomaly pattern analysis


#------------------------------
Stage 7 — Monitoring
#------------------------------
Includes:
--Data schema checks
--Volume freshness checks
--Statistical drift detection
--Model alert rate monitoring
--Prediction drift checks
Outputs health status:
--🟢 Healthy
--🟡 Warning
--🔴 Critical


###================== CI/Lite CD(MLOps Layer) ==================###

This project includes GitHub Actions CI.

On every push to main:
--Environment setup (Python 3.10)
--Dependencies installation
--Pipeline execution in CI mode
--Synthetic data injection
--Full pipeline run validation

CI ensures:
--Reproducibility
--Automated testing
--Code reliability
--Deployment readiness
This demonstrates foundational MLOps capability.


###================== Tech Stack ==================###

--Python 3.10
--Pandas
--NumPy
--Scikit-learn
--Isolation Forest
--Logging Framework
--GitHub Actions (CI/CD)


###================== Sample Output ==================###

Pipeline output includes:
--baseline_anomaly_flag
--ml_anomaly_flag
--ml_anomaly_score
--Evaluation metrics
--Monitoring reports
--Feature summary (80+ features generated)

 metric_date  daily_revenue  ...  ml_anomaly_flag  ml_anomaly_score
42  2024-02-12   21515.372136  ...                0         -0.088579
43  2024-02-13   20530.185628  ...                0         -0.097466
44  2024-02-14   20583.434503  ...                0         -0.095230
45  2024-02-15   18765.696453  ...                0         -0.061334
46  2024-02-16   22485.451532  ...                0         -0.065957


###================== How to Run Locally ==================###

1️.Install dependencies
pip install -r requirements.txt

2.Run the pipeline
python -m src.pipeline.run_pipeline


###================== Key Highlights ==================###

--Modular architecture
--Baseline + ML hybrid approach
--Business-aware anomaly evaluation
--Monitoring layer included
--CI-enabled reproducibility
--Production-style logging


###================== Future Enhancements ==================###

--Integrate real-time streaming data (Kafka / API ingestion)
--Add experiment tracking (MLflow)
--Containerize with Docker for production deployment
--Deploy as REST API (FastAPI)
--Add automated model retraining
--Integrate monitoring dashboard (Grafana / Streamlit)


###================== Project Classification ==================###

This project represents:
"End-to-End ML Pipeline + Lite MLOps Implementation"
Not just model training — but full lifecycle design.


Author

Hanah Al Haris
AI / ML Engineering Student