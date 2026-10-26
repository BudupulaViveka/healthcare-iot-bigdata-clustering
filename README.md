# Scalable Clustering of Heterogeneous Big Data via Distributed Ensemble Gradient-boosting Models (XGBoost on Spark/Dask), Applied to Healthcare IoT Patient Monitoring at Scale

**Course:** Big Data Analytics  
**Faculty Guide:** Ms. G. Hema Sri  
**Academic Year:** 2026–2027  

---

## 1. Project Overview

This project presents a scalable, high-throughput distributed big data machine learning pipeline designed for **Healthcare Internet of Things (IoT) Patient Monitoring**. Modern intensive care and remote ambulatory setups generate continuous, heterogeneous data streams combining:
- **Continuous high-frequency vitals:** Heart Rate ($HR$), Systolic and Diastolic Blood Pressure ($SBP/DBP$), Oxygen Saturation ($SpO_2$), Respiration Rate ($RR$), Body Temperature, and ECG Heart Rate Variability ($SDNN$).
- **Static / categorical clinical context:** Demographics (Age bracket, Gender) and chronic comorbidities (Hypertension, Diabetes, COPD, Cardiovascular history).
- **IoT hardware telemetry:** Signal Quality Indicator (SQI), packet transmission latency, battery percentages.

### Why Distributed Gradient-Boosting for Clustering?
Traditional distance-based clustering algorithms (such as standard Euclidean K-Means, DBSCAN, or GMM) deteriorate when applied to multi-modal heterogeneous medical streams due to scale disparities and non-linear physiological couplings. 

This framework implements **Unsupervised Contrastive Gradient Boosting (Distributed XGBoost on Apache Spark)**:
1. Constructs an empirical marginal reference distribution destroying joint physiological correlations.
2. Trains a distributed XGBoost ensemble using histogram-based split finding across Spark worker partitions.
3. Maps raw patient records into high-dimensional **sparse leaf-node embeddings** $\mathbf{\Phi}(\mathbf{x}) \in \{0, 1\}^{T \cdot L}$.
4. Deploys **Distributed Bisecting K-Means** on Apache Spark MLlib over the learned latent representation to discover 5 clinically meaningful patient phenotypes without human supervision:
   - **Cluster 0:** Stable Baseline / Low Risk
   - **Cluster 1:** Sepsis / Early Septic Shock Onset
   - **Cluster 2:** Acute Respiratory Distress (ARDS) / Hypoxemia
   - **Cluster 3:** Hypertensive Crisis / Severe Cardiac Strain
   - **Cluster 4:** Telemetry Sensor Noise & Electrode Detachment

---

## 2. Directory Structure

```text
healthcare_iot_bigdata_clustering/
│
├── README.md                          # Project documentation and quick start
├── requirements.txt                   # Python and Big Data library dependencies
│
├── docs/                              # Academic project documentation
│   ├── SYNOPSIS.md                    # Formal Project Synopsis & Proposal for Ms. G. Hema Sri
│   └── ARCHITECTURE.md                # System Architecture, sequence diagrams & sizing
│
├── src/                               # Source code modules
│   ├── data_generator.py              # Heterogeneous Healthcare IoT telemetry generator
│   ├── spark_pipeline.py              # Apache PySpark distributed MLlib clustering pipeline
│   ├── clustering_engine.py           # Contrastive XGBoost leaf-embedding & clustering engine
│   ├── evaluate_and_visualize.py      # Radar charts, feature importance & scalability plots
│   └── run_pipeline.py                # Master CLI orchestrator
│
├── data/                              # Simulated IoT telemetry datasets (CSV / Parquet)
└── results/                           # Generated plots, metrics JSON, and clustered outputs
```

---

## 3. Installation & Prerequisites

### 3.1 Python Environment
Ensure Python 3.9+ is installed:
```bash
pip install -r requirements.txt
```

### 3.2 Apache Spark Setup (Optional for Distributed Cluster Mode)
If running on a local machine or standalone Spark cluster:
- **Java:** JDK 8 or 11 (`JAVA_HOME` set)
- **Apache Spark:** Version 3.4.0+ (`SPARK_HOME` set)

*Note: The system contains an auto-detecting fallback engine that seamlessly runs the exact same contrastive XGBoost algorithm in high-performance multi-threaded CPU mode if PySpark or Java is not present.*

---

## 4. How to Run

### Quick Start (Master CLI)
Execute the end-to-end pipeline with 25,000 patient telemetry records:
```bash
python src/run_pipeline.py --samples 25000 --clusters 5 --framework auto
```

### Distributed Spark Cluster Execution
To execute directly via PySpark on a distributed cluster:
```bash
python src/run_pipeline.py --samples 100000 --clusters 5 --framework spark
```
Or via `spark-submit`:
```bash
spark-submit --master spark://master:7077 --deploy-mode client \
  --driver-memory 4G --executor-memory 8G --executor-cores 4 \
  src/run_pipeline.py --samples 250000 --framework spark
```

### Generating Custom Scale Datasets
Generate 500,000 synthetic patient telemetry records in Parquet format:
```bash
python src/data_generator.py --samples 500000 --out data/iot_stream_500k.parquet
```

---

## 5. Experimental Results & Deliverables

Running the pipeline automatically generates publication-ready artifacts in `results/`:
1. `clinical_phenotypes_radar.png`: Polar multi-vital radar chart profiling the 5 patient cohorts.
2. `feature_importance.png`: Relative information gain ranking of physiological vitals.
3. `metric_comparison.png`: Comparison of Silhouette score ($SC$) and Davies-Bouldin index ($DBI$) against baseline raw Euclidean K-Means.
4. `scalability_benchmark.png`: Execution speedup and wall-clock times from 10K to 1M records.
5. `patient_clustered_phenotypes.csv`: Telemetry records with patient IDs and assigned cluster labels.
6. `metrics_summary.json`: Persisted validation metrics.

---

## 6. Citations & References
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. ACM SIGKDD.
- Zaharia, M., et al. (2016). *Apache Spark: A unified engine for big data processing*. Communications of the ACM.
- Goldberger, A. L., et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet: Complex physiologic signals*. Circulation.
