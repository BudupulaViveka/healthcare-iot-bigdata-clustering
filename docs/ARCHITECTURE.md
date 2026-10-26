# Distributed System Architecture: Scalable Clustering of Heterogeneous Healthcare IoT Big Data

**Project Title:** Scalable Clustering of Heterogeneous Big Data via Distributed Ensemble Gradient-boosting Models (XGBoost on Spark/Dask), Applied to Healthcare IoT Patient Monitoring at Scale  
**Faculty Guide:** Ms. G. Hema Sri  
**Domain:** Big Data Analytics & Distributed Machine Learning  

---

## 1. High-Level End-to-End System Architecture

The following diagram illustrates the multi-tier streaming and batch architecture:

```mermaid
flowchart TD
    subgraph SENSORS ["Layer 1: Healthcare IoT Sensing & Telemetry"]
        ECG["Wearable ECG Patch (HRV, Rhythm)"]
        OX["Pulse Oximeter (SpO2, Pulse)"]
        BP["Smart BP Cuff (SBP, DBP)"]
        THERM["Continuous Temp Sensor"]
        RESP["Chest Impedance Band (Respiration Rate)"]
    end

    subgraph INGESTION ["Layer 2: Distributed Ingestion Tier"]
        KAFKA["Apache Kafka Telemetry Topics (iot-vitals-stream)"]
        SPARK_STREAM["Apache Spark Structured Streaming / Parquet Storage"]
    end

    subgraph PREPROCESSING ["Layer 3: Distributed Spark ML Feature Pipeline"]
        IMPUTE["Spark ML Imputer (Median / Forward-fill)"]
        CAT_ENC["StringIndexer + OneHotEncoder (Demographics, Conditions)"]
        SCALER["StandardScaler + Robust VectorAssembler"]
        HETERO_VEC["Heterogeneous Feature Vector: x_i in R^D"]
    end

    subgraph EMBEDDING ["Layer 4: Distributed Ensemble Gradient-Boosting Representation"]
        SYNTH_GEN["Synthetic Marginal Permutation Generator (D_synth, y=0)"]
        CONTRAST["Contrastive Dataset (D_real, y=1 vs D_synth, y=0)"]
        XGB_DIST["Distributed XGBoost on Spark (Histogram Splitting)"]
        LEAF_EXTRACT["Sparse Leaf Embedding Transformer: Phi(x_i)"]
    end

    subgraph CLUSTERING ["Layer 5: Distributed Clustering & Clinical Phenotyping"]
        SPARK_KMEANS["Distributed Bisecting K-Means on Spark MLlib"]
        CLUSTERS["Clinical Patient Clusters (k=5 Cohorts)"]
    end

    subgraph ALERTING ["Layer 6: Clinical Decision Support & Telemetry Dashboard"]
        C0["Phenotype 0: Stable Baseline"]
        C1["Phenotype 1: Sepsis Warning"]
        C2["Phenotype 2: ARDS / Hypoxemia"]
        C3["Phenotype 3: Hypertensive Crisis"]
        C4["Phenotype 4: Telemetry Sensor Artifact"]
        DASH["Real-Time ICU Monitoring Dashboard"]
    end

    SENSORS --> KAFKA
    KAFKA --> SPARK_STREAM
    SPARK_STREAM --> IMPUTE
    IMPUTE --> CAT_ENC
    CAT_ENC --> SCALER
    SCALER --> HETERO_VEC

    HETERO_VEC --> SYNTH_GEN
    HETERO_VEC --> CONTRAST
    SYNTH_GEN --> CONTRAST
    CONTRAST --> XGB_DIST
    XGB_DIST --> LEAF_EXTRACT

    LEAF_EXTRACT --> SPARK_KMEANS
    SPARK_KMEANS --> CLUSTERS

    CLUSTERS --> C0
    CLUSTERS --> C1
    CLUSTERS --> C2
    CLUSTERS --> C3
    CLUSTERS --> C4

    C0 & C1 & C2 & C3 & C4 --> DASH
```

---

## 2. Distributed Execution Dataflow on Apache Spark

```mermaid
sequenceDiagram
    autonumber
    participant D as IoT Wearables / Data Stream
    participant SM as Spark Driver Master
    participant W1 as Spark Worker 1 (Core 1-4)
    participant W2 as Spark Worker 2 (Core 1-4)
    participant XGB as Distributed XGBoost Engine
    participant KM as Spark MLlib K-Means

    D->>SM: Stream Telemetry Batch (100k records)
    SM->>W1: Partition 1 (50k records)
    SM->>W2: Partition 2 (50k records)
    
    par Distributed Preprocessing
        W1->>W1: Cleanse, Impute, One-Hot Encode Demographics
    and
        W2->>W2: Cleanse, Impute, One-Hot Encode Demographics
    end

    SM->>XGB: Initialize Distributed Tree Boosting (tree_method='hist')
    XGB-->>W1: Build Histogram Bins across features
    XGB-->>W2: Build Histogram Bins across features
    W1->>XGB: Local Gradient & Hessian Sums
    W2->>XGB: Local Gradient & Hessian Sums
    XGB->>SM: Optimal Tree Splits Identified

    par Leaf Embedding Generation
        W1->>W1: Transform raw x -> sparse leaf index vector Phi(x)
    and
        W2->>W2: Transform raw x -> sparse leaf index vector Phi(x)
    end

    SM->>KM: Fit Distributed Bisecting K-Means(k=5)
    KM-->>W1: Compute Distance to Centroids (Map)
    KM-->>W2: Compute Distance to Centroids (Map)
    W1->>KM: Centroid Update Sums (Reduce)
    W2->>KM: Centroid Update Sums (Reduce)

    KM->>SM: Converged Cluster Centroids & Predictions
    SM->>D: Trigger Clinical Phenotype Alerts
```

---

## 3. Clinical Phenotype Profiles (Cluster Taxonomy)

The 5 patient clusters represent clinically distinct physiological states:

| Cluster ID | Phenotype Name | Physiological Signature | Clinical Risk Level | Recommended Action |
|---|---|---|---|---|
| **0** | **Stable Baseline** | HR: 65–80 bpm, $SpO_2$: 97–99%, SBP: 110–125 mmHg, RR: 12–16 bpm, Temp: 36.6–37.2°C | Normal / Safe | Routine periodic monitoring (every 4 hours). |
| **1** | **Early Sepsis / Septic Shock** | HR: 110–145 bpm, Temp: >38.5°C or <36°C, RR: >24 bpm, SBP: <90 mmHg (hypotension) | **Critical (Emergency)** | Immediate SOFA evaluation, blood cultures, IV fluids, and broad-spectrum antibiotics. |
| **2** | **Acute Respiratory Distress (ARDS)** | $SpO_2 < 88\%$, RR: 28–38 bpm, HR: 105–125 bpm, Normal/Moderate BP | **High (Urgent)** | Supplemental oxygen therapy, mechanical ventilation assessment, arterial blood gas. |
| **3** | **Hypertensive Crisis / Cardiac Strain** | SBP > 180 mmHg, DBP > 110 mmHg, HR: 95–120 bpm, Low ECG HRV | **High (Urgent)** | Immediate IV antihypertensive administration, 12-lead ECG, troponin cardiac biomarkers. |
| **4** | **Telemetry Sensor Artifact** | SQI < 0.40, Erratic instantaneous jumps in vitals, high latency / packet loss | Non-clinical (Technical) | Alert nursing staff to reposition wireless sensors or replace electrodes. |

---

## 4. Hardware & Distributed Infrastructure Sizing

For deployment in a hospital data center or cloud environment:

- **Cluster Manager:** Apache Spark Standalone or Kubernetes (YARN optional)
- **Driver Node:** 4 vCPUs, 16 GB RAM
- **Worker Nodes:** Minimum 2 Nodes, each 8 vCPUs, 32 GB RAM, 10 Gbps interconnect
- **Storage:** HDFS or Apache Parquet on AWS S3 / Azure ADLS / Google Cloud Storage
- **Throughput Target:** > 50,000 telemetry events/sec at < 500 ms p99 latency
