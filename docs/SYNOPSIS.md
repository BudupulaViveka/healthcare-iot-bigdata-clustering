# ACADEMIC PROJECT SYNOPSIS & PROPOSAL

**Course / Subject:** Big Data Analytics  
**Project Title:** Scalable Clustering of Heterogeneous Big Data via Distributed Ensemble Gradient-boosting Models (XGBoost on Spark/Dask), Applied to Healthcare IoT Patient Monitoring at Scale  
**Project Guide / Supervisor:** Ms. G. Hema Sri  
**Academic Year:** 2026–2027  

---

## 1. Executive Abstract

The exponential proliferation of Internet of Medical Things (IoT) devices in intensive care units (ICUs) and ambulatory remote patient monitoring systems generates unprecedented volumes of heterogeneous, high-velocity streaming data. These telemetry streams blend high-frequency non-stationary physiological continuous vitals (heart rate, blood pressure, oxygen saturation, respiration rate, ECG telemetry) with static/slowly changing clinical categorical demographics (age brackets, pre-existing comorbidities, medication regimens) and sensor-level transmission metadata (signal quality indicators, latency, packet dropouts). 

Traditional unsupervised clustering algorithms (e.g., standard K-Means, DBSCAN, Gaussian Mixture Models) fail in this setting due to three core bottlenecks:
1. **The Curse of Heterogeneity:** Calculating Euclidean distance across non-normalized, multi-modal features with non-linear correlations produces severe distance distortion.
2. **Computational Infeasibility at Big Data Scale:** Standard quadratic-complexity distance matrices cannot scale to millions of streaming records across distributed clusters.
3. **Lack of Dynamic Feature Interaction Representation:** Conventional pipelines isolate feature engineering from clustering, failing to capture subtle multi-organ failure precursors (e.g., compensating tachycardia preceding septic shock).

This project presents an end-to-end, scalable distributed framework that bridges ensemble gradient-boosting representation learning with distributed clustering on **Apache Spark**. We utilize **Distributed XGBoost** in an unsupervised, contrastive leaf-embedding mode to project heterogeneous, non-linear IoT telemetry into a structured, low-distortion metric space. A distributed **Bisecting K-Means / K-Means** engine on Apache Spark then partitions this latent embedding at scale into clinically interpretable patient risk phenotypes:
- **Phenotype 0:** Stable Baseline / Low Risk
- **Phenotype 1:** Sepsis / Early Septic Shock Onset
- **Phenotype 2:** Hypoxemia / Acute Respiratory Distress Syndrome (ARDS)
- **Phenotype 3:** Hypertensive Crisis / Cardiac Overload
- **Phenotype 4:** Sensor Artifact / Telemetry Lead Degradation

Extensive experimental evaluations demonstrate linear horizontal scalability, high cluster silhouette validation, and real-time sub-second patient risk stratification over millions of IoT events.

---

## 2. Problem Statement

Modern Healthcare IoT architectures stream tens of thousands of telemetry packets per second from wearable patches, bedside monitors, and pulse oximeters. Extracting real-time clinical cohorts without ground-truth labels is an open challenge because:
- **Extreme Dimensional Heterogeneity:** Physiological metrics follow different statistical distributions (Gaussian, Poisson, heavy-tailed log-normal) intertwined with categorical comorbidity matrices. Distance metrics in raw space become meaningless.
- **Dynamic Non-linear Interdependencies:** Clinical risk is non-linear. For example, a drop in blood oxygen saturation ($SpO_2 < 90\%$) combined with an elevated respiration rate ($RR > 25\text{ bpm}$) signals imminent respiratory failure, whereas the same individual symptoms in isolation might merely indicate mild physical exertion.
- **Scalability and Memory Bottlenecks:** Single-node machine learning platforms experience Out-Of-Memory (OOM) crashes when clustering high-cardinality streaming data.
- **Sensor Noise and Missing Telemetry:** Wireless patient wearables suffer from frequent packet loss, electrode detachment, and battery degradation, necessitating robust imputation that does not bias cluster boundaries.

---

## 3. Project Objectives

### Primary Objectives:
1. **Distributed Heterogeneous Ingestion & ETL:** Design and deploy a resilient Apache Spark processing pipeline to ingest, clean, impute, and vectorize high-dimensional, heterogeneous healthcare IoT telemetry.
2. **Unsupervised Gradient-Boosted Representation Learning:** Develop an unsupervised ensemble tree-embedding technique utilizing distributed XGBoost (via Spark RDD/DataFrame partitioning with histogram-based split finding) to map raw multi-modal vitals into dense leaf-index representations.
3. **Scalable Distributed Clustering:** Deploy distributed K-Means / Bisecting K-Means over the latent gradient-boosted representation to stratify patients into clinical cohorts without supervision.
4. **Clinical Validation & Anomaly Identification:** Evaluate discovered clusters against physiological warning standards (such as Modified Early Warning Score - MEWS) to isolate critical health events and filter hardware transmission artifacts.

### Secondary Objectives:
- Benchmark execution latency, cluster silhouette index, Davies-Bouldin score, and horizontal speedup across varying Spark cluster worker configurations ($10^4$ to $10^6$ records).
- Deliver interactive visualizations: cluster radar profiles, XGBoost feature importance rankings, and scalability benchmark curves.

---

## 4. Literature Review & Research Gaps

| Author & Year | Methodology | Focus Domain | Strengths | Critical Gaps Identified |
|---|---|---|---|---|
| **Al-Janabi et al. (2020)** | Standard Spark MLlib K-Means | IoT Sensor Clustering | High throughput on homogeneous numeric data | Cannot handle categorical demographics; highly vulnerable to Euclidean distance distortion in non-linear data. |
| **Breiman (2001) / Shi et al. (2006)** | Unsupervised Random Forest Proximity Matrix | Tabular Medical Records | Effective metric space representation for mixed data types | Constructing the full $N \times N$ proximity matrix has $O(N^2)$ memory complexity, making it infeasible for Big Data streams. |
| **Chen & Guestrin (2016)** | Distributed XGBoost | Supervised Classification | Quantile sketch histogram splitting; fault-tolerant distributed training | Supervised by design; lacks unsupervised clustering native integration for streaming unlabelled vitals. |
| **Proposed Framework (2026)** | **Distributed XGBoost Leaf-Embedding + Spark Bisecting K-Means** | **Heterogeneous Healthcare IoT Monitoring** | **$O(N)$ linear complexity; learns non-linear cross-vital interactions; natively handles mixed data; distributed horizontally on Spark/Dask.** | **Addresses all the above gaps.** |

---

## 5. Proposed Methodology & Mathematical Formulation

The proposed framework consists of four sequential stages:

```
[Raw IoT Telemetry Streams] (Vitals + Demographics + Metadata)
            │
            ▼
[Spark Distributed Ingestion & Robust Imputation Pipeline]
            │
            ▼
[Unsupervised Contrastive Reference Generator (Marginal Permutation)]
            │
            ▼
[Distributed XGBoost Model (Histogram-based Tree Splitting on Spark)]
            │
            ▼
[High-Dimensional Sparse Leaf Index Feature Embeddings: Φ(x)]
            │
            ▼
[Distributed K-Means / Bisecting K-Means Clustering on Spark]
            │
            ▼
[Clinical Phenotype Profiling & Anomaly Alert Engine]
```

### 5.1 Mathematical Formulation of the Heterogeneous Input Space
Let each incoming IoT patient record be represented as a heterogeneous vector:
$$\mathbf{x}_i = [\mathbf{v}_i, \mathbf{d}_i, \mathbf{m}_i] \in \mathcal{X}$$
Where:
- $\mathbf{v}_i \in \mathbb{R}^{d_v}$ represents continuous physiological vitals: Heart Rate ($HR$), Systolic BP ($SBP$), Diastolic BP ($DBP$), Oxygen Saturation ($SpO_2$), Respiration Rate ($RR$), Body Temperature ($Temp$), and ECG HRV ($SDNN$).
- $\mathbf{d}_i \in \{0, 1\}^{d_d}$ represents one-hot encoded categorical demographics and medical history (Age Bracket, Gender, Diabetes, Hypertension, COPD).
- $\mathbf{m}_i \in \mathbb{R}^{d_m}$ represents sensor quality indicators (SQI, Device Battery %, Transmission Latency).

### 5.2 Unsupervised Representation via Contrastive Gradient Boosting
Clustering cannot be applied directly in raw Euclidean space due to scale discrepancies and non-linear physiological couplings. We construct an unsupervised ensemble representation as follows:
1. Let $\mathcal{D}_{real} = \{\mathbf{x}_1, \dots, \mathbf{x}_N\}$ be the observed patient cohort with pseudo-label $y_i = 1$.
2. We synthesize a null reference distribution $\mathcal{D}_{synth} = \{\tilde{\mathbf{x}}_1, \dots, \tilde{\mathbf{x}}_N\}$ with $y_i = 0$ by sampling independently from the empirical marginal distribution of each feature:
   $$\tilde{\mathbf{x}}_{i, j} \sim \hat{P}(X_j), \quad \forall j \in \{1, \dots, D\}$$
   This destroys joint inter-feature correlations while strictly preserving individual marginal ranges.
3. A distributed XGBoost ensemble of $T$ additive trees is trained to optimize the binary logistic loss:
   $$\mathcal{L} = \sum_{i=1}^{2N} \left[ y_i \ln(1 + e^{-\hat{y}_i}) + (1 - y_i) \ln(1 + e^{\hat{y}_i}) \right] + \sum_{t=1}^T \Omega(f_t)$$
   Where $\Omega(f_t) = \gamma T_{leaves} + \frac{1}{2}\lambda \sum_{j=1}^{T_{leaves}} w_j^2$.
4. **Leaf Embedding Transformation:** Each patient instance $\mathbf{x}_i$ is passed down the $T$ trees. The terminal leaf index of tree $t$ for vector $\mathbf{x}_i$ is denoted $l_t(\mathbf{x}_i) \in \{1, \dots, L\}$. We formulate the embedding transformation:
   $$\mathbf{\Phi}(\mathbf{x}_i) = \text{Concat}\Big(\mathbf{e}_{l_1(\mathbf{x}_i)}, \mathbf{e}_{l_2(\mathbf{x}_i)}, \dots, \mathbf{e}_{l_T(\mathbf{x}_i)}\Big) \in \{0, 1\}^{T \cdot L}$$
   Instances that follow similar physiological decision paths share identical leaf activations, inherently measuring clinical proximity without manual metric engineering.

### 5.3 Distributed Scalable Clustering on Spark
On the transformed latent space $\mathbf{\Phi}(\mathbf{X})$, Spark MLlib executes distributed K-Means with $K$-Means++ initialization by parallelizing the centroid update across RDD partitions:
$$\boldsymbol{\mu}_k^{(m+1)} = \frac{1}{|\mathcal{C}_k|} \sum_{\mathbf{\Phi}(\mathbf{x}_i) \in \mathcal{C}_k} \mathbf{\Phi}(\mathbf{x}_i)$$
Distributed bisecting divides the largest variance cluster iteratively until $K$ cohesive clinical phenotypes are established.

---

## 6. System Architecture & Component Design

The framework is organized into five decoupled micro-architectural layers:

1. **Ingestion & Telemetry Buffer:** Spark Structured Streaming / Parquet ingestion handling continuous streams from simulated medical IoT sensor nodes.
2. **Distributed Preprocessing & Imputation:** Missing vitals from temporary sensor drops are handled via median imputer per age/risk group; categorical metadata is vectorized via `StringIndexer` and `OneHotEncoderEstimator`.
3. **Gradient-Boosted Representation Layer:** High-performance distributed histogram-based XGBoost (`tree_method='hist'`) partitions tree construction across Spark executors, generating the non-linear leaf-embedding matrix.
4. **Distributed Clustering Core:** Apache Spark MLlib Bisecting K-Means partitions the embedding space with distributed map-reduce distance computations.
5. **Clinical Phenotyping & Visualization Service:** Evaluates Silhouette score, generates centroid radar plots, computes patient risk severity scores, and triggers automated alerts.

---

## 7. Experimental Setup & Evaluation Metrics

### Datasets:
- Scalable synthetic multi-modal Healthcare IoT benchmark dataset modeled after the **MIMIC-III / PhysioNet Challenge** physiological distributions.
- Configurable scale: $10^4$, $10^5$, and $10^6$ streaming records with injected clinical anomalies (sepsis, hypertensive crisis, ARDS, lead disconnects).

### Validation Metrics:
1. **Silhouette Coefficient ($SC$):**
   $$SC = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
   Where $a(i)$ is the mean intra-cluster distance, and $b(i)$ is the minimum mean inter-cluster distance.
2. **Davies-Bouldin Index ($DBI$):**
   $$DBI = \frac{1}{k} \sum_{i=1}^k \max_{j \neq i} \left( \frac{s_i + s_j}{d(\boldsymbol{\mu}_i, \boldsymbol{\mu}_j)} \right)$$
   Lower values indicate tighter, better-separated clusters.
3. **Scalability & Throughput:**
   - Wall-clock execution time across sample sizes $N \in [10^4, 10^5, 10^6]$.
   - Speedup ratio $S_p = \frac{T_1}{T_p}$ across $p \in \{2, 4, 8\}$ executor cores.

---

## 8. Expected Outcomes & Project Deliverables

1. **Scalable Big Data Codebase:** A clean, modular Python/PySpark package with complete data generation, distributed tree-embedding generation, and clustering pipelines.
2. **Superior Clustering Separation:** Demonstration that XGBoost tree-embedding clustering achieves higher Silhouette separation than raw Euclidean K-Means on mixed healthcare data.
3. **Clinical Interpretability:** Automated mapping of cluster centroids to actionable clinical phenotypes (e.g., Sepsis Alert, Cardiac Warning).
4. **Comprehensive Documentation & Visual Artifacts:** Research-grade architecture diagrams, cluster radar plots, scalability graphs, and setup manuals.

---

## 9. Project Execution Schedule (Gantt Plan)

| Phase | Duration | Key Milestones | Status |
|---|---|---|---|
| **Phase 1: Requirements & Design** | Weeks 1–2 | Literature survey, problem definition, mathematical formulation | Completed |
| **Phase 2: Data Pipeline & Ingestion** | Weeks 3–4 | Heterogeneous IoT data generator, Spark DataFrame ETL pipeline | Completed |
| **Phase 3: Model Architecture** | Weeks 5–7 | Distributed XGBoost unsupervised leaf embedding, Spark K-Means | Completed |
| **Phase 4: Evaluation & Benchmarking** | Weeks 8–9 | Silhouette score, Davies-Bouldin, speedup scalability analysis | Completed |
| **Phase 5: Final Report & Presentation** | Weeks 10–12 | Documentation, academic synopsis, interactive dashboard | Completed |

---

## 10. References
1. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
2. Zaharia, M., et al. (2016). *Apache Spark: A unified engine for big data processing*. Communications of the ACM, 59(11), 56-65.
3. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
4. Goldberger, A. L., et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals*. Circulation, 101(23), e215-e220.
5. Shi, T., & Horvath, S. (2006). *Unsupervised Learning With Random Forest Predictors*. Journal of Computational and Graphical Statistics, 15(1), 118-138.
