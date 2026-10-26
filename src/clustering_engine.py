"""
Core Clustering Engine: Unsupervised Distributed Ensemble Gradient-Boosting
Implements the contrastive tree embedding formulation and scalable clustering.
Supports both PySpark execution and optimized multi-core CPU execution.
"""

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb


class GradientBoostedClusteringEngine:
    def __init__(self, n_clusters=5, n_estimators=40, max_depth=5, learning_rate=0.1, random_state=42):
        self.n_clusters = n_clusters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.xgb_model = None
        self.leaf_encoder = None
        self.svd = None
        self.clustering_model = None
        self.feature_names = None

    def fit_transform(self, df):
        """
        1. Encodes heterogeneous features (continuous vitals + categorical demographics + IoT metadata).
        2. Synthesizes contrastive null reference distribution (marginal permutation).
        3. Trains XGBoost with histogram tree splitting to discriminate real from synthetic data.
        4. Extracts multi-hot leaf-node embeddings capturing high-order non-linear interactions.
        5. Performs scalable clustering and stratifies patient phenotypes.
        """
        start_time = time.time()
        print(f"[Engine] Starting Gradient-Boosted Representation & Clustering on {len(df):,} records...")

        # Feature separation
        continuous_cols = [
            "heart_rate", "systolic_bp", "diastolic_bp", "spo2",
            "resp_rate", "temperature", "ecg_hrv", "has_hypertension",
            "has_diabetes", "has_copd", "has_cardiovascular", "sqi", "latency_ms"
        ]
        categorical_cols = ["age_group", "gender"]

        # One-Hot Encoding for Demographics
        ohe_cat = OneHotEncoder(sparse_output=False, drop="first")
        cat_encoded = ohe_cat.fit_transform(df[categorical_cols])
        cat_names = list(ohe_cat.get_feature_names_out(categorical_cols))

        X_cont = df[continuous_cols].values
        X_real = np.hstack([X_cont, cat_encoded])
        self.feature_names = continuous_cols + cat_names
        N, D = X_real.shape

        # Step 2: Unsupervised Reference Distribution (Breiman Contrastive Method)
        print(f"[Engine] Generating synthetic reference marginals for {D} heterogeneous features...")
        X_synth = np.zeros_like(X_real)
        np.random.seed(self.random_state)
        for j in range(D):
            X_synth[:, j] = np.random.permutation(X_real[:, j])

        # Formulate binary discrimination task (1 = observed telemetry, 0 = synthetic independent marginals)
        X_contrast = np.vstack([X_real, X_synth])
        y_contrast = np.hstack([np.ones(N, dtype=np.int32), np.zeros(N, dtype=np.int32)])

        # Step 3: Train Histogram-Based Distributed XGBoost
        print(f"[Engine] Training XGBoost ensemble ({self.n_estimators} trees, max_depth={self.max_depth}, tree_method='hist')...")
        t_xgb_start = time.time()
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            tree_method="hist",
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.xgb_model.fit(X_contrast, y_contrast)
        t_xgb = time.time() - t_xgb_start
        print(f"[Engine] XGBoost training finished in {t_xgb:.2f}s.")

        # Step 4: Extract Non-Linear Leaf Node Embeddings
        print("[Engine] Extracting high-dimensional sparse leaf representations: Phi(x)...")
        leaf_indices = self.xgb_model.apply(X_real)

        self.leaf_encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
        sparse_embeddings = self.leaf_encoder.fit_transform(leaf_indices)
        total_leaf_features = sparse_embeddings.shape[1]
        print(f"[Engine] Generated sparse leaf embedding of dimension: {total_leaf_features}")

        # Dimensionality Reduction for Scalable Distance Computation
        self.svd = TruncatedSVD(n_components=16, random_state=self.random_state)
        dense_latent = self.svd.fit_transform(sparse_embeddings)

        # Step 5: Scalable Clustering (MiniBatch / Bisecting K-Means)
        print(f"[Engine] Executing Scalable Clustering (k={self.n_clusters})...")
        t_cluster_start = time.time()
        self.clustering_model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=2048,
            random_state=self.random_state,
            n_init="auto"
        )
        cluster_labels = self.clustering_model.fit_predict(dense_latent)
        t_cluster = time.time() - t_cluster_start
        print(f"[Engine] Clustering converged in {t_cluster:.2f}s.")

        # Step 6: Evaluation Metrics
        print("[Engine] Computing validation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)...")
        # Subsample for Silhouette if N > 15,000 to keep metric evaluation fast
        if N > 15000:
            eval_idx = np.random.choice(N, size=10000, replace=False)
            sil_score = float(silhouette_score(dense_latent[eval_idx], cluster_labels[eval_idx]))
        else:
            sil_score = float(silhouette_score(dense_latent, cluster_labels))

        db_score = float(davies_bouldin_score(dense_latent, cluster_labels))
        ch_score = float(calinski_harabasz_score(dense_latent, cluster_labels))

        # Baseline comparison: Standard Euclidean K-Means on Raw Normalized Features
        scaler = StandardScaler()
        X_raw_norm = scaler.fit_transform(X_real)
        baseline_km = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=2048, random_state=self.random_state, n_init="auto")
        baseline_labels = baseline_km.fit_predict(X_raw_norm)
        if N > 15000:
            baseline_sil = float(silhouette_score(X_raw_norm[eval_idx], baseline_labels[eval_idx]))
        else:
            baseline_sil = float(silhouette_score(X_raw_norm, baseline_labels))

        df_out = df.copy()
        df_out["predicted_cluster"] = cluster_labels

        total_elapsed = time.time() - start_time

        results = {
            "processed_df": df_out,
            "silhouette_score": sil_score,
            "baseline_silhouette": baseline_sil,
            "davies_bouldin_index": db_score,
            "calinski_harabasz_index": ch_score,
            "embedding_dimension": total_leaf_features,
            "feature_importance": dict(zip(self.feature_names, [float(v) for v in self.xgb_model.feature_importances_])),
            "total_elapsed_seconds": total_elapsed,
            "xgb_training_seconds": t_xgb,
            "clustering_seconds": t_cluster,
            "n_samples": N,
            "n_clusters": self.n_clusters
        }

        print(f"[Engine] XGBoost-Latent Silhouette: {sil_score:.4f} vs Baseline Euclidean: {baseline_sil:.4f}")
        print(f"[Engine] Davies-Bouldin Index: {db_score:.4f} (Lower is better)")
        print(f"[Engine] Total Execution Time: {total_elapsed:.2f} seconds.")

        return results
