# -*- coding: utf-8 -*-
"""
Clustering Model Retraining / Regeneration Script (based on jk_ML_v2 (clustering).ipynb logic)

Input:
- startup_ver.csv (or a feature table similar to startup_ml_final created in v1)

Required columns:
- category_4 (categorical)
- degree_level_filled, log1p_n_offices, log1p_n_founding, log1p_relationships (numeric)
- is_degree_level_missing, is_n_offices_missing, is_n_founding_missing (missing-value flags)

Training:
- One-hot encode (category_4) + numeric features
- StandardScaler
- PCA(n_components=0.9)
- KMeans(n_clusters=5)

Output:
- startup_ver_with_cluster.csv (updated cluster column)
- artifacts/scaler.joblib, artifacts/pca.joblib, artifacts/kmeans.joblib (optional)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

try:
    import joblib
except Exception:
    joblib = None


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ART_DIR = BASE_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

INPUT = DATA_DIR / "startup_ver.csv"
OUTPUT = DATA_DIR / "startup_ver_with_cluster.csv"

N_CLUSTERS = 5
RANDOM_STATE = 42

REQ_COLS = [
    "objects_cfpr_id",
    "category_4",
    "degree_level_filled",
    "log1p_n_offices",
    "log1p_n_founding",
    "log1p_relationships",
    "is_degree_level_missing",
    "is_n_offices_missing",
    "is_n_founding_missing",
]

def main():
    df = pd.read_csv(INPUT, low_memory=False)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"startup_ver.csv is missing required columns: {missing}")

    df["objects_cfpr_id"] = df["objects_cfpr_id"].astype(str)

    # 1) Build X (jk_ML_v2 (clustering).ipynb logic)
    X_cat = pd.get_dummies(df["category_4"], drop_first=False)

    X_num = df[
        [
            "degree_level_filled",
            "log1p_n_offices",
            "log1p_n_founding",
            "log1p_relationships",
            "is_degree_level_missing",
            "is_n_offices_missing",
            "is_n_founding_missing",
        ]
    ].copy()

    X = pd.concat([X_num, X_cat], axis=1)

    # 2) Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3) PCA (retain 90% variance)
    pca = PCA(n_components=0.9, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    # 4) KMeans
    kmeans = KMeans(n_clusters=N_CLUSTERS, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(X_pca)

    out = df.copy()
    out["cluster"] = labels.astype(int)
    out.to_csv(OUTPUT, index=False)
    print(f"[OK] saved: {OUTPUT}  shape={out.shape}")

    # 5) (Optional) Save artifacts
    if joblib is not None:
        joblib.dump(scaler, ART_DIR / "cluster_scaler.joblib")
        joblib.dump(pca, ART_DIR / "cluster_pca.joblib")
        joblib.dump(kmeans, ART_DIR / "cluster_kmeans.joblib")
        print(f"[OK] saved artifacts to: {ART_DIR}")

if __name__ == "__main__":
    main()
