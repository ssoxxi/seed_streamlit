# -*- coding: utf-8 -*-
"""
군집 모델 재학습/재생성 스크립트 (jk_ML_v2(군집).ipynb 로직 기반)

입력:
- startup_ver.csv (또는 v1에서 만든 startup_ml_final과 유사한 피처 테이블)
필수 컬럼:
- category_4 (범주형)
- degree_level_filled, log1p_n_offices, log1p_n_founding, log1p_relationships (수치형)
- is_degree_level_missing, is_n_offices_missing, is_n_founding_missing (결측 플래그)

학습:
- One-hot(category_4) + 수치형
- StandardScaler
- PCA(n_components=0.9)
- KMeans(n_clusters=5)

출력:
- startup_ver_with_cluster.csv (cluster 컬럼 갱신)
- artifacts/scaler.joblib, artifacts/pca.joblib, artifacts/kmeans.joblib (선택)
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
        raise ValueError(f"startup_ver.csv에 필요한 컬럼이 없습니다: {missing}")

    df["objects_cfpr_id"] = df["objects_cfpr_id"].astype(str)

    # 1) X 구성 (jk_ML_v2(군집).ipynb 로직)
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

    # 2) 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3) PCA(90% 분산 유지)
    pca = PCA(n_components=0.9, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    # 4) KMeans
    kmeans = KMeans(n_clusters=N_CLUSTERS, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(X_pca)

    out = df.copy()
    out["cluster"] = labels.astype(int)
    out.to_csv(OUTPUT, index=False)
    print(f"[OK] saved: {OUTPUT}  shape={out.shape}")

    # 5) (선택) 아티팩트 저장
    if joblib is not None:
        joblib.dump(scaler, ART_DIR / "cluster_scaler.joblib")
        joblib.dump(pca, ART_DIR / "cluster_pca.joblib")
        joblib.dump(kmeans, ART_DIR / "cluster_kmeans.joblib")
        print(f"[OK] saved artifacts to: {ART_DIR}")

if __name__ == "__main__":
    main()
