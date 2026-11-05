"""Clusterização: K-Means, DBSCAN, HDBSCAN, GMM e Agglomerative."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.cluster import HDBSCAN
except ImportError:  # sklearn < 1.3
    HDBSCAN = None

BASE_FEATURES = [
    "n_recv_tx",
    "recv_total_btc",
    "recv_mean_btc",
    "recv_median_btc",
    "recv_cv",
    "frac_near_median",
    "frac_round_values",
    "distinct_value_ratio",
    "mean_value_share_in_tx",
    "frac_unspent",
    "lifetime_days",
    "active_days",
    "recv_per_active_day",
    "mean_gap_h",
    "hour_entropy",
    "weekday_entropy",
    "mean_payer_n_in",
    "mean_payer_n_out",
    "mean_payer_fee_rate",
    "n_unique_senders",
    "senders_per_tx",
    "n_spend_tx",
    "mean_spend_n_in",
    "mean_spend_n_out",
    "spend_consol_share",
    "dwell_days",
    "never_spent",
    "type_p2pkh",
    "type_p2sh",
    "type_p2wpkh",
    "type_p2wsh",
]
GRAPH_FEATURES = [
    "entity_n_addr",
    "entity_n_seeds",
    "in_degree",
    "out_degree",
    "in_strength_btc",
    "out_strength_btc",
    "pagerank",
]
LOG_FEATURES = [
    "n_recv_tx",
    "recv_total_btc",
    "recv_mean_btc",
    "recv_median_btc",
    "lifetime_days",
    "active_days",
    "recv_per_active_day",
    "mean_gap_h",
    "mean_payer_n_in",
    "mean_payer_n_out",
    "mean_payer_fee_rate",
    "n_unique_senders",
    "n_spend_tx",
    "mean_spend_n_in",
    "mean_spend_n_out",
    "dwell_days",
    "entity_n_addr",
    "entity_n_seeds",
    "in_degree",
    "out_degree",
    "in_strength_btc",
    "out_strength_btc",
]


@dataclass
class ClusterResult:
    labels: pd.DataFrame
    Z: np.ndarray
    columns: list
    dropped_corr: list
    sweep: pd.DataFrame
    dbscan_grid: pd.DataFrame
    gmm_bic: pd.DataFrame
    agreement: pd.DataFrame
    info: dict = field(default_factory=dict)


def prepare_matrix(feat: pd.DataFrame, cols: list, corr_thr: float = 0.95):
    cols = [c for c in cols if c in feat.columns]
    X = feat[cols].astype(float).copy()
    for c in LOG_FEATURES:
        if c in X:
            X[c] = np.log1p(X[c].clip(lower=0))
    X = X.loc[:, X.std() > 0]
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if (upper[c] > corr_thr).any()]
    X = X.drop(columns=drop)
    Z = StandardScaler().fit_transform(X)
    return X, Z, drop


def _safe_scores(Z, labels):
    mask = labels != -1
    n_cl = len(set(labels[mask]))
    if n_cl < 2 or mask.sum() <= n_cl:
        return np.nan, np.nan, np.nan
    return (
        silhouette_score(Z[mask], labels[mask]),
        davies_bouldin_score(Z[mask], labels[mask]),
        calinski_harabasz_score(Z[mask], labels[mask]),
    )


def run_clustering(
    feat: pd.DataFrame,
    use_graph_features: bool = False,
    k: int | None = None,
    k_max: int = 12,
    seed: int = 42,
) -> ClusterResult:
    n = len(feat)
    if n < 6:
        raise ValueError(f"Poucos endereços ({n}) para clusterizar")
    cols = BASE_FEATURES + (GRAPH_FEATURES if use_graph_features else [])
    X, Z, dropped = prepare_matrix(feat, cols)
    k_max = max(2, min(k_max, n - 1))

    # ---------- K-Means: varredura de k
    rows = []
    for kk in range(2, k_max + 1):
        km = KMeans(n_clusters=kk, n_init=20, random_state=seed).fit(Z)
        sil, db, ch = _safe_scores(Z, km.labels_)
        rows.append(
            {
                "k": kk,
                "inertia": km.inertia_,
                "silhouette": sil,
                "davies_bouldin": db,
                "calinski_harabasz": ch,
            }
        )
    sweep = pd.DataFrame(rows)
    k_best = int(k or sweep.loc[sweep["silhouette"].idxmax(), "k"])
    labels = pd.DataFrame({"address": feat["address"]})
    labels["kmeans"] = KMeans(
        n_clusters=k_best, n_init=50, random_state=seed
    ).fit_predict(Z)

    # ---------- DBSCAN: busca de eps por silhouette
    min_samples = max(4, round(1.5 * np.log(n)))
    kd = (
        NearestNeighbors(n_neighbors=min(min_samples, n - 1))
        .fit(Z)
        .kneighbors(Z)[0][:, -1]
    )
    grid = []
    for q in np.arange(0.40, 0.99, 0.04):
        eps = float(np.quantile(kd, q))
        lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)
        sil, _db, _ = _safe_scores(Z, lab)
        grid.append(
            {
                "quantile": round(q, 2),
                "eps": eps,
                "n_clusters": len(set(lab) - {-1}),
                "noise_frac": float((lab == -1).mean()),
                "silhouette": sil,
            }
        )
    dgrid = pd.DataFrame(grid)
    ok = dgrid[(dgrid["noise_frac"] <= 0.4) & dgrid["silhouette"].notna()]
    eps_best = (
        float(ok.loc[ok["silhouette"].idxmax(), "eps"])
        if len(ok)
        else float(np.quantile(kd, 0.9))
    )
    labels["dbscan"] = DBSCAN(eps=eps_best, min_samples=min_samples).fit_predict(Z)

    # ---------- HDBSCAN
    if HDBSCAN is not None:
        labels["hdbscan"] = HDBSCAN(
            min_cluster_size=max(5, n // 50), copy=True
        ).fit_predict(Z)

        # ---------- GMM (BIC)
    gmms = [
        GaussianMixture(
            kk, covariance_type="diag", n_init=5, reg_covar=1e-4, random_state=seed
        ).fit(Z)
        for kk in range(1, k_max + 1)
    ]
    brows = [
        {"k": kk, "bic": gm.bic(Z), "aic": gm.aic(Z)}
        for kk, gm in enumerate(gmms, start=1)
    ]
    best_gmm = gmms[int(np.argmin([r["bic"] for r in brows]))]
    labels["gmm"] = best_gmm.predict(Z)

    # ---------- Aglomerativo (Ward) com o mesmo k do K-Means
    labels["agglomerative"] = AgglomerativeClustering(n_clusters=k_best).fit_predict(Z)

    methods = [c for c in labels.columns if c != "address"]
    agree = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for a in methods:
        for b in methods:
            agree.loc[a, b] = adjusted_rand_score(labels[a], labels[b])

    summary = {}
    for m in methods:
        sil, db, ch = _safe_scores(Z, labels[m].to_numpy())
        summary[m] = {
            "n_clusters": len(set(labels[m]) - {-1}),
            "noise_frac": float((labels[m] == -1).mean()),
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
        }
    info = {
        "k_best": k_best,
        "dbscan_eps": eps_best,
        "dbscan_min_samples": min_samples,
        "gmm_k": len(best_gmm.weights_),
        "n": n,
        "n_features": Z.shape[1],
        "methods": summary,
    }
    return ClusterResult(
        labels=labels,
        Z=Z,
        columns=list(X.columns),
        dropped_corr=dropped,
        sweep=sweep,
        dbscan_grid=dgrid,
        gmm_bic=pd.DataFrame(brows),
        agreement=agree,
        info=info,
    )


def pca_2d(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = PCA(n_components=2, random_state=0).fit(Z)
    return p.transform(Z), p.explained_variance_ratio_


def save_clustering(res: ClusterResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    res.labels.to_csv(out_dir / "cluster_labels.csv", index=False)
    res.sweep.to_csv(out_dir / "kmeans_sweep.csv", index=False)
    res.dbscan_grid.to_csv(out_dir / "dbscan_grid.csv", index=False)
    res.gmm_bic.to_csv(out_dir / "gmm_bic.csv", index=False)
    res.agreement.to_csv(out_dir / "method_agreement_ari.csv")
