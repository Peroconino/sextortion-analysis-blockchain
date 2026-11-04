"""Gráficos (matplotlib, backend Agg)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .clustering import ClusterResult, pca_2d  # noqa: E402

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2",
           "#EECA3B", "#9D755D", "#BAB0AC", "#FF9DA6", "#79706E", "#86BCB6"]


def color_for(label: int) -> str:
    return "#BBBBBB" if label == -1 else PALETTE[int(label) % len(PALETTE)]


def plot_sweep(res: ClusterResult, out: Path) -> None:
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    s = res.sweep
    ax[0].plot(s["k"], s["inertia"], "o-"); ax[0].set_title("K-Means: inércia (elbow)")
    ax[1].plot(s["k"], s["silhouette"], "o-"); ax[1].set_title("Silhouette (maior = melhor)")
    ax[2].plot(s["k"], s["davies_bouldin"], "o-"); ax[2].set_title("Davies-Bouldin (menor = melhor)")
    ax[3].plot(res.gmm_bic["k"], res.gmm_bic["bic"], "o-"); ax[3].set_title("GMM: BIC")
    for a in ax:
        a.set_xlabel("k"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "01_selecao_k.png", dpi=130); plt.close(fig)


def plot_pca(res: ClusterResult, out: Path) -> None:
    P, var = pca_2d(res.Z)
    methods = [c for c in res.labels.columns if c != "address"]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 4))
    axes = np.atleast_1d(axes)
    for a, m in zip(axes, methods):
        lab = res.labels[m].to_numpy()
        a.scatter(P[:, 0], P[:, 1], c=[color_for(l) for l in lab], s=14, alpha=.8, lw=0)
        a.set_title(f"{m} ({len(set(lab) - {-1})} clusters)")
        a.set_xlabel(f"PC1 ({var[0]:.0%})"); a.set_ylabel(f"PC2 ({var[1]:.0%})")
    fig.tight_layout(); fig.savefig(out / "02_pca_metodos.png", dpi=130); plt.close(fig)


def plot_boxplots(feat: pd.DataFrame, label_col: str, cols: list, out: Path) -> None:
    cols = [c for c in cols if c in feat.columns]
    n = len(cols)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    groups = sorted(feat[label_col].unique())
    for a, c in zip(axes, cols):
        data = [np.log10(feat.loc[feat[label_col] == g, c].clip(lower=0) + 1e-9)
                if c.endswith("_btc") or c.startswith("n_") else feat.loc[feat[label_col] == g, c]
                for g in groups]
        a.boxplot(data, tick_labels=[str(g) for g in groups], showfliers=False)
        a.set_title(c + (" (log10)" if c.endswith("_btc") or c.startswith("n_") else ""))
        a.set_xlabel(label_col)
    for a in axes[n:]:
        a.axis("off")
    fig.tight_layout(); fig.savefig(out / "03_boxplots_clusters.png", dpi=130); plt.close(fig)


def plot_flow_heatmap(flow: pd.DataFrame, out: Path) -> None:
    if flow.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(np.log10(flow.to_numpy() + 1e-3), cmap="viridis")
    ax.set_xticks(range(len(flow.columns)), flow.columns); ax.set_yticks(range(len(flow.index)), flow.index)
    ax.set_xlabel("cluster destino"); ax.set_ylabel("cluster origem")
    ax.set_title("Fluxo entre clusters (BTC, log10)")
    for i in range(flow.shape[0]):
        for j in range(flow.shape[1]):
            ax.text(j, i, f"{flow.iloc[i, j]:.2g}", ha="center", va="center", color="w", fontsize=7)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out / "04_fluxo_entre_clusters.png", dpi=130)
    plt.close(fig)


def plot_seed_graph(SG: nx.Graph, ent_cluster: dict, out: Path) -> None:
    if SG.number_of_nodes() == 0:
        return
    pos = nx.spring_layout(SG, weight="weight", seed=1, k=1.2 / np.sqrt(max(SG.number_of_nodes(), 1)))
    fig, ax = plt.subplots(figsize=(9, 8))
    nx.draw_networkx_edges(SG, pos, ax=ax, alpha=.25, width=.6)
    nodes = list(SG.nodes)
    nx.draw_networkx_nodes(SG, pos, nodelist=nodes, ax=ax, node_size=[20 + 15 * SG.nodes[n].get("n_seeds", 1) for n in nodes],
                           node_color=[color_for(ent_cluster.get(n, -1)) for n in nodes], linewidths=0)
    ax.set_title("Grafo entre entidades-semente (cor = cluster de atributos)"); ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "05_grafo_sementes.png", dpi=130); plt.close(fig)
