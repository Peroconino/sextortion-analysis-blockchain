"""Gráficos (matplotlib, backend Agg)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .clustering import ClusterResult, pca_2d

PALETTE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#EECA3B",
    "#9D755D",
    "#BAB0AC",
    "#FF9DA6",
    "#79706E",
    "#86BCB6",
]


def color_for(label: int) -> str:
    return "#BBBBBB" if label == -1 else PALETTE[int(label) % len(PALETTE)]


def plot_sweep(res: ClusterResult, out: Path) -> None:
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    s = res.sweep
    ax[0].plot(s["k"], s["inertia"], "o-")
    ax[0].set_title("K-Means: inércia (elbow)")
    ax[1].plot(s["k"], s["silhouette"], "o-")
    ax[1].set_title("Silhouette (maior = melhor)")
    ax[2].plot(s["k"], s["davies_bouldin"], "o-")
    ax[2].set_title("Davies-Bouldin (menor = melhor)")
    ax[3].plot(res.gmm_bic["k"], res.gmm_bic["bic"], "o-")
    ax[3].set_title("GMM: BIC")
    for a in ax:
        a.set_xlabel("k")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "01_selecao_k.png", dpi=130)
    plt.close(fig)


def plot_pca(res: ClusterResult, out: Path) -> None:
    P, var = pca_2d(res.Z)
    methods = [c for c in res.labels.columns if c != "address"]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 4))
    axes = np.atleast_1d(axes)
    for a, m in zip(axes, methods):
        lab = res.labels[m].to_numpy()
        a.scatter(
            P[:, 0], P[:, 1], c=[color_for(l) for l in lab], s=14, alpha=0.8, lw=0
        )
        a.set_title(f"{m} ({len(set(lab) - {-1})} clusters)")
        a.set_xlabel(f"PC1 ({var[0]:.0%})")
        a.set_ylabel(f"PC2 ({var[1]:.0%})")
    fig.tight_layout()
    fig.savefig(out / "02_pca_metodos.png", dpi=130)
    plt.close(fig)


def _packed_layout(G: nx.Graph, seed: int = 1) -> dict:
    """Um layout por componente conexo, empacotados em prateleiras (sem sobreposição)."""
    comps = cast(
        list[set[Any]],
        sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True),
    )
    radius = [0.9 + (0.6 if len(c) < 6 else 1.0) * np.sqrt(len(c)) for c in comps]
    gap = 0.5
    area = sum((2 * r + gap) ** 2 for r in radius)
    row_w = max(2 * radius[0] + gap, np.sqrt(area) * 1.3)
    pos, x, y, row_h = {}, 0.0, 0.0, 0.0
    for c, r in zip(comps, radius):
        H = G.subgraph(c).to_undirected()
        if len(c) == 1:
            p = {next(iter(c)): np.zeros(2)}
        elif len(c) == 2:
            a, b = list(c)
            p = {a: np.array([-1.0, 0.0]), b: np.array([1.0, 0.0])}
        else:
            p = nx.spring_layout(H, seed=seed, k=2.0 / np.sqrt(len(c)), iterations=400)
        arr = np.array(list(p.values()))
        arr = arr - (arr.max(0) + arr.min(0)) / 2  # centraliza
        scale = np.abs(arr).max() or 1.0
        if x + 2 * r > row_w and x > 0:
            x, y, row_h = 0.0, y - row_h - gap, 0.0
        for n, v in zip(p.keys(), arr):
            pos[n] = np.array(
                [x + r + v[0] / scale * r * 0.8, y - r + v[1] / scale * r * 0.8]
            )
        x += 2 * r + gap
        row_h = max(row_h, 2 * r)
    return pos


def _short(a: str, k: int = 6) -> str:
    return f"{a[:k]}…"


def _legend(ax, clusters, extra=()):
    from matplotlib.lines import Line2D

    h = [
        Line2D([], [], marker="o", ls="", color=color_for(c), label=f"cluster {c}")
        for c in sorted(clusters)
    ]
    ax.legend(
        handles=[*h, *extra],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        fontsize=8,
        frameon=False,
    )


def plot_seed_graph(
    SG: nx.Graph, ent_cluster: dict, out: Path, btc: dict | None = None
) -> None:
    """Grafo entre entidades-semente SEM os nós isolados, um layout por componente."""
    from matplotlib.lines import Line2D

    n_all = SG.number_of_nodes()
    H = SG.copy()
    H.remove_nodes_from([n for n in list(H) if H.degree(n) == 0])
    if H.number_of_nodes() == 0:
        return
    pos = _packed_layout(H)
    fig, ax = plt.subplots(figsize=(10, 8))
    for u, v, d in H.edges(data=True):
        if d.get("flow_btc", 0) > 0:
            col, ls = "#E45756", "-"
        elif d.get("jaccard_sinks", 0) >= d.get("jaccard_senders", 0):
            col, ls = "#4C78A8", "-"
        else:
            col, ls = "#54A24B", "--"
        ax.plot(
            *zip(pos[u], pos[v]),
            color=col,
            ls=ls,
            lw=1.2 + 4 * max(d.get("jaccard_senders", 0), d.get("jaccard_sinks", 0)),
            alpha=0.7,
            zorder=1,
        )
    nodes = list(H.nodes)
    sizes = [
        160 + 140 * np.log1p((btc or {}).get(n, 0)) + 60 * H.nodes[n].get("n_seeds", 1)
        for n in nodes
    ]
    nx.draw_networkx_nodes(
        H,
        pos,
        nodelist=nodes,
        ax=ax,
        node_size=sizes,
        linewidths=1.2,
        edgecolors="white",
        node_color=[color_for(ent_cluster.get(n, -1)) for n in nodes],
    )
    nx.draw_networkx_labels(
        H,
        pos,
        labels={n: _short(n) for n in nodes},
        font_size=7,
        ax=ax,
        verticalalignment="bottom",
        bbox={"fc": "white", "ec": "none", "alpha": 0.7, "pad": 0.5},
    )
    _legend(
        ax,
        {ent_cluster.get(n, -1) for n in nodes},
        extra=[
            Line2D([], [], color="#4C78A8", label="destinos em comum"),
            Line2D([], [], color="#54A24B", ls="--", label="pagadores em comum"),
            Line2D([], [], color="#E45756", label="fluxo direto"),
        ],
    )
    ax.set_title(
        f"Entidades-semente com alguma ligação ({H.number_of_nodes()} de {n_all}; "
        f"{n_all - H.number_of_nodes()} isoladas omitidas)"
    )
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out / "05_grafo_sementes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_backbone(
    B: nx.DiGraph, ent_cluster: dict, out: Path, n_full: int | None = None
) -> None:
    """Sementes + contrapartes compartilhadas (círculo = semente, quadrado = contraparte)."""
    from matplotlib.lines import Line2D

    if B.number_of_nodes() == 0:
        return
    pos = _packed_layout(B)
    fig, ax = plt.subplots(figsize=(11, 8))
    w = np.array([d.get("weight", 0) for *_, d in B.edges(data=True)])
    wmax = w.max() if len(w) else 1.0
    nx.draw_networkx_edges(
        B,
        pos,
        ax=ax,
        arrows=True,
        arrowsize=10,
        alpha=0.55,
        edge_color="#555555",
        width=[0.6 + 3.5 * (x / wmax) ** 0.5 for x in w],
        node_size=300,
        connectionstyle="arc3,rad=0.08",
    )
    seeds = [n for n in B if B.nodes[n].get("kind") == "seed"]
    shared = [n for n in B if B.nodes[n].get("kind") != "seed"]
    flow = {
        n: sum(d.get("weight", 0) for *_, d in B.edges(n, data=True))
        + sum(d.get("weight", 0) for *_, d in B.in_edges(n, data=True))
        for n in B
    }
    nx.draw_networkx_nodes(
        B,
        pos,
        nodelist=seeds,
        ax=ax,
        node_shape="o",
        linewidths=1.2,
        edgecolors="white",
        node_size=[200 + 120 * np.log1p(flow[n]) for n in seeds],
        node_color=[color_for(ent_cluster.get(n, -1)) for n in seeds],
    )
    nx.draw_networkx_nodes(
        B,
        pos,
        nodelist=shared,
        ax=ax,
        node_shape="s",
        node_color="#DDDDDD",
        edgecolors="#666666",
        linewidths=1,
        node_size=[140 + 90 * np.log1p(flow[n]) for n in shared],
    )
    nx.draw_networkx_labels(
        B,
        pos,
        labels={n: _short(n) for n in B},
        font_size=6.5,
        ax=ax,
        bbox={"fc": "white", "ec": "none", "alpha": 0.7, "pad": 0.4},
    )
    _legend(
        ax,
        {ent_cluster.get(n, -1) for n in seeds},
        extra=[
            Line2D(
                [],
                [],
                marker="s",
                ls="",
                mfc="#DDDDDD",
                mec="#666666",
                label="contraparte compartilhada",
            )
        ],
    )
    sub = f" (grafo completo: {n_full} nós)" if n_full else ""
    ax.set_title(
        f"Backbone: sementes + contrapartes ligadas a ≥2 sementes — {B.number_of_nodes()} nós{sub}\n"
        "largura da aresta ∝ √BTC",
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out / "06_grafo_backbone.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
