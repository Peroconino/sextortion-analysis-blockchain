"""Grafo de transações: heurística de input comum, grafo de entidades e de sementes."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .loader import Dataset


class UnionFind:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        root = x
        while self.p[root] != root:
            root = self.p[root]
        while self.p[x] != root:  # compressão de caminho
            self.p[x], x = root, self.p[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


@dataclass
class GraphResult:
    entity_graph: nx.DiGraph
    seed_graph: nx.Graph
    seed_metrics: pd.DataFrame  # por endereço-semente
    seed_pairs: pd.DataFrame  # pares de entidades-semente relacionadas
    sinks: pd.DataFrame  # destinos dos fundos das sementes
    seed_entity: dict
    stats: dict


def _jaccard_pairs(sets: dict, hub_max: int) -> dict:
    inv = defaultdict(list)
    for node, s in sets.items():
        for x in s:
            inv[x].append(node)
    shared = Counter()
    for _x, nodes in inv.items():
        if len(nodes) < 2 or len(nodes) > hub_max:
            continue
        for a, b in combinations(sorted(nodes), 2):
            shared[(a, b)] += 1
    return {
        (a, b): c / (len(sets[a]) + len(sets[b]) - c) for (a, b), c in shared.items()
    }


def build_graph(
    ds: Dataset, jaccard_min: float = 0.05, hub_max: int = 300, seed: int = 42
) -> GraphResult:
    ios, txs = ds.ios, ds.txs.set_index("txid")
    seeds = ds.seeds["address"].tolist()
    out_ios = ios[ios["role"] == "out"]

    # --- detecção simples de CoinJoin (não aplicar heurística de input comum)
    top_equal = out_ios.groupby(["txid", "value"]).size().groupby("txid").max()
    top_equal = top_equal.reindex(txs.index).fillna(0)
    cj = (txs["n_in"] >= 3) & (top_equal >= np.maximum(3, 0.5 * txs["n_out"]))
    cj_set = set(cj[cj].index)

    # --- entidades por input comum
    uf = UnionFind()
    inputs = ios[
        (ios["role"] == "in") & ios["address"].notna() & ~ios["txid"].isin(cj_set)
    ]
    in_groups = inputs.groupby("txid")["address"].apply(list)
    src_of = {}
    for txid, addrs in in_groups.items():
        first = addrs[0]
        uf.find(first)
        for a in addrs[1:]:
            uf.union(first, a)
        src_of[txid] = first
    for a in ios["address"].dropna().unique():
        uf.find(a)
    src_of = {t: uf.find(a) for t, a in src_of.items()}
    sizes = Counter(uf.find(a) for a in list(uf.p))

    # --- grafo de fluxo entre entidades
    edges = defaultdict(lambda: [0, 0])
    for r in out_ios[out_ios["txid"].isin(src_of.keys())].itertuples(index=False):
        if r.address is None or (isinstance(r.address, float) and np.isnan(r.address)):
            continue
        s, d = src_of[r.txid], uf.find(r.address)
        if s == d:
            continue
        e = edges[(s, d)]
        e[0] += r.value
        e[1] += 1
    G = nx.DiGraph()
    for (s, d), (v, c) in edges.items():
        G.add_edge(s, d, weight=v / 1e8, n_tx=c)

    seed_entity = {a: uf.find(a) for a in seeds}
    seed_ents = set(seed_entity.values())
    ent_seeds = defaultdict(list)
    for a, e in seed_entity.items():
        ent_seeds[e].append(a)
    for e in seed_ents:
        G.add_node(e)
    for n in G.nodes:
        G.nodes[n]["is_seed"] = int(n in seed_ents)
        G.nodes[n]["n_addr"] = int(sizes.get(n, 1))

    # --- métricas por entidade
    try:
        pr = nx.pagerank(G, weight="weight")
    except Exception:  # noqa: BLE001
        pr = {n: np.nan for n in G}
    U = nx.Graph()
    U.add_nodes_from(G.nodes)
    for u, v, d in G.edges(data=True):
        if U.has_edge(u, v):
            U[u][v]["weight"] += d["n_tx"]
        else:
            U.add_edge(u, v, weight=d["n_tx"])
    comm_ent = {}
    for i, c in enumerate(
        nx.community.louvain_communities(U, weight="weight", seed=seed)
    ):
        for n in c:
            comm_ent[n] = i

    rows = []
    for a in seeds:
        e = seed_entity[a]
        rows.append(
            {
                "address": a,
                "entity": e,
                "entity_n_addr": sizes.get(e, 1),
                "entity_n_seeds": len(ent_seeds[e]),
                "in_degree": G.in_degree(e),
                "out_degree": G.out_degree(e),
                "in_strength_btc": G.in_degree(e, weight="weight"),
                "out_strength_btc": G.out_degree(e, weight="weight"),
                "pagerank": pr.get(e, np.nan),
                "entity_community": comm_ent.get(e, -1),
            }
        )
    metrics = pd.DataFrame(rows)

    # --- grafo entre entidades-semente (relações de pagadores / destinos em comum)
    senders = {e: set(G.predecessors(e)) - {e} for e in seed_ents}
    sinks_s = {e: set(G.successors(e)) - {e} for e in seed_ents}
    js = _jaccard_pairs(senders, hub_max)
    jk = _jaccard_pairs(sinks_s, hub_max)
    SG = nx.Graph()
    for e in seed_ents:
        SG.add_node(e, n_seeds=len(ent_seeds[e]))
    pair_rows = []
    keys = set(js) | set(jk)
    flows = {
        (u, v): d["weight"]
        for u, v, d in G.edges(data=True)
        if u in seed_ents and v in seed_ents
    }
    for (u, v), _w in flows.items():
        keys.add(tuple(sorted((u, v))))
    for a, b in keys:
        s_, k_ = js.get((a, b), 0.0), jk.get((a, b), 0.0)
        fl = flows.get((a, b), 0.0) + flows.get((b, a), 0.0)
        if max(s_, k_) < jaccard_min and fl == 0:
            continue
        w = 1.0 if fl > 0 else max(s_, k_)
        SG.add_edge(a, b, weight=w, jaccard_senders=s_, jaccard_sinks=k_, flow_btc=fl)
        pair_rows.append(
            {"a": a, "b": b, "jaccard_senders": s_, "jaccard_sinks": k_, "flow_btc": fl}
        )
    pairs = pd.DataFrame(
        pair_rows, columns=["a", "b", "jaccard_senders", "jaccard_sinks", "flow_btc"]
    )
    comm_seed = {}
    for i, c in enumerate(
        nx.community.louvain_communities(SG, weight="weight", seed=seed)
    ):
        for n in c:
            comm_seed[n] = i
    metrics["seed_graph_community"] = metrics["entity"].map(comm_seed)
    for n in SG.nodes:
        SG.nodes[n]["community"] = int(comm_seed.get(n, -1))

    # --- destinos (sinks) dos fundos das entidades-semente
    srows = [
        {
            "src_entity": u,
            "dst_entity": v,
            "btc": d["weight"],
            "n_out": d["n_tx"],
            "dst_is_seed": int(v in seed_ents),
        }
        for u, v, d in G.edges(data=True)
        if u in seed_ents
    ]
    sinks = pd.DataFrame(
        srows, columns=["src_entity", "dst_entity", "btc", "n_out", "dst_is_seed"]
    )

    stats = {
        "n_seeds": len(seeds),
        "n_seed_entities": len(seed_ents),
        "seeds_merged_by_cospend": len(seeds) - len(seed_ents),
        "n_entities_total": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_coinjoin_like_txs": len(cj_set),
        "n_txs": len(txs),
        "weak_components": nx.number_weakly_connected_components(G),
        "largest_weak_component": max(
            (len(c) for c in nx.weakly_connected_components(G)), default=0
        ),
        "seed_graph_nodes": SG.number_of_nodes(),
        "seed_graph_edges": SG.number_of_edges(),
        "seed_graph_communities": len(set(comm_seed.values())),
        "hub_max": hub_max,
        "jaccard_min": jaccard_min,
    }
    return GraphResult(G, SG, metrics, pairs, sinks, seed_entity, stats)


def backbone_graph(G: nx.DiGraph, min_seed_links: int = 2) -> nx.DiGraph:
    """Grafo reduzido: entidades-semente + contrapartes ligadas a >= `min_seed_links` sementes.

    Contrapartes (pagadores ou destinos) de uma única semente são "folhas" de estrela e não dizem
    nada sobre relação entre sementes, então saem. Sementes que sobram sem aresta também saem.
    """
    seeds = {n for n, d in G.nodes(data=True) if d.get("is_seed")}
    keep = set()
    for n in G.nodes:
        if n in seeds:
            continue
        linked = (set(G.predecessors(n)) | set(G.successors(n))) & seeds
        if len(linked) >= min_seed_links:
            keep.add(n)
    keep |= seeds
    H = G.subgraph(keep).copy()
    H.remove_nodes_from([n for n in list(H) if H.degree(n) == 0])
    for n in H.nodes:
        H.nodes[n]["kind"] = "seed" if n in seeds else "shared"
    return H


def export_graphs(
    gr: GraphResult, cluster_col: pd.Series | None, out_dir: Path, full: bool = False
) -> list[str]:
    """Exporta o GEXF completo de entidades para o Gephi, só se `full=True` (pode ser grande)."""
    if not full:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    ent_cluster = {}
    if cluster_col is not None:
        df = pd.DataFrame(
            {
                "e": [gr.seed_entity[a] for a in cluster_col.index],
                "c": cluster_col.values,
            }
        )
        ent_cluster = (
            df.groupby("e")["c"].agg(lambda s: int(s.mode().iloc[0])).to_dict()
        )
    G = gr.entity_graph.copy()
    for n in G.nodes:
        G.nodes[n]["cluster"] = int(ent_cluster.get(n, -9))
    nx.write_gexf(G, out_dir / "entity_graph_completo.gexf")
    return ["entity_graph_completo.gexf"]
