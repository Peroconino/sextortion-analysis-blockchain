"""Entry point: `uv run sextortion-analysis --addresses addresses --out results`."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import plots
from .analysis import cluster_flows, entity_cluster_map, write_report
from .clustering import run_clustering, save_clustering
from .features import build_features
from .graph_analysis import build_graph, export_graphs
from .loader import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--addresses", default="addresses", help="pasta com um arquivo JSON por endereço")
    ap.add_argument("--out", default="results", help="pasta de saída")
    ap.add_argument("--primary", default="kmeans",
                    choices=["kmeans", "dbscan", "hdbscan", "gmm", "agglomerative"],
                    help="método usado nas análises por cluster")
    ap.add_argument("--k", type=int, default=None, help="força k no K-Means/Aglomerativo")
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--with-graph-features", action="store_true",
                    help="inclui métricas de grafo (grau, pagerank, entidade) na clusterização")
    ap.add_argument("--jaccard-min", type=float, default=0.05)
    ap.add_argument("--hub-max", type=int, default=300,
                    help="ignora pagadores/destinos ligados a mais de N entidades-semente")
    ap.add_argument("--max-files", type=int, default=None, help="limita arquivos (teste rápido)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def step(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    step(f"Lendo {a.addresses}/ ...")
    ds = load_dataset(a.addresses, a.max_files)
    step(f"{len(ds.seeds)} endereços, {len(ds.txs)} txs únicas, {len(ds.ios)} inputs/outputs; "
         f"{len(ds.bad_files)} arquivos com erro, {int(ds.seeds['truncated'].sum())} possivelmente truncados")
    for name, err in ds.bad_files[:5]:
        print("  erro:", name, err)

    step("Features por endereço ...")
    feat = build_features(ds)
    feat.to_csv(out / "features.csv", index=False)

    step("Grafo (input comum, entidades, grafo de sementes) ...")
    gr = build_graph(ds, a.jaccard_min, a.hub_max)
    gr.seed_metrics.to_csv(out / "seed_graph_metrics.csv", index=False)
    step(f"  {gr.stats['n_seed_entities']} entidades-semente "
         f"({gr.stats['seeds_merged_by_cospend']} endereços fundidos por co-gasto), "
         f"{gr.stats['n_edges']} arestas, {gr.stats['seed_graph_communities']} comunidades")

    step("Clusterização ...")
    X = feat.merge(gr.seed_metrics, on="address", how="left")
    cres = run_clustering(X, use_graph_features=a.with_graph_features, k=a.k, k_max=a.k_max)
    save_clustering(cres, out)
    primary = a.primary if a.primary in cres.labels.columns else "kmeans"
    step("  " + ", ".join(f"{m}: {v['n_clusters']} clusters" for m, v in cres.info["methods"].items()))

    step("Gráficos, exportação e relatório ...")
    plots.plot_sweep(cres, out)
    plots.plot_pca(cres, out)
    full = X.merge(cres.labels, on="address")
    plots.plot_boxplots(full, primary, ["recv_total_btc", "n_recv_tx", "recv_median_btc", "recv_cv",
                                        "lifetime_days", "mean_gap_h", "n_unique_senders",
                                        "dwell_days", "spend_consol_share"], out)
    ent_cl = entity_cluster_map(full, primary)
    plots.plot_flow_heatmap(cluster_flows(gr, ent_cl), out)
    plots.plot_seed_graph(gr.seed_graph, ent_cl, out)
    gexf = export_graphs(gr, cres.labels.set_index("address")[primary], out)
    files = [p.name for p in sorted(out.glob("*.png"))] + gexf
    summary = write_report(feat, cres.labels, cres, gr, ds, primary, out, files)
    step(f"Pronto. Relatório: {out / 'report.md'}")
    print(f"  k={summary['clustering']['k_best']}  ARI(cluster×grafo)="
          f"{summary['ari_cluster_vs_comunidade']:.3f}")


if __name__ == "__main__":
    main()
