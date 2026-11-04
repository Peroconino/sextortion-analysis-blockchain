"""Análise dos clusters (financeira + grafo) e geração do relatório em Markdown."""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .graph_analysis import GraphResult

PROFILE_COLS = [
    "n_recv_tx", "recv_total_btc", "recv_median_btc", "recv_cv", "frac_near_median",
    "lifetime_days", "mean_gap_h", "n_unique_senders", "mean_payer_n_out",
    "mean_payer_fee_rate", "dwell_days", "never_spent", "spend_consol_share",
    "frac_unspent", "in_degree", "out_degree", "entity_n_addr",
]


def md(df: pd.DataFrame, floatfmt: str = ".4g", index: bool = True) -> str:
    d = df.reset_index() if index else df
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return "" if np.isnan(x) else format(x, floatfmt)
        return str(x)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def cluster_profile(df: pd.DataFrame, label: str) -> pd.DataFrame:
    cols = [c for c in PROFILE_COLS if c in df.columns]
    prof = df.groupby(label)[cols].median()
    prof.insert(0, "n_enderecos", df.groupby(label).size())
    prof.insert(1, "pct_enderecos", 100 * prof["n_enderecos"] / len(df))
    tot = df["recv_total_btc"].sum()
    prof.insert(2, "btc_recebido", df.groupby(label)["recv_total_btc"].sum())
    prof.insert(3, "pct_btc", 100 * prof["btc_recebido"] / tot if tot else 0.0)
    return prof


def entity_cluster_map(df: pd.DataFrame, label: str) -> dict:
    return df.groupby("entity")[label].agg(lambda s: int(s.mode().iloc[0])).to_dict()


def cluster_flows(gr: GraphResult, ent_cluster: dict) -> pd.DataFrame:
    rows = []
    for u, v, d in gr.entity_graph.edges(data=True):
        if u in ent_cluster and v in ent_cluster:
            rows.append((ent_cluster[u], ent_cluster[v], d["weight"]))
    if not rows:
        return pd.DataFrame()
    f = pd.DataFrame(rows, columns=["src", "dst", "btc"])
    return f.pivot_table(index="src", columns="dst", values="btc", aggfunc="sum", fill_value=0.0)


def sink_concentration(gr: GraphResult, ent_cluster: dict) -> pd.DataFrame:
    s = gr.sinks[gr.sinks["dst_is_seed"] == 0].copy()
    s["cluster"] = s["src_entity"].map(ent_cluster)
    rows = []
    for c, grp in s.groupby("cluster"):
        by_dst = grp.groupby("dst_entity")["btc"].sum().sort_values(ascending=False)
        tot = by_dst.sum()
        share = by_dst / tot if tot else by_dst
        rows.append(dict(cluster=c, btc_saida=tot, n_destinos=len(by_dst),
                         top1_share=share.iloc[0] if len(share) else 0,
                         top5_share=share.iloc[:5].sum() if len(share) else 0,
                         hhi=float((share ** 2).sum()) if len(share) else 0,
                         top1_destino=by_dst.index[0] if len(by_dst) else ""))
    return pd.DataFrame(rows).set_index("cluster") if rows else pd.DataFrame()


def link_density(gr: GraphResult, ent_cluster: dict) -> pd.DataFrame:
    sizes = pd.Series(ent_cluster).value_counts()
    linked_in = {c: 0 for c in sizes.index}
    linked_between = 0
    for a, b in gr.seed_graph.edges():
        ca, cb = ent_cluster.get(a), ent_cluster.get(b)
        if ca is None or cb is None:
            continue
        if ca == cb:
            linked_in[ca] += 1
        else:
            linked_between += 1
    rows = []
    for c, n in sizes.items():
        tot = comb(n, 2)
        rows.append(dict(cluster=c, entidades=n, pares_possiveis=tot, pares_ligados=linked_in[c],
                         densidade_interna=linked_in[c] / tot if tot else np.nan))
    N = int(sizes.sum())
    tot_between = comb(N, 2) - sum(comb(n, 2) for n in sizes)
    out = pd.DataFrame(rows).set_index("cluster").sort_index()
    out.attrs["between_density"] = linked_between / tot_between if tot_between else np.nan
    return out


def write_report(feat: pd.DataFrame, labels: pd.DataFrame, cres, gr: GraphResult, ds,
                 primary: str, out_dir: Path, plot_files: list[str]) -> dict:
    df = feat.merge(gr.seed_metrics, on="address", how="left").merge(labels, on="address")
    df.to_csv(out_dir / "features_with_labels.csv", index=False)
    ent_cl = entity_cluster_map(df, primary)

    prof = cluster_profile(df, primary)
    prof.to_csv(out_dir / "cluster_profiles.csv")
    flow = cluster_flows(gr, ent_cl)
    flow.to_csv(out_dir / "cluster_flows_btc.csv")
    sink_c = sink_concentration(gr, ent_cl)
    sink_c.to_csv(out_dir / "cluster_sinks.csv")
    dens = link_density(gr, ent_cl)
    dens.to_csv(out_dir / "cluster_link_density.csv")
    top_sinks = (gr.sinks[gr.sinks["dst_is_seed"] == 0].groupby("dst_entity")
                 .agg(btc=("btc", "sum"), n_origens=("src_entity", "nunique"))
                 .sort_values("btc", ascending=False).head(15))
    top_sinks.to_csv(out_dir / "top_sinks.csv")
    gr.sinks.sort_values("btc", ascending=False).to_csv(out_dir / "sinks.csv", index=False)
    gr.seed_pairs.to_csv(out_dir / "seed_pairs.csv", index=False)
    df.sort_values("pagerank", ascending=False)[
        ["address", primary, "recv_total_btc", "n_recv_tx", "in_degree", "out_degree", "pagerank",
         "entity", "entity_n_seeds"]].head(20).to_csv(out_dir / "top_pagerank.csv", index=False)

    comm = df["seed_graph_community"].fillna(-1).astype(int)
    ari = adjusted_rand_score(df[primary], comm)
    nmi = normalized_mutual_info_score(df[primary], comm)
    xtab = pd.crosstab(df[primary], comm, rownames=[primary], colnames=["comunidade_grafo"])
    xtab.to_csv(out_dir / "crosstab_cluster_vs_comunidade.csv")

    ds_info = dict(
        n_enderecos=len(ds.seeds), n_transacoes_unicas=len(ds.txs),
        enderecos_truncados=int(ds.seeds["truncated"].sum()),
        arquivos_com_erro=len(ds.bad_files),
        periodo_utc=[str(pd.to_datetime(ds.txs["time"].min(), unit="s", utc=True)),
                     str(pd.to_datetime(ds.txs["time"].max(), unit="s", utc=True))],
        btc_total_recebido=float(feat["recv_total_btc"].sum()))
    summary = dict(dataset=ds_info, clustering=cres.info, grafo=gr.stats,
                   primary_method=primary, ari_cluster_vs_comunidade=ari, nmi_cluster_vs_comunidade=nmi,
                   densidade_entre_clusters=dens.attrs.get("between_density"),
                   features_usadas=cres.columns, features_removidas_por_correlacao=cres.dropped_corr)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str, ensure_ascii=False))

    methods = pd.DataFrame(cres.info["methods"]).T
    L = []
    L += ["# Relatório: clusterização e grafo dos endereços de sextortion\n",
          "Cole este arquivo (e, se quiser, os CSVs citados) na conversa para a análise.\n",
          "## 1. Dataset", "```json", json.dumps(ds_info, indent=2, ensure_ascii=False), "```\n"]
    if ds_info["enderecos_truncados"]:
        L.append(f"> ⚠ {ds_info['enderecos_truncados']} endereços têm tantas transações quanto o "
                 f"`limit` do arquivo (paginação possivelmente incompleta); features de volume "
                 f"desses endereços estão subestimadas.\n")
    L += ["## 2. Métodos de clusterização", md(methods, index=True),
          f"\nk escolhido (K-Means, silhouette): **{cres.info['k_best']}**; DBSCAN eps="
          f"{cres.info['dbscan_eps']:.3f}, min_samples={cres.info['dbscan_min_samples']}; "
          f"GMM (BIC) k={cres.info['gmm_k']}. Método principal para a análise: **{primary}**.\n",
          "### Concordância entre métodos (ARI)", md(cres.agreement, ".2f"),
          "\n### Varredura de k (K-Means)", md(cres.sweep, ".4g", index=False),
          f"\nFeatures usadas ({len(cres.columns)}): {', '.join(cres.columns)}\n",
          f"Removidas por correlação > 0.95: {', '.join(cres.dropped_corr) or 'nenhuma'}\n",
          f"## 3. Perfil dos clusters ({primary}), medianas", md(prof, ".4g"), "",
          "## 4. Grafo", "```json", json.dumps(gr.stats, indent=2), "```\n",
          f"Concordância cluster de atributos × comunidade do grafo: ARI={ari:.3f}, NMI={nmi:.3f}\n",
          md(xtab, "d"), "",
          "### Densidade de ligações (pagadores/destinos em comum ou fluxo direto)",
          md(dens, ".3g"),
          f"\nDensidade entre clusters: {dens.attrs.get('between_density', float('nan')):.4g}\n",
          "### Fluxo entre clusters (BTC)", md(flow, ".4g") if not flow.empty else "_sem fluxo entre sementes_",
          "\n### Concentração de destinos por cluster", md(sink_c, ".3g") if not sink_c.empty else "_n/d_",
          "\n### Principais destinos (entidades) dos fundos", md(top_sinks, ".4g"),
          "\n## 5. Arquivos gerados", *[f"- `{p}`" for p in sorted(set(plot_files))]]
    (out_dir / "report.md").write_text("\n".join(L))
    return summary
