"""Engenharia de features por endereço (financeiras, temporais, estruturais)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import Dataset, script_type


def _entropy(values: np.ndarray, bins: int) -> float:
    counts = np.bincount(values.astype(int), minlength=bins).astype(float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def build_features(ds: Dataset) -> pd.DataFrame:
    seeds = ds.seeds["address"].tolist()
    seed_set = set(seeds)
    txs = ds.txs.set_index("txid")
    ios = ds.ios
    t_cols = ["time", "n_in", "n_out", "total_out", "fee_rate"]

    out = ios[(ios["role"] == "out") & ios["address"].isin(seed_set)].join(txs[t_cols], on="txid")
    inp = ios[(ios["role"] == "in") & ios["address"].isin(seed_set)].join(txs[t_cols], on="txid")
    out = out.copy()
    out["spent"] = out["spent"].fillna(False).astype(bool)

    ts = pd.to_datetime(out["time"], unit="s", utc=True)
    out["hour"] = ts.dt.hour
    out["weekday"] = ts.dt.weekday
    out["day"] = ts.dt.floor("D")
    out["is_round"] = out["value"] % 100_000 == 0
    out["value_share"] = out["value"] / out["total_out"].replace(0, np.nan)
    g0 = out.groupby("address")
    med = g0["value"].transform("median")
    out["near_median"] = (out["value"] - med).abs() <= 0.1 * med
    out = out.sort_values(["address", "time"])
    out["gap_h"] = out.groupby("address")["time"].diff() / 3600.0
    g = out.groupby("address")

    f = pd.DataFrame(index=pd.Index(seeds, name="address"))
    # --- recebimento
    f["n_recv_outputs"] = g.size()
    f["n_recv_tx"] = g["txid"].nunique()
    f["recv_total_btc"] = g["value"].sum() / 1e8
    f["recv_mean_btc"] = g["value"].mean() / 1e8
    f["recv_median_btc"] = g["value"].median() / 1e8
    f["recv_max_btc"] = g["value"].max() / 1e8
    f["recv_std_btc"] = g["value"].std(ddof=0) / 1e8
    f["recv_cv"] = f["recv_std_btc"] / f["recv_mean_btc"].replace(0, np.nan)
    f["frac_near_median"] = g["near_median"].mean()
    f["frac_round_values"] = g["is_round"].mean()
    f["distinct_value_ratio"] = g["value"].nunique() / g.size()
    f["mean_value_share_in_tx"] = g["value_share"].mean()
    unspent = out[~out["spent"]].groupby("address")
    f["unspent_btc"] = unspent["value"].sum() / 1e8
    f["frac_unspent"] = unspent.size() / f["n_recv_outputs"]

    # --- tempo
    f["first_recv"] = g["time"].min()
    f["last_recv"] = g["time"].max()
    f["lifetime_days"] = (f["last_recv"] - f["first_recv"]) / 86400.0
    f["active_days"] = g["day"].nunique()
    f["recv_per_active_day"] = f["n_recv_tx"] / f["active_days"]
    f["mean_gap_h"] = g["gap_h"].mean()
    f["hour_entropy"] = g["hour"].apply(lambda s: _entropy(s.to_numpy(), 24))
    f["weekday_entropy"] = g["weekday"].apply(lambda s: _entropy(s.to_numpy(), 7))

    # --- transações que pagam o endereço (perfil dos pagadores)
    pt = out.drop_duplicates(["address", "txid"]).groupby("address")
    f["mean_payer_n_in"] = pt["n_in"].mean()
    f["mean_payer_n_out"] = pt["n_out"].mean()
    f["mean_payer_fee_rate"] = pt["fee_rate"].mean()
    pairs = out[["address", "txid"]].drop_duplicates().rename(columns={"address": "seed"})
    senders = ios[(ios["role"] == "in") & ios["address"].notna()][["txid", "address"]]
    senders = senders.rename(columns={"address": "sender"})
    m = pairs.merge(senders, on="txid")
    m = m[m["sender"] != m["seed"]]
    f["n_unique_senders"] = m.groupby("seed")["sender"].nunique()
    f["senders_per_tx"] = f["n_unique_senders"] / f["n_recv_tx"]

    # --- gasto (consolidação / saída)
    if len(inp):
        gi = inp.groupby("address")
        f["n_spend_inputs"] = gi.size()
        f["spent_btc"] = gi["value"].sum() / 1e8
        f["n_spend_tx"] = gi["txid"].nunique()
        st = inp.drop_duplicates(["address", "txid"]).copy()
        st["consol"] = (st["n_in"] >= 5) & (st["n_out"] <= 2)
        gs = st.groupby("address")
        f["mean_spend_n_in"] = gs["n_in"].mean()
        f["mean_spend_n_out"] = gs["n_out"].mean()
        f["spend_consol_share"] = gs["consol"].mean()
        f["first_spend"] = gi["time"].min()
    else:
        for c in ["n_spend_inputs", "spent_btc", "n_spend_tx", "mean_spend_n_in",
                  "mean_spend_n_out", "spend_consol_share", "first_spend"]:
            f[c] = np.nan
    t_max = float(ds.txs["time"].max())
    f["never_spent"] = f["n_spend_inputs"].isna() | (f["n_spend_inputs"] == 0)
    dwell = (f["first_spend"] - f["first_recv"]) / 86400.0
    f["dwell_days"] = dwell.clip(lower=0).fillna((t_max - f["first_recv"]) / 86400.0)

    # --- tipo de endereço e metadados do arquivo
    f["addr_type"] = [script_type(a) for a in f.index]
    for t in ["p2pkh", "p2sh", "p2wpkh", "p2wsh"]:
        f[f"type_{t}"] = (f["addr_type"] == t).astype(int)
    meta = ds.seeds.set_index("address")[["n_tx_file", "truncated"]]
    f = f.join(meta)

    num = f.select_dtypes(include=[np.number]).columns
    f[num] = f[num].fillna(0)
    f["never_spent"] = f["never_spent"].astype(int)
    f["first_recv_utc"] = pd.to_datetime(f["first_recv"], unit="s", utc=True, errors="coerce")
    f["last_recv_utc"] = pd.to_datetime(f["last_recv"], unit="s", utc=True, errors="coerce")
    return f.reset_index()
