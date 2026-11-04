"""Leitura dos arquivos JSON de `addresses/` e normalização em tabelas.

Cada arquivo tem o nome do endereço (sem extensão) e o conteúdo é a resposta
paginada de um explorer: {"transactions": [...], "limit": 50, "offset": 0}.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


def script_type(addr) -> str:
    if not isinstance(addr, str) or not addr:
        return "unknown"
    if addr.startswith("1"):
        return "p2pkh"
    if addr.startswith("3"):
        return "p2sh"
    if addr.startswith("bc1p"):
        return "p2tr"
    if addr.startswith("bc1q"):
        return "p2wpkh" if len(addr) == 42 else "p2wsh"
    return "other"


@dataclass
class Dataset:
    seeds: pd.DataFrame  # address, file, n_tx_file, limit, truncated
    txs: pd.DataFrame  # uma linha por transação (deduplicada entre arquivos)
    ios: pd.DataFrame  # uma linha por input/output
    bad_files: list = field(default_factory=list)


def load_dataset(folder: str | Path, max_files: int | None = None) -> Dataset:
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {folder}")

    seeds, tx_rows, io_rows, bad = [], [], [], []
    seen: set[str] = set()

    for p in files:
        address = p.name[:-5] if p.suffix == ".json" else p.name
        try:
            data = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            bad.append((p.name, str(exc)))
            continue
        txs = data.get("transactions", []) if isinstance(data, dict) else data
        limit = data.get("limit") if isinstance(data, dict) else None
        seeds.append(
            dict(
                address=address,
                file=p.name,
                n_tx_file=len(txs),
                limit=limit,
                truncated=bool(limit and len(txs) >= limit),
            )
        )
        for tx in txs:
            txid = tx.get("txId") or tx.get("txid")
            if not txid or txid in seen:
                continue
            seen.add(txid)
            ins, outs = tx.get("inputs", []), tx.get("outputs", [])
            weight = tx.get("weight") or (tx.get("size") or 0) * 4
            fee = tx.get("fee") or 0
            tx_rows.append(
                dict(
                    txid=txid,
                    block_height=tx.get("blockHeight"),
                    time=tx.get("time"),
                    fee=fee,
                    size=tx.get("size"),
                    weight=weight,
                    fee_rate=fee / (weight / 4) if weight else 0.0,
                    rbf=bool(tx.get("rbf")),
                    version=tx.get("version"),
                    locktime=tx.get("locktime"),
                    n_in=len(ins),
                    n_out=len(outs),
                    total_in=sum(i.get("value") or 0 for i in ins),
                    total_out=sum(o.get("value") or 0 for o in outs),
                )
            )
            for i in ins:
                io_rows.append(
                    dict(
                        txid=txid,
                        role="in",
                        address=i.get("address"),
                        value=i.get("value") or 0,
                        ref_txid=i.get("txid"),
                        ref_out=i.get("output"),
                        spent=None,
                        pos=None,
                    )
                )
            for k, o in enumerate(outs):
                io_rows.append(
                    dict(
                        txid=txid,
                        role="out",
                        address=o.get("address"),
                        value=o.get("value") or 0,
                        ref_txid=None,
                        ref_out=None,
                        spent=o.get("spent"),
                        pos=k,
                    )
                )

    return Dataset(
        seeds=pd.DataFrame(seeds),
        txs=pd.DataFrame(tx_rows),
        ios=pd.DataFrame(io_rows),
        bad_files=bad,
    )
