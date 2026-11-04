# sextortion-analysis

Clusterização + análise de grafo de endereços Bitcoin (um arquivo JSON por endereço em `addresses/`).

## Uso
```bash
uv add numpy pandas scikit-learn networkx matplotlib   # se ainda não tiver
uv run sextortion-analysis --addresses addresses --out results
# teste rápido: --max-files 200 | forçar k: --k 4 | método principal: --primary dbscan
# incluir métricas de grafo na clusterização: --with-graph-features
```
Sem o script instalado: `uv run python -m sextortion_analysis.cli ...`

Saída em `results/`: `report.md` (cole na conversa), `summary.json`, CSVs, PNGs e GEXF (Gephi).
