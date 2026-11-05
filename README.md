# sextortion-analysis

Clusterização + análise de grafo de endereços Bitcoin (um arquivo JSON por endereço em `addresses/`).

Pra rodar o código, instale o package manager uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Instale as dependencias:
```bash
uv sync

```
Rode o projeto:
```bash
uv run sextortion-analysis --addresses addresses --out out 
```
