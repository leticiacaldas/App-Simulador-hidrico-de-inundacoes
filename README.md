# SimHidrion – Sistema de Simulação Hidrodinâmica para Análise de Inundações

SimHidrion é um aplicativo Streamlit para simulação hidrodinâmica simplificada baseada em NumPy, com foco em agilidade, visualização e geração de produtos geoespaciais padronizados. Opcionalmente, integra-se ao LISFLOOD via Docker para cenários avançados.

## Recursos principais

- Simulação vetorizada em grade raster (NumPy), com chuva uniforme ou espacial por atributo de vetores
- Hietograma CSV (tempo_min, mm_h) para chuva temporal por ciclo
- Infiltração constante (mm/h) ou por raster GeoTIFF reamostrado ao DEM
- Visualização com DOM/ortofoto de fundo, relevo (hillshade) e contraste ajustável
- IA (RandomForest) para mapa de probabilidade de inundação e validação (curvas ROC/PR)
- Exportação completa de artefatos pós-processados:
  - GeoTIFF: lâmina d’água (`lamina_agua.tif`), mancha binária (`mancha_agua.tif`), acumulação (`acumulacao_agua.tif`), intensidade da chuva (`chuva_intensidade.tif`), excedência multibanda (`excedencia_multibanda.tif`)
  - GeoTIFF estilizado (RGBA): água simulada com transparência sob limiar (`lamina_agua_rgba.tif`)
  - Vetores: polígonos de inundação (`inundacao.geojson`, `inundacao.gpkg`)
  - PNGs de overlay: DOM + água simulada, DOM + probabilidade IA
  - CSVs: dados da simulação (`dados_simulacao.csv`), área de inundação (`area_inundacao.csv`), série temporal de ponto (`serie_ponto.csv`)
  - ZIP do cenário: `simhidrion_relatorio_cenario.zip` agrega todos os artefatos

## Requisitos

- Python 3.11+
- ffmpeg (opcional, para MP4) ou `imageio-ffmpeg`
- Docker (opcional, somente se usar LISFLOOD)

## Como executar (local)

```bash
# Instalar dependências
pip install -r requirements.txt

# Rodar o app principal
streamlit run release_streamlit_app/app_lisflood_streamlit.py

# Acessar no navegador
# http://localhost:8501
```

Versão leve (somente NumPy, menos dependências):

```bash
pip install -r requirements-minimal.txt
streamlit run app_numpy_streamlit.py
```

## Estrutura

- `release_streamlit_app/app_lisflood_streamlit.py`: app principal (SimHidrion)
- `app_numpy_streamlit.py`: app leve (NumPy puro)
- `lisflood_integration.py`: utilitários de integração com LISFLOOD (Docker) – opcional
- `design.py`, `shapes/`: estilos e cabeçalho
- `templates/`: templates XML do LISFLOOD – opcional

## Extensões recomendadas (VS Code)

Para melhor experiência no desenvolvimento:

- `ms-python.vscode-pylance`
- `ms-python.python`
- `ms-toolsai.jupyter`
- `eamodio.gitlens`
- `ms-ceintl.vscode-language-pack-pt-BR` (opcional)

Um arquivo `.vscode/extensions.json` foi incluído com estas recomendações.

## Deploy no Streamlit Cloud

Implante com 1‑clique:

[Deploy 1‑clique](https://share.streamlit.io/deploy?repository=leticiacaldas/SimHidrion&branch=main&mainModule=release_streamlit_app/app_lisflood_streamlit.py)

Manual:

1. Acesse <https://share.streamlit.io>
2. Conecte sua conta do GitHub
3. Selecione o repositório `leticiacaldas/SimHidrion`, branch `main`, e `release_streamlit_app/app_lisflood_streamlit.py` como arquivo principal
4. Configure:
   - Python: 3.11 (definido em `runtime.txt`)
   - Requirements: `requirements.txt`
5. Lance o app e acompanhe os logs

## Publicar alterações no GitHub

```bash
git add .
git commit -m "refactor: renomeia para SimHidrion e atualiza README"
git push
```
