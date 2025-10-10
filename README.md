# Simulador híbrido de inundações

Aplicativo Streamlit para simulação rápida (NumPy) e execução avançada do LISFLOOD via Docker.

## Requisitos

- Python 3.10+ (recomendado usar `.venv`)
- Docker (para a aba LISFLOOD)
- ffmpeg (opcional, para MP4) ou `imageio-ffmpeg`

## Configuração rápida

```bash
# Ativar ambiente virtual (Linux)
source .venv/bin/activate

# Rodar o app
python -m streamlit run app_lisflood_streamlit.py
```

Acesse: <http://localhost:8501>

## Estrutura principal

- `app_lisflood_streamlit.py`: app principal (NumPy + LISFLOOD)
- `lisflood_integration.py`: utilitários de integração com LISFLOOD (Docker)
- `design.py` e `shapes/`: estilos e cabeçalho
- `templates/`: templates XML do LISFLOOD (opcional)

## Notas

- Dados geoespaciais grandes (GeoTIFF, shapefiles, etc.) são ignorados pelo Git via `.gitignore`.
- A integração com Opik (telemetria) é opcional e desativa-se automaticamente se o pacote não estiver instalado.

## Deploy no Streamlit Cloud

Clique para implantar este app no Streamlit Cloud:

[Deploy 1‑clique no Streamlit Cloud](https://share.streamlit.io/deploy?repository=leticiacaldas/App-Simulador-hidrico-de-inundacoes&branch=main&mainModule=app_lisflood_streamlit.py)

Caso prefira fazer manualmente:

1. Acesse <https://share.streamlit.io>
1. Conecte sua conta do GitHub
1. Em "New app": selecione `leticiacaldas/App-Simulador-hidrico-de-inundacoes`, branch `main`, e `app_lisflood_streamlit.py` como arquivo principal
1. Verifique as configurações avançadas:
   - Python: 3.11 (o arquivo `runtime.txt` já define)
   - Requirements: será lido de `requirements.txt`
1. Lance o app e acompanhe os logs

## Publicar no GitHub

1. Inicialize o repositório e primeiro commit:

 ```bash
 git init
 git add .
 git commit -m "chore: projeto inicial"
 ```

1. Crie um repositório vazio no GitHub e pegue a URL (por exemplo, `https://github.com/<usuario>/<repo>.git`).

1. Configure o remoto e envie:

 ```bash
 git remote add origin <URL_DO_REPO>
 git branch -M main
 git push -u origin main
 ```
