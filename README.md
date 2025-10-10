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
