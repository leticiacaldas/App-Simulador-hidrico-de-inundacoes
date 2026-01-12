# Simulador Híbrido de Inundações (Streamlit)

Este pacote contém apenas os arquivos necessários para publicar no Streamlit Community Cloud.

## Rodar localmente
- Python 3.11
- Instale dependências:
```
pip install -r requirements.txt
```
- Execute:
```
streamlit run app_lisflood_streamlit.py
```

## Publicar no Streamlit Cloud
1. Suba esta pasta (release_streamlit_app) como um repositório no GitHub.
2. Em streamlit.io → New app → selecione o repo/branch.
3. Main file: `app_lisflood_streamlit.py`.
4. Deploy.

## Observações
- Os dados de entrada (DEM, vetores etc.) devem ser enviados via upload na própria interface.
- Ícones e logos ficam em `logos/`.
- Configurações do servidor/tema: `.streamlit/config.toml`.
