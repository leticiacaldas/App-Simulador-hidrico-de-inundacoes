#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "release_streamlit_app"

FILES_ROOT = [
    "app_lisflood_streamlit.py",
    "requirements.txt",
    "runtime.txt",
]

# Pasta shapes (somente código, sem dados)
SHAPES_FILES = [
    ROOT / "shapes" / "__init__.py",
    ROOT / "shapes" / "design.py",
]

# Ícones e logos sugeridos
ICON_NAMES = [
    "florestamento.png", "diques.png", "sistemaDrenagem.png", "aterroNoTerreno.png",
    # alternativas aceitas pelo código
    "tree.png", "arvore.png", "dike.png", "barragem.png", "drainage.png", "drenagem.png", "fill.png", "aterro.png",
]
LOGO_NAMES = ["logo.png", "logoPlataforma.png"]

STREAMLIT_DIR = ROOT / ".streamlit"


def safe_copy(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def main():
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True, exist_ok=True)

    # Copiar arquivos raiz
    for rel in FILES_ROOT:
        src = ROOT / rel
        dst = TARGET / Path(rel).name
        if not safe_copy(src, dst):
            print(f"[WARN] Arquivo não encontrado: {src}")

    # Copiar .streamlit/config.toml (se existir) ou criar um mínimo
    cfg_src = STREAMLIT_DIR / "config.toml"
    cfg_dst = TARGET / ".streamlit" / "config.toml"
    if not safe_copy(cfg_src, cfg_dst):
        cfg_dst.parent.mkdir(parents=True, exist_ok=True)
        cfg_dst.write_text("""
[server]
headless = true
enableCORS = true

[browser]
gatherUsageStats = false

[theme]
base = "light"
""".strip()+"\n", encoding="utf-8")

    # Copiar pasta shapes essencial
    for f in SHAPES_FILES:
        dst = TARGET / "shapes" / f.name
        if not safe_copy(f, dst):
            print(f"[WARN] Shapes faltando: {f}")

    # Copiar logos e ícones se existirem
    logos_src_dir = ROOT / "logos"
    if logos_src_dir.exists():
        # Logos
        for name in LOGO_NAMES:
            safe_copy(logos_src_dir / name, TARGET / "logos" / name)
        # Ícones
        for name in ICON_NAMES:
            safe_copy(logos_src_dir / name, TARGET / "logos" / name)

    # Criar .gitignore
    (TARGET / ".gitignore").write_text("""
# Python
__pycache__/
*.pyc
.venv/
.env

# Dados pesados (não subir)
*.tif
*.tiff
*.ovr
*.aux.xml
*.gpkg
*.shp
*.shx
*.dbf
*.prj
*.qmd
*.asc
*.tfw
*.zip

# Artefatos/gerados
*.mp4
*.gif
*.png

# IDE/OS
.vscode/
.DS_Store
Thumbs.db
""".strip()+"\n", encoding="utf-8")

    # Criar README
    (TARGET / "README.md").write_text("""
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
""".strip()+"\n", encoding="utf-8")

    print(f"Release preparada em: {TARGET}")


if __name__ == "__main__":
    main()
