# Dockerfile para deploy do seu app Streamlit no Render
FROM python:3.11-slim

# Instala dependências do sistema (ajuste conforme necessário)
RUN apt-get update && apt-get install -y \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Define diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto
COPY . /app

# Instala as dependências Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expõe a porta padrão do Streamlit
EXPOSE 8501

# Comando para iniciar o app
CMD ["streamlit", "run", "release_streamlit_app/app_lisflood_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
