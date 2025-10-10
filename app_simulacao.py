import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
import geopandas as gpd
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from matplotlib.animation import FuncAnimation
import numpy.ma as ma
from io import BytesIO
import os
import tempfile
import shutil
import contextily as ctx
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# ===================================================================
# FUNÇÃO AUXILIAR PARA MANIPULAR SHAPEFILES
# ===================================================================
def process_shapefile_upload(uploaded_files):
    """
    Salva os arquivos de shapefile enviados em um diretório temporário
    e retorna o caminho para o arquivo .shp principal.
    Retorna o caminho do .shp e o caminho do diretório temporário.
    """
    if not uploaded_files:
        return None, None

    temp_dir = tempfile.mkdtemp()
    shp_path = None

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if uploaded_file.name.lower().endswith('.shp'):
            shp_path = file_path

    if not shp_path:
        shutil.rmtree(temp_dir)
        st.error("Nenhum arquivo .shp encontrado nos arquivos enviados. Por favor, inclua o arquivo .shp e seus arquivos relacionados (.shx, .dbf).")
        return None, None

    return shp_path, temp_dir


def get_basemap_source(tipo_mapa):
    """
    Retorna a fonte do mapa base baseada no tipo selecionado
    """
    map_sources = {
        'Google Satellite': ctx.providers.GoogleStreet,  # Google Hybrid como fallback
        'OpenStreetMap': ctx.providers.OpenStreetMap.Mapnik,
        'CartoDB Positron': ctx.providers.CartoDB.Positron,
        'Stamen Terrain': ctx.providers.Stamen.Terrain
    }
    return map_sources.get(tipo_mapa, ctx.providers.OpenStreetMap.Mapnik)


# ===================================================================
# CONFIGURAÇÃO DA PÁGINA DO STREAMLIT
# ===================================================================
st.set_page_config(layout="wide", page_title="Simulador de Inundação", page_icon="🌊")

# --- ESTADO DA SESSÃO ---
if 'preview_image' not in st.session_state:
    st.session_state.preview_image = None
if 'simulation_gif' not in st.session_state:
    st.session_state.simulation_gif = None

# ===================================================================
# PAINEL DE CONTROLE (SIDEBAR)
# ===================================================================
with st.sidebar:
    st.title(" Simulador de Enchentes")
    st.markdown("Configure os parâmetros e execute a simulação.")

    with st.expander("📂 Arquivos de Entrada", expanded=True):
        uploaded_dem_file = st.file_uploader(
            "Selecione o arquivo DEM/DOM (.tif)",
            type=['tif', 'tiff']
        )
        uploaded_shp_files = st.file_uploader(
            "Selecione os arquivos do Shapefile (.shp, .shx, .dbf, etc.)",
            type=['shp', 'shx', 'dbf', 'prj', 'cpg', 'sbn', 'sbx'],
            accept_multiple_files=True
        )

    if uploaded_dem_file and uploaded_shp_files:
        with st.expander("⚙️ Parâmetros da Simulação"):
            fator_reducao = st.slider("Fator de Redução (Detalhe)", 1, 10, 4, help="Aumente para acelerar, diminua para mais detalhes.")
            chuva_mm_por_ciclo = st.number_input("Chuva por Ciclo (mm)", 0.0, 100.0, 5.0, 1.0)
            modo_agua = st.selectbox("Modo de Adição de Água", ['CHUVA_UNIFORME', 'FONTE_NO_RIO'], index=1)

        with st.expander("🗺️ Configurações do Mapa"):
            usar_mapa_base = st.checkbox("Mostrar mapa base (Google Earth/OpenStreetMap)", value=True)
            if usar_mapa_base:
                tipo_mapa = st.selectbox("Tipo de Mapa Base", [
                    'Google Satellite',
                    'OpenStreetMap', 
                    'CartoDB Positron',
                    'Stamen Terrain'
                ], index=0)
                transparencia_dem = st.slider("Transparência do DEM", 0.0, 1.0, 0.7, help="0 = Transparente, 1 = Opaco")

        with st.expander("🎬 Parâmetros da Animação"):
            total_ciclos = st.slider("Duração (Ciclos)", 10, 500, 150)
            intervalo_ms = st.slider("Intervalo por Ciclo (ms)", 50, 1000, 200, help="Aumente para uma animação mais lenta.")

        st.subheader("⚡ Ações")
        col1, col2, col3 = st.columns(3)
        with col1:
            run_button = st.button("Executar", type="primary")
        with col2:
            preview_button = st.button("Visualizar")
        with col3:
            reset_button = st.button("Limpar")
    else:
        st.info("Por favor, carregue o arquivo DEM e os arquivos do Shapefile para continuar.")
        run_button = preview_button = reset_button = False

# ===================================================================
# ÁREA PRINCIPAL
# ===================================================================
st.title("Painel de Visualização")

status_placeholder = st.empty()
output_placeholder = st.empty()

if reset_button:
    st.session_state.preview_image = None
    st.session_state.simulation_gif = None
    st.rerun()

# Ação do botão Visualizar
if preview_button:
    if not uploaded_dem_file or not uploaded_shp_files:
        status_placeholder.warning("Por favor, carregue todos os arquivos antes de visualizar.")
    else:
        temp_dir_preview = None
        try:
            with st.spinner("Gerando visualização prévia..."):
                shp_path, temp_dir_preview = process_shapefile_upload(uploaded_shp_files)
                if shp_path:
                    with rio.open(uploaded_dem_file) as src:
                        raster_crs = src.crs
                        transform_original = src.transform
                        nova_altura = src.height // fator_reducao
                        novo_comprimento = src.width // fator_reducao
                        transform_reduzida = from_origin(transform_original.xoff, transform_original.yoff, transform_original.a * fator_reducao, transform_original.e * fator_reducao)
                        dem_data = src.read(1, out_shape=(nova_altura, novo_comprimento), resampling=Resampling.average)
                        left, top = transform_reduzida * (0, 0)
                        right, bottom = transform_reduzida * (novo_comprimento, nova_altura)
                        extent = [left, right, bottom, top]

                    with rio.Env(SHAPE_RESTORE_SHX='YES'):
                        fontes_gdf = gpd.read_file(shp_path)
                    
                    if fontes_gdf.crs is None:
                        fontes_gdf = fontes_gdf.set_crs(raster_crs)
                    
                    if fontes_gdf.crs != raster_crs:
                        fontes_gdf = fontes_gdf.to_crs(raster_crs)

                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Adiciona o mapa base se selecionado
                    if usar_mapa_base:
                        try:
                            # Converte para Web Mercator (EPSG:3857) para compatibilidade com mapas base
                            fontes_gdf_web = fontes_gdf.to_crs('EPSG:3857')
                            
                            # Calcula os bounds em Web Mercator para o contexto do mapa
                            minx, miny, maxx, maxy = fontes_gdf_web.total_bounds
                            
                            # Adiciona o mapa base
                            basemap_source = get_basemap_source(tipo_mapa)
                            ctx.add_basemap(ax, crs='EPSG:3857', source=basemap_source, 
                                          extent=[minx, maxx, miny, maxy])
                            
                            # Ajusta o extent e CRS do plot
                            ax.set_xlim(minx, maxx)
                            ax.set_ylim(miny, maxy)
                            
                            # Reprojecta o DEM para Web Mercator e exibe com transparência
                            # Aqui usamos uma abordagem simplificada - apenas o shapefile com mapa base
                            fontes_gdf_web.plot(ax=ax, facecolor='blue', edgecolor='blue', alpha=0.7)
                            ax.set_title(f"Visualização com {tipo_mapa}: Fontes de Água")
                            
                        except Exception as e:
                            # Se falhar, volta para o modo normal sem mapa base
                            st.warning(f"Não foi possível carregar o mapa base ({e}). Usando visualização padrão.")
                            ax.imshow(dem_data, cmap='terrain', extent=extent, alpha=transparencia_dem)
                            fontes_gdf.plot(ax=ax, facecolor='blue', edgecolor='blue')
                            ax.set_title("Visualização Prévia: Terreno e Fontes de Água")
                    else:
                        # Modo padrão sem mapa base
                        ax.imshow(dem_data, cmap='terrain', extent=extent)
                        fontes_gdf.plot(ax=ax, facecolor='blue', edgecolor='blue')
                        ax.set_title("Visualização Prévia: Terreno e Fontes de Água")
                    
                    ax.set_xlabel("Coordenada X (metros)")
                    ax.set_ylabel("Coordenada Y (metros)")
                    
                    buf = BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    st.session_state.preview_image = buf
                    st.session_state.simulation_gif = None
                    plt.close(fig)
                    status_placeholder.success("Visualização prévia gerada!")

        except Exception as e:
            status_placeholder.error(f"Ocorreu um erro inesperado na visualização: {e}")
        finally:
            if temp_dir_preview and os.path.exists(temp_dir_preview):
                shutil.rmtree(temp_dir_preview)

# MOTOR DA SIMULAÇÃO
if run_button:
    if not uploaded_dem_file or not uploaded_shp_files:
        status_placeholder.warning("Por favor, carregue todos os arquivos antes de executar.")
    else:
        st.session_state.preview_image = None
        temp_dir_run = None
        try:
            status_placeholder.info("Iniciando a simulação... Por favor, aguarde.")
            
            with st.spinner("Etapa 1/4: Carregando e processando o terreno..."):
                with rio.open(uploaded_dem_file) as src:
                    raster_crs = src.crs
                    transform_original = src.transform
                    nova_altura = src.height // fator_reducao
                    novo_comprimento = src.width // fator_reducao
                    dem_data = src.read(1, out_shape=(nova_altura, novo_comprimento), resampling=Resampling.average)
            
            with st.spinner("Etapa 2/4: Alinhando as fontes de água..."):
                shp_path, temp_dir_run = process_shapefile_upload(uploaded_shp_files)
                if not shp_path:
                    raise Exception("Falha ao processar os arquivos do shapefile.")

                with rio.Env(SHAPE_RESTORE_SHX='YES'):
                    fontes_gdf = gpd.read_file(shp_path)

                if fontes_gdf.crs is None:
                    fontes_gdf = fontes_gdf.set_crs(raster_crs)

                if fontes_gdf.crs != raster_crs:
                    fontes_gdf = fontes_gdf.to_crs(raster_crs)
                
                transform_reduzida = from_origin(transform_original.xoff, transform_original.yoff, transform_original.a * fator_reducao, transform_original.e * fator_reducao)
                mascara_fontes = rasterize(shapes=fontes_gdf.geometry, out_shape=dem_data.shape, transform=transform_reduzida, fill=0, all_touched=True, dtype=np.float32)
                mascara_fontes_bool = mascara_fontes.astype(bool)

            water_level = np.zeros_like(dem_data, dtype=float)
            rain_amount = chuva_mm_por_ciclo / 1000.0

            with st.spinner(f"Etapa 3/4: Executando {total_ciclos} ciclos da simulação..."):
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Configuração do fundo baseada na escolha do usuário
                if usar_mapa_base:
                    try:
                        # Para animação, é mais complexo usar mapa base em tempo real
                        # Usamos apenas o DEM com transparência por enquanto
                        ax.imshow(dem_data, cmap='terrain', alpha=transparencia_dem)
                    except:
                        ax.imshow(dem_data, cmap='terrain')
                else:
                    ax.imshow(dem_data, cmap='terrain')
                    
                water_im = ax.imshow(np.zeros_like(dem_data), cmap='Blues', alpha=0.0)
                
                progress_bar = st.progress(0)

                def update(frame):
                    if modo_agua == 'CHUVA_UNIFORME':
                        water_level[:] += rain_amount
                    elif modo_agua == 'FONTE_NO_RIO':
                        water_level[mascara_fontes_bool] += rain_amount
                    
                    flow_level = np.copy(water_level)
                    for y in range(1, dem_data.shape[0] - 1):
                        for x in range(1, dem_data.shape[1] - 1):
                            if flow_level[y, x] <= 0: continue
                            total_height_here = dem_data[y, x] + flow_level[y, x]
                            neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
                            lower_neighbors, total_diff = [], 0
                            for ny, nx in neighbors:
                                total_height_neighbor = dem_data[ny, nx] + flow_level[ny, nx]
                                if total_height_here > total_height_neighbor:
                                    diff = total_height_here - total_height_neighbor
                                    lower_neighbors.append({'pos': (ny, nx), 'diff': diff})
                                    total_diff += diff
                            if total_diff > 0:
                                total_flow = min(flow_level[y, x], total_diff * 0.1)
                                water_level[y, x] -= total_flow
                                for neighbor in lower_neighbors:
                                    flow_share = (neighbor['diff'] / total_diff) * total_flow
                                    water_level[neighbor['pos']] += flow_share
                    
                    masked_water = ma.masked_where(water_level <= 0, water_level)
                    water_im.set_data(masked_water)
                    water_im.set_alpha(1.0)
                    progress_bar.progress((frame + 1) / total_ciclos)
                    return [water_im]

                anim = FuncAnimation(fig, update, frames=total_ciclos, blit=True, interval=intervalo_ms)
                
                with st.spinner("Etapa 4/4: Gerando o arquivo GIF..."):
                    # Salva em arquivo temporário primeiro
                    temp_gif_path = os.path.join(tempfile.gettempdir(), "temp_simulation.gif")
                    anim.save(temp_gif_path, writer='imagemagick')
                    
                    # Lê o arquivo e converte para BytesIO
                    gif_buffer = BytesIO()
                    with open(temp_gif_path, 'rb') as f:
                        gif_buffer.write(f.read())
                    gif_buffer.seek(0)  # Volta para o início do buffer
                    
                    # Remove o arquivo temporário
                    if os.path.exists(temp_gif_path):
                        os.remove(temp_gif_path)
                    
                    st.session_state.simulation_gif = gif_buffer
                    plt.close(fig)
            
            status_placeholder.success("Simulação concluída com sucesso!")

        except Exception as e:
            status_placeholder.error(f"Ocorreu um erro inesperado na simulação: {e}")
        finally:
            if temp_dir_run and os.path.exists(temp_dir_run):
                shutil.rmtree(temp_dir_run)
            if 'progress_bar' in locals():
                progress_bar.empty()

# MOSTRAR RESULTADOS
if st.session_state.preview_image:
    output_placeholder.image(st.session_state.preview_image, caption="Visualização Prévia")
elif st.session_state.simulation_gif:
    output_placeholder.image(st.session_state.simulation_gif, caption="Animação da Simulação de Inundação")
else:
    output_placeholder.info("Aguardando ação: carregue os arquivos e clique em 'Visualizar' ou 'Executar'.")
