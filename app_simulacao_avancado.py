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
import time

# ===================================================================
# CONFIGURAÇÃO DA PÁGINA AVANÇADA
# ===================================================================
st.set_page_config(
    layout="wide", 
    page_title="Simulador de Inundação Avançado", 
    page_icon="🌊",
    initial_sidebar_state="expanded"
)

# ===================================================================
# ESTADO DA SESSÃO EXPANDIDO
# ===================================================================
if 'preview_image' not in st.session_state:
    st.session_state.preview_image = None
if 'simulation_gif' not in st.session_state:
    st.session_state.simulation_gif = None
if 'simulation_stats' not in st.session_state:
    st.session_state.simulation_stats = None
if 'file_info' not in st.session_state:
    st.session_state.file_info = {}
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = []

# ===================================================================
# FUNÇÕES AUXILIARES EXPANDIDAS
# ===================================================================
def process_shapefile_upload(uploaded_files):
    """Processa arquivos de shapefile e retorna informações"""
    if not uploaded_files:
        return None, None, {}

    temp_dir = tempfile.mkdtemp()
    shp_path = None
    file_info = {
        'total_files': len(uploaded_files),
        'file_types': [],
        'total_size_mb': 0
    }

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if uploaded_file.name.lower().endswith('.shp'):
            shp_path = file_path
        
        file_info['file_types'].append(uploaded_file.name.split('.')[-1].upper())
        file_info['total_size_mb'] += uploaded_file.size / (1024 * 1024)

    if not shp_path:
        shutil.rmtree(temp_dir)
        st.error("❌ Nenhum arquivo .shp encontrado!")
        return None, None, {}

    return shp_path, temp_dir, file_info

def get_basemap_source(tipo_mapa):
    """Retorna fonte do mapa base com provedores válidos"""
    try:
        map_sources = {
            'Google Satellite': ctx.providers.Google.Satellite,
            'OpenStreetMap': ctx.providers.OpenStreetMap.Mapnik,
            'CartoDB Positron': ctx.providers.CartoDB.Positron,
            'CartoDB Dark Matter': ctx.providers.CartoDB.DarkMatter,
            'Stamen Terrain': ctx.providers.Stamen.Terrain,
            'Stamen Watercolor': ctx.providers.Stamen.Watercolor,
            'ESRI World Imagery': ctx.providers.Esri.WorldImagery
        }
        return map_sources.get(tipo_mapa, ctx.providers.OpenStreetMap.Mapnik)
    except Exception as e:
        # Fallback para OpenStreetMap se houver erro
        st.warning(f"⚠️ Provedor {tipo_mapa} não disponível. Usando OpenStreetMap.")
        return ctx.providers.OpenStreetMap.Mapnik

def analyze_dem_file(dem_file):
    """Analisa arquivo DEM e retorna estatísticas"""
    try:
        with rio.open(dem_file) as src:
            data = src.read(1)
            stats = {
                'width': src.width,
                'height': src.height,
                'crs': str(src.crs),
                'min_elevation': float(np.nanmin(data)),
                'max_elevation': float(np.nanmax(data)),
                'mean_elevation': float(np.nanmean(data)),
                'resolution_x': abs(src.transform[0]),
                'resolution_y': abs(src.transform[4]),
                'area_km2': (src.width * abs(src.transform[0]) * src.height * abs(src.transform[4])) / 1000000
            }
        return stats
    except Exception as e:
        st.error(f"Erro ao analisar DEM: {e}")
        return {}

def calculate_simulation_stats(water_level, dem_data, rain_amount, frame):
    """Calcula estatísticas da simulação"""
    flooded_pixels = np.sum(water_level > 0)
    total_pixels = water_level.size
    flooded_percentage = (flooded_pixels / total_pixels) * 100
    total_water_volume = np.sum(water_level)
    max_water_depth = np.max(water_level)
    avg_water_depth = np.mean(water_level[water_level > 0]) if flooded_pixels > 0 else 0
    
    return {
        'frame': frame,
        'flooded_area_percent': flooded_percentage,
        'flooded_pixels': flooded_pixels,
        'total_water_volume': total_water_volume,
        'max_depth': max_water_depth,
        'avg_depth': avg_water_depth,
        'rain_added': rain_amount
    }

# ===================================================================
# INTERFACE PRINCIPAL AVANÇADA
# ===================================================================
st.title("🌊 Simulador de Inundação Avançado")
st.markdown("**Versão Premium** • Análise Avançada • Múltiplos Formatos • Estatísticas em Tempo Real")

# Layout principal em colunas
col_main, col_info = st.columns([3, 1])

with col_main:
    # ===================================================================
    # PAINEL DE CONTROLE EXPANDIDO
    # ===================================================================
    with st.sidebar:
        st.header("⚙️ Controles")
        
        # Seção de arquivos com informações
        with st.expander("📂 Arquivos de Entrada", expanded=True):
            uploaded_dem_file = st.file_uploader(
                "📊 Arquivo DEM/DOM (.tif)",
                type=['tif', 'tiff'],
                help="Modelo Digital de Elevação"
            )
            uploaded_shp_files = st.file_uploader(
                "🗺️ Arquivos do Shapefile",
                type=['shp', 'shx', 'dbf', 'prj', 'cpg', 'sbn', 'sbx'],
                accept_multiple_files=True,
                help="Selecione TODOS os arquivos (.shp, .shx, .dbf, etc.)"
            )
            
            # Análise de arquivos
            if uploaded_dem_file:
                with st.spinner("Analisando DEM..."):
                    dem_stats = analyze_dem_file(uploaded_dem_file)
                    st.session_state.file_info['dem'] = dem_stats
            
            if uploaded_shp_files:
                shp_path, temp_dir_info, shp_info = process_shapefile_upload(uploaded_shp_files)
                if shp_info:
                    st.session_state.file_info['shapefile'] = shp_info

        # Parâmetros avançados apenas se arquivos estiverem carregados
        if uploaded_dem_file and uploaded_shp_files:
            
            # Parâmetros de simulação
            with st.expander("🔧 Parâmetros da Simulação"):
                col1, col2 = st.columns(2)
                with col1:
                    fator_reducao = st.slider("🎯 Resolução (Detalhe)", 1, 8, 2, 
                                            help="1 = Máximo detalhe (lento), 8 = Baixo detalhe (rápido)")
                    chuva_mm_por_ciclo = st.number_input("🌧️ Chuva/Ciclo (mm)", 0.0, 100.0, 5.0, 1.0)
                    
                    # Informação sobre o fator de redução
                    if fator_reducao <= 2:
                        st.info("🔍 **Alta resolução**: Máximo detalhe do terreno")
                    elif fator_reducao <= 4:
                        st.info("⚖️ **Resolução balanceada**: Bom equilíbrio")
                    else:
                        st.warning("⚡ **Baixa resolução**: Rápido mas menos preciso")
                        
                with col2:
                    modo_agua = st.selectbox("💧 Modo", ['CHUVA_UNIFORME', 'FONTE_NO_RIO'], index=1)
                    fisica_agua = st.selectbox("⚗️ Física", ['SIMPLES', 'AVANÇADA'], help="Avançada: mais realista, mais lenta")

            # Configurações de mapa
            with st.expander("🗺️ Configurações de Visualização"):
                usar_mapa_base = st.checkbox("🛰️ Mapa Base", value=True)
                if usar_mapa_base:
                    tipo_mapa = st.selectbox("Tipo", [
                        'ESRI World Imagery',      # Mais confiável para satélite
                        'OpenStreetMap',           # Sempre disponível
                        'CartoDB Positron',        # Limpo e claro
                        'CartoDB Dark Matter',     # Estilo escuro
                        'Stamen Terrain',          # Boa para topografia
                        'Google Satellite'         # Pode não funcionar sempre
                    ], index=0, help="ESRI World Imagery é o mais confiável para imagens de satélite")
                    transparencia_dem = st.slider("🔍 Transparência DEM", 0.0, 1.0, 0.3, 
                                                 help="0 = Totalmente transparente, 1 = Totalmente opaco")
                else:
                    transparencia_dem = 1.0
                    
                # Configurações adicionais de visualização
                mostrar_elevacao = st.checkbox("🏔️ Colorir por Elevação", value=True, 
                                             help="Usa cores para mostrar diferentes altitudes")
                estilo_fontes = st.selectbox("💧 Estilo das Fontes", [
                    'Azul Sólido', 'Azul Transparente', 'Azul Contorno'
                ], help="Como mostrar as fontes de água")

            # Parâmetros de animação
            with st.expander("🎬 Animação & Export"):
                col1, col2 = st.columns(2)
                with col1:
                    total_ciclos = st.slider("⏱️ Duração (ciclos)", 10, 500, 100)
                    intervalo_ms = st.slider("⏰ Velocidade (ms)", 50, 1000, 150)
                with col2:
                    formato_export = st.selectbox("💾 Formato", ['GIF', 'MP4'], index=0)
                    qualidade = st.selectbox("🎨 Qualidade", ['Alta', 'Média', 'Baixa'], index=1)

            # Opções avançadas
            with st.expander("🚀 Opções Avançadas"):
                preview_tempo_real = st.checkbox("👁️ Preview Tempo Real", help="Mostra evolução durante simulação")
                salvar_dados = st.checkbox("💾 Salvar Dados", help="Exporta dados da simulação")
                estatisticas_detalhadas = st.checkbox("📊 Stats Detalhadas", value=True)

            # Botões de ação
            st.subheader("🎮 Ações")
            col1, col2 = st.columns(2)
            with col1:
                visualizar_btn = st.button("👁️ Visualizar", type="secondary", width="stretch")
                executar_btn = st.button("🚀 Executar", type="primary", width="stretch")
            with col2:
                limpar_btn = st.button("🗑️ Limpar", width="stretch")
                export_btn = st.button("💾 Exportar", width="stretch")

        else:
            st.info("📁 **Carregue os arquivos DEM e Shapefile para começar**")
            visualizar_btn = executar_btn = limpar_btn = export_btn = False

    # ===================================================================
    # ÁREA DE RESULTADOS
    # ===================================================================
    if limpar_btn:
        st.session_state.preview_image = None
        st.session_state.simulation_gif = None
        st.session_state.simulation_stats = None
        st.session_state.simulation_data = []
        st.rerun()

    # Placeholder para resultados
    resultado_container = st.container()

    # ===================================================================
    # LÓGICA DE VISUALIZAÇÃO AVANÇADA
    # ===================================================================
    if visualizar_btn and uploaded_dem_file and uploaded_shp_files:
        with resultado_container:
            st.subheader("🔍 Visualização Prévia")
            
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0, text="Preparando visualização...")
                
            temp_dir_preview = None
            try:
                # Processamento
                progress_bar.progress(20, text="Carregando DEM...")
                shp_path, temp_dir_preview, _ = process_shapefile_upload(uploaded_shp_files)
                
                if shp_path:
                    progress_bar.progress(40, text="Processando terreno...")
                    with rio.open(uploaded_dem_file) as src:
                        raster_crs = src.crs
                        transform_original = src.transform
                        nova_altura = src.height // fator_reducao
                        novo_comprimento = src.width // fator_reducao
                        transform_reduzida = from_origin(transform_original.xoff, transform_original.yoff, 
                                                       transform_original.a * fator_reducao, transform_original.e * fator_reducao)
                        dem_data = src.read(1, out_shape=(nova_altura, novo_comprimento), resampling=Resampling.average)
                        left, top = transform_reduzida * (0, 0)
                        right, bottom = transform_reduzida * (novo_comprimento, nova_altura)
                        extent = [left, right, bottom, top]

                    progress_bar.progress(60, text="Carregando shapefile...")
                    with rio.Env(SHAPE_RESTORE_SHX='YES'):
                        fontes_gdf = gpd.read_file(shp_path)
                    
                    if fontes_gdf.crs is None:
                        fontes_gdf = fontes_gdf.set_crs(raster_crs)
                    if fontes_gdf.crs != raster_crs:
                        fontes_gdf = fontes_gdf.to_crs(raster_crs)

                    progress_bar.progress(80, text="Gerando visualização...")
                    
                    fig, ax = plt.subplots(figsize=(14, 10))
                    
                    # Configurar estilo das fontes de água
                    if estilo_fontes == 'Azul Sólido':
                        fonte_color, fonte_alpha, fonte_edge = 'blue', 0.8, 'navy'
                    elif estilo_fontes == 'Azul Transparente':
                        fonte_color, fonte_alpha, fonte_edge = 'cyan', 0.5, 'blue'
                    else:  # Azul Contorno
                        fonte_color, fonte_alpha, fonte_edge = 'none', 1.0, 'blue'
                    
                    # Aplicar mapa base se selecionado
                    if usar_mapa_base:
                        try:
                            fontes_gdf_web = fontes_gdf.to_crs('EPSG:3857')
                            minx, miny, maxx, maxy = fontes_gdf_web.total_bounds
                            
                            basemap_source = get_basemap_source(tipo_mapa)
                            ctx.add_basemap(ax, crs='EPSG:3857', source=basemap_source)
                            
                            ax.set_xlim(minx, maxx)
                            ax.set_ylim(miny, maxy)
                            
                            # Plotar fontes com estilo selecionado
                            fontes_gdf_web.plot(ax=ax, facecolor=fonte_color, edgecolor=fonte_edge, 
                                              alpha=fonte_alpha, linewidth=2)
                            ax.set_title(f"Visualizacao com {tipo_mapa} (Resolucao: {fator_reducao})", 
                                       fontsize=14, fontweight='bold')
                            
                        except Exception as e:
                            st.warning(f"⚠️ Erro no mapa base: {e}. Usando visualização padrão.")
                            # Fallback: mostrar terreno com elevação
                            if mostrar_elevacao:
                                im = ax.imshow(dem_data, cmap='terrain', extent=extent, alpha=transparencia_dem)
                                plt.colorbar(im, ax=ax, label='Elevação (m)', shrink=0.8)
                            else:
                                ax.imshow(dem_data, cmap='gray', extent=extent, alpha=transparencia_dem)
                            
                            fontes_gdf.plot(ax=ax, facecolor=fonte_color, edgecolor=fonte_edge, 
                                          alpha=fonte_alpha, linewidth=2)
                            ax.set_title(f"Visualizacao: Terreno e Fontes (Resolucao: {fator_reducao})", 
                                       fontsize=14, fontweight='bold')
                    else:
                        # Visualização apenas com terreno
                        if mostrar_elevacao:
                            im = ax.imshow(dem_data, cmap='terrain', extent=extent)
                            plt.colorbar(im, ax=ax, label='Elevação (m)', shrink=0.8)
                        else:
                            ax.imshow(dem_data, cmap='gray', extent=extent)
                        
                        fontes_gdf.plot(ax=ax, facecolor=fonte_color, edgecolor=fonte_edge, 
                                      alpha=fonte_alpha, linewidth=2)
                        ax.set_title(f"Visualizacao: Terreno e Fontes (Resolucao: {fator_reducao})", 
                                   fontsize=14, fontweight='bold')
                    
                    ax.set_xlabel("Coordenada X (metros)", fontsize=12)
                    ax.set_ylabel("Coordenada Y (metros)", fontsize=12)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    progress_bar.progress(100, text="✅ Visualização concluída!")
                    
                    buf = BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                    st.session_state.preview_image = buf
                    plt.close(fig)
                    
                    time.sleep(0.5)
                    progress_container.empty()

            except Exception as e:
                st.error(f"❌ Erro na visualização: {e}")
            finally:
                if temp_dir_preview and os.path.exists(temp_dir_preview):
                    shutil.rmtree(temp_dir_preview)

    # Mostrar resultado da visualização
    if st.session_state.preview_image:
        with resultado_container:
            st.image(st.session_state.preview_image, caption="🖼️ Prévia da Área de Simulação", width="stretch")

    # ===================================================================
    # MOTOR DA SIMULAÇÃO COMPLETA COM PROGRESSO
    # ===================================================================
    if executar_btn and uploaded_dem_file and uploaded_shp_files:
        with resultado_container:
            st.subheader("🚀 Simulação de Inundação")
            
            # Container para o progresso
            progress_main = st.container()
            simulation_container = st.container()
            
            temp_dir_sim = None
            try:
                with progress_main:
                    st.info("🔄 **Iniciando simulação completa...**")
                    main_progress = st.progress(0, text="Preparando simulação...")
                    
                # Etapa 1: Carregar DEM
                main_progress.progress(5, text="📊 Carregando arquivo DEM...")
                shp_path, temp_dir_sim, _ = process_shapefile_upload(uploaded_shp_files)
                
                if shp_path:
                    with rio.open(uploaded_dem_file) as src:
                        raster_crs = src.crs
                        transform_original = src.transform
                        nova_altura = src.height // fator_reducao
                        novo_comprimento = src.width // fator_reducao
                        transform_reduzida = from_origin(transform_original.xoff, transform_original.yoff, 
                                                       transform_original.a * fator_reducao, transform_original.e * fator_reducao)
                        dem_data = src.read(1, out_shape=(nova_altura, novo_comprimento), resampling=Resampling.average)

                    main_progress.progress(15, text="🗺️ Processando shapefile...")
                    
                    # Carregar shapefile
                    with rio.Env(SHAPE_RESTORE_SHX='YES'):
                        fontes_gdf = gpd.read_file(shp_path)
                    
                    if fontes_gdf.crs is None:
                        fontes_gdf = fontes_gdf.set_crs(raster_crs)
                    if fontes_gdf.crs != raster_crs:
                        fontes_gdf = fontes_gdf.to_crs(raster_crs)
                    
                    # Criar máscara de fontes
                    mascara_fontes = rasterize(shapes=fontes_gdf.geometry, out_shape=dem_data.shape, 
                                             transform=transform_reduzida, fill=0, all_touched=True, dtype=np.float32)
                    mascara_fontes_bool = mascara_fontes.astype(bool)

                    main_progress.progress(25, text="⚗️ Inicializando simulação...")
                    
                    # Inicializar simulação
                    water_level = np.zeros_like(dem_data, dtype=float)
                    rain_amount = chuva_mm_por_ciclo / 1000.0
                    
                    # Preparar animação
                    fig, ax = plt.subplots(figsize=(14, 10))
                    
                    # Fundo baseado nas configurações
                    if mostrar_elevacao:
                        dem_im = ax.imshow(dem_data, cmap='terrain', alpha=0.8)
                        plt.colorbar(dem_im, ax=ax, label='Elevação (m)', shrink=0.6)
                    else:
                        dem_im = ax.imshow(dem_data, cmap='gray', alpha=0.8)
                    
                    # Inicializar camada de água
                    water_im = ax.imshow(np.zeros_like(dem_data), cmap='Blues', alpha=0.0)
                    
                    ax.set_title(f"Simulação de Inundação (Progresso: 0%)", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Coordenada X", fontsize=12)
                    ax.set_ylabel("Coordenada Y", fontsize=12)
                    ax.grid(True, alpha=0.3)
                    
                    main_progress.progress(30, text="🎬 Executando simulação...")
                    
                    # Container para estatísticas em tempo real
                    with simulation_container:
                        col_sim1, col_sim2 = st.columns([2, 1])
                        
                        with col_sim2:
                            st.subheader("📊 Stats em Tempo Real")
                            area_metric = st.empty()
                            volume_metric = st.empty()
                            depth_metric = st.empty()
                            ciclo_metric = st.empty()
                        
                        with col_sim1:
                            chart_placeholder = st.empty()
                    
                    # Lista para dados da simulação
                    simulation_data = []
                    
                    # Loop principal da simulação
                    for frame in range(total_ciclos):
                        # Atualizar progresso principal
                        progress_percent = 30 + int((frame / total_ciclos) * 65)
                        main_progress.progress(progress_percent, 
                                            text=f"🌊 Simulando ciclo {frame+1}/{total_ciclos} ({(frame/total_ciclos)*100:.1f}%)")
                        
                        # Adicionar água
                        if modo_agua == 'CHUVA_UNIFORME':
                            water_level += rain_amount
                        elif modo_agua == 'FONTE_NO_RIO':
                            water_level[mascara_fontes_bool] += rain_amount
                        
                        # Física da água (baseada na configuração)
                        if fisica_agua == 'AVANÇADA' or frame % 2 == 0:  # Física avançada ou processo a cada 2 frames
                            flow_level = np.copy(water_level)
                            for y in range(1, dem_data.shape[0] - 1):
                                for x in range(1, dem_data.shape[1] - 1):
                                    if flow_level[y, x] <= 0: 
                                        continue
                                    
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
                                        flow_rate = 0.15 if fisica_agua == 'AVANÇADA' else 0.1
                                        total_flow = min(flow_level[y, x], total_diff * flow_rate)
                                        water_level[y, x] -= total_flow
                                        
                                        for neighbor in lower_neighbors:
                                            flow_share = (neighbor['diff'] / total_diff) * total_flow
                                            water_level[neighbor['pos']] += flow_share
                        
                        # Calcular estatísticas
                        stats = calculate_simulation_stats(water_level, dem_data, rain_amount, frame)
                        simulation_data.append(stats)
                        
                        # Atualizar visualização a cada 5 frames ou se preview em tempo real está ativo
                        if preview_tempo_real and (frame % 5 == 0 or frame == total_ciclos - 1):
                            # Atualizar métricas
                            area_metric.metric("🌊 Área Inundada", f"{stats['flooded_area_percent']:.1f}%")
                            volume_metric.metric("💧 Volume Total", f"{stats['total_water_volume']:.2f} m³")
                            depth_metric.metric("📏 Profundidade Máx", f"{stats['max_depth']:.2f} m")
                            ciclo_metric.metric("⏱️ Ciclo Atual", f"{frame+1}/{total_ciclos}")
                            
                            # Atualizar gráfico
                            if len(simulation_data) > 1:
                                df_temp = pd.DataFrame(simulation_data)
                                fig_temp = px.line(df_temp, x='frame', y='flooded_area_percent', 
                                                  title="Evolução da Inundação",
                                                  labels={'frame': 'Ciclos', 'flooded_area_percent': 'Área Inundada (%)'})
                                chart_placeholder.plotly_chart(fig_temp, width="stretch", config={'displayModeBar': False})
                        
                        # Atualizar imagem da água na animação
                        masked_water = ma.masked_where(water_level <= 0, water_level)
                        water_im.set_data(masked_water)
                        water_im.set_alpha(0.9)
                        ax.set_title(f"Simulação de Inundação (Progresso: {(frame+1)/total_ciclos*100:.1f}%)", 
                                   fontsize=14, fontweight='bold')
                    
                    main_progress.progress(95, text="🎬 Gerando animação final...")
                    
                    # Salvar animação
                    def update_frame(frame_num):
                        # Recalcular frame específico para animação suave
                        frame_water = np.zeros_like(dem_data, dtype=float)
                        
                        for f in range(frame_num + 1):
                            if modo_agua == 'CHUVA_UNIFORME':
                                frame_water += rain_amount
                            elif modo_agua == 'FONTE_NO_RIO':
                                frame_water[mascara_fontes_bool] += rain_amount
                        
                        masked_water = ma.masked_where(frame_water <= 0, frame_water)
                        water_im.set_data(masked_water)
                        water_im.set_alpha(0.9)
                        return [water_im]
                    
                    # Criar animação
                    anim = FuncAnimation(fig, update_frame, frames=total_ciclos, blit=True, 
                                       interval=intervalo_ms, repeat=True)
                    
                    # Salvar GIF
                    temp_gif_path = os.path.join(tempfile.gettempdir(), "simulation_advanced.gif")
                    anim.save(temp_gif_path, writer='pillow' if formato_export == 'GIF' else 'ffmpeg')
                    
                    # Carregar para BytesIO
                    gif_buffer = BytesIO()
                    with open(temp_gif_path, 'rb') as f:
                        gif_buffer.write(f.read())
                    gif_buffer.seek(0)
                    
                    if os.path.exists(temp_gif_path):
                        os.remove(temp_gif_path)
                    
                    # Salvar resultados
                    st.session_state.simulation_gif = gif_buffer
                    st.session_state.simulation_data = simulation_data
                    st.session_state.simulation_stats = simulation_data[-1] if simulation_data else {}
                    
                    plt.close(fig)
                    
                    main_progress.progress(100, text="✅ Simulação concluída com sucesso!")
                    time.sleep(1)
                    progress_main.empty()
                    
                    # Mostrar resultado final
                    st.success(f"🎉 **Simulação concluída!** {total_ciclos} ciclos processados")
                    
                    col_result1, col_result2 = st.columns([2, 1])
                    with col_result1:
                        st.image(gif_buffer, caption=f"🎬 Animação da Simulação ({formato_export})")
                    
                    with col_result2:
                        st.subheader("📊 Resultados Finais")
                        final_stats = simulation_data[-1]
                        st.metric("🌊 Área Final Inundada", f"{final_stats['flooded_area_percent']:.1f}%")
                        st.metric("💧 Volume Total de Água", f"{final_stats['total_water_volume']:.2f} m³")
                        st.metric("📏 Profundidade Máxima", f"{final_stats['max_depth']:.2f} m")
                        
                        if salvar_dados:
                            # Salvar dados CSV
                            df_results = pd.DataFrame(simulation_data)
                            csv_buffer = BytesIO()
                            df_results.to_csv(csv_buffer, index=False)
                            st.download_button(
                                "💾 Download Dados CSV", 
                                csv_buffer.getvalue(),
                                f"simulacao_dados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )

            except Exception as e:
                st.error(f"❌ Erro na simulação: {e}")
                main_progress.progress(0, text="❌ Simulação falhou")
            finally:
                if temp_dir_sim and os.path.exists(temp_dir_sim):
                    shutil.rmtree(temp_dir_sim)

# ===================================================================
# PAINEL DE INFORMAÇÕES LATERAL
# ===================================================================
with col_info:
    st.subheader("📋 Informações")
    
    # Informações dos arquivos
    if st.session_state.file_info:
        with st.expander("📊 Análise de Arquivos", expanded=True):
            
            if 'dem' in st.session_state.file_info:
                dem_info = st.session_state.file_info['dem']
                st.metric("🏔️ Elevação Mín/Máx", f"{dem_info.get('min_elevation', 0):.1f}m / {dem_info.get('max_elevation', 0):.1f}m")
                st.metric("📐 Resolução", f"{dem_info.get('resolution_x', 0):.1f}m")
                st.metric("📍 CRS", dem_info.get('crs', 'N/A')[:20] + "...")
                st.metric("🗺️ Área", f"{dem_info.get('area_km2', 0):.2f} km²")
            
            if 'shapefile' in st.session_state.file_info:
                shp_info = st.session_state.file_info['shapefile']
                st.metric("📁 Arquivos", shp_info.get('total_files', 0))
                st.metric("💾 Tamanho", f"{shp_info.get('total_size_mb', 0):.1f} MB")

    # Estatísticas da simulação
    if st.session_state.simulation_stats:
        with st.expander("📈 Estatísticas da Simulação", expanded=True):
            stats = st.session_state.simulation_stats
            st.metric("🌊 Área Inundada", f"{stats.get('flooded_area_percent', 0):.1f}%")
            st.metric("💧 Volume Total", f"{stats.get('total_water_volume', 0):.2f} m³")
            st.metric("📏 Profundidade Máx", f"{stats.get('max_depth', 0):.2f} m")
            
            # Gráfico de evolução
            if st.session_state.simulation_data:
                df = pd.DataFrame(st.session_state.simulation_data)
                fig = px.line(df, x='frame', y='flooded_area_percent', 
                             title="Evolução da Inundação",
                             labels={'frame': 'Tempo (ciclos)', 'flooded_area_percent': 'Área Inundada (%)'})
                st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})

    # Status do sistema
    with st.expander("⚙️ Status do Sistema"):
        st.info("🟢 Sistema Operacional")
        if uploaded_dem_file and uploaded_shp_files:
            st.success("📁 Arquivos Carregados")
        else:
            st.warning("📁 Aguardando Arquivos")

# ===================================================================
# RODAPÉ
# ===================================================================
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("🌊 **Simulador de Inundação Avançado** • Feito com Streamlit")
