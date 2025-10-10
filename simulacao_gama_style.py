import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import contextily as ctx
import tempfile
import os
import shutil
from typing import Tuple, Optional

# --- Modelo de Simulação Vetorizado com NumPy ---
class GamaFloodModelNumpy:
    """Motor de simulação de inundação vetorizado com NumPy."""
    def __init__(self, dem_data: np.ndarray, sources_mask: np.ndarray, diffusion_rate: float, flood_threshold: float, cell_size_meters: float):
        self.height, self.width = dem_data.shape
        self.diffusion_rate = diffusion_rate
        self.flood_threshold = flood_threshold
        self.cell_area = cell_size_meters * cell_size_meters
        self.altitude = dem_data.astype(np.float32)
        self.is_source = sources_mask.astype(bool)
        self.water_height = np.zeros_like(self.altitude, dtype=np.float32)
        self.active_cells_coords = set(zip(*np.where(self.is_source)))
        self.simulation_time_minutes = 0
        self.overflow_time_minutes: Optional[int] = None
        self.history = []

    def add_water(self, rain_mm: float):
        water_to_add_meters = rain_mm / 1000.0
        if water_to_add_meters <= 0: return
        self.water_height[self.is_source] += water_to_add_meters
        self.active_cells_coords.update(zip(*np.where(self.is_source)))

    def run_flow_step(self):
        if not self.active_cells_coords: return
        newly_active_cells, cells_to_deactivate = set(), set()
        water_height_prev = self.water_height.copy()
        for y, x in list(self.active_cells_coords):
            current_cell_water = water_height_prev[y, x]
            if current_cell_water <= 0.001:
                cells_to_deactivate.add((y, x)); continue
            current_total_elevation = self.altitude[y, x] + current_cell_water
            neighbors_coords = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width: neighbors_coords.append((ny, nx))
            if not neighbors_coords: continue
            n_y, n_x = zip(*neighbors_coords)
            n_total_elevation = self.altitude[n_y, n_x] + water_height_prev[n_y, n_x]
            lower_neighbors_mask = n_total_elevation < current_total_elevation
            if not np.any(lower_neighbors_mask):
                cells_to_deactivate.add((y, x)); continue
            lower_n_coords = np.array(neighbors_coords)[lower_neighbors_mask]
            lower_n_total_elevation = n_total_elevation[lower_neighbors_mask]
            total_diff = np.sum(current_total_elevation - lower_n_total_elevation)
            if total_diff <= 0: continue
            water_to_distribute = current_cell_water * self.diffusion_rate
            for n_idx, (ny, nx) in enumerate(lower_n_coords):
                diff = current_total_elevation - lower_n_total_elevation[n_idx]
                fraction = diff / total_diff
                water_moved = min(water_to_distribute * fraction, diff / 2.0)
                if water_moved > 0:
                    self.water_height[y, x] -= water_moved
                    self.water_height[ny, nx] += water_moved
                    newly_active_cells.add((ny, nx))
        self.active_cells_coords.difference_update(cells_to_deactivate)
        self.active_cells_coords.update(newly_active_cells)

    def update_stats(self, time_step_minutes: int):
        self.simulation_time_minutes += time_step_minutes
        inundated_mask = self.water_height > self.flood_threshold
        if self.overflow_time_minutes is None and np.any(inundated_mask & ~self.is_source):
            self.overflow_time_minutes = self.simulation_time_minutes
        self.history.append({
            "time_minutes": self.simulation_time_minutes,
            "flooded_percent": np.sum(inundated_mask) / inundated_mask.size * 100,
            "active_cells": len(self.active_cells_coords),
            "max_depth": np.max(self.water_height) if self.water_height.size > 0 else 0,
            "total_water_volume_m3": np.sum(self.water_height * self.cell_area)
        })

# --- Funções Auxiliares ---
def process_uploaded_files(dem_file, vector_files, ortho_file) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    if not dem_file or not vector_files: return None, None, None, None
    temp_dir = tempfile.mkdtemp()
    dem_path = os.path.join(temp_dir, dem_file.name)
    with open(dem_path, "wb") as f: f.write(dem_file.getbuffer())
    
    vector_path = None
    for f in vector_files:
        path = os.path.join(temp_dir, f.name)
        with open(path, "wb") as out: out.write(f.getbuffer())
        if f.name.lower().endswith(('.gpkg', '.shp')):
            vector_path = path

    ortho_path = None
    if ortho_file:
        ortho_path = os.path.join(temp_dir, ortho_file.name)
        with open(ortho_path, "wb") as f: f.write(ortho_file.getbuffer())

    return dem_path, vector_path, ortho_path, temp_dir

def setup_geodata(dem_path, vector_path, grid_reduction_factor):
    with rio.open(dem_path) as dem_src:
        dem_crs = dem_src.crs
        original_transform = dem_src.transform
        new_height, new_width = dem_src.height // grid_reduction_factor, dem_src.width // grid_reduction_factor
        dem_data = dem_src.read(1, out_shape=(new_height, new_width), resampling=Resampling.bilinear)
        rescaled_transform = from_origin(original_transform.xoff, original_transform.yoff, original_transform.a * grid_reduction_factor, original_transform.e * grid_reduction_factor)
        if dem_crs.is_geographic:
            st.warning(f"O CRS do DEM é geográfico ({dem_crs.to_string()}). Considere usar um DEM em CRS projetado (ex: UTM).")
            cell_size_meters = abs(rescaled_transform.a) * 111320
        else:
            cell_size_meters = abs(rescaled_transform.a)
            st.info(f"CRS do DEM: {dem_crs.to_string()}. Tamanho da célula: {cell_size_meters:.2f} metros.")
    sources_gdf = gpd.read_file(vector_path).to_crs(dem_crs)
    sources_mask = rasterize(sources_gdf.geometry, out_shape=dem_data.shape, transform=rescaled_transform, fill=0, all_touched=True, dtype=np.uint8)
    return dem_data, sources_mask, rescaled_transform, dem_crs, cell_size_meters

def setup_visualization(dem_data, rescaled_transform, dem_crs, basemap_choice, ortho_path=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    bounds = rio.transform.array_bounds(dem_data.shape[0], dem_data.shape[1], rescaled_transform)
    
    valid_dem_data = dem_data[~np.isnan(dem_data) & (dem_data > 0)]
    vmin, vmax = np.percentile(valid_dem_data, (5, 95)) if valid_dem_data.size > 0 else (0, 1)
    
    title_color = 'black'
    
    # --- NOVA LÓGICA PARA ESCOLHER O FUNDO ---
    if basemap_choice == "Meu Ortomozaico" and ortho_path:
        try:
            with rio.open(ortho_path) as ortho_src:
                # Lê os 3 primeiros canais (RGB)
                img_data = ortho_src.read((1, 2, 3))
                # Transpõe de (canal, linha, coluna) para (linha, coluna, canal) para o imshow
                img_rgb = np.transpose(img_data, (1, 2, 0))
                ax.imshow(img_rgb, extent=ortho_src.bounds, zorder=0)
            dem_alpha = 0.5 # DEM fica mais transparente sobre o ortomosaico
        except Exception as e:
            st.error(f"Erro ao ler o arquivo de ortomosaico: {e}")
            basemap_choice = "Mapa Online" # Volta para o online se der erro
    
    if basemap_choice == "Mapa Online":
        dem_alpha = 1.0 # DEM opaco por padrão
        try:
            ctx.add_basemap(ax, crs=dem_crs, source=ctx.providers.OpenStreetMap.Mapnik, zorder=0)
            dem_alpha = 0.6 # Fica transparente se o mapa online carregar
        except Exception as e:
            st.warning(f"Aviso: Não foi possível carregar o mapa base online. Exibindo apenas o DEM. Erro: {e}")
            ax.set_facecolor('lightgray')

    dem_plot = ax.imshow(dem_data, extent=bounds, cmap='terrain', alpha=dem_alpha, vmin=vmin, vmax=vmax, zorder=1)
    water_layer = ax.imshow(np.zeros_like(dem_data), extent=bounds, cmap='Blues', alpha=0.0, vmin=0, vmax=1, zorder=2)
    rain_particles, = ax.plot([], [], '.', color='royalblue', markersize=1.0, alpha=0.7, zorder=10)
    title = ax.set_title("Simulação de Inundação | Tempo: 0h 0m", color=title_color, fontweight='bold', zorder=11)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    return fig, ax, water_layer, rain_particles, title, bounds

# --- Interface Streamlit ---
def main():
    st.set_page_config(page_title="Simulador de Inundação", layout="wide", page_icon="🌊")
    st.title("🌊 Simulador de Inundação")

    with st.sidebar:
        st.header("📂 Arquivos de Entrada")
        dem_file = st.file_uploader("1. DEM (.tif)", type=["tif", "tiff"])
        vector_files = st.file_uploader("2. Fontes de Água (.gpkg, .shp)", type=["gpkg", "shp", "shx", "dbf", "prj"], accept_multiple_files=True)
        
        st.header("🎨 Visualização")
        # --- NOVAS OPÇÕES DE MAPA BASE ---
        basemap_choice = st.radio(
            "Escolha o mapa de fundo:",
            ("Mapa Online", "Meu Ortomozaico"),
            help="Use um mapa padrão da internet ou envie sua própria imagem de drone georreferenciada."
        )
        ortho_file = None
        if basemap_choice == "Meu Ortomozaico":
            ortho_file = st.file_uploader("Envie seu Ortomozaico (.tif)", type=["tif", "tiff"])
            st.info("Seu ortomosaico deve ter o mesmo sistema de coordenadas (CRS) do DEM.", icon="ℹ️")

        st.header("⚙️ Parâmetros da Simulação")
        time_step_minutes = st.slider("Duração de cada ciclo (min)", 1, 60, 10)
        rain_mm_per_cycle = st.slider("Chuva por ciclo (mm)", 0.0, 50.0, 5.0, 0.5)
        diffusion_rate = st.slider("Taxa de Difusão da Água", 0.1, 1.0, 0.5, 0.05)
        flood_threshold = st.slider("Limiar de Inundação (m)", 0.01, 1.0, 0.1, 0.01)
        
        grid_reduction_factor = st.select_slider("Resolução da Grade", options=[1, 2, 4, 8, 16], value=4)
        animation_format = st.selectbox("Formato da Animação", ["GIF", "MP4"])
        total_cycles = st.number_input("Total de Ciclos", 10, 500, 100)
        animation_interval = st.slider("Intervalo da Animação (ms)", 100, 2000, 500, 50, help="Valores maiores = animação mais lenta.")
    
    st.markdown("---")
    stats_cols = st.columns(4)
    overflow_stat_placeholder, time_stat_placeholder, flooded_stat_placeholder, water_volume_stat_placeholder = [c.empty() for c in stats_cols]
    st.markdown("---")
    animation_placeholder = st.empty()
    
    overflow_stat_placeholder.metric("Tempo para Transbordar", "N/A")
    time_stat_placeholder.metric("Tempo de Simulação", "0h 0m")
    flooded_stat_placeholder.metric("Área Inundada", "0.00%")
    water_volume_stat_placeholder.metric("Volume de Água", "0.00 m³")

    if not dem_file or not vector_files or (basemap_choice == "Meu Ortomozaico" and not ortho_file):
        st.info("Por favor, envie todos os arquivos necessários para iniciar.")
        return

    if st.button("🚀 Iniciar Simulação", type="primary"):
        dem_path, vector_path, ortho_path, temp_dir = process_uploaded_files(dem_file, vector_files, ortho_file)
        if not vector_path:
            st.error("Falha ao processar arquivos vetoriais."); return

        progress_bar = st.progress(0, "Inicializando...")
        try:
            dem_data, sources_mask, transform, crs, cell_size = setup_geodata(dem_path, vector_path, grid_reduction_factor)
            progress_bar.progress(15, "Inicializando modelo...")
            model = GamaFloodModelNumpy(dem_data, sources_mask, diffusion_rate, flood_threshold, cell_size)
            progress_bar.progress(25, "Preparando visualização...")
            fig, ax, water_layer, rain_particles, title, plot_bounds = setup_visualization(dem_data, transform, crs, basemap_choice, ortho_path)

            def update(frame):
                model.add_water(rain_mm_per_cycle)
                model.run_flow_step()
                model.update_stats(time_step_minutes)
                water_matrix = model.water_height
                masked_water = np.ma.masked_where(water_matrix < 0.01, water_matrix)
                water_layer.set_data(masked_water)
                water_layer.set_alpha(0.5)
                num_particles = int(rain_mm_per_cycle * 150)
                x_min, y_min, x_max, y_max = plot_bounds
                rain_x, rain_y = np.random.uniform(x_min, x_max, num_particles), np.random.uniform(y_min, y_max, num_particles)
                rain_particles.set_data(rain_x, rain_y)
                hours, rem_minutes = divmod(model.simulation_time_minutes, 60)
                title.set_text(f"Simulação de Inundação | Tempo: {hours}h {rem_minutes}m")
                latest_stats = model.history[-1]
                if model.overflow_time_minutes is not None:
                    h_o, m_o = divmod(model.overflow_time_minutes, 60)
                    overflow_stat_placeholder.metric("Tempo para Transbordar", f"{h_o}h {m_o}m")
                time_stat_placeholder.metric("Tempo de Simulação", f"{hours}h {rem_minutes}m")
                flooded_stat_placeholder.metric("Área Inundada", f"{latest_stats['flooded_percent']:.2f}%")
                water_volume_stat_placeholder.metric("Volume de Água", f"{latest_stats['total_water_volume_m3']:.2f} m³")
                progress = 25 + int(70 * ((frame + 1) / total_cycles))
                progress_bar.progress(progress, f"Simulando ciclo {frame+1}/{total_cycles}")
                return [water_layer, rain_particles, title]

            anim = FuncAnimation(fig, update, frames=total_cycles, blit=True, interval=animation_interval)
            
            progress_bar.progress(95, "Gerando animação (pode demorar)...")
            ext = animation_format.lower()
            temp_anim_path = os.path.join(temp_dir, f"simulation.{ext}")
            anim.save(temp_anim_path, writer='pillow' if ext == 'gif' else 'ffmpeg', dpi=150)
            progress_bar.progress(100, "Simulação concluída!")
            
            with open(temp_anim_path, "rb") as f:
                if ext == 'gif': animation_placeholder.image(f.read())
                else: animation_placeholder.video(f.read())

            st.markdown("---")
            st.subheader("⬇️ Download dos Resultados")
            col1, col2 = st.columns(2)
            with open(temp_anim_path, "rb") as f:
                col1.download_button(f"📥 Baixar Animação ({ext.upper()})", f.read(), f"simulacao.{ext}")
            df_history = pd.DataFrame(model.history)
            csv = df_history.to_csv(index=False).encode('utf-8')
            col2.download_button("📥 Baixar Dados (CSV)", csv, "dados_simulacao.csv", "text/csv")
        except Exception as e:
            st.error(f"Ocorreu um erro durante a simulação: {e}")
            st.exception(e)
        finally:
            if 'temp_dir' in locals() and os.path.exists(temp_dir): shutil.rmtree(temp_dir)
            plt.close('all')

if __name__ == "__main__":
    main()