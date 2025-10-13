import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.transform import from_origin, array_bounds
from rasterio.features import rasterize
import contextily as ctx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LightSource
import os
import contextlib
import tempfile
import shutil
import io
from typing import Optional, Tuple
import time
import subprocess
import sys
import importlib as importlib_mod

# Telemetria (Opik) - import opcional
try:
    opik = importlib_mod.import_module("opik")  # type: ignore
except Exception:
    opik = None  # type: ignore

def maybe_track(enabled: bool, **kwargs):
    """Retorna um decorador de no-op quando telemetria desativada ou opik ausente."""
    def _noop_decorator(func):
        return func
    if enabled and opik is not None:
        try:
            return opik.track(**kwargs)
        except (AttributeError, TypeError, ValueError):
            return _noop_decorator
    return _noop_decorator

# Import das funções de design (reexportadas em shapes/__init__.py)
from shapes import apply_custom_styles, create_header
from lisflood_integration import (
    prepare_input_folder as lf_prepare_input_folder,
    copy_rasters as lf_copy_rasters,
    ensure_mask_pcraster as lf_ensure_mask_pcraster,
    generate_ldd_from_dem as lf_generate_ldd_from_dem,
    run_lisflood_docker as lf_run_lisflood,
    list_outputs as lf_list_outputs,
    convert_stl_to_dsm as lf_convert_stl_to_dsm,
    patch_settings_xml_with_defaults as lf_patch_settings_xml_with_defaults,
)

# ========= SIMULADOR NUMÉRICO =========
class GamaFloodModelNumpy:
    def __init__(self, dem_data: np.ndarray, sources_mask: np.ndarray, diffusion_rate: float, flood_threshold: float, cell_size_meters: float, river_mask: Optional[np.ndarray] = None):
        self.height, self.width = dem_data.shape
        self.diffusion_rate = diffusion_rate
        self.flood_threshold = flood_threshold
        self.cell_area = cell_size_meters * cell_size_meters
        self.altitude = dem_data.astype(np.float32)
        self.is_source = sources_mask.astype(bool)
        self.river_mask = river_mask.astype(bool) if river_mask is not None else None
        self.water_height = np.zeros_like(self.altitude, dtype=np.float32)
        self.active_cells_coords = set(zip(*np.where(self.is_source)))
        self.simulation_time_minutes = 0
        self.overflow_time_minutes = None
        self.history = []
        self.uniform_rain = False

    def add_water(self, rain_mm: float):
        water_to_add_meters = rain_mm / 1000.0
        if water_to_add_meters <= 0:
            return
        if self.uniform_rain or not np.any(self.is_source):
            self.water_height += water_to_add_meters
            ys, xs = np.where(self.water_height > 0)
            self.active_cells_coords.update(zip(ys, xs))
        else:
            self.water_height[self.is_source] += water_to_add_meters
            self.active_cells_coords.update(zip(*np.where(self.is_source)))

    def run_flow_step(self):
        if not self.active_cells_coords:
            return
        newly_active = set()
        to_deactivate = set()
        wh_prev = self.water_height.copy()
        for y, x in list(self.active_cells_coords):
            w = wh_prev[y, x]
            if w <= 0.001:
                to_deactivate.add((y, x))
                continue
            current_total = self.altitude[y, x] + w
            neigh = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        neigh.append((ny, nx))
            if not neigh:
                continue
            n_y, n_x = zip(*neigh)
            n_total = self.altitude[n_y, n_x] + wh_prev[n_y, n_x]
            mask_lower = n_total < current_total
            if not np.any(mask_lower):
                to_deactivate.add((y, x))
                continue
            lower_coords = np.array(neigh)[mask_lower]
            lower_total = n_total[mask_lower]
            total_diff = np.sum(current_total - lower_total)
            if total_diff <= 0:
                continue
            to_distribute = w * self.diffusion_rate
            for i, (ny, nx) in enumerate(lower_coords):
                diff = current_total - lower_total[i]
                frac = diff / total_diff
                moved = min(to_distribute * frac, diff / 2.0)
                if moved > 0:
                    self.water_height[y, x] -= moved
                    self.water_height[ny, nx] += moved
                    newly_active.add((ny, nx))
        self.active_cells_coords.difference_update(to_deactivate)
        self.active_cells_coords.update(newly_active)

    def update_stats(self, step_min: int):
        self.simulation_time_minutes += step_min
        inundated = self.water_height > self.flood_threshold
        baseline_mask = self.river_mask if self.river_mask is not None else self.is_source
        flooded_outside = inundated & ~baseline_mask
        if self.overflow_time_minutes is None and np.any(flooded_outside):
            self.overflow_time_minutes = self.simulation_time_minutes
        flooded_percent = float(np.sum(inundated)) / inundated.size * 100.0
        total_water = float(np.sum(self.water_height * self.cell_area))
        max_depth = float(np.max(self.water_height)) if self.water_height.size else 0.0
        self.history.append({
            "time_minutes": self.simulation_time_minutes,
            "flooded_percent": flooded_percent,
            "active_cells": len(self.active_cells_coords),
            "max_depth": max_depth,
            "total_water_volume_m3": total_water
        })

def _process_uploaded_files(dem_file, vector_files) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not dem_file or not vector_files:
        return None, None, None
    tmp = tempfile.mkdtemp(prefix="sim_numpy_")
    dem_path = os.path.join(tmp, dem_file.name)
    with open(dem_path, "wb") as f:
        f.write(dem_file.getbuffer())
    vector_path = None
    for f in vector_files:
        p = os.path.join(tmp, f.name)
        with open(p, "wb") as out:
            out.write(f.getbuffer())
        if f.name.lower().endswith(".gpkg"):
            vector_path = p
        elif f.name.lower().endswith(".shp") and not vector_path:
            vector_path = p
    return dem_path, vector_path, tmp

def _setup_geodata(dem_path, vector_path: Optional[str], grid_reduction_factor, river_vector_path: Optional[str] = None):
    with rio.open(dem_path) as dem_src:
        dem_crs = dem_src.crs
        orig_transform = dem_src.transform
        new_h = max(1, dem_src.height // grid_reduction_factor)
        new_w = max(1, dem_src.width // grid_reduction_factor)
        dem_data = dem_src.read(1, out_shape=(new_h, new_w), resampling=Resampling.bilinear)
        transform = from_origin(
            orig_transform.xoff,
            orig_transform.yoff,
            orig_transform.a * grid_reduction_factor,
            abs(orig_transform.e) * grid_reduction_factor,
        )
        # Tamanho da célula
        if dem_crs and dem_crs.is_geographic:
            st.warning(f"CRS geográfico detectado ({dem_crs.to_string()}). Converta para UTM para resultados precisos.")
            # Cálculo mais realista considerando latitude do centro do DEM
            bounds = dem_src.bounds
            center_lat = (bounds.top + bounds.bottom) / 2.0
            meters_per_degree_lat = 111_320
            meters_per_degree_lon = 111_320 * np.cos(np.radians(center_lat))
            cell_width_m = abs(dem_src.res[0]) * meters_per_degree_lon * grid_reduction_factor
            cell_height_m = abs(dem_src.res[1]) * meters_per_degree_lat * grid_reduction_factor
            cell_size_m = (cell_width_m + cell_height_m) / 2.0
        else:
            # CRS projetado (e.g., UTM): resolução já em metros
            cell_size_m = abs(dem_src.res[0]) * grid_reduction_factor
        st.info(f"CRS do DEM: {dem_crs}. Tamanho de célula efetivo: {cell_size_m:.3f} m")
    if vector_path is None:
        st.info("Sem vetor de fontes enviado. Usando a área inteira como fonte de chuva.")
        sources_mask = np.ones(dem_data.shape, dtype=np.uint8)
    else:
        gdf = gpd.read_file(vector_path).to_crs(dem_crs)
        sources_mask = rasterize(
            gdf.geometry,
            out_shape=dem_data.shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        )
        if not np.any(sources_mask > 0):
            st.warning("Nenhum pixel de 'fonte de água' foi rasterizado sobre o DEM. Aplicando chuva uniforme em toda a área.")
            sources_mask = np.ones_like(sources_mask, dtype=np.uint8)
    river_mask = None
    if river_vector_path:
        try:
            gdf_river = gpd.read_file(river_vector_path).to_crs(dem_crs)
            river_mask = rasterize(
                gdf_river.geometry,
                out_shape=dem_data.shape,
                transform=transform,
                fill=0,
                all_touched=True,
                dtype=np.uint8,
            )
        except (OSError, ValueError, RuntimeError) as e:
            st.warning(f"Falha ao ler/rasterizar rio: {e}")
            river_mask = None
    return dem_data, sources_mask, transform, dem_crs, cell_size_m, river_mask

def _setup_visualization(dem_data, transform, dem_crs, background_rgb=None, apply_hillshade=False, hs_intensity=0.6):
    fig, ax = plt.subplots(figsize=(12, 9))
    bounds = array_bounds(dem_data.shape[0], dem_data.shape[1], transform)
    if background_rgb is not None:
        img = background_rgb
        if apply_hillshade:
            dem_norm = dem_data.astype(float)
            dem_norm = (dem_norm - np.nanmin(dem_norm)) / (np.nanmax(dem_norm) - np.nanmin(dem_norm) + 1e-9)
            ls = LightSource(azdeg=315, altdeg=45)
            try:
                shaded = ls.shade_rgb(img, dem_norm, fraction=hs_intensity)
                ax.imshow(shaded, extent=bounds)
            except (ValueError, RuntimeError, TypeError):
                ax.imshow(img, extent=bounds)
        else:
            ax.imshow(img, extent=bounds)
    else:
        vmin, vmax = (np.percentile(dem_data[dem_data > 0], (5, 95)) if (dem_data > 0).any() else (0, 1))
        ax.imshow(dem_data, extent=bounds, cmap="terrain", alpha=0.85, vmin=vmin, vmax=vmax)
        with contextlib.suppress(OSError, ValueError, RuntimeError):
            ctx.add_basemap(ax, crs=dem_crs, source='CartoDB.Positron')
    water_layer = ax.imshow(np.zeros_like(dem_data), extent=bounds, cmap="Blues", alpha=0.0, vmin=0, vmax=1)
    x_min, y_min, x_max, y_max = bounds
    rain_particles, = ax.plot([], [], ".", color="royalblue", markersize=1.0, alpha=0.7, zorder=10)
    title = ax.set_title("Simulação de Inundação | Tempo: 0h 0m", color="black", fontweight="bold")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    return fig, ax, water_layer, rain_particles, title, (x_min, y_min, x_max, y_max)

def _prepare_background(dom_path: str, target_shape, target_crs) -> Optional[np.ndarray]:
    try:
        with rio.open(dom_path) as src:
            if src.crs and target_crs and (src.crs != target_crs):
                st.warning(f"CRS do DOM ({src.crs}) difere do DEM ({target_crs}). A visualização será reamostrada por dimensão apenas.")
            H, W = target_shape
            count = min(src.count, 3)
            bands = []
            for b in range(1, count+1):
                band = src.read(b, out_shape=(H, W), resampling=Resampling.bilinear)
                bands.append(band)
            if len(bands) == 1:
                rgb = np.stack([bands[0]]*3, axis=-1)
            else:
                rgb = np.stack(bands, axis=-1)
            if rgb.dtype != np.uint8:
                p2, p98 = np.percentile(rgb[np.isfinite(rgb)], (2, 98)) if np.isfinite(rgb).any() else (0, 1)
                p2 = max(p2, 0)
                denom = (p98 - p2) if (p98 - p2) != 0 else 1
                rgb = np.clip((rgb - p2) / denom, 0, 1)
            return rgb
    except (OSError, ValueError, RuntimeError) as e:
        st.warning(f"Falha ao preparar fundo (DOM): {e}")
    return None

def _visualize_geotiff(fp: str):
    with rio.open(fp) as src:
        data = src.read(1)
        bounds = array_bounds(src.height, src.width, src.transform)
        crs = src.crs
    fig, ax = plt.subplots(figsize=(10, 8))
    vmin, vmax = np.nanpercentile(data, (5, 95)) if np.isfinite(data).any() else (0, 1)
    ax.imshow(data, extent=bounds, cmap="Blues", alpha=0.8, vmin=vmin, vmax=vmax)
    with contextlib.suppress(OSError, ValueError, RuntimeError):
        ctx.add_basemap(ax, crs=crs, source='CartoDB.Positron')
    ax.set_title(os.path.basename(fp))
    ax.set_axis_off()
    st.pyplot(fig, clear_figure=True)

def _check_docker_available(timeout: float = 6.0) -> Tuple[bool, str]:
    """Verifica se o Docker está disponível no PATH e retorna (ok, detalhe)."""
    try:
        res = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=timeout, check=False)
        if res.returncode == 0:
            out = (res.stdout or res.stderr or "").strip()
            return True, out
        return False, (res.stderr or res.stdout or "").strip()
    except FileNotFoundError:
        return False, "binário 'docker' não encontrado no PATH"
    except subprocess.TimeoutExpired:
        return False, "tempo esgotado ao consultar 'docker --version'"
    except (OSError, RuntimeError) as e:
        return False, f"erro ao consultar docker: {e}"

def _check_trimesh_installed() -> Tuple[bool, str]:
    try:
        # Primeiro tenta obter versão via metadata
        try:
            from importlib import metadata as importlib_metadata  # py3.8+
            ver = importlib_metadata.version("trimesh")  # type: ignore
            return True, f"trimesh {ver}"
        except importlib_metadata.PackageNotFoundError:  # type: ignore[attr-defined]
            pass
        except (ValueError, RuntimeError):
            pass
        # Fallback: tenta import direto
        try:
            trimesh = importlib_mod.import_module("trimesh")  # type: ignore
            ver = getattr(trimesh, "__version__", "")
            label = f"trimesh {ver}".strip() if ver else "trimesh"
            return True, label
        except ImportError:
            return False, "não instalado"
        except (RuntimeError, AttributeError) as e:
            return False, f"erro ao checar: {e}"
    except (RuntimeError, AttributeError) as e:
        return False, f"erro ao checar: {e}"

def _install_trimesh() -> Tuple[int, str, str]:
    """Instala trimesh no Python atual. Retorna (code, stdout, stderr)."""
    try:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "trimesh"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return res.returncode, res.stdout, res.stderr
    except (OSError, RuntimeError) as e:
        return 1, "", str(e)

def create_lisflood_minimal_xml(output_path, replacements):
    """Cria um XML mínimo compatível com LISFLOOD - VERSÃO ROBUSTA"""
    import xml.etree.ElementTree as ET

    # Valores padrão COMPLETOS
    default_replacements = {
        "MaskMap": "/input/MASK.map",
        "Ldd": "/input/Ldd.map",
        "PathOut": "/input/output/",
        "StepStart": "01/01/2024 00:00",
        "StepEnd": "01/01/2024 01:00",
        "DtSec": "3600",
        "DtInit": "3600",
        "RepStep": "1",
        "CalendarConvention": "proleptic_gregorian",
    }

    # Atualiza com os valores fornecidos
    default_replacements.update(replacements or {})

    # Construção XML
    root = ET.Element("lisfloodSettings")
    lfuser = ET.SubElement(root, "lfuser")

    params = [
        ("MaskMap", default_replacements["MaskMap"]),
        ("Ldd", default_replacements["Ldd"]),
        ("PathOut", default_replacements["PathOut"]),
        ("StepStart", default_replacements["StepStart"]),
        ("StepEnd", default_replacements["StepEnd"]),
        ("DtSec", default_replacements["DtSec"]),
        ("DtInit", default_replacements["DtInit"]),
        # Sinônimos para compatibilidade
        ("timestep", default_replacements["DtSec"]),
        ("timestep_init", default_replacements["DtInit"]),
        ("CalendarType", default_replacements["CalendarConvention"]),
        ("RepStep", default_replacements["RepStep"]),
        ("CalendarConvention", default_replacements["CalendarConvention"]),
        ("simulateWaterBodies", "0"),
        ("simulateLakes", "0"),
        ("simulateReservoirs", "0"),
        ("simulateSnow", "0"),
        ("simulateGlaciers", "0"),
        ("simulateFrost", "0"),
        ("simulateInfiltration", "0"),
        ("simulatePercolation", "0"),
        ("simulateGroundwater", "0"),
        ("simulateCapillaryRise", "0"),
        ("simulateInterception", "0"),
        ("simulateEvapotranspiration", "0"),
        ("simulateWaterQuality", "0"),
        ("simulateSediment", "0"),
        ("simulateNutrients", "0"),
        ("RepMapSteps", "1"),
        ("RepStateFiles", "0"),
        ("RepDischarge", "0"),
        ("RepStateVars", "0"),
        ("InitialConditions", "0"),
        ("InitLisflood", "1"),
    ]
    for name, value in params:
        ET.SubElement(lfuser, "textvar", name=name, value=value)

    lfbinding = ET.SubElement(root, "lfbinding")
    # Essenciais no binding (incluir variações)
    ET.SubElement(lfbinding, "textvar", name="CalendarConvention", value=default_replacements["CalendarConvention"])
    ET.SubElement(lfbinding, "textvar", name="CalendarType", value=default_replacements["CalendarConvention"])  # alias
    # Algumas versões usam tag <text> em vez de <textvar>
    ET.SubElement(lfbinding, "text", name="CalendarConvention", value=default_replacements["CalendarConvention"])
    ET.SubElement(lfbinding, "text", name="CalendarType", value=default_replacements["CalendarConvention"])  # alias
    ET.SubElement(lfbinding, "map", name="MASK", file="MASK.map")
    ET.SubElement(lfbinding, "map", name="LDD", file="Ldd.map")

    # Formatar e gravar
    try:
        ET.indent(root, space="\t", level=0)  # py>=3.9
    except Exception:
        pass
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    # Verificação básica
    try:
        _ = ET.parse(output_path)
        print(f"✅ XML válido criado em: {output_path}")
    except ET.ParseError as e:
        print(f"❌ Erro no XML gerado: {e}")
        raise

# ========= FUNÇÕES AUXILIARES PARA PÓS-PROCESSAMENTO =========
def _create_evolution_plots(model):
    """Cria gráficos de evolução temporal"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
   
    # Gráfico 1: Área inundada e volume
    times = [h['time_minutes']/60 for h in model.history]
    areas = [h['flooded_percent'] for h in model.history]
    volumes = [h['total_water_volume_m3'] for h in model.history]
   
    ax1.plot(times, areas, 'b-', linewidth=2, label='Área Inundada (%)')
    ax1.set_xlabel('Tempo (horas)')
    ax1.set_ylabel('Área Inundada (%)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
   
    ax1_twin = ax1.twinx()
    ax1_twin.plot(times, volumes, 'r-', linewidth=2, label='Volume (m³)')
    ax1_twin.set_ylabel('Volume de Água (m³)', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
   
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.set_title('Evolução da Área Inundada e Volume')
   
    # Gráfico 2: Profundidade e células ativas
    depths = [h['max_depth'] for h in model.history]
    cells = [h['active_cells'] for h in model.history]
   
    ax2.plot(times, depths, 'g-', linewidth=2, label='Profundidade Máx (m)')
    ax2.set_xlabel('Tempo (horas)')
    ax2.set_ylabel('Profundidade Máxima (m)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.grid(True, alpha=0.3)
   
    ax2_twin = ax2.twinx()
    ax2_twin.plot(times, cells, 'orange', linewidth=2, label='Células Ativas')
    ax2_twin.set_ylabel('Células Ativas', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
   
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.set_title('Evolução da Profundidade e Células Ativas')
   
    plt.tight_layout()
    return fig

def _post_process_simulation(model, _tmp_dir, anim_path, anim_format, total_rain, cell_size, sources_mask):
    """Processa resultados e gera downloads"""
   
    st.markdown("---")
    st.subheader("📊 Resultados da Simulação")
   
    # 1. Estatísticas finais
    total_rain_m3 = float(np.sum((total_rain/1000.0) * (sources_mask>0) * (cell_size*cell_size)))
    final_time = model.simulation_time_minutes
    h, m = divmod(final_time, 60)
    transbordo = (f"{divmod(model.overflow_time_minutes,60)[0]}h {divmod(model.overflow_time_minutes,60)[1]}m"
                  if model.overflow_time_minutes else "N/A")
   
    # Tabela resumo
    resumo = pd.DataFrame([{
        "Chuva total (mm)": total_rain,
        "Volume de chuva (m³)": total_rain_m3,
        "Tempo simulado": f"{h}h {m}m",
        "Transbordamento": transbordo,
        "Área máxima inundada": f"{model.history[-1]['flooded_percent']:.2f}%"
    }])
    st.dataframe(resumo)
   
    # 2. Downloads
    st.subheader("📥 Downloads")
    col1, col2, col3 = st.columns(3)
   
    # CSV com dados
    with col1:
        df = pd.DataFrame(model.history)
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📊 Dados (CSV)",
            csv_data,
            "dados_simulacao.csv",
            "text/csv"
        )
   
    # Animação
    with col2:
            if anim_path and os.path.exists(anim_path):
                try:
                    with open(anim_path, "rb") as f:
                        anim_data = f.read()
                    ext = anim_format.lower()
                    st.download_button(
                        f"🎬 Animação ({ext.upper()})",
                        anim_data,
                        f"simulacao.{ext}",
                        f"video/{ext}" if ext == 'mp4' else "image/gif"
                    )
                except Exception as e:
                    st.error(f"Erro ao carregar animação: {e}")
            else:
                st.info("Ative 'Salvar animação' para download")
   
    # Gráficos
    with col3:
        if len(model.history) > 0:
            # Criar gráficos em memória
            fig = _create_evolution_plots(model)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
           
            st.download_button(
                "📈 Gráficos (PNG)",
                buf.getvalue(),
                "evolucao_simulacao.png",
                "image/png"
            )
   
    # 3. Gráficos de evolução
    if len(model.history) > 0:
        st.subheader("📈 Evolução Temporal")
        fig_display = _create_evolution_plots(model)
        st.pyplot(fig_display)

# ========= FUNÇÕES LISFLOOD (SIMULADAS) =========
# Removidas: usando implementações reais via logica_lisflood

def main():
    # Configuração da página (título exatamente como solicitado e sem emoji/prefixo)
    st.set_page_config(
        page_title="Simulador hibrido de inundações",
        page_icon=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logos", "logo.png"),
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Aplicar estilos customizados
    apply_custom_styles()

    # Cabeçalho com as duas logos e o título central sem prefixos
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOGO_MAIN = os.path.join(_BASE_DIR, "logos", "logo.png")
    LOGO_PLATFORM = os.path.join(_BASE_DIR, "logos", "logoPlataforma.png")

    with st.container():
        col_l, col_c, col_r = st.columns([1, 4, 1])
        with col_l:
            if os.path.exists(LOGO_MAIN):
                st.image(LOGO_MAIN, width=160)
        with col_c:
            st.markdown(
                "<h1 style='text-align:center; margin: 0;'>Simulador Híbrido de Inundações</h1>",
                unsafe_allow_html=True,
            )
        with col_r:
            if os.path.exists(LOGO_PLATFORM):
                st.image(LOGO_PLATFORM, width=140)
   
    # Divisor visual
    st.markdown("---")
   
    # ========= ABAS PRINCIPAIS =========
    tab_numpy, tab_lisflood = st.tabs([" Simulação Rápida", "⚙️ LISFLOOD Avançado"])
   
    with tab_numpy:
        st.header("Simulação Vetorizada (NumPy)")
        # ===== Telemetria Opik =====
        with st.expander("🧭 Telemetria (Opik)"):
            enable_telemetry = st.checkbox(
                "Ativar telemetria Opik",
                value=False,
                help="Registra eventos da simulação para análise."
            )
            colt1, colt2 = st.columns(2)
            with colt1:
                opik_api_key = st.text_input("API Key (opcional)", type="password")
                opik_workspace = st.text_input("Workspace (opcional)")
            with colt2:
                opik_url = st.text_input("Servidor Opik (opcional)")
                use_local = st.checkbox("Usar modo local", value=True, help="Sem servidor externo; guarda localmente.")
            # Status do Opik
            if opik is None:
                st.caption("Opik: ⚠️ não instalado/disponível no ambiente atual.")
            else:
                ver = getattr(opik, "__version__", "")
                st.caption(f"Opik: ✅ disponível{(' - versão ' + ver) if ver else ''}.")

        # Configurar opik se habilitado
        if 'enable_telemetry' not in locals():
            enable_telemetry = False
        if enable_telemetry and opik is not None:
            try:
                if "_opik_configured" not in st.session_state:
                    opik.configure(
                        api_key=opik_api_key or None,
                        workspace=opik_workspace or None,
                        url=opik_url or None,
                        use_local=use_local,
                        force=False,
                    )
                    st.session_state["_opik_configured"] = True
            except (RuntimeError, ValueError, TypeError) as _e:
                st.warning(f"Opik não pôde ser configurado: {_e}")
                enable_telemetry = False
        # Persistir estado de telemetria para outras abas
        st.session_state["telemetry_enabled"] = bool(enable_telemetry)
       
        # Container para uploads
        with st.container():
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("📁 Dados de Entrada")
            col1, col2 = st.columns(2)
           
            with col1:
                st.markdown("**🗺️ Arquivos Obrigatórios**")
                dem_file = st.file_uploader(
                    "Modelo Digital de Elevação (DEM)",
                    type=["tif", "tiff"],
                    help="Arquivo raster com a elevação do terreno",
                    key="np_dem"
                )
               
            with col2:
                st.markdown("**💧 Fontes de Água (Opcional)**")
                vector_files = st.file_uploader(
                    "Áreas de chuva/fontes",
                    type=["gpkg", "shp", "shx", "dbf", "prj"],
                    accept_multiple_files=True,
                    help="Arquivos vetoriais definindo onde a chuva será aplicada",
                    key="np_vec"
                )
           
            st.markdown('</div>', unsafe_allow_html=True)
       
        # Container para parâmetros
        with st.container():
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("⚙️ Parâmetros da Simulação")
           
            col_params1, col_params2, col_params3 = st.columns(3)
           
            with col_params1:
                st.markdown("**🌧️ Precipitação**")
                rain_mm_per_cycle = st.number_input(
                    "Chuva por ciclo (mm)",
                    min_value=0.1,
                    max_value=1000.0,
                    value=5.0,
                    step=0.1,
                    help="Volume de chuva adicionado em cada ciclo de simulação"
                )
               
                total_cycles = st.number_input(
                    "Total de ciclos",
                    min_value=1,
                    max_value=2000,
                    value=100,
                    help="Número total de etapas da simulação"
                )
               
                time_step_minutes = st.number_input(
                    "Duração de cada ciclo (minutos)",
                    min_value=1,
                    max_value=1440,
                    value=10,
                    help="Duração de cada etapa da simulação em minutos"
                )
           
            with col_params2:
                st.markdown("**💧 Comportamento da Água**")
                diffusion_rate = st.number_input(
                    "Taxa de difusão",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.5,
                    step=0.01,
                    help="Controla a velocidade com que a água se espalha (0.1=lento, 1.0=rápido)"
                )
               
                flood_threshold = st.number_input(
                    "Limiar de inundação (metros)",
                    min_value=0.001,
                    max_value=2.0,
                    value=0.1,
                    step=0.01,
                    help="Altura mínima de água para considerar área inundada"
                )
               
                rain_mode = st.selectbox(
                    "Modo de chuva",
                    ["Uniforme na área", "Somente nas fontes"],
                    index=0,
                    help="Uniforme: chuva em toda área | Fontes: apenas nas áreas definidas"
                )
           
            with col_params3:
                st.markdown("**🎬 Visualização**")
                animation_format = st.selectbox("Formato da animação", ["GIF", "MP4"], index=0)
               
                animation_duration = st.slider(
                    "Duração da animação (segundos)",
                    min_value=2,
                    max_value=60,
                    value=10,
                    help="Duração total do vídeo/GIF gerado"
                )
               
                water_alpha = st.slider(
                    "Opacidade da água",
                    0.05, 0.9, 0.35, 0.05,
                    help="Transparência da visualização da água"
                )
                water_min_threshold = st.slider(
                    "Limiar mínimo de água para visualizar (m)",
                    0.0, 0.1, 0.01, 0.001,
                    help="Valores de água abaixo deste limiar não serão mostrados para destacar áreas realmente inundadas"
                )
           
            # Parâmetros adicionais
            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                grid_reduction_factor = st.select_slider(
                    "Resolução da grade",
                    options=[1, 2, 4, 8, 16],
                    value=4,
                    help="Fator de redução da resolução para simulação mais rápida"
                )
               
                quick_preview = st.checkbox(
                    "Pré-visualização rápida (não salvar animação)",
                    value=False,
                    help="Executa simulação sem salvar arquivos de animação"
                )
           
            with col_adv2:
                dom_bg_file = st.file_uploader(
                    "Imagem de fundo (DOM) opcional",
                    type=["tif", "tiff"],
                    help="Imagem de satélite/ortofoto para fundo da visualização",
                    key="np_dom_bg"
                )
               
                river_vector_files = st.file_uploader(
                    "Rio (opcional)",
                    type=["gpkg", "shp", "shx", "dbf", "prj"],
                    accept_multiple_files=True,
                    help="Arquivos vetoriais definindo áreas de rio para cálculo de transbordamento",
                    key="np_river"
                )
               
                apply_hs = st.checkbox(
                    "Aplicar relevo (hillshade) sobre DOM",
                    value=True,
                    help="Aplica efeito de relevo sobre a imagem de fundo"
                )
               
                hs_intensity = st.slider(
                    "Intensidade do relevo",
                    0.0, 1.0, 0.6, 0.05,
                    help="Intensidade do efeito de relevo sobre a imagem de fundo"
                )
           
            st.markdown('</div>', unsafe_allow_html=True)
       
        # Métricas em tempo real
        st.subheader("📊 Métricas em Tempo Real")
        stats_cols = st.columns(4)
        overflow_ph, time_ph, flooded_ph, vol_ph = stats_cols
        overflow_ph.metric("Tempo para Transbordar", "N/A")
        time_ph.metric("Tempo de Simulação", "0h 0m")
        flooded_ph.metric("Área Inundada", "0.00%")
        vol_ph.metric("Volume de Água", "0.00 m³")
       
        # Botão de ação
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            simular = st.button(
                "🚀 INICIAR SIMULAÇÃO",
                type="primary",
                use_container_width=True,
                key="simular_numpy"
            )
        with col_info:
            if not dem_file:
                st.warning("⚠️ Faça upload do DEM para iniciar a simulação")
            else:
                st.success("✅ Pronto para simular!")
       
        # Área para a animação
        anim_area = st.empty()

        # ========= LÓGICA DE SIMULAÇÃO NUMÉRICA =========
        if simular and dem_file:
            tmp = None
            tmp_anim = None  # Caminho do arquivo de animação (se gerado)
            try:
                sim_start_ts = time.time()
                vector_path = None
                river_path = None
               
                # Processar DEM e vetores de fonte
                if vector_files:
                    dem_path, vector_path, tmp = _process_uploaded_files(dem_file, vector_files)
                    if not vector_path:
                        st.error("Falha ao processar o arquivo vetorial.")
                        raise RuntimeError("Arquivo vetorial inválido")
                else:
                    # Salvar apenas DEM
                    tmp = tempfile.mkdtemp(prefix="sim_numpy_")
                    dem_path = os.path.join(tmp, dem_file.name)
                    with open(dem_path, "wb") as f:
                        f.write(dem_file.getbuffer())
               
                # Processar vetor de rio se enviado
                if river_vector_files:
                    for f in river_vector_files:
                        if f.name.lower().endswith((".gpkg", ".shp")):
                            rp = os.path.join(tmp or tempfile.mkdtemp(prefix="sim_numpy_"), f"river_{f.name}")
                            with open(rp, "wb") as out:
                                out.write(f.getbuffer())
                            if f.name.lower().endswith(".gpkg"):
                                river_path = rp
                            elif f.name.lower().endswith(".shp") and river_path is None:
                                river_path = rp
               
                # Configurar dados geoespaciais
                dem_data, sources_mask, transform, crs, cell_size, river_mask = _setup_geodata(
                    dem_path, vector_path, grid_reduction_factor, river_path
                )
               
                # Inicializar modelo
                model = GamaFloodModelNumpy(
                    dem_data, sources_mask, diffusion_rate,
                    flood_threshold, cell_size, river_mask
                )
                model.uniform_rain = (rain_mode == "Uniforme na área")
               
                # Preparar fundo visual
                background_rgb = None
                if dom_bg_file is not None and tmp:
                    dom_tmp = os.path.join(tmp, f"bg_{dom_bg_file.name}")
                    with open(dom_tmp, "wb") as out:
                        out.write(dom_bg_file.getbuffer())
                    background_rgb = _prepare_background(dom_tmp, dem_data.shape, crs)
               
                # Configurar visualização
                fig, _, water_layer, rain_particles, title, bounds = _setup_visualization(
                    dem_data, transform, crs, background_rgb, apply_hs, hs_intensity
                )
                x_min, y_min, x_max, y_max = bounds
               
                progress = st.progress(0, text="Inicializando...")
               
                # Função de atualização da animação
                @maybe_track(enable_telemetry and (opik is not None), name="numpy_update", type="general", tags=["sim:update"])
                def update(frame):
                    # Adicionar chuva
                    model.add_water(rain_mm_per_cycle)
                   
                    # Executar passo de fluxo
                    model.run_flow_step()
                   
                    # Atualizar estatísticas
                    model.update_stats(time_step_minutes)
                   
                    # Atualizar visualização da água
                    water = model.water_height
                    masked = np.ma.masked_where(water < water_min_threshold, water)
                    water_layer.set_data(masked)
                    water_layer.set_alpha(water_alpha)
                   
                    # Partículas de chuva (efeito visual)
                    n = int(rain_mm_per_cycle * 150)
                    rx = np.random.uniform(x_min, x_max, n)
                    ry = np.random.uniform(y_min, y_max, n)
                    rain_particles.set_data(rx, ry)
                   
                    # Atualizar título e métricas
                    h, m = divmod(model.simulation_time_minutes, 60)
                    title.set_text(f"Simulação de Inundação | Tempo: {h}h {m}m")
                   
                    # Atualizar métricas em tempo real
                    latest = model.history[-1]
                    if model.overflow_time_minutes is not None:
                        ho, mo = divmod(model.overflow_time_minutes, 60)
                        overflow_ph.metric("Tempo para Transbordar", f"{ho}h {mo}m")
                    time_ph.metric("Tempo de Simulação", f"{h}h {m}m")
                    flooded_ph.metric("Área Inundada", f"{latest['flooded_percent']:.2f}%")
                    vol_ph.metric("Volume de Água", f"{latest['total_water_volume_m3']:.2f} m³")
                   
                    # Atualizar barra de progresso
                    progress.progress(
                        int(100 * (frame + 1) / total_cycles),
                        text=f"Simulando ciclo {frame + 1}/{total_cycles}"
                    )
                   
                    return [water_layer, rain_particles, title]
               
                # Executar simulação
                if quick_preview:
                    # Modo pré-visualização: executar sem salvar animação
                    for frame in range(total_cycles):
                        update(frame)
                    # Mostrar resultado final
                    anim_area.pyplot(fig, clear_figure=False)
                else:
                    # Modo completo: gerar animação
                    fps = max(1, total_cycles // animation_duration)
                    interval = 1000 // fps  # ms entre frames
                   
                    anim = FuncAnimation(
                        fig, update, frames=total_cycles,
                        blit=True, interval=interval
                    )
                   
                    # Salvar animação
                    ext = animation_format.lower()
                    tmp_anim = os.path.join(tmp or tempfile.gettempdir(), f"simulation.{ext}")
                    try:
                        if ext == 'gif':
                            # GIF via Pillow sempre disponível
                            anim.save(tmp_anim, writer='pillow', dpi=150, fps=fps)
                        else:
                            # MP4: garantir ffmpeg disponível via imageio-ffmpeg
                            ffmpeg_bin = None
                            try:
                                import imageio_ffmpeg  # type: ignore
                                ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
                            except (ImportError, OSError):
                                # Tentar instalar imageio-ffmpeg on-the-fly
                                subprocess.run([sys.executable, '-m', 'pip', 'install', 'imageio-ffmpeg', '--quiet'], capture_output=True, check=False)
                                # Importar após tentativa de instalação
                                try:
                                    import importlib as _im
                                    imageio_ffmpeg = _im.import_module('imageio_ffmpeg')  # type: ignore
                                except ModuleNotFoundError:
                                    ffmpeg_bin = None
                                else:
                                    try:
                                        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
                                    except OSError:
                                        ffmpeg_bin = None

                            if ffmpeg_bin:
                                # Apontar Matplotlib para o binário específico
                                # Definir caminho do ffmpeg para o Matplotlib
                                plt.rcParams['animation.ffmpeg_path'] = ffmpeg_bin
                                # Usar FFMpegWriter com parâmetros compatíveis
                                from matplotlib.animation import FFMpegWriter
                                writer = FFMpegWriter(
                                    fps=fps,
                                    codec='libx264',
                                    bitrate=1800,
                                    extra_args=['-pix_fmt', 'yuv420p']
                                )
                                anim.save(tmp_anim, writer=writer, dpi=150)
                            else:
                                # Última tentativa: writer='ffmpeg' (usa PATH/rcParams). Passar extra_args para compatibilidade
                                anim.save(
                                    tmp_anim,
                                    writer='ffmpeg',
                                    dpi=150,
                                    fps=fps,
                                    extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
                                )
                        # Telemetria do salvamento
                        maybe_track(enable_telemetry and (opik is not None), name="save_animation", type="general", tags=[ext.upper()])(lambda: None)()
                    except (RuntimeError, ValueError, OSError) as e:
                        # Fallback para GIF
                        st.warning(f"Falha ao salvar em {ext.upper()} ({e}). Tentando GIF...")
                        tmp_anim = os.path.join(tmp or tempfile.gettempdir(), "simulation.gif")
                        anim.save(tmp_anim, writer='pillow', dpi=150, fps=fps)
                        ext = 'gif'
                   
                    # Exibir animação
                    with open(tmp_anim, "rb") as f:
                        if ext == 'gif':
                            anim_area.image(f.read())
                        else:
                            anim_area.video(f.read())
               
                # Pós-processamento e downloads
                total_rain_mm = rain_mm_per_cycle * total_cycles
                _post_process_simulation(
                    model, tmp,
                    tmp_anim,
                    animation_format,
                    total_rain_mm,
                    cell_size,
                    sources_mask
                )
                # Telemetria de conclusão
                sim_dur = time.time() - sim_start_ts
                maybe_track(enable_telemetry and (opik is not None), name="simulation_completed", type="general", tags=["numpy"], metadata={
                    "duration_sec": round(sim_dur, 3),
                    "total_cycles": int(total_cycles),
                    "rain_mm_per_cycle": float(rain_mm_per_cycle),
                })(lambda: None)()

            except (RuntimeError, ValueError, OSError) as e:
                st.error(f"Erro na simulação: {e}")
                import traceback
                st.error(traceback.format_exc())
                maybe_track(enable_telemetry and (opik is not None), name="simulation_error", type="general", tags=["error"])(lambda: None)()
            finally:
                # Limpeza
                if tmp and os.path.exists(tmp):
                    shutil.rmtree(tmp, ignore_errors=True)
                plt.close('all')
   
    with tab_lisflood:
        st.header("Simulação LISFLOOD (Docker)")
       
        # Sidebar com parâmetros LISFLOOD
        with st.sidebar:
            st.markdown("### 📁 Dados de Entrada LISFLOOD")
           
            dem_file_lf = st.file_uploader(
                "DEM .tif",
                type=["tif", "tiff"],
                key="lf_dem"
            )
           
            dom_file_lf = st.file_uploader(
                "DOM .tif (opcional)",
                type=["tif", "tiff"],
                key="lf_dom"
            )
           
            dsm_file_lf = st.file_uploader(
                "DSM .tif (opcional)",
                type=["tif", "tiff"],
                key="lf_dsm"
            )
           
            stl_file_lf = st.file_uploader(
                "Modelo 3D (.stl) → gerar DSM",
                type=["stl"],
                key="lf_stl"
            )
           
            # Parâmetros STL→DSM
            with st.expander("🔧 Parâmetros STL→DSM"):
                col1, col2 = st.columns(2)
                with col1:
                    x_offset = st.number_input("Offset X", value=0.0, step=1.0)
                    y_offset = st.number_input("Offset Y", value=0.0, step=1.0)
                with col2:
                    z_scale = st.number_input("Escala Z", value=1.0, step=0.1)
                    rotation = st.number_input("Rotação Z (°)", value=0.0, step=1.0)
                preview_dsm = st.checkbox("Pré-visualizar DSM gerado", value=True)
           
            mask_file_lf = st.file_uploader(
                "Máscara .tif (opcional)",
                type=["tif", "tiff"],
                key="lf_mask"
            )
           
            template_xml = st.file_uploader(
                "Template XML (opcional)",
                type=["xml"],
                key="lf_tpl"
            )
           
            # Configurações de simulação
            st.markdown("### ⚙️ Parâmetros de Simulação")
            step_start = st.text_input("Início (dd/mm/aaaa HH:MM)", "01/01/2024 00:00")
            step_end = st.text_input("Fim (dd/mm/aaaa HH:MM)", "01/01/2024 01:00")
            timestep = st.number_input("Intervalo (segundos)", min_value=60, value=3600, step=60)
           
            # Opções avançadas
            st.markdown("### 🔧 Opções Avançadas")
            generate_ldd = st.checkbox("Gerar LDD do DEM", value=True)
            validate_only = st.checkbox("Apenas validar (--initonly)", value=True)
            use_minimal_template = st.checkbox("Usar template mínimo", value=True)

            # Telemetria para LISFLOOD
            with st.expander("🧭 Telemetria (Opik)"):
                telemetry_lf = st.checkbox(
                    "Ativar telemetria Opik (LISFLOOD)",
                    value=st.session_state.get("telemetry_enabled", False),
                    help="Registra eventos de execução do LISFLOOD."
                )
                st.session_state["telemetry_enabled"] = bool(telemetry_lf)
                if opik is None:
                    st.caption("Opik: ⚠️ não instalado/disponível.")
                else:
                    ver = getattr(opik, "__version__", "")
                    cfg = "_opik_configured" in st.session_state
                    st.caption(f"Opik: ✅ disponível{(' - versão ' + ver) if ver else ''}. Configurado: {'sim' if cfg else 'não'}.")
                    if telemetry_lf and not cfg:
                        # Configurar rapidamente em modo local, sem credenciais
                        try:
                            opik.configure(api_key=None, workspace=None, url=None, use_local=True, force=False)
                            st.session_state["_opik_configured"] = True
                            st.success("Opik configurado em modo local.")
                        except Exception as _e:
                            st.warning(f"Falha ao configurar Opik local: {_e}")
       
        # Área principal de execução LISFLOOD
        col_run, col_status = st.columns([2, 1])
       
        with col_run:
            st.subheader("🎯 Execução do LISFLOOD")
           
            if st.button("🚀 Executar Simulação LISFLOOD", type="primary", key="run_lisflood"):
                if not dem_file_lf:
                    st.error("⚠️ É necessário enviar um arquivo DEM para executar o LISFLOOD.")
                else:
                    # Preparar ambiente temporário
                    tmp_dir = tempfile.mkdtemp(prefix="lisflood_run_")
                   
                    try:
                        with st.spinner("🔄 Preparando ambiente de execução..."):
                            # Criar estrutura de pastas
                            input_folder, output_folder = lf_prepare_input_folder(tmp_dir)
                           
                            # Processar e salvar arquivos enviados
                            def save_uploaded_file(uploaded_file, filename):
                                if not uploaded_file:
                                    return None
                                path = os.path.join(input_folder, filename)
                                with open(path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                return path
                           
                            # Salvar arquivos principais
                            dem_path = save_uploaded_file(dem_file_lf, "dem.tif")
                            dom_path = save_uploaded_file(dom_file_lf, "dom.tif")
                            dsm_path = save_uploaded_file(dsm_file_lf, "dsm.tif")
                            mask_path = save_uploaded_file(mask_file_lf, "mask.tif")
                           
                            # Processar STL se enviado
                            if stl_file_lf and not dsm_path:
                                st.info("🔄 Convertendo STL para DSM...")
                                stl_path = save_uploaded_file(stl_file_lf, "model.stl")
                                dsm_output = os.path.join(input_folder, "dsm_from_stl.tif")
                                # Garantias mínimas
                                if not dem_path or not os.path.exists(dem_path):
                                    raise RuntimeError("DEM inválido ao converter STL→DSM.")
                                if not stl_path or not os.path.exists(stl_path):
                                    raise RuntimeError("STL inválido ao converter STL→DSM.")
                                success, message = lf_convert_stl_to_dsm(
                                    dem_path, stl_path, dsm_output,
                                    x_offset=x_offset, y_offset=y_offset,
                                    z_scale=z_scale, rot_deg=rotation
                                )
                               
                                if success:
                                    dsm_path = dsm_output
                                    st.success("✅ DSM gerado com sucesso!")
                                    if preview_dsm:
                                        st.subheader("Pré-visualização do DSM")
                                        _visualize_geotiff(dsm_path)
                                else:
                                    st.error(f"❌ Falha na conversão: {message}")
                           
                            # Copiar e converter rasters
                            st.info("🔄 Processando dados de entrada...")
                            if not dem_path or not os.path.exists(dem_path):
                                raise RuntimeError("DEM ausente após upload.")
                            lf_copy_rasters(input_folder, dem_path, dom_path, dsm_path, mask_path)
                           
                            # Garantir máscara PCRaster
                            if os.path.exists(os.path.join(input_folder, "MASK.tif")):
                                exit_code, stdout, stderr = lf_ensure_mask_pcraster(input_folder)
                                if exit_code != 0:
                                    st.error(f"Erro ao criar MASK.map: {stderr}")
                           
                            # Gerar LDD se necessário
                            if generate_ldd:
                                st.info("🔄 Gerando rede de drenagem (LDD)...")
                                exit_code, stdout, stderr = lf_generate_ldd_from_dem(input_folder)
                                if exit_code == 0:
                                    st.success("✅ LDD gerado com sucesso!")
                                else:
                                    st.error(f"❌ Erro ao gerar LDD: {stderr}")
                           
                            # Preparar arquivo de configuração
                            st.info("🔄 Configurando arquivo de controle...")
                            xml_path = os.path.join(input_folder, "lisflood_config.xml")
                           
                            replacements = {
                                "MaskMap": "/input/MASK.map",
                                "Ldd": "/input/Ldd.map",
                                "PathOut": "/input/output/",
                                "StepStart": step_start,
                                "StepEnd": step_end,
                                "DtSec": str(timestep),
                                "DtInit": str(timestep),  # ADICIONADO
                                "RepStep": "1",           # ADICIONADO
                                "CalendarConvention": "proleptic_gregorian",
                            }
                           
                            if use_minimal_template or not template_xml:
                                create_lisflood_minimal_xml(xml_path, replacements)
                            else:
                                # Usar template personalizado com patch seguro
                                template_path = save_uploaded_file(template_xml, "template.xml")
                                if template_path and os.path.exists(template_path):
                                    lf_patch_settings_xml_with_defaults(
                                        template_path, xml_path, replacements
                                    )
                                else:
                                    st.warning("Falha ao salvar template. Usando template mínimo.")
                                    create_lisflood_minimal_xml(xml_path, replacements)

                            # Verificar se o XML foi criado corretamente
                            if os.path.exists(xml_path):
                                st.success(f"✅ Arquivo de configuração criado: {xml_path}")
                                # Verificação crítica: tags essenciais e elemento lfuser
                                try:
                                    import xml.etree.ElementTree as ET
                                    tree = ET.parse(xml_path)
                                    root = tree.getroot()
                                    lfuser_el = root.find('lfuser')
                                    with open(xml_path, 'r', encoding='utf-8') as f:
                                        xml_content = f.read()
                                    missing = [tag for tag in ['<lfuser>', '<lfbinding>', 'DtInit', 'CalendarConvention'] if tag not in xml_content]
                                    if lfuser_el is None or missing:
                                        st.error(f"❌ XML incompleto. Faltando: {missing or ['lfuser']} ")
                                        st.code(xml_content, language='xml')
                                        st.stop()
                                    if st.checkbox("Mostrar conteúdo do XML (debug)"):
                                        st.code(xml_content, language='xml')
                                except Exception as e:
                                    st.error(f"❌ Erro ao verificar XML: {e}")
                                    st.stop()
                            else:
                                st.error("❌ Falha ao criar arquivo de configuração XML")
                                st.stop()

                        # Telemetria: início da execução LISFLOOD
                        _telemetry_on = bool(st.session_state.get("telemetry_enabled", False) and (opik is not None))
                        maybe_track(_telemetry_on, name="lisflood_run_start", type="general", tags=["lisflood", "start"], metadata={
                            "validate_only": bool(validate_only),
                            "generate_ldd": bool(generate_ldd),
                            "use_minimal_template": bool(use_minimal_template),
                            "step_start": step_start,
                            "step_end": step_end,
                            "timestep_sec": int(timestep),
                            "has_dom": bool(dom_file_lf is not None),
                            "has_dsm": bool(dsm_file_lf is not None or stl_file_lf is not None),
                            "has_mask": bool(mask_file_lf is not None),
                        })(lambda: None)()
                       
                        # Executar LISFLOOD
                        # Checagem real do Docker
                        ok_docker, docker_msg = _check_docker_available()
                        if not ok_docker:
                            st.error("Docker não disponível: " + docker_msg)
                            st.stop()
                        st.info("🎯 Executando LISFLOOD no Docker...")
                       
                        extra_args = ["--initonly"] if validate_only else None
                        exit_code, stdout, stderr = lf_run_lisflood(
                            input_folder, "lisflood_config.xml", extra_args=extra_args
                        )
                       
                        # Processar resultados
                        st.subheader("📊 Resultados da Execução")
                       
                        # Exibir logs
                        with st.expander("🔍 Logs de Execução"):
                            st.text(f"Código de saída: {exit_code}")
                            st.text_area("Saída padrão:", stdout, height=200)
                            st.text_area("Erros:", stderr, height=200)
                       
                        if exit_code == 0:
                            st.success("✅ Simulação LISFLOOD concluída com sucesso!")
                            maybe_track(_telemetry_on, name="lisflood_run_success", type="general", tags=["lisflood", "success"], metadata={
                                "exit_code": int(exit_code),
                                "out_len": len(stdout or ""),
                            })(lambda: None)()
                           
                            # Listar arquivos de saída
                            output_files = lf_list_outputs(output_folder)
                            if output_files:
                                st.subheader("📁 Arquivos Gerados")
                               
                                # Visualizar rasters
                                tif_files = [f for f in output_files if f.lower().endswith(('.tif', '.tiff'))]
                                if tif_files:
                                    selected_file = st.selectbox(
                                        "Selecione um arquivo para visualizar:",
                                        tif_files,
                                        format_func=os.path.basename
                                    )
                                    if selected_file:
                                        _visualize_geotiff(selected_file)
                               
                                # Download de resultados
                                st.subheader("📥 Download dos Resultados")
                                for file_path in output_files[:10]:  # Limitar a 10 arquivos
                                    file_name = os.path.basename(file_path)
                                    with open(file_path, "rb") as f:
                                        st.download_button(
                                            label=f"📥 {file_name}",
                                            data=f.read(),
                                            file_name=file_name,
                                            mime="application/octet-stream",
                                            key=f"download_{file_name}"
                                        )
                            else:
                                st.warning("⚠️ Nenhum arquivo de saída foi gerado.")
                       
                        else:
                            st.error("❌ A simulação LISFLOOD encontrou erros.")
                            maybe_track(_telemetry_on, name="lisflood_run_error", type="general", tags=["lisflood", "error"], metadata={
                                "exit_code": int(exit_code),
                                "err_len": len(stderr or ""),
                            })(lambda: None)()
                           
                            # Análise de erros comuns
                            if "missing" in stderr.lower():
                                st.info("💡 Dica: Verifique se todos os arquivos necessários foram fornecidos.")
                            if "permission" in stderr.lower():
                                st.info("💡 Dica: Problema de permissão. Verifique o Docker.")
                   
                    except (RuntimeError, OSError, ValueError) as e:
                        st.error(f"❌ Erro durante a execução: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                        _telemetry_on = bool(st.session_state.get("telemetry_enabled", False) and (opik is not None))
                        maybe_track(_telemetry_on, name="lisflood_run_exception", type="general", tags=["lisflood", "exception"], metadata={
                            "error": str(e)[:500],
                        })(lambda: None)()
                   
                    finally:
                        # Limpeza
                        if os.path.exists(tmp_dir):
                            shutil.rmtree(tmp_dir, ignore_errors=True)
       
        with col_status:
            st.subheader("📋 Status do Sistema")
           
            # Verificar dependências
            st.markdown("#### ✅ Verificações do Sistema")
            ok_docker, docker_msg = _check_docker_available()
            if ok_docker:
                st.success("Docker: ✅ " + docker_msg)
            else:
                st.warning("Docker: ⚠️ " + docker_msg)
           
            # Verificar arquivos de entrada
            if dem_file_lf:
                st.success("DEM: ✅ Carregado")
            else:
                st.warning("DEM: ⚠️ Aguardando upload")

            # Status do trimesh (necessário para STL→DSM)
            st.markdown("#### 📦 Dependência opcional: trimesh")
            tri_ok, tri_msg = _check_trimesh_installed()
            if tri_ok:
                st.success("trimesh: ✅ " + tri_msg)
            else:
                st.warning("trimesh: ⚠️ " + tri_msg)
                if st.button("Instalar trimesh agora", key="btn_install_trimesh"):
                    with st.spinner("Instalando trimesh..."):
                        code, out, err = _install_trimesh()
                    if code == 0:
                        st.success("trimesh instalado. Talvez seja necessário recarregar a página do app.")
                    else:
                        st.error("Falha ao instalar trimesh.")
                        with st.expander("Logs de instalação"):
                            st.code(out or "", language="bash")
                            st.code(err or "", language="bash")
           
            # Informações de uso
            st.markdown("#### 📋 Guia Rápido")
            st.markdown("""
            1. **Upload do DEM** (obrigatório)
            2. **Configure parâmetros** na sidebar
            3. **Execute a simulação**
            4. **Visualize resultados** e faça downloads
            """)
           
            # Link para documentação
            st.markdown("#### 📚 Recursos")
            st.markdown("[Documentação LISFLOOD](https://ec-jrc.github.io/lisflood/)")

if __name__ == "__main__":
    main()