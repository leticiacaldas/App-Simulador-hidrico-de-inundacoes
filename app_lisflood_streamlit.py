from shapes import apply_custom_styles, create_header

# Imports padrão e bibliotecas utilizadas no app
import os
import io
import sys
import time
import shutil
import tempfile
import subprocess
import contextlib
from typing import Optional, Tuple, Any, Dict, cast

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import rasterio as rio
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from rasterio.transform import array_bounds, from_origin
from rasterio.features import rasterize
import geopandas as gpd
import contextily as ctx
from scipy import ndimage
from matplotlib.patches import Patch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from sklearn.cluster import DBSCAN
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
)

# Opik (telemetria) é opcional
try:
    import opik  # type: ignore
except Exception:
    opik = None  # type: ignore

# Feature-flag para LISFLOOD (desativado por padrão)
ENABLE_LISFLOOD = str(os.environ.get("ENABLE_LISFLOOD", "")
                      ).strip().lower() in {"1", "true", "yes", "on"}


def maybe_track(enabled: bool = False, name: str = "", type: str = "", tags: Optional[list] = None, metadata: Optional[dict] = None):
    """Decorator/closure no-op para telemetria segura.

    Uso esperado: @maybe_track(cond, name="evt")(func) ou maybe_track(cond, name=...)(lambda: None)()
    """
    def _decorator(fn):
        def _wrapped(*args, **kwargs):
            try:
                if enabled and opik is not None:
                    # Evita falhas se opik não estiver configurado
                    _ = (name, type, tags, metadata)
                return fn(*args, **kwargs)
            except Exception:
                return fn(*args, **kwargs)
        return _wrapped
    return _decorator


class GamaFloodModelNumpy:
    """Motor de simulação de inundação vetorizado com NumPy (versão local).

    Parâmetros:
    - dem_data: matriz 2D com altitudes
    - sources_mask: máscara 2D (uint8/bool) com fontes de chuva
    - diffusion_rate: fração de água que pode se mover por passo
    - flood_threshold: limiar de inundação (m)
    - cell_size_meters: tamanho da célula (m)
    - river_mask: máscara opcional de rio (não obrigatório)
    """

    def __init__(self, dem_data: np.ndarray, sources_mask: np.ndarray, diffusion_rate: float, flood_threshold: float, cell_size_meters: float, river_mask: Optional[np.ndarray] = None):
        self.height, self.width = dem_data.shape
        self.diffusion_rate = float(diffusion_rate)
        self.flood_threshold = float(flood_threshold)
        self.cell_area = float(cell_size_meters) * float(cell_size_meters)
        self.altitude = dem_data.astype(np.float32)
        self.is_source = (sources_mask.astype(
            bool) if sources_mask is not None else np.zeros_like(self.altitude, dtype=bool))
        self.river_mask = (river_mask.astype(
            bool) if river_mask is not None else np.zeros_like(self.altitude, dtype=bool))
        self.water_height = np.zeros_like(self.altitude, dtype=np.float32)
        self.active_cells_coords = set(zip(*np.where(self.is_source)))
        self.simulation_time_minutes = 0
        self.overflow_time_minutes: Optional[int] = None
        self.history: list[dict] = []
        self.uniform_rain: bool = True

    def add_water(self, rain_mm: float):
        water_to_add_meters = float(rain_mm) / 1000.0
        if water_to_add_meters <= 0:
            return
        # Chuva uniforme: aplica em toda a grade
        if self.uniform_rain:
            self.water_height += water_to_add_meters
            # Ativar todas as células que agora têm água
            ys, xs = np.where(self.water_height > 0)
            self.active_cells_coords.update(zip(ys, xs))
            return

        # Caso não seja uniforme: aplicar nas fontes; se não houver fontes, usar rio como fallback; se ainda assim não houver, aplicar uniforme para evitar simulação "em branco"
        if np.any(self.is_source):
            self.water_height[self.is_source] += water_to_add_meters
            ys, xs = np.where(self.is_source)
            self.active_cells_coords.update(zip(ys, xs))
        elif np.any(self.river_mask):
            self.water_height[self.river_mask] += water_to_add_meters * 0.2
            ys, xs = np.where(self.river_mask)
            self.active_cells_coords.update(zip(ys, xs))
        else:
            # Fallback: sem fontes nem rio definidos, distribuir uniformemente
            self.water_height += water_to_add_meters
            ys, xs = np.where(self.water_height > 0)
            self.active_cells_coords.update(zip(ys, xs))

    def run_flow_step(self):
        if not self.active_cells_coords:
            return
        newly_active, to_deactivate = set(), set()
        prev = self.water_height.copy()
        H, W = self.height, self.width
        for y, x in list(self.active_cells_coords):
            cur_w = prev[y, x]
            if cur_w <= 1e-3:
                to_deactivate.add((y, x))
                continue
            cur_total = self.altitude[y, x] + cur_w
            neigh = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        neigh.append((ny, nx))
            if not neigh:
                continue
            ny, nx = zip(*neigh)
            n_total = self.altitude[ny, nx] + prev[ny, nx]
            mask_lower = n_total < cur_total
            if not np.any(mask_lower):
                to_deactivate.add((y, x))
                continue
            lower_coords = np.array(neigh)[mask_lower]
            lower_total = n_total[mask_lower]
            total_diff = float(np.sum(cur_total - lower_total))
            if total_diff <= 0:
                continue
            move_amount = cur_w * self.diffusion_rate
            if move_amount <= 0:
                continue
            for i, (ny2, nx2) in enumerate(lower_coords):
                diff = cur_total - lower_total[i]
                frac = float(diff) / total_diff
                wmv = min(move_amount * frac, diff / 2.0)
                if wmv > 0:
                    self.water_height[y, x] -= wmv
                    self.water_height[ny2, nx2] += wmv
                    newly_active.add((ny2, nx2))
        self.active_cells_coords.difference_update(to_deactivate)
        self.active_cells_coords.update(newly_active)

    def update_stats(self, time_step_minutes: int):
        self.simulation_time_minutes += int(time_step_minutes)
        inundated = self.water_height > self.flood_threshold
        if self.overflow_time_minutes is None and np.any(inundated & ~self.is_source):
            self.overflow_time_minutes = self.simulation_time_minutes
        self.history.append({
            "time_minutes": self.simulation_time_minutes,
            "flooded_percent": float(np.sum(inundated)) / float(inundated.size) * 100.0,
            "active_cells": int(len(self.active_cells_coords)),
            "max_depth": float(np.max(self.water_height)) if self.water_height.size > 0 else 0.0,
            "total_water_volume_m3": float(np.sum(self.water_height * self.cell_area)),
        })


def _prepare_background(img_path: str, target_shape: Tuple[int, int], target_crs) -> Optional[np.ndarray]:
    """Carrega um raster de fundo (ex. DOM ortomosaico), reamostra para target_shape e retorna RGB float [0,1]."""
    try:
        H, W = target_shape
        with rio.open(img_path) as src:
            count = min(3, src.count or 1)
            bands: list[np.ndarray] = []
            for b in range(1, count + 1):
                band = src.read(
                    b,
                    out_shape=(H, W),
                    resampling=Resampling.bilinear,
                ).astype(float)
                bands.append(band)
            if len(bands) == 1:
                rgb = np.stack([bands[0]] * 3, axis=-1)
            else:
                rgb = np.stack(bands, axis=-1)
            # Normalização robusta para [0,1]
            if not np.issubdtype(rgb.dtype, np.floating):
                rgb = rgb.astype(float)
            finite_vals = rgb[np.isfinite(rgb)]
            if finite_vals.size > 0:
                p2, p98 = np.percentile(finite_vals, (2, 98))
                p2 = max(float(p2), 0.0)
                denom = (float(p98) - p2) or 1.0
                rgb = np.clip((rgb - p2) / denom, 0, 1)
            else:
                rgb = np.zeros_like(rgb, dtype=float)
            return rgb
    except (OSError, ValueError, RuntimeError) as e:
        st.warning(f"Falha ao preparar fundo (DOM): {e}")
        return None


def _process_uploaded_files(dem_file, vector_files):
    """Salva DEM e arquivos vetoriais (primeiro .gpkg/.shp encontrado). Retorna (dem_path, vector_path, tmp_dir)."""
    if not dem_file:
        return None, None, None
    tmp = tempfile.mkdtemp(prefix="sim_numpy_")
    dem_path = os.path.join(tmp, dem_file.name)
    with open(dem_path, "wb") as f:
        f.write(dem_file.getbuffer())
    vector_path = None
    for f in (vector_files or []):
        p = os.path.join(tmp, f.name)
        with open(p, "wb") as out:
            out.write(f.getbuffer())
        if f.name.lower().endswith((".gpkg", ".shp")) and vector_path is None:
            vector_path = p
    return dem_path, vector_path, tmp


def _setup_geodata(dem_path: str, vector_path: Optional[str], grid_reduction_factor: int, river_path: Optional[str] = None):
    """Lê DEM, reamostra grade, rasteriza fontes (e opcionalmente rio) e retorna (dem, sources_mask, transform, crs, cell_size_m, river_mask)."""
    with rio.open(dem_path) as dem_src:
        crs = dem_src.crs
        t = dem_src.transform
        factor = max(1, int(grid_reduction_factor))
        h2 = max(1, dem_src.height // factor)
        w2 = max(1, dem_src.width // factor)
        dem_data = dem_src.read(1, out_shape=(
            h2, w2), resampling=Resampling.bilinear)
        # Construir novo transform com base na origem e no tamanho de pixel reamostrado
        px = abs(float(t.a)) * factor
        py = abs(float(t.e)) * factor
        transform = from_origin(float(getattr(t, 'c', 0.0) or t.c), float(
            getattr(t, 'f', 0.0) or t.f), px, py)
        # Tamanho de célula (m) aproximado
        if crs and crs.is_geographic:
            cell_size = px * 111320.0
        else:
            cell_size = px

    # Rasterizar fontes (vetor)
    if vector_path:
        gdf = gpd.read_file(vector_path)
        if gdf.crs is None and crs is not None:
            gdf = gdf.set_crs(crs)
        elif crs is not None and gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        sources_mask = rasterize(
            gdf.geometry,
            out_shape=dem_data.shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        )
    else:
        sources_mask = np.zeros_like(dem_data, dtype=np.uint8)

    # Rasterizar rio (opcional)
    river_mask = np.zeros_like(dem_data, dtype=bool)
    if river_path and os.path.exists(river_path):
        try:
            rgdf = gpd.read_file(river_path)
            if rgdf.crs is None and crs is not None:
                rgdf = rgdf.set_crs(crs)
            elif crs is not None and rgdf.crs != crs:
                rgdf = rgdf.to_crs(crs)
            river_mask = rasterize(
                rgdf.geometry,
                out_shape=dem_data.shape,
                transform=transform,
                fill=0,
                all_touched=True,
                dtype=np.uint8,
            ).astype(bool)
        except Exception:
            river_mask = np.zeros_like(dem_data, dtype=bool)

    return dem_data, sources_mask, transform, crs, cell_size, river_mask


def _setup_visualization(dem_data: np.ndarray, transform, crs, background_rgb: Optional[np.ndarray], apply_hs: bool, hs_intensity: float):
    """Prepara a figura principal e camadas (água e partículas de chuva)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    bounds = array_bounds(dem_data.shape[0], dem_data.shape[1], transform)

    # Fundo: DOM/Orto se fornecido, caso contrário DEM
    if background_rgb is not None and background_rgb.shape[:2] == dem_data.shape:
        img = (np.clip(background_rgb, 0, 1) * 255).astype(np.uint8)
        ax.imshow(img, extent=bounds, alpha=1.0)
    else:
        dem_b = dem_data.astype(float)
        vmin, vmax = np.nanpercentile(
            dem_b, (5, 95)) if np.isfinite(dem_b).any() else (0, 1)
        ax.imshow(dem_b, extent=bounds, cmap="terrain",
                  vmin=vmin, vmax=vmax, alpha=0.85)

    # Opcional: mapa base online por trás
    with contextlib.suppress(Exception):
        ctx.add_basemap(ax, crs=crs, source='CartoDB.Positron')

    water_layer = ax.imshow(np.zeros_like(
        dem_data), extent=bounds, cmap='Blues', alpha=0.0, vmin=0, vmax=1)
    rain_particles, = ax.plot(
        [], [], '.', color='royalblue', markersize=1.0, alpha=0.7)
    title = ax.set_title("Simulação de Inundação | Tempo: 0h 0m")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    return fig, ax, water_layer, rain_particles, title, bounds


def _visualize_geotiff(fp: str):
    with rio.open(fp) as src:
        data = src.read(1)
        bounds = array_bounds(src.height, src.width, src.transform)
        crs = src.crs
    fig, ax = plt.subplots(figsize=(10, 8))
    vmin, vmax = np.nanpercentile(
        data, (5, 95)) if np.isfinite(data).any() else (0, 1)
    ax.imshow(data, extent=bounds, cmap="Blues",
              alpha=0.8, vmin=vmin, vmax=vmax)
    with contextlib.suppress(OSError, ValueError, RuntimeError):
        ctx.add_basemap(ax, crs=crs, source='CartoDB.Positron')
    ax.set_title(os.path.basename(fp))
    ax.set_axis_off()
    st.pyplot(fig, clear_figure=True)

# ========= FUNÇÕES DE ANÁLISE E MITIGAÇÃO =========


def _analyze_terrain_for_mitigation(dem: np.ndarray, flood_prob: np.ndarray, river_mask: Optional[np.ndarray] = None,
                                    prob_threshold: float = 0.7, min_slope: float = 0.01, cell_size_m: float = 1.0) -> tuple[np.ndarray, dict]:
    """
    Analisa o terreno para sugerir medidas de mitigação.
    Retorna:
    - intervention_mask: máscara com tipos de intervenção (1=florestação, 2=dique, 3=drenagem, 4=aterro)
    - suggestions: dicionário com análises detalhadas
    """
    H, W = dem.shape
    suggestions: Dict[str, Any] = {
        "florestamento": {"areas": [], "percentual": 0.0, "beneficio_estimado": 0.0},
        "diques": {"locais": [], "comprimento_estimado": 0.0, "areas_protegidas": []},
        "sistemas_drenagem": {"locais": [], "volume_estimado": 0.0},
        "aterro_terreno": {"areas": [], "volume_estimado": 0.0},
        "reservatorios": {"locais": [], "volume_estimado": 0.0}
    }

    intervention_mask = np.zeros_like(dem, dtype=np.uint8)
    try:
        dem_f = dem.astype(float)
        gy, gx = np.gradient(dem_f)
        # Normalizar declividade por tamanho de célula (m/m). Evita thresholds inconsistentes.
        cs = max(1e-9, float(cell_size_m))
        slope = np.hypot(gx, gy) / cs

        # Risco de inundação (booleano)
        prob_f = np.clip(flood_prob.astype(float), 0.0, 1.0)
        high_flood_risk = prob_f >= float(prob_threshold)
        # Médio risco: valores intermediários para alimentar drenagem
        medium_flood_risk = (prob_f >= 0.25) & (prob_f < float(prob_threshold))

        # 1) Florestamento
        gentle_slope = slope < float(min_slope)
        forest_candidates = high_flood_risk & gentle_slope
        if np.any(forest_candidates):
            labeled_forest, num_forest = cast(
                Tuple[np.ndarray, int], ndimage.label(forest_candidates))
            for i in range(1, num_forest + 1):
                area_mask = labeled_forest == i
                area_size = int(np.sum(area_mask))
                if area_size > 10:
                    y_coords, x_coords = np.where(area_mask)
                    centroid_y, centroid_x = float(
                        np.mean(y_coords)), float(np.mean(x_coords))
                    suggestions["florestamento"]["areas"].append({
                        "centroide": (centroid_x, centroid_y),
                        "tamanho_pixels": area_size,
                        "beneficio_estimado": float(min(1.0, area_size / (H * W) * 10))
                    })
                    intervention_mask[area_mask] = 1
        suggestions["florestamento"]["percentual"] = float(
            np.sum(forest_candidates)) / float(H * W) * 100.0

        # 2) Diques próximos ao rio
        if river_mask is not None and np.any(river_mask):
            structure = np.ones((7, 7), dtype=bool)
            river_bool = river_mask.astype(bool)
            river_buffer = ndimage.binary_dilation(
                river_bool, structure=structure).astype(bool)
            high_risk_near_river = (
                high_flood_risk.astype(bool) & river_buffer & (~river_bool))
            if np.any(high_risk_near_river):
                labeled_dikes, num_dikes = cast(
                    Tuple[np.ndarray, int], ndimage.label(high_risk_near_river))
                total_length = 0.0
                for i in range(1, num_dikes + 1):
                    dike_mask = labeled_dikes == i
                    if int(np.sum(dike_mask)) > 5:
                        y_coords, x_coords = np.where(dike_mask)
                        if len(y_coords) > 1:
                            coords = np.column_stack([x_coords, y_coords])
                            clustering = DBSCAN(
                                eps=2, min_samples=3).fit(coords)
                            for cluster_id in set(clustering.labels_):
                                if cluster_id == -1:
                                    continue
                                cluster_coords = coords[clustering.labels_ == cluster_id]
                                if len(cluster_coords) > 2:
                                    length = float(len(cluster_coords)) * 0.5
                                    total_length += length
                                    centroid = np.mean(cluster_coords, axis=0)
                                    suggestions["diques"]["locais"].append({
                                        "centroide": (float(centroid[0]), float(centroid[1])),
                                        "comprimento_estimado": length,
                                        "areas_protegidas": int(len(cluster_coords))
                                    })
                                    intervention_mask[dike_mask] = 2
                suggestions["diques"]["comprimento_estimado"] = total_length

        # 3) Sistemas de drenagem
        local_min = ndimage.minimum_filter(dem_f, size=5) == dem_f
        drainage_candidates = local_min & medium_flood_risk
        if np.any(drainage_candidates):
            volume_total = 0.0
            labeled_drainage, num_drainage = cast(
                Tuple[np.ndarray, int], ndimage.label(drainage_candidates))
            for i in range(1, num_drainage + 1):
                drain_mask = labeled_drainage == i
                area_size = int(np.sum(drain_mask))
                if area_size > 3:
                    depression_depth = float(np.percentile(
                        dem_f[drain_mask], 25) - np.min(dem_f[drain_mask]))
                    volume = area_size * max(0.1, depression_depth)
                    volume_total += volume
                    y_coords, x_coords = np.where(drain_mask)
                    centroid_y, centroid_x = float(
                        np.mean(y_coords)), float(np.mean(x_coords))
                    suggestions["sistemas_drenagem"]["locais"].append({
                        "centroide": (centroid_x, centroid_y),
                        "volume_estimado": float(volume),
                        "area_pixels": area_size
                    })
                    intervention_mask[drain_mask] = 3
            suggestions["sistemas_drenagem"]["volume_estimado"] = float(
                volume_total)

        # 4) Aterro em áreas baixas críticas
        valid = dem_f[np.isfinite(dem_f)]
        p25 = float(np.percentile(valid, 25)) if valid.size > 0 else (
            float(np.nanmin(dem_f)) if np.isfinite(dem_f).any() else 0.0)
        low_areas = dem_f < p25
        critical_low_areas = low_areas & high_flood_risk
        if np.any(critical_low_areas):
            volume_total = 0.0
            labeled_fill, num_fill = cast(
                Tuple[np.ndarray, int], ndimage.label(critical_low_areas))
            for i in range(1, num_fill + 1):
                fill_mask = labeled_fill == i
                area_size = int(np.sum(fill_mask))
                if area_size > 5:
                    vals = dem_f[fill_mask]
                    fill_height = float(
                        np.percentile(vals, 75) - np.mean(vals))
                    volume = area_size * max(0.5, fill_height)
                    volume_total += volume
                    y_coords, x_coords = np.where(fill_mask)
                    centroid_y, centroid_x = float(
                        np.mean(y_coords)), float(np.mean(x_coords))
                    suggestions["aterro_terreno"]["areas"].append({
                        "centroide": (centroid_x, centroid_y),
                        "volume_necessario": float(volume),
                        "area_pixels": area_size,
                        "altura_media_aterro": float(fill_height)
                    })
                    intervention_mask[fill_mask] = 4
            suggestions["aterro_terreno"]["volume_estimado"] = float(
                volume_total)

        total_benefit = (
            suggestions["florestamento"]["percentual"] * 0.1 +
            len(suggestions["diques"]["locais"]) * 0.3 +
            len(suggestions["sistemas_drenagem"]["locais"]) * 0.2 +
            len(suggestions["aterro_terreno"]["areas"]) * 0.4
        )
        suggestions["beneficio_total_estimado"] = float(
            min(10.0, total_benefit))
    except Exception as e:
        print(f"Erro na análise de mitigação: {e}")
    return intervention_mask, suggestions


def _discover_icon_paths(icon_dir: Optional[str] = None) -> Dict[int, Optional[str]]:
    """Descobre caminhos de ícones por classe usando pastas conhecidas e um diretório opcional.
    Classes: 1=florestamento, 2=diques, 3=drenagem, 4=aterro.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates: list[str] = []
    if icon_dir:
        candidates.append(icon_dir)
    candidates.extend([
        os.path.join(base_dir, 'logos', 'icons'),
        os.path.join(base_dir, 'icons'),
        os.path.join(base_dir, 'logos'),
    ])

    def _find_icon(names: list[str]) -> Optional[str]:
        for d in candidates:
            for n in names:
                p = os.path.join(d, n)
                if os.path.exists(p):
                    return p
        return None

    icon_map: Dict[int, Optional[str]] = {
        1: _find_icon(['tree.png', 'florestamento.png', 'arvore.png']),
        2: _find_icon(['dike.png', 'diques.png', 'barragem.png']),
        3: _find_icon(['drainage.png', 'drenagem.png', 'sistemaDrenagem.png']),
        4: _find_icon(['fill.png', 'aterro.png', 'aterroNoTerreno.png']),
    }
    return icon_map


def _create_mitigation_map(dem: np.ndarray, intervention_mask: np.ndarray, suggestions: dict,
                           transform, crs, background_rgb: Optional[np.ndarray] = None,
                           use_icons: bool = False, icon_dir: Optional[str] = None, icon_size: int = 24):
    """Cria mapa visual com sugestões de mitigação."""
    bounds = array_bounds(dem.shape[0], dem.shape[1], transform)
    fig, ax = plt.subplots(figsize=(12, 10))
    # Base
    if background_rgb is not None and background_rgb.shape[:2] == dem.shape:
        img = (np.clip(background_rgb, 0, 1) * 255).astype(np.uint8)
        ax.imshow(img, extent=bounds, alpha=0.8)
    else:
        dem_f = np.asarray(dem, dtype=float)
        finite = np.isfinite(dem_f)
        if np.any(finite):
            vmin, vmax = np.nanpercentile(dem_f[finite], (5, 95))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = float(np.nanmin(dem_f[finite])), float(
                    np.nanmax(dem_f[finite]) + 1e-6)
            ax.imshow(dem_f, extent=bounds, cmap="terrain",
                      vmin=vmin, vmax=vmax, alpha=0.8)
        else:
            # Fallback: gradiente simples para evitar figura em branco
            grad = np.linspace(
                0, 1, dem_f.size, dtype=float).reshape(dem_f.shape)
            ax.imshow(grad, extent=bounds, cmap="Greys",
                      vmin=0, vmax=1, alpha=0.6)

    # Paleta e rótulos (RGBA 0..1)
    colors = {1: (0, 0.7, 0, 0.45), 2: (0.8, 0.2, 0.2, 0.5),
              3: (0, 0.5, 1, 0.45), 4: (0.9, 0.7, 0, 0.45)}
    labels = {1: "Florestamento/Vegetação", 2: "Diques/Proteção Ribeirinha",
              3: "Sistemas de Drenagem", 4: "Aterro/Elevação do Terreno"}

    # Overlay de cores por classe (mais leve e limpo que scatter por pixel)
    any_points = np.any(intervention_mask > 0)
    if any_points:
        H, W = intervention_mask.shape
        overlay = np.zeros((H, W, 4), dtype=float)
        for cls in [1, 2, 3, 4]:
            mk = intervention_mask == cls
            if np.any(mk):
                r, g, b, a = colors[cls]
                overlay[mk, 0] = r
                overlay[mk, 1] = g
                overlay[mk, 2] = b
                overlay[mk, 3] = a
        ax.imshow(overlay, extent=bounds, alpha=1.0)

    # Ícones (opcional) nas posições de centróides das sugestões
    def _try_icon(path: str, size_px: int) -> Optional[OffsetImage]:
        try:
            im = Image.open(path).convert('RGBA')
            im_arr = np.asarray(im)
            return OffsetImage(im_arr, zoom=max(0.1, size_px/64.0))
        except Exception:
            return None

    if use_icons and any_points:
        icon_map = _discover_icon_paths(icon_dir)

        # Preparar listas de centros por classe
        centers_by_cls: dict[int, list[tuple[float, float]]] = {
            1: [], 2: [], 3: [], 4: []}
        from rasterio.transform import xy as _xy
        # Florestamento
        for a in suggestions.get('florestamento', {}).get('areas', []):
            cx, cy = a.get('centroide', (None, None))
            if cx is None or cy is None:
                continue
            xw, yw = _xy(transform, [int(cy)], [int(cx)])
            centers_by_cls[1].append((xw[0], yw[0]))
        # Diques
        for d in suggestions.get('diques', {}).get('locais', []):
            cx, cy = d.get('centroide', (None, None))
            if cx is None or cy is None:
                continue
            xw, yw = _xy(transform, [int(cy)], [int(cx)])
            centers_by_cls[2].append((xw[0], yw[0]))
        # Drenagem
        for d in suggestions.get('sistemas_drenagem', {}).get('locais', []):
            cx, cy = d.get('centroide', (None, None))
            if cx is None or cy is None:
                continue
            xw, yw = _xy(transform, [int(cy)], [int(cx)])
            centers_by_cls[3].append((xw[0], yw[0]))
        # Aterro
        for a in suggestions.get('aterro_terreno', {}).get('areas', []):
            cx, cy = a.get('centroide', (None, None))
            if cx is None or cy is None:
                continue
            xw, yw = _xy(transform, [int(cy)], [int(cx)])
            centers_by_cls[4].append((xw[0], yw[0]))

        # Desenhar ícones
        for cls in [1, 2, 3, 4]:
            icon_path = icon_map.get(cls)
            if not icon_path:
                continue
            img = _try_icon(icon_path, int(icon_size))
            if img is None:
                continue
            for (xw, yw) in centers_by_cls.get(cls, []):
                ab = AnnotationBbox(
                    img, (xw, yw), frameon=False, pad=0.0, zorder=20)
                ax.add_artist(ab)

    with contextlib.suppress(Exception):
        ctx.add_basemap(ax, crs=crs, source='CartoDB.Positron', alpha=0.5)

    if any_points:
        # Legenda com patches de cor
        handles = [Patch(facecolor=colors[c][:3], alpha=colors[c]
                         [3], label=labels[c]) for c in [1, 2, 3, 4]]
        ax.legend(handles=handles, loc='upper right', fontsize=10)
    else:
        # Mensagem informativa na figura quando não há intervenções
        cx = (bounds[0] + bounds[2]) / 2.0
        cy = (bounds[1] + bounds[3]) / 2.0
        ax.text(cx, cy, "Sem áreas elegíveis com os parâmetros atuais",
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))
    ax.set_title("Mapa de Sugestões de Mitigação de Inundações",
                 fontsize=14, fontweight='bold')
    ax.set_axis_off()
    return fig


def _generate_mitigation_report(suggestions: dict, cell_size: float) -> str:
    """Gera relatório textual das sugestões."""
    report = []
    report.append("# 📋 RELATÓRIO DE SUGESTÕES DE MITIGAÇÃO")
    report.append("")
    benefit_score = float(suggestions.get("beneficio_total_estimado", 0))
    if benefit_score > 7:
        rating = " MUITO ALTO "
    elif benefit_score > 5:
        rating = " ALTO "
    elif benefit_score > 3:
        rating = " MÉDIO "
    else:
        rating = " BAIXO "
    report.append(f"## 📊 Benefício Esperado: {rating}")
    report.append("")

    def _section(title, lines):
        if lines:
            report.append(title)
            report.extend(lines)
            report.append("")

    forest = suggestions.get("florestamento", {})
    if forest.get("areas"):
        area_px = sum(a.get("tamanho_pixels", 0) for a in forest["areas"])
        area_km2 = (float(area_px) * cell_size * cell_size) / 1e6
        _section("## 🌳 FLORESTAMENTO E VEGETAÇÃO", [
            f"- **Área total recomendada:** {area_km2:.2f} km²",
            f"- **Número de áreas:** {len(forest['areas'])}",
            f"- **Benefício:** Aumenta infiltração, reduz escoamento superficial",
        ])

    dikes = suggestions.get("diques", {})
    if dikes.get("locais"):
        total_length = sum(d.get("comprimento_estimado", 0.0)
                           for d in dikes["locais"]) * cell_size
        _section("## 🏗️ DIQUES E PROTEÇÕES RIBEIRINHAS", [
            f"- **Comprimento total:** {total_length:.0f} m",
            f"- **Número de trechos:** {len(dikes['locais'])}",
            f"- **Benefício:** Proteção direta contra transbordamento de rios",
        ])

    drainage = suggestions.get("sistemas_drenagem", {})
    if drainage.get("locais"):
        total_volume = float(drainage.get(
            "volume_estimado", 0.0)) * cell_size * cell_size
        _section("## 💧 SISTEMAS DE DRENAGEM", [
            f"- **Volume de água a ser drenado:** {total_volume:.0f} m³",
            f"- **Número de depressões críticas:** {len(drainage['locais'])}",
            f"- **Benefício:** Eliminação de pontos de acumulação de água",
        ])

    fill = suggestions.get("aterro_terreno", {})
    if fill.get("areas"):
        total_volume = float(fill.get("volume_estimado", 0.0)
                             ) * cell_size * cell_size
        _section("## 🗻 ATERRO E ELEVAÇÃO DO TERRENO", [
            f"- **Volume total de terra:** {total_volume:.0f} m³",
            f"- **Número de áreas críticas:** {len(fill['areas'])}",
            f"- **Benefício:** Elevação de áreas baixas vulneráveis",
        ])

    report.append("## 💡 RECOMENDAÇÕES GERAIS")
    if forest.get("areas"):
        report.append(
            "- **Priorize o florestamento** em áreas planas com alto risco")
    if dikes.get("locais"):
        report.append(
            "- **Implemente diques** nos trechos críticos junto aos rios")
    if drainage.get("locais"):
        report.append(
            "- **Instale sistemas de drenagem** nas depressões identificadas")
    if fill.get("areas"):
        report.append("- **Execute aterros seletivos** nas áreas mais baixas")
    if not any([forest.get("areas"), dikes.get("locais"), drainage.get("locais"), fill.get("areas")]):
        report.append(
            "- **Situação favorável**: O terreno apresenta boa resiliência natural")
        report.append(
            "- **Ações preventivas**: Mantenha a vegetação existente e monitore áreas ribeirinhas")
    return "\n".join(report)


# ========= Funções auxiliares de IA =========

def _compute_ia_features(dem: np.ndarray) -> np.ndarray:
    """Gera atributos simples a partir do DEM: elevação normalizada e declividade aproximada."""
    H, W = dem.shape
    dem = dem.astype(float)
    # Normalização robusta por percentis
    if np.isfinite(dem).any():
        p2, p98 = np.nanpercentile(dem, (2, 98))
        denom = (p98 - p2) if (p98 - p2) != 0 else 1.0
        dem_norm = np.clip((dem - p2) / denom, 0, 1)
    else:
        dem_norm = np.zeros_like(dem, dtype=float)
    # Declividade aproximada
    gy, gx = np.gradient(np.nan_to_num(dem, nan=0.0))
    slope = np.hypot(gx, gy)
    if np.isfinite(slope).any():
        s_p2, s_p98 = np.nanpercentile(slope, (2, 98))
        s_denom = (s_p98 - s_p2) if (s_p98 - s_p2) != 0 else 1.0
        slope_norm = np.clip((slope - s_p2) / s_denom, 0, 1)
    else:
        slope_norm = np.zeros_like(slope, dtype=float)
    X = np.stack([dem_norm, slope_norm], axis=-1).reshape(H * W, 2)
    return X


def _train_ia_model(dem: np.ndarray, water: np.ndarray, threshold: float, n_estimators: int = 100, max_depth: int = 12) -> RandomForestClassifier:
    """Treina um RandomForest simples para classificar pixels inundados (water>threshold)."""
    X = _compute_ia_features(dem)
    y = (water.reshape(-1) > float(threshold)).astype(np.uint8)
    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X, y)
    return clf


def _predict_probability(model: RandomForestClassifier, dem: np.ndarray) -> np.ndarray:
    X = _compute_ia_features(dem)
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        p1 = np.zeros((dem.size,), dtype=float)
    else:
        p1 = proba[:, 1]
    return p1.reshape(dem.shape)


def _plot_probability_overlay(prob: np.ndarray, transform, crs, ia_threshold: float, ia_alpha: float, dem_back: Optional[np.ndarray] = None):
    """Plota um mapa com o DEM (se fornecido) e sobrepõe a probabilidade (IA) com transparência abaixo do limiar."""
    bounds = array_bounds(prob.shape[0], prob.shape[1], transform)
    fig, ax = plt.subplots(figsize=(10, 8))
    if dem_back is not None and np.isfinite(dem_back).any():
        vmin, vmax = np.nanpercentile(dem_back, (5, 95)) if np.isfinite(
            dem_back).any() else (0, 1)
        ax.imshow(dem_back, extent=bounds, cmap="terrain",
                  alpha=0.75, vmin=vmin, vmax=vmax)
    # Prob com máscara/under transparente
    reds = plt.get_cmap("Reds").copy()
    reds.set_under((0, 0, 0, 0.0))
    masked = np.ma.masked_less_equal(prob, ia_threshold)
    im = ax.imshow(masked, extent=bounds, cmap=reds, vmin=max(
        1e-6, ia_threshold + 1e-6), vmax=1.0, alpha=ia_alpha)
    with contextlib.suppress(Exception):
        ctx.add_basemap(ax, crs=crs, source='CartoDB.Positron')
    ax.set_title("Probabilidade de Inundação (IA)")
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, fraction=0.026, pad=0.02, label="Probabilidade")
    st.pyplot(fig, clear_figure=True)


def _probability_geotiff_bytes(prob: np.ndarray, transform, crs) -> bytes:
    """Gera um GeoTIFF em memória com a probabilidade (0..1)."""
    prob = prob.astype(np.float32)
    profile = {
        'driver': 'GTiff',
        'height': prob.shape[0],
        'width': prob.shape[1],
        'count': 1,
        'dtype': 'float32',
        'compress': 'deflate',
        'nodata': np.nan,
        'transform': transform,
        'crs': crs,
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(prob, 1)
        return memfile.read()


def _probability_rgba_geotiff_bytes(prob: np.ndarray, transform, crs, vmin: float = 0.0, vmax: float = 1.0, cmap_name: str = "Reds", under_transparent: bool = True) -> bytes:
    """Gera um GeoTIFF RGBA (uint8) com estilo aplicado (colormap) e transparência abaixo de vmin.

    - vmin: valores <= vmin ficam transparentes quando under_transparent=True.
    - vmax: topo do mapeamento de cores.
    - cmap_name: nome do colormap Matplotlib.
    """
    data = np.asarray(prob, dtype=float)
    H, W = data.shape
    # Normalização para 0..1 dentro do intervalo informado
    vmin_eff = max(1e-6, float(vmin))
    vmax_eff = max(vmin_eff + 1e-6, float(vmax))
    norm = mcolors.Normalize(vmin=vmin_eff, vmax=vmax_eff, clip=True)

    cmap = plt.get_cmap(cmap_name).copy()
    # Transparência para valores 'under' e mascarados
    if under_transparent:
        cmap.set_under((0, 0, 0, 0.0))
        cmap.set_bad((0, 0, 0, 0.0))
        masked = np.ma.masked_less_equal(data, vmin_eff)
        mapped = cmap(norm(masked))  # HxWx4 em float 0..1
    else:
        mapped = cmap(norm(data))

    rgba = (np.clip(mapped, 0, 1) * 255).astype(np.uint8)
    # Separar bandas
    r = rgba[:, :, 0]
    g = rgba[:, :, 1]
    b = rgba[:, :, 2]
    a = rgba[:, :, 3]

    profile = {
        'driver': 'GTiff',
        'height': H,
        'width': W,
        'count': 4,
        'dtype': 'uint8',
        'compress': 'deflate',
        'transform': transform,
        'crs': crs,
        # Mantemos photometric padrão; a banda 4 serve como alpha
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(r, 1)
            dst.write(g, 2)
            dst.write(b, 3)
            dst.write(a, 4)
        return memfile.read()


def _check_docker_available(timeout: float = 6.0) -> Tuple[bool, str]:
    """Verifica se o Docker está disponível no PATH e retorna (ok, detalhe)."""
    try:
        res = subprocess.run(["docker", "--version"], capture_output=True,
                             text=True, timeout=timeout, check=False)
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
        # type: ignore[attr-defined]
        except importlib_metadata.PackageNotFoundError:
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
    ET.SubElement(lfbinding, "textvar", name="CalendarConvention",
                  value=default_replacements["CalendarConvention"])
    ET.SubElement(lfbinding, "textvar", name="CalendarType",
                  value=default_replacements["CalendarConvention"])  # alias
    # Algumas versões usam tag <text> em vez de <textvar>
    ET.SubElement(lfbinding, "text", name="CalendarConvention",
                  value=default_replacements["CalendarConvention"])
    ET.SubElement(lfbinding, "text", name="CalendarType",
                  value=default_replacements["CalendarConvention"])  # alias
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
    # Se não houver fontes definidas (ou a simulação usou chuva uniforme), considere chuva sobre toda a área
    if np.any(sources_mask):
        rain_area_cells = np.sum(sources_mask > 0)
    else:
        rain_area_cells = sources_mask.size
    total_rain_m3 = float((total_rain/1000.0) *
                          rain_area_cells * (cell_size*cell_size))
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

    # 2. Downloads (gerar artefatos e persistir no session_state para sobreviver ao rerun)
    st.subheader("📥 Downloads")
    col1, col2, col3 = st.columns(3)

    # CSV com dados
    df = pd.DataFrame(model.history)
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.session_state["dl_history_csv"] = csv_data
    with col1:
        st.download_button(
            "📊 Dados (CSV)",
            csv_data,
            "dados_simulacao.csv",
            "text/csv",
            key="dl_csv_now",
        )

    # Animação
    anim_bytes = None
    anim_mime = None
    anim_ext = None
    if anim_path and os.path.exists(anim_path):
        try:
            with open(anim_path, "rb") as f:
                anim_bytes = f.read()
            anim_ext = anim_format.lower()
            anim_mime = f"video/{anim_ext}" if anim_ext == 'mp4' else "image/gif"
            st.session_state["dl_anim_bytes"] = anim_bytes
            st.session_state["dl_anim_ext"] = anim_ext
            st.session_state["dl_anim_mime"] = anim_mime
        except Exception as e:
            st.error(f"Erro ao carregar animação: {e}")
    with col2:
        if anim_bytes is not None:
            _ext_label = (anim_ext or "").upper(
            ) if isinstance(anim_ext, str) else ""
            st.download_button(
                f"🎬 Animação ({_ext_label})",
                anim_bytes,
                f"simulacao.{anim_ext}",
                anim_mime,
                key="dl_anim_now",
            )
        else:
            st.info("Ative 'Pré‑visualização rápida' DESMARCADA para salvar animação")

    # Gráficos
    graph_png_bytes = None
    if len(model.history) > 0:
        fig = _create_evolution_plots(model)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        graph_png_bytes = buf.getvalue()
        st.session_state["dl_graph_png"] = graph_png_bytes
    with col3:
        if graph_png_bytes is not None:
            st.download_button(
                "📈 Gráficos (PNG)",
                graph_png_bytes,
                "evolucao_simulacao.png",
                "image/png",
                key="dl_graph_now",
            )

    # Overlay PNG da água simulada (não IA) sobre DOM/DEM
    try:
        dem_last = st.session_state.get("last_dem_data")
        transform_last = st.session_state.get("last_transform")
        bg = st.session_state.get("last_background_rgb")
        if dem_last is not None and transform_last is not None:
            bounds_sim = array_bounds(
                dem_last.shape[0], dem_last.shape[1], transform_last)
            water = np.asarray(model.water_height, dtype=float)
            masked_sim = np.ma.masked_less_equal(water, float(
                max(1e-6, getattr(model, 'flood_threshold', 0.0))))
            fig_sim, ax_sim = plt.subplots(figsize=(10, 8))
            if bg is not None:
                img = (bg * 255).astype(np.uint8) if np.issubdtype(bg.dtype,
                                                                   np.floating) and bg.max() <= 1.0 else bg.astype(np.uint8)
                ax_sim.imshow(img, extent=bounds_sim, alpha=1.0)
            else:
                dem_b = dem_last.astype(float)
                vmin_b, vmax_b = np.nanpercentile(
                    dem_b, (5, 95)) if np.isfinite(dem_b).any() else (0, 1)
                ax_sim.imshow(dem_b, extent=bounds_sim, cmap="terrain",
                              vmin=vmin_b, vmax=vmax_b, alpha=0.85)
            # Usar o mesmo colormap da água e PowerNorm se disponível
            water_cmap = mcolors.LinearSegmentedColormap.from_list(
                "water_export",
                [
                    (0.70, 0.82, 1.00),
                    (0.18, 0.34, 0.85),
                    (0.05, 0.12, 0.45),
                    (0.00, 0.02, 0.18),
                ], N=256,
            )
            water_cmap.set_under((0, 0, 0, 0.0))
            vmin_w = float(
                max(1e-6, getattr(model, 'flood_threshold', 0.0) + 1e-6))
            vmax_w = float(np.nanmax(masked_sim)) if np.isfinite(
                masked_sim).any() else vmin_w + 1e-3
            try:
                norm = mcolors.PowerNorm(gamma=float(st.session_state.get(
                    "water_gamma", 0.7)), vmin=vmin_w, vmax=max(vmin_w + 1e-6, vmax_w))
                ax_sim.imshow(masked_sim, extent=bounds_sim, cmap=water_cmap, norm=norm, alpha=float(
                    st.session_state.get("water_alpha", 0.5)))
            except Exception:
                ax_sim.imshow(masked_sim, extent=bounds_sim, cmap=water_cmap, vmin=vmin_w, vmax=max(
                    vmin_w + 1e-6, vmax_w), alpha=float(st.session_state.get("water_alpha", 0.5)))
            ax_sim.set_axis_off()
            buf_sim = io.BytesIO()
            fig_sim.savefig(buf_sim, format='png', dpi=200,
                            bbox_inches='tight', pad_inches=0)
            plt.close(fig_sim)
            buf_sim.seek(0)
            overlay_sim_png = buf_sim.getvalue()
            st.session_state["dl_overlay_sim_png"] = overlay_sim_png
            with st.container():
                st.download_button(
                    "🖼️ PNG (DOM + água simulada)",
                    overlay_sim_png,
                    "overlay_dom_agua_simulada.png",
                    "image/png",
                    key="dl_overlay_sim_now",
                )
    except Exception as _e_sim_png:
        st.caption(f"(PNG de água simulada indisponível: {_e_sim_png})")

    # 3. Gráficos de evolução
    if len(model.history) > 0:
        st.subheader("📈 Evolução Temporal")
        fig_display = _create_evolution_plots(model)
        st.pyplot(fig_display)

    # 4. Painel persistente de downloads recentes (sobrevive ao rerun)
    with st.expander("📦 Downloads recentes (persistentes)"):
        cols = st.columns(3)
        with cols[0]:
            csv_buf = st.session_state.get("dl_history_csv")
            if csv_buf:
                st.download_button(
                    "📊 Dados (CSV)",
                    csv_buf,
                    "dados_simulacao.csv",
                    "text/csv",
                    key="dl_csv_persist",
                )
        with cols[1]:
            anim_buf = st.session_state.get("dl_anim_bytes")
            anim_ext = st.session_state.get("dl_anim_ext") or ""
            anim_mime = st.session_state.get(
                "dl_anim_mime") or "application/octet-stream"
            if anim_buf:
                st.download_button(
                    f"🎬 Animação ({str(anim_ext).upper()})",
                    anim_buf,
                    f"simulacao.{anim_ext}",
                    anim_mime,
                    key="dl_anim_persist",
                )
        with cols[2]:
            graph_buf = st.session_state.get("dl_graph_png")
            if graph_buf:
                st.download_button(
                    "📈 Gráficos (PNG)",
                    graph_buf,
                    "evolucao_simulacao.png",
                    "image/png",
                    key="dl_graph_persist",
                )
        # Linha adicional para overlay PNG
        cols2 = st.columns(3)
        with cols2[0]:
            overlay_buf = st.session_state.get("dl_overlay_png")
            if overlay_buf:
                st.download_button(
                    "🖼️ PNG (DOM + probabilidade)",
                    overlay_buf,
                    "overlay_dom_probabilidade.png",
                    "image/png",
                    key="dl_overlay_persist",
                )
        with cols2[1]:
            overlay_sim_buf = st.session_state.get("dl_overlay_sim_png")
            if overlay_sim_buf:
                st.download_button(
                    "🖼️ PNG (DOM + água simulada)",
                    overlay_sim_buf,
                    "overlay_dom_agua_simulada.png",
                    "image/png",
                    key="dl_overlay_sim_persist",
                )

# ========= FUNÇÕES LISFLOOD (SIMULADAS) =========
# Removidas: usando implementações reais via logica_lisflood


def main():
    # Configuração da página (título exatamente como solicitado e sem emoji/prefixo)
    st.set_page_config(
        page_title="Simulador hibrido de inundações",
        page_icon=os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "logos", "logo.png"),
        layout="wide",
        initial_sidebar_state="collapsed"
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

    # ========= ABAS PRINCIPAIS (apenas vetorizado e validação) =========
    tab_numpy, tab_validation = st.tabs([
        " Simulação Rápida", " Validação (IA)"
    ])

    with tab_numpy:
        st.header("Simulação Vetorizada")
        # ===== Telemetria Opik =====
        with st.expander("🧭 Telemetria (Opik)"):
            enable_telemetry = st.checkbox(
                "Ativar telemetria Opik",
                value=False,
                help="Registra eventos da simulação para análise."
            )
            colt1, colt2 = st.columns(2)
            with colt1:
                opik_api_key = st.text_input(
                    "API Key (opcional)", type="password")
                opik_workspace = st.text_input("Workspace (opcional)")
            with colt2:
                opik_url = st.text_input("Servidor Opik (opcional)")
                use_local = st.checkbox(
                    "Usar modo local", value=True, help="Sem servidor externo; guarda localmente.")
            # Status do Opik
            if opik is None:
                st.caption(
                    "Opik: ⚠️ não instalado/disponível no ambiente atual.")
            else:
                ver = getattr(opik, "__version__", "")
                st.caption(
                    f"Opik: ✅ disponível{(' - versão ' + ver) if ver else ''}.")

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
                animation_format = st.selectbox(
                    "Formato da animação", ["GIF", "MP4"], index=0)

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
                water_gamma = st.slider(
                    "Contraste da água (gamma)",
                    0.3, 2.0, 0.7, 0.05,
                    help="<1 realça diferenças em águas rasas (mais contraste); >1 suaviza."
                )

            # Persistir parâmetros visuais para uso em exportações
            st.session_state["water_alpha"] = float(water_alpha)
            st.session_state["water_gamma"] = float(water_gamma)

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

        # Normalizações/casts para tipos estáveis (evita alertas do Pylance)
        total_cycles_int = int(total_cycles)
        time_step_minutes_int = int(time_step_minutes)

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

        # ========== IA (na mesma aba) ==========
        with st.expander("🤖 IA (Beta) - Probabilidade de Inundação"):
            st.caption(
                "Treine um classificador simples a partir do resultado da simulação atual.")
            col_ia1, col_ia2 = st.columns(2)
            with col_ia1:
                ia_show = st.checkbox(
                    "Exibir mapa de probabilidade (IA)", value=False)
                ia_threshold = st.slider(
                    "Limiar de probabilidade para destacar", 0.0, 1.0, 0.5, 0.05)
                ia_alpha = st.slider(
                    "Opacidade da probabilidade", 0.05, 1.0, 0.6, 0.05)
            with col_ia2:
                ia_trees = st.slider(
                    "Nº de árvores (RandomForest)", 10, 300, 80, 10)
                ia_max_depth = st.slider("Profundidade máx.", 2, 30, 12, 1)
                ia_train = st.button(
                    "Treinar IA com simulação atual", type="secondary")

            st.caption(
                "Usa atributos do terreno (elevação normalizada, declividade aproximada) e rótulos de inundação da simulação.")

        # Botão de treino IA (usa última simulação concluída)
        if 'ia_model' not in st.session_state:
            st.session_state['ia_model'] = None
        if ia_train:
            dem_last = st.session_state.get('last_dem_data')
            water_last = st.session_state.get('last_water_height')
            if dem_last is None or water_last is None:
                st.warning("Rode uma simulação primeiro para treinar a IA.")
            else:
                with st.spinner("Treinando modelo IA (RandomForest)..."):
                    try:
                        clf = _train_ia_model(
                            dem_last, water_last, threshold=water_min_threshold, n_estimators=ia_trees, max_depth=ia_max_depth)
                        st.session_state['ia_model'] = clf
                        st.success("Modelo IA treinado com sucesso!")
                    except Exception as e:
                        st.error(f"Falha ao treinar IA: {e}")

        # Predição/overlay de probabilidade (em tempo real após treino)
        if ia_show and st.session_state.get('ia_model') is not None:
            try:
                dem_last = st.session_state.get('last_dem_data')
                transform_last = st.session_state.get('last_transform')
                crs_last = st.session_state.get('last_crs')
                if dem_last is None or transform_last is None or crs_last is None:
                    st.warning(
                        "Rode uma simulação primeiro para gerar o mapa de probabilidade.")
                else:
                    prob = _predict_probability(
                        st.session_state['ia_model'], dem_last)
                    # guardar para validação e download
                    st.session_state['ia_last_prob'] = prob
                    _plot_probability_overlay(
                        prob, transform_last, crs_last, ia_threshold, ia_alpha, dem_back=dem_last)
                    # botão de download do raster de probabilidade
                    try:
                        gtiff_bytes = _probability_geotiff_bytes(
                            prob, transform_last, crs_last)
                        st.download_button(
                            label="⬇️ Baixar probabilidade (GeoTIFF)",
                            data=gtiff_bytes,
                            file_name="prob_inundacao_ia.tif",
                            mime="image/tiff",
                            use_container_width=True,
                        )
                        # Versão estilizada (RGBA) com transparência abaixo do limiar para melhor visualização em QGIS/ArcGIS
                        rgba_bytes = _probability_rgba_geotiff_bytes(
                            prob, transform_last, crs_last,
                            vmin=max(1e-6, ia_threshold), vmax=1.0,
                            cmap_name="Reds", under_transparent=True,
                        )
                        st.download_button(
                            label="⬇️ Baixar probabilidade estilizada (GeoTIFF RGBA)",
                            data=rgba_bytes,
                            file_name="prob_inundacao_ia_rgba.tif",
                            mime="image/tiff",
                            use_container_width=True,
                        )
                        # PNG do overlay (DOM/DEM + probabilidade) para visualização direta
                        try:
                            bg = st.session_state.get("last_background_rgb")
                            bounds_png = array_bounds(
                                prob.shape[0], prob.shape[1], transform_last)
                            fig_png, ax_png = plt.subplots(figsize=(10, 8))
                            if bg is not None:
                                img = (bg * 255).astype(np.uint8) if np.issubdtype(bg.dtype,
                                                                                   np.floating) and bg.max() <= 1.0 else bg.astype(np.uint8)
                                ax_png.imshow(
                                    img, extent=bounds_png, alpha=1.0)
                            else:
                                dem_back = dem_last.astype(float)
                                vmin_b, vmax_b = np.nanpercentile(
                                    dem_back, (5, 95)) if np.isfinite(dem_back).any() else (0, 1)
                                ax_png.imshow(
                                    dem_back, extent=bounds_png, cmap="terrain", vmin=vmin_b, vmax=vmax_b, alpha=0.85)
                            reds = plt.get_cmap("Reds").copy()
                            reds.set_under((0, 0, 0, 0.0))
                            masked_png = np.ma.masked_less_equal(
                                prob, ia_threshold)
                            ax_png.imshow(masked_png, extent=bounds_png, cmap=reds, vmin=max(
                                1e-6, ia_threshold+1e-6), vmax=1.0, alpha=ia_alpha)
                            ax_png.set_axis_off()
                            buf_png = io.BytesIO()
                            fig_png.savefig(
                                buf_png, format='png', dpi=200, bbox_inches='tight', pad_inches=0)
                            plt.close(fig_png)
                            buf_png.seek(0)
                            overlay_png = buf_png.getvalue()
                            st.session_state["dl_overlay_png"] = overlay_png
                            st.download_button(
                                label="🖼️ Baixar PNG (DOM + probabilidade)",
                                data=overlay_png,
                                file_name="overlay_dom_probabilidade.png",
                                mime="image/png",
                                use_container_width=True,
                            )
                        except Exception as _e_png:
                            st.caption(
                                f"(PNG de overlay indisponível: {_e_png})")
                    except Exception as e:
                        st.warning(
                            f"Falha ao gerar GeoTIFF de probabilidade: {e}")
            except Exception as e:
                st.error(f"Falha ao gerar probabilidade: {e}")

    # ========= ABA: VALIDAÇÃO (IA) =========
    with tab_validation:
        st.header("📈 Validação da IA")
        st.caption(
            "Compara a probabilidade prevista (IA) com a inundação simulada. Gere a probabilidade na aba 'Simulação Rápida' primeiro.")
        dem_last = st.session_state.get('last_dem_data')
        water_last = st.session_state.get('last_water_height')
        prob_last = st.session_state.get('ia_last_prob')
        if any(x is None for x in [dem_last, water_last, prob_last]):
            st.info(
                "⚠️ Rode uma simulação, treine a IA e gere o mapa de probabilidade para habilitar a validação.")
        else:
            colv1, colv2 = st.columns(2)
            with colv1:
                label_threshold = st.slider(
                    "Limiar de água para rótulo positivo (m)", 0.0, 0.3, 0.01, 0.005,
                    help="Define o que é considerado inundado no rótulo (simulação)."
                )
            with colv2:
                st.caption("As curvas abaixo usam todos os pixels válidos.")

            water = np.asarray(water_last, dtype=float)
            y_true = (water.reshape(-1) >
                      float(label_threshold)).astype(np.uint8)
            y_score = np.asarray(prob_last, dtype=float).reshape(-1)
            valid = np.isfinite(y_true) & np.isfinite(y_score)
            if valid.sum() < 2:
                st.warning("Dados insuficientes para validar.")
            else:
                yt = y_true[valid]
                ys = y_score[valid]
                # ROC
                try:
                    fpr, tpr, _ = roc_curve(yt, ys)
                    auc_roc = roc_auc_score(yt, ys)
                except Exception as e:
                    fpr, tpr, auc_roc = np.array(
                        [0, 1]), np.array([0, 1]), float('nan')
                    st.warning(f"Falha ROC: {e}")
                # PR
                try:
                    prec, rec, _ = precision_recall_curve(yt, ys)
                    ap = average_precision_score(yt, ys)
                except Exception as e:
                    prec, rec, ap = np.array(
                        [1, 0]), np.array([0, 1]), float('nan')
                    st.warning(f"Falha PR: {e}")

                fig_val, axs = plt.subplots(1, 2, figsize=(12, 5))
                axs[0].plot(fpr, tpr, label=f"AUC = {auc_roc:.3f}")
                axs[0].plot([0, 1], [0, 1], 'k--', alpha=0.4)
                axs[0].set_title("Curva ROC")
                axs[0].set_xlabel("Falso Positivo (FPR)")
                axs[0].set_ylabel("Verdadeiro Positivo (TPR)")
                axs[0].grid(True, alpha=0.3)
                axs[0].legend()
                axs[1].plot(rec, prec, label=f"AP = {ap:.3f}")
                axs[1].set_title("Curva Precisão-Revocação (PR)")
                axs[1].set_xlabel("Revocação")
                axs[1].set_ylabel("Precisão")
                axs[1].grid(True, alpha=0.3)
                axs[1].legend()
                st.pyplot(fig_val, clear_figure=True)

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
                    dem_path, vector_path, tmp = _process_uploaded_files(
                        dem_file, vector_files)
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
                            rp = os.path.join(tmp or tempfile.mkdtemp(
                                prefix="sim_numpy_"), f"river_{f.name}")
                            with open(rp, "wb") as out:
                                out.write(f.getbuffer())
                            if f.name.lower().endswith(".gpkg"):
                                river_path = rp
                            elif f.name.lower().endswith(".shp") and river_path is None:
                                river_path = rp

                # Configurar dados geoespaciais
                assert dem_path is not None, "Caminho do DEM não pode ser None"
                gf = int(grid_reduction_factor[0]) if isinstance(
                    grid_reduction_factor, tuple) else int(grid_reduction_factor)
                dem_data, sources_mask, transform, crs, cell_size, river_mask = _setup_geodata(
                    dem_path, vector_path, gf, river_path
                )

                # Inicializar modelo
                model = GamaFloodModelNumpy(
                    dem_data, sources_mask, diffusion_rate,
                    flood_threshold, cell_size, river_mask
                )
                model.uniform_rain = (rain_mode == "Uniforme na área")

                # Fallback: se usuário escolheu "Somente nas fontes", mas não há fontes nem rio, aplicar chuva uniforme para evitar resultado em branco
                if (rain_mode != "Uniforme na área") and (not np.any(sources_mask)) and (not np.any(river_mask)):
                    st.info(
                        "Nenhuma fonte vetorial ou rio definidos. Aplicando chuva uniforme na área para evitar resultado vazio.")
                    model.uniform_rain = True

                # Preparar fundo visual
                background_rgb = None
                if dom_bg_file is not None and tmp:
                    dom_tmp = os.path.join(tmp, f"bg_{dom_bg_file.name}")
                    with open(dom_tmp, "wb") as out:
                        out.write(dom_bg_file.getbuffer())
                    background_rgb = _prepare_background(
                        dom_tmp, dem_data.shape, crs)
                # Guardar o fundo atual (DOM reamostrado) para futuras exportações
                st.session_state["last_background_rgb"] = background_rgb

                # Configurar visualização
                fig, _, water_layer, rain_particles, title, bounds = _setup_visualization(
                    dem_data, transform, crs, background_rgb, apply_hs, hs_intensity
                )
                x_min, y_min, x_max, y_max = bounds

                # Grelha nas coordenadas geográficas para alinhar contorno com o imshow (extent)
                xs = np.linspace(x_min, x_max, dem_data.shape[1])
                # inverter eixo Y (origin='upper')
                ys = np.linspace(y_max, y_min, dem_data.shape[0])
                Xw, Yw = np.meshgrid(xs, ys)

                # Coleção para contornos de água (atualizados a cada frame)
                water_contour_artists = []

                progress = st.progress(0, text="Inicializando...")

                # Função de atualização da animação
                @maybe_track(enable_telemetry and (opik is not None), name="numpy_update", type="general", tags=["sim:update"])
                def update(frame):
                    # Adicionar chuva
                    model.add_water(rain_mm_per_cycle)

                    # Executar passo de fluxo
                    model.run_flow_step()

                    # Atualizar estatísticas
                    model.update_stats(time_step_minutes_int)

                    # Atualizar visualização da água: somente onde acumula
                    water = model.water_height
                    masked = np.ma.masked_less_equal(
                        water, water_min_threshold)
                    # definir vmin ligeiramente acima do threshold para que valores abaixo fiquem 'under' (transparentes)
                    current_vmin = max(1e-9, float(water_min_threshold) + 1e-9)
                    masked = np.ma.masked_less_equal(
                        water, water_min_threshold)
                    # vmax dinâmico com base no valor máximo atual acima do limiar
                    if np.ma.is_masked(masked):
                        try:
                            wmax = float(np.nanmax(masked))
                        except ValueError:
                            wmax = current_vmin + 1e-3
                    else:
                        wmax = float(np.nanmax(masked)) if np.isfinite(
                            masked).any() else current_vmin + 1e-3
                    vmax_eff = max(current_vmin + 1e-6, wmax)
                    water_layer.set_clim(vmin=current_vmin, vmax=vmax_eff)
                    # Aplicar PowerNorm para aumentar contraste visual
                    try:
                        norm = mcolors.PowerNorm(gamma=float(
                            water_gamma), vmin=current_vmin, vmax=vmax_eff)
                        water_layer.set_norm(norm)
                    except Exception:
                        pass
                    water_layer.set_data(masked)
                    water_layer.set_alpha(water_alpha)

                    # Atualizar contorno das áreas com água (destacar limites)
                    # Remover contornos anteriores
                    try:
                        for artist in water_contour_artists:
                            artist.remove()
                    except Exception:
                        pass
                    water_contour_artists.clear()

                    # Desenhar novo contorno se houver água acima do threshold
                    if np.nanmax(water) > water_min_threshold:
                        ax_plot = water_layer.axes
                        cs = ax_plot.contour(
                            Xw, Yw, water,
                            levels=[water_min_threshold],
                            colors=[(0.0, 0.8, 1.0, 0.95)],  # ciano forte
                            linewidths=1.6,
                            zorder=11,
                        )
                        # Guardar artistas para remoção no próximo frame (compatível com type checker)
                        water_contour_artists.extend(
                            getattr(cs, 'collections', []))

                    # Partículas de chuva (efeito visual)
                    n = int(rain_mm_per_cycle * 150)
                    rx = np.random.uniform(x_min, x_max, n)
                    ry = np.random.uniform(y_min, y_max, n)
                    rain_particles.set_data(rx, ry)

                    # Atualizar título e métricas
                    h, m = divmod(model.simulation_time_minutes, 60)
                    title.set_text(
                        f"Simulação de Inundação | Tempo: {h}h {m}m")

                    # Atualizar métricas em tempo real
                    latest = model.history[-1]
                    if model.overflow_time_minutes is not None:
                        ho, mo = divmod(model.overflow_time_minutes, 60)
                        overflow_ph.metric(
                            "Tempo para Transbordar", f"{ho}h {mo}m")
                    time_ph.metric("Tempo de Simulação", f"{h}h {m}m")
                    flooded_ph.metric(
                        "Área Inundada", f"{latest['flooded_percent']:.2f}%")
                    vol_ph.metric("Volume de Água",
                                  f"{latest['total_water_volume_m3']:.2f} m³")

                    # Atualizar barra de progresso
                    progress.progress(
                        int(100 * (frame + 1) / max(1, total_cycles_int)),
                        text=f"Simulando ciclo {frame + 1}/{total_cycles_int}"
                    )

                    return [water_layer, rain_particles, title]

                # Executar simulação
                if quick_preview:
                    # Modo pré-visualização: executar sem salvar animação
                    for frame in range(total_cycles_int):
                        update(frame)
                    # Mostrar resultado final
                    anim_area.pyplot(fig, clear_figure=False)
                else:
                    # Modo completo: gerar animação
                    fps = max(1, total_cycles_int //
                              max(1, int(animation_duration)))
                    interval = 1000 // fps  # ms entre frames

                    anim = FuncAnimation(
                        fig, update, frames=total_cycles_int,
                        blit=True, interval=interval
                    )

                    # Salvar animação
                    ext = str(animation_format).lower()
                    tmp_anim = os.path.join(
                        tmp or tempfile.gettempdir(), f"simulation.{ext}")
                    try:
                        if ext == 'gif':
                            # GIF via Pillow sempre disponível
                            anim.save(tmp_anim, writer='pillow',
                                      dpi=150, fps=fps)
                        else:
                            # MP4: garantir ffmpeg disponível via imageio-ffmpeg
                            ffmpeg_bin = None
                            try:
                                import imageio_ffmpeg  # type: ignore
                                ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
                            except (ImportError, OSError):
                                # Tentar instalar imageio-ffmpeg on-the-fly
                                subprocess.run([sys.executable, '-m', 'pip', 'install',
                                               'imageio-ffmpeg', '--quiet'], capture_output=True, check=False)
                                # Importar após tentativa de instalação
                                try:
                                    import importlib as _im
                                    imageio_ffmpeg = _im.import_module(
                                        'imageio_ffmpeg')  # type: ignore
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
                                    extra_args=['-vcodec', 'libx264',
                                                '-pix_fmt', 'yuv420p']
                                )
                        # Telemetria do salvamento
                        maybe_track(enable_telemetry and (
                            opik is not None), name="save_animation", type="general", tags=[ext.upper()])(lambda: None)()
                    except (RuntimeError, ValueError, OSError) as e:
                        # Fallback para GIF
                        st.warning(
                            f"Falha ao salvar em {ext.upper()} ({e}). Tentando GIF...")
                        tmp_anim = os.path.join(
                            tmp or tempfile.gettempdir(), "simulation.gif")
                        anim.save(tmp_anim, writer='pillow', dpi=150, fps=fps)
                        ext = 'gif'
                        tmp_anim = os.path.join(
                            tmp or tempfile.gettempdir(), "simulation.gif")
                        anim.save(tmp_anim, writer='pillow', dpi=150, fps=fps)
                        ext = 'gif'

                    # Exibir animação
                    with open(tmp_anim, "rb") as f:
                        if ext == 'gif':
                            anim_area.image(f.read())
                        else:
                            anim_area.video(f.read())

                # Salvar dados para IA (estado da sessão)
                st.session_state["last_dem_data"] = dem_data
                st.session_state["last_transform"] = transform
                st.session_state["last_crs"] = crs
                st.session_state["last_water_height"] = model.water_height.copy()
                st.session_state["last_river_mask"] = river_mask
                st.session_state["last_cell_size"] = float(cell_size)
                st.session_state["last_flood_threshold"] = float(
                    flood_threshold)

                # Pós-processamento e downloads
                total_rain_mm = float(rain_mm_per_cycle) * \
                    float(total_cycles_int)
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
                    "total_cycles": int(total_cycles_int),
                    "rain_mm_per_cycle": float(rain_mm_per_cycle),
                })(lambda: None)()

            except (RuntimeError, ValueError, OSError) as e:
                st.error(f"Erro na simulação: {e}")
                import traceback
                st.error(traceback.format_exc())
                maybe_track(enable_telemetry and (
                    opik is not None), name="simulation_error", type="general", tags=["error"])(lambda: None)()
            finally:
                # Limpeza
                if tmp and os.path.exists(tmp):
                    shutil.rmtree(tmp, ignore_errors=True)
                plt.close('all')

        # ========= SEÇÃO DE MITIGAÇÃO =========
        with st.expander("🛡️ Mitigação e Intervenções (Beta)"):
            st.caption(
                "Sugestões baseadas em IA para reduzir riscos de inundação")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                mit_threshold = st.slider(
                    "Limiar de risco para intervenções",
                    0.1, 1.0, 0.45, 0.05,
                    help="Probabilidade mínima (IA) para considerar área de risco"
                )
                mit_min_slope = st.slider(
                    "Declividade máxima para florestamento",
                    0.001, 0.2, 0.05, 0.005,
                    help="Áreas mais planas favorecem florestamento/vegetação"
                )
            with col_m2:
                show_mitigation = st.checkbox(
                    "Mostrar mapa de intervenções", value=False)
                generate_report = st.button(
                    "📄 Gerar Relatório de Mitigação", type="secondary")
                use_icons = st.checkbox("Usar ícones nas intervenções", value=False,
                                        help="Sobrepõe pequenos ícones (árvore, dique, drenagem, aterro) sobre as áreas sugeridas")
                icon_size = st.slider("Tamanho dos ícones (px)", 12, 64, 24, 2)
                icon_dir = st.text_input("Pasta de ícones (opcional)", value="",
                                         help="Coloque seus arquivos PNG (tree.png, dike.png, drainage.png, fill.png). Se vazio, buscarei em ./logos/icons, ./icons e ./logos")

            if st.button("🔍 Analisar Terreno para Mitigação", type="primary"):
                dem_last = st.session_state.get('last_dem_data')
                prob_last = st.session_state.get('ia_last_prob')
                river_last = st.session_state.get('last_river_mask')
                transform_last = st.session_state.get('last_transform')
                crs_last = st.session_state.get('last_crs')
                bg_last = st.session_state.get('last_background_rgb')
                cell_size = float(st.session_state.get(
                    'last_cell_size') or 10.0)

                if dem_last is None:
                    st.warning("Execute uma simulação primeiro.")
                else:
                    # Se a probabilidade IA ainda não foi gerada, mas há modelo treinado, calcule agora
                    if prob_last is None and st.session_state.get('ia_model') is not None:
                        try:
                            prob_last = _predict_probability(
                                st.session_state['ia_model'], dem_last)
                            st.session_state['ia_last_prob'] = prob_last
                        except Exception as _e_pred:
                            st.warning(
                                f"Falha ao gerar probabilidade automaticamente: {_e_pred}")

                    if prob_last is None:
                        # Fallback: estimar probabilidade a partir da lâmina d'água simulada
                        water_last = st.session_state.get('last_water_height')
                        ft = float(st.session_state.get(
                            'last_flood_threshold') or 0.1)
                        if water_last is not None:
                            w = np.asarray(water_last, dtype=float)
                            if np.isfinite(w).any():
                                wmax = float(np.nanmax(w))
                                if wmax > 0:
                                    eps = 1e-6
                                    prob_last = np.clip(
                                        (w - ft) / max(eps, (wmax - ft)), 0.0, 1.0)
                                    st.info(
                                        "Usando probabilidade aproximada derivada da lâmina d'água simulada.")
                                    st.session_state['ia_last_prob'] = prob_last
                        if prob_last is None:
                            st.warning(
                                "Gere o mapa de probabilidade na seção de IA primeiro ou aumente a chuva/ciclos para obter lâmina d'água.")
                    if prob_last is not None:
                        with st.spinner("Analisando terreno e gerando sugestões..."):
                            try:
                                intervention_mask, suggestions = _analyze_terrain_for_mitigation(
                                    np.asarray(dem_last), np.asarray(
                                        prob_last),
                                    river_last, mit_threshold, mit_min_slope, cell_size
                                )
                                st.session_state['mitigation_data'] = {
                                    'intervention_mask': intervention_mask,
                                    'suggestions': suggestions,
                                    'cell_size': cell_size,
                                    'transform': transform_last,
                                    'crs': crs_last,
                                    'background': bg_last,
                                }
                                st.success("Análise de mitigação concluída!")
                            except Exception as e:
                                st.error(f"Erro na análise: {e}")

            mitigation_data = st.session_state.get('mitigation_data')
            if show_mitigation and mitigation_data:
                dem_last = st.session_state.get('last_dem_data')
                transform_last = mitigation_data.get('transform')
                crs_last = mitigation_data.get('crs')
                bg_last = mitigation_data.get('background')
                if dem_last is not None and transform_last is not None:
                    fig_mit = _create_mitigation_map(
                        np.asarray(dem_last),
                        mitigation_data['intervention_mask'],
                        mitigation_data['suggestions'],
                        transform_last, crs_last, bg_last,
                        use_icons=bool(use_icons),
                        icon_dir=(icon_dir or None),
                        icon_size=int(icon_size)
                    )
                    st.pyplot(fig_mit, clear_figure=True)
                    if not np.any(mitigation_data['intervention_mask']):
                        st.info(
                            "Nenhuma intervenção foi identificada com os parâmetros atuais. Ajuste o limiar de risco ou a declividade máxima e tente novamente.")
                    buf = io.BytesIO()
                    fig_mit.savefig(buf, format='png', dpi=300,
                                    bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        "⬇️ Baixar Mapa de Intervenções (PNG)",
                        buf.getvalue(),
                        "mapa_intervencoes_mitigacao.png",
                        "image/png",
                        use_container_width=True,
                    )

            if generate_report and mitigation_data:
                report_text = _generate_mitigation_report(
                    mitigation_data['suggestions'], float(
                        mitigation_data.get('cell_size') or 10.0)
                )
                st.markdown("---")
                st.markdown("### 📋 Relatório de Mitigação")
                st.markdown(report_text)
                st.download_button(
                    "⬇️ Baixar Relatório (TXT)",
                    report_text.encode('utf-8'),
                    "relatorio_mitigacao.txt",
                    "text/plain",
                    use_container_width=True,
                )
                import json as _json
                suggestions_json = _json.dumps(
                    mitigation_data['suggestions'], indent=2, ensure_ascii=False)
                st.download_button(
                    "⬇️ Baixar Dados das Intervenções (JSON)",
                    suggestions_json.encode('utf-8'),
                    "dados_intervencoes.json",
                    "application/json",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()


# NOTA: Sugestões prontas (todas totalizam 174 mm em 24 h):

# Ciclo de 5 min: 288 ciclos; 0,604 mm por ciclo
# Ciclo de 10 min: 144 ciclos; 1,208 mm por ciclo
# Ciclo de 15 min: 96 ciclos; 1,8125 mm por ciclo
# Ciclo de 30 min: 48 ciclos; 3,625 mm por ciclo
# Ciclo de 60 min: 24 ciclos; 7,25 mm por ciclo
