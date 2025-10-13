import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
from typing import Optional, List, Tuple
import time
import os

# ==========================
# Configuração da página
# ==========================
st.set_page_config(
    page_title="Simulador Híbrido de Inundações", 
    page_icon="💧", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Esconde completamente a sidebar e aplica tema do design_numpy
st.markdown(
    """
    <style>
    /* Ocultar sidebar */
    [data-testid="stSidebar"] { display: none !important; }
    
    /* Tema do Simulador Híbrido */
    .main > div { 
        padding-left: 0 !important; 
        padding-right: 0 !important;
        background-color: #f5f7fa;
    }
    
    .block-container { 
        max-width: 1400px; 
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Cores do design_numpy */
    h1, h2, h3 { 
        color: #1e5799 !important; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    h1 {
        font-size: 28px !important;
        margin-bottom: 10px !important;
    }
    
    h2 {
        font-size: 20px !important;
        margin-bottom: 15px !important;
        border-bottom: 1px solid #eaeaea;
        padding-bottom: 8px;
    }
    
    h3 {
        font-size: 18px !important;
        margin: 15px 0 10px !important;
        color: #207cca !important;
    }
    
    /* Botões no estilo do design */
    .stButton>button {
        border-radius: 5px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        border: none !important;
    }
    
    .stButton>button[kind="primary"] { 
        background: #1e5799 !important;
        color: white !important;
    }
    
    .stButton>button[kind="primary"]:hover { 
        background: #16457a !important;
    }
    
    .stButton>button[kind="secondary"] { 
        background: #6c757d !important;
        color: white !important;
    }
    
    .stButton>button[kind="secondary"]:hover { 
        background: #545b62 !important;
    }
    
    /* Containers no estilo do design */
    .section-box { 
        border: 1px solid #eaeaea !important; 
        border-radius: 8px !important; 
        padding: 20px 30px !important; 
        background: #ffffff !important;
        margin-bottom: 20px !important;
    }
    
    /* Checkboxes no estilo do design */
    .stCheckbox > label {
        font-weight: normal !important;
    }
    
    /* Sliders e inputs */
    .stSlider > div > div {
        color: #1e5799 !important;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 4px !important;
    }
    
    /* Métricas em tempo real */
    .metric-container {
        background: white;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #eaeaea;
        margin-bottom: 15px;
    }
    
    /* Status bar */
    .status-bar {
        background-color: #e8f4fd;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 10px 0;
        font-size: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .status-ready { background-color: #28a745; }
    .status-running { background-color: #ffc107; }
    .status-error { background-color: #dc3545; }
    
    /* Tabelas de métricas */
    .metric-table {
        width: 100%;
        border-collapse: collapse;
        background-color: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    
    .metric-table th {
        background-color: #eef5ff;
        padding: 12px 15px;
        text-align: left;
        font-weight: 600;
        color: #1e5799;
        border-bottom: 1px solid #ddd;
    }
    
    .metric-table td {
        padding: 12px 15px;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .highlight {
        background-color: #fff9e6 !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================
# Utilitários de I/O simples
# ==========================
def load_dem_from_upload(file) -> Optional[np.ndarray]:
    """Carrega um DEM a partir de CSV, NPY, PNG ou JPG."""
    if file is None:
        return None

    name = file.name.lower()
    data = file.read()

    if name.endswith(".npy"):
        return np.load(BytesIO(data))
    elif name.endswith(".csv"):
        text = data.decode("utf-8")
        try:
            arr = np.loadtxt(BytesIO(text.encode("utf-8")), delimiter=",")
        except Exception:
            arr = np.loadtxt(BytesIO(text.encode("utf-8")))
        return arr.astype(float)
    elif name.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(BytesIO(data)).convert("L")
        return np.asarray(img, dtype=float)
    else:
        raise ValueError("Formato não suportado. Use .csv, .npy, .png ou .jpg")

def load_mask_from_upload(file, shape) -> np.ndarray:
    """Carrega uma máscara binária nas mesmas dimensões do DEM."""
    if file is None:
        return np.zeros(shape, dtype=bool)

    arr_opt = load_dem_from_upload(file)
    if arr_opt is None:
        return np.zeros(shape, dtype=bool)
    arr = (arr_opt > 0).astype(bool)
    if arr.shape != shape:
        arr = resize_nearest(arr.astype(float), shape) > 0.5
    return arr

def resize_nearest(arr: np.ndarray, new_shape) -> np.ndarray:
    """Redimensiona uma matriz 2D via amostragem por vizinho mais próximo."""
    h, w = arr.shape
    nh, nw = new_shape
    y_idx = (np.linspace(0, h - 1, nh)).astype(int)
    x_idx = (np.linspace(0, w - 1, nw)).astype(int)
    return arr[y_idx][:, x_idx]

def downscale_if_needed(arr: np.ndarray, max_size: int) -> np.ndarray:
    """Reduz matriz (mantendo proporção) para no máximo max_size no maior lado."""
    h, w = arr.shape
    m = max(h, w)
    if m <= max_size:
        return arr
    scale = m / max_size
    step = int(np.ceil(scale))
    return arr[::step, ::step]

def normalize_minmax(a: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normaliza array para range [0, 1]."""
    amin, amax = float(np.nanmin(a)), float(np.nanmax(a))
    if amax - amin < eps:
        return np.zeros_like(a, dtype=float)
    return (a - amin) / (amax - amin)

def compose_frame_adv(
    dem: np.ndarray,
    water: np.ndarray,
    water_max_display: float,
    dom_rgb: Optional[np.ndarray],
    dem_opacity: float,
    relief_intensity: float,
    river_mask: Optional[np.ndarray],
    river_color: Tuple[float, float, float] = (0.0, 0.2, 1.0),
    river_opacity: float = 0.7,
    water_opacity: float = 0.7,
    water_min_threshold: float = 0.0,
) -> np.ndarray:
    """Gera frame RGB (uint8) combinando DOM, relevo sombreado e água."""
    H, W = dem.shape
    
    # Base: DOM ou preto
    if dom_rgb is not None and dom_rgb.shape[:2] == (H, W):
        base = dom_rgb.copy()
    else:
        base = np.zeros((H, W, 3), dtype=float)

    # Relevo sombreado
    dem_n = normalize_minmax(dem)
    gamma = 1.0 / max(1e-6, relief_intensity)
    dem_shade = np.power(dem_n, gamma)
    dem_rgb = np.stack([dem_shade, dem_shade, dem_shade], axis=2)
    if dem_opacity > 0:
        a = np.clip(dem_opacity, 0.0, 1.0)
        base = base * (1 - a) + dem_rgb * a

    # Rio (overlay)
    if river_mask is not None and river_mask.shape == (H, W):
        rc = np.array(river_color, dtype=float).reshape(1, 1, 3)
        a = np.clip(river_opacity, 0.0, 1.0)
        river_overlay = np.zeros_like(base)
        river_overlay[...] = rc
        mask3 = np.repeat(river_mask[..., None], 3, axis=2)
        base = np.where(mask3, base * (1 - a) + river_overlay * a, base)

    # Água (overlay azul)
    water_disp = np.clip((water - water_min_threshold) / max(1e-6, water_max_display), 0.0, 1.0)
    a = np.clip(water_opacity, 0.0, 1.0) * water_disp
    a3 = np.repeat(a[..., None], 3, axis=2)
    water_rgb = np.zeros_like(base)
    water_rgb[..., 2] = 1.0
    out = base * (1 - a3) + water_rgb * a3
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)

# ==========================
# Cabeçalho do Simulador
# ==========================
def _load_logos() -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """Carrega logos se disponíveis."""
    left, right = None, None
    logos_dir = os.path.join(os.getcwd(), "logos")
    if os.path.isdir(logos_dir):
        imgs = [f for f in os.listdir(logos_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".svg"))]
        imgs.sort()
        if imgs:
            try:
                left = Image.open(os.path.join(logos_dir, imgs[0]))
            except Exception:
                left = None
        if len(imgs) > 1:
            try:
                right = Image.open(os.path.join(logos_dir, imgs[-1]))
            except Exception:
                right = None
    return left, right

# Header com gradiente similar ao design_numpy
st.markdown(
    """
    <div style='
        background: linear-gradient(135deg, #1e5799 0%, #207cca 100%);
        color: white;
        padding: 25px 30px;
        border-radius: 0;
        margin-bottom: 20px;
    '>
    """,
    unsafe_allow_html=True
)

logo_left, logo_right = _load_logos()
colh1, colh2, colh3 = st.columns([1, 6, 1])
with colh1:
    if logo_left:
        st.image(logo_left, use_container_width=True)
with colh2:
    st.markdown("<h1 style='text-align:center; color:white !important; margin-bottom: 10px;'>Simulador Híbrido de Inundações</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; color:white; font-size: 20px;'>Simulação Vectorizada (NumPy)</div>", unsafe_allow_html=True)
    
    # Checkboxes no estilo do design
    col_cb1, col_cb2, col_cb3 = st.columns(3)
    with col_cb1:
        st.checkbox("Elemento (DIA)", value=True, key="elemento_dia")
    with col_cb2:
        st.checkbox("Texto", value=False, key="texto1")
    with col_cb3:
        st.checkbox("Texto", value=False, key="texto2")

with colh3:
    if logo_right:
        st.image(logo_right, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# Linha divisória
st.markdown("<hr style='border: none; border-top: 1px dashed #ccc; margin: 20px 0;'>", unsafe_allow_html=True)

# ==========================
# Seção: Dados de Entrada
# ==========================
st.markdown("<h2>Dados de Entrada</h2>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    
    st.markdown("<h3>Aquinas e Reguladas</h3>", unsafe_allow_html=True)
    st.markdown("<p><strong>Noutra Região de Emenda: 2014</strong></p>", unsafe_allow_html=True)
    
    col_input1, col_input2, col_input3 = st.columns([2, 2, 1])
    
    with col_input1:
        dem_file = st.file_uploader(
            "Matriz do Terreno (DEM)", 
            type=["csv", "npy", "png", "jpg", "jpeg"],
            help="Carregue DEM em CSV, NPY, PNG ou JPG"
        )
        gerar_sintetico = st.checkbox("Gerar DEM sintético", value=(dem_file is None))
        
    with col_input2:
        max_dim = st.slider("Tamanho máximo do grid (px)", 64, 300, 160)
        if gerar_sintetico:
            tipo = st.selectbox("Tipo de DEM sintético", 
                              ["plano_inclinado", "colinas_gaussianas", "vale_canal"], 
                              index=1)
            size = st.slider("Tamanho (N x N)", 64, 300, 160)
            seed = st.number_input("Semente", 0, 10_000, 42)
        else:
            tipo, size, seed = None, None, 0
            
    with col_input3:
        st.markdown("""
        <div style='background: #f9f9f9; padding: 15px; border-radius: 8px; margin-top: 10px;'>
            <ul style='list-style-type: none; padding-left: 10px;'>
                <li style='margin-bottom: 8px; position: relative; padding-left: 15px;'>
                    <span style='position: absolute; left: 0; color: #207cca;'>•</span>
                    Doça para doze Dia Anos
                </li>
                <li style='margin-bottom: 8px; position: relative; padding-left: 15px;'>
                    <span style='position: absolute; left: 0; color: #207cca;'>•</span>
                    Lunes 2020 por dia no TCP 7/8
                </li>
                <li style='margin-bottom: 8px; position: relative; padding-left: 15px;'>
                    <span style='position: absolute; left: 0; color: #207cca;'>•</span>
                    BRAÇAIS_XXXIAT_1.2.100
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# Seção: Parâmetros da Simulação
# ==========================
st.markdown("<h2>Parâmetros da Simulação</h2>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    
    col_params1, col_params2, col_params3 = st.columns(3)
    
    with col_params1:
        st.markdown("**Prestupidez | Comportamento da Água**")
        st.text("O mesmo estilo para | Varela alusada")
        escap_value = st.number_input("ESCAP", 0.0, 1.0, 0.5, 0.1, key="escap")
        
    with col_params2:
        st.markdown("**Testa de dados | Listas de inovação (Tempo)**")
        testa_dados = st.number_input("Testa de dados", 0, 100, 0, key="testa_dados")
        listas_inovacao = st.number_input("Listas de inovação", 0.0, 1.0, 0.5, 0.1, key="listas_inovacao")
        
    with col_params3:
        st.markdown("**Oração de usina este (Invanto) | Nota de chave**")
        oracao_usina = st.number_input("Oração de usina", 0, 100, 30, key="oracao_usina")
        st.text("Utilizamos os anos")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# Seção: Configurações Avançadas
# ==========================
st.markdown("<h2>Configurações da Simulação</h2>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    
    col_sim1, col_sim2, col_sim3 = st.columns(3)
    
    with col_sim1:
        modo = st.selectbox("Modo de adição de água", 
                          ["CHUVA_UNIFORME", "FONTES_MASCARA", "FONTES_PONTUAIS"], 
                          index=0)
        chuva_mm = st.number_input("Precipitação por ciclo (mm)", 0.0, 100.0, 5.0, 0.5)
        comportamento = st.selectbox("Comportamento da água", [
            "Distribuição proporcional (4 vizinhos)",
            "Sem escoamento (acumulação)",
        ], index=0)
        
    with col_sim2:
        steps = st.slider("Total de ciclos", 10, 1000, 200)
        intervalo_ms = st.slider("Duração de cada ciclo (ms)", 0, 2000, 100)
        water_max_display = st.number_input("Prof. máx. visualização (m)", 0.001, 2.0, 0.1, format="%.3f")
        
    with col_sim3:
        record_every = st.slider("Salvar 1 frame a cada N ciclos", 1, 50, 5)
        flow_coeff = st.slider("Coef. de escoamento", 0.01, 0.5, 0.1)
        metrics_live = st.checkbox("Mostrar métricas em tempo real", value=True)
        
        # Fontes específicas
        sources_file = None
        point_sources: List[Tuple[int, int]] = []
        point_radius = 1
        
        if modo == "FONTES_MASCARA":
            sources_file = st.file_uploader("Máscara de fontes", 
                                          type=["csv", "npy", "png", "jpg", "jpeg"])
        elif modo == "FONTES_PONTUAIS":
            n_points = st.slider("Nº de fontes", 1, 5, 2)
            point_radius = st.slider("Raio da fonte (px)", 1, 10, 2)
            for i in range(n_points):
                y = st.number_input(f"Fonte {i+1} — linha (y)", 0, 9999, 20, key=f"fy{i}")
                x = st.number_input(f"Fonte {i+1} — coluna (x)", 0, 9999, 20, key=f"fx{i}")
                point_sources.append((int(y), int(x)))
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# Seção: Inovação na Grada
# ==========================
st.markdown("<h2>Inovação na grada</h2>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    
    col_inov1, col_inov2, col_inov3 = st.columns(3)
    
    with col_inov1:
        st.checkbox("Pós-investirado a água boba sobre animação", key="inovacao1")
    with col_inov2:
        st.checkbox("Segundo o inovação", key="inovacao2")
    with col_inov3:
        st.checkbox("Para o inovação", key="inovacao3")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# Seção: Visualização
# ==========================
st.markdown("<h2>Visualização</h2>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    
    col_viz1, col_viz2, col_viz3 = st.columns(3)
    
    with col_viz1:
        output_format = st.selectbox("Formato do vídeo", ["GIF", "MP4"], index=0)
        anim_duration_s = st.slider("Duração da animação (s)", 1, 60, 
                                  max(1, int(steps * intervalo_ms / 1000)))
        water_opacity = st.slider("Opacidade da água", 0.1, 1.0, 0.7)
        
    with col_viz2:
        water_min_threshold = st.number_input("Limiar mínimo de água (m)", 0.0, 1.0, 0.0, format="%.3f")
        relief_intensity = st.slider("Intensidade do relevo", 0.5, 2.0, 1.0)
        dem_opacity = st.slider("Transparência do DEM", 0.0, 1.0, 1.0)
        
    with col_viz3:
        dom_file = st.file_uploader("Imagem de fundo (DOM) opcional", 
                                  type=["png", "jpg", "jpeg"])
        river_file = st.file_uploader("Rio (máscara) opcional", 
                                    type=["csv", "npy", "png", "jpg", "jpeg"])
        chuva_mask_file = st.file_uploader("Máscara de chuva (opcional)", 
                                         type=["csv", "npy", "png", "jpg", "jpeg"])
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# Seção: Métricas em Tempo Real
# ==========================
st.markdown("<h2>Métricas em Tempo Real</h2>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    
    # Status bar
    status_col1, status_col2 = st.columns([3, 1])
    with status_col1:
        st.markdown("""
        <div class="status-bar">
            <div>
                <span class="status-indicator status-ready"></span>
                <span id="statusText">Pronto para simular</span>
            </div>
            <div id="simulationTime">Tempo: 00:00:00</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Métricas - Tabela 1
    col_met1, col_met2 = st.columns(2)
    
    with col_met1:
        st.markdown("""
        <table class="metric-table">
            <tr>
                <th>Tempo em Trabalho</th>
                <th>Tempo de Inovação</th>
                <th>Aterrorizada</th>
                <th>Vazionar dúvida</th>
            </tr>
            <tr>
                <td id="tempoTrabalho">N/A</td>
                <td id="tempoInovacao1">0h 0m</td>
                <td id="aterrorizada1">0,00%</td>
                <td id="vazionar1">0,00 m³</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)
    
    with col_met2:
        st.markdown("""
        <table class="metric-table">
            <tr>
                <th>Tempo de Inovação</th>
                <th>Aterrorizada</th>
                <th>Vazionar dúvida</th>
            </tr>
            <tr class="highlight">
                <td id="tempoInovacao2">Dia 20m</td>
                <td id="aterrorizada2">0,00%</td>
                <td id="vazionar2">20,02 AES m³</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)
    
    # Painel de controle
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        run_btn = st.button("Iniciar Simulação", type="primary", use_container_width=True)
    with col_btn2:
        clear_btn = st.button("Pausar", use_container_width=True)
    with col_btn3:
        reset_btn = st.button("Resetar", use_container_width=True)
    with col_btn4:
        export_btn = st.button("Exportar Dados", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# Funções Auxiliares
# ==========================
def make_synthetic_dem(kind: str, n: int, seed: int) -> np.ndarray:
    """Gera DEM sintético."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n]
    if kind == "plano_inclinado":
        dem = x.astype(float) * 0.01 + y.astype(float) * 0.005
    elif kind == "colinas_gaussianas":
        dem = np.zeros((n, n), dtype=float)
        for _ in range(5):
            cx, cy = rng.integers(0, n), rng.integers(0, n)
            sx, sy = rng.integers(max(2, n//20), max(3, n//6), size=2)
            amp = rng.uniform(0.5, 2.0)
            dem += amp * np.exp(-(((x - cx) ** 2) / (2 * sx ** 2) + ((y - cy) ** 2) / (2 * sy ** 2)))
    else:  # vale_canal
        dem = (np.abs(x - n / 2) * 0.02 + (y * 0.002)).astype(float)
        dem += 0.3 * np.sin(x / 8.0) + 0.2 * np.cos(y / 10.0)
    dem -= dem.min()
    return dem

def prepare_dem_and_masks() -> Optional[Tuple[np.ndarray, Tuple[Optional[np.ndarray], Optional[List[Tuple[int, int]]]]]]:
    """Prepara DEM e máscaras para simulação."""
    if gerar_sintetico:
        assert tipo is not None
        assert size is not None
        dem = make_synthetic_dem(tipo, size, seed)
    else:
        try:
            dem = load_dem_from_upload(dem_file)
        except Exception as e:
            st.error(f"Falha ao carregar DEM: {e}")
            return None
        if dem is None:
            st.error("Nenhum DEM fornecido.")
            return None

    dem = dem.astype(float)
    dem = downscale_if_needed(dem, max_dim)
    H, W = dem.shape

    # Ajusta fontes
    if modo == "FONTES_MASCARA":
        try:
            src_mask = load_mask_from_upload(sources_file, (H, W))
        except Exception as e:
            st.error(f"Falha ao carregar máscara de fontes: {e}")
            return None
        p_sources = None
    elif modo == "FONTES_PONTUAIS":
        clamped = []
        for (yy, xx) in point_sources:
            clamped.append((int(np.clip(yy, 0, H - 1)), int(np.clip(xx, 0, W - 1))))
        src_mask = None
        p_sources = clamped
    else:
        src_mask = None
        p_sources = None

    return dem, (src_mask, p_sources)

def load_image_rgb_from_upload(file, shape_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """Carrega imagem RGB e adapta para shape do DEM."""
    if file is None:
        return None
    try:
        img = Image.open(file).convert("RGB")
        arr = np.asarray(img, dtype=float) / 255.0
        H, W = shape_hw
        if arr.shape[0] != H or arr.shape[1] != W:
            r = resize_nearest(arr[..., 0], (H, W))
            g = resize_nearest(arr[..., 1], (H, W))
            b = resize_nearest(arr[..., 2], (H, W))
            arr = np.stack([r, g, b], axis=2)
        return np.clip(arr, 0, 1)
    except Exception as e:
        st.warning(f"Falha ao carregar imagem de fundo: {e}")
        return None

# ==========================
# Execução da Simulação
# ==========================
col_left, col_right = st.columns([1, 1])

if 'last_clear' not in st.session_state:
    st.session_state.last_clear = 0

if clear_btn:
    st.session_state.last_clear += 1
    st.rerun()

if run_btn:
    prepared = prepare_dem_and_masks()
    if prepared is None:
        st.stop()
    
    dem, (src_mask, p_sources) = prepared
    chuva_m = float(chuva_mm) / 1000.0
    
    # Carregar camadas de visualização
    H, W = dem.shape
    dom_rgb = load_image_rgb_from_upload(dom_file, (H, W))
    
    river_mask = None
    if river_file is not None:
        try:
            rm = load_mask_from_upload(river_file, (H, W))
            river_mask = rm.astype(bool)
        except Exception as e:
            st.warning(f"Falha ao carregar máscara de rio: {e}")
    
    chuva_mask = None
    if chuva_mask_file is not None:
        try:
            cm = load_mask_from_upload(chuva_mask_file, (H, W))
            chuva_mask = cm.astype(bool)
        except Exception as e:
            st.warning(f"Falha ao carregar máscara de chuva: {e}")

    # Simulação
    with st.spinner("Rodando simulação..."):
        water = np.zeros_like(dem, dtype=float)
        frames_rgb: List[np.ndarray] = []

        # Placeholders para atualização em tempo real
        ph_img_left = col_left.empty()
        ph_progress = st.progress(0)

        start = time.perf_counter()
        
        # Pré-calcular fontes pontuais
        if p_sources:
            yy, xx = np.ogrid[:H, :W]
            source_disks = []
            r2 = point_radius * point_radius
            for (sy, sx) in p_sources:
                mask = (yy - sy) * (yy - sy) + (xx - sx) * (xx - sx) <= r2
                source_disks.append(mask)
        else:
            source_disks = []

        for i in range(1, steps + 1):
            # Adição de água
            if modo == "CHUVA_UNIFORME":
                if chuva_mask is not None:
                    water[chuva_mask] += chuva_m
                else:
                    water += chuva_m
            elif modo == "FONTES_MASCARA" and src_mask is not None:
                water[src_mask] += chuva_m
            elif modo == "FONTES_PONTUAIS" and source_disks:
                for mask in source_disks:
                    water[mask] += chuva_m

            # Escoamento
            if comportamento.startswith("Distribuição proporcional"):
                flow = water.copy()
                for y in range(1, H - 1):
                    for x in range(1, W - 1):
                        if flow[y, x] <= 0:
                            continue
                        here = dem[y, x] + flow[y, x]
                        lowers = []
                        tot_diff = 0.0
                        for ny, nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                            there = dem[ny, nx] + flow[ny, nx]
                            if here > there:
                                d = here - there
                                lowers.append((ny, nx, d))
                                tot_diff += d
                        if tot_diff > 0:
                            max_out = min(flow[y, x], tot_diff * flow_coeff)
                            water[y, x] -= max_out
                            for ny, nx, d in lowers:
                                share = (d / tot_diff) * max_out
                                water[ny, nx] += share

            # Coletar frame
            if i % max(1, record_every) == 0:
                frame = compose_frame_adv(
                    dem, water, water_max_display,
                    dom_rgb=dom_rgb,
                    dem_opacity=dem_opacity,
                    relief_intensity=relief_intensity,
                    river_mask=river_mask,
                    river_color=(0.0, 0.2, 1.0),
                    river_opacity=0.7,
                    water_opacity=water_opacity,
                    water_min_threshold=water_min_threshold,
                )
                frames_rgb.append(frame)
                ph_img_left.image(frame, caption=f"Ciclo {i}/{steps}")

            # Atualizar progresso
            ph_progress.progress(i / steps)

            # Ritmo de visualização
            if intervalo_ms > 0:
                time.sleep(intervalo_ms / 1000.0)

        # Geração do GIF/MP4
        if not frames_rgb:
            frames_rgb.append(
                compose_frame_adv(
                    dem, water, water_max_display,
                    dom_rgb=dom_rgb,
                    dem_opacity=dem_opacity,
                    relief_intensity=relief_intensity,
                    river_mask=river_mask,
                    river_color=(0.0, 0.2, 1.0),
                    river_opacity=0.7,
                    water_opacity=water_opacity,
                    water_min_threshold=water_min_threshold,
                )
            )
        
        pil_frames = [Image.fromarray(f) for f in frames_rgb]
        gif_buf = BytesIO()
        duration_ms = int(max(50, (anim_duration_s * 1000) / max(1, len(pil_frames))))
        
        if output_format == "GIF":
            pil_frames[0].save(
                gif_buf,
                format="GIF",
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0,
                disposal=2,
            )
            gif_buf.seek(0)
            media_buf = gif_buf
            media_caption = "Evolução da lâmina d'água (GIF)"
        else:
            pil_frames[0].save(
                gif_buf,
                format="GIF", 
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0,
                disposal=2,
            )
            gif_buf.seek(0)
            media_buf = gif_buf
            media_caption = "Evolução da lâmina d'água (GIF gerado)"

    with col_right:
        st.subheader("Animação")
        st.image(media_buf, caption=media_caption)

else:
    st.info("Preencha os dados e clique em 'Iniciar Simulação' para executar a Simulação Vetorizada (NumPy).")

# ==========================
# Rodapé
# ==========================
st.markdown("<hr style='border: none; border-top: 1px dashed #ccc; margin: 40px 0 20px 0;'>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 14px;'>"
    "Simulador Híbrido de Inundações • Versão 1.0 • Desenvolvido com Streamlit e NumPy"
    "</div>",
    unsafe_allow_html=True
)