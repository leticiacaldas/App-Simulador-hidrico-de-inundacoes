# design.py
import streamlit as st
import base64
from typing import Optional

def _path_to_data_uri(path: str) -> str:
    """Converte um caminho de imagem local para data URI base64 (image/png) para exibir no HTML."""
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except (OSError, ValueError):
        # Se falhar, retorna o caminho original (pode funcionar se servido estaticamente)
        return path

def apply_custom_styles():
    """Aplica estilos CSS customizados baseados no tema"""
    st.markdown("""
    <style>
    /* ===== ESTILOS GERAIS ===== */
    .main {
        background-color: #F7F9FA;
    }
    
    /* ===== CABEÇALHO E LOGOS ===== */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 2px solid #DCE3E5;
        margin-bottom: 2rem;
    }
    
    .main-logo {
        height: 80px;
        transition: transform 0.3s ease;
    }
    
    .main-logo:hover {
        transform: scale(1.05);
    }
    
    .secondary-logo {
        height: 50px;
        opacity: 0.7;
        transition: opacity 0.3s ease;
    }
    
    .secondary-logo:hover {
        opacity: 1;
    }
    
    .title-section {
        text-align: center;
        flex-grow: 1;
        margin: 0 2rem;
    }
    
    /* ===== TIPOGRAFIA ===== */
    h1, h2, h3 {
        color: #003F5C;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    h1 {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #3FB8AF, #007F91);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Logo inline no título */
    .title-logo {
        height: 42px;
        vertical-align: middle;
        margin-right: 12px;
        border-radius: 6px;
    }
    
    /* ===== BOTÕES ===== */
    .stButton > button {
        background: linear-gradient(135deg, #3FB8AF, #007F91);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(63, 184, 175, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(63, 184, 175, 0.3);
    }
    
    /* ===== ABAS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #DCE3E5;
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3FB8AF !important;
        color: white !important;
    }
    
    /* ===== CARDS E CONTAINERS ===== */
    .custom-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3FB8AF;
        box-shadow: 0 2px 8px rgba(0, 63, 92, 0.1);
        margin-bottom: 1rem;
    }
    
    /* ===== SLIDERS E INPUTS ===== */
    .stSlider [data-testid="stThumb"] {
        background-color: #3FB8AF;
    }
    
    .stNumberInput input, .stTextInput input {
        border: 2px solid #DCE3E5;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: #3FB8AF;
        box-shadow: 0 0 0 2px rgba(63, 184, 175, 0.2);
    }
    
    /* ===== MÉTRICAS ===== */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, white, #F7F9FA);
        border: 1px solid #DCE3E5;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 6px rgba(0, 63, 92, 0.1);
    }
    
    /* ===== UPLOAD DE ARQUIVOS ===== */
    .stFileUploader > div > div {
        border: 2px dashed #3FB8AF;
        border-radius: 12px;
        background-color: rgba(63, 184, 175, 0.05);
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        background-color: rgba(63, 184, 175, 0.1);
        border-color: #007F91;
    }
    
    /* ===== SIDEBAR ===== */
    .css-1d391kg {
        background-color: white;
        border-right: 1px solid #DCE3E5;
    }
    
    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3FB8AF, #007F91);
    }
    
    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader {
        background-color: #F7F9FA;
        border: 1px solid #DCE3E5;
        border-radius: 8px;
        font-weight: 600;
        color: #003F5C;
    }
    
    /* ===== TOOLTIPS ===== */
    .stTooltip {
        background-color: #003F5C;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def create_header(logo_main_path: Optional[str], logo_secondary_path: Optional[str]):
    """Cria o cabeçalho com as duas logos. Aceita caminhos locais e converte para data URI quando possível."""
    if logo_main_path:
        logo_main_path = _path_to_data_uri(logo_main_path)
    if logo_secondary_path:
        logo_secondary_path = _path_to_data_uri(logo_secondary_path)
    st.markdown("""
    <div class="header-container">
        <div class="logo-main">
            <img src="{logo_main}" class="main-logo" alt="Logo Principal">
        </div>
        <div class="title-section">
            <h1>Simulador hibrido de inundações</h1>
        </div>
        <div class="logo-secondary">
            <img src="{logo_secondary}" class="secondary-logo" alt="Logo do Projeto">
        </div>
    </div>
    """.format(logo_main=logo_main_path or "", logo_secondary=logo_secondary_path or ""), unsafe_allow_html=True)