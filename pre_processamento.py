# Bibliotecas
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from scipy.interpolate import griddata
import os

def preparar_tif_para_gama(input_tif_path, output_dir):
    """
    Lê um arquivo TIF, garante que ele tenha apenas uma banda e o salva
    em um formato simples e sem compressão para máxima compatibilidade com o GAMA.
    """
    print(f"Preparando TIF para o GAMA: {os.path.basename(input_tif_path)}")
    output_path = os.path.join(output_dir, "DOM_GAMA.tif")
    
    try:
        with rio.open(input_tif_path) as src:
            # Lê apenas a primeira banda
            data = src.read(1)
            
            # Pega o perfil original e simplifica para garantir compatibilidade
            profile = src.profile
            profile.update(
                count=1,          # Garante que há apenas uma banda
                compress='none',  # Remove qualquer compressão
                driver='GTiff'
            )
            
            # Salva o novo arquivo TIF simplificado
            with rio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)
                
        print(f"Arquivo TIF compatível com GAMA salvo em: {output_path}")
        return output_path
    except Exception as e:
        print(f"Ocorreu um erro ao preparar o TIF: {e}")
        return None

def resample_raster(input_path, output_path, scale_factor):
    """Reduz a resolução de um raster por um fator de escala."""
    print(f"Reamostrando {os.path.basename(input_path)}...")
    with rio.open(input_path) as src:
        new_height = int(src.height * scale_factor)
        new_width = int(src.width * scale_factor)
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bilinear
        )
        transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )
        profile = src.profile
        profile.update(
            transform=transform,
            width=new_width,
            height=new_height,
            driver='GTiff'
        )
        with rio.open(output_path, 'w', **profile) as dst:
            dst.write(data)
    print(f"Raster reamostrado e salvo em: {output_path}")
    return output_path

def calcular_declividade(dem_path):
    """Calcula a declividade a partir de um DEM e salva em um novo arquivo."""
    print('Calculando declividade...')
    with rio.open(dem_path) as src:
        dem = src.read(1)
        x_res, y_res = src.transform[0], -src.transform[4]
        dx, dy = np.gradient(dem, y_res, x_res)
        declividade = np.arctan(np.sqrt(dx**2 + dy**2)) * (180 / np.pi)
        declividade_path = dem_path.replace("_resampled.tif", "_declividade.tif")
        profile = src.profile
        profile.update(dtype=rio.float32)
        with rio.open(declividade_path, 'w', **profile) as dst:
            dst.write(declividade.astype(rio.float32), 1)
    print(f"Declividade salva em: {declividade_path}")
    return declividade_path

def calcular_direcao_fluxo(dem_path):
    """Cria um raster de direção de fluxo (placeholder)."""
    print('Calculando direção de fluxo...')
    direcao_fluxo_path = dem_path.replace("_resampled.tif", "_direcao_fluxo.tif")
    with rio.open(dem_path) as src, rio.open(direcao_fluxo_path, 'w', **src.profile) as dst:
        dst.write(src.read())
    print(f"Direção de fluxo (placeholder) salva em: {direcao_fluxo_path}")
    return direcao_fluxo_path

def processar_terraceamento(terraceamento_path, dem_resampled_path):
    """Alinha o raster de terraceamento ao DEM de baixa resolução."""
    print('Processando terraceamento...')
    terraceamento_processado_path = terraceamento_path.replace("_resampled.tif", "_processado.tif")
    with rio.open(dem_resampled_path) as dem_src:
        dem_profile = dem_src.profile
        dem_shape = dem_src.shape
    with rio.open(terraceamento_path) as src:
        data = src.read(1, out_shape=dem_shape, resampling=Resampling.nearest)
        profile = dem_profile.copy()
        profile.update(dtype=data.dtype)
        with rio.open(terraceamento_processado_path, 'w', **profile) as dst:
            dst.write(data, 1)
    print(f"Terraceamento processado e salvo em: {terraceamento_processado_path}")
    return terraceamento_processado_path

def processar_penetrometro(penetrometro_path, dem_resampled_path):
    """Interpola os dados de profundidade do solo para a grade do DEM."""
    print('Processando dados do penetrômetro...')
    profundidade_processada_path = penetrometro_path.replace(".csv", "_processado.tif")
    df = pd.read_csv(penetrometro_path)
    with rio.open(dem_resampled_path) as src:
        dem_data = src.read(1)
        transform = src.transform
        coords = np.indices(dem_data.shape)
        x_coords = transform[2] + coords[1] * transform[0]
        y_coords = transform[5] + coords[0] * transform[4]
    points = df[['longitude', 'latitude']].values
    values = df['profundidade_cm'].values
    grid_z = griddata(points, values, (x_coords, y_coords), method='cubic')
    grid_z = np.nan_to_num(grid_z)
    profile = src.profile
    profile.update(dtype=rio.float32)
    with rio.open(profundidade_processada_path, 'w', **profile) as dst:
        dst.write(grid_z.astype(rio.float32), 1)
    print(f"Dados do penetrômetro processados e salvos em: {profundidade_processada_path}")
    return profundidade_processada_path

def extrair_valores_para_csv(output_csv_path, **kwargs):
    """Extrai valores de múltiplos rasters e salva em um arquivo CSV."""
    print('Extraindo valores para CSV...')
    with rio.open(kwargs['dem']) as src:
        data = {'posicao_x': [], 'posicao_y': []}
        for i in range(src.width):
            for j in range(src.height):
                x, y = src.xy(j, i)
                data['posicao_x'].append(x)
                data['posicao_y'].append(y)
    df = pd.DataFrame(data)
    for key, path in kwargs.items():
        with rio.open(path) as src:
            values = list(src.sample(zip(df['posicao_x'], df['posicao_y'])))
            df[key] = np.array([v[0] for v in values])
    df.to_csv(output_csv_path, index=False)
    print(f"Arquivo CSV final salvo em: {output_csv_path}")

# --- Função Principal ---
if __name__ == "__main__":
    # Define o caminho base como o diretório onde o script está localizado
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Caminho para o arquivo TIF original
    dom_original_path = os.path.join(base_dir, "DOM_AOI.tif")
    
    # Define o caminho para a pasta 'data' do projeto GAMA
    # Este caminho assume que a pasta do projeto GAMA está no mesmo nível que a pasta 'projeto_rs'
    # Ex: /Desktop/Gama_Workspace/simulacao_de_inundacao/
    #     /Desktop/projeto_rs/
    gama_project_dir = os.path.abspath(os.path.join(base_dir, '..', 'Gama_Workspace', 'simulacao_de_inundacao'))
    gama_data_dir = os.path.join(gama_project_dir, 'data')
    
    # Garante que o diretório de dados do GAMA exista
    print(f"Verificando/criando diretório de dados do GAMA em: {gama_data_dir}")
    os.makedirs(gama_data_dir, exist_ok=True)
    
    # --- PASSO PRINCIPAL: Preparar o TIF para o GAMA ---
    # Este passo cria uma versão limpa e compatível do TIF de elevação.
    if os.path.exists(dom_original_path):
        preparar_tif_para_gama(dom_original_path, gama_data_dir)
        print("\nPré-processamento para GAMA concluído com sucesso.")
        print(f"O arquivo 'DOM_GAMA.tif' foi criado/atualizado na pasta '{gama_data_dir}'.")
        print("Por favor, use este novo arquivo no seu modelo GAMA.")
    else:
        print(f"ERRO: Arquivo de entrada não encontrado em '{dom_original_path}'. Verifique o caminho.")

    # --------------------------------------------------------------------
    # O código abaixo, que gerava o arquivo CSV, foi desativado, pois o
    # novo modelo GAML não precisa mais dele. Se for necessário no futuro,
    # você pode descomentar as linhas.
    # --------------------------------------------------------------------
    # print("\nIniciando processamento antigo para gerar CSV (desativado)...")
    # scale_factor = 0.1 
    # dem_resampled_path = resample_raster(dom_path, dom_path.replace(".tif", "_resampled.tif"), scale_factor)
    # declividade_path = calcular_declividade(dem_resampled_path)
    # direcao_fluxo_path = calcular_direcao_fluxo(dem_resampled_path)
    # terraceamento_resampled_path = resample_raster(terraceamento_path, terraceamento_path.replace(".tif", "_resampled.tif"), scale_factor)
    # terraceamento_processado_path = processar_terraceamento(terraceamento_resampled_path, dem_resampled_path)
    # profundidade_processada_path = processar_penetrometro(penetrometro_path, dem_resampled_path)
    # extrair_valores_para_csv(
    #     os.path.join(gama_data_dir, "celulas_terreno.csv"),
    #     dem=dem_resampled_path,
    #     declividade=declividade_path,
    #     fluxo=direcao_fluxo_path,
    #     terraceamento=terraceamento_processado_path,
    #     profundidade=profundidade_processada_path
    # )