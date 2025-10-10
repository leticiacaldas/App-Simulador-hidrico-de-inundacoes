import geopandas as gpd
import os

# =============================================================================
# CONFIGURAÇÕES - Altere estes valores
# =============================================================================

# Caminho para o seu shapefile GRANDE e ORIGINAL
ARQUIVO_ENTRADA = 'DEM_AOI_1.shp' 

# Nome para o novo shapefile, que será PEQUENO e SIMPLIFICADO
ARQUIVO_SAIDA = 'DEM_AOI_2_simplificado.shp'

# FATOR DE SIMPLIFICAÇÃO (TOLERÂNCIA)
# Como estamos dissolvendo primeiro, podemos usar valores de tolerância menores.
# Tente começar com 100 e aumente se a redução não for suficiente.
TOLERANCIA = 1000  # Simplificação de 1000 metros

# =============================================================================
# ALGORITMO DE DISSOLUÇÃO E SIMPLIFICAÇÃO
# =============================================================================

def processar_shapefile(entrada, saida, tolerancia):
    """
    Lê um shapefile, dissolve todas as feições em uma, simplifica a geometria
    resultante e salva em um novo arquivo.
    """
    if not os.path.exists(entrada):
        print(f"Erro: O arquivo de entrada não foi encontrado em '{entrada}'")
        return

    try:
        print(f"1/5 - Lendo o arquivo '{entrada}'... (Isso pode demorar MUITO)")
        gdf = gpd.read_file(entrada)
        tamanho_antes = os.path.getsize(entrada) / (1024 * 1024)
        print(f"Arquivo lido com sucesso. Tamanho original: {tamanho_antes:.2f} MB")

        # ETAPA DE DISSOLUÇÃO
        print("\n2/5 - Dissolvendo todas as geometrias em uma única feição...")
        print("      Isso pode consumir muita memória e tempo.")
        # Criamos uma coluna 'dummy' com um valor constante para dissolver tudo junto
        gdf['dissolve_field'] = 1
        dissolved_gdf = gdf.dissolve(by='dissolve_field')
        print("Geometrias dissolvidas com sucesso.")

        # ETAPA DE SIMPLIFICAÇÃO
        print(f"\n3/5 - Simplificando a geometria dissolvida com tolerância de {tolerancia}...")
        dissolved_gdf['geometry'] = dissolved_gdf.geometry.simplify(tolerance=tolerancia, preserve_topology=True)
        print("Geometria simplificada.")

        # ETAPA DE SALVAMENTO
        print(f"\n4/5 - Salvando o novo arquivo em '{saida}'...")
        dissolved_gdf.to_file(saida, driver='ESRI Shapefile')
        print("Arquivo salvo com sucesso!")

        # RELATÓRIO FINAL
        tamanho_depois = os.path.getsize(saida) / (1024 * 1024)
        reducao = 100 * (1 - tamanho_depois / tamanho_antes) if tamanho_antes > 0 else 0

        print("\n--- RELATÓRIO FINAL ---")
        print(f"Tamanho original: {tamanho_antes:.2f} MB")
        print(f"Tamanho final: {tamanho_depois:.2f} MB")
        print(f"Redução de tamanho: {reducao:.2f}%")
        print("5/5 - Processo concluído!")

    except Exception as e:
        print(f"Ocorreu um erro durante o processo: {e}")
        print("Isso pode acontecer por falta de memória. Tente fechar outros programas.")

if __name__ == '__main__':
    processar_shapefile(ARQUIVO_ENTRADA, ARQUIVO_SAIDA, TOLERANCIA)
