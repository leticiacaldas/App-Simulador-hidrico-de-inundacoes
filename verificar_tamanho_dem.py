import rasterio

def analyze_dem(dem_path):
    with rasterio.open(dem_path) as src:
        print(f"Dimensões: {src.width} x {src.height}")
        print(f"Contagem de bandas: {src.count}")
        print(f"CRS: {src.crs}")
        print(f"Transformação: {src.transform}")
        print(f"Resolução: {src.res}")
        print(f"Bounds: {src.bounds}")
        
        # Ler a banda 1
        dem_data = src.read(1)
        print(f"Valor mínimo: {dem_data.min()}")
        print(f"Valor máximo: {dem_data.max()}")
        print(f"Valor médio: {dem_data.mean()}")
        print(f"Área (número de pixels): {dem_data.size}")
        
        # Calcular área em m² e km²
        transform = src.transform
        pixel_area = transform[0] * (-transform[4])  # pixel width * pixel height (em unidades do CRS, geralmente metros)
        total_area = pixel_area * dem_data.size
        print(f"Área total (m²): {total_area:.2f}")
        print(f"Área total (km²): {total_area/1e6:.2f}")

if __name__ == "__main__":
    analyze_dem("DEM_AOI_50CM.tif")