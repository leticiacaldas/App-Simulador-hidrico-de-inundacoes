#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

try:
    from pysheds.grid import Grid
except Exception as e:
    raise SystemExit("Erro: pysheds não está instalado. Instale com 'pip install pysheds' no seu ambiente.")


def calcular_area_bacia(
    dem_path: Path,
    stream_quantile: float = 85.0,
    outlet_mode: str = "border_acc",
    out_row: int | None = None,
    out_col: int | None = None,
) -> dict:
    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM não encontrado: {dem_path}")

    grid = Grid.from_raster(str(dem_path))
    dem = grid.read_raster(str(dem_path))

    # Garantir float e substituir valores absurdos por NaN
    dem = dem.astype(float)

    # Preencher depressões e resolver áreas planas
    grid.fill_depressions(dem, in_place=True)
    grid.resolve_flats(dem, in_place=True)

    # Direção de fluxo (D8) e acumulação
    fdir = grid.flowdir(dem)
    acc = grid.accumulation(fdir)

    # Derivar rios: limiar por quantil alto da acumulação (configurável)
    q = float(max(0.0, min(100.0, float(stream_quantile))))
    thr = float(np.nanpercentile(acc, q))
    streams = acc >= thr

    # Construir máscara da borda
    h, w = dem.shape
    border_mask = np.zeros_like(streams, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    # Seleção do exutório
    sel_mode = str(outlet_mode or "border_stream").lower()
    if out_row is not None and out_col is not None:
        row_out, col_out = int(out_row), int(out_col)
        outlet_info = "manual"
    elif sel_mode == "border_stream":
        # Preferir rios na borda, senão maior acumulação na borda
        candidates = np.where(border_mask & streams, acc, -np.inf)
        if np.isfinite(candidates).any():
            flat_idx = int(np.nanargmax(candidates))
            outlet_info = "border_stream"
        else:
            border_acc = np.where(border_mask, acc, -np.inf)
            flat_idx = int(np.nanargmax(border_acc))
            outlet_info = "border_acc"
        row_out, col_out = np.unravel_index(flat_idx, dem.shape)
    elif sel_mode == "border_acc":
        border_acc = np.where(border_mask, acc, -np.inf)
        flat_idx = int(np.nanargmax(border_acc))
        row_out, col_out = np.unravel_index(flat_idx, dem.shape)
        outlet_info = "border_acc"
    elif sel_mode == "border_elev":
        border_vals = np.where(border_mask, dem, np.nan)
        flat_idx = int(np.nanargmin(border_vals))
        row_out, col_out = np.unravel_index(flat_idx, dem.shape)
        outlet_info = "border_elev"
    elif sel_mode == "max_acc":
        flat_idx = int(np.nanargmax(acc))
        row_out, col_out = np.unravel_index(flat_idx, dem.shape)
        outlet_info = "max_acc"
    else:
        # fallback padrão
        border_acc = np.where(border_mask, acc, -np.inf)
        flat_idx = int(np.nanargmax(border_acc))
        row_out, col_out = np.unravel_index(flat_idx, dem.shape)
        outlet_info = "border_acc"

    # Delimitar bacia contribuinte
    # dirmap D8 esperado pelo pysheds como array 3x3 linearizado (índices 0..8)
    # Mapa na convenção (linhas):
    #  [32, 64, 128,
    #   16,  0,   1,
    #    8,  4,   2]
    d8_dirmap_arr = np.array([32, 64, 128, 16, 0, 1, 8, 4, 2], dtype=np.int64)
    catch = grid.catchment(x=int(col_out), y=int(row_out), fdir=fdir, dirmap=d8_dirmap_arr, xytype='index')

    # Área (número de pixels verdadeiros * área do pixel)
    n_pix = int(np.sum(catch.astype(bool)))
    pix_area_m2 = abs(grid.affine.a * grid.affine.e)
    area_m2 = float(n_pix * pix_area_m2)
    area_km2 = area_m2 / 1e6

    # Área total do AOI (raster completo)
    a = grid.affine
    aoi_area_m2 = abs(a.a * a.e) * (dem.shape[0] * dem.shape[1])

    return {
        "pixels": n_pix,
        "pixel_area_m2": pix_area_m2,
        "area_m2": area_m2,
        "area_km2": area_km2,
        "outlet_row": int(row_out),
        "outlet_col": int(col_out),
        "aoi_area_m2": float(aoi_area_m2),
        "aoi_area_km2": float(aoi_area_m2) / 1e6,
        "stream_quantile": q,
        "acc_threshold": thr,
        "outlet_mode": outlet_info,
        "outlet_acc": float(acc[row_out, col_out]) if np.isfinite(acc[row_out, col_out]) else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description="Verificar tamanho da bacia a partir de um DEM (pysheds)")
    parser.add_argument("--dem", type=str, default=str(Path("DEM 50cm") / "DEM_AOI_50CM.tif"), help="Caminho do DEM GeoTIFF")
    parser.add_argument("--stream-quantile", type=float, default=85.0, help="Quantil (0-100) da acumulação para definir rios (padrão: 85.0)")
    parser.add_argument("--outlet-mode", type=str, default="border_acc", choices=["border_stream","border_acc","border_elev","max_acc"], help="Critério de seleção do exutório")
    parser.add_argument("--out-row", type=int, default=None, help="Linha do exutório (sobrescreve o modo)")
    parser.add_argument("--out-col", type=int, default=None, help="Coluna do exutório (sobrescreve o modo)")
    args = parser.parse_args()

    res = calcular_area_bacia(
        Path(args.dem),
        stream_quantile=args.stream_quantile,
        outlet_mode=args.outlet_mode,
        out_row=args.out_row,
        out_col=args.out_col,
    )

    print("=" * 60)
    print("RESULTADO: ÁREA DA BACIA (DELINEAÇÃO AUTOMÁTICA)")
    print("=" * 60)
    print(f"DEM: {args.dem}")
    print(f"Área estimada da bacia: {res['area_km2']:.6f} km² ({res['area_m2']:,.0f} m²)")
    print(f"Nº de pixels na bacia: {res['pixels']}")
    print(f"Área por pixel: {res['pixel_area_m2']:.2f} m²")
    print(f"Ponto de saída (linha, coluna): ({res['outlet_row']}, {res['outlet_col']})")
    print(f"Quantil de acumulação usado: {res['stream_quantile']:.1f}% (limiar = {res['acc_threshold']:.1f})")
    print(f"Modo de exutório: {res['outlet_mode']} (acc={res['outlet_acc']:.1f})")
    print("-")
    print(f"Área total do DEM (AOI): {res['aoi_area_km2']:.6f} km² ({res['aoi_area_m2']:,.0f} m²)")


if __name__ == "__main__":
    main()
