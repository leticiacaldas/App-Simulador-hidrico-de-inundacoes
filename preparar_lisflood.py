import os
import subprocess
import tempfile
import shutil

def preparar_ambiente_lisflood(dem_tif_path: str, output_dir: str) -> bool:
    """Prepara ambiente completo para LISFLOOD criando PCRaster e mapas climáticos zerados."""
    input_dir = os.path.join(output_dir, "input")
    os.makedirs(input_dir, exist_ok=True)

    # Copiar DEM
    dem_input = os.path.join(input_dir, "dem.tif")
    shutil.copy(dem_tif_path, dem_input)

    print("🔄 Convertendo arquivos para PCRaster...")

    cmd = (
        f"docker run --rm -v {input_dir}:/input efas/lisflood:latest bash -c "
        "'gdal_translate -q -of PCRaster /input/dem.tif /input/dem.map && "
        "pcr2map --clone /input/dem.map --mask /input/dem.map /input/MASK.map && "
        "lddcreate /input/dem.map /input/Ldd.map && "
        "pcr2map --clone /input/dem.map --constant 0 /input/pr.map && "
        "pcr2map --clone /input/dem.map --constant 0 /input/tavg.map && "
        "pcr2map --clone /input/dem.map --constant 0 /input/et.map && "
        "echo \"✅ Todos os mapas criados com sucesso!\"'"
    )

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Erro na conversão: {result.stderr}")
        return False

    print("✅ Mapas PCRaster criados:")
    for mapa in ["MASK.map", "Ldd.map", "pr.map", "tavg.map", "et.map"]:
        status = "OK" if os.path.exists(os.path.join(input_dir, mapa)) else "FALTA"
        print(f"   - {mapa}: {status}")

    return True


def executar_lisflood(output_dir: str) -> bool:
    """Executa LISFLOOD com ambiente preparado e template completo."""
    input_dir = os.path.join(output_dir, "input")
    template_src = os.path.join(os.path.dirname(__file__), "templates", "template_completo.xml")
    template_dst = os.path.join(input_dir, "settings.xml")
    shutil.copy(template_src, template_dst)

    print("🚀 Executando LISFLOOD (--initonly)...")
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{input_dir}:/input",
        "efas/lisflood:latest",
        "/input/settings.xml", "--initonly",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    print("=== STDOUT ===")
    print(result.stdout)
    print("=== STDERR ===")
    print(result.stderr)
    print(f"=== EXIT CODE: {result.returncode} ===")

    return result.returncode == 0


if __name__ == "__main__":
    # Ajuste este caminho se necessário
    dem_path = os.path.join(os.path.dirname(__file__), "DEM 50cm", "DEM_AOI_50CM.tif")
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"📁 Diretório temporário: {tmp_dir}")
        if preparar_ambiente_lisflood(dem_path, tmp_dir):
            ok = executar_lisflood(tmp_dir)
            print("🎉 LISFLOOD executado com SUCESSO!" if ok else "❌ LISFLOOD falhou")
        else:
            print("❌ Falha na preparação do ambiente")
