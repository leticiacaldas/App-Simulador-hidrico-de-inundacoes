"""
Integração com LISFLOOD via Docker.

Funções principais:
- prepare_input_folder(base_dir): cria /input e /input/output
- copy_rasters: copia DEM/DOM/DSM (e opcional máscara) para nomes padronizados
- patch_settings_xml: atualiza um template XML (textvar) com caminhos e parâmetros
- run_lisflood_docker: executa o container e retorna stdout/stderr e exit code
- list_outputs: lista arquivos gerados em /input/output
"""
from __future__ import annotations

import os
import shutil
import tempfile
import subprocess
from typing import Optional, Dict, Tuple, List
import xml.etree.ElementTree as ET
import numpy as np
import rasterio as rio
import csv


def prepare_input_folder(base_dir: Optional[str] = None) -> Tuple[str, str]:
    """Cria pasta de input (montada no Docker) e subpasta output.

    Retorna (input_folder, output_folder)
    """
    if base_dir is None:
        base_dir = tempfile.mkdtemp(prefix="lisflood_")
    input_folder = base_dir
    output_folder = os.path.join(input_folder, "output")
    os.makedirs(output_folder, exist_ok=True)
    return input_folder, output_folder


def copy_rasters(input_folder: str, dem_path: str, dom_path: Optional[str] = None,
                 dsm_path: Optional[str] = None, mask_path: Optional[str] = None) -> Dict[str, str]:
    """Copia rasters para nomes padronizados no input_folder.

    Retorna dict com caminhos padronizados dentro do input_folder.
    """
    targets = {}
    def _cp(src: str, dst_name: str):
        if src and os.path.exists(src):
            dst = os.path.join(input_folder, dst_name)
            shutil.copyfile(src, dst)
            targets[dst_name] = dst
            return dst
        return ""

    dem_dst = _cp(dem_path, "DEM.tif")
    if dom_path:
        _cp(dom_path, "DOM.tif")
    if dsm_path:
        _cp(dsm_path, "DSM.tif")
    if mask_path:
        _cp(mask_path, "MASK.tif")
    else:
        # Gerar máscara básica a partir do DEM (1 para pixels válidos, 0 para NoData)
        try:
            if dem_dst and os.path.exists(dem_dst):
                with rio.open(dem_dst) as src:
                    profile = src.profile.copy()
                    # Usar máscara do raster: 0 (nodata) / 255 (válido)
                    bandmask = src.read_masks(1)
                    mask = (bandmask > 0).astype(np.uint8)
                    profile.update(dtype=rio.uint8, count=1, nodata=0, compress='lzw')
                    out_path = os.path.join(input_folder, "MASK.tif")
                    with rio.open(out_path, 'w', **profile) as dst:
                        dst.write(mask, 1)
                    targets["MASK.tif"] = out_path
        except (OSError, ValueError):
            # silencioso: se falhar, deixa sem máscara (o chamador pode tratar)
            pass
    return targets


def _find_or_create(parent: ET.Element, tag: str) -> ET.Element:
    els = parent.findall(tag)
    if els:
        return els[0]
    return ET.SubElement(parent, tag)


def _set_textvar(parent: ET.Element, name: str, value: str) -> None:
    # procura textvar existente
    for tv in parent.findall("textvar"):
        if tv.get("name") == name:
            tv.set("value", value)
            return
    # cria novo
    ET.SubElement(parent, "textvar", name=name, value=value)


def patch_settings_xml(template_xml: str, out_xml: str, replacements: Dict[str, str]) -> None:
    """Atualiza um template LISFLOOD com textvars necessários.

    Versão simplificada para templates mínimos.
    """
    tree = ET.parse(template_xml)
    root = tree.getroot()

    # Seções padrão - versão simplificada
    lfuser = _find_or_create(root, "lfuser")
    lfbinding = _find_or_create(root, "lfbinding")

    # Aplicar substituições em lfuser
    for k, v in replacements.items():
        _set_textvar(lfuser, k, v)

    # Aplicar substituições em lfbinding para mapas essenciais
    map_bindings = {
        "MaskMap": "MASK.map",
        "Ldd": "Ldd.map", 
        "dem": "DEM.map",
        "dom": "DOM.map",
        "dsm": "DSM.map"
    }
    
    for xml_key, file_name in map_bindings.items():
        if xml_key in replacements:
            _set_textvar(lfbinding, xml_key, replacements[xml_key])
        else:
            # Tentar criar binding padrão se o arquivo existir
            file_path = os.path.join(os.path.dirname(template_xml), file_name)
            if os.path.exists(file_path):
                _set_textvar(lfbinding, xml_key, f"/input/{file_name}")

    # Garantir bindings essenciais
    essential_bindings = {
        "MaskMap": "/input/MASK.map",
        "Ldd": "/input/Ldd.map"
    }
    
    for key, default_value in essential_bindings.items():
        if not any(tv.get("name") == key for tv in lfbinding.findall("textvar")):
            _set_textvar(lfbinding, key, default_value)

    # grava
    ET.indent(tree, space="\t", level=0)
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)


def _inject_defaults_from_csv(xml_path: str, defaults_csv_path: str) -> None:
    """Abre o XML e garante que textvars listadas no CSV existam no lfbinding com valor "0".

    O CSV deve ter colunas module,setting. Apenas o nome do setting é usado.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        lfbinding = _find_or_create(root, "lfbinding")
        existing = {tv.get("name"): tv for tv in lfbinding.findall("textvar")}
        with open(defaults_csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            _ = next(reader, None)  # descarta cabeçalho, se houver
            for row in reader:
                if not row:
                    continue
                setting = row[-1].strip().strip("[]'\"")
                if not setting:
                    continue
                if setting not in existing:
                    ET.SubElement(lfbinding, "textvar", name=setting, value="0")
        ET.indent(tree, space="\t", level=0)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    except (OSError, ET.ParseError, csv.Error, UnicodeError):
        # se falhar, não interrompe o fluxo principal
        pass


def patch_settings_xml_with_defaults(template_xml: str, out_xml: str, replacements: Dict[str, str],
                                     defaults_csv_path: Optional[str] = None) -> None:
    """Como patch_settings_xml, mas injeta defaults neutros dos módulos faltantes a partir de um CSV.

    Útil para rodar --initonly sem precisar de todos os mapas/curvas quando módulos estão inativos.
    """
    patch_settings_xml(template_xml, out_xml, replacements)
    if defaults_csv_path and os.path.exists(defaults_csv_path):
        _inject_defaults_from_csv(out_xml, defaults_csv_path)


def run_lisflood_docker(input_folder: str, xml_filename: str,
                         extra_args: Optional[List[str]] = None,
                         stream: bool = False) -> Tuple[int, str, str]:
    """Executa o container LISFLOOD com o XML dado.

    Se stream=True, imprime a saída em tempo real e ainda retorna stdout/err acumulados.
    """
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{input_folder}:/input",
        "jrce1/lisflood:latest",
        f"/input/{xml_filename}",
    ]

    if extra_args:
        cmd.extend(extra_args)

    if stream:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out_lines: List[str] = []
        if proc.stdout:
            for line in proc.stdout:
                print(line, end="")
                out_lines.append(line)
        code = proc.wait()
        return code, "".join(out_lines), ""
    else:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return res.returncode, res.stdout, res.stderr


def list_outputs(output_folder: str) -> List[str]:
    files = []
    if os.path.isdir(output_folder):
        for root, _, fnames in os.walk(output_folder):
            for fn in fnames:
                files.append(os.path.join(root, fn))
    return sorted(files)


def generate_ldd_from_dem(input_folder: str, dem_name: str = "DEM.tif", ldd_name: str = "Ldd.map") -> Tuple[int, str, str]:
    """Gera um LDD (Local Drain Direction) a partir do DEM usando PCRaster no container.

    Saída: /input/<ldd_name> no formato PCRaster .map
    """
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{input_folder}:/input",
        "--entrypoint", "bash",
        "jrce1/lisflood:latest",
        "-lc",
        (
            "source /opt/conda/etc/profile.d/conda.sh && conda activate lisflood && "
            f"gdal_translate -q -of PCRaster -ot Float32 /input/{dem_name} /input/DEM.map && "
            "python - <<'PY'\n"
            "import pcraster as pcr\n"
            "pcr.setclone('/input/DEM.map')\n"
            "dem = pcr.readmap('/input/DEM.map')\n"
            "ldd = pcr.lddcreate(dem, 1e31, 1e31, 1e31, 1e31)\n"
            f"pcr.report(ldd, '/input/{ldd_name}')\n"
            "PY"
        )
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return res.returncode, res.stdout, res.stderr


def convert_geotiff_to_pcraster(input_folder: str, tif_name: str, map_name: str) -> Tuple[int, str, str]:
    """Converte um GeoTIFF para PCRaster .map usando GDAL dentro do container.

    Ex.: convert_geotiff_to_pcraster('/abs/input', 'Channels.tif', 'Channels.map')
    """
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{input_folder}:/input",
        "--entrypoint", "bash",
        "jrce1/lisflood:latest",
        "-lc",
        (
            "source /opt/conda/etc/profile.d/conda.sh && conda activate lisflood && "
            f"gdal_translate -q -of PCRaster -ot Float32 /input/{tif_name} /input/{map_name}"
        )
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return res.returncode, res.stdout, res.stderr


def convert_stl_to_dsm(dem_tif: str, stl_path: str, out_tif: str,
                       x_offset: float = 0.0, y_offset: float = 0.0,
                       z_scale: float = 1.0, z_offset: float = 0.0,
                       rot_deg: float = 0.0, chunk_rows: int = 128) -> Tuple[bool, str]:
    """Converte um STL (malha 3D) em um DSM GeoTIFF alinhado ao DEM.

    Estratégia: ray casting vertical (Z-) por linhas da grade do DEM; pega a 1ª interseção.
    Requer trimesh instalado. Retorna (ok, log_text).
    """
    try:
        import importlib
        trimesh = importlib.import_module("trimesh")  # type: ignore
        from rasterio.transform import Affine  # usa rio do escopo superior
    except ImportError as e:
        return False, f"Dependências ausentes para conversão STL→DSM (instale 'trimesh'): {e}"

    try:
        mesh = trimesh.load(stl_path, force='mesh')
        if mesh.is_empty:
            return False, "STL vazio ou inválido."
        # Transformação: escala Z, rotação Z, translação XYZ
        T = np.eye(4)
        # escala Z
        S = np.eye(4); S[2,2] = z_scale
        # rotação Z
        R = trimesh.transformations.rotation_matrix(np.deg2rad(rot_deg), [0,0,1])
        # translação
        Tr = np.eye(4); Tr[:3,3] = [x_offset, y_offset, z_offset]
        M = Tr @ R @ S @ T
        mesh.apply_transform(M)

        # Intersector
        try:
            ray_mod = importlib.import_module("trimesh.ray.ray_pyembree")
            RayMeshIntersector = getattr(ray_mod, "RayMeshIntersector")
            intersector = RayMeshIntersector(mesh)
        except ImportError:
            ray_mod = importlib.import_module("trimesh.ray.ray_triangle")
            RayMeshIntersector = getattr(ray_mod, "RayMeshIntersector")
            intersector = RayMeshIntersector(mesh)

        with rio.open(dem_tif) as src:
            dem = src.read(1)
            profile = src.profile.copy()
            transform: Affine = src.transform
            height, width = src.height, src.width

        dsm = dem.astype(np.float32).copy()
        # Altura máxima para origem dos raios
        zmax = float(mesh.bounds[1,2]) + 10.0

        # Pré-calcular X para colunas
        a, c = transform.a, transform.c
        e, f = transform.e, transform.f
        cols = np.arange(width, dtype=np.float64)
        xs = c + (cols + 0.5) * a

        for r0 in range(0, height, chunk_rows):
            r1 = min(height, r0 + chunk_rows)
            for r in range(r0, r1):
                y = f + (r + 0.5) * e
                origins = np.column_stack([
                    xs,
                    np.full(width, y, dtype=np.float64),
                    np.full(width, zmax, dtype=np.float64)
                ])
                directions = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float64), (width, 1))
                try:
                    locations, idx_ray, _ = intersector.intersects_location(origins, directions, multiple_hits=False)
                except (RuntimeError, ValueError) as ie:
                    return False, f"Falha no ray casting: {ie}"
                if len(idx_ray) > 0:
                    cols_hit = cols[idx_ray]
                    z_hit = locations[:, 2].astype(np.float32)
                    # Usa o máximo entre DEM e STL (estruturas sobre terreno)
                    dsm[r, cols_hit] = np.maximum(dsm[r, cols_hit], z_hit)

        profile.update(dtype='float32', count=1, compress='lzw', nodata=None)
        with rio.open(out_tif, 'w', **profile) as dst:
            dst.write(dsm, 1)
        return True, f"DSM gerado em {out_tif}"
    except (OSError, ValueError, RuntimeError) as e:
        return False, f"Erro na conversão STL→DSM: {e}"


def ensure_mask_pcraster(input_folder: str, tif_name: str = "MASK.tif", map_name: str = "MASK.map") -> Tuple[int, str, str]:
    """Gera MASK.map em PCRaster a partir de MASK.tif dentro do container.

    Retorna (exit_code, stdout, stderr)."""
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{input_folder}:/input",
        "--entrypoint", "bash",
        "jrce1/lisflood:latest",
        "-lc",
        (
            "source /opt/conda/etc/profile.d/conda.sh && conda activate lisflood && "
            f"gdal_translate -q -of PCRaster -ot Byte /input/{tif_name} /input/{map_name}"
        )
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return res.returncode, res.stdout, res.stderr


def create_constant_maps(input_folder: str, values: Dict[str, float]) -> Tuple[int, str, str]:
    """Cria mapas PCRaster constantes (Scalar) com base no clone /input/MASK.map.

    values: dict { 'MapName': value } criará /input/MapName.map com o valor.
    """
    # Monta script Python que recebe nomes/valores por env e cria os mapas
    import json
    payload = json.dumps(values)
    script = (
        "source /opt/conda/etc/profile.d/conda.sh && conda activate lisflood && "
        "MAPS='" + payload.replace("'", "'\"'\"'") + "' python - <<'PY'\n"
        "import os, json\n"
        "import pcraster as pcr\n"
        "maps = json.loads(os.environ.get('MAPS','{}'))\n"
        "pcr.setclone('/input/MASK.map')\n"
        "for name, val in maps.items():\n"
        "    m = pcr.scalar(float(val))\n"
        "    pcr.report(m, f'/input/{name}.map')\n"
        "print('Const maps criados:', ','.join(maps.keys()))\n"
        "PY"
    )
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{input_folder}:/input",
        "--entrypoint", "bash",
        "jrce1/lisflood:latest",
        "-lc",
        script,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return res.returncode, res.stdout, res.stderr