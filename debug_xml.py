import os
import xml.etree.ElementTree as ET

def debug_xml(xml_path):
    """Verifica se o XML está correto"""
    print("=== DEBUG DO ARQUIVO XML ===")

    if not os.path.exists(xml_path):
        print(f"❌ Arquivo não existe: {xml_path}")
        return

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        print(f"✅ XML válido: {xml_path}")

        # Verificar parâmetros essenciais
        essential_params = ['DtSec', 'DtInit', 'CalendarConvention']

        for param in essential_params:
            found = False
            for elem in root.iter('textvar'):
                if elem.get('name') == param:
                    value = elem.get('value')
                    print(f"✅ {param}: {value}")
                    found = True
                    break
            if not found:
                print(f"❌ {param}: NÃO ENCONTRADO")

    except Exception as e:
        print(f"❌ Erro ao ler XML: {e}")

if __name__ == "__main__":
    # Exemplo de uso local (ajuste o caminho se necessário)
    debug_xml("input/lisflood_config.xml")
