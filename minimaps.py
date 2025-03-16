from PIL import Image
import os
import hashlib
import chardet
from multiprocessing import Pool

caminho_base = '/home/munif-gebara-junior/temp'
caminho_saida = 'dataset/novo'
encrypt_flag = True

if encrypt_flag:
    caminho_saida = caminho_saida + '_encrypted'

fixed_size = False
if fixed_size:
    caminho_saida = caminho_saida + '_fixed_size'


def encrypt_chars(asc_code):
    if asc_code < 33:  # não-imprimíveis
        return 32
    if asc_code < 48:  # símbolos
        return asc_code
    if asc_code < 58:  # números
        return 53
    if asc_code < 65:  # símbolos
        return asc_code
    if asc_code < 91:  # maiúsculas
        return 77
    if asc_code < 97:  # símbolos
        return asc_code
    if asc_code < 123:  # minúsculas
        return 109
    if asc_code < 127:  # símbolos
        return asc_code
    return 130


def create_16_digit_hash(input_string):
    input_bytes = input_string.encode('ISO-8859-1')
    md5_hash = hashlib.md5(input_bytes).hexdigest()
    hash_int = int(md5_hash, 16)
    truncated_hash = hash_int % (10 ** 16)
    return f"{truncated_hash:016d}"


def processar_linhas(caminho_arquivo, encoding):
    """Lê o arquivo apenas uma vez e retorna as linhas processadas"""
    contador_linhas = 0
    linha_mais_larga = 0
    linhas = []

    with open(caminho_arquivo, 'r', encoding=encoding) as arquivo:
        for linha in arquivo:
            linhas.append(linha.strip())
            contador_linhas += 1
            linha_mais_larga = max(linha_mais_larga, len(linha.strip()))

    return linhas, contador_linhas, linha_mais_larga


def geraImagem(caminho_arquivo):
    try:
        if any(caminho_arquivo.endswith(ext) for ext in [".zip", ".png", ".jpg"]) or '.git' in caminho_arquivo:
            return

        with open(caminho_arquivo, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        linhas, contador_linhas, linha_mais_larga = processar_linhas(caminho_arquivo, encoding)

        if fixed_size:
            largura, altura = 128, 128
        else:
            largura, altura = min(linha_mais_larga + 2 * 8, 512), min(contador_linhas + 2 * 8, 512)

        border = 8
        imagem = Image.new('RGB', (largura, altura), 'black')
        y = 0

        for linha in linhas:
            for x in range(len(linha)):
                if (x + 2 * border) >= largura:
                    break
                v = ord(linha[x])
                v = encrypt_chars(v)
                v = min(v, 255)
                try:
                    imagem.putpixel((x + border, y + border), (v, v, v))
                except IndexError:
                    print(largura, altura, x, y, v)
            y += 1
            if (y + 2 * border) >= altura:
                break

        tipo = "other"
        extensoes = {
            ".sql": "sql", ".sh": "sh", ".yml": "yml", ".json": "json", ".xml": "xml", ".properties": "properties",
            ".svg": "svg", ".js": "js", ".css": "css", ".jsp": "javajsp", ".htm": "html", ".html": "html"
        }
        for ext, tipo_detectado in extensoes.items():
            if caminho_arquivo.endswith(ext):
                tipo = tipo_detectado
                break

        caminho_saida_com_tipo = os.path.join(caminho_saida, tipo)
        os.makedirs(caminho_saida_com_tipo, exist_ok=True)
        arquivoImagem = create_16_digit_hash(caminho_arquivo) + '.png'
        imagem.save(os.path.join(caminho_saida_com_tipo, arquivoImagem))
    except Exception as ex:
        print('ignorando', caminho_arquivo, ex)


def processar_arquivo(caminho_arquivo):
    geraImagem(caminho_arquivo)


def list_files_parallel(path):
    arquivos = []
    for root, _, files in os.walk(path):
        for file in files:
            arquivos.append(os.path.join(root, file))

    with Pool(processes=4) as pool:  # Processamento paralelo
        pool.map(processar_arquivo, arquivos)


list_files_parallel(caminho_base)
