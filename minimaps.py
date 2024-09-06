from PIL import Image
import os
import hashlib
import chardet

# caminho_base = 'D:/doutorado/nexus'
caminho_base = 'C:/Users/Nufinha/work/seeu/web'
caminho_saida = 'D:/doutorado/lbp_example-main/dataset/javaprojetct02_encrypted/'


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
    hash_16_digit = f"{truncated_hash:016d}"
    return hash_16_digit


def list_files_recursive(path='.'):
    print (path)
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            list_files_recursive(full_path)
        else:
            geraImagem(full_path)


def geraImagem(caminho_arquivo):
    try:
        contador_linhas = 0
        linha_mais_larga = 0

        tipo = 'other'

        if caminho_arquivo.endswith(".sql"):
            tipo = 'sql'
        elif caminho_arquivo.endswith("IT.java"):
            tipo = 'javaintegrationtest'
        elif caminho_arquivo.endswith("Test.java"):
            tipo = 'javaunittest'
        elif caminho_arquivo.endswith(".xhtml"):
            tipo = 'javajsf'
        elif caminho_arquivo.endswith("DTO.java"):
            tipo = 'javadto'
        elif caminho_arquivo.endswith("TO.java"):
            tipo = 'javato'
        elif caminho_arquivo.endswith(".jrxml"):
            tipo = "javajasper"
        elif caminho_arquivo.endswith("Builder.java"):
            tipo = "javabuilder"
        elif caminho_arquivo.endswith("Impl.java"):  # vai dar problema....
            tipo = "javaimplementation"
        elif caminho_arquivo.endswith(".sh"):
            tipo = "sh"
        elif caminho_arquivo.endswith(".yml"):
            tipo = "yml"
        elif caminho_arquivo.endswith(".json"):
            tipo = "json"
        elif caminho_arquivo.endswith(".xml"):
            tipo = "xml"
        elif caminho_arquivo.endswith(".properties"):
            tipo = "properties"
        elif caminho_arquivo.endswith("DataProvider.java"):
            tipo = "javadataprovider"
        elif caminho_arquivo.endswith(".svg"):
            tipo = "svg"
        elif caminho_arquivo.endswith("Configuration.java"):
            tipo = "javaconfiguration"
        elif caminho_arquivo.endswith(".js"):
            tipo = "js"
        elif caminho_arquivo.endswith(".css"):
            tipo = "css"
        elif caminho_arquivo.endswith("Converter.java"):
            tipo = "javaconverter"
        elif caminho_arquivo.endswith(".jsp"):
            tipo = "javajsp"
        elif caminho_arquivo.endswith(".htm"):
            tipo = "html"
        elif caminho_arquivo.endswith(".html"):
            tipo = "html"

        with open(caminho_arquivo, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        with open(caminho_arquivo, 'r', encoding=encoding) as arquivo:
            for linha in arquivo:
                if caminho_arquivo.endswith(".java"):
                    if '@Entity' in linha:
                        tipo = 'javaentity'
                    elif '@Repository' in linha:
                        tipo = 'javarepository'
                    elif '@Service' in linha:
                        tipo = 'javaservice'
                    elif '@Controller' in linha:
                        tipo = 'javacontroller'
                    elif '@RestController' in linha:
                        tipo = 'javarestcontroller'
                    elif '@RestController' in linha:
                        tipo = 'javarestcontroller'
                    elif '@Interface' in linha:
                        tipo = 'javaannotation'
                    elif 'Interface' in linha:
                        tipo = 'javainterface'

                contador_linhas += 1  # Conta as linhas
                linha_mais_larga = max(linha_mais_larga, len(linha.strip()))

        if tipo == 'other':
            return

        largura, altura = linha_mais_larga, contador_linhas
        imagem = Image.new('RGB', (largura, altura), 'black')
        y = 0

        with open(caminho_arquivo, 'r', encoding=encoding) as arquivo:
            for linha in arquivo:
                for x in range(len(linha.strip())):
                    v = ord(linha[x])
                    v = encrypt_chars(v)
                    if v > 255:
                        v = 255
                    try:
                        imagem.putpixel((x, y), (v, v, v))
                    except IndexError:
                        print(largura, altura, x, y, v)
                y += 1

        caminho_saida_com_tipo = caminho_saida + '/' + tipo + '/'
        os.makedirs(caminho_saida_com_tipo, exist_ok=True)
        arquivoImagem = create_16_digit_hash(caminho_arquivo) + '.png'
        imagem.save(caminho_saida_com_tipo + arquivoImagem)

    except Exception as ex:
        print('ignorando', caminho_arquivo, ex)


list_files_recursive(caminho_base)
