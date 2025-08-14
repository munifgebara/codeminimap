from PIL import Image
import os
import sys
import csv
import json
import hashlib
import chardet
from multiprocessing import Pool, Value, Lock
import subprocess

# ===================== Configurações =====================
caminho_base = '/home/munif/dataset'
caminho_saida = '/mnt/7937a629-8811-4af0-ae7a-097a78cc4d2c/datasets/99Dataset'
encrypt_flag = True

if encrypt_flag:
    caminho_saida = caminho_saida + '_encrypted'

fixed_size = True
if fixed_size:
    caminho_saida = caminho_saida + '_fixed_size'

ARQUIVO_CSV = os.path.join(caminho_saida, 'catalogo.csv')
ARQ_METADATA = 'metadata.json'  # nome esperado do arquivo de metadados na raiz do repo

# Dicionário de extensões conhecidas -> tipo
EXTENSOES = {
    '.adoc': 'asciidoc', '.asciidoc': 'asciidoc', '.asm': 'asm', '.bat': 'batch',
    '.c': 'c', '.cc': 'cpp', '.cfg': 'config', '.clj': 'clojure', '.cmake': 'cmake',
    '.coffee': 'coffeescript', '.conf': 'config', '.cpp': 'cpp', '.cs': 'csharp',
    '.css': 'css', '.cxx': 'cpp', '.dart': 'dart', '.dockerfile': 'dockerfile',
    '.editorconfig': 'editorconfig', '.ejs': 'ejs', '.env': 'env',
    '.gitattributes': 'gitattributes', '.gitignore': 'gitignore', '.go': 'go',
    '.gradle': 'gradle', '.graphql': 'graphql', '.groovy': 'groovy', '.h': 'c-header',
    '.hh': 'cpp-header', '.hpp': 'cpp-header', '.htm': 'html', '.html': 'html',
    '.hxx': 'cpp-header', '.ini': 'ini', '.java': 'java', '.jl': 'julia', '.js': 'js',
    '.json': 'json', '.json5': 'json', '.jsonc': 'json', '.jsp': 'javajsp',
    '.jsx': 'javascript-jsx', '.kt': 'kotlin', '.kts': 'kotlin-script', '.less': 'less',
    '.lua': 'lua', '.m': 'objc', '.makefile': 'makefile', '.md': 'markdown',
    '.mk': 'makefile', '.mm': 'objc-cpp', '.php': 'php', '.pl': 'perl', '.pm': 'perl',
    '.properties': 'properties', '.ps1': 'powershell', '.psql': 'sql', '.py': 'python',
    '.r': 'r', '.rb': 'ruby', '.rs': 'rust', '.rst': 'rst', '.s': 'asm', '.sass': 'sass',
    '.scala': 'scala', '.scss': 'scss', '.sh': 'sh', '.sql': 'sql', '.svg': 'svg',
    '.swift': 'swift', '.toml': 'toml', '.ts': 'typescript', '.tsx': 'typescript-jsx',
    '.twig': 'twig', '.txt': 'text', '.vue': 'vue', '.xml': 'xml', '.xsl': 'xsl',
    '.yaml': 'yaml', '.yml': 'yaml'
}
# =========================================================

# --- Globais compartilhadas para progresso/CSV ---
_progress_val = None
_progress_total = None
_progress_lock = None
_csv_lock = None

def init_globals(progress_val, progress_total, progress_lock, csv_lock):
    """Inicializador do Pool para compartilhar locks/contadores."""
    global _progress_val, _progress_total, _progress_lock, _csv_lock
    _progress_val = progress_val
    _progress_total = progress_total
    _progress_lock = progress_lock
    _csv_lock = csv_lock

# ===================== Utilidades =====================

def encrypt_chars(asc_code: int) -> int:
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

def create_16_digit_hash(input_string: str) -> str:
    input_bytes = input_string.encode('ISO-8859-1', errors='ignore')
    md5_hash = hashlib.md5(input_bytes).hexdigest()
    hash_int = int(md5_hash, 16)
    truncated_hash = hash_int % (10 ** 16)
    return f"{truncated_hash:016d}"

def processar_linhas(caminho_arquivo: str, encoding_detectada: str):
    """Lê o arquivo e retorna linhas, contagem, e maior largura (com fallbacks de encoding)."""
    contador_linhas = 0
    linha_mais_larga = 0
    linhas = []
    encs = [encoding_detectada or 'utf-8', 'utf-8', 'latin-1']
    for enc in encs:
        try:
            with open(caminho_arquivo, 'r', encoding=enc, errors='strict') as arquivo:
                for linha in arquivo:
                    s = linha.rstrip('\n\r')
                    linhas.append(s)
                    contador_linhas += 1
                    if len(s) > linha_mais_larga:
                        linha_mais_larga = len(s)
            return linhas, contador_linhas, linha_mais_larga
        except Exception:
            continue
    # Último recurso: ignora erros
    with open(caminho_arquivo, 'r', encoding='utf-8', errors='ignore') as arquivo:
        for linha in arquivo:
            s = linha.rstrip('\n\r')
            linhas.append(s)
            contador_linhas += 1
            if len(s) > linha_mais_larga:
                linha_mais_larga = len(s)
    return linhas, contador_linhas, linha_mais_larga

def is_text_file_quick(path: str, sample_size: int = 65536) -> bool:
    """
    Heurística rápida para decidir se é arquivo de texto:
    - lê até 64KB
    - se contém NUL -> binário
    - tenta decodificar como UTF-8; se falhar, usa latin-1 e mede taxa de imprimíveis
    """
    try:
        with open(path, 'rb') as f:
            data = f.read(sample_size)
        if b'\x00' in data:
            return False
        try:
            data.decode('utf-8')
            return True
        except UnicodeDecodeError:
            decoded = data.decode('latin-1', errors='ignore')
            printable = sum(32 <= ord(c) <= 126 or c in '\r\n\t' for c in decoded)
            ratio = printable / max(1, len(decoded))
            return ratio > 0.85
    except Exception:
        return False

def listar_arquivos_classificar(base_dir: str):
    """Retorna:
       - lista [(caminho_absoluto, tipo)] apenas para extensões conhecidas
       - set de extensões de arquivos de texto desconhecidas
    """
    conhecidos = []
    desconhecidas_texto = set()
    for root, dirs, files in os.walk(base_dir):
        if '.git' in root:
            continue
        for nome in files:
            caminho = os.path.join(root, nome)
            ext = os.path.splitext(nome)[1].lower()
            if ext in EXTENSOES:
                conhecidos.append((caminho, EXTENSOES[ext]))
            else:
                if is_text_file_quick(caminho):
                    desconhecidas_texto.add(ext if ext else '(sem extensão)')
    return conhecidos, desconhecidas_texto

def escrever_extensoes_desconhecidas(colecao_exts, destino_dir: str):
    try:
        os.makedirs(destino_dir, exist_ok=True)
        saida = os.path.join(destino_dir, 'extensoes_texto_desconhecidas.txt')
        with open(saida, 'w', encoding='utf-8', newline='') as f:
            for ext in sorted(colecao_exts):
                f.write(f"{ext}\n")
        print(f"Extensões de texto desconhecidas salvas em: {saida}")
    except Exception as e:
        print("Falha ao salvar extensões desconhecidas:", e, file=sys.stderr)

def decompor_caminho_para_csv(caminho_arquivo: str):
    """
    Retorna (projeto_original, caminho_dentro_do_projeto, nome_arquivo_original).
    'projeto_original' é o primeiro diretório após caminho_base.
    """
    rel = os.path.relpath(caminho_arquivo, start=caminho_base)
    partes = rel.split(os.sep)
    if len(partes) == 1:
        projeto = '(raiz)'
        nome = partes[0]
        caminho_dentro = ''
    else:
        projeto = partes[0]
        nome = partes[-1]
        caminho_dentro = os.path.join(*partes[1:-1]) if len(partes) > 2 else ''
    return projeto, caminho_dentro, nome

# -------- Autor via metadata.json --------

def _deep_find_authorish(obj):
    """
    Procura, de forma tolerante, algo que pareça 'autor/owner' dentro do JSON.
    Cobre padrões comuns:
      - 'owner', 'author', 'user', 'organization'
      - dicionários com chaves 'login', 'name', 'username'
      - estruturas aninhadas como repository.owner.login
    Retorna string ou None.
    """
    CAND_KEYS = {'owner', 'author', 'user', 'organization', 'repo_owner', 'maintainer'}
    USER_KEYS = ('login', 'name', 'username', 'user', 'owner', 'display_name')

    def extract_from_owner_dict(d):
        # tenta várias chaves plausíveis
        for k in USER_KEYS:
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # como fallback, primeira string do dict
        for v in d.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    # 1) tentativas diretas em dict de topo
    if isinstance(obj, dict):
        # caso clássico: repository → owner
        repo = obj.get('repository')
        if isinstance(repo, dict):
            own = repo.get('owner')
            if isinstance(own, dict):
                cand = extract_from_owner_dict(own)
                if cand:
                    return cand
            if isinstance(own, str) and own.strip():
                return own.strip()

        # checa chaves candidatas no topo
        for k in CAND_KEYS:
            if k in obj:
                v = obj[k]
                if isinstance(v, dict):
                    cand = extract_from_owner_dict(v)
                    if cand:
                        return cand
                elif isinstance(v, str) and v.strip():
                    return v.strip()

        # alguns dumps trazem 'owner' em lista (ex.: múltiplos mantenedores)
        for k in CAND_KEYS:
            v = obj.get(k)
            if isinstance(v, list) and v:
                # pega o primeiro que for string/dict aproveitável
                for item in v:
                    if isinstance(item, str) and item.strip():
                        return item.strip()
                    if isinstance(item, dict):
                        cand = extract_from_owner_dict(item)
                        if cand:
                            return cand

        # fallback: varre recursivamente
        for v in obj.values():
            res = _deep_find_authorish(v)
            if res:
                return res

    elif isinstance(obj, list):
        for it in obj:
            res = _deep_find_authorish(it)
            if res:
                return res

    return None

def obter_autor_via_metadata(repo_dir: str) -> str | None:
    """Tenta abrir <repo_dir>/metadata.json e extrair um nome de autor/owner de forma tolerante."""
    meta_path = os.path.join(repo_dir, ARQ_METADATA)
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        autor = _deep_find_authorish(data)
        if autor and isinstance(autor, str):
            return autor.strip()
    except Exception:
        return None
    return None

def obter_autor_projeto(caminho_arquivo: str) -> str:
    """
    Autor:
      1) Se houver .git no diretório do projeto → último autor do arquivo (git log).
      2) Caso contrário → lê <repo>/metadata.json e tenta extrair owner/autor.
      3) Caso não encontre → '(desconhecido)'.
    """
    projeto, _, _ = decompor_caminho_para_csv(caminho_arquivo)
    if projeto == '(raiz)':
        return '(desconhecido)'

    repo_dir = os.path.join(caminho_base, projeto)
    git_dir = os.path.join(repo_dir, '.git')

    # 1) Tenta pelo git
    if os.path.isdir(git_dir):
        try:
            autor = subprocess.check_output(
                ['git', '-C', repo_dir, 'log', '--format=%an', '-n', '1', '--', caminho_arquivo],
                stderr=subprocess.DEVNULL
            ).decode('utf-8', errors='ignore').strip()
            if autor:
                return autor
        except Exception:
            pass

    # 2) metadata.json
    autor_meta = obter_autor_via_metadata(repo_dir)
    if autor_meta:
        return autor_meta

    # 3) Fallback
    return '(desconhecido)'

# -------- CSV / progresso --------

def append_csv_row(novo_nome: str, tipo: str, projeto: str, caminho_dentro: str, nome_original: str, autor: str):
    """Acrescenta linha no CSV de forma thread-safe e cria header se necessário."""
    with _csv_lock:
        os.makedirs(os.path.dirname(ARQUIVO_CSV), exist_ok=True)
        write_header = not os.path.exists(ARQUIVO_CSV) or os.path.getsize(ARQUIVO_CSV) == 0
        with open(ARQUIVO_CSV, 'a', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    'novo nome', 'tipo', 'projeto original',
                    'caminho dentro do projeto', 'nome do arquivo original', 'autor'
                ])
            w.writerow([novo_nome, tipo, projeto, caminho_dentro, nome_original, autor])

def atualizar_progresso():
    with _progress_lock:
        _progress_val.value += 1
        print(f"[{_progress_val.value} de {_progress_total.value}] processado", flush=True)

def get_unique_filename(dest_dir: str, base_name_no_ext: str, ext: str = '.png') -> str:
    """Gera um nome de arquivo único no diretório de destino, serializando a checagem via lock."""
    with _csv_lock:
        candidate = f"{base_name_no_ext}{ext}"
        n = 1
        while os.path.exists(os.path.join(dest_dir, candidate)):
            candidate = f"{base_name_no_ext}_{n}{ext}"
            n += 1
        return candidate

# ===================== Núcleo do processamento =====================

def gera_imagem_e_csv(caminho_arquivo: str, tipo_destino: str):
    try:
        # Detecta encoding uma vez
        with open(caminho_arquivo, 'rb') as f:
            raw = f.read()
            result = chardet.detect(raw)
            encoding = (result or {}).get('encoding')

        linhas, contador_linhas, linha_mais_larga = processar_linhas(caminho_arquivo, encoding)

        # Define dimensões da imagem
        if fixed_size:
            largura, altura = 128, 128
        else:
            border = 8
            largura = min(linha_mais_larga + 2 * border, 512)
            altura = min(contador_linhas + 2 * border, 512)

        border = 8
        imagem = Image.new('RGB', (largura, altura), 'black')
        y = 0

        for linha in linhas:
            max_x = min(len(linha), largura - 2 * border)
            for x in range(max_x):
                v = ord(linha[x])
                v = encrypt_chars(v)
                if v > 255:
                    v = 255
                try:
                    imagem.putpixel((x + border, y + border), (v, v, v))
                except IndexError:
                    break
            y += 1
            if (y + 2 * border) >= altura:
                break

        # Pastas e nome único
        caminho_tipo = os.path.join(caminho_saida, tipo_destino)
        os.makedirs(caminho_tipo, exist_ok=True)
        base_hash = create_16_digit_hash(caminho_arquivo)
        novo_nome = get_unique_filename(caminho_tipo, base_hash, '.png')
        caminho_final = os.path.join(caminho_tipo, novo_nome)
        imagem.save(caminho_final)

        # Metadados -> CSV
        projeto, caminho_dentro, nome_original = decompor_caminho_para_csv(caminho_arquivo)
        autor = obter_autor_projeto(caminho_arquivo)
        append_csv_row(novo_nome, tipo_destino, projeto, caminho_dentro, nome_original, autor)

    except Exception as ex:
        print('ignorando', caminho_arquivo, ex, file=sys.stderr)

def processar_arquivo(args):
    caminho_arquivo, tipo_destino = args
    gera_imagem_e_csv(caminho_arquivo, tipo_destino)
    atualizar_progresso()

# ===================== Main =====================

def main():
    # 1) Classificar arquivos
    conhecidos, desconhecidas_texto = listar_arquivos_classificar(caminho_base)

    total = len(conhecidos)
    print(f"Arquivos a processar (extensões conhecidas): {total}")

    # 2) Salvar extensões desconhecidas que parecem texto
    escrever_extensoes_desconhecidas(desconhecidas_texto, caminho_saida)

    if total == 0:
        return

    # 3) Inicializa CSV com header (uma vez)
    os.makedirs(caminho_saida, exist_ok=True)
    if not os.path.exists(ARQUIVO_CSV) or os.path.getsize(ARQUIVO_CSV) == 0:
        with open(ARQUIVO_CSV, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'novo nome', 'tipo', 'projeto original',
                'caminho dentro do projeto', 'nome do arquivo original', 'autor'
            ])

    # 4) Pool com progresso e lock de CSV
    progress_val = Value('i', 0)
    progress_total = Value('i', total)
    progress_lock = Lock()
    csv_lock = Lock()

    with Pool(
        processes=os.cpu_count() or 4,
        initializer=init_globals,
        initargs=(progress_val, progress_total, progress_lock, csv_lock)
    ) as pool:
        pool.map(processar_arquivo, conhecidos)

    print("Concluído.")
    print(f"CSV salvo em: {ARQUIVO_CSV}")

if __name__ == "__main__":
    main()
