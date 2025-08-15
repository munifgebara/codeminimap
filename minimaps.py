from PIL import Image
import os
import sys
import csv
import json
import hashlib
import chardet
import subprocess
from multiprocessing import Pool, Value, Lock, Manager
from typing import Optional, Tuple

# ===================== Configurações =====================
caminho_base = '/home/munif/dataset2'  # raiz onde estão as pastas <Linguagem>/<owner>__<repo>__<ref>
caminho_saida = '/mnt/7937a629-8811-4af0-ae7a-097a78cc4d2c/datasets/100Dataset'
encrypt_flag = True

if encrypt_flag:
    caminho_saida = caminho_saida + '_encrypted'

fixed_size = True
if fixed_size:
    caminho_saida = caminho_saida + '_fixed_size'

ARQUIVO_CSV = os.path.join(caminho_saida, 'catalogo.csv')
ARQ_METADATA = 'metadata.json'
# =========================================================

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

# --- Globais compartilhadas (multiprocessing) ---
_progress_val = None
_progress_total = None
_progress_lock = None
_csv_lock = None
_repo_root_cache = None   # caminho_arquivo -> repo_root
_author_cache = None      # (repo_root, relpath) -> autor

def init_globals(progress_val, progress_total, progress_lock, csv_lock, repo_root_cache, author_cache):
    global _progress_val, _progress_total, _progress_lock, _csv_lock, _repo_root_cache, _author_cache
    _progress_val = progress_val
    _progress_total = progress_total
    _progress_lock = progress_lock
    _csv_lock = csv_lock
    _repo_root_cache = repo_root_cache
    _author_cache = author_cache

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

def processar_linhas(caminho_arquivo: str, encoding_detectada: Optional[str]):
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
    # Último recurso
    with open(caminho_arquivo, 'r', encoding='utf-8', errors='ignore') as arquivo:
        for linha in arquivo:
            s = linha.rstrip('\n\r')
            linhas.append(s)
            contador_linhas += 1
            if len(s) > linha_mais_larga:
                linha_mais_larga = len(s)
    return linhas, contador_linhas, linha_mais_larga

def is_text_file_quick(path: str, sample_size: int = 65536) -> bool:
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
    conhecidos = []
    desconhecidas_texto = set()
    for root, dirs, files in os.walk(base_dir):
        # ignora controles de git
        if os.path.basename(root) in ('.git',):
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

def find_repo_root(path_file: str) -> Optional[str]:
    """Sobe a partir do arquivo até encontrar um diretório contendo '.git'."""
    # cache
    cached = _repo_root_cache.get(path_file) if _repo_root_cache is not None else None
    if cached:
        return cached

    base_abs = os.path.abspath(caminho_base)
    cur = os.path.abspath(os.path.dirname(path_file))
    while True:
        if os.path.isdir(os.path.join(cur, '.git')):
            if _repo_root_cache is not None:
                _repo_root_cache[path_file] = cur
            return cur
        parent = os.path.dirname(cur)
        if parent == cur or os.path.commonpath([parent, base_abs]) != base_abs:
            break
        cur = parent
    return None

def load_author_from_metadata(repo_root: str) -> Optional[str]:
    """Tenta extrair owner/autor de <repo_root>/metadata.json."""
    meta_path = os.path.join(repo_root, ARQ_METADATA)
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # tentativas comuns
        repo = data.get('repository') or data.get('repo') or {}
        own = repo.get('owner') or data.get('owner') or {}
        # owner pode ser string ou dict
        if isinstance(own, str) and own.strip():
            return own.strip()
        if isinstance(own, dict):
            for k in ('login', 'name', 'username', 'user', 'owner', 'display_name'):
                v = own.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        # full_name: owner/repo
        full = (repo if isinstance(repo, dict) else {}).get('full_name') or data.get('full_name')
        if isinstance(full, str) and '/' in full:
            return full.split('/', 1)[0].strip()
    except Exception:
        return None
    return None

def obter_autor_git(path_file: str) -> Optional[str]:
    """Retorna autor via git log (último commit que tocou o arquivo) ou None."""
    repo_root = find_repo_root(path_file)
    if not repo_root:
        return None
    rel = os.path.relpath(path_file, repo_root)
    try:
        out = subprocess.check_output(
            ['git', '-C', repo_root, 'log', '--format=%an', '-n', '1', '--', rel],
            stderr=subprocess.DEVNULL
        ).decode('utf-8', errors='ignore').strip()
        return out or None
    except Exception:
        return None

def obter_autor_projeto(path_file: str) -> str:
    """Autor por ordem: git -> metadata.json -> '(desconhecido)' (com cache por arquivo)."""
    key = None
    repo_root = find_repo_root(path_file)
    if repo_root:
        rel = os.path.relpath(path_file, repo_root)
        key = (repo_root, rel)

    if key and _author_cache is not None:
        cached = _author_cache.get(str(key))
        if isinstance(cached, str) and cached:
            return cached

    # 1) Git
    autor = obter_autor_git(path_file)
    if autor:
        if key and _author_cache is not None:
            _author_cache[str(key)] = autor
        return autor

    # 2) metadata.json na raiz do repo
    if repo_root:
        autor_meta = load_author_from_metadata(repo_root)
        if autor_meta:
            if key and _author_cache is not None:
                _author_cache[str(key)] = autor_meta
            return autor_meta

    # 3) fallback
    if key and _author_cache is not None:
        _author_cache[str(key)] = '(desconhecido)'
    return '(desconhecido)'

def decompor_para_csv(caminho_arquivo: str) -> Tuple[str, str, str]:
    """
    Retorna (projeto_original, caminho_dentro_do_projeto, nome_arquivo_original)
    Onde 'projeto_original' é a pasta do repo (ex.: owner__repo__ref).
    """
    rel_to_base = os.path.relpath(caminho_arquivo, start=caminho_base)
    partes = rel_to_base.split(os.sep)
    # Estrutura esperada: <LinguagemSlug>/<owner>__<repo>__<ref>/...
    if len(partes) >= 2:
        projeto = partes[1]
        nome = partes[-1]
        # caminho dentro do projeto = tudo após <LinguagemSlug>/<owner>__<repo>__<ref> sem o nome
        if len(partes) > 2:
            caminho_dentro = os.path.join(*partes[2:-1]) if len(partes) > 3 else ''
        else:
            caminho_dentro = ''
        return projeto, caminho_dentro, nome
    # fallback genérico
    projeto = '(desconhecido)'
    nome = partes[-1] if partes else os.path.basename(caminho_arquivo)
    caminho_dentro = ''
    return projeto, caminho_dentro, nome

def append_csv_row(novo_nome: str, tipo: str, projeto: str, caminho_dentro: str, nome_original: str, autor: str):
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
        # Detecta encoding
        with open(caminho_arquivo, 'rb') as f:
            raw = f.read()
            result = chardet.detect(raw)
            encoding = (result or {}).get('encoding')

        linhas, contador_linhas, linha_mais_larga = processar_linhas(caminho_arquivo, encoding)

        # Dimensões
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

        # Pasta destino por tipo e nome único
        caminho_tipo = os.path.join(caminho_saida, tipo_destino)
        os.makedirs(caminho_tipo, exist_ok=True)
        base_hash = create_16_digit_hash(caminho_arquivo)
        novo_nome = get_unique_filename(caminho_tipo, base_hash, '.png')
        imagem.save(os.path.join(caminho_tipo, novo_nome))

        # Metadados -> CSV
        projeto, caminho_dentro, nome_original = decompor_para_csv(caminho_arquivo)
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
    # 1) Seleção de arquivos
    conhecidos, desconhecidas_texto = listar_arquivos_classificar(caminho_base)
    total = len(conhecidos)
    print(f"Arquivos a processar (extensões conhecidas): {total}")

    # 2) Escreve extensões desconhecidas
    escrever_extensoes_desconhecidas(desconhecidas_texto, caminho_saida)

    if total == 0:
        return

    # 3) Inicializa CSV se vazio
    os.makedirs(caminho_saida, exist_ok=True)
    if not os.path.exists(ARQUIVO_CSV) or os.path.getsize(ARQUIVO_CSV) == 0:
        with open(ARQUIVO_CSV, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'novo nome', 'tipo', 'projeto original',
                'caminho dentro do projeto', 'nome do arquivo original', 'autor'
            ])

    # 4) Pool com locks e caches
    manager = Manager()
    progress_val = Value('i', 0)
    progress_total = Value('i', total)
    progress_lock = Lock()
    csv_lock = Lock()
    repo_root_cache = manager.dict()
    author_cache = manager.dict()

    with Pool(
        processes=os.cpu_count() or 4,
        initializer=init_globals,
        initargs=(progress_val, progress_total, progress_lock, csv_lock, repo_root_cache, author_cache)
    ) as pool:
        pool.map(processar_arquivo, conhecidos)

    print("Concluído.")
    print(f"CSV salvo em: {ARQUIVO_CSV}")

if __name__ == "__main__":
    main()
