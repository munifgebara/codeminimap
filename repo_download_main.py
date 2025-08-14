#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baixa 100 repositórios (10 por linguagem) e salva cada repo em uma pasta da linguagem.
Gera metadata.json por repositório e arquivos agregados CSV/JSONL na pasta DESTINO.

Melhorias:
- Usa token do GitHub para evitar rate-limit: lê de $GITHUB_TOKEN ou de ~/.github_token (1ª linha).
- Organiza saída em /DESTINO/<LINGUAGEM>/<owner>__<repo>__<ref>/
"""

import csv
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import unicodedata
import re


from repo_dowload_plus import GithubDownloader, process_one

DESTINO = "/home/munif/dataset"
MAX_WORKERS = 3            # paralelismo de DOWNLOAD
META_WORKERS = 1           # paralelismo de METADADOS (1 ajuda a evitar 403/429)
MAX_ZIP_MB = 200
FALLBACK_GIT = True


# ------------------------
# Lista: 10 linguagens x 10 aplicações (100)
# ------------------------
LANG_REPOS: Dict[str, List[str]] = {
    "JavaScript": [
        "https://github.com/freeCodeCamp/freeCodeCamp",
        "https://github.com/TryGhost/Ghost",
        "https://github.com/RocketChat/Rocket.Chat",
        "https://github.com/ether/etherpad-lite",
        "https://github.com/node-red/node-red",
        "https://github.com/NodeBB/NodeBB",
        "https://github.com/jitsi/jitsi-meet",
        "https://github.com/vercel/hyper",
        "https://github.com/standardnotes/app",
        "https://github.com/BoostIO/Boostnote",
    ],
    "Python": [
        "https://github.com/home-assistant/core",
        "https://github.com/odoo/odoo",
        "https://github.com/yt-dlp/yt-dlp",
        "https://github.com/getsentry/sentry",
        "https://github.com/saltstack/salt",
        "https://github.com/ansible/ansible",
        "https://github.com/paperless-ngx/paperless-ngx",
        "https://github.com/kovidgoyal/calibre",
        "https://github.com/commaai/openpilot",
        "https://github.com/openai/whisper",
    ],
    "Java": [
        "https://github.com/jenkinsci/jenkins",
        "https://github.com/keycloak/keycloak",
        "https://github.com/elastic/elasticsearch",
        "https://github.com/apache/solr",
        "https://github.com/trinodb/trino",
        "https://github.com/apache/hadoop",
        "https://github.com/apache/nifi",
        "https://github.com/geoserver/geoserver",
        "https://github.com/Graylog2/graylog2-server",
        "https://github.com/apache/cassandra",
    ],
    "TypeScript": [
        "https://github.com/microsoft/vscode",
        "https://github.com/desktop/desktop",
        "https://github.com/outline/outline",
        "https://github.com/directus/directus",
        "https://github.com/jgraph/drawio",
        "https://github.com/excalidraw/excalidraw",
        "https://github.com/laurent22/joplin",
        "https://github.com/immich-app/immich",
        "https://github.com/openlens/openlens",
        "https://github.com/supabase/supabase",
    ],
    "C#": [
        "https://github.com/jellyfin/jellyfin",
        "https://github.com/ShareX/ShareX",
        "https://github.com/files-community/Files",
        "https://github.com/icsharpcode/ILSpy",
        "https://github.com/dnSpyEx/dnSpy",
        "https://github.com/Jackett/Jackett",
        "https://github.com/Radarr/Radarr",
        "https://github.com/Sonarr/Sonarr",
        "https://github.com/Ombi-app/Ombi",
        "https://github.com/bitwarden/server",
    ],
    "C++": [
        "https://github.com/bitcoin/bitcoin",
        "https://github.com/obsproject/obs-studio",
        "https://github.com/notepad-plus-plus/notepad-plus-plus",
        "https://github.com/qbittorrent/qBittorrent",
        "https://github.com/telegramdesktop/tdesktop",
        "https://github.com/godotengine/godot",
        "https://github.com/musescore/MuseScore",
        "https://github.com/qgis/QGIS",
        "https://github.com/KDE/krita",
        "https://github.com/openttd/openttd",
    ],
    "PHP": [
        "https://github.com/WordPress/WordPress",
        "https://github.com/nextcloud/server",
        "https://github.com/matomo-org/matomo",
        "https://github.com/phpmyadmin/phpmyadmin",
        "https://github.com/moodle/moodle",
        "https://github.com/BookStackApp/BookStack",
        "https://github.com/flarum/flarum",
        "https://github.com/LycheeOrg/Lychee",
        "https://github.com/osTicket/osTicket",
        "https://github.com/pterodactyl/panel",
    ],
    "C": [
        "https://github.com/torvalds/linux",
        "https://github.com/FFmpeg/FFmpeg",
        "https://github.com/neovim/neovim",
        "https://github.com/vim/vim",
        "https://github.com/curl/curl",
        "https://github.com/tmux/tmux",
        "https://github.com/redis/redis",
        "https://github.com/nginx/nginx",
        "https://github.com/git/git",
        "https://github.com/systemd/systemd",
    ],
    "Go": [
        "https://github.com/kubernetes/kubernetes",
        "https://github.com/docker/cli",
        "https://github.com/prometheus/prometheus",
        "https://github.com/grafana/loki",
        "https://github.com/influxdata/influxdb",
        "https://github.com/hashicorp/terraform",
        "https://github.com/traefik/traefik",
        "https://github.com/minio/minio",
        "https://github.com/caddyserver/caddy",
        "https://github.com/go-gitea/gitea",
    ],
    "Ruby": [
        "https://github.com/discourse/discourse",
        "https://github.com/mastodon/mastodon",
        "https://github.com/Homebrew/brew",
        "https://github.com/jekyll/jekyll",
        "https://github.com/redmine/redmine",
        "https://github.com/gitlabhq/gitlabhq",
        "https://github.com/opf/openproject",
        "https://github.com/spree/spree",
        "https://github.com/forem/forem",
        "https://github.com/diaspora/diaspora",
    ],
}


# ------------------------
# Utilidades
# ------------------------
def load_github_token() -> Optional[str]:
    """Lê token do $GITHUB_TOKEN; se vazio, tenta ~/.github_token (1ª linha)."""
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        return tok.strip()
    try:
        path = os.path.expanduser("~/.github_token")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                line = f.readline().strip()
                return line or None
    except Exception:
        pass
    return None

def slugify(text: str) -> str:
    """Slug genérico (fallback) – mantém letras/números/_-. e troca espaços por _."""
    text = unicodedata.normalize("NFKD", text)
    text = text.strip()
    text = re.sub(r"[\s/]+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_\-\.]", "", text)
    return text or "misc"

def language_slug(lang: str) -> str:
    """Mapeia nomes de linguagem para pastas estáveis (evita colidir C, C# e C++)."""
    mapping = {
        "C": "C",
        "C#": "CSharp",
        "C++": "Cpp",
        "Objective-C": "Objective-C",
        "Objective-C++": "Objective-Cpp",
        "JavaScript": "JavaScript",
        "TypeScript": "TypeScript",
        "Python": "Python",
        "Java": "Java",
        "Go": "Go",
        "Ruby": "Ruby",
        "PHP": "PHP",
        # você pode estender aqui se adicionar outras linguagens
    }
    return mapping.get(lang, slugify(lang))

def guess_default_branch(dl: GithubDownloader, owner: str, repo: str, git_bin: str = "git") -> str:
    """
    Resolve branch padrão SEM usar API (reduz 403/429):
    1) tenta `git ls-remote --symref ... HEAD`
    2) testa URLs ZIP de main/master
    3) fallback: 'main'
    """
    try:
        proc = subprocess.run(
            [git_bin, "ls-remote", "--symref", f"https://github.com/{owner}/{repo}.git", "HEAD"],
            capture_output=True, text=True, timeout=25
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if line.startswith("ref: ") and line.endswith("HEAD"):
                    m = re.search(r"refs/heads/([^\s]+)", line)
                    if m:
                        return m.group(1)
    except Exception:
        pass

    for cand in ("main", "master"):
        zip_head = f"https://github.com/{owner}/{repo}/archive/refs/heads/{cand}.zip"
        try:
            r = dl.session.head(zip_head, timeout=dl.timeout, allow_redirects=True)
            if 200 <= r.status_code < 300:
                return cand
        except Exception:
            pass

    return "main"


def fetch_repo_metadata_with_backoff(dl: GithubDownloader, owner: str, repo: str, max_retries: int = 6) -> Optional[dict]:
    """Busca metadados usando API com respeito a rate limit; usa token se disponível."""
    url = f"https://api.github.com/repos/{owner}/{repo}"
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        resp = dl.session.get(url, timeout=dl.timeout)
        if resp.status_code == 200:
            repo_json = resp.json()
            owner_login = (repo_json.get("owner") or {}).get("login")
            owner_json = {}
            if owner_login:
                o = dl.session.get(f"https://api.github.com/users/{owner_login}", timeout=dl.timeout)
                if o.status_code == 200:
                    owner_json = o.json()
            return {
                "repo": {
                    "full_name": repo_json.get("full_name"),
                    "name": repo_json.get("name"),
                    "owner": owner_login,
                    "description": repo_json.get("description"),
                    "stars": repo_json.get("stargazers_count"),
                    "forks": repo_json.get("forks_count"),
                    "watchers": repo_json.get("subscribers_count"),
                    "open_issues": repo_json.get("open_issues_count"),
                    "language": repo_json.get("language"),
                    "topics": repo_json.get("topics"),
                    "license": (repo_json.get("license") or {}).get("spdx_id"),
                    "visibility": repo_json.get("visibility"),
                    "archived": repo_json.get("archived"),
                    "disabled": repo_json.get("disabled"),
                    "size_kb": repo_json.get("size"),
                    "default_branch": repo_json.get("default_branch"),
                    "created_at": repo_json.get("created_at"),
                    "updated_at": repo_json.get("updated_at"),
                    "pushed_at": repo_json.get("pushed_at"),
                    "homepage": repo_json.get("homepage"),
                },
                "owner": {
                    "login": owner_json.get("login"),
                    "type": owner_json.get("type"),
                    "name": owner_json.get("name"),
                    "company": owner_json.get("company"),
                    "blog": owner_json.get("blog"),
                    "location": owner_json.get("location"),
                    "public_repos": owner_json.get("public_repos"),
                    "followers": owner_json.get("followers"),
                    "following": owner_json.get("following"),
                    "created_at": owner_json.get("created_at"),
                },
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

        if resp.status_code in (403, 429):
            remaining = resp.headers.get("X-RateLimit-Remaining")
            reset_at = resp.headers.get("X-RateLimit-Reset")
            wait_s = 5 * attempt
            if remaining == "0" and reset_at:
                try:
                    reset_epoch = int(reset_at)
                    now = int(time.time())
                    wait_s = max(reset_epoch - now + 2, wait_s)
                except ValueError:
                    pass
            print(f"[META] Rate limit {resp.status_code}. Aguardando {wait_s}s…")
            time.sleep(wait_s)
            continue

        if 500 <= resp.status_code < 600:
            wait_s = 3 * attempt
            print(f"[META] HTTP {resp.status_code}. Retentando em {wait_s}s…")
            time.sleep(wait_s)
            continue

        print(f"[META] Falha ao buscar metadados ({resp.status_code}): {resp.text[:120]}")
        return None
    print("[META] Abandonei após múltiplas tentativas.")
    return None


def write_metadata_file(path_dir_repo: str, data: dict) -> None:
    os.makedirs(path_dir_repo, exist_ok=True)
    with open(os.path.join(path_dir_repo, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def flatten_for_csv(lang: str, owner: str, repo: str, target_dir: str, meta: dict) -> dict:
    r = meta.get("repo", {})
    o = meta.get("owner", {})
    return {
        "language_group": lang,
        "owner": owner,
        "repo": repo,
        "path": target_dir,
        "full_name": r.get("full_name"),
        "description": r.get("description"),
        "stars": r.get("stars"),
        "forks": r.get("forks"),
        "watchers": r.get("watchers"),
        "open_issues": r.get("open_issues"),
        "language": r.get("language"),
        "license": r.get("license"),
        "default_branch": r.get("default_branch"),
        "visibility": r.get("visibility"),
        "archived": r.get("archived"),
        "disabled": r.get("disabled"),
        "size_kb": r.get("size_kb"),
        "created_at": r.get("created_at"),
        "updated_at": r.get("updated_at"),
        "pushed_at": r.get("pushed_at"),
        "owner_type": o.get("type"),
        "owner_name": o.get("name"),
        "owner_followers": o.get("followers"),
        "owner_following": o.get("following"),
        "owner_public_repos": o.get("public_repos"),
    }


def main():
    os.makedirs(DESTINO, exist_ok=True)

    token = load_github_token()
    if token:
        os.environ["GITHUB_TOKEN"] = token  # só para referência futura
    dl = GithubDownloader(token=token)

    # Monta a fila de tarefas
    tasks: List[Tuple[str, str]] = []
    for lang, repos in LANG_REPOS.items():
        for url in repos:
            tasks.append((url, lang))

    # 1) Downloads em paralelo, cada linguagem na sua pasta
    download_results: List[Tuple[str, str, str, str, str]] = []  # (lang, owner, repo, ref, out_root_lang)

    def download_worker(url: str, lang: str) -> Optional[Tuple[str, str, str, str, str]]:
        try:
            owner, repo = dl.parse_repo(url)
            ref = guess_default_branch(dl, owner, repo, git_bin="git")
            out_root_lang = os.path.join(DESTINO, slugify(lang))
            os.makedirs(out_root_lang, exist_ok=True)

            process_one(
                dl=dl,
                url=url,
                out_root=out_root_lang,  # <-- SALVA DENTRO DA PASTA DA LINGUAGEM
                ref=ref,
                overwrite=False,
                use_latest=None,
                tag_pattern=None,
                subdir=None,
                include=None,
                exclude=None,
                json_log=None,
                max_zip_mb=MAX_ZIP_MB,
                fallback_git=FALLBACK_GIT,
                git_bin="git",
            )
            return (lang, owner, repo, ref, out_root_lang)
        except Exception as e:
            print(f"[ERRO] {url}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_map = {ex.submit(download_worker, url, lang): (url, lang) for url, lang in tasks}
        for fut in as_completed(future_map):
            res = fut.result()
            if res:
                download_results.append(res)

    # 2) Metadados (quase serial) e agregados
    results_for_csv: List[dict] = []

    def meta_worker(item: Tuple[str, str, str, str, str]) -> Optional[dict]:
        lang, owner, repo, ref, out_root_lang = item
        target_dir = os.path.join(out_root_lang, f"{owner}__{repo}__{ref}")
        meta = fetch_repo_metadata_with_backoff(dl, owner, repo)
        if not meta:
            return None
        meta["language_group"] = lang
        write_metadata_file(target_dir, meta)
        return flatten_for_csv(lang, owner, repo, target_dir, meta)

    if META_WORKERS <= 1:
        for item in download_results:
            rec = meta_worker(item)
            if rec:
                results_for_csv.append(rec)
    else:
        with ThreadPoolExecutor(max_workers=META_WORKERS) as ex:
            for fut in as_completed({ex.submit(meta_worker, it): it for it in download_results}):
                rec = fut.result()
                if rec:
                    results_for_csv.append(rec)

    # 3) Escreve agregados na PASTA RAIZ (DESTINO)
    csv_path = os.path.join(DESTINO, "repos_metadata.csv")
    jsonl_path = os.path.join(DESTINO, "repos_metadata.jsonl")

    if results_for_csv:
        fieldnames = list(results_for_csv[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results_for_csv)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in results_for_csv:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n✓ Metadados agregados salvos:\n  • {csv_path}\n  • {jsonl_path}")
    else:
        print("\n[AVISO] Nenhum metadado agregado foi gerado (todas as tarefas falharam?).")


if __name__ == "__main__":
    main()
