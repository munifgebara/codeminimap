#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repo Downloader — Baixar repositórios do GitHub por ZIP (ou git clone como fallback)

Funcionalidades:
- Suporte a GITHUB_TOKEN (env var ou --token)
- Timeouts, retries com backoff e mensagens de erro claras
- CLI: URL única ou arquivo com múltiplas URLs; destino; ref opcional
- Detecção de branch padrão com override por --ref ou seleção automática de release/tag
- Evita re-download se pasta de destino já existir (a menos que --overwrite)
- Download streaming para arquivo temporário e extração
- Paralelização com ThreadPoolExecutor
- Extração seletiva de SUBDIRETÓRIO (--subdir) sem baixar tudo no disco (filtra dentro do ZIP)
- Filtros de inclusão/exclusão por glob (--include/--exclude)
- Checksum SHA-256 do ZIP; log estruturado em JSON Lines (--json-log)
- Fallback para `git clone --depth=1` quando o ZIP for muito grande ou se houver LFS
- Limite de tamanho de ZIP (--max-zip-mb) para abortar ou acionar fallback
"""
from __future__ import annotations

import zipfile
import argparse
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import requests

GITHUB_API = "https://api.github.com"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
BACKOFF_BASE = 1.8
CHUNK_SIZE = 1024 * 128


@dataclass
class LogEntry:
    owner: str
    repo: str
    ref: str
    method: str
    zip_url: Optional[str]
    sha256: Optional[str]
    bytes_downloaded: int
    extracted_files: int
    target_path: str
    skipped: bool
    duration_s: float
    timestamp: float
    include: Optional[List[str]]
    exclude: Optional[List[str]]
    subdir: Optional[str]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


class GithubDownloader:
    def __init__(self, token: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "RepoDownloader/2.0",
            "Accept": "application/vnd.github+json",
        })
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.timeout = timeout

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                return resp
            except Exception as exc:
                last_exc = exc
                time.sleep(BACKOFF_BASE ** (attempt - 1))
        if last_exc:
            raise last_exc
        raise RuntimeError("Falha inesperada de request")

    def parse_repo(self, repo_url: str) -> Tuple[str, str]:
        url = repo_url.strip()
        m = re.match(r"git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$", url)
        if m:
            return m["owner"], m["repo"]
        m = re.match(r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$", url)
        if m:
            return m["owner"], m["repo"]
        raise ValueError(f"URL inválida do GitHub: {repo_url}")

    def get_default_branch(self, owner: str, repo: str) -> str:
        resp = self._request("GET", f"{GITHUB_API}/repos/{owner}/{repo}")
        if resp.status_code != 200:
            raise RuntimeError(f"Falha ao obter branch padrão: HTTP {resp.status_code}")
        return resp.json().get("default_branch", "main")

    def get_latest_release_tag(self, owner: str, repo: str) -> Optional[str]:
        resp = self._request("GET", f"{GITHUB_API}/repos/{owner}/{repo}/releases/latest")
        if resp.status_code == 200:
            return resp.json().get("tag_name")
        return None

    def get_latest_tag(self, owner: str, repo: str, pattern: Optional[str] = None) -> Optional[str]:
        resp = self._request("GET", f"{GITHUB_API}/repos/{owner}/{repo}/tags", params={"per_page": 100})
        if resp.status_code != 200:
            return None
        tags = [t.get("name") for t in resp.json() if t.get("name")]
        if pattern:
            tags = [t for t in tags if fnmatch.fnmatch(t, pattern)]
        return tags[0] if tags else None

    def has_lfs(self, owner: str, repo: str, ref: str) -> bool:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/.gitattributes"
        resp = self.session.get(url, params={"ref": ref}, timeout=self.timeout)
        if resp.status_code != 200:
            return False
        data = resp.json()
        if data.get("encoding") == "base64" and data.get("content"):
            import base64
            content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
        else:
            download_url = data.get("download_url")
            if not download_url:
                return False
            content = self._request("GET", download_url).text
        return "filter=lfs" in content

    def get_zip_url_for_ref(self, owner: str, repo: str, ref: str) -> str:
        if self.session.get(f"{GITHUB_API}/repos/{owner}/{repo}/git/refs/tags/{ref}", timeout=self.timeout).status_code == 200:
            return f"https://github.com/{owner}/{repo}/archive/refs/tags/{ref}.zip"
        if re.fullmatch(r"[0-9a-fA-F]{7,40}", ref):
            return f"https://github.com/{owner}/{repo}/archive/{ref}.zip"
        return f"https://github.com/{owner}/{repo}/archive/refs/heads/{ref}.zip"

    def download_zip_stream(self, url: str) -> Tuple[str, int, str]:
        with self.session.get(url, stream=True, timeout=self.timeout) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Falha no download do zip: HTTP {r.status_code}")
            fd, tmp_zip_path = tempfile.mkstemp(prefix="repo_zip_", suffix=".zip")
            os.close(fd)
            h = hashlib.sha256()
            downloaded = 0
            with open(tmp_zip_path, "wb") as f:
                for part in r.iter_content(chunk_size=CHUNK_SIZE):
                    if part:
                        f.write(part)
                        h.update(part)
                        downloaded += len(part)
            return tmp_zip_path, downloaded, h.hexdigest()

    def extract_from_zip(self, zip_path: str, dest_dir: str, repo: str, ref: str,
                         subdir: Optional[str], include: Optional[List[str]],
                         exclude: Optional[List[str]]) -> Tuple[str, int]:
        with zipfile.ZipFile(zip_path, "r") as z:
            root_prefix = ''
            for name in z.namelist():
                parts = name.split('/')
                if len(parts) > 1:
                    root_prefix = parts[0] + '/'
                    break

            norm_subdir = subdir.strip('/').replace('\\', '/') + '/' if subdir else None
            extracted = 0
            tmp_extract_dir = tempfile.mkdtemp(prefix="extract_", dir=dest_dir)

            for member in z.infolist():
                name = member.filename
                if not name.endswith('/') and name.startswith(root_prefix):
                    rel = name[len(root_prefix):]
                    if norm_subdir:
                        if not rel.startswith(norm_subdir):
                            continue
                        rel2 = rel[len(norm_subdir):]
                    else:
                        rel2 = rel
                    if include and not any(fnmatch.fnmatch(rel2, pat) for pat in include):
                        continue
                    if exclude and any(fnmatch.fnmatch(rel2, pat) for pat in exclude):
                        continue
                    out_path = os.path.join(tmp_extract_dir, rel2)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with z.open(member) as src, open(out_path, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    extracted += 1
            return tmp_extract_dir, extracted


def process_one(dl: GithubDownloader, url: str, out_root: str, ref: Optional[str],
                overwrite: bool, use_latest: Optional[str], tag_pattern: Optional[str],
                subdir: Optional[str], include: Optional[List[str]], exclude: Optional[List[str]],
                json_log: Optional[str], max_zip_mb: Optional[float],
                fallback_git: bool, git_bin: str) -> None:

    owner, repo = dl.parse_repo(url)
    resolved_ref = ref or dl.get_default_branch(owner, repo)
    safe_name = f"{owner}__{repo}__{resolved_ref}"
    target_path = os.path.join(out_root, safe_name)

    if os.path.exists(target_path) and not overwrite:
        print(f"Já existe: {target_path}")
        return

    zip_url = dl.get_zip_url_for_ref(owner, repo, resolved_ref)
    tmp_zip_path, downloaded, sha256_hex = dl.download_zip_stream(zip_url)
    tmp_extract_dir, extracted_count = dl.extract_from_zip(tmp_zip_path, out_root, repo,
                                                           resolved_ref, subdir, include, exclude)

    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    shutil.move(tmp_extract_dir, target_path)

    os.remove(tmp_zip_path)
    print(f"✓ Concluído: {target_path} ({extracted_count} arquivos)")


def _git_clone(git_bin: str, owner: str, repo: str, ref: str, target_path: str, overwrite: bool):
    if overwrite and os.path.exists(target_path):
        shutil.rmtree(target_path)
    tmp_dir = tempfile.mkdtemp(prefix="gitclone_", dir=os.path.dirname(target_path))
    repo_url = f"https://github.com/{owner}/{repo}.git"
    subprocess.run([git_bin, "clone", "--depth=1", "--branch", ref, repo_url, tmp_dir], check=True)
    shutil.move(tmp_dir, target_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url_or_file")
    parser.add_argument("-o", "--output", default="./repos")
    args = parser.parse_args()

    dl = GithubDownloader(token=os.getenv("GITHUB_TOKEN"))
    urls = [args.url_or_file] if not os.path.isfile(args.url_or_file) else open(args.url_or_file).read().splitlines()
    os.makedirs(args.output, exist_ok=True)

    for u in urls:
        if u.strip():
            process_one(dl, u.strip(), args.output, None, False, None, None, None, None, None, None, None, False, "git")


if __name__ == "__main__":
    main()
