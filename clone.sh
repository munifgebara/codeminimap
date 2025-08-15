#!/usr/bin/env bash
set -euo pipefail

# ===== Configuração inicial =====
BASE_DIR="${1:-$HOME/dataset2}"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "[ERRO] GITHUB_TOKEN não definido. Ex.: export GITHUB_TOKEN='seu_token'"
  exit 1
fi

mkdir -p "$BASE_DIR"

# Constrói URL autenticada sem persistir o token no 'origin'
authed_url() {
  local url="$1"
  # Insere o token na URL apenas para a operação de rede
  echo "${url/https:\/\/github.com\//https:\/\/${GITHUB_TOKEN}@github.com/}"
}

clone_or_update() {
  local lang="$1"
  local url="$2"

  local repo_name
  repo_name="$(basename "$url")"
  repo_name="${repo_name%.git}"

  local dest_dir="$BASE_DIR/$lang/$repo_name"
  mkdir -p "$BASE_DIR/$lang"

  if [[ -d "$dest_dir/.git" ]]; then
    echo "[ATUALIZANDO] $lang/$repo_name"
    # Troca temporariamente o remote para a URL autenticada só para puxar
    git -C "$dest_dir" remote set-url origin "$(authed_url "$url")"
    # Atualiza com segurança
    git -C "$dest_dir" fetch --all --prune --depth=1
    # Tenta puxar avanço rápido na branch atual
    if ! git -C "$dest_dir" pull --ff-only; then
      echo "[AVISO] Pull com ff-only falhou em $lang/$repo_name. Mantendo estado local."
    fi
    # Restaura o remote para a URL sem token
    git -C "$dest_dir" remote set-url origin "$url"
  else
    echo "[CLONANDO] $lang/$repo_name -> $dest_dir"
    git clone --depth 1 "$(authed_url "$url")" "$dest_dir"
    # Limpa o token do remote
    git -C "$dest_dir" remote set-url origin "$url"
  fi
}

# ===== Lista de repositórios por linguagem =====

JavaScript_REPOS=(
  "https://github.com/freeCodeCamp/freeCodeCamp"
  "https://github.com/TryGhost/Ghost"
  "https://github.com/RocketChat/Rocket.Chat"
  "https://github.com/ether/etherpad-lite"
  "https://github.com/node-red/node-red"
  "https://github.com/NodeBB/NodeBB"
  "https://github.com/jitsi/jitsi-meet"
  "https://github.com/vercel/hyper"
  "https://github.com/standardnotes/app"
  "https://github.com/BoostIO/Boostnote"
)

Python_REPOS=(
  "https://github.com/home-assistant/core"
  "https://github.com/odoo/odoo"
  "https://github.com/yt-dlp/yt-dlp"
  "https://github.com/getsentry/sentry"
  "https://github.com/saltstack/salt"
  "https://github.com/ansible/ansible"
  "https://github.com/paperless-ngx/paperless-ngx"
  "https://github.com/kovidgoyal/calibre"
  "https://github.com/commaai/openpilot"
  "https://github.com/openai/whisper"
)

Java_REPOS=(
  "https://github.com/jenkinsci/jenkins"
  "https://github.com/keycloak/keycloak"
  "https://github.com/elastic/elasticsearch"
  "https://github.com/apache/solr"
  "https://github.com/trinodb/trino"
  "https://github.com/apache/hadoop"
  "https://github.com/apache/nifi"
  "https://github.com/geoserver/geoserver"
  "https://github.com/Graylog2/graylog2-server"
  "https://github.com/apache/cassandra"
)

TypeScript_REPOS=(
  "https://github.com/microsoft/vscode"
  "https://github.com/desktop/desktop"
  "https://github.com/outline/outline"
  "https://github.com/directus/directus"
  "https://github.com/jgraph/drawio"
  "https://github.com/excalidraw/excalidraw"
  "https://github.com/laurent22/joplin"
  "https://github.com/immich-app/immich"
  "https://github.com/Open-Lens/lens"
  "https://github.com/supabase/supabase"
)

Csharp_REPOS=(
  "https://github.com/jellyfin/jellyfin"
  "https://github.com/ShareX/ShareX"
  "https://github.com/files-community/Files"
  "https://github.com/icsharpcode/ILSpy"
  "https://github.com/dnSpyEx/dnSpy"
  "https://github.com/Jackett/Jackett"
  "https://github.com/Radarr/Radarr"
  "https://github.com/Sonarr/Sonarr"
  "https://github.com/Ombi-app/Ombi"
  "https://github.com/bitwarden/server"
)

Cpp_REPOS=(
  "https://github.com/bitcoin/bitcoin"
  "https://github.com/obsproject/obs-studio"
  "https://github.com/notepad-plus-plus/notepad-plus-plus"
  "https://github.com/qbittorrent/qBittorrent"
  "https://github.com/telegramdesktop/tdesktop"
  "https://github.com/godotengine/godot"
  "https://github.com/musescore/MuseScore"
  "https://github.com/qgis/QGIS"
  "https://github.com/KDE/krita"
  "https://github.com/openttd/openttd"
)

PHP_REPOS=(
  "https://github.com/WordPress/WordPress"
  "https://github.com/nextcloud/server"
  "https://github.com/matomo-org/matomo"
  "https://github.com/phpmyadmin/phpmyadmin"
  "https://github.com/moodle/moodle"
  "https://github.com/BookStackApp/BookStack"
  "https://github.com/flarum/flarum"
  "https://github.com/LycheeOrg/Lychee"
  "https://github.com/osTicket/osTicket"
  "https://github.com/pterodactyl/panel"
)

C_REPOS=(
  "https://github.com/torvalds/linux"
  "https://github.com/FFmpeg/FFmpeg"
  "https://github.com/neovim/neovim"
  "https://github.com/vim/vim"
  "https://github.com/curl/curl"
  "https://github.com/tmux/tmux"
  "https://github.com/redis/redis"
  "https://github.com/nginx/nginx"
  "https://github.com/git/git"
  "https://github.com/systemd/systemd"
)

Go_REPOS=(
  "https://github.com/kubernetes/kubernetes"
  "https://github.com/docker/cli"
  "https://github.com/prometheus/prometheus"
  "https://github.com/grafana/loki"
  "https://github.com/influxdata/influxdb"
  "https://github.com/hashicorp/terraform"
  "https://github.com/traefik/traefik"
  "https://github.com/minio/minio"
  "https://github.com/caddyserver/caddy"
  "https://github.com/go-gitea/gitea"
)

Ruby_REPOS=(
  "https://github.com/discourse/discourse"
  "https://github.com/mastodon/mastodon"
  "https://github.com/Homebrew/brew"
  "https://github.com/jekyll/jekyll"
  "https://github.com/redmine/redmine"
  "https://github.com/gitlabhq/gitlabhq"
  "https://github.com/opf/openproject"
  "https://github.com/spree/spree"
  "https://github.com/forem/forem"
  "https://github.com/diaspora/diaspora"
)

# ===== Execução =====

process_group() {
  local lang="$1"; shift
  local -n arr="$1"
  for url in "${arr[@]}"; do
    clone_or_update "$lang" "$url"
  done
}

process_group "JavaScript" JavaScript_REPOS
process_group "Python"     Python_REPOS
process_group "Java"       Java_REPOS
process_group "TypeScript" TypeScript_REPOS
process_group "C#"         Csharp_REPOS
process_group "C++"        Cpp_REPOS
process_group "PHP"        PHP_REPOS
process_group "C"          C_REPOS
process_group "Go"         Go_REPOS
process_group "Ruby"       Ruby_REPOS

echo "[OK] Tudo pronto em: $BASE_DIR"

