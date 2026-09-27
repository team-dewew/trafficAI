#!/usr/bin/env bash
# Traffic AI website: one-shot, idempotent installer for Ubuntu 22.04/24.04 (x86_64, no GPU needed).
#
#   sudo bash deploy/install.sh --domain trafficai.dewew.dev [--email you@example.com]
#
# What it does (safe to re-run; it never stops or reconfigures other services):
#   1. preflight: root, OS, disk, memory (+ a swap file if swap is small), DNS of the domain -> this server,
#      who owns ports 80/443
#   2. apt packages: python3-venv, git, nginx (only if nothing else serves 80/443), certbot
#   3. system user 'trafficai', code in /opt/trafficai (git checkout -> site copy of the tracked files)
#   4. Python venv with the website's CPU dependency set (requirements-web.txt)
#   5. demo detector weights, kept outside the site copy so updates do not re-download them
#   6. a free localhost port (default range 8600-8699) that no other service uses
#   7. systemd service (memory-capped, restarts on failure, starts on boot)
#   8. nginx site for the domain -> 127.0.0.1:<port> (websockets, 800 MB uploads)
#   9. Let's Encrypt certificate (HTTP -> HTTPS redirect) with automatic renewal + nginx reload hook
#  10. health checks over localhost and HTTPS, and a real run of the demo pipeline on the bundled clip
set -Eeuo pipefail

# ------------------------------------------------------------------ defaults
DOMAIN=""
EMAIL=""
REPO_URL="https://github.com/team-dewew/trafficAI.git"
BRANCH="master"
APP_USER="trafficai"
APP_DIR="/opt/trafficai"
PORT=""
PORT_RANGE_START=8600
PORT_RANGE_END=8699
MEMORY_MAX="3000M"        # measured: 768 MB upload held by Streamlit + 4K processing -> 2.5 GB peak RSS
MEMORY_HIGH="2700M"
UPLOAD_MB=800
SWAP_SIZE="2G"            # swap file added when the server has less than 3.5 GB of swap; 0 = never
SKIP_DNS_CHECK=0
SKIP_SMOKE=0
CONTAINER_TEST=0          # CI/containers without systemd: skip systemd/nginx/certbot/DNS
SERVICE="trafficai"
ENV_FILE="/etc/trafficai/trafficai.env"

log()  { printf '\033[1;36m[trafficai]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[trafficai] WARNING:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[trafficai] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }
trap 'die "failed at line $LINENO: $BASH_COMMAND"' ERR

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  cat <<EOF

Options:
  --domain NAME        public domain (required unless --container-test)
  --email ADDRESS      Let's Encrypt account e-mail (expiry notices); omitted = registered without e-mail
  --repo URL           git repository (default: $REPO_URL)
  --branch NAME        branch or tag to deploy (default: $BRANCH)
  --dir PATH           install directory (default: $APP_DIR)
  --port N             localhost port for Streamlit (default: first free in $PORT_RANGE_START-$PORT_RANGE_END)
  --memory-max SIZE    systemd MemoryMax for the service (default: $MEMORY_MAX)
  --swap SIZE          swap file /swapfile-trafficai if swap < 3.5 GB (default: $SWAP_SIZE; 0 = none)
  --skip-dns-check     do not require the domain's A record to point at this server
  --skip-smoke         skip the demo pipeline smoke test
  --container-test     install + run without systemd/nginx/certbot (for testing in a container)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain) DOMAIN="$2"; shift 2 ;;
    --email) EMAIL="$2"; shift 2 ;;
    --repo) REPO_URL="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --dir) APP_DIR="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --memory-max) MEMORY_MAX="$2"; shift 2 ;;
    --swap) SWAP_SIZE="$2"; shift 2 ;;
    --skip-dns-check) SKIP_DNS_CHECK=1; shift ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --container-test) CONTAINER_TEST=1; SKIP_DNS_CHECK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

[[ $EUID -eq 0 ]] || die "run as root: sudo bash $0 --domain <name>"
[[ $CONTAINER_TEST -eq 1 || -n "$DOMAIN" ]] || die "--domain is required"
[[ -z "$PORT" || "$PORT" =~ ^[0-9]+$ ]] || die "--port must be a number"

REPO_DIR="$APP_DIR/repo"
SITE_DIR="$APP_DIR/site"
VENV_DIR="$APP_DIR/venv"
DATA_DIR="$APP_DIR/data"
TMP_DIR="$APP_DIR/tmp"
export DEBIAN_FRONTEND=noninteractive

port_in_use() { ss -ltnH "( sport = :$1 )" 2>/dev/null | grep -q .; }
listeners_on() { { ss -ltnpH "( sport = :$1 )" 2>/dev/null | grep -o 'users:(("[^"]*"' | sed 's/users:(("//' | sort -u | tr '\n' ' '; } || true; }

# ------------------------------------------------------------------ 1. preflight
log "1/10 preflight checks"
. /etc/os-release 2>/dev/null || die "cannot read /etc/os-release"
[[ "${ID:-}" == "ubuntu" ]] || warn "tested on Ubuntu; this is ${PRETTY_NAME:-unknown}"
[[ "$(uname -m)" == "x86_64" ]] || die "x86_64 required (CPU torch wheels); this is $(uname -m)"

mkdir -p "$APP_DIR"
free_gb=$(df -BG --output=avail "$APP_DIR" | tail -1 | tr -dc '0-9')
(( free_gb >= 4 )) || die "need >= 4 GB free disk in $APP_DIR (have ${free_gb} GB)"

avail_mb=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
swap_free_mb=$(awk '/SwapFree/ {print int($2/1024)}' /proc/meminfo)
log "memory available: ${avail_mb} MB, swap free: ${swap_free_mb} MB"
if (( avail_mb < 1500 )); then
  warn "only ${avail_mb} MB RAM available. The demo peaks at ~1.2-1.4 GB while processing a video."
  warn "It is capped at MemoryMax=${MEMORY_MAX} so it cannot starve other services, but free some RAM"
  warn "(or add swap) for reliable demos. Continuing."
fi

# An 800 MB upload sits in memory while it is processed. On a small server, a swap file keeps
# a demo from being killed instead of starving other services. Created once, kept across updates.
SWAP_FILE="/swapfile-trafficai"
swap_total_mb=$(awk '/SwapTotal/ {print int($2/1024)}' /proc/meminfo)
if [[ $CONTAINER_TEST -eq 0 && "$SWAP_SIZE" != "0" && ! -f "$SWAP_FILE" ]] && (( swap_total_mb < 3500 )); then
  swap_gb=$(tr -dc '0-9' <<< "$SWAP_SIZE")
  root_free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  if (( root_free_gb >= swap_gb + 6 )); then
    log "adding a ${SWAP_SIZE} swap file $SWAP_FILE (swap now ${swap_total_mb} MB)"
    if { fallocate -l "$SWAP_SIZE" "$SWAP_FILE" 2>/dev/null \
           || dd if=/dev/zero of="$SWAP_FILE" bs=1M count=$(( swap_gb * 1024 )) status=none; } \
       && chmod 600 "$SWAP_FILE" && mkswap "$SWAP_FILE" >/dev/null && swapon "$SWAP_FILE"; then
      grep -q "^$SWAP_FILE " /etc/fstab || echo "$SWAP_FILE none swap sw 0 0" >> /etc/fstab
    else
      rm -f "$SWAP_FILE"
      warn "could not add swap; continuing without it"
    fi
  else
    warn "not adding swap: only ${root_free_gb} GB free on /"
  fi
fi

command -v ss >/dev/null || { apt-get update -qq && apt-get install -y -qq iproute2 >/dev/null; }
if [[ $CONTAINER_TEST -eq 0 ]]; then
  command -v systemctl >/dev/null && systemctl --version >/dev/null 2>&1 || die "systemd is required"
  for p in 80 443; do
    owners=$(listeners_on "$p")
    if [[ -n "$owners" && "$owners" != *nginx* ]]; then
      die "port $p is served by: $owners. This installer uses nginx on 80/443 and will not stop other web servers.
       Options: put a reverse-proxy rule for $DOMAIN -> 127.0.0.1:<port> into that server yourself, or move it."
    fi
  done
fi

if [[ $SKIP_DNS_CHECK -eq 0 ]]; then
  apt-get install -y -qq curl dnsutils >/dev/null 2>&1 || true
  server_ip=$(curl -4 -fsS --max-time 10 https://api.ipify.org || curl -4 -fsS --max-time 10 https://ifconfig.me || true)
  domain_ips=$({ getent ahostsv4 "$DOMAIN" | awk '{print $1}' | sort -u | tr '\n' ' '; } || true)
  log "server public IPv4: ${server_ip:-unknown}; $DOMAIN resolves to: ${domain_ips:-nothing}"
  [[ -n "$domain_ips" ]] || die "$DOMAIN does not resolve. Create an A record $DOMAIN -> $server_ip and retry."
  if [[ -n "$server_ip" && " $domain_ips " != *" $server_ip "* ]]; then
    die "$DOMAIN points to ${domain_ips}but this server is $server_ip. Let's Encrypt would fail.
       Fix the A record (or pass --skip-dns-check if a proxy/NAT in front is intended)."
  fi
fi

# ------------------------------------------------------------------ 2. packages
log "2/10 installing system packages"
apt-get update -qq
PKGS=(ca-certificates curl git python3 python3-venv python3-pip libglib2.0-0 iproute2 openssl util-linux)
if [[ $CONTAINER_TEST -eq 0 ]]; then PKGS+=(nginx certbot python3-certbot-nginx); fi
apt-get install -y -qq --no-install-recommends "${PKGS[@]}" >/dev/null
PY=$(command -v python3)
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' || die "Python >= 3.10 required"

# ------------------------------------------------------------------ 3. user + code
log "3/10 user '$APP_USER' and code ($REPO_URL @ $BRANCH)"
id -u "$APP_USER" >/dev/null 2>&1 || useradd --system --home-dir "$APP_DIR" --shell /usr/sbin/nologin "$APP_USER"
mkdir -p "$DATA_DIR/weights" "$TMP_DIR" /etc/trafficai
if [[ -d "$REPO_DIR/.git" ]]; then
  git -c safe.directory="$REPO_DIR" -C "$REPO_DIR" fetch --quiet --depth 1 origin "$BRANCH"
  git -c safe.directory="$REPO_DIR" -C "$REPO_DIR" reset --quiet --hard FETCH_HEAD
else
  rm -rf "$REPO_DIR"
  git clone --quiet --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi
COMMIT=$(git -c safe.directory="$REPO_DIR" -C "$REPO_DIR" rev-parse --short HEAD)
log "deploying commit $COMMIT"

# the site runs from a copy of the tracked files, so a later fetch never changes a running site
rm -rf "$SITE_DIR.new"
mkdir -p "$SITE_DIR.new"
git -c safe.directory="$REPO_DIR" -C "$REPO_DIR" archive --format=tar HEAD | tar -x -C "$SITE_DIR.new"
# keep weights outside the site copy; link them in
rm -f "$SITE_DIR.new/weights/"*.pt
for f in "$DATA_DIR"/weights/*.pt; do
  if [[ -e "$f" ]]; then ln -sf "$f" "$SITE_DIR.new/weights/$(basename "$f")"; fi
done
rm -rf "$SITE_DIR.old"
if [[ -d "$SITE_DIR" ]]; then mv "$SITE_DIR" "$SITE_DIR.old"; fi
mv "$SITE_DIR.new" "$SITE_DIR"
rm -rf "$SITE_DIR.old"

# ------------------------------------------------------------------ 4. venv
log "4/10 Python environment (CPU)"
REQ_FILE="$REPO_DIR/requirements-web.txt"
REQ_HASH=$( (cat "$REQ_FILE"; "$PY" --version) | sha256sum | cut -c1-16)
if [[ ! -x "$VENV_DIR/bin/python" || "$(cat "$VENV_DIR/.req_hash" 2>/dev/null)" != "$REQ_HASH" ]]; then
  [[ -x "$VENV_DIR/bin/python" ]] || "$PY" -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --quiet --no-cache-dir --upgrade pip wheel
  "$VENV_DIR/bin/pip" install --quiet --no-cache-dir -r "$REQ_FILE"
  echo "$REQ_HASH" > "$VENV_DIR/.req_hash"
else
  log "dependencies unchanged ($REQ_HASH)"
fi
"$VENV_DIR/bin/python" -c "import torch, cv2, ultralytics, supervision, streamlit; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| streamlit', streamlit.__version__)"

# ------------------------------------------------------------------ 5. weights
log "5/10 demo detector weights"
if [[ ! -s "$DATA_DIR/weights/yolo11s.pt" ]]; then
  ( cd "$SITE_DIR/weights" && "$VENV_DIR/bin/python" -c "import download; download.download(['yolo11s.pt'])" )
  mv "$SITE_DIR/weights/yolo11s.pt" "$DATA_DIR/weights/yolo11s.pt"
fi
ln -sf "$DATA_DIR/weights/yolo11s.pt" "$SITE_DIR/weights/yolo11s.pt"
( cd "$SITE_DIR/weights" && grep ' yolo11s.pt$' SHA256SUMS | sha256sum -c --quiet - ) || die "yolo11s.pt checksum mismatch"

chown -R "$APP_USER:$APP_USER" "$APP_DIR"
# the git checkout stays root-owned: the service never needs to write it, and root's git refuses
# to work in a repository owned by another user
chown -R root:root "$REPO_DIR"
chmod 750 "$APP_DIR"

# ------------------------------------------------------------------ 6. port
log "6/10 choosing a free localhost port"
if [[ -z "$PORT" && -f "$ENV_FILE" ]]; then
  PORT=$(sed -n 's/^PORT=//p' "$ENV_FILE")
fi
if [[ -n "$PORT" ]] && port_in_use "$PORT"; then
  owners=$(listeners_on "$PORT")
  if [[ $CONTAINER_TEST -eq 1 || "$(systemctl is-active "$SERVICE" 2>/dev/null)" != "active" || "$owners" != *streamlit* ]]; then
    warn "port $PORT is taken by: ${owners:-unknown}; picking another"
    PORT=""
  fi
fi
if [[ -z "$PORT" ]]; then
  for p in $(seq "$PORT_RANGE_START" "$PORT_RANGE_END"); do
    port_in_use "$p" || { PORT=$p; break; }
  done
fi
[[ -n "$PORT" ]] || die "no free port in $PORT_RANGE_START-$PORT_RANGE_END"
log "Streamlit will listen on 127.0.0.1:$PORT (not exposed publicly; nginx proxies to it)"
cat > "$ENV_FILE" <<EOF
# written by deploy/install.sh
DOMAIN=$DOMAIN
PORT=$PORT
APP_DIR=$APP_DIR
BRANCH=$BRANCH
REPO_URL=$REPO_URL
COMMIT=$COMMIT
EOF

run_smoke() {
  log "demo pipeline smoke test on the bundled clip (takes ~1 min on 2 vCPU)"
  ( cd "$SITE_DIR" && runuser -u "$APP_USER" -- env TMPDIR="$TMP_DIR" OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \
      "$VENV_DIR/bin/python" -c "
from pathlib import Path
from src.demo import run_demo
r = run_demo('samples/demo/C3896_60-95s_720p.mp4', Path('$TMP_DIR/smoke'), 'smoke')
labels = sorted({e[2] for e in r.events})
print(f'smoke: {len(r.events)} events {labels}, {len(r.risk)} risk samples, {len(r.clips)} clips, {r.elapsed:.0f} s')
assert {'stop_line', 'red_light'} <= set(labels), r.events
" ) || die "demo smoke test failed"
  rm -rf "$TMP_DIR/smoke"
}

STREAMLIT_ARGS="run app.py --server.address 127.0.0.1 --server.port $PORT --server.headless true \
--server.enableCORS true --server.enableXsrfProtection true --server.maxUploadSize $UPLOAD_MB \
--server.fileWatcherType none --browser.gatherUsageStats false"

if [[ $CONTAINER_TEST -eq 1 ]]; then
  [[ $SKIP_SMOKE -eq 1 ]] || run_smoke
  log "container test: starting Streamlit in the background for a health check"
  ( cd "$SITE_DIR" && runuser -u "$APP_USER" -- env TMPDIR="$TMP_DIR" OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \
      nohup "$VENV_DIR/bin/streamlit" $STREAMLIT_ARGS > "$TMP_DIR/streamlit.log" 2>&1 & )
  for _ in $(seq 60); do curl -fsS "http://127.0.0.1:$PORT/_stcore/health" >/dev/null 2>&1 && break; sleep 2; done
  curl -fsS "http://127.0.0.1:$PORT/_stcore/health" | grep -q ok || { cat "$TMP_DIR/streamlit.log"; die "health check failed"; }
  log "container test OK: http://127.0.0.1:$PORT/_stcore/health -> ok (commit $COMMIT)"
  exit 0
fi

# ------------------------------------------------------------------ 7. systemd
log "7/10 systemd service '$SERVICE'"
cat > /etc/tmpfiles.d/trafficai.conf <<EOF
# uploads and rendered clips of the demo; files older than 1 day are removed by systemd-tmpfiles-clean.timer
d $TMP_DIR 0750 $APP_USER $APP_USER 1d
EOF
systemd-tmpfiles --create /etc/tmpfiles.d/trafficai.conf || true

cat > "/etc/systemd/system/$SERVICE.service" <<EOF
[Unit]
Description=Traffic AI website (WIUT Hackathon 2026, team dewew)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$APP_USER
Group=$APP_USER
WorkingDirectory=$SITE_DIR
Environment=HOME=$APP_DIR
Environment=TMPDIR=$TMP_DIR
Environment=OMP_NUM_THREADS=2
Environment=MKL_NUM_THREADS=2
Environment=CUDA_VISIBLE_DEVICES=
Environment="OPENCV_FFMPEG_CAPTURE_OPTIONS=threads;2"
Environment=YOLO_OFFLINE=1
Environment=PYTHONUNBUFFERED=1
ExecStart=$VENV_DIR/bin/streamlit $STREAMLIT_ARGS
Restart=always
RestartSec=5
Nice=5
MemoryHigh=$MEMORY_HIGH
MemoryMax=$MEMORY_MAX
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=true
ReadWritePaths=$APP_DIR

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --quiet "$SERVICE"
systemctl restart "$SERVICE"

for _ in $(seq 60); do curl -fsS "http://127.0.0.1:$PORT/_stcore/health" >/dev/null 2>&1 && break; sleep 2; done
curl -fsS "http://127.0.0.1:$PORT/_stcore/health" | grep -q ok \
  || { journalctl -u "$SERVICE" -n 60 --no-pager; die "service did not become healthy on port $PORT"; }
log "service healthy on 127.0.0.1:$PORT"

# ------------------------------------------------------------------ 8. nginx
log "8/10 nginx site for $DOMAIN"
NGINX_SITE="/etc/nginx/sites-available/trafficai.conf"
NGINX_MAP="/etc/nginx/conf.d/trafficai-upgrade-map.conf"
cat > "$NGINX_MAP" <<'EOF'
# websocket upgrade for the Traffic AI site (own variable name: no clash with other sites' maps)
map $http_upgrade $trafficai_connection_upgrade {
    default upgrade;
    ''      close;
}
EOF
# keep the TLS lines certbot added on earlier runs; otherwise start from a plain HTTP server block
if [[ -f "$NGINX_SITE" ]] && grep -q "managed by Certbot" "$NGINX_SITE"; then
  sed -i -E "s#proxy_pass http://127\.0\.0\.1:[0-9]+;#proxy_pass http://127.0.0.1:$PORT;#" "$NGINX_SITE"
  sed -i -E "s#client_max_body_size [0-9]+m;#client_max_body_size $(( UPLOAD_MB + 10 ))m;#" "$NGINX_SITE"
else
  cat > "$NGINX_SITE" <<EOF
# Traffic AI website -> Streamlit on 127.0.0.1:$PORT (written by deploy/install.sh)
server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN;

    client_max_body_size $(( UPLOAD_MB + 10 ))m;

    location / {
        proxy_pass http://127.0.0.1:$PORT;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection \$trafficai_connection_upgrade;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        proxy_buffering off;
        proxy_request_buffering off;
    }
}
EOF
fi
ln -sf "$NGINX_SITE" /etc/nginx/sites-enabled/trafficai.conf
if ! nginx -t 2>/tmp/trafficai-nginx-test.log; then
  cat /tmp/trafficai-nginx-test.log >&2
  rm -f /etc/nginx/sites-enabled/trafficai.conf "$NGINX_MAP"
  die "nginx config test failed; the Traffic AI site was removed again, other sites are untouched"
fi
systemctl enable --quiet nginx
systemctl is-active --quiet nginx && systemctl reload nginx || systemctl start nginx

if command -v ufw >/dev/null && ufw status 2>/dev/null | grep -q "Status: active"; then
  log "ufw is active: allowing 'Nginx Full' (80/443)"
  ufw allow 'Nginx Full' >/dev/null
fi

# ------------------------------------------------------------------ 9. certificate
log "9/10 Let's Encrypt certificate for $DOMAIN"
CERTBOT_ACCOUNT=(--register-unsafely-without-email)
if [[ -n "$EMAIL" ]]; then CERTBOT_ACCOUNT=(-m "$EMAIL"); fi
certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos "${CERTBOT_ACCOUNT[@]}" \
  --redirect --keep-until-expiring
mkdir -p /etc/letsencrypt/renewal-hooks/deploy
cat > /etc/letsencrypt/renewal-hooks/deploy/trafficai-reload-nginx.sh <<'EOF'
#!/bin/sh
# reload nginx after a renewed certificate is written (installed by Traffic AI deploy/install.sh)
nginx -t && systemctl reload nginx
EOF
chmod 755 /etc/letsencrypt/renewal-hooks/deploy/trafficai-reload-nginx.sh
if systemctl list-unit-files certbot.timer >/dev/null 2>&1 && systemctl list-unit-files | grep -q '^certbot.timer'; then
  systemctl enable --now --quiet certbot.timer
  RENEWAL="certbot.timer (systemd)"
elif systemctl list-unit-files | grep -q '^snap.certbot.renew.timer'; then
  RENEWAL="snap.certbot.renew.timer (snap)"
else
  echo "17 3,15 * * * root certbot renew --quiet" > /etc/cron.d/trafficai-certbot-renew
  RENEWAL="/etc/cron.d/trafficai-certbot-renew"
fi
log "automatic renewal: $RENEWAL; testing it with a dry run"
certbot renew --dry-run --cert-name "$DOMAIN" --quiet || warn "renewal dry run failed; check 'certbot renew --dry-run'"

# ------------------------------------------------------------------ 10. verification
log "10/10 verification"
[[ $SKIP_SMOKE -eq 1 ]] || run_smoke
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "https://$DOMAIN/_stcore/health" || true)
[[ "$code" == "200" ]] || die "https://$DOMAIN/_stcore/health returned HTTP $code"
redirect=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "http://$DOMAIN/" || true)
expiry=$(echo | openssl s_client -servername "$DOMAIN" -connect "$DOMAIN:443" 2>/dev/null | openssl x509 -noout -enddate | cut -d= -f2)

cat <<EOF

==================================================================
 Traffic AI is live:  https://$DOMAIN
 commit:              $COMMIT ($BRANCH)
 service:             systemctl status $SERVICE   (logs: journalctl -u $SERVICE -f)
 internal port:       127.0.0.1:$PORT (not public)
 http -> https:       HTTP $redirect
 certificate expires: $expiry  (auto-renewal: $RENEWAL)
 update later:        sudo bash $REPO_DIR/deploy/update.sh
 status check:        sudo bash $REPO_DIR/deploy/check.sh
==================================================================
EOF

