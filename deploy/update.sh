#!/usr/bin/env bash
# Deploy the latest commit of the configured branch (re-runs install.sh with the saved settings).
#
#   sudo bash /opt/trafficai/repo/deploy/update.sh [--branch NAME] [--skip-smoke]
set -Eeuo pipefail
ENV_FILE="/etc/trafficai/trafficai.env"
[[ $EUID -eq 0 ]] || { echo "run as root: sudo bash $0" >&2; exit 1; }
[[ -f "$ENV_FILE" ]] || { echo "$ENV_FILE not found: run deploy/install.sh first" >&2; exit 1; }
# shellcheck disable=SC1090
. "$ENV_FILE"

EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --branch) BRANCH="$2"; shift 2 ;;
    --skip-smoke) EXTRA+=(--skip-smoke); shift ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

REPO_DIR="$APP_DIR/repo"
git -c safe.directory="$REPO_DIR" -C "$REPO_DIR" fetch --quiet --depth 1 origin "$BRANCH"
git -c safe.directory="$REPO_DIR" -C "$REPO_DIR" reset --quiet --hard FETCH_HEAD
echo "[trafficai] updating to $(git -c safe.directory="$REPO_DIR" -C "$REPO_DIR" rev-parse --short HEAD) ($BRANCH)"
exec bash "$REPO_DIR/deploy/install.sh" --domain "$DOMAIN" --branch "$BRANCH" --repo "$REPO_URL" \
  --dir "$APP_DIR" --port "$PORT" "${EXTRA[@]}"
