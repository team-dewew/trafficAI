#!/usr/bin/env bash
# Remove the Traffic AI website. Other nginx sites and services are not touched.
#
#   sudo bash deploy/uninstall.sh            # stop + remove service and nginx site (keeps files and certificate)
#   sudo bash deploy/uninstall.sh --purge    # also delete /opt/trafficai, the user, settings and the certificate
set -Eeuo pipefail
[[ $EUID -eq 0 ]] || { echo "run as root" >&2; exit 1; }
PURGE=0; [[ "${1:-}" == "--purge" ]] && PURGE=1
ENV_FILE="/etc/trafficai/trafficai.env"
DOMAIN=""; APP_DIR="/opt/trafficai"
# shellcheck disable=SC1090
[[ -f "$ENV_FILE" ]] && . "$ENV_FILE"

systemctl disable --now trafficai 2>/dev/null || true
rm -f /etc/systemd/system/trafficai.service /etc/tmpfiles.d/trafficai.conf
systemctl daemon-reload
rm -f /etc/nginx/sites-enabled/trafficai.conf /etc/nginx/sites-available/trafficai.conf /etc/nginx/conf.d/trafficai-upgrade-map.conf
if command -v nginx >/dev/null && nginx -t 2>/dev/null; then systemctl reload nginx || true; fi
rm -f /etc/letsencrypt/renewal-hooks/deploy/trafficai-reload-nginx.sh /etc/cron.d/trafficai-certbot-renew

if [[ $PURGE -eq 1 ]]; then
  [[ -n "$DOMAIN" ]] && certbot delete --cert-name "$DOMAIN" --non-interactive 2>/dev/null || true
  rm -rf "$APP_DIR" /etc/trafficai
  id -u trafficai >/dev/null 2>&1 && userdel trafficai || true
  if [[ -f /swapfile-trafficai ]]; then
    swapoff /swapfile-trafficai 2>/dev/null || true
    sed -i '\#^/swapfile-trafficai #d' /etc/fstab
    rm -f /swapfile-trafficai
  fi
fi
echo "[trafficai] removed (purge=$PURGE)"
