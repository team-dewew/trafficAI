#!/usr/bin/env bash
# Health report for the deployed website. Exit code 0 = everything OK.
#
#   sudo bash /opt/trafficai/repo/deploy/check.sh
set -uo pipefail
ENV_FILE="/etc/trafficai/trafficai.env"
[[ -f "$ENV_FILE" ]] || { echo "not installed ($ENV_FILE missing)"; exit 1; }
# shellcheck disable=SC1090
. "$ENV_FILE"
fail=0
ok()  { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
bad() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; fail=1; }

echo "Traffic AI  (domain $DOMAIN, port $PORT, commit $COMMIT)"
systemctl is-active --quiet trafficai && ok "service trafficai active" || bad "service trafficai not active (journalctl -u trafficai -n 100)"
systemctl is-enabled --quiet trafficai && ok "service starts on boot" || bad "service not enabled"
[[ "$(curl -fsS --max-time 10 "http://127.0.0.1:$PORT/_stcore/health" 2>/dev/null)" == "ok" ]] \
  && ok "local health 127.0.0.1:$PORT" || bad "local health check failed"
systemctl is-active --quiet nginx && ok "nginx active" || bad "nginx not active"
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "https://$DOMAIN/_stcore/health")
[[ "$code" == "200" ]] && ok "https://$DOMAIN reachable" || bad "https://$DOMAIN/_stcore/health -> HTTP $code"
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "http://$DOMAIN/")
[[ "$code" == "301" || "$code" == "308" ]] && ok "http redirects to https ($code)" || bad "http -> https redirect missing (HTTP $code)"
end=$(echo | openssl s_client -servername "$DOMAIN" -connect "$DOMAIN:443" 2>/dev/null | openssl x509 -noout -enddate 2>/dev/null | cut -d= -f2)
if [[ -n "$end" ]]; then
  days=$(( ( $(date -d "$end" +%s) - $(date +%s) ) / 86400 ))
  (( days > 10 )) && ok "certificate valid for $days more days ($end)" || bad "certificate expires in $days days"
else
  bad "could not read the certificate"
fi
if systemctl is-active --quiet certbot.timer || systemctl is-active --quiet snap.certbot.renew.timer || [[ -f /etc/cron.d/trafficai-certbot-renew ]]; then
  ok "certificate auto-renewal scheduled"
else
  bad "no certbot renewal timer/cron"
fi
[[ -L "$APP_DIR/site/weights/yolo11s.pt" && -s "$APP_DIR/data/weights/yolo11s.pt" ]] && ok "demo weights present" || bad "demo weights missing"
mem=$(systemctl show trafficai -p MemoryCurrent --value 2>/dev/null)
[[ "$mem" =~ ^[0-9]+$ ]] && echo "  info memory used by the service: $(( mem / 1024 / 1024 )) MB"
echo "  info server memory available: $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo) MB"
exit $fail
