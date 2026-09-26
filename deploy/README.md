# Deploying the Traffic AI website to a server

> **O'zbekcha qisqacha:** bu papka saytni o'z serveringizga (Ubuntu, GPU'siz) `https://trafficai.dewew.dev`
> manzilida o'rnatadi. Serverdagi agentga shu faylni bering. U 1–4-qadamlarni bajaradi, natijani
> `check.sh` bilan tekshiradi va xulosani sizga aytadi. Boshqa ishlayotgan xizmatlarga tegilmaydi.

These instructions are written for an operator or an AI agent working on the target server.
Follow them in order. Do not improvise around a failed check: every script stops with a
message that says what is wrong.

## What gets installed

| piece | where | notes |
|---|---|---|
| code | `/opt/trafficai/repo` (git, `master`), site bundle in `/opt/trafficai/site` | built by `scripts/build_space.py` |
| Python env | `/opt/trafficai/venv` | CPU-only torch; ~2 GB of disk |
| weights | `/opt/trafficai/data/weights/yolo11s.pt` (19 MB) | the website demo's detector |
| service | systemd `trafficai` → Streamlit on **127.0.0.1:<free port 8600-8699>** | memory-capped (`MemoryMax=2200M`), `Nice=5`, restarts on failure, starts on boot |
| web | nginx site `/etc/nginx/sites-available/trafficai.conf` for the domain | websockets, 300 MB uploads |
| TLS | Let's Encrypt via `certbot --nginx`, HTTP → HTTPS redirect | auto-renewal (certbot timer) + nginx reload hook |
| settings | `/etc/trafficai/trafficai.env` | domain, port, branch, deployed commit |

Nothing else is changed. Existing nginx sites stay as they are, other services are never stopped,
and the Streamlit port is chosen so that it does not collide with anything already listening.

## Requirements (checked automatically)

- Ubuntu 22.04 or 24.04, x86_64, systemd, root access (`sudo`).
- **DNS:** an `A` record `trafficai.dewew.dev` → this server's public IPv4. The installer compares both and stops if they differ.
- Ports 80 and 443 reachable from the internet, and either free or already served by **nginx**.
  If Apache, Caddy, Traefik or a Docker proxy owns them, the installer stops without touching anything
  (see Troubleshooting).
- At least 4 GB free disk in `/opt`.
- **RAM:** processing one demo video peaks at about 1.2–1.4 GB. With less than 1.5 GB available the
  installer only warns. The service is capped at 2.2 GB, so it cannot starve other services, but a
  demo may then be killed and restarted.

## Steps

**Step 1 depends on whether the GitHub repository is public.** Check with
`curl -s -o /dev/null -w '%{http_code}\n' https://github.com/DeWeWO/wiut` (200 = public, 404 = private).
The competition requires a public repository, so the owner should make it public before submission.
Until then, use the private variant (1b).

```bash
# 1a. public repository: get the code, no credentials needed
sudo apt-get update && sudo apt-get install -y git
sudo git clone --depth 1 https://github.com/DeWeWO/wiut.git /opt/trafficai/repo
REPO=https://github.com/DeWeWO/wiut.git

# 1b. private repository: read-only deploy key (never ask for or store a personal token/password)
sudo apt-get update && sudo apt-get install -y git
sudo ssh-keygen -t ed25519 -N "" -C "trafficai-server" -f /root/.ssh/trafficai_deploy
sudo cat /root/.ssh/trafficai_deploy.pub
#     -> the owner adds this public key at github.com/DeWeWO/wiut -> Settings -> Deploy keys (read-only), then:
sudo tee -a /root/.ssh/config >/dev/null <<'EOF'
Host github.com
    IdentityFile /root/.ssh/trafficai_deploy
    IdentitiesOnly yes
EOF
sudo ssh-keyscan github.com | sudo tee -a /root/.ssh/known_hosts >/dev/null
sudo git clone --depth 1 git@github.com:DeWeWO/wiut.git /opt/trafficai/repo
REPO=git@github.com:DeWeWO/wiut.git

# 2. install everything (10-20 minutes, mostly the Python packages)
sudo bash /opt/trafficai/repo/deploy/install.sh --domain trafficai.dewew.dev --repo "$REPO" --email <owner e-mail>

# 3. verify
sudo bash /opt/trafficai/repo/deploy/check.sh

# 4. open https://trafficai.dewew.dev in a browser -> "Live Demo" -> "Bundled sample clip" -> Run.
#    Expected: 2 events (stop_line, red_light), a risk curve and 2 annotated clips, in about 1 minute.
```

`--email` is optional. Without it the certificate is registered without an e-mail, so there are no
expiry warnings, but renewal is automatic anyway.

`install.sh` is idempotent: running it again repairs or updates the installation and keeps the same port.

The last lines of `install.sh` print the URL, the internal port, the certificate expiry and the
renewal mechanism. `check.sh` must end with every line `OK` (exit code 0).

## Later

| task | command |
|---|---|
| deploy the latest commit | `sudo bash /opt/trafficai/repo/deploy/update.sh` |
| status report | `sudo bash /opt/trafficai/repo/deploy/check.sh` |
| logs | `journalctl -u trafficai -f` |
| restart | `sudo systemctl restart trafficai` |
| test certificate renewal | `sudo certbot renew --dry-run` |
| remove (keep files + certificate) | `sudo bash /opt/trafficai/repo/deploy/uninstall.sh` |
| remove everything | `sudo bash /opt/trafficai/repo/deploy/uninstall.sh --purge` |

## Troubleshooting

- **git clone fails with "Repository not found" / "Authentication failed"**: the repository is private.
  Use variant 1b (deploy key), or ask the owner to make it public, which the competition requires anyway.

- **"port 80/443 is served by: apache2 / caddy / docker-proxy ..."**: another web server owns the public
  ports. Do not stop it. Either add a reverse-proxy rule in that server for `trafficai.dewew.dev` →
  `http://127.0.0.1:<PORT>` (with websocket upgrade and a 300 MB body limit) and obtain the certificate
  there, or run `install.sh` with `--skip-dns-check` only after nginx owns 80/443.
- **"DOMAIN points to X but this server is Y"**: fix the DNS `A` record and wait for it to propagate
  (`getent ahostsv4 trafficai.dewew.dev`). Behind a NAT or proxy on purpose, use `--skip-dns-check`.
- **certbot fails**: port 80 is not reachable from the internet (cloud firewall or security group).
  Open 80/443 there and re-run `install.sh`.
- **The demo stops mid-run or the service restarts**: out of memory. Check with
  `journalctl -u trafficai | grep -i -e oom -e memory`, free RAM on the server or add swap, then retry.
- **Slow first page load**: normal after a restart, while the models are imported (about 10–20 s).

## What the website runs

The live demo runs the submission pipeline in a CPU setting: YOLO11-S at 768 px on every 6th
frame, frames downscaled to 1280 px, no crash/fire model, and one demo at a time (see `src/demo.py`).
The submission itself runs on the organizers' GPU with `requirements.txt` and `run_submission.py`.
This deployment does not affect it.
