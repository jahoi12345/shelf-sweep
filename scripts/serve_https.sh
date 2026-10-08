#!/usr/bin/env bash
# HTTPS on the LAN for the iPhone Safari test (getUserMedia needs a secure context).
#
#   scripts/serve_https.sh            # serve on https://<LAN IP>:4183 (the HTTP site on :4182 is untouched)
#   scripts/serve_https.sh --print    # create/refresh the certificate, print the URL and the phone steps, do not start the server
#
# One-time: a local CA (EC P-256, 10 years) under runs/_server_state/tls/ (never committed). Every start checks the leaf certificate
# (SAN = LAN IP, <LocalHostName>.local, *.local, localhost, 127.0.0.1; 365 days) and re-issues it only when the LAN IP/host name changed
# or it expires within 30 days, so the CA the phone trusts stays valid. mkcert is used for the leaf when it is on PATH (FORCE_OPENSSL=1 skips it).
# Environment overrides: LAN_IP, HTTPS_PORT (4183), LIBRARY_TLS_DIR, PYTHON.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"
PORT="${HTTPS_PORT:-4183}"
RUNS="${LIBRARY_RUNS_DIR:-$ROOT/runs}"
TLS="${LIBRARY_TLS_DIR:-$RUNS/_server_state/tls}"
PRINT_ONLY=0
[ "${1:-}" = "--print" ] && PRINT_ONLY=1

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in "$ROOT/.venv/bin/python" "$(git rev-parse --git-common-dir 2>/dev/null)/../.venv/bin/python"; do
    [ -x "$c" ] && PY="$c" && break
  done
fi
[ -n "$PY" ] || { echo "No .venv/bin/python found; set PYTHON=..." >&2; exit 1; }

# ---- LAN address and mDNS name
IFACE="$(route -n get default 2>/dev/null | awk '/interface:/{print $2}' || true)"
LAN_IP="${LAN_IP:-}"
for i in "$IFACE" en0 en1; do
  [ -n "$LAN_IP" ] && break
  [ -n "$i" ] && LAN_IP="$(ipconfig getifaddr "$i" 2>/dev/null || true)"
done
[ -n "$LAN_IP" ] || { echo "Could not find the Mac's LAN IP (is Wi-Fi up?). Set LAN_IP=192.168.x.y" >&2; exit 1; }
HOST_LOCAL="$(scutil --get LocalHostName 2>/dev/null || hostname -s).local"
SAN="DNS:localhost,DNS:${HOST_LOCAL},DNS:*.local,IP:127.0.0.1,IP:${LAN_IP}"

mkdir -p "$TLS"
chmod 700 "$TLS"
cd "$TLS"

# ---- pairing key (kept across restarts so the phone's cookie stays valid)
[ -s lan_token ] || { umask 077; openssl rand -hex 16 > lan_token; }
TOKEN="$(cat lan_token)"

# ---- CA (once) and leaf certificate (when the SAN set changes or it is about to expire)
USE_MKCERT=0
if command -v mkcert >/dev/null 2>&1 && [ -z "${FORCE_OPENSSL:-}" ]; then USE_MKCERT=1; fi

need_leaf=0
if [ ! -s server.crt ] || [ ! -s server.key ] || [ "$(cat server.san 2>/dev/null)" != "$SAN" ] \
   || ! openssl x509 -checkend $((30 * 86400)) -noout -in server.crt >/dev/null 2>&1; then need_leaf=1; fi

if [ "$USE_MKCERT" = 1 ]; then
  if [ "$need_leaf" = 1 ]; then
    mkcert -cert-file server.crt -key-file server.key localhost 127.0.0.1 "$LAN_IP" "$HOST_LOCAL" "*.local" >/dev/null 2>&1
    echo "$SAN" > server.san
  fi
  cp "$(mkcert -CAROOT)/rootCA.pem" ca.crt
else
  if [ ! -s ca.crt ] || [ ! -s ca.key ]; then
    (umask 077; openssl ecparam -name prime256v1 -genkey -noout -out ca.key)
    printf 'basicConstraints=critical,CA:TRUE\nkeyUsage=critical,keyCertSign,cRLSign\nsubjectKeyIdentifier=hash\n' > ca.ext
    openssl req -new -key ca.key -subj "/CN=Library Claim LAN CA" -out ca.csr
    openssl x509 -req -in ca.csr -signkey ca.key -days 3650 -sha256 -extfile ca.ext -out ca.crt >/dev/null 2>&1
    rm -f ca.csr server.crt   # a new CA invalidates any old leaf
    need_leaf=1
  fi
  if [ "$need_leaf" = 1 ]; then
    (umask 077; openssl ecparam -name prime256v1 -genkey -noout -out server.key)
    printf 'basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\nextendedKeyUsage=serverAuth\nsubjectAltName=%s\nauthorityKeyIdentifier=keyid\n' "$SAN" > server.ext
    openssl req -new -key server.key -subj "/CN=${LAN_IP}" -out server.csr
    openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -days 365 -sha256 -extfile server.ext -out server.crt >/dev/null 2>&1
    rm -f server.csr
    echo "$SAN" > server.san
  fi
fi
chmod 600 server.key lan_token
cd "$ROOT"

URL="https://${LAN_IP}:${PORT}/"
PAIR="${URL}?k=${TOKEN}"
CA_FILE="$TLS/ca.crt"

cat <<EOF

=== iPhone Safari over HTTPS (LAN) ===
Mac LAN address : ${LAN_IP}   (also ${HOST_LOCAL})
Open on the iPhone (same Wi-Fi), after the one-time trust below:

    ${PAIR}

The ?k=... part is the pairing key: the first visit stores it in a cookie, then Safari drops it from the address bar.
Without it every request from the phone is refused (403).

One-time certificate trust on the iPhone (the CA stays valid when the Mac's IP changes):
  1. AirDrop this file to the iPhone:  ${CA_FILE}
     (or in Safari open  https://${LAN_IP}:${PORT}/lan/ca.crt  and tap Advanced > proceed once)
  2. iPhone: Settings > "Profile Downloaded" > Install (enter the passcode) > Install.
  3. iPhone: Settings > General > About > Certificate Trust Settings > switch ON "Library Claim LAN CA".
  4. Open the URL above in Safari. Tap Start (camera + mic prompts appear once; allow both).
  5. Hold the phone sideways (landscape). Upright (portrait) frames show about half as much shelf, so every book is seen in half as many frames.
  6. Optional: add ?res=4k to the URL (e.g. ${URL}?res=4k) to ask the phone for 3840x2160; the Sweep page then shows the size it granted and the frame rate it sustained.
If the Mac asks "Allow python to accept incoming network connections?" click Allow. Wi-Fi with client isolation (some office/campus
networks) blocks phone-to-Mac traffic: use the phone's Personal Hotspot with the Mac joined to it.
EOF

if [ "$PRINT_ONLY" = 1 ]; then exit 0; fi

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already in use; is serve_https.sh already running?" >&2
  exit 1
fi

# The HTTP site on :4182 keeps running; this server shares runs/ with it, so it must not re-queue that server's unfinished builds at start.
RESUME=1
if lsof -nP -iTCP:4182 -sTCP:LISTEN >/dev/null 2>&1; then RESUME=0; fi

echo
echo "Serving https://0.0.0.0:${PORT} (private-range clients with the pairing key only). Ctrl-C to stop."
export LIBRARY_LAN_MODE=1
export LIBRARY_LAN_HOSTS="${LAN_IP},${HOST_LOCAL}"
export LIBRARY_LAN_TOKEN="$TOKEN"
export LIBRARY_LAN_CA="$CA_FILE"
export LIBRARY_POST_SWEEP_RESUME="$RESUME"
export LIBRARY_CANONICAL_URL="$URL"
exec nice "$PY" -m uvicorn app.server:app --host 0.0.0.0 --port "$PORT" \
  --ssl-keyfile "$TLS/server.key" --ssl-certfile "$TLS/server.crt"
