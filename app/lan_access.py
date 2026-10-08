"""Opt-in LAN access for the phone test.

The demo is loopback-only (see ``local_access`` in app/server.py). ``scripts/serve_https.sh`` starts a second server on :4183 with
LIBRARY_LAN_MODE=1, which additionally accepts

  * a Host header that is exactly one of LIBRARY_LAN_HOSTS (the Mac's LAN IP and ``<name>.local``; no wildcards, so a rebinding page
    that points an attacker hostname at the Mac is still refused),
  * a client address in a private range (RFC 1918, link-local, ULA): never a public address,
  * from a non-loopback client, the pairing key LIBRARY_LAN_TOKEN, sent once as ``?k=`` and then kept in the ``lan_key`` cookie.

Without LIBRARY_LAN_MODE every function reduces to the original loopback rule. With it but without a token, non-loopback clients are
refused (fail closed).
"""
from __future__ import annotations

import hmac
import ipaddress
import os

LOOPBACK_HOSTS = {"localhost", "127.0.0.1"}
LOOPBACK_CLIENTS = {"127.0.0.1", "::1", "testclient"}
COOKIE = "lan_key"
CA_PATH = "/lan/ca.crt"        # public CA certificate for the phone to install; reachable without the key


def enabled() -> bool:
    return os.environ.get("LIBRARY_LAN_MODE", "").strip().lower() in ("1", "true", "yes", "on")


def lan_hosts() -> set[str]:
    return {h.strip().lower() for h in os.environ.get("LIBRARY_LAN_HOSTS", "").split(",") if h.strip()}


def host_ok(host: str) -> bool:
    name = host.split(":")[0].strip().lower()
    return name in LOOPBACK_HOSTS or (enabled() and name in lan_hosts())


def private_client(addr: str | None) -> bool:
    try:
        ip = ipaddress.ip_address((addr or "").split("%")[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_loopback or ip.is_link_local or (ip.is_private and not ip.is_unspecified)


def client_ok(addr: str | None) -> bool:
    return addr in LOOPBACK_CLIENTS or (enabled() and private_client(addr))


def needs_key(addr: str | None) -> bool:
    """Loopback clients (the Mac itself, 4182) never need the pairing key; every other client does."""
    return addr not in LOOPBACK_CLIENTS


def token() -> str:
    return os.environ.get("LIBRARY_LAN_TOKEN", "")


def key_ok(value: str | None) -> bool:
    t = token()
    return bool(t) and bool(value) and hmac.compare_digest(str(value), t)


def ca_file() -> str | None:
    p = os.environ.get("LIBRARY_LAN_CA", "")
    return p if p and os.path.isfile(p) else None
