"""LAN/HTTPS phone test access: loopback behaviour is unchanged; LAN mode is opt-in, host-allowlisted, private-range only and key-gated."""
import pytest
from fastapi.testclient import TestClient

from app import lan_access as LAN
from app import server as s

IP = "10.0.0.41"
KEY = "k" * 32


@pytest.fixture()
def lan(monkeypatch, tmp_path):
    ca = tmp_path / "ca.crt"
    ca.write_text("-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n")
    monkeypatch.setenv("LIBRARY_LAN_MODE", "1")
    monkeypatch.setenv("LIBRARY_LAN_HOSTS", f"{IP},Mac-Test.local")
    monkeypatch.setenv("LIBRARY_LAN_TOKEN", KEY)
    monkeypatch.setenv("LIBRARY_LAN_CA", str(ca))


def phone(client_ip=IP, base=f"https://{IP}:4183"):
    return TestClient(s.app, base_url=base, client=(client_ip, 5000))


def test_default_is_loopback_only(monkeypatch):
    for k in ("LIBRARY_LAN_MODE", "LIBRARY_LAN_HOSTS", "LIBRARY_LAN_TOKEN", "LIBRARY_LAN_CA"):
        monkeypatch.delenv(k, raising=False)
    assert LAN.host_ok("127.0.0.1:4182") and LAN.host_ok("localhost") and not LAN.host_ok(f"{IP}:4183")
    assert LAN.client_ok("127.0.0.1") and LAN.client_ok("::1") and not LAN.client_ok(IP)
    assert phone().get("/api/build").status_code == 403
    assert TestClient(s.app, base_url="http://127.0.0.1:4182").get("/api/build").status_code == 200


def test_private_ranges_only(lan):
    for ok in ("10.0.0.5", "172.16.3.4", "192.168.1.9", "169.254.1.1", "::1", "fd00::1", "::ffff:192.168.1.9"):
        assert LAN.client_ok(ok), ok
    for bad in ("8.8.8.8", "100.64.0.1", "2001:4860:4860::8888", "", None, "not-an-ip", "0.0.0.0"):
        assert not LAN.client_ok(bad), bad


def test_host_is_an_exact_allowlist(lan):
    assert LAN.host_ok(f"{IP}:4183") and LAN.host_ok("mac-test.local:4183") and LAN.host_ok("localhost")
    assert not LAN.host_ok("evil.example:4183") and not LAN.host_ok("10.0.0.42:4183") and not LAN.host_ok("x.mac-test.local")


def test_public_client_is_refused_even_with_the_key(lan):
    c = phone(client_ip="8.8.8.8")
    assert c.get(f"/?k={KEY}", follow_redirects=False).status_code == 403


def test_phone_needs_the_pairing_key_then_keeps_a_cookie(lan):
    c = phone()
    assert c.get("/", follow_redirects=False).status_code == 403
    assert c.get("/?k=wrong", follow_redirects=False).status_code == 403
    r = c.get(f"/?k={KEY}&x=1", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/?x=1"           # the key never stays in the address bar
    sc = r.headers["set-cookie"].lower()
    assert "lan_key=" in sc and "httponly" in sc and "secure" in sc and "samesite=lax" in sc
    assert c.get("/", follow_redirects=False).status_code == 200                # cookie jar now carries it
    assert c.post("/api/sessions", headers={"origin": f"https://{IP}:4183"}).status_code == 200
    assert c.post("/api/sessions", headers={"origin": "https://evil.example"}).status_code == 403   # origin rule unchanged


def test_key_in_query_is_not_accepted_for_posts(lan):
    r = phone().post(f"/api/sessions?k={KEY}", headers={"origin": f"https://{IP}:4183"}, follow_redirects=False)
    assert r.status_code == 403


def test_lan_mode_without_a_token_fails_closed(lan, monkeypatch):
    monkeypatch.delenv("LIBRARY_LAN_TOKEN")
    assert phone().get("/?k=", follow_redirects=False).status_code == 403
    assert TestClient(s.app, base_url="http://127.0.0.1:4182").get("/api/build").status_code == 200   # the Mac itself is unaffected


def test_ca_certificate_is_public_in_lan_mode_only(lan, monkeypatch):
    r = phone().get("/lan/ca.crt")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-x509-ca-cert") and b"BEGIN CERTIFICATE" in r.content
    monkeypatch.delenv("LIBRARY_LAN_MODE")
    assert TestClient(s.app, base_url="http://127.0.0.1:4182").get("/lan/ca.crt").status_code == 404


def test_websocket_requires_the_key_cookie(lan, monkeypatch):
    # TestClient websockets always report client "testclient", Host "testserver" and scheme ws, so the non-loopback rule is forced and the origin uses http;
    # the real wss path (TLS, LAN client, cookie) was exercised against scripts/serve_https.sh (iPhone Safari).
    base = f"http://{IP}:4183"
    # This tests LAN authentication only, regardless of credentials in .env.
    # Never open a real hosted voice connection from this unit test.
    monkeypatch.setattr(s, "voice_status", lambda: (False, "test: LAN authentication only"))
    c = TestClient(s.app, base_url=base)
    monkeypatch.setattr(LAN, "needs_key", lambda addr: True)
    assert c.get(f"/?k={KEY}", follow_redirects=False).status_code == 303
    r = c.post("/api/sessions", headers={"origin": base})
    sid, owner = r.json()["session_id"], c.cookies.get("intake_owner")
    assert owner
    with c.websocket_connect(f"/ws/live?session_id={sid}", headers={"host": f"{IP}:4183", "origin": base, "cookie": f"intake_owner={owner}; lan_key={KEY}"}) as ws:
        assert ws.receive_json()["type"] == "session"
    for cookie in (f"intake_owner={owner}", f"intake_owner={owner}; lan_key=wrong"):       # the session owner cookie alone is not enough from the LAN
        with pytest.raises(Exception):
            with c.websocket_connect(f"/ws/live?session_id={sid}", headers={"host": f"{IP}:4183", "origin": base, "cookie": cookie}):
                pass
