"""Canlı altyazı WebRTC ile (POST /webrtc/teklif) ve HTTPS sertifikası.

Gerçek bir WebRTC bağlantısı kurulur (aiortc istemci <-> sunucu, aynı bilgisayarda UDP): ses
Opus ile gider, mesajlar veri kanalından gelir. Modeller yerine sahte motorlar (test_live).
"""
import asyncio
import fractions
import json

import numpy as np
import pytest

from test_live import TWO, make_local  # aynı klasör (pytest rootdir-dışı içe aktarma)

aiortc = pytest.importorskip("aiortc")
av = pytest.importorskip("av")
httpx = pytest.importorskip("httpx")


class PcmTrack(aiortc.MediaStreamTrack):
    """16 kHz float sesi 48 kHz'lik 20 ms Opus çerçeveleri olarak yollar; bitince sessizlik."""
    kind = "audio"

    def __init__(self, audio16k: np.ndarray):
        super().__init__()
        a48 = np.repeat(audio16k, 3)  # 16 -> 48 kHz (test için yeterli)
        self.pcm = (a48 * 32767).astype("<i2")
        self.pos = 0
        self.sent_all = asyncio.Event()

    async def recv(self):
        await asyncio.sleep(0.004)  # gerçek zamandan hızlı, ama alıcı kuyruğu taşmasın
        n = 960
        chunk = self.pcm[self.pos:self.pos + n]
        if len(chunk) < n:
            self.sent_all.set()
            chunk = np.concatenate([chunk, np.zeros(n - len(chunk), "<i2")])
        frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate, frame.pts = 48000, self.pos
        frame.time_base = fractions.Fraction(1, 48000)
        self.pos += n
        return frame


async def rtc_session(app, audio, ayarlar=None, headers=None):
    """WebRTC oturumu: teklif -> yanıt, ses, 'stop', 'done'e kadar mesajlar."""
    pc = aiortc.RTCPeerConnection(aiortc.RTCConfiguration(iceServers=[]))
    channel = pc.createDataChannel("altyazi", ordered=True)
    track = PcmTrack(audio)
    pc.addTrack(track)
    msgs, done = [], asyncio.Event()

    @channel.on("message")
    def on_message(m):
        msgs.append(json.loads(m))
        if msgs[-1]["type"] == "done":
            done.set()

    await pc.setLocalDescription(await pc.createOffer())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/webrtc/teklif", json={"sdp": pc.localDescription.sdp, "type": "offer",
                                                  "ayarlar": ayarlar or {}}, headers=headers or {})
        if r.status_code != 200:
            await pc.close()
            return r, msgs
        await pc.setRemoteDescription(aiortc.RTCSessionDescription(**{k: r.json()[k] for k in ("sdp", "type")}))
        await asyncio.wait_for(track.sent_all.wait(), 30)
        await asyncio.sleep(0.5)  # son paketler ulaşsın
        channel.send(json.dumps({"type": "stop"}))
        await asyncio.wait_for(done.wait(), 30)
    await pc.close()
    return r, msgs


def test_webrtc_ile_canli_oturum_uctan_uca(tmp_path, monkeypatch):
    app, store, sent = make_local(tmp_path, monkeypatch)
    r, msgs = asyncio.run(rtc_session(app, TWO, {"partial_sec": "0.8"}))

    assert r.status_code == 200 and r.json()["type"] == "answer"
    assert msgs[0]["type"] == "ready" and msgs[0]["session_id"] == r.json()["session_id"]
    finals = [m for m in msgs if m["type"] == "final"]
    assert [m["id"] for m in finals] == [0, 1]           # iki cümle: Opus'tan geçen ses bölündü
    assert all(m["text"].startswith("cümle") for m in finals)
    done = msgs[-1]
    assert done["type"] == "done" and done["job_id"]
    assert sent == [(done["job_id"], "tr")]               # kayıt konuşmacı ayrımı için toplu işe
    assert len(list((tmp_path / "rec").glob("*.wav"))) == 1


def test_webrtc_ayar_hatasi_400_ve_anahtar_401(tmp_path, monkeypatch):
    app, _, _ = make_local(tmp_path, monkeypatch)
    r, _ = asyncio.run(rtc_session(app, TWO[:1600], {"language": "xx"}))
    assert r.status_code == 400 and "açık değil" in r.json()["detail"]

    app2, _, _ = make_local(tmp_path / "b", monkeypatch, api_key="gizli")
    r, _ = asyncio.run(rtc_session(app2, TWO[:1600]))
    assert r.status_code == 401
    r, msgs = asyncio.run(rtc_session(app2, TWO[:1600], headers={"X-API-Key": "gizli"}))
    assert r.status_code == 200 and msgs[-1]["type"] == "done"


def test_saglik_webrtc_ve_istemci_dosyasi(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app, _, _ = make_local(tmp_path, monkeypatch)
    with TestClient(app) as c:
        assert c.get("/live/health").json()["webrtc"] is True
        js = c.get("/canli_istemci.js")
        assert js.status_code == 200 and "LiveClient" in js.text
        assert "canli_istemci.js" in c.get("/canli").text and "canli_istemci.js" in c.get("/").text
        r = c.post("/webrtc/teklif", json={"sdp": "x", "type": "answer"})
        assert r.status_code == 400


def test_kendinden_imzali_sertifika_uretilir_ve_yeniden_kullanilir(tmp_path):
    from cryptography import x509

    from viet_asr.infrastructure.tls import ensure_self_signed, local_addresses

    cert, key = ensure_self_signed(tmp_path / "tls")
    c = x509.load_pem_x509_certificate(open(cert, "rb").read())
    san = c.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    ips = {str(a) for a in san.get_values_for_type(x509.IPAddress)}
    assert "127.0.0.1" in ips and set(local_addresses()) <= ips
    assert "localhost" in san.get_values_for_type(x509.DNSName)
    before = open(cert, "rb").read()
    assert ensure_self_signed(tmp_path / "tls") == (cert, key)   # aynı adresler: yeniden üretmez
    assert open(cert, "rb").read() == before


def test_acilis_adresleri(monkeypatch):
    from viet_asr.interface import serve
    from viet_asr.interface.settings import Settings

    monkeypatch.setattr("viet_asr.infrastructure.tls.local_addresses", lambda: ["127.0.0.1", "192.168.1.20"])
    assert serve.addresses(Settings(https=True), "0.0.0.0", 8001) == [
        "https://127.0.0.1:8001", "https://192.168.1.20:8001"]
    assert serve.addresses(Settings(https=False), "127.0.0.1", 8001) == ["http://127.0.0.1:8001"]
    assert serve.tls_files(Settings(https=False)) is None
