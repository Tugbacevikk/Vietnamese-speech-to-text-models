"""Canlı akış: RTP çözme, sıra tamponu, alıcı; akış çözümleyici; /streams uçları."""
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from viet_asr.application.stream import StreamTranscriber
from viet_asr.domain import VadSegment, Word
from viet_asr.infrastructure.rtp import ALAW, ULAW, JitterBuffer, RtpSource, decode, parse

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from rtp_gonder import packets, ulaw_encode  # noqa: E402


def pkt(seq, payload=b"\xff" * 160, pt=0, ts=0, ssrc=1, ext=False):
    b0 = 0x80 | (0x10 if ext else 0)
    hdr = struct.pack("!BBHII", b0, pt, seq & 0xFFFF, ts, ssrc)
    if ext:
        hdr += struct.pack("!HH", 0xBEDE, 1) + b"\0\0\0\0"
    return hdr + payload


def test_baslik_cozme_ve_g711():
    p = parse(pkt(7, b"\x01\x02", pt=8, ts=320, ssrc=0xABCD, ext=True))
    assert (p.pt, p.seq, p.ts, p.ssrc, p.payload) == (8, 7, 320, 0xABCD, b"\x01\x02")
    assert parse(b"\x00" * 20) is None and parse(b"\x80\xc8" + b"\0" * 10) is None   # RTP değil / RTCP
    assert ULAW[0xFF] == 0.0 and ULAW[0x00] == pytest.approx(-32124 / 32768)
    assert ALAW[0xD5] == pytest.approx(8 / 32768)
    x = np.sin(np.linspace(0, 20, 800)).astype(np.float32) * 0.5
    y = decode(ulaw_encode(x), "pcmu")
    assert 10 * np.log10((x ** 2).mean() / ((x - y) ** 2).mean()) > 30            # G.711 ~38 dB
    assert np.allclose(decode(struct.pack(">2h", 16384, -16384), "l16"), [0.5, -0.5])


def test_sira_tamponu_siralar_kaybi_doldurur_sarmayi_bilir():
    jb = JitterBuffer(depth=2)
    seqs = [65534, 0, 65535, 1, 3, 4, 5, 6]          # 0 ile 65535 yer değiştirmiş, 2 kayıp, sarma var
    out = []
    for s in seqs:
        out += jb.push(parse(pkt(s)))
    out += jb.drain()
    assert [p.seq if p else None for p in out] == [65534, 65535, 0, 1, None, 3, 4, 5, 6]
    assert jb.push(parse(pkt(1))) == []                 # geç gelen eski paket atılır


def test_rtp_alici_gercek_udp_kayip_ve_yabanci_akis():
    src = RtpSource("127.0.0.1", 0, idle_timeout=0.5)
    audio = (0.3 * np.sin(2 * np.pi * 300 * np.arange(16000) / 16000)).astype(np.float32)
    ps = list(packets(audio, "pcmu"))                    # 50 paket x 20 ms
    def send():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for i, p in enumerate(ps):
            if i == 10:
                continue                                 # kayıp
            s.sendto(p, ("127.0.0.1", src.port))
            if i == 5:
                s.sendto(pkt(999, ssrc=42), ("127.0.0.1", src.port))   # başka akış
            time.sleep(0.001)
    threading.Thread(target=send).start()
    got = np.concatenate(list(src.frames()))
    assert abs(len(got) - 16000) < 400                   # kayıp yeri sessizlikle dolar, süre korunur
    assert src.stats["kayip"] == 1 and src.stats["yabanci"] == 1 and src.stats["paket"] == 49


class FakeVad:
    """Her 1 sn'de bir parça biter; konuşma sürerken ongoing() son 0,5 sn'yi verir."""

    def __init__(self):
        self.buf, self.t = [], 0

    def feed(self, x):
        self.buf.append(x)
        n = sum(len(b) for b in self.buf)
        if n >= 16000:
            a = np.concatenate(self.buf)
            seg = VadSegment(self.t / 16000, a[:16000])
            self.t += 16000
            self.buf = [a[16000:]]
            return [seg]
        return []

    def ongoing(self):
        n = sum(len(b) for b in self.buf)
        return VadSegment(self.t / 16000, np.zeros(n, np.float32)) if n >= 8000 else None

    def flush(self):
        a = np.concatenate(self.buf) if self.buf else np.zeros(0, np.float32)
        return [VadSegment(self.t / 16000, a)] if len(a) else []


class FakeAsr:
    name = "zipformer-test"

    def transcribe(self, segs):
        s = segs[0]
        return [Word(f"k{int(s.start)}", s.start, s.start + 0.3, conf=0.4), Word("x", s.start + 0.4, s.end, conf=0.9)]


class ListSource:
    def __init__(self, frames):
        self.fr, self.closed = frames, False

    def frames(self):
        yield from self.fr

    def close(self):
        self.closed = True


def test_akis_cozumleyici_final_ara_kayit():
    frames = [np.zeros(1600, np.float32)] * 25            # 2,5 sn, 0,1 sn'lik parçalar
    src, events, rec = ListSource(frames), [], []
    sm = StreamTranscriber(FakeVad, interim_every=0.5, min_interim=0.4).run(
        src, FakeAsr(), events.append, record=rec.append)
    finals = [e for e in events if e.final]
    assert [(e.index, e.start, e.text) for e in finals] == [(0, 0.0, "k0 x"), (1, 1.0, "k1 x"), (2, 2.0, "k2 x")]
    assert finals[0].low_conf == (0,) and finals[0].conf == pytest.approx(0.65)
    assert any(not e.final for e in events)               # ara sonuçlar da geldi
    assert sum(len(r) for r in rec) == 40000 and src.closed
    assert sm.segments == 3 and sm.duration == pytest.approx(2.5)
    ev2 = []
    StreamTranscriber(FakeVad).run(ListSource(frames), FakeAsr(), ev2.append, interim=False)
    assert all(e.final for e in ev2)


def test_akis_uclari_ve_websocket(tmp_path):
    from fastapi.testclient import TestClient

    from viet_asr.infrastructure.memory_store import MemoryJobStore
    from viet_asr.interface.api import create_app
    from viet_asr.interface.settings import Settings
    from viet_asr.interface.streams import StreamManager

    class Q:
        def __init__(self):
            self.sent = []

        def enqueue_detection(self, j):
            self.sent.append(j)

        def enqueue_transcription(self, j, lang):
            self.sent.append((j, lang))

    from viet_asr.application.jobs import SubmitJob
    store, q = MemoryJobStore(), Q()
    s = Settings(data_dir=str(tmp_path), languages=["vi"], enable_streams=True)
    mgr = StreamManager({"vi": FakeAsr()}, None, tmp_path, submit=SubmitJob(store, q, ["vi"]),
                        vad_factory=FakeVad)
    c = TestClient(create_app(store, q, s, streams=mgr))
    assert c.get("/health").json()["features"]["streams"] is True
    assert c.post("/streams", json={"kaynak": "ftp://x", "dil": "vi"}).status_code == 422
    assert c.post("/streams", json={"kaynak": "/etc/passwd", "dil": "vi"}).status_code == 422  # dosya kapalı
    assert c.post("/streams", json={"kaynak": "rtp://127.0.0.1:0", "dil": "tr"}).status_code == 422
    r = c.post("/streams", json={"kaynak": "rtp://127.0.0.1:0", "dil": "vi", "idle_timeout": 1.0})
    assert r.status_code == 201, r.text
    sid, port = r.json()["id"], r.json()["port"]
    audio = (0.3 * np.sin(2 * np.pi * 300 * np.arange(16000 * 3) / 16000)).astype(np.float32)
    def send():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for p in packets(audio, "pcmu"):
            sock.sendto(p, ("127.0.0.1", port))
            time.sleep(0.002)
    threading.Thread(target=send).start()
    events = []
    with c.websocket_connect(f"/streams/{sid}/ws") as ws:
        while True:
            ev = ws.receive_json()
            events.append(ev)
            if ev["tur"] == "bitti":
                break
    finals = [e for e in events if e["tur"] == "parca" and e["final"]]
    assert len(finals) == 3 and finals[0]["text"] == "k0 x"
    end = events[-1]
    assert end["durum"] == "bitti" and end["ozet"]["rtp"]["paket"] == 150 and end["son_is"]
    rec = store.get(end["son_is"])
    assert rec.job.external_id == f"akis:{sid}" and Path(rec.job.audio_path).is_file() and q.sent
    got = c.get(f"/streams/{sid}").json()
    assert got["parca_sayisi"] == 3 and got["parcalar"][2]["text"] == "k2 x"
    assert c.delete(f"/streams/{sid}").json()["durum"] == "bitti"
    off = TestClient(create_app(store, q, Settings(data_dir=str(tmp_path), languages=["vi"])))
    assert off.post("/streams", json={"kaynak": "rtp://127.0.0.1:0"}).status_code == 404


class FullAsr(FakeAsr):
    segmentation = "full"

    def __init__(self):
        self.seen = []

    def transcribe(self, segs):
        self.seen.append((round(segs[0].start, 3), round(segs[0].end, 3)))
        return super().transcribe(segs)


class GapVad(FakeVad):
    """Konuşma 0,5-1,0 sn'de; sonrasında hiç parça bitmez (müzik altında konuşma gibi)."""

    def __init__(self, **kw):
        super().__init__()
        self.kw, self.done = kw, False

    def feed(self, x):
        self.t += len(x)
        if not self.done and self.t >= 16000:
            self.done = True
            return [VadSegment(0.5, np.zeros(8000, np.float32))]
        return []

    def ongoing(self):
        return None

    def flush(self):
        return []


def test_bitisik_mod_sesi_atlamaz_ve_zorla_keser():
    frames = [np.zeros(1600, np.float32)] * 50             # 5 sn
    asr, made = FullAsr(), []

    def factory(**kw):
        made.append(kw)
        return GapVad(**kw)

    StreamTranscriber(factory, max_chunk=2.0, min_chunk=0.5).run(ListSource(frames), asr, lambda e: None, interim=False)
    assert made == [{"max_speech": 4.0}]                    # VAD zorla kesmesin, biz keselim
    assert asr.seen[0] == (0.0, 1.0)                        # VAD'ın kaçırdığı 0-0,5 sn de çözülür
    assert all(b[0] == a[1] for a, b in zip(asr.seen, asr.seen[1:]))   # bitişik, boşluk yok
    assert asr.seen[-1][1] == 5.0 and all(e - s <= 2.0 + 1e-6 for s, e in asr.seen)


def test_akis_ucu_sdp_metniyle(tmp_path):
    from fastapi.testclient import TestClient

    from viet_asr.infrastructure.memory_store import MemoryJobStore
    from viet_asr.interface.api import create_app
    from viet_asr.interface.settings import Settings
    from viet_asr.interface.streams import StreamManager

    store = MemoryJobStore()
    mgr = StreamManager({"vi": FakeAsr()}, None, tmp_path, submit=None, vad_factory=FakeVad)
    c = TestClient(create_app(store, None, Settings(data_dir=str(tmp_path), languages=["vi"], enable_streams=True),
                              streams=mgr))
    assert c.post("/streams", json={"dil": "vi"}).status_code == 422          # kaynak da sdp de yok
    sdp_text = "v=0\nc=IN IP4 127.0.0.1\nm=audio 0 RTP/AVP 96\na=rtpmap:96 L16/16000/1\n"
    r = c.post("/streams", json={"sdp": sdp_text, "kaynak": "rtp://127.0.0.1:0", "dil": "vi", "idle_timeout": 1.0})
    assert r.status_code == 201, r.text
    port = r.json()["port"]
    audio = (0.3 * np.sin(2 * np.pi * 300 * np.arange(32000) / 16000)).astype(np.float32)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for p in packets(audio, "l16"):                                        # yük türü 96, L16 16 kHz
        sock.sendto(p, ("127.0.0.1", port))
        time.sleep(0.002)
    for _ in range(100):
        if mgr.get(r.json()["id"]).durum != "dinleniyor":
            break
        time.sleep(0.1)
    got = c.get(f"/streams/{r.json()['id']}").json()
    assert got["durum"] == "bitti" and got["parca_sayisi"] == 2 and got["ozet"]["rtp"]["paket"] == 100
