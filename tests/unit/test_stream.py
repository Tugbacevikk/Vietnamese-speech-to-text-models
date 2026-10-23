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


# --- dışarıdan servis: parça parça ses, SSE, canlı bildirim, port aralığı ---

def _pcm16(seconds, rate=16000):
    x = (0.3 * np.sin(2 * np.pi * 300 * np.arange(int(seconds * rate)) / rate) * 32767).astype("<i2")
    return x.tobytes()


def _push_app(tmp_path, **mgr_kw):
    from fastapi.testclient import TestClient

    from viet_asr.infrastructure.memory_store import MemoryJobStore
    from viet_asr.interface.api import create_app
    from viet_asr.interface.settings import Settings
    from viet_asr.interface.streams import StreamManager

    store = MemoryJobStore()

    class Q:
        def enqueue_detection(self, j): pass
        def enqueue_transcription(self, j, lang): pass

    s = Settings(data_dir=str(tmp_path), languages=["vi"], enable_streams=True)
    mgr = StreamManager({"vi": FakeAsr()}, None, tmp_path, submit=None, vad_factory=FakeVad, **mgr_kw)
    return TestClient(create_app(store, Q(), s, streams=mgr)), mgr


def test_push_kaynagi_siralama_tekrar_geri_basinc():
    from viet_asr.infrastructure.push_source import BacklogFull, PushSource, SeqError

    src = PushSource(8000, idle_timeout=0.4, max_backlog=1.0)
    assert src.feed(_pcm16(0.5, 8000), 0) is True
    assert src.feed(_pcm16(0.5, 8000), 0) is False          # tekrar: atlanır
    with pytest.raises(SeqError) as e:
        src.feed(b"\0\0", 5)
    assert e.value.expected == 1
    src.feed(b"\x01", None)                                  # örnek ortasında bölünmüş parça (1 bayt arta kalır)
    src.feed(_pcm16(1.0, 8000)[1:], 1)                       # bir sonraki parça baytı tamamlar
    with pytest.raises(BacklogFull):
        src.feed(_pcm16(0.1, 8000), 2)
    src.end()
    frames = list(src.frames())
    assert sum(len(f) for f in frames) == pytest.approx(16000 * 1.5, abs=64)   # 8 kHz -> 16 kHz, 1,5 sn
    assert src.stats["tekrar"] == 1


def test_push_http_sse_ve_canli_bildirim(tmp_path):
    posted = []
    c, mgr = _push_app(tmp_path, notify_secret="gizli",
                       notify_post=lambda url, body, headers: posted.append((url, body, headers)) or True)
    assert c.post("/streams/yok/audio", content=b"\0\0").status_code == 404
    r = c.post("/streams", json={"kaynak": "push", "dil": "vi", "canli_url": "http://servis.local/hook",
                                 "idle_timeout": 5.0})
    assert r.status_code == 201, r.text
    sid = r.json()["id"]
    assert r.json()["port"] is None
    for i in range(3):                                       # 3 x 1 sn, sıra numaralı
        a = c.post(f"/streams/{sid}/audio?seq={i}", content=_pcm16(1.0))
        assert a.status_code == 202, a.text
    assert a.json()["sonraki_sira"] == 3
    assert c.post(f"/streams/{sid}/audio?seq=0", content=_pcm16(1.0)).status_code == 202   # tekrar: zararsız
    assert c.post(f"/streams/{sid}/audio?seq=9", content=_pcm16(1.0)).headers["X-Expected-Seq"] == "3"
    assert c.post(f"/streams/{sid}/audio?son=true", content=b"").status_code == 202
    body = ""
    with c.stream("GET", f"/streams/{sid}/events") as resp:
        for line in resp.iter_lines():
            body += line + "\n"
    assert "event: parca" in body and "event: bitti" in body and "id: 2" in body
    with c.stream("GET", f"/streams/{sid}/events", headers={"Last-Event-ID": "1"}) as resp:   # kaldığı yerden
        again = "\n".join(resp.iter_lines())
    assert "id: 2" in again and "id: 0" not in again and "id: 1\n" not in again
    deadline = time.time() + 5
    while time.time() < deadline and not any(h["X-Event"] == "stream.ended" for _, _, h in posted):
        time.sleep(0.05)
    events = [h["X-Event"] for _, _, h in posted]
    assert events == ["stream.segment"] * 3 + ["stream.ended"]
    url, raw, hdr = posted[0]
    from viet_asr.infrastructure.webhook import verify_signature
    assert url == "http://servis.local/hook"
    assert verify_signature("gizli", hdr["X-Timestamp"], raw, hdr["X-Signature"])
    assert c.post(f"/streams/{sid}/audio", content=b"\0\0").status_code == 409        # bitti


def test_push_websocket_tek_baglanti(tmp_path):
    c, mgr = _push_app(tmp_path)
    sid = c.post("/streams", json={"kaynak": "push", "dil": "vi", "ses_hizi": 8000}).json()["id"]
    got = []
    with c.websocket_connect(f"/streams/{sid}/audio") as ws:
        for _ in range(2):
            ws.send_bytes(_pcm16(1.0, 8000))
        ws.send_text("son")
        while True:
            ev = ws.receive_json()
            got.append(ev)
            if ev["tur"] == "bitti":
                break
    finals = [e for e in got if e["tur"] == "parca" and e["final"]]
    assert len(finals) == 2 and got[-1]["durum"] == "bitti"
    assert c.post("/streams", json={"kaynak": "rtp://127.0.0.1:0"}).status_code == 201   # rtp akışına push olmaz
    rtp = [s for s in mgr.sessions.values() if s.spec.kaynak.startswith("rtp")][0]
    assert c.post(f"/streams/{rtp.id}/audio", content=b"\0\0").status_code == 409
    c.delete(f"/streams/{rtp.id}")


def test_rtp_port_araligi_ve_adres(tmp_path):
    base = 47600
    c, mgr = _push_app(tmp_path, rtp_ports=(base, base + 1), public_host="asr.ornek.lan")
    a = c.post("/streams", json={"kaynak": "rtp://0.0.0.0:0", "dil": "vi"}).json()
    b = c.post("/streams", json={"kaynak": "rtp://0.0.0.0:0", "dil": "vi"}).json()
    assert sorted([a["port"], b["port"]]) == [base, base + 1]
    assert a["rtp_adres"] == f"asr.ornek.lan:{a['port']}"
    assert c.post("/streams", json={"kaynak": "rtp://0.0.0.0:0", "dil": "vi"}).status_code == 409   # aralık dolu
    for x in (a, b):
        c.delete(f"/streams/{x['id']}")


def test_akis_kaynagi_dis_adres_denetimi(tmp_path):
    from viet_asr.interface.security import stream_source_problem
    from viet_asr.interface.settings import Settings

    dev = Settings(data_dir=str(tmp_path))
    assert stream_source_problem("http://10.0.0.5/akim.ogg", dev) is None            # geliştirmede serbest
    lst = Settings(data_dir=str(tmp_path), stream_allowed_hosts=["medya.lan"])
    assert stream_source_problem("rtsp://medya.lan/kanal1", lst) is None
    assert "izin listesinde değil" in stream_source_problem("rtsp://baska.lan/x", lst)
    for ok in ("push", "rtp://0.0.0.0:0", "udp://0.0.0.0:5004", "/yol/kayit.wav", ""):   # dışarı bağlanmaz
        assert stream_source_problem(ok, lst) is None
    prod = Settings(data_dir=str(tmp_path), app_env="production", api_key="k")
    assert "iç ağ" in stream_source_problem("http://127.0.0.1/x.ogg", prod)


def test_akis_ucu_dis_adresi_reddeder_ve_backlog_ayari(tmp_path):
    from fastapi.testclient import TestClient

    from viet_asr.infrastructure.memory_store import MemoryJobStore
    from viet_asr.interface.api import create_app
    from viet_asr.interface.settings import Settings
    from viet_asr.interface.streams import StreamManager

    s = Settings(data_dir=str(tmp_path), languages=["vi"], enable_streams=True,
                 stream_allowed_hosts=["medya.lan"], stream_max_backlog=0.5)
    mgr = StreamManager.from_settings(s, {"vi": FakeAsr()}, None)
    mgr.vad_factory = FakeVad

    class Q:
        def enqueue_detection(self, j): pass
        def enqueue_transcription(self, j, lang): pass

    c = TestClient(create_app(MemoryJobStore(), Q(), s, streams=mgr))
    r = c.post("/streams", json={"kaynak": "http://baska.lan/akim.ogg", "dil": "vi"})
    assert r.status_code == 422 and "STREAM_ALLOWED_HOSTS" in r.text
    sid = c.post("/streams", json={"kaynak": "push", "dil": "vi"}).json()["id"]
    assert c.post(f"/streams/{sid}/audio", content=_pcm16(1.0)).status_code == 202
    assert mgr.get(sid).source.max_backlog == 0.5                 # STREAM_MAX_BACKLOG kaynağa ulaştı
    c.delete(f"/streams/{sid}")


def test_kuyruk_bekleme_ve_toplam_gecikme_metrikleri():
    from viet_asr.infrastructure.push_source import PushSource
    from viet_asr.interface.streams import segment_dict

    src = PushSource(16000, idle_timeout=0.5)
    # 1. parça gönderilir
    src.feed(_pcm16(1.0), 0)
    # Kuyrukta bilerek 100 ms bekletme simülasyonu
    time.sleep(0.1)
    # 2. parça gönderilir
    src.feed(_pcm16(1.0), 1)
    src.end()

    events = []
    sm = StreamTranscriber(FakeVad).run(src, FakeAsr(), events.append, interim=False)
    finals = [e for e in events if e.final]
    assert len(finals) == 2

    # Her iki parçada da kuyruk bekleme ve toplam gecikme hesaplanmış olmalı
    for seg in finals:
        assert hasattr(seg, "kuyruk_bekleme_sn")
        assert hasattr(seg, "toplam_gecikme_sn")
        assert seg.kuyruk_bekleme_sn >= 0.0
        assert seg.toplam_gecikme_sn >= seg.latency
        assert seg.toplam_gecikme_sn >= seg.kuyruk_bekleme_sn

        d = segment_dict(seg)
        assert "kuyruk_bekleme_sn" in d
        assert "toplam_gecikme_sn" in d
        assert "latency" in d

    # 1. parça en az 100 ms kuyrukta beklemiş olmalı (t_pop - t_ingress)
    assert finals[0].kuyruk_bekleme_sn >= 0.05
    assert sm.max_kuyruk_bekleme >= 0.05
    assert sm.max_toplam_gecikme >= sm.max_latency


class FakeRefinerAsr:
    name = "faster-whisper-medium-mock"

    def transcribe(self, segs):
        out = []
        for s in segs:
            out.append(Word("Düzeltilmiş", s.start, s.start + 0.4, conf=0.98))
            out.append(Word("metin.", s.start + 0.5, s.end, conf=0.99))
        return out


def test_iki_asamali_akis_duzeltme():
    frames = [np.zeros(1600, np.float32)] * 20  # 2,0 sn
    src = ListSource(frames)
    events = []
    sm = StreamTranscriber(FakeVad).run(
        src, FakeAsr(), events.append, interim=False, refiner=FakeRefinerAsr())

    initial_finals = [e for e in events if e.final and not e.duzeltildi]
    corrections = [e for e in events if e.duzeltildi]

    assert len(initial_finals) == 2
    assert initial_finals[0].text == "k0 x"
    assert initial_finals[1].text == "k1 x"

    # Arka plan düzeltmesi her parça için gelmiş olmalı
    assert len(corrections) == 2
    assert corrections[0].index == 0
    assert corrections[0].text == "Düzeltilmiş metin."
    assert corrections[1].index == 1
    assert corrections[1].text == "Düzeltilmiş metin."
    assert sm.duzeltmeler == 2
    assert sm.duzeltme_pencere >= 1  # sayaç çalışıyor
    assert sm.duzeltme_pencere <= 2
    # düzeltme olayının 'latency' alanı çözme süresi (saniye); eskiden monotonik saatten ses konumu çıkarılıyordu (yüzlerce sn)
    assert all(0.0 <= c.latency < 5.0 for c in corrections)


def test_duzeltme_grubu_sabitleri_olculen_degerde():
    # 2 sn ve 10 sn bekleme ölçümde fark yaratmadı; eski davranış (2.0 / 29.5) korunur.
    from viet_asr.application import stream as st
    assert st.REFINE_IDLE_SEC == 2.0
    assert st.REFINE_FLUSH_SPAN == 29.5


def test_stream_manager_iki_asamali_websocket(tmp_path):
    from fastapi.testclient import TestClient
    from viet_asr.infrastructure.memory_store import MemoryJobStore
    from viet_asr.interface.api import create_app
    from viet_asr.interface.settings import Settings
    from viet_asr.interface.streams import StreamManager

    class DummyQ:
        def enqueue_detection(self, j): pass
        def enqueue_transcription(self, j, lang): pass

    s = Settings(data_dir=str(tmp_path), languages=["tr"], enable_streams=True,
                 stream_two_stage=True)
    fast_mock = FakeAsr()
    refiner_mock = FakeRefinerAsr()
    mgr = StreamManager({"tr": refiner_mock}, None, tmp_path,
                        submit=None, vad_factory=FakeVad,
                        fast_engines={"tr": fast_mock}, stream_two_stage=True)

    c = TestClient(create_app(MemoryJobStore(), DummyQ(), s, streams=mgr))
    r = c.post("/streams", json={"kaynak": "push", "dil": "tr", "iki_asama": True})
    assert r.status_code == 201
    sid = r.json()["id"]

    events = []
    # WebSocket dinleyici
    def listen():
        with c.websocket_connect(f"/streams/{sid}/ws") as ws:
            while True:
                ev = ws.receive_json()
                events.append(ev)
                if ev["tur"] == "bitti":
                    break

    t_listen = threading.Thread(target=listen)
    t_listen.start()
    time.sleep(0.05)

    # 2 sn ses gönder
    c.post(f"/streams/{sid}/audio?seq=0", content=_pcm16(1.0))
    c.post(f"/streams/{sid}/audio?seq=1", content=_pcm16(1.0))
    c.post(f"/streams/{sid}/audio?son=true", content=b"")

    t_listen.join(timeout=5.0)

    # Olayları incele
    parcalar = [e for e in events if e["tur"] == "parca" and e["final"]]
    duzeltmeler = [e for e in events if e["tur"] == "duzeltme"]

    assert len(parcalar) == 2
    assert len(duzeltmeler) == 2
    assert duzeltmeler[0]["text"] == "Düzeltilmiş metin."

    # Oturumdaki parçalar düzeltilmiş metinle güncellenmiş olmalı
    session_data = c.get(f"/streams/{sid}").json()
    assert session_data["parca_sayisi"] == 2
    assert session_data["parcalar"][0]["text"] == "Düzeltilmiş metin."
    assert session_data["ozet"]["duzeltmeler"] == 2
    assert "toplam_gecikme_p95_sn" in session_data["ozet"]


def test_iki_asama_kotasi_ikinci_akisi_hizli_modele_dusurur(tmp_path):
    from fastapi.testclient import TestClient
    from viet_asr.infrastructure.memory_store import MemoryJobStore
    from viet_asr.interface.api import create_app
    from viet_asr.interface.settings import Settings
    from viet_asr.interface.streams import StreamManager

    class DummyQ:
        def enqueue_detection(self, j): pass
        def enqueue_transcription(self, j, lang): pass

    s = Settings(data_dir=str(tmp_path), languages=["tr"], enable_streams=True, stream_two_stage=True)
    mgr = StreamManager({"tr": FakeRefinerAsr()}, None, tmp_path, submit=None, vad_factory=FakeVad,
                        fast_engines={"tr": FakeAsr()}, stream_two_stage=True, two_stage_max=1)
    c = TestClient(create_app(MemoryJobStore(), DummyQ(), s, streams=mgr))

    def ac():
        r = c.post("/streams", json={"kaynak": "push", "dil": "tr", "iki_asama": True})
        assert r.status_code == 201
        return r.json()["id"]

    def bitir(sid):
        c.post(f"/streams/{sid}/audio?seq=0", content=_pcm16(1.0))
        c.post(f"/streams/{sid}/audio?seq=1", content=_pcm16(1.0))
        c.post(f"/streams/{sid}/audio?son=true", content=b"")
        for _ in range(100):
            d = c.get(f"/streams/{sid}").json()
            if d["durum"] != "dinleniyor":
                return d
            time.sleep(0.05)
        raise AssertionError("akış bitmedi")

    a = ac()                    # kotayı tutar (henüz ses gelmedi, açık)
    time.sleep(0.3)
    b = ac()                    # kota dolu: yalnız hızlı model
    db = bitir(b)
    assert db["ozet"]["duzeltmeler"] == 0
    assert db["ozet"]["iki_asama"] is False and db["ozet"]["iki_asama_kota_doldu"] is True
    assert db["parca_sayisi"] == 2  # yine de anlık metin geldi

    da = bitir(a)
    assert da["ozet"]["duzeltmeler"] == 2 and da["ozet"]["iki_asama"] is True
    assert mgr._two_stage_active == 0  # kota geri verildi


def test_iki_asama_hizli_model_yoksa_eski_davranisa_doner(tmp_path):
    from fastapi.testclient import TestClient
    from viet_asr.infrastructure.memory_store import MemoryJobStore
    from viet_asr.interface.api import create_app
    from viet_asr.interface.settings import Settings
    from viet_asr.interface.streams import StreamManager

    class DummyQ:
        def enqueue_detection(self, j): pass
        def enqueue_transcription(self, j, lang): pass

    class Bozuk(dict):
        def __getitem__(self, k):
            raise FileNotFoundError("omnilingual-300m modeli yok")

    s = Settings(data_dir=str(tmp_path), languages=["tr"], enable_streams=True, stream_two_stage=True)
    mgr = StreamManager({"tr": FakeAsr()}, None, tmp_path, submit=None, vad_factory=FakeVad,
                        fast_engines=Bozuk(tr=1), stream_two_stage=True)
    c = TestClient(create_app(MemoryJobStore(), DummyQ(), s, streams=mgr))
    sid = c.post("/streams", json={"kaynak": "push", "dil": "tr", "iki_asama": True}).json()["id"]
    c.post(f"/streams/{sid}/audio?seq=0", content=_pcm16(1.0))
    c.post(f"/streams/{sid}/audio?seq=1", content=_pcm16(1.0))
    c.post(f"/streams/{sid}/audio?son=true", content=b"")
    for _ in range(100):
        d = c.get(f"/streams/{sid}").json()
        if d["durum"] != "dinleniyor":
            break
        time.sleep(0.05)
    assert d["durum"] == "bitti" and d["hata"] is None
    assert d["ozet"]["iki_asama"] is False and d["parca_sayisi"] == 2
