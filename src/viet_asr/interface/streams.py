"""Canlı akış oturumları: kaynağı aç, gerçek zamanlı çöz, kaydet, bitince tam analize ver.

Kaynak yazımı (POST /streams "kaynak", CLI --kaynak):
    rtp://0.0.0.0:5004        UDP portunda RTP dinle (G.711 PCMU/PCMA, L16; bizim alıcı)
    rtp://0.0.0.0:0           boş port seçilir, yanıtta "port" döner
    /yol/cagri.sdp            RTP + SDP (Opus, G.722...; ffmpeg)
    http(s)://.../akim.ogg    OGG/MP3/WAV yayını (ffmpeg); rtsp://, srt://, udp:// da olur
    /yol/kayit.wav            dosyayı gerçek zaman hızında oynat (deneme; ALLOW_LOCAL_PATHS=1)

İki geçiş: canlı metin anında (StreamTranscriber), akış bitince kaydedilen ses normal işe
dönüşür (konuşmacı ayırma, güvenlik ağı, temiz transkript). İşin kimliği oturumda "son_is".
"""
from __future__ import annotations

import logging
import queue as queue_mod
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import soundfile as sf

from viet_asr.application.config import VadConfig
from viet_asr.application.stream import StreamTranscriber
from viet_asr.domain import AudioJob, StreamSegment
from viet_asr.infrastructure.ffmpeg_source import FfmpegSource
from viet_asr.infrastructure.rtp import RtpSource
from viet_asr.infrastructure.sherpa_vad import SileroStreamingVad

log = logging.getLogger(__name__)

URL_SCHEMES = ("http", "https", "rtsp", "srt", "udp", "rtmp")
# ara sonuç yalnızca hızlı motorlarda (Whisper ara çözümleri canlı akışı geride bırakır)
FAST_ENGINES = ("zipformer", "omnilingual")


class StreamError(ValueError):
    pass


@dataclass
class StreamSpec:
    kaynak: str
    dil: str = "vi"
    codec: str = "auto"
    clock_rate: int | None = None
    son_analiz: bool = True
    num_speakers: int | None = None
    callback_url: str | None = None
    idle_timeout: float = 5.0


def parse_source(spec: StreamSpec, allow_files: bool):
    """Kaynak yazımından AudioStream nesnesi."""
    k = spec.kaynak.strip()
    u = urlparse(k)
    if u.scheme == "rtp" and not k.lower().endswith(".sdp"):
        if u.port is None:
            raise StreamError("rtp://adres:port biçiminde olmalı (port 0 = boş port seç)")
        if spec.codec not in ("auto", "pcmu", "pcma", "l16"):
            raise StreamError("codec: auto | pcmu | pcma | l16 (Opus vb. için SDP dosyası verin)")
        return RtpSource(u.hostname or "0.0.0.0", u.port, spec.codec, spec.clock_rate,
                         idle_timeout=spec.idle_timeout)
    if u.scheme in URL_SCHEMES:
        return FfmpegSource(k)
    p = Path(k)
    if not allow_files:
        raise StreamError("sunucudaki dosya/SDP yolu kapalı (ALLOW_LOCAL_PATHS=1 ile açılır)")
    if not p.is_file():
        raise StreamError(f"dosya yok: {k}")
    return FfmpegSource(str(p), realtime=p.suffix.lower() != ".sdp")


@dataclass
class StreamSession:
    id: str
    spec: StreamSpec
    started_at: str
    durum: str = "dinleniyor"          # dinleniyor | bitti | hata | durduruldu
    port: int | None = None
    segments: list[dict] = field(default_factory=list)
    ara: dict | None = None
    ozet: dict | None = None
    hata: str | None = None
    son_is: str | None = None
    kayit: str | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    source: object = None
    subscribers: list[queue_mod.Queue] = field(default_factory=list)

    def to_dict(self, with_segments: bool = True) -> dict:
        d = {"id": self.id, "kaynak": self.spec.kaynak, "dil": self.spec.dil, "durum": self.durum,
             "port": self.port, "baslangic": self.started_at, "parca_sayisi": len(self.segments),
             "ozet": self.ozet, "hata": self.hata, "son_is": self.son_is}
        if with_segments:
            d["parcalar"] = self.segments
            d["ara"] = self.ara
        return d


def segment_dict(s: StreamSegment) -> dict:
    return {"index": s.index, "start": s.start, "end": s.end, "text": s.text, "final": s.final,
            "confidence": s.conf, "low_confidence_words": list(s.low_conf), "latency": s.latency}


class StreamManager:
    def __init__(self, engines, vad_cfg: VadConfig, data_dir: str | Path, submit=None,
                 languages: list[str] | None = None, max_sessions: int = 8, allow_files: bool = False,
                 max_speech: float = 10.0, vad_factory=None):
        self.engines, self.vad_cfg, self.submit = engines, vad_cfg, submit
        self.dir = Path(data_dir) / "akislar"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.languages = languages or list(engines)
        self.max_sessions, self.allow_files, self.max_speech = max_sessions, allow_files, max_speech
        self.vad_factory = vad_factory or (lambda max_speech=None: SileroStreamingVad(
            self.vad_cfg, max_speech=max_speech or self.max_speech))
        self.sessions: dict[str, StreamSession] = {}
        self._lock = threading.Lock()

    def active(self) -> int:
        return sum(1 for s in self.sessions.values() if s.durum == "dinleniyor")

    def start(self, spec: StreamSpec) -> StreamSession:
        if spec.dil not in self.languages:
            raise StreamError(f"dil '{spec.dil}' yok; seçenekler: {', '.join(self.languages)}")
        with self._lock:
            if self.active() >= self.max_sessions:
                raise StreamError(f"en fazla {self.max_sessions} canlı akış (STREAM_MAX_SESSIONS)")
            source = parse_source(spec, self.allow_files)
            self._prune()
            sid = uuid.uuid4().hex
            s = StreamSession(sid, spec, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              port=getattr(source, "port", None), source=source)
            self.sessions[sid] = s
        threading.Thread(target=self._run, args=(s,), daemon=True, name=f"akis-{sid[:8]}").start()
        return s

    def _prune(self, keep: int = 100) -> None:
        """Biten oturumların en eskilerini bellekten atar (kayıt ve tam analiz işi kalır)."""
        done = [k for k, v in self.sessions.items() if v.durum != "dinleniyor"]
        for k in done[:max(0, len(done) - keep)]:
            del self.sessions[k]

    def get(self, sid: str) -> StreamSession | None:
        return self.sessions.get(sid)

    def stop(self, sid: str) -> StreamSession | None:
        s = self.sessions.get(sid)
        if s is not None and s.durum == "dinleniyor":
            s.stop.set()
            s.source.close()
        return s

    def subscribe(self, sid: str) -> queue_mod.Queue | None:
        s = self.sessions.get(sid)
        if s is None:
            return None
        q: queue_mod.Queue = queue_mod.Queue()
        for seg in s.segments:  # geç bağlanan da baştan görsün
            q.put({"tur": "parca", **seg})
        if s.durum != "dinleniyor":
            q.put({"tur": "bitti", **s.to_dict(with_segments=False)})
        s.subscribers.append(q)
        return q

    def _publish(self, s: StreamSession, event: dict) -> None:
        for q in list(s.subscribers):
            q.put(event)

    def _run(self, s: StreamSession) -> None:
        path = self.dir / f"{s.id}.wav"
        wav = sf.SoundFile(str(path), "w", samplerate=16000, channels=1, subtype="PCM_16")
        s.kayit = str(path)

        def on_segment(seg: StreamSegment) -> None:
            d = segment_dict(seg)
            if seg.final:
                s.segments.append(d)
                s.ara = None
            else:
                s.ara = d
            self._publish(s, {"tur": "parca", **d})

        try:
            asr = self.engines[s.spec.dil]
            fast = any(k in getattr(asr, "name", "") for k in FAST_ENGINES)
            st = StreamTranscriber(self.vad_factory)
            t0 = time.monotonic()
            sm = st.run(s.source, asr, on_segment, record=wav.write, interim=fast, stop=s.stop)
            s.ozet = {"sure_sn": round(sm.duration, 1), "parca": sm.segments, "kelime": sm.words,
                      "gecikme_p95_sn": round(sm.p95_latency, 2), "gecikme_max_sn": round(sm.max_latency, 2),
                      "duvar_sn": round(time.monotonic() - t0, 1),
                      **({"rtp": dict(s.source.stats)} if hasattr(s.source, "stats") else {})}
            s.durum = "durduruldu" if s.stop.is_set() else "bitti"
        except Exception as e:  # kaynak açılamadı, ffmpeg hatası, model yok...
            log.exception("akış %s hata", s.id)
            s.durum, s.hata = "hata", f"{type(e).__name__}: {e}"
        finally:
            wav.close()
        dur = (s.ozet or {}).get("sure_sn", 0)
        if s.spec.son_analiz and self.submit is not None and dur >= 1.0:
            try:
                s.son_is = self.submit(AudioJob(uuid.uuid4().hex, str(path), s.spec.dil, s.spec.callback_url,
                                                s.spec.num_speakers, external_id=f"akis:{s.id}"))
            except Exception as e:
                log.warning("akış %s tam analize verilemedi: %s", s.id, e)
        elif dur < 1.0:
            path.unlink(missing_ok=True)
            s.kayit = None
        log.info("akış %s %s: %s", s.id, s.durum, s.ozet)
        self._publish(s, {"tur": "bitti", **s.to_dict(with_segments=False)})
