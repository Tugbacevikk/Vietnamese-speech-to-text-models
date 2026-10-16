"""RTP ile gelen canlı sesi dinler ve 16 kHz mono float32 parçalar halinde verir.

RTP (RFC 3550): santral/IP telefon sesi UDP üzerinden ~20 ms'lik paketlerle taşır. Her pakette
12 baytlık başlık (sürüm, yük türü, sıra no, zaman damgası, SSRC) ve sıkıştırılmış ses var.
Burada doğrudan çözülen biçimler (ek kütüphane gerekmez):
  PCMU (G.711 mu-law, yük türü 0, 8 kHz)   telefonun en yaygın biçimi
  PCMA (G.711 A-law, yük türü 8, 8 kHz)
  L16  (16 bit ham ses, büyük-endian; yük türü 10/11 ya da dinamik, clock_rate ile)
Opus, G.722 vb. için: ffmpeg_source.FfmpegSource + SDP dosyası.

Ağ gerçekleri:
  - Paketler sırasız gelebilir: kısa bir sıra tamponu (jitter buffer) sıra numarasına göre dizer.
  - Paket kaybolabilir: eksik paketin yeri sessizlikle doldurulur (zaman çizgisi kaymasın).
  - Aynı porta başka bir akış (SSRC) karışabilir: ilk gelen akışa kilitlenilir, diğerleri atılır.
  - Görüşme bitince paket gelmez: idle_timeout sn sessizlikten sonra akış biter.
"""
from __future__ import annotations

import logging
import socket
import struct
import threading
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import soxr

log = logging.getLogger(__name__)

SR = 16000
STATIC_PT = {0: ("pcmu", 8000), 8: ("pcma", 8000), 10: ("l16", 44100), 11: ("l16", 44100)}


def _ulaw_table() -> np.ndarray:
    u = ~np.arange(256, dtype=np.int32) & 0xFF
    sign, exp, mant = u & 0x80, (u >> 4) & 0x07, u & 0x0F
    mag = (((mant << 3) + 0x84) << exp) - 0x84
    return (np.where(sign, -mag, mag) / 32768.0).astype(np.float32)


def _alaw_table() -> np.ndarray:
    a = np.arange(256, dtype=np.int32) ^ 0x55
    sign, exp, mant = a & 0x80, (a >> 4) & 0x07, a & 0x0F
    mag = np.where(exp == 0, (mant << 4) + 8, ((mant << 4) + 0x108) << np.maximum(exp - 1, 0))
    return (np.where(sign, mag, -mag) / 32768.0).astype(np.float32)


ULAW, ALAW = _ulaw_table(), _alaw_table()


def decode(payload: bytes, codec: str) -> np.ndarray:
    if codec == "pcmu":
        return ULAW[np.frombuffer(payload, dtype=np.uint8)]
    if codec == "pcma":
        return ALAW[np.frombuffer(payload, dtype=np.uint8)]
    if codec == "l16":
        return np.frombuffer(payload, dtype=">i2").astype(np.float32) / 32768.0
    raise ValueError(f"desteklenmeyen RTP biçimi: {codec} (ffmpeg kaynağını kullanın)")


@dataclass(frozen=True)
class RtpPacket:
    pt: int
    seq: int
    ts: int
    ssrc: int
    payload: bytes


def parse(data: bytes) -> RtpPacket | None:
    """RTP başlığını çözer; RTP değilse (ör. RTCP, bozuk) None."""
    if len(data) < 12 or data[0] >> 6 != 2:
        return None
    b0, b1, seq, ts, ssrc = struct.unpack("!BBHII", data[:12])
    pt = b1 & 0x7F
    if 72 <= pt <= 76:  # RTCP aynı porta düşmüş
        return None
    off = 12 + 4 * (b0 & 0x0F)
    if b0 & 0x10:  # uzantı başlığı
        if len(data) < off + 4:
            return None
        off += 4 + 4 * struct.unpack("!H", data[off + 2:off + 4])[0]
    end = len(data)
    if b0 & 0x20 and end > off:  # dolgu
        end -= data[-1]
    if off > end:
        return None
    return RtpPacket(pt, seq, ts, ssrc, data[off:end])


class JitterBuffer:
    """Sıra numarasına göre dizer; `depth` paket bekler, sonra eksik olanı kayıp sayar.
    Çıktı: (paket ya da None=kayıp) sırayla. 16 bit sıra numarasının sarmasını bilir."""

    def __init__(self, depth: int = 8):
        self.depth = depth
        self.next: int | None = None
        self.pending: dict[int, RtpPacket] = {}

    def push(self, p: RtpPacket) -> list[RtpPacket | None]:
        if self.next is None:
            self.next = p.seq
        if (p.seq - self.next) & 0xFFFF >= 0x8000:  # geç kalmış (zaten geçildi): at
            return []
        self.pending[p.seq] = p
        out: list[RtpPacket | None] = []
        while self.pending:
            if self.next in self.pending:
                out.append(self.pending.pop(self.next))
            elif len(self.pending) > self.depth:
                out.append(None)  # bekledik, gelmedi: kayıp
            else:
                break
            self.next = (self.next + 1) & 0xFFFF
        return out

    def drain(self) -> list[RtpPacket | None]:
        out: list[RtpPacket | None] = []
        while self.pending:
            out.append(self.pending.pop(self.next, None))
            self.next = (self.next + 1) & 0xFFFF
        return out


class RtpSource:
    """UDP portunu dinler; `frames()` 16 kHz float32 parçalar verir (AudioStream portu).

    codec: "auto" (yük türünden: 0 PCMU, 8 PCMA), "pcmu", "pcma", "l16".
    clock_rate: L16 ya da dinamik yük türü için örnekleme hızı.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5004, codec: str = "auto",
                 clock_rate: int | None = None, idle_timeout: float = 5.0, first_timeout: float = 60.0,
                 jitter_depth: int = 8):
        self.host, self.port, self.codec, self.clock_rate = host, port, codec, clock_rate
        self.idle_timeout, self.first_timeout = idle_timeout, first_timeout
        self.jitter = JitterBuffer(jitter_depth)
        self._stop = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self.sock.bind((host, port))
        self.port = self.sock.getsockname()[1]  # port=0 verildiyse işletim sisteminin seçtiği
        self.stats = {"paket": 0, "kayip": 0, "sirasiz": 0, "yabanci": 0}
        self.ssrc: int | None = None
        self._last_ts: int | None = None
        self._frame_len: int | None = None
        self._resampler = None

    def close(self) -> None:
        self._stop.set()

    def _codec_of(self, pt: int) -> tuple[str, int]:
        if self.codec != "auto":
            rate = self.clock_rate or (8000 if self.codec in ("pcmu", "pcma") else 16000)
            return self.codec, rate
        if pt in STATIC_PT:
            name, rate = STATIC_PT[pt]
            return name, self.clock_rate or rate
        raise ValueError(f"dinamik yük türü {pt}: codec ve clock_rate verin ya da ffmpeg+SDP kullanın")

    def _to16k(self, x: np.ndarray, rate: int, last: bool = False) -> np.ndarray:
        if rate == SR:
            return x
        if self._resampler is None:
            self._resampler = soxr.ResampleStream(rate, SR, 1, dtype="float32")
        return self._resampler.resample_chunk(x, last=last)

    def frames(self) -> Iterator[np.ndarray]:
        self.sock.settimeout(0.2)
        waited, codec, rate = 0.0, None, None
        try:
            while not self._stop.is_set():
                try:
                    data = self.sock.recv(4096)
                except socket.timeout:
                    waited += 0.2
                    limit = self.idle_timeout if self.ssrc is not None else self.first_timeout
                    if waited >= limit:
                        break
                    continue
                p = parse(data)
                if p is None:
                    continue
                if self.ssrc is None:
                    self.ssrc = p.ssrc
                    codec, rate = self._codec_of(p.pt)
                    log.info("RTP akışı: ssrc=%08x yük=%d biçim=%s %d Hz", p.ssrc, p.pt, codec, rate)
                elif p.ssrc != self.ssrc:
                    self.stats["yabanci"] += 1
                    continue
                waited = 0.0
                self.stats["paket"] += 1
                for q in self.jitter.push(p):
                    out = self._emit(q, codec, rate)
                    if len(out):
                        yield out
            for q in self.jitter.drain():
                out = self._emit(q, codec, rate)
                if len(out):
                    yield out
            if rate is not None and rate != SR and self._resampler is not None:
                tail = self._to16k(np.zeros(0, np.float32), rate, last=True)
                if len(tail):
                    yield tail
        finally:
            self.sock.close()

    def _emit(self, p: RtpPacket | None, codec: str, rate: int) -> np.ndarray:
        if p is None:  # kayıp paket: bir paket boyu sessizlik
            self.stats["kayip"] += 1
            pcm = np.zeros(self._frame_len or rate // 50, np.float32)
        else:
            pcm = decode(p.payload, codec)
            self._frame_len = len(pcm)
        return self._to16k(pcm, rate)
