"""ffmpeg ile her türlü canlı/kayıtlı kaynaktan 16 kHz mono akış (AudioStream portu).

ffmpeg'in okuyabildiği her şey: RTP + SDP dosyası (Opus, G.722, AAC...), rtp://ip:port,
http(s) üzerinden OGG/MP3/WAV yayını (Icecast vb.), srt://, udp://, yerel dosya.
Paket sıralama, kayıp ve ses çözme işini ffmpeg yapar; biz stdout'tan ham 16 bit ses okuruz.

    FfmpegSource("cagri.sdp")                    # RTP, biçim SDP'de
    FfmpegSource("http://radyo/akim.ogg")
    FfmpegSource("kayit.wav", realtime=True)     # dosyayı gerçek zaman hızında oynat (deneme)
"""
from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Iterator

import numpy as np

from viet_asr.infrastructure.audio import ffmpeg_bin

log = logging.getLogger(__name__)
SR = 16000


class FfmpegSource:
    def __init__(self, src: str, realtime: bool = False, chunk_sec: float = 0.1,
                 extra_input_args: list[str] | None = None):
        self.src, self.realtime, self.chunk = src, realtime, int(chunk_sec * SR) * 2
        self.extra = extra_input_args or []
        self.proc: subprocess.Popen | None = None
        self._stop = threading.Event()

    def command(self) -> list[str]:
        exe = ffmpeg_bin()
        if exe is None:
            raise RuntimeError("ffmpeg bulunamadı: bu kaynak türü için ffmpeg kurulu olmalı")
        args = [exe, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if self.src.lower().endswith(".sdp"):
            args += ["-protocol_whitelist", "file,udp,rtp", "-reorder_queue_size", "64"]
        if self.realtime:
            args += ["-re"]
        return args + self.extra + ["-i", self.src, "-vn", "-ac", "1", "-ar", str(SR),
                                    "-f", "s16le", "-"]

    def close(self) -> None:
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def frames(self) -> Iterator[np.ndarray]:
        self.proc = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        rest = b""
        try:
            while not self._stop.is_set():
                data = self.proc.stdout.read1(self.chunk) if hasattr(self.proc.stdout, "read1") \
                    else self.proc.stdout.read(self.chunk)
                if not data:
                    break
                data, rest = rest + data, b""
                if len(data) % 2:
                    data, rest = data[:-1], data[-1:]
                yield np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        finally:
            self.close()
            if self.proc:
                try:
                    _, err = self.proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    err = b""
                if self.proc.returncode not in (0, None, -15, 255) and err:
                    log.warning("ffmpeg: %s", err.decode(errors="replace").strip()[-500:])
