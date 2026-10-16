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
                 extra_input_args: list[str] | None = None, idle_timeout: float | None = None,
                 cleanup: str | None = None):
        """idle_timeout: RTP/UDP'de bu kadar sn paket gelmezse akış biter (görüşme bitti).
        cleanup: akış bitince silinecek dosya (geçici SDP)."""
        self.src, self.realtime, self.chunk = src, realtime, int(chunk_sec * SR) * 2
        self.extra = extra_input_args or []
        self.idle_timeout, self.cleanup = idle_timeout, cleanup
        self.first_timeout = 60.0
        self.proc: subprocess.Popen | None = None
        self._stop = threading.Event()

    def command(self) -> list[str]:
        exe = ffmpeg_bin()
        if exe is None:
            raise RuntimeError("ffmpeg bulunamadı: bu kaynak türü için ffmpeg kurulu olmalı")
        args = [exe, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if self.src.lower().endswith(".sdp"):
            args += ["-protocol_whitelist", "file,udp,rtp", "-reorder_queue_size", "64"]
            # biçim SDP'de yazılı: ffmpeg'in akışı 5 sn dinleyip biçim araması (gecikme) gereksiz
            args += ["-fflags", "nobuffer", "-probesize", "32", "-analyzeduration", "0"]
        if self.realtime:
            args += ["-re"]
        return args + self.extra + ["-i", self.src, "-vn", "-ac", "1", "-ar", str(SR),
                                    "-f", "s16le", "-"]

    def close(self) -> None:
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            self.proc.kill()  # UDP'de bekleyen ffmpeg SIGTERM'e geç yanıt veriyor; yazılacak bir şey yok

    def _chunks(self) -> Iterator[bytes]:
        """stdout'tan ham baytlar. idle_timeout verilmişse okuma ayrı iş parçacığında yapılır ve
        o kadar sn veri gelmezse akış biter (RTP'de görüşme bitince ffmpeg kendiliğinden çıkmaz).
        İlk veri için first_timeout (60 sn) beklenir."""
        read = self.proc.stdout.read1 if hasattr(self.proc.stdout, "read1") else self.proc.stdout.read
        if not self.idle_timeout:
            while not self._stop.is_set():
                data = read(self.chunk)
                if not data:
                    return
                yield data
            return
        import queue

        q: queue.Queue = queue.Queue(maxsize=256)

        def pump():
            while True:
                data = read(self.chunk)
                q.put(data)
                if not data:
                    return

        threading.Thread(target=pump, daemon=True, name="ffmpeg-oku").start()
        got_any = False
        while not self._stop.is_set():
            try:
                data = q.get(timeout=self.idle_timeout if got_any else self.first_timeout)
            except queue.Empty:
                log.info("ffmpeg kaynağı: %.0f sn veri yok, akış bitti", self.idle_timeout)
                return
            if not data:
                return
            got_any = True
            yield data

    def frames(self) -> Iterator[np.ndarray]:
        self.proc = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        rest = b""
        try:
            for data in self._chunks():
                data, rest = rest + data, b""
                if len(data) % 2:
                    data, rest = data[:-1], data[-1:]
                yield np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        finally:
            self.close()
            if self.proc:
                try:  # stdout'u okuma iş parçacığı tutuyor olabilir: communicate değil wait
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
                err = self.proc.stderr.read() if self.proc.stderr else b""
                if self.proc.returncode not in (0, None, -15, 255) and err:
                    log.warning("ffmpeg: %s", err.decode(errors="replace").strip()[-500:])
            if self.cleanup:
                import os

                try:
                    os.unlink(self.cleanup)
                except OSError:
                    pass
