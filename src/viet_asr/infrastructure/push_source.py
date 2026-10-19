"""Dışarıdan parça parça gelen ses (HTTP ya da WebSocket ile itilen): AudioStream kaynağı.

Karşıdaki servis RTP kuramıyorsa sesi kendisi parçalara bölüp yollar (ham PCM16, little-endian,
tek kanal; hız `ses_hizi`). Parçalar bir kuyrukta birikir, `frames()` 16 kHz float32 olarak verir.

- Geri basınç: kuyrukta `max_backlog` saniyeden fazla ses birikirse `BacklogFull` (HTTP 429);
  gönderen yavaşlar. Sınırsız büyütmek, modelin yetişemediği yayında belleği doldururdu.
- Tekrar gönderim: `seq` verilirse aynı parça iki kez gelirse atlanır, atlanan parça varsa
  hata verilir (ses sırası bozulmasın). Sıra numarası vermeyen WebSocket'te TCP sırayı korur.
- Bitiş: `end()` ya da `idle_timeout` sn boyunca hiç ses gelmemesi (ilk ses için first_timeout).
"""
from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

import numpy as np
import soxr

SR = 16000


class BacklogFull(Exception):
    def __init__(self, backlog: float):
        super().__init__(f"{backlog:.1f} sn ses işlenmeyi bekliyor")
        self.backlog = backlog


class SeqError(ValueError):
    def __init__(self, expected: int):
        super().__init__(f"beklenen sıra no {expected}")
        self.expected = expected


class PushSource:
    def __init__(self, rate: int = SR, idle_timeout: float = 30.0, first_timeout: float = 120.0,
                 max_backlog: float = 30.0):
        self.rate, self.idle_timeout, self.first_timeout = rate, idle_timeout, first_timeout
        self.max_backlog = max_backlog
        self.port = None
        self.stats = {"parca": 0, "tekrar": 0}
        self._q: queue.Queue[np.ndarray | None] = queue.Queue()
        self._lock = threading.Lock()
        self._backlog = 0.0       # kuyruktaki ses, sn
        self._received = 0.0      # toplam gelen ses, sn
        self._odd = b""           # çift olmayan bayt (örnek ortasında bölünmüş parça)
        self._next_seq = 0
        self._ended = False
        self._stop = threading.Event()
        self._resampler = soxr.ResampleStream(rate, SR, 1, dtype="float32") if rate != SR else None

    @property
    def backlog(self) -> float:
        return self._backlog

    @property
    def received(self) -> float:
        return self._received

    @property
    def next_seq(self) -> int:
        return self._next_seq

    def feed(self, data: bytes, seq: int | None = None) -> bool:
        """Parçayı kuyruğa koyar. False: aynı sıra no daha önce alınmıştı (atlandı)."""
        with self._lock:
            if self._ended or self._stop.is_set():
                raise ValueError("akış kapandı")
            if seq is not None:
                if seq < self._next_seq:
                    self.stats["tekrar"] += 1
                    return False
                if seq > self._next_seq:
                    raise SeqError(self._next_seq)
            if self._backlog > self.max_backlog:
                raise BacklogFull(self._backlog)
            if seq is not None:
                self._next_seq = seq + 1
            data = self._odd + data
            n = len(data) // 2 * 2
            self._odd = data[n:]
            self.stats["parca"] += 1
            if not n:
                return True
            x = np.frombuffer(data[:n], dtype="<i2").astype(np.float32) / 32768.0
            secs = len(x) / self.rate
            self._backlog += secs
            self._received += secs
            self._q.put(x)
            return True

    def end(self) -> None:
        with self._lock:
            if not self._ended:
                self._ended = True
                self._q.put(None)

    def close(self) -> None:
        self._stop.set()
        self._q.put(None)

    def frames(self) -> Iterator[np.ndarray]:
        waited, got = 0.0, False
        while not self._stop.is_set():
            try:
                x = self._q.get(timeout=0.2)
            except queue.Empty:
                waited += 0.2
                if waited >= (self.idle_timeout if got else self.first_timeout):
                    break
                continue
            if x is None:
                break
            waited, got = 0.0, True
            with self._lock:
                self._backlog = max(0.0, self._backlog - len(x) / self.rate)
            out = self._resampler.resample_chunk(x) if self._resampler else x
            if len(out):
                yield out
        if self._resampler is not None:
            tail = self._resampler.resample_chunk(np.zeros(0, np.float32), last=True)
            if len(tail):
                yield tail
