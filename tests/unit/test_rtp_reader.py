"""Alıcı, tüketici yavaşken de soketi boşaltır (ASR çalışırken paket düşmesin)."""
import socket
import struct
import time

from viet_asr.infrastructure.rtp import RtpSource


def _pkt(seq: int) -> bytes:
    return struct.pack("!BBHII", 0x80, 0, seq, seq * 160, 0xABCD) + b"\xff" * 160  # PCMU, 20 ms


def test_yavas_tuketici_paket_kaybettirmez():
    src = RtpSource("127.0.0.1", 0, "pcmu", 8000, idle_timeout=0.6)
    gonder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    frames = src.frames()
    # okuyucu iş parçacığı ilk next() ile başlar; titreşim tamponu dolana dek ilk paketler gönderilir
    for i in range(12):
        gonder.sendto(_pkt(i), ("127.0.0.1", src.port))
        time.sleep(0.002)
    toplam = len(next(frames))
    src.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)  # artık çok küçük çekirdek tamponu
    n = 400
    for i in range(12, n):
        gonder.sendto(_pkt(i), ("127.0.0.1", src.port))
        time.sleep(0.001)
    time.sleep(0.5)  # tüketici (ASR gibi) bu sırada hiç okumuyor
    for f in frames:
        toplam += len(f)
    assert src.stats["kayip"] == 0
    assert src.stats["paket"] == n
    assert toplam == n * 320  # 20 ms @16 kHz = 320 örnek
