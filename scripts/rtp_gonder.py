"""Deneme için: bir ses dosyasını RTP olarak, gerçek zaman hızında gönderir (santral yerine).

    python scripts/rtp_gonder.py kayit.wav --hedef 127.0.0.1:5004               # PCMU 8 kHz
    python scripts/rtp_gonder.py kayit.wav --hedef 127.0.0.1:5004 --codec l16   # 16 kHz ham
    python scripts/rtp_gonder.py kayit.wav --kayip 0.02 --karistir 0.05          # ağ bozukluğu

Gerçek santral yerine kullanılır; ffmpeg ile de gönderilebilir:
    ffmpeg -re -i kayit.wav -ar 8000 -ac 1 -c:a pcm_mulaw -f rtp rtp://127.0.0.1:5004
"""
from __future__ import annotations

import argparse
import random
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def ulaw_encode(x: np.ndarray) -> bytes:
    s = np.clip(x * 32768.0, -32768, 32767).astype(np.int32)
    sign = (s < 0).astype(np.int32) << 7
    mag = np.minimum(np.abs(s), 32635) + 0x84
    exp = np.floor(np.log2(mag)).astype(np.int32) - 7
    exp = np.clip(exp, 0, 7)
    mant = (mag >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | mant) & 0xFF).astype(np.uint8).tobytes()


def packets(audio: np.ndarray, codec: str, ptime: float = 0.02):
    rate = 8000 if codec == "pcmu" else 16000
    pt = 0 if codec == "pcmu" else 96
    x = soxr.resample(audio, 16000, rate) if rate != 16000 else audio
    n = int(rate * ptime)
    for i, k in enumerate(range(0, len(x), n)):
        chunk = x[k:k + n]
        payload = ulaw_encode(chunk) if codec == "pcmu" else \
            (np.clip(chunk, -1, 1) * 32767).astype(">i2").tobytes()
        hdr = struct.pack("!BBHII", 0x80, pt | (0x80 if i == 0 else 0), i & 0xFFFF, (k) & 0xFFFFFFFF,
                          0x1234ABCD)
        yield hdr + payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dosya")
    ap.add_argument("--hedef", default="127.0.0.1:5004")
    ap.add_argument("--codec", choices=["pcmu", "l16"], default="pcmu")
    ap.add_argument("--hiz", type=float, default=1.0, help="1 = gerçek zaman, 0 = beklemeden")
    ap.add_argument("--kayip", type=float, default=0.0, help="paket kaybı oranı (0-1)")
    ap.add_argument("--karistir", type=float, default=0.0, help="sırası bozulan paket oranı")
    ap.add_argument("--tohum", type=int, default=0)
    a = ap.parse_args(argv)
    audio, sr = sf.read(a.dosya, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        audio = soxr.resample(audio, sr, 16000)
    host, port = a.hedef.rsplit(":", 1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rng = random.Random(a.tohum)
    t0, held, sent = time.monotonic(), None, 0
    for i, pkt in enumerate(packets(audio, a.codec)):
        if a.hiz > 0:
            delay = t0 + i * 0.02 / a.hiz - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        if rng.random() < a.kayip:
            continue
        if held is None and rng.random() < a.karistir:
            held = pkt  # bir sonrakinden sonra gönder: sıra bozulur
            continue
        sock.sendto(pkt, (host, int(port)))
        sent += 1
        if held is not None:
            sock.sendto(held, (host, int(port)))
            held, sent = None, sent + 1
    print(f"{sent} paket gönderildi ({len(audio) / 16000:.1f} sn ses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
