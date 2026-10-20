"""Dışarıdaki bir servisin canlı akışı nasıl kullanacağının örneği (ve elle deneme aracı).

    python scripts/dis_servis_ornek.py kayit.wav                        # HTTP parçaları + SSE
    python scripts/dis_servis_ornek.py kayit.wav --uc ws                # tek WebSocket bağlantısı
    python scripts/dis_servis_ornek.py kayit.wav --api http://sunucu:8000 --anahtar GIZLI --dil vi
    python scripts/dis_servis_ornek.py kayit.wav --hiz 0                # gerçek zamandan hızlı (yük denemesi)

Sesi parça parça (varsayılan 1 sn) yollar, metni aynı anda alır, parça başına gecikmeyi yazar.
Gönderen taraf kuralları (docs/entegrasyon.md "Canlı akış"):
  - ses: ham PCM16 little-endian, tek kanal; hızı akış açılırken `ses_hizi` ile bildirilir
  - HTTP: her parçaya artan `seq`; 429 gelirse Retry-After kadar bekleyip AYNI parçayı yeniden yolla
  - bitirirken son=true (WebSocket'te "son" metni)
"""
from __future__ import annotations

import argparse
import json
import threading
import time

import httpx
import numpy as np
import soundfile as sf
import soxr


def pcm_chunks(path: str, rate: int, seconds: float):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != rate:
        x = soxr.resample(x, sr, rate)
    n = int(rate * seconds)
    for i in range(0, len(x), n):
        yield (np.clip(x[i:i + n], -1, 1) * 32767).astype("<i2").tobytes()


def listen(api: str, sid: str, headers: dict, sent_at: dict, out: list, done: threading.Event) -> None:
    """SSE: kopan bağlantıda Last-Event-ID ile kaldığı yerden devam eder."""
    last = -1
    while not done.is_set():
        try:
            h = dict(headers, **({"Last-Event-ID": str(last)} if last >= 0 else {}))
            with httpx.stream("GET", f"{api}/streams/{sid}/events", headers=h, timeout=None) as r:
                ev = None
                for line in r.iter_lines():
                    if line.startswith("event:"):
                        ev = line[6:].strip()
                    elif line.startswith("data:"):
                        d = json.loads(line[5:])
                        if ev in ("parca", "duzeltme") and d.get("final"):
                            last = d["index"]
                            out.append(d)
                            tag = " [düzeltildi]" if d.get("duzeltildi") else ""
                            print(f"[{d['start']:6.1f}-{d['end']:6.1f}]{tag} {d['text']}")
                        elif ev == "bitti":
                            print("bitti:", d.get("durum"), d.get("ozet"))
                            done.set()
                            return
        except httpx.HTTPError as e:
            print("SSE koptu, yeniden bağlanılıyor:", e)
            time.sleep(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ses")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--anahtar", default="")
    ap.add_argument("--dil", default="vi")
    ap.add_argument("--uc", choices=["http", "ws"], default="http")
    ap.add_argument("--hiz", type=float, default=1.0, help="1 = gerçek zaman, 0 = beklemeden")
    ap.add_argument("--parca", type=float, default=1.0, help="parça süresi, sn")
    ap.add_argument("--ses-hizi", type=int, default=16000)
    ap.add_argument("--canli-url", default=None, help="parçaların POST edileceği adres (imzalı)")
    a = ap.parse_args()
    headers = {"X-API-Key": a.anahtar} if a.anahtar else {}

    r = httpx.post(f"{a.api}/streams", headers=headers, timeout=10, json={
        "kaynak": "push", "dil": a.dil, "ses_hizi": a.ses_hizi, "son_analiz": True,
        "idle_timeout": 30, **({"canli_url": a.canli_url} if a.canli_url else {})})
    r.raise_for_status()
    sid = r.json()["id"]
    print("akış:", sid)
    t0, out, done = time.monotonic(), [], threading.Event()

    if a.uc == "ws":
        from websockets.sync.client import connect

        url = a.api.replace("http", "ws", 1) + f"/streams/{sid}/audio" + (f"?key={a.anahtar}" if a.anahtar else "")
        with connect(url, max_size=None) as ws:
            def rx():
                for m in ws:
                    d = json.loads(m)
                    if d.get("tur") in ("parca", "duzeltme") and d.get("final"):
                        out.append(d)
                        tag = " [düzeltildi]" if d.get("duzeltildi") else ""
                        print(f"[{d['start']:6.1f}-{d['end']:6.1f}]{tag} {d['text']}")
                    elif d["tur"] == "bitti":
                        print("bitti:", d.get("durum"), d.get("ozet"))
                        return
            th = threading.Thread(target=rx)
            th.start()
            for i, c in enumerate(pcm_chunks(a.ses, a.ses_hizi, a.parca)):
                ws.send(c)
                if a.hiz:
                    time.sleep(a.parca / a.hiz)
            ws.send("son")
            th.join()
    else:
        sent: dict = {}
        th = threading.Thread(target=listen, args=(a.api, sid, headers, sent, out, done))
        th.start()
        chunks = list(pcm_chunks(a.ses, a.ses_hizi, a.parca))
        for i, c in enumerate(chunks):
            while True:
                p = httpx.post(f"{a.api}/streams/{sid}/audio", params={"seq": i}, content=c, headers=headers, timeout=15)
                if p.status_code == 429:                       # sunucu yetişemiyor: bekle, aynı parçayı yolla
                    time.sleep(float(p.headers.get("Retry-After", 1)))
                    continue
                p.raise_for_status()
                break
            if a.hiz:
                time.sleep(a.parca / a.hiz)
        httpx.post(f"{a.api}/streams/{sid}/audio", params={"son": "true"}, content=b"", headers=headers, timeout=15)
        th.join()
    print(f"{len(out)} parça, toplam {time.monotonic() - t0:.1f} sn")


if __name__ == "__main__":
    main()
