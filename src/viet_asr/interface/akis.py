"""Komut satırından canlı akış: kaynağı dinler, metni anında satır satır yazar.

    python -m viet_asr.interface.akis --kaynak rtp://0.0.0.0:5004 --dil vi
    python -m viet_asr.interface.akis --kaynak cagri.sdp --dil vi --json
    python -m viet_asr.interface.akis --kaynak http://radyo/akim.ogg --kayit akim.wav
    python -m viet_asr.interface.akis --kaynak rtp://0.0.0.0:5004 --api http://127.0.0.1:8001

--api verilirse akış bitince kaydedilen ses o sunucuya normal iş olarak gönderilir (konuşmacı
ayırma, temiz transkript). Tam modda (API ayrı süreç, modeller işçilerde) canlı akış bu
komutla çalıştırılır; yerel modda POST /streams ile sunucu içinden de açılabilir.
Durdurmak: Ctrl+C (o ana kadarki ses yine kaydedilir/gönderilir).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from pathlib import Path

import soundfile as sf

from viet_asr.application.stream import StreamTranscriber
from viet_asr.infrastructure.sherpa_vad import SileroStreamingVad
from viet_asr.interface.composition import build_engines, config_from_settings
from viet_asr.interface.settings import Settings
from viet_asr.interface.streams import FAST_ENGINES, StreamError, StreamSpec, parse_source, segment_dict


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kaynak", required=True)
    ap.add_argument("--dil", default="vi")
    ap.add_argument("--codec", default="auto", choices=["auto", "pcmu", "pcma", "l16"])
    ap.add_argument("--clock-rate", type=int, default=None)
    ap.add_argument("--bekleme", type=float, default=5.0, help="bu kadar sn paket gelmezse akış biter")
    ap.add_argument("--kayit", default=None, help="sesi bu WAV dosyasına da yaz")
    ap.add_argument("--api", default=None, help="bitince tam analiz için sunucu (ör. http://127.0.0.1:8001)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--json", action="store_true", help="her olayı JSON satırı olarak yaz")
    ap.add_argument("--ara", action="store_true", help="ara sonuçları da yaz (Zipformer)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    s = Settings()
    cfg = config_from_settings(s)
    if a.dil not in cfg.languages:
        print(f"dil '{a.dil}' yok: {', '.join(cfg.languages)}", file=sys.stderr)
        return 2
    spec = StreamSpec(a.kaynak, a.dil, a.codec, a.clock_rate, idle_timeout=a.bekleme)
    try:
        source = parse_source(spec, allow_files=True)
    except StreamError as e:
        print(e, file=sys.stderr)
        return 2
    if getattr(source, "port", None):
        print(f"dinleniyor: UDP {source.port}", file=sys.stderr, flush=True)
    asr = build_engines(cfg)[a.dil]
    kayit = a.kayit or (str(Path(s.data_dir) / "akislar" / "son_akis.wav") if a.api else None)
    wav = None
    if kayit:
        Path(kayit).parent.mkdir(parents=True, exist_ok=True)
        wav = sf.SoundFile(kayit, "w", samplerate=16000, channels=1, subtype="PCM_16")

    def on_segment(seg):
        if not seg.final and not a.ara:
            return
        if a.json:
            print(json.dumps(segment_dict(seg), ensure_ascii=False), flush=True)
        else:
            m, sec = divmod(int(seg.start), 60)
            print(f"[{m:02d}:{sec:02d}]{'' if seg.final else ' …'} {seg.text}", flush=True)

    stop = threading.Event()
    st = StreamTranscriber(lambda max_speech=None: SileroStreamingVad(
        cfg.vad, max_speech=max_speech or s.stream_max_speech))
    fast = any(k in getattr(asr, "name", "") for k in FAST_ENGINES)
    try:
        sm = st.run(source, asr, on_segment, record=wav.write if wav else None, interim=fast and a.ara,
                    stop=stop)
    except KeyboardInterrupt:
        stop.set()
        source.close()
        sm = None
    finally:
        if wav:
            wav.close()
    if sm:
        print(f"bitti: {sm.duration:.1f} sn, {sm.segments} parça, gecikme p95 {sm.p95_latency:.2f} sn",
              file=sys.stderr)
    if a.api and kayit:
        import httpx

        with open(kayit, "rb") as f:
            r = httpx.post(f"{a.api.rstrip('/')}/jobs", files={"file": ("akis.wav", f, "audio/wav")},
                           data={"language": a.dil},
                           headers={"X-API-Key": a.api_key} if a.api_key else {}, timeout=120)
        print(f"tam analiz: {r.status_code} {r.text}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
