"""Çalışan sisteme uçtan uca duman testi: dosyaları API'ye yollar, webhook'ları dinler.

    python scripts/smoke_test.py --api http://localhost:8000 kayit1.wav kayit2.flac
    python scripts/smoke_test.py --api http://localhost:8000 --lang tr kayit.wav

Webhook'u almak için bu betik küçük bir HTTP sunucusu açar (--hook-port, varsayılan 9009).
API Docker içindeyse --hook-host olarak API'nin bu makineye ulaştığı adresi verin
(Docker Desktop'ta host.docker.internal).
Her iş için: durum, dil, süre ve ilk satırlar yazılır; webhook gelmezse hata kodu 1.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx

received: dict[str, dict] = {}


class Hook(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        received[self.headers["X-Job-Id"]] = {"event": self.headers["X-Event"], **body}
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--lang", default="auto")
    p.add_argument("--hook-host", default="localhost")
    p.add_argument("--hook-port", type=int, default=9009)
    p.add_argument("--timeout", type=float, default=600)
    a = p.parse_args()

    srv = HTTPServer(("0.0.0.0", a.hook_port), Hook)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    hook = f"http://{a.hook_host}:{a.hook_port}/hook"

    jobs = {}
    with httpx.Client(base_url=a.api, timeout=60) as c:
        print("health:", c.get("/health").json())
        t0 = time.time()
        for f in a.files:
            with open(f, "rb") as fh:
                r = c.post("/jobs", files={"file": (Path(f).name, fh)},
                           data={"language": a.lang, "callback_url": hook})
            print(f"{Path(f).name}: HTTP {r.status_code} {r.json()}")
            if r.status_code == 202:
                jobs[r.json()["job_id"]] = f
        while len(received) < len(jobs) and time.time() - t0 < a.timeout:
            time.sleep(0.5)
        dt = time.time() - t0
        for jid, f in jobs.items():
            body = c.get(f"/jobs/{jid}").json()
            hook_ev = received.get(jid, {}).get("event", "GELMEDİ")
            lang = body.get("language") or {}
            print(f"\n{Path(f).name}: durum={body['status']} webhook={hook_ev} "
                  f"dil={lang.get('code')}({lang.get('source')}, {lang.get('confidence')}) "
                  f"hata={body.get('error')}")
            for u in (body.get("result") or {}).get("utterances", [])[:4]:
                print(f"   [{u['start']:6.2f}-{u['end']:6.2f}] {u['speaker']}: {u['text']}")
    print(f"\n{len(received)}/{len(jobs)} webhook, toplam {dt:.1f} sn")
    sys.exit(0 if len(received) == len(jobs) else 1)


if __name__ == "__main__":
    main()
