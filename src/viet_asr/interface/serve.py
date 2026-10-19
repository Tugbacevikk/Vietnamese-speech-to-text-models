"""Sunucuyu başlatma (yerel ve tam mod ortak): HTTPS ve açılış adresleri.

HTTPS=1 ise TLS_CERT/TLS_KEY, yoksa TLS_DIR'de kendinden imzalı sertifika kullanılır.
HOST=0.0.0.0 ise bu bilgisayarın tüm ağ adresleri yazılır: telefondan/başka bilgisayardan
açılacak adres bunlardan biri.
"""
from __future__ import annotations

import logging

import uvicorn

log = logging.getLogger(__name__)
_ANY = ("0.0.0.0", "::", "")


def tls_files(settings) -> tuple[str, str] | None:
    if not settings.https:
        return None
    if settings.tls_cert and settings.tls_key:
        return settings.tls_cert, settings.tls_key
    from viet_asr.infrastructure.tls import ensure_self_signed

    return ensure_self_signed(settings.tls_dir)


def addresses(settings, host: str, port: int) -> list[str]:
    from viet_asr.infrastructure.tls import local_addresses

    scheme = "https" if settings.https else "http"
    hosts = local_addresses() if host in _ANY else [host]
    return [f"{scheme}://{h}:{port}" for h in hosts]


def announce(settings, host: str, port: int) -> None:
    for url in addresses(settings, host, port):
        log.info("adres: %s   (canlı altyazı: %s/canli)", url, url)
    if host in ("127.0.0.1", "localhost"):
        log.info("başka cihazdan (telefon, diğer bilgisayar) açmak için .env: HOST=0.0.0.0 ve HTTPS=1")
    elif not settings.https:
        log.warning("HOST=%s ama HTTPS kapalı: başka cihazda sayfa açılır, mikrofon açılmaz "
                    "(tarayıcılar mikrofonu yalnızca https'de verir). .env: HTTPS=1", host)
    if host not in ("127.0.0.1", "localhost") and not settings.auth_enabled:
        log.warning("Sunucu ağa açık ve API_KEY yok: aynı ağdaki herkes işleri görebilir, silebilir, "
                    "canlı altyazı açabilir. .env: API_KEY=<uzun rastgele bir değer> (sayfada Ayarlar > "
                    "API anahtarı alanına da girilir)")
    if settings.https:
        log.info("HTTPS: kendinden imzalı sertifika; tarayıcıdaki uyarıda 'Gelişmiş > devam et' (bir kez)")


def run(app, settings, host: str, port: int) -> None:
    files = tls_files(settings)
    kw = {"ssl_certfile": files[0], "ssl_keyfile": files[1]} if files else {}
    announce(settings, host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning", **kw)
