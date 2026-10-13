"""Outbox göndericisi: bekleyen bildirimleri müşterinin callback adresine POST eder.

- 2xx yanıt: gönderildi.
- Diğer her şey (hata kodu, zaman aşımı, bağlantı yok): artan aralıkla tekrar
  (10 sn, 20 sn, 40 sn ... en fazla 1 saat), max_attempts denemeden sonra bırakılır;
  satır outbox'ta last_error ile kalır, elle incelenebilir.
- Gövde: iş sonucu JSON'u. Başlıklar: X-Event (job.done / job.failed / job.rejected),
  X-Job-Id, X-Delivery-Id (aynı bildirim tekrar gelirse alıcı bununla ayıklar).
- İmza (WEBHOOK_SECRET tanımlıysa): X-Timestamp ve
  X-Signature: sha256=HMAC_SHA256(secret, f"{timestamp}.{gövde}")
  Alıcı aynı hesabı yapıp karşılaştırır; ayrıca zaman damgası çok eskiyse
  (ör. 5 dk) reddederek eski isteğin tekrar oynatılmasını engeller.
  (fine-tuning reposundaki WebhookService ile aynı biçim.)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from datetime import timedelta

import httpx

log = logging.getLogger(__name__)


def sign(secret: str, timestamp: str, body: bytes) -> str:
    mac = hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256)
    return "sha256=" + mac.hexdigest()


def verify_signature(secret: str, timestamp: str, body: bytes, signature: str,
                     max_age_s: int = 300, now: float | None = None) -> bool:
    """Alıcı tarafı için örnek doğrulama (testlerde de kullanılır)."""
    if abs((now or time.time()) - int(timestamp)) > max_age_s:
        return False
    return hmac.compare_digest(sign(secret, timestamp, body), signature)


def backoff(attempts: int, base: float = 10.0, cap: float = 3600.0) -> timedelta:
    return timedelta(seconds=min(cap, base * (2 ** attempts)))


class OutboxDispatcher:
    """store: claim_outbox / mark_sent / mark_failed sunan depo (PostgreSQL ya da SQLite)."""

    def __init__(self, store, client: httpx.Client | None = None,
                 max_attempts: int = 8, batch: int = 20, timeout: float = 10.0,
                 lease: timedelta = timedelta(seconds=60), secret: str = ""):
        self.store, self.secret = store, secret
        self.client = client or httpx.Client(timeout=timeout)
        self.max_attempts, self.batch, self.lease = max_attempts, batch, lease
        self._stop = False

    def run_once(self) -> int:
        """Bir tur: en fazla `batch` bildirim gönderir. Gönderilen sayısını döndürür."""
        sent = 0
        for row in self.store.claim_outbox(self.batch, self.max_attempts, self.lease):
            try:
                body = json.dumps(row["payload"], ensure_ascii=False,
                                  separators=(",", ":")).encode("utf-8")
                headers = {"Content-Type": "application/json", "X-Event": row["event"],
                           "X-Job-Id": row["job_id"], "X-Delivery-Id": str(row["id"])}
                if self.secret:
                    ts = str(int(time.time()))
                    headers["X-Timestamp"] = ts
                    headers["X-Signature"] = sign(self.secret, ts, body)
                r = self.client.post(row["url"], content=body, headers=headers)
                if 200 <= r.status_code < 300:
                    self.store.mark_sent(row["id"])
                    sent += 1
                    continue
                err = f"HTTP {r.status_code}"
            except httpx.HTTPError as e:
                err = f"{type(e).__name__}: {e}"
            wait = backoff(row["attempts"])
            self.store.mark_failed(row["id"], err, wait)
            log.warning("webhook başarısız job=%s deneme=%d %s; %ss sonra tekrar",
                        row["job_id"], row["attempts"] + 1, err, int(wait.total_seconds()))
        return sent

    def stop(self) -> None:
        self._stop = True

    def run_forever(self, interval: float = 2.0) -> None:
        log.info("outbox göndericisi çalışıyor (her %.0f sn)", interval)
        while not self._stop:
            try:
                n = self.run_once()
            except Exception:  # veritabanı geçici olarak yoksa çökme, bekle
                log.exception("outbox turu başarısız")
                n = 0
            if n < self.batch:
                time.sleep(interval)
