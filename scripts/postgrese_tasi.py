"""Yerel moddaki SQLite verisini PostgreSQL'e taşır: işler, konuşma satırları (düzeltmeler ve
silinenler dahil), bekleyen/gönderilmiş bildirimler. Ses dosyalarına dokunmaz (yerlerinde kalır).

    python scripts/postgrese_tasi.py                 # hedef .env'deki LOCAL_DATABASE_URL
    python scripts/postgrese_tasi.py --hedef postgresql://asr:asr@localhost:5432/asr
    python scripts/postgrese_tasi.py --dene          # yalnızca sayar, hiçbir şey yazmaz

Sıra (postgrese_gec.bat bunu yapar):
  1. PostgreSQL kurulu ve veritabanı oluşturulmuş olsun (README: "PostgreSQL'e geçiş").
  2. .env: LOCAL_DATABASE_URL=postgresql://... (yerel mod, baslat.bat; tam modda DATABASE_URL).
  3. Sunucuyu durdurun (durdur.bat): taşıma sırasında SQLite'a yeni iş yazılmasın.
  4. Bu betik (tekrar çalıştırılabilir: hedefte olan iş atlanır).
  5. baslat.bat. Eski SQLite dosyası (DATA_DIR/jobs.db) yedek olarak kalır, silinmez.

Kaynak varsayılanı: DATA_DIR/jobs.db. Hedef varsayılanı: LOCAL_DATABASE_URL PostgreSQL ise o,
değilse DATABASE_URL.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from viet_asr.infrastructure.db.transfer import transfer  # noqa: E402
from viet_asr.interface.settings import Settings  # noqa: E402

NAMES = {"jobs": "iş", "utterances": "konuşma satırı", "outbox": "bildirim", "audit_log": "erişim kaydı"}


def main(argv: list[str] | None = None) -> int:
    s = Settings()
    default_src = f"sqlite:///{Path(s.data_dir) / 'jobs.db'}"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kaynak", default=default_src, help=f"varsayılan {default_src}")
    target = s.local_database_url if s.local_database_url.startswith("postgresql") else s.database_url
    ap.add_argument("--hedef", default=target, help="varsayılan LOCAL_DATABASE_URL ya da DATABASE_URL (.env)")
    ap.add_argument("--dene", action="store_true", help="yazmadan say")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if a.kaynak.startswith("sqlite:///"):
        path = Path(a.kaynak.removeprefix("sqlite:///"))
        if not path.is_file():
            print(f"Kaynak yok: {path}  (--kaynak ile verin)")
            return 1
    if not a.hedef.startswith("postgresql"):
        print(f"Hedef PostgreSQL değil: {a.hedef}\n.env'de LOCAL_DATABASE_URL=postgresql://kullanici:parola@localhost:5432/asr "
              "yazın ya da --hedef verin.")
        return 1
    try:
        rep = transfer(a.kaynak, a.hedef, dry_run=a.dene)
    except Exception as e:  # bağlantı / izin: ne yapılacağı söylensin
        print(f"Taşınamadı: {type(e).__name__}: {e}")
        if "connect" in str(e).lower() or "connection" in str(e).lower():
            print("PostgreSQL çalışıyor mu, adres/kullanıcı/parola doğru mu, veritabanı oluşturuldu mu?")
        return 1

    for t, n in rep.source.items():
        if a.dene:
            print(f"  {NAMES[t]:15s} {n} (taşınacak)")
        else:
            print(f"  {NAMES[t]:15s} kaynakta {n}, kopyalanan {rep.copied[t]}, zaten vardı "
                  f"{rep.skipped[t]}, hedefte şimdi {rep.target[t]}")
    if a.dene:
        return 0
    if not rep.complete:
        print("UYARI: hedefte kaynaktan az satır var; çıktıyı kontrol edin.")
        return 2
    print("Tamam. Şimdi .env'de LOCAL_DATABASE_URL (yerel mod) ya da DATABASE_URL (tam mod) "
          "bu adrese ayarlanıp sunucu başlatılabilir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
