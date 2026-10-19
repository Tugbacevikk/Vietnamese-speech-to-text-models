"""Veritabanı yedeği: al, listele, geri yükle. Sunucu açıkken bakım bunu günde bir kendisi yapar
(BACKUP_EVERY_HOURS); bu betik elle yedek ve geri yükleme içindir.

    python scripts/yedekle.py                      # şimdi yedek al
    python scripts/yedekle.py listele
    python scripts/yedekle.py geri-yukle data\\yedek\\yedek_2026-10-19_030000.dump

Hangi veritabanı: LOCAL_DATABASE_URL (yerel mod) ya da --tam ile DATABASE_URL (tam mod).
Yedekler BACKUP_DIR'de (varsayılan data/yedek), BACKUP_KEEP_DAYS günden eskiler silinir
(en yeni 3 yedek her zaman kalır).

Geri yüklemeden önce sunucuyu durdurun (durdur.bat). Geri yükleme, veritabanındaki tabloları
yedektekilerle DEĞİŞTİRİR: yedekten sonra gelen işler gider. Bu yüzden önce o anki hâlin de
yedeği alınır.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from viet_asr.infrastructure.backup import BackupError, DatabaseBackup  # noqa: E402
from viet_asr.infrastructure.db.url import normalize_url  # noqa: E402
from viet_asr.interface.settings import Settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("komut", nargs="?", default="al", choices=["al", "listele", "geri-yukle"])
    ap.add_argument("dosya", nargs="?", help="geri-yukle için yedek dosyası")
    ap.add_argument("--tam", action="store_true", help="tam modun veritabanı (DATABASE_URL)")
    ap.add_argument("--evet", action="store_true", help="geri yüklemede onay sorma")
    a = ap.parse_args(argv)
    s = Settings()
    url = normalize_url(s.database_url if a.tam else s.local_db)
    b = DatabaseBackup(url, s.backup_dir, keep_days=s.backup_keep_days,
                       docker_container=s.backup_docker_container, pg_dump=s.pg_dump)
    try:
        if a.komut == "al":
            p = b.run()
            print(f"Yedek alındı ve doğrulandı: {p} ({p.stat().st_size / 1e6:.1f} MB)")
        elif a.komut == "listele":
            items = b.backups()
            if not items:
                print(f"Yedek yok ({s.backup_dir})")
            for p in items:
                when = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                print(f"  {when}  {p.stat().st_size / 1e6:7.1f} MB  {p}")
        else:
            if not a.dosya:
                print("Hangi yedek? python scripts/yedekle.py listele")
                return 1
            if not a.evet:
                ans = input(f"{a.dosya} geri yüklenecek; veritabanındaki mevcut veri bununla değişecek. "
                            "Sunucu durduruldu mu? [e/H] ").strip().lower()
                if ans not in ("e", "evet"):
                    print("Vazgeçildi.")
                    return 1
            before = b.run()  # geri dönüş yolu: o anki hâl
            print(f"Önce mevcut hâlin yedeği alındı: {before}")
            b.restore(a.dosya)
            print("Geri yüklendi. Sunucuyu başlatabilirsiniz.")
    except BackupError as e:
        print(f"Olmadı: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
