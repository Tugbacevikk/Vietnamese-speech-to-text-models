"""Bakım süreci (tam modda; yerel modda local_server bunu kendi içinde çalıştırır).

    python -m viet_asr.interface.bakim            # her 60 sn
"""
from __future__ import annotations

import logging
import time

from viet_asr.infrastructure.db import SqlAlchemyJobStore
from viet_asr.interface.composition import build_maintenance
from viet_asr.interface.settings import Settings


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = Settings()
    store = SqlAlchemyJobStore(s.database_url, pool_size=1)
    store.migrate()
    m = build_maintenance(store, s)
    while True:
        try:
            m.run_once()
        except Exception:
            logging.exception("bakım turu başarısız")
        time.sleep(60)


if __name__ == "__main__":
    main()
