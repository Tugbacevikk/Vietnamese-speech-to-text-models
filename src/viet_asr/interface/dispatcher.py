"""Webhook göndericisi süreci:  python -m viet_asr.interface.dispatcher"""
from __future__ import annotations

import logging

from viet_asr.infrastructure.db import SqlAlchemyJobStore
from viet_asr.infrastructure.webhook import OutboxDispatcher
from viet_asr.interface.settings import Settings


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    store = SqlAlchemyJobStore(Settings().database_url, pool_size=1)
    store.migrate()
    OutboxDispatcher(store, secret=Settings().webhook_secret).run_forever()


if __name__ == "__main__":
    main()
