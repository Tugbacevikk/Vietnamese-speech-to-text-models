"""Webhook göndericisi süreci:  python -m viet_asr.interface.dispatcher"""
from __future__ import annotations

import logging

from viet_asr.infrastructure.postgres import PostgresJobStore
from viet_asr.infrastructure.webhook import OutboxDispatcher
from viet_asr.interface.settings import Settings


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    store = PostgresJobStore(Settings().database_url, max_size=1)
    store.migrate()
    OutboxDispatcher(store).run_forever()


if __name__ == "__main__":
    main()
