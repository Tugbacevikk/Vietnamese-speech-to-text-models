# Mimari kararlar (ADR)

Her dosya bir kararı kaydeder: bağlam, karar, sonuçları. Karar değişirse dosya silinmez,
"Yerini aldı: NNNN" notu eklenir ve yeni ADR yazılır.

| No | Karar | Durum |
|----|-------|-------|
| [0001](0001-kuyruk-redis-streams.md) | Kuyruk: Celery yerine Redis Streams | Kabul |
| [0002](0002-veritabani-code-first.md) | Veritabanı: SQLAlchemy + Alembic, SQLite ve PostgreSQL tek kod | Kabul |
| [0003](0003-konusmaci-ayirma.md) | Konuşmacı ayırma: CAM++ imzası + spektral kümeleme | Kabul |
| [0004](0004-dil-ve-modeller.md) | Dil seçimi ve dil başına model | Kabul |
| [0005](0005-rust-gecisi.md) | Rust'a kademeli geçiş | Öneri |
