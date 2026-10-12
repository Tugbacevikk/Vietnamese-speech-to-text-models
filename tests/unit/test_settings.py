from viet_asr.interface.settings import Settings, available_cpus


def test_eszamanlilik_otomatik_ve_elle(monkeypatch):
    monkeypatch.setenv("ONNX_THREADS", "2")
    monkeypatch.delenv("CONCURRENCY", raising=False)
    assert Settings().concurrency == max(1, available_cpus() // 2)
    monkeypatch.setenv("CONCURRENCY", "5")
    assert Settings().concurrency == 5
    monkeypatch.setenv("CONCURRENCY", "0")
    assert Settings().concurrency == 1
