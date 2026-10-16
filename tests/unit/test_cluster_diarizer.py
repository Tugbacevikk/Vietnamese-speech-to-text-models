"""tests/unit/test_cluster_diarizer.py

nme_spectral ve EmbeddingClusterDiarizer birim testleri.
"""

import numpy as np

from viet_asr.infrastructure.cluster_diarizer import nme_spectral


def test_nme_spectral_single_speaker():
    """Tek kişiye ait gömmelerde k = 1 tahmin edilir ve tüm etiketler 0 olur."""
    rng = np.random.default_rng(0)
    center = rng.standard_normal(192)
    center /= np.linalg.norm(center)

    # Tek konuşmacı etrafında küçük gürültülü 50 pencere
    E = np.tile(center, (50, 1)) + 0.05 * rng.standard_normal((50, 192))
    E /= np.linalg.norm(E, axis=1, keepdims=True)

    labels = nme_spectral(E)
    assert len(labels) == 50
    assert np.all(labels == 0)
    assert len(np.unique(labels)) == 1


def test_nme_spectral_two_speakers():
    """İki farklı konuşmacıya ait gömmelerde k = 2 küme bulunur."""
    rng = np.random.default_rng(42)
    v1 = rng.standard_normal(192)
    v1 /= np.linalg.norm(v1)
    v2 = rng.standard_normal(192)
    v2 /= np.linalg.norm(v2)

    E1 = np.tile(v1, (25, 1)) + 0.05 * rng.standard_normal((25, 192))
    E2 = np.tile(v2, (25, 1)) + 0.05 * rng.standard_normal((25, 192))
    E = np.vstack([E1, E2])
    E /= np.linalg.norm(E, axis=1, keepdims=True)

    labels = nme_spectral(E)
    assert len(labels) == 50
    assert len(np.unique(labels)) == 2


def test_nme_spectral_short_input():
    """3'ten az pencerede kümeleme yapılmaz, doğrudan sıfır döner."""
    E = np.zeros((2, 192))
    labels = nme_spectral(E)
    assert len(labels) == 2
    assert np.all(labels == 0)


def test_nme_spectral_explicit_k():
    """Kullanıcı k parametresi verdiğinde o sayıya uyulur."""
    rng = np.random.default_rng(1)
    E = rng.standard_normal((30, 192))
    E /= np.linalg.norm(E, axis=1, keepdims=True)

    labels_k1 = nme_spectral(E, k=1)
    assert np.all(labels_k1 == 0)

    labels_k3 = nme_spectral(E, k=3)
    assert len(np.unique(labels_k3)) == 3


def test_merge_close_threshold():
    """merge_distance eşiğine göre yakın kümeler birleştirilir, uzaklar korunur."""
    from viet_asr.application.config import DiarizerConfig
    from viet_asr.infrastructure.cluster_diarizer import EmbeddingClusterDiarizer

    c1 = np.zeros(192)
    c1[0] = 1.0
    c2 = np.zeros(192)
    c2[0] = 0.70
    c2[1] = np.sqrt(1.0 - 0.70**2)

    E = np.vstack([np.tile(c1, (10, 1)), np.tile(c2, (10, 1))])
    lab = np.array([0] * 10 + [1] * 10)

    # merge_distance = 0.35 iken (kosinüs uzaklığı 0.30 < 0.35) tek kümeye birleşir
    d35 = EmbeddingClusterDiarizer.__new__(EmbeddingClusterDiarizer)
    d35.cfg = DiarizerConfig(merge_distance=0.35)
    res35 = d35._merge_close(E, lab.copy())
    assert len(np.unique(res35)) == 1

    # merge_distance = 0.28 iken (0.30 >= 0.28) ayrı kalır
    d28 = EmbeddingClusterDiarizer.__new__(EmbeddingClusterDiarizer)
    d28.cfg = DiarizerConfig(merge_distance=0.28)
    res28 = d28._merge_close(E, lab.copy())
    assert len(np.unique(res28)) == 2

