from pathlib import Path
import pytest

from viet_asr.application.config import AsrParams, ModelRef
from viet_asr.infrastructure.normalizer import ViNormalizer
from viet_asr.infrastructure.sherpa_asr import SherpaZipformerAsr


def test_tibbi_hotword_dosyasi_gecerli():
    hw_path = Path("hotwords/tibbi_vi.txt")
    assert hw_path.exists(), "hotwords/tibbi_vi.txt dosyası bulunamadı"
    lines = [line.strip() for line in hw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 50, f"En az 50 tıbbi terim bekleniyordu, {len(lines)} bulundu"
    # Tüm terimler BÜYÜK HARFLE olmalı (sherpa-onnx şartı)
    for term in lines:
        assert term == term.upper(), f"Hotword büyük harfle olmalı: {term}"
    assert "BÁC SĨ" in lines
    assert "BỆNH VIỆN" in lines
    assert "HUYẾT ÁP" in lines
    assert "TIỂU ĐƯỜNG" in lines


def test_hotword_modified_beam_search_disinda_hata_verir(tmp_path):
    hw = tmp_path / "hw.txt"
    hw.write_text("BÁC SĨ\n", encoding="utf-8")
    bpe = tmp_path / "bpe.vocab"
    bpe.write_text("dummy\n", encoding="utf-8")

    mref = ModelRef(
        name="zipformer-68m",
        kind="zipformer-transducer",
        folder=tmp_path,
        files={"encoder": "enc.onnx", "decoder": "dec.onnx", "joiner": "join.onnx",
               "tokens": "tokens.txt", "bpe_vocab": "bpe.vocab"}
    )
    # greedy ile hotword verilemez
    params = AsrParams(decoding_method="greedy", hotwords_file=str(hw))
    with pytest.raises(ValueError, match="hotword yalnızca modified_beam_search"):
        SherpaZipformerAsr(mref, params, ViNormalizer())


def test_hotword_bpe_vocab_olmadan_hata_verir(tmp_path):
    hw = tmp_path / "hw.txt"
    hw.write_text("BÁC SĨ\n", encoding="utf-8")

    mref = ModelRef(
        name="zipformer-68m",
        kind="zipformer-transducer",
        folder=tmp_path,
        files={"encoder": "enc.onnx", "decoder": "dec.onnx", "joiner": "join.onnx",
               "tokens": "tokens.txt"}
    )
    params = AsrParams(decoding_method="modified_beam_search", hotwords_file=str(hw))
    with pytest.raises(FileNotFoundError, match="bpe.vocab yok"):
        SherpaZipformerAsr(mref, params, ViNormalizer())
