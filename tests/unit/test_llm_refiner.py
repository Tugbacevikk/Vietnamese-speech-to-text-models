"""tests/unit/test_llm_refiner.py

LlmSpeakerRefiner birim testleri (mock ve fail-open güvenceleri).
"""
import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

from viet_asr.application.config import LlmRefinerConfig
from viet_asr.domain import Utterance
from viet_asr.infrastructure.llm_refiner import LlmSpeakerRefiner


def _dummy_utts() -> list[Utterance]:
    return [
        Utterance(speaker="speaker_00", start=0.0, end=5.0, text="Em xin hỏi bác sĩ là nguyên nhân do đâu?"),
        Utterance(speaker="speaker_00", start=5.5, end=10.0, text="Cảm ơn câu hỏi, xin mời quý vị tiếp tục."),
        Utterance(speaker="speaker_00", start=10.5, end=20.0, text="Dạ thưa quý vị, để điều trị rụng tóc thì cần dùng viên uống."),
    ]


def test_llm_refiner_disabled():
    cfg = LlmRefinerConfig(enabled=False)
    refiner = LlmSpeakerRefiner(cfg)
    utts = _dummy_utts()
    res = refiner.refine(utts)
    assert res == utts


def test_llm_refiner_single_utterance():
    cfg = LlmRefinerConfig(enabled=True)
    refiner = LlmSpeakerRefiner(cfg)
    single = [Utterance(speaker="speaker_00", start=0.0, end=5.0, text="Tek cümle.")]
    res = refiner.refine(single)
    assert res == single


def test_llm_refiner_success_mock():
    cfg = LlmRefinerConfig(enabled=True)
    refiner = LlmSpeakerRefiner(cfg)
    utts = _dummy_utts()

    mock_response = json.dumps({
        "response": json.dumps({"0": "speaker_00", "1": "speaker_01", "2": "speaker_02"})
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_response
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = refiner.refine(utts, language="vi")

    assert len(res) == 3
    assert res[0].speaker == "speaker_00"
    assert res[1].speaker == "speaker_01"  # MC düzeltildi
    assert res[2].speaker == "speaker_02"  # Doktor düzeltildi
    assert res[0].text == utts[0].text
    assert res[2].start == utts[2].start


def test_llm_refiner_fail_open_on_timeout():
    """Ollama zaman aşımına uğrarsa transkript bozulmaz veya hata fırlatmaz (fail-open)."""
    cfg = LlmRefinerConfig(enabled=True, timeout_sec=1)
    refiner = LlmSpeakerRefiner(cfg)
    utts = _dummy_utts()

    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        res = refiner.refine(utts, language="vi")

    assert res == utts
    assert all(u.speaker == "speaker_00" for u in res)


def test_llm_refiner_fail_open_on_connection_refused():
    """Ollama kapalıysa (URLError) hata patlamaz, orijinal liste döner."""
    cfg = LlmRefinerConfig(enabled=True)
    refiner = LlmSpeakerRefiner(cfg)
    utts = _dummy_utts()

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        res = refiner.refine(utts, language="vi")

    assert res == utts


def test_llm_refiner_malformed_json():
    """Model bozuk veya eksik JSON üretirse orijinal konuşmacılar korunur."""
    cfg = LlmRefinerConfig(enabled=True)
    refiner = LlmSpeakerRefiner(cfg)
    utts = _dummy_utts()

    mock_response = json.dumps({
        "response": "Bu bir JSON değil: {bozuk"
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_response
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = refiner.refine(utts, language="vi")

    assert res == utts


def test_llm_refiner_prompt_builder():
    cfg = LlmRefinerConfig(enabled=True)
    refiner = LlmSpeakerRefiner(cfg)
    utts = _dummy_utts()

    prompt_vi = refiner._build_prompt(utts, "vi")
    assert "MC / Sunucu" in prompt_vi
    assert "Bác sĩ" in prompt_vi
    assert utts[0].text in prompt_vi

    prompt_tr = refiner._build_prompt(utts, "tr")
    assert "Doktor / Uzman" in prompt_tr
    assert "Sunucu" in prompt_tr
