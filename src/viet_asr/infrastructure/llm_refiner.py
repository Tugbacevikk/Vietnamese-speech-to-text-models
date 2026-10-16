"""LLM destekli konuşmacı ayrıştırma ve rol düzeltme servisi (Ollama / OpenAI).

Akustik diarization modeli (CAM++ / Pyannote) aynı ortamda (arka plan müziği, stüdyo yankısı)
konuşan aynı cinsiyetteki kişileri (örneğin soru soran kadın katılımcılar ile cevap veren kadın doktoru)
aynı konuşmacı kümesine birleştirebilir.

LlmSpeakerRefiner, transkriptteki diyalog akışını, soru-cevap yapısını ve hitapları inceleyerek:
  - Soru soran kişi (Khán giả / Arayan)
  - Sunucu / Moderatör (MC)
  - Uzman / Doktor (Bác sĩ / Chuyên gia)
rollerini tespit eder ve konuşmacı kimliklerini düzeltir.

Fail-Open Güvencesi:
  Ollama erişilemezse, zaman aşımına uğrarsa veya yanıt ayrıştırılamazsa işlem aksamaz;
  orijinal akustik transkript olduğu gibi korunur.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import replace

from viet_asr.application.config import LlmRefinerConfig
from viet_asr.domain import Utterance

log = logging.getLogger(__name__)


class LlmSpeakerRefiner:
    def __init__(self, cfg: LlmRefinerConfig | None = None):
        self.cfg = cfg or LlmRefinerConfig()
        self.enabled = self.cfg.enabled

    def refine(self, utterances: list[Utterance], language: str = "vi") -> list[Utterance]:
        if not self.enabled or len(utterances) <= 1:
            return utterances

        try:
            spk_map = self._query_llm(utterances, language)
            if not spk_map:
                return utterances

            updated: list[Utterance] = []
            changes = 0
            for idx, u in enumerate(utterances):
                new_spk = spk_map.get(str(idx))
                if new_spk is None:
                    new_spk = spk_map.get(idx)

                if new_spk:
                    new_spk_str = str(new_spk).strip().lower()
                    m = re.search(r"(speaker_\d+|\d+)", new_spk_str)
                    if m:
                        matched = m.group(1)
                        if matched.isdigit():
                            formatted_spk = f"speaker_{int(matched):02d}"
                        else:
                            formatted_spk = matched
                    else:
                        formatted_spk = u.speaker

                    if formatted_spk != u.speaker:
                        changes += 1
                        updated.append(replace(u, speaker=formatted_spk))
                    else:
                        updated.append(u)
                else:
                    updated.append(u)

            if changes > 0:
                log.info("LLM konuşmacı düzeltmesi uygulandı: %d satır güncellendi.", changes)
            return updated

        except Exception as e:
            log.warning("LLM konuşmacı düzeltmesi başarısız oldu (%s). Orijinal konuşmacılar korunuyor.", e)
            return utterances

    def _query_llm(self, utterances: list[Utterance], language: str) -> dict[str | int, str] | None:
        prompt = self._build_prompt(utterances, language)
        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "num_ctx": 2048,
                "num_predict": self.cfg.num_predict,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.cfg.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_sec) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                raw_response = result.get("response", "")
                return self._parse_response(raw_response)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            log.warning("Ollama LLM servisine erişilemedi veya zaman aşımı (%s: %s).", type(e).__name__, e)
            return None

    def _build_prompt(self, utterances: list[Utterance], language: str) -> str:
        lines = "\n".join(f"[{i}] {u.text}" for i, u in enumerate(utterances))

        if language == "vi":
            return f"""Bạn là chuyên gia phân loại người nói trong chương trình truyền hình đối thoại y tế.
Hãy xác định mã người nói cho từng câu:

1. Khán giả (người đặt câu hỏi cho bác sĩ, ví dụ: 'em xin hỏi bác sĩ', 'thưa bác sĩ'): gán "speaker_00"
2. MC / Sunucu (người điều phối chương trình, cảm ơn câu hỏi, mời khán giả tiếp theo, mời bác sĩ): gán "speaker_01"
3. Bác sĩ / Chuyên gia (người trả lời tư vấn y tế, hướng dẫn liệu trình điều trị, viên uống): gán "speaker_02"

QUY TẮC BẮT BUỘC:
- Bác sĩ trả lời tư vấn TUYỆT ĐỐI KHÔNG ĐƯỢC gán cùng người với khán giả hỏi. Bác sĩ phải là "speaker_02".
- MC điều phối chương trình phải là "speaker_01".
- Khán giả hỏi là "speaker_00".

Dữ liệu các câu thoại:
{lines}

Trả về DUY NHẤT một JSON map từ chỉ số chuỗi sang mã speaker:"""

        # Türkçe ve diğer diller için
        return f"""Diyalog dökümündeki konuşmacı rollerini analiz et ve her cümlenin konuşmacı kodunu belirle:
- Soru Soran / Arayan / Katılımcı: "speaker_00"
- Sunucu / Moderatör / Temsilci: "speaker_01"
- Doktor / Uzman (cevap veren): "speaker_02"

KURAL: Soru soran kişi ile cevap veren uzman aynı kişi olamaz.
Diyalog satırları:
{lines}

SADECE satır indeksinden konuşmacı koduna eşleme yapan JSON sözlüğü döndür:"""

    def _parse_response(self, text: str) -> dict[str | int, str] | None:
        if not text:
            return None
        clean = text.strip()
        if "```json" in clean:
            clean = clean.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in clean:
            clean = clean.split("```", 1)[1].split("```", 1)[0].strip()

        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            clean = m.group(0)

        try:
            parsed = json.loads(clean)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return None
