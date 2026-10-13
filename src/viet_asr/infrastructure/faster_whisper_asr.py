"""Türkçe ASR: faster-whisper (CTranslate2), fine-tuning projesindeki model (varsayılan medium).

Projedeki FasterWhisperAdapter'dan uyarlandı; düzeltilenler:
  1. Ayarlar transcribe() içinde genel Settings/ortam değişkeninden okunuyordu (gizli
     bağımlılık, testte değiştirilemez). Burada hepsi kurucuda verilir (WhisperParams).
  2. Toplu (batched) çözme hatası yakalanamıyordu: faster-whisper sonuçları tembel
     üretir (generator), hata transcribe() çağrısında değil döngüde çıkar; try bloğu
     boşa çalışıyordu. Burada sonuçlar try içinde listeye alınır, hata olursa tek tek
     çözmeye düşülür.
  3. Tekrar süzgeci noktalamaya bakıyordu ("evet," ile "evet" farklı sayılıyordu);
     burada yalnızca harf/rakamla karşılaştırılır.
  4. Kendi VAD'ı yerine hattın VAD parçaları kullanılır (clip_timestamps): konuşmacı
     ayrımı ve ASR aynı parçalara bakar, ses iki kez taranmaz.
  5. log_prob_threshold ve istem (initial_prompt): bkz. WhisperParams.

Model klasörü (çevrimdışı): models/faster-whisper-medium/ (ya da -small)/{model.bin, config.json,
tokenizer.json, vocabulary.txt}; scripts/download_models.py indirir.
"""
from __future__ import annotations

import logging

import numpy as np

from viet_asr.application.config import AsrParams, ModelRef
from viet_asr.application.ports import TextNormalizer
from viet_asr.domain import VadSegment, Word

log = logging.getLogger(__name__)
SR = 16000


def _key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def drop_repeats(words: list[Word], max_repeat: int) -> list[Word]:
    """Aynı kelime art arda max_repeat'ten fazla gelirse fazlasını atar
    ("evet evet evet evet" -> "evet evet"): Whisper'ın döngüye girmesine karşı."""
    out: list[Word] = []
    run = 0
    for w in words:
        run = run + 1 if out and _key(out[-1].text) == _key(w.text) else 1
        if run <= max_repeat:
            out.append(w)
    return out


class FasterWhisperAsr:
    segmentation = "vad"  # Whisper sessizlik/müzikte metin uyduruyor: yalnızca konuşma parçaları

    def __init__(self, model: ModelRef, params: AsrParams, normalizer: TextNormalizer,
                 language: str, whisper_model=None, batched=None):
        """whisper_model/batched: testler için sahte nesne verilebilir; verilmezse yüklenir."""
        self.name = model.name
        self.language = language
        self.params = params
        self.wp = params.whisper
        self.normalizer = normalizer
        self._drop = {_key(p) for p in params.drop_phrases}
        if self.wp.initial_prompt:  # istem sessiz yerde aynen yazılırsa süzülsün
            self._drop.add(_key(self.wp.initial_prompt))
        if whisper_model is None:
            from pathlib import Path

            from faster_whisper import BatchedInferencePipeline, WhisperModel

            if not (Path(model.folder) / "model.bin").exists():
                # klasör yoksa kütüphane adı Hugging Face'te arar: internetsiz sunucuda
                # anlaşılmaz bir hata yerine ne yapılacağı söylenir
                raise FileNotFoundError(
                    f"{model.folder}/model.bin yok: python scripts/download_models.py --tr {model.name}")
            whisper_model = WhisperModel(str(model.folder), device="cpu",
                                         compute_type=self.wp.compute_type,
                                         cpu_threads=params.num_threads, local_files_only=True)
            batched = BatchedInferencePipeline(model=whisper_model)
        self.model = whisper_model
        self.batched = batched

    def _options(self) -> dict:
        wp = self.wp
        return dict(language=self.language, task="transcribe", beam_size=wp.beam_size,
                    word_timestamps=True, condition_on_previous_text=False, temperature=0.0,
                    repetition_penalty=wp.repetition_penalty,
                    no_repeat_ngram_size=wp.no_repeat_ngram_size,
                    no_speech_threshold=wp.no_speech_threshold,
                    compression_ratio_threshold=wp.compression_ratio_threshold,
                    log_prob_threshold=wp.log_prob_threshold,
                    initial_prompt=wp.initial_prompt, hotwords=wp.hotwords)

    def transcribe(self, segments: list[VadSegment]) -> list[Word]:
        if not segments:
            return []
        raw = None
        if self.batched is not None:
            try:
                raw = self._batched(segments)
            except Exception as e:  # ör. bellek: tek tek çözmeye düş
                log.warning("toplu çözme başarısız (%s), parçalar tek tek çözülüyor", e)
        if raw is None:
            raw = self._one_by_one(segments)
        words: list[Word] = []
        for seg_text, seg_words in raw:
            if _key(seg_text) in self._drop:
                continue
            for w in seg_words:
                text = self.normalizer.normalize(w.word)
                if text:
                    words.append(Word(text=text, start=float(w.start), end=float(max(w.start, w.end))))
        return drop_repeats(words, self.wp.max_repeat)

    def _batched(self, segments: list[VadSegment]):
        """Parçalar asıl zamanlarıyla tek bir diziye yerleştirilir; clip_timestamps ile
        yalnızca onlar çözülür ve kelime zamanları doğrudan mutlak gelir."""
        end = max(s.end for s in segments)
        audio = np.zeros(int(np.ceil(end * SR)) + 1, dtype=np.float32)
        clips = []
        for s in segments:
            i = int(round(s.start * SR))
            n = min(len(s.samples), len(audio) - i)
            audio[i:i + n] = s.samples[:n]
            clips.append({"start": i / SR, "end": (i + n) / SR})
        result, _ = self.batched.transcribe(audio, clip_timestamps=clips,
                                            batch_size=self.wp.batch_size, **self._options())
        return [(r.text, list(r.words or [])) for r in result]  # list(): hatalar burada çıksın

    def _one_by_one(self, segments: list[VadSegment]):
        out = []
        for s in segments:
            result, _ = self.model.transcribe(s.samples, vad_filter=False, **self._options())
            for r in result:
                out.append((r.text, [_Shifted(w, s.start) for w in (r.words or [])]))
        return out


class _Shifted:
    """Tek parça çözümünde zamanlar parçanın başına göredir: mutlak zamana çevirir."""

    __slots__ = ("word", "start", "end")

    def __init__(self, w, offset: float):
        self.word, self.start, self.end = w.word, w.start + offset, w.end + offset
