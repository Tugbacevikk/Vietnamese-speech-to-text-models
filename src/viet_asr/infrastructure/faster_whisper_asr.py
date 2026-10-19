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
import os
import sys
import threading
from pathlib import Path

import numpy as np

from viet_asr.application.config import AsrParams, ModelRef
from viet_asr.application.ports import TextNormalizer
from viet_asr.domain import VadSegment, Word

log = logging.getLogger(__name__)
SR = 16000


def _cuda_ok() -> bool:
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() <= 0:
            return False
        if sys.platform == "win32":
            import ctypes
            candidates = [
                Path(os.environ.get("CUDA_DIR", "")) if os.environ.get("CUDA_DIR") else None,
                Path(os.environ.get("CUDA_DLL_DIR", "")) if os.environ.get("CUDA_DLL_DIR") else None,
                Path("D:/sesAnalizi_venv/Lib/site-packages/torch/lib"),
                Path(os.environ.get("USERPROFILE", "")) / "AppData" / "Local" / "Programs" / "Ollama" / "lib" / "ollama" / "cuda_v12",
                Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
                Path(sys.prefix) / "nvidia" / "cublas" / "bin",
                Path(os.environ.get("CUDA_PATH", "")) / "bin" if os.environ.get("CUDA_PATH") else None,
            ]
            for cand in candidates:
                if cand and cand.is_dir():
                    s = str(cand)
                    if s not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = s + ";" + os.environ.get("PATH", "")
                    if hasattr(os, "add_dll_directory"):
                        try:
                            os.add_dll_directory(s)
                        except Exception:
                            pass
            try:
                ctypes.CDLL("cublas64_12.dll")
            except Exception:
                try:
                    ctypes.CDLL("cublas64_11.dll")
                except Exception:
                    return False
        return True
    except Exception:
        return False


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
    supports_fast = True    # feedback profili: beam 1 (açgözlü), projedeki gibi
    supports_rescue = True  # güvenlik ağı: yazısız kalan konuşma aralıklarını yeniden çözer

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
            from faster_whisper import BatchedInferencePipeline, WhisperModel

            if not (Path(model.folder) / "model.bin").exists():
                # klasör yoksa kütüphane adı Hugging Face'te arar: internetsiz sunucuda
                # anlaşılmaz bir hata yerine ne yapılacağı söylenir
                raise FileNotFoundError(
                    f"{model.folder}/model.bin yok: python scripts/download_models.py --tr {model.name}")

            dev = getattr(self.wp, "device", "auto")
            cuda_ok = _cuda_ok()
            if dev == "auto":
                dev = "cuda" if cuda_ok else "cpu"
            elif dev == "cuda" and not cuda_ok:
                log.warning("CUDA istendi ama CUDA kütüphaneleri bulunamadı; CPU moduna düşülüyor.")
                dev = "cpu"

            c_type = getattr(self.wp, "compute_type", "auto")
            if c_type == "auto":
                c_type = "float16" if dev == "cuda" else "int8"

            dev_idx = getattr(self.wp, "device_index", 0)
            threads = self.wp.cpu_threads or params.num_threads
            log.info("FasterWhisper yükleniyor: model=%s, device=%s, compute_type=%s, threads=%d",
                     model.folder, dev, c_type, threads)

            try:
                whisper_model = WhisperModel(str(model.folder), device=dev,
                                             compute_type=c_type,
                                             device_index=dev_idx,
                                             cpu_threads=threads, local_files_only=True,
                                             num_workers=self.wp.num_workers)
                batched = BatchedInferencePipeline(model=whisper_model)
            except Exception as load_err:
                if "mkl_malloc" in str(load_err) or isinstance(load_err, MemoryError):
                    log.warning("Bellek yetersiz (%s); num_workers=1 ile yükleniyor.", load_err)
                    whisper_model = WhisperModel(str(model.folder), device="cpu",
                                                 compute_type="int8",
                                                 cpu_threads=min(4, threads), local_files_only=True,
                                                 num_workers=1)
                    batched = BatchedInferencePipeline(model=whisper_model)
                elif dev == "cuda":
                    log.warning("CUDA yükleme hatası (%s), CPU moduna düşülüyor.", load_err)
                    dev, c_type = "cpu", "int8"
                    whisper_model = WhisperModel(str(model.folder), device="cpu",
                                                 compute_type="int8",
                                                 cpu_threads=threads, local_files_only=True,
                                                 num_workers=self.wp.num_workers)
                    batched = BatchedInferencePipeline(model=whisper_model)
                else:
                    raise load_err
        self.model = whisper_model
        self.batched = batched
        # canlı ekranda gösterilir ("medium · GPU"): CTranslate2 modelinin gerçekten açıldığı cihaz
        self.device = str(getattr(getattr(whisper_model, "model", None), "device", "cpu"))
        # Tek model nesnesi aynı anda tek çözüm: Windows'ta eşzamanlı çağrıda CTranslate2 0xc0000005
        # ile çöküyordu; ayrıca BatchedInferencePipeline kelime zamanı için ortak durum tutuyor
        # (last_speech_timestamp), eşzamanlı işlerde zamanlar karışır. Bedeli: canlı Türkçe altyazı
        # ve paralel Türkçe işler aynı modeli sırayla kullanır.
        self._lock = threading.RLock()

    # sıcaklık denemesine kalan en fazla parça (en uzunlar): her deneme ayrı 30 sn'lik çözüm
    RESCUE_MAX_RETRY = 4

    def rescue(self, segments: list[VadSegment]) -> list[Word]:
        """Güvenlik ağı: ilk geçişte kelime çıkmayan konuşma aralıklarını yeniden çözer.

        Önceki sürüm (ve projedeki) her aralığı tek tek, 4 sıcaklıkla ve sıcaklık başına 5 örnekle
        (best_of) çözüyordu. Whisper her çağrıda 30 sn'lik pencere işlediği için 2 sn'lik boşluk
        da 30 sn'lik ses kadar sürüyordu. Şimdi:
          1. tüm aralıklar TEK toplu çözümde, sıcaklık 0, gevşek eşiklerle;
          2. hâlâ boş kalanların en uzun RESCUE_MAX_RETRY tanesi tek tek, sıcaklık 0,2 ve 0,4,
             tek örnek (best_of=1), beam 1.
        """
        if not segments:
            return []
        with self._lock:
            raw, todo = [], list(segments)
            if self.batched is not None:
                try:
                    raw = self._batched(segments, opts=self._options(rescue=True))
                    todo = [s for s in segments if not _has_word(raw, s)]
                except Exception as e:
                    log.warning("güvenlik ağı toplu çözümü başarısız (%s), tek tek", e)
                    raw, todo = [], list(segments)
            todo = sorted(todo, key=lambda s: len(s.samples), reverse=True)[:self.RESCUE_MAX_RETRY]
            raw += self._one_by_one(todo, rescue=True, retry=True)
            return drop_repeats(self._words(raw), self.wp.max_repeat)

    def _words(self, raw) -> list[Word]:
        words: list[Word] = []
        for seg_text, seg_words in raw:
            if _key(seg_text) in self._drop:
                continue
            for w in seg_words:
                text = self.normalizer.normalize(w.word)
                if text:
                    p = getattr(w, "probability", None)
                    words.append(Word(text=text, start=float(w.start), end=float(max(w.start, w.end)),
                                      conf=round(float(p), 3) if p is not None else None))
        return words

    def _options(self, fast: bool = False, rescue: bool = False, retry: bool = False) -> dict:
        wp = self.wp
        if rescue:
            return dict(language=self.language, task="transcribe",
                        beam_size=1 if retry else wp.beam_size, best_of=1,
                        word_timestamps=True, condition_on_previous_text=False,
                        temperature=(0.2, 0.4) if retry else 0.0,
                        repetition_penalty=wp.repetition_penalty,
                        no_repeat_ngram_size=wp.no_repeat_ngram_size, no_speech_threshold=0.8,
                        compression_ratio_threshold=wp.compression_ratio_threshold,
                        log_prob_threshold=-1.0, initial_prompt=wp.initial_prompt, hotwords=wp.hotwords)
        return dict(language=self.language, task="transcribe", beam_size=1 if fast else wp.beam_size,
                    word_timestamps=True, condition_on_previous_text=False, temperature=0.0,
                    repetition_penalty=wp.repetition_penalty,
                    no_repeat_ngram_size=wp.no_repeat_ngram_size,
                    no_speech_threshold=wp.no_speech_threshold,
                    compression_ratio_threshold=wp.compression_ratio_threshold,
                    log_prob_threshold=wp.log_prob_threshold,
                    initial_prompt=wp.initial_prompt, hotwords=wp.hotwords)

    def transcribe(self, segments: list[VadSegment], fast: bool = False) -> list[Word]:
        if not segments:
            return []
        with self._lock:
            raw = None
            if self.batched is not None:
                try:
                    raw = self._batched(segments, fast)
                except Exception as e:  # ör. bellek: tek tek çözmeye düş
                    log.warning("toplu çözme başarısız (%s), parçalar tek tek çözülüyor", e)
            if raw is None:
                raw = self._one_by_one(segments, fast)
            return drop_repeats(self._words(raw), self.wp.max_repeat)

    def transcribe_text(self, samples: np.ndarray, fast: bool = False, prompt: str | None = None,
                        normalizer: TextNormalizer | None = None) -> str:
        """Canlı altyazı: tek cümlenin sesi -> metin (kelime zamanı yok, daha hızlı). Live-stream'den.

        fast=True: ara metin (beam 1, sıcaklık geri çekilmesi yok).
        prompt: önceki cümleler; Whisper'a bağlam olur (özel adlar, konu tutarlılığı).
        normalizer: ekran biçimi (büyük harf ve noktalama korunur); verilmezse toplu iş biçimi."""
        if samples.size == 0:
            return ""
        with self._lock:
            wp = self.wp
            result, _ = self.model.transcribe(
                samples, language=self.language, task="transcribe", vad_filter=False,
                beam_size=1 if fast else wp.beam_size,
                temperature=0.0 if fast else (0.0, 0.2, 0.4),
                initial_prompt=prompt or wp.initial_prompt, hotwords=wp.hotwords,
                condition_on_previous_text=False, without_timestamps=True,
                repetition_penalty=wp.repetition_penalty, no_repeat_ngram_size=wp.no_repeat_ngram_size,
                no_speech_threshold=wp.no_speech_threshold,
                compression_ratio_threshold=wp.compression_ratio_threshold,
                log_prob_threshold=wp.log_prob_threshold)
            norm = normalizer or self.normalizer
            parts = []
            for r in result:
                if _key(r.text) in self._drop or (prompt and _key(r.text) == _key(prompt)):
                    continue  # uydurma kalıp ya da istemin aynen tekrarı
                text = norm.normalize(r.text)
                if text:
                    parts.append(text)
            words = [Word(text=w, start=0.0, end=0.0) for w in " ".join(parts).split()]
            return " ".join(w.text for w in drop_repeats(words, wp.max_repeat))

    def _batched(self, segments: list[VadSegment], fast: bool = False, opts: dict | None = None):
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
                                            batch_size=self.wp.batch_size, **(opts or self._options(fast)))
        return [(r.text, list(r.words or [])) for r in result]  # list(): hatalar burada çıksın

    def _one_by_one(self, segments: list[VadSegment], fast: bool = False, rescue: bool = False,
                    retry: bool = False):
        out = []
        for s in segments:
            result, _ = self.model.transcribe(s.samples, vad_filter=False,
                                              **self._options(fast, rescue, retry))
            for r in result:
                out.append((r.text, [_Shifted(w, s.start) for w in (r.words or [])]))
        return out


def _has_word(raw, seg: VadSegment) -> bool:
    """Toplu çözümde bu aralığa düşen kelime var mı (kelime zamanları mutlak)."""
    a, b = seg.start, seg.start + len(seg.samples) / SR
    return any(a <= float(w.start) < b for _, ws in raw for w in ws)


class _Shifted:
    """Tek parça çözümünde zamanlar parçanın başına göredir: mutlak zamana çevirir."""

    __slots__ = ("word", "start", "end", "probability")

    def __init__(self, w, offset: float):
        self.word, self.start, self.end = w.word, w.start + offset, w.end + offset
        self.probability = getattr(w, "probability", None)
