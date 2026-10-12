"""Zipformer transducer ASR (sherpa-onnx)."""
from __future__ import annotations

from pathlib import Path

import sherpa_onnx

from viet_asr.application.config import AsrParams, ModelRef
from viet_asr.application.ports import TextNormalizer
from viet_asr.domain import Syllable, VadSegment

_WORD_START = (" ", "▁")  # BPE'de yeni hece başlangıcı işareti


def tokens_to_syllables(tokens: list[str], timestamps: list[float], seg_start: float,
                        seg_end: float, normalizer: TextNormalizer,
                        max_duration: float = 1.0) -> list[Syllable]:
    """BPE parçalarını heceye birleştirir ve mutlak zaman verir.

    sherpa-onnx yalnızca her parçanın BAŞLANGIÇ zamanını verir ve bu zaman VAD
    parçasının başına göredir. Bu yüzden:
      - başlangıca seg_start eklenir,
      - bitiş = sonraki hecenin başlangıcı (en fazla max_duration sonra).
    """
    sylls: list[list] = []
    for tok, t in zip(tokens, timestamps):
        if tok.startswith(_WORD_START) or not sylls:
            sylls.append([tok.strip(" ▁"), seg_start + t])
        else:
            sylls[-1][0] += tok
    out = []
    for i, (text, start) in enumerate(sylls):
        nxt = sylls[i + 1][1] if i + 1 < len(sylls) else seg_end
        text = normalizer.normalize(text)
        if text:
            out.append(Syllable(text=text, start=start, end=max(start, min(nxt, start + max_duration))))
    return out


class SherpaOnnxAsr:
    def __init__(self, model: ModelRef, params: AsrParams, normalizer: TextNormalizer):
        self.name = model.name
        self.params = params
        self.normalizer = normalizer
        kw = {}
        if params.hotwords_file:
            # bpe ayarı olmadan sherpa-onnx hotword dosyasını SESSİZCE yok sayar.
            if not model.bpe_vocab or not Path(model.path(model.bpe_vocab)).exists():
                raise FileNotFoundError(
                    "hotwords_file verildi ama bpe.vocab yok; hotword sessizce çalışmaz. "
                    "scripts/make_bpe_vocab.py ile üretin.")
            if params.decoding_method != "modified_beam_search":
                raise ValueError("hotword yalnızca modified_beam_search ile çalışır")
            kw = dict(hotwords_file=params.hotwords_file, hotwords_score=params.hotwords_score,
                      modeling_unit="bpe", bpe_vocab=model.path(model.bpe_vocab))
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=model.path(model.encoder), decoder=model.path(model.decoder),
            joiner=model.path(model.joiner), tokens=model.path(model.tokens),
            num_threads=params.num_threads, decoding_method=params.decoding_method,
            provider="cpu", **kw)

    def transcribe(self, segments: list[VadSegment]) -> list[Syllable]:
        streams = []
        for seg in segments:
            st = self.recognizer.create_stream()
            st.accept_waveform(seg.sample_rate, seg.samples)
            streams.append(st)
        self.recognizer.decode_streams(streams)  # toplu çözme: tek tek çağırmaktan hızlı
        out: list[Syllable] = []
        for seg, st in zip(segments, streams):
            r = st.result
            out += tokens_to_syllables(list(r.tokens), list(r.timestamps), seg.start, seg.end,
                                       self.normalizer, self.params.max_syllable_duration)
        return out
