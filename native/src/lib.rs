//! viet_asr Rust çekirdeği. Saf algoritmalar `fusion` gibi modüllerde, Python bağı
//! yalnızca `python` özelliği açıkken derlenir (Rust servisi PyO3'süz kullanabilsin).

pub mod chunker;
pub mod fusion;
pub mod tokens;

#[cfg(feature = "python")]
mod py {
    use super::{chunker, fusion, tokens};
    use numpy::PyReadonlyArray1;
    use pyo3::prelude::*;

    /// energy_cuts(audio: np.ndarray[float32], sr, max_len, min_len, frame, silence_rms)
    ///   -> list[(başlangıç, bitiş)]  (numpy dizisi kopyalanmadan okunur)
    #[pyfunction]
    fn energy_cuts(
        py: Python<'_>,
        audio: PyReadonlyArray1<'_, f32>,
        sr: usize,
        max_len: f64,
        min_len: f64,
        frame: f64,
        silence_rms: f64,
    ) -> PyResult<Vec<(usize, usize)>> {
        let a = audio.as_slice()?;
        Ok(py.allow_threads(|| chunker::energy_cuts(a, sr, max_len, min_len, frame, silence_rms)))
    }

    /// assign_labels(word_starts, word_ends, seg_starts, seg_ends, seg_speakers,
    ///               min_dominance, max_gap) -> list[int]   (-1 = belirsiz)
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn assign_labels(
        py: Python<'_>,
        word_starts: Vec<f64>,
        word_ends: Vec<f64>,
        seg_starts: Vec<f64>,
        seg_ends: Vec<f64>,
        seg_speakers: Vec<i64>,
        min_dominance: f64,
        max_gap: f64,
    ) -> PyResult<Vec<i64>> {
        if word_starts.len() != word_ends.len()
            || seg_starts.len() != seg_ends.len()
            || seg_starts.len() != seg_speakers.len()
        {
            return Err(pyo3::exceptions::PyValueError::new_err("dizi uzunlukları uyuşmuyor"));
        }
        let words: Vec<(f64, f64)> = word_starts.into_iter().zip(word_ends).collect();
        let segs: Vec<fusion::Segment> = (0..seg_starts.len())
            .map(|i| fusion::Segment { start: seg_starts[i], end: seg_ends[i], speaker: seg_speakers[i] })
            .collect();
        let p = fusion::Params { min_dominance, max_gap };
        // hesap sırasında Python kilidi bırakılır: diğer iş parçacıkları beklemesin
        Ok(py.allow_threads(|| fusion::assign(&words, &segs, p)))
    }

    type RawWordTuple = (String, f64, f64, Option<f64>);

    /// group_tokens(tokens, timestamps, seg_start, seg_end, max_duration, log_probs=None)
    ///   -> list[(metin, başlangıç, bitiş, log olasılık ortalaması | None)]
    /// Normalizer uygulanmamış ham kelimeler; boş çıkanları Python ayıklar.
    #[pyfunction]
    #[pyo3(signature = (tokens, timestamps, seg_start, seg_end, max_duration, log_probs=None))]
    fn group_tokens(
        py: Python<'_>,
        tokens: Vec<String>,
        timestamps: Vec<f64>,
        seg_start: f64,
        seg_end: f64,
        max_duration: f64,
        log_probs: Option<Vec<f64>>,
    ) -> PyResult<Vec<RawWordTuple>> {
        Ok(py.allow_threads(|| {
            tokens::group(&tokens, &timestamps, seg_start, seg_end, max_duration, log_probs.as_deref())
                .into_iter()
                .map(|w| (w.text, w.start, w.end, w.mean_log_prob))
                .collect()
        }))
    }

    #[pymodule]
    fn viet_asr_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(assign_labels, m)?)?;
        m.add_function(wrap_pyfunction!(energy_cuts, m)?)?;
        m.add_function(wrap_pyfunction!(group_tokens, m)?)?;
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        Ok(())
    }
}
