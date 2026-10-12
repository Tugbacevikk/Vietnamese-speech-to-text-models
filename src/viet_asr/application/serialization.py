"""Sonucun dışarıya giden JSON biçimi (API yanıtı ve webhook gövdesi aynı)."""
from __future__ import annotations

from viet_asr.domain import JobRecord, LanguageDecision, Transcript


def decision_to_dict(d: LanguageDecision | None) -> dict | None:
    if d is None:
        return None
    return {"code": d.code, "confidence": d.confidence, "source": d.source.value,
            "uncertain": d.uncertain}


def transcript_to_dict(t: Transcript) -> dict:
    return {
        "job_id": t.job_id,
        "duration": round(t.duration, 3),
        "model": t.model,
        "language": decision_to_dict(t.language),
        "text": t.text,
        "utterances": [{"speaker": u.speaker, "start": round(u.start, 3), "end": round(u.end, 3),
                        "text": u.text} for u in t.utterances],
    }


def record_to_dict(r: JobRecord, transcript: Transcript | None = None) -> dict:
    out = {
        "job_id": r.job.job_id,
        "status": r.status.value,
        "requested_language": r.job.language,
        "language": decision_to_dict(r.decision),
        "error": r.error,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
    }
    if transcript is not None:
        out["result"] = transcript_to_dict(transcript)
    return out
