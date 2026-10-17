"""DARG web API (the project form's "API ve sunucu" step): ask, give feedback, see test statistics.

  uvicorn src.api.app:app --host 0.0.0.0 --port 8000      # docs: http://localhost:8000/docs

POST /soru           {"soru": "...", "oturum_id": "..."} -> answer, how it was understood, sources, timing
POST /geri-bildirim  {"kayit_id": "...", "puan": "iyi" | "kotu", "not": "..."}
GET  /istatistik     answered questions, positive-feedback share (form: >= 70%), bad-answer share (form: < 10%)
GET  /saglik         liveness check
GET  /kutuphane                  document library: every source document with title, category, province, size
GET  /kutuphane/{id}/dosya       the document itself (PDF or image), for reading or offline download
GET  /kutuphane/{id}/metin       its page texts, for keyword search on the phone without internet

Every question/answer and every feedback is appended to data/logs/ (git-ignored: questions may contain
personal information). Conversations are kept in memory per oturum_id (last turns only).
"""

from __future__ import annotations

import json
import threading
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.api import library
from src.rag.context import clean_url
from src.rag.generate_answer import generate_answer_v3
from src.rag.pipeline_v3 import MAX_HISTORY_TURNS

LOG_DIR = Path("data/logs")
QA_LOG = LOG_DIR / "qa_log.jsonl"
FEEDBACK_LOG = LOG_DIR / "feedback.jsonl"
MAX_SESSIONS = 1000

app = FastAPI(title="DARG", description="AFAD belgelerine dayalı Türkçe deprem bilgi asistanı", version="1.0")
# The mobile app's web build (flutter run -d chrome) calls the API from another origin.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])
_sessions: dict[str, list[dict[str, str]]] = {}
_lock = threading.Lock()


class Question(BaseModel):
    soru: str = Field(min_length=1, max_length=1000)
    oturum_id: str | None = Field(default=None, description="Aynı sohbetin soruları için aynı kimlik")


class Source(BaseModel):
    etiket: str
    tur: Literal["afad", "web", "il_karti"]
    kaynak: str
    sayfa: str | None = None
    baslik: str | None = Field(default=None, description="AFAD belgesinin başlığı")
    belge_id: str | None = Field(default=None, description="Kütüphanedeki belge (/kutuphane/{belge_id}/dosya)")
    ilk_sayfa: int | None = Field(default=None, description="Alıntının başladığı sayfa numarası")


class Sentence(BaseModel):
    metin: str
    kaynaklar: list[int] = Field(description="Cümlenin dayandığı kaynak numaraları; boşsa genel bilgi")


class Answer(BaseModel):
    kayit_id: str
    oturum_id: str
    cevap: str
    cumleler: list[Sentence] = Field(description="Cevabın cümleleri (uygulamalar kaynakları cümle cümle gösterir); "
                                                 "boşsa yalnız 'cevap' metni gösterilir")
    anlasilan_soru: str
    kaynaklar: list[Source]
    web_kullanildi: bool
    sure_sn: float


class Feedback(BaseModel):
    kayit_id: str
    puan: Literal["iyi", "kotu"]
    not_: str | None = Field(default=None, alias="not", max_length=2000)


def _append(path: Path, record: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@lru_cache(maxsize=1)
def _library_by_file() -> dict[str, library.LibraryDocument]:
    """Library documents by file name: an AFAD source links to the document the app can open."""
    return {document.file_name: document for document in library.catalog()}


def _source(meta: dict) -> Source:
    chunk_id = str(meta.get("chunk_id", ""))
    if chunk_id.startswith("web:"):
        return Source(etiket=meta["source_label"], tur="web", kaynak=clean_url(str(meta.get("source_file"))))
    kind = "il_karti" if chunk_id.startswith("il_karti:") else "afad"
    name = str(meta.get("source_file"))
    document = _library_by_file().get(unicodedata.normalize("NFC", name)) if kind == "afad" else None
    return Source(etiket=meta["source_label"], tur=kind, kaynak=name, sayfa=meta.get("page_label") or None,
                  baslik=document.title if document else None, belge_id=document.id if document else None,
                  ilk_sayfa=meta.get("page_start") if document else None)


@app.get("/saglik")
def health() -> dict:
    return {"durum": "ok"}


@app.get("/kutuphane")
def library_catalog() -> dict:
    documents = library.catalog()
    return {"belgeler": [document.to_api() for document in documents],
            "temel_paket": [document.id for document in documents if document.core],
            "kategoriler": library.CATEGORIES}


def _library_document(document_id: str) -> library.LibraryDocument:
    document = library.find(document_id)
    if document is None or not document.path.exists():
        raise HTTPException(status_code=404, detail="Bu numarayla bir belge bulunamadı.")
    return document


@app.get("/kutuphane/{document_id}/dosya")
def library_file(document_id: str) -> FileResponse:
    document = _library_document(document_id)
    return FileResponse(document.path, media_type=document.media_type, filename=document.file_name)


@app.get("/kutuphane/{document_id}/metin")
def library_text(document_id: str) -> dict:
    return {"id": document_id, "sayfalar": library.page_texts(_library_document(document_id))}


@app.post("/soru", response_model=Answer)
def ask(question: Question) -> Answer:
    session_id = question.oturum_id or uuid.uuid4().hex
    with _lock:
        history = list(_sessions.get(session_id, []))
    start = time.perf_counter()
    result = generate_answer_v3(question.soru, history=history)
    seconds = round(time.perf_counter() - start, 1)
    record_id = uuid.uuid4().hex
    sources = [_source(meta) for meta in result["sources"]]
    with _lock:
        if session_id not in _sessions and len(_sessions) >= MAX_SESSIONS:
            _sessions.pop(next(iter(_sessions)))
        turns = _sessions.setdefault(session_id, [])
        turns.append(result.get("turn", {"question": result["standalone_question"], "answer": result["answer"]}))
        del turns[:-MAX_HISTORY_TURNS]
    _append(QA_LOG, {"kayit_id": record_id, "zaman": datetime.now(timezone.utc).isoformat(), "oturum_id": session_id,
                     "soru": question.soru, "anlasilan_soru": result["standalone_question"], "cevap": result["answer"],
                     "kaynaklar": [source.model_dump() for source in sources], "web_kullanildi": result["used_web"],
                     "sure_sn": seconds})
    sentences = [Sentence(metin=s["text"], kaynaklar=s.get("sources", [])) for s in result.get("sentences", [])]
    return Answer(kayit_id=record_id, oturum_id=session_id, cevap=result["answer"], cumleler=sentences,
                  anlasilan_soru=result["standalone_question"], kaynaklar=sources,
                  web_kullanildi=result["used_web"], sure_sn=seconds)


@app.post("/geri-bildirim")
def feedback(item: Feedback) -> dict:
    if not any(record["kayit_id"] == item.kayit_id for record in _read(QA_LOG)):
        raise HTTPException(status_code=404, detail="Bu kayıt numarasıyla bir cevap bulunamadı.")
    _append(FEEDBACK_LOG, {"kayit_id": item.kayit_id, "zaman": datetime.now(timezone.utc).isoformat(),
                           "puan": item.puan, "not": item.not_})
    return {"durum": "kaydedildi"}


@app.get("/istatistik")
def statistics() -> dict:
    answers = _read(QA_LOG)
    latest: dict[str, str] = {}
    for record in _read(FEEDBACK_LOG):  # the last feedback per answer counts
        latest[record["kayit_id"]] = record["puan"]
    rated = len(latest)
    good = sum(1 for verdict in latest.values() if verdict == "iyi")
    times = sorted(record["sure_sn"] for record in answers)
    return {
        "soru_sayisi": len(answers),
        "geri_bildirim_sayisi": rated,
        "olumlu_orani": round(good / rated, 3) if rated else None,      # form: test users >= 70% positive
        "kotu_cevap_orani": round((rated - good) / rated, 3) if rated else None,  # form: error rate < 10%
        "ortanca_sure_sn": times[len(times) // 2] if times else None,
        "web_kullanim_orani": round(sum(r["web_kullanildi"] for r in answers) / len(answers), 3) if answers else None,
    }
