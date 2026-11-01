"""Request limits: keep the OpenAI bill bounded and the server usable when the API is public.

Per client address: a burst limit (per minute) and a daily limit, kept in memory only. Addresses are never written to
disk or logs; a restart forgets them. Whole service: daily caps on questions and on web searches (the expensive step),
stored in the SQLite database so a restart does not reset them. When the web cap is reached, answers are written
from the AFAD documents only. At most a few answers are written at the same time; the rest wait briefly in line.

The client address comes from Caddy's X-Forwarded-For header (uvicorn trusts it through FORWARDED_ALLOW_IPS,
see Dockerfile). Days are UTC days (they start at 03:00 Turkish time). Numbers come from .env (see .env.example).
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from fastapi import HTTPException, Request


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


QUESTIONS_PER_MINUTE = _env_int("KISI_DAKIKA_SORU", 5)
QUESTIONS_PER_DAY = _env_int("KISI_GUNLUK_SORU", 30)
DAILY_QUESTIONS = _env_int("GUNLUK_SORU_TAVANI", 150)
DAILY_WEB_SEARCHES = _env_int("GUNLUK_WEB_TAVANI", 40)
CONCURRENT_ANSWERS = _env_int("ES_ZAMANLI_SORU", 3)
QUEUE_WAIT_SECONDS = 30

TOO_MANY_MESSAGE = ("Kısa sürede çok fazla istek gönderildi. Lütfen biraz sonra tekrar deneyin. "
                    "Acil bir durumdaysanız hemen 112'yi arayın.")
DAILY_CAP_MESSAGE = ("DARG bugün için soru sınırına ulaştı; yarın tekrar deneyebilirsiniz. "
                     "Acil bir durumdaysanız hemen 112'yi arayın.")
BUSY_MESSAGE = ("DARG şu anda çok yoğun. Lütfen birkaç saniye sonra tekrar deneyin. "
                "Acil bir durumdaysanız hemen 112'yi arayın.")

_lock = threading.Lock()
_recent: dict[tuple[str, str], deque[float]] = {}   # (rule, address) -> request times in the last minute
_daily: dict[tuple[str, str], int] = {}             # (rule, address) -> requests today
_daily_day = ""
_blocked: dict[str, int] = {}                        # rule -> refused requests today (for /istatistik)
_answer_slots = threading.BoundedSemaphore(CONCURRENT_ANSWERS)


def today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def client_address(request: Request) -> str:
    return request.client.host if request.client else "bilinmiyor"


class Rule:
    """A FastAPI dependency: `dependencies=[Depends(Rule("soru", per_minute=5, per_day=30))]`."""

    def __init__(self, name: str, *, per_minute: int | None = None, per_day: int | None = None) -> None:
        self.name, self.per_minute, self.per_day = name, per_minute, per_day

    def __call__(self, request: Request) -> None:
        global _daily_day
        key = (self.name, client_address(request))
        moment = time.monotonic()
        with _lock:
            if _daily_day != today():
                _daily.clear()
                _blocked.clear()
                _daily_day = today()
            recent = _recent.setdefault(key, deque())
            while recent and moment - recent[0] > 60:
                recent.popleft()
            if ((self.per_minute is not None and len(recent) >= self.per_minute)
                    or (self.per_day is not None and _daily.get(key, 0) >= self.per_day)):
                _blocked[self.name] = _blocked.get(self.name, 0) + 1
                raise HTTPException(status_code=429, detail=TOO_MANY_MESSAGE)
            recent.append(moment)
            _daily[key] = _daily.get(key, 0) + 1
            if len(_recent) > 50_000:  # forget idle addresses
                for idle in [k for k, times in _recent.items() if not times or moment - times[-1] > 60]:
                    del _recent[idle]


def reset() -> None:
    """Forget the per-address counters (tests)."""
    with _lock:
        _recent.clear()
        _daily.clear()
        _blocked.clear()


def note_blocked(name: str) -> None:
    with _lock:
        _blocked[name] = _blocked.get(name, 0) + 1


def blocked_today() -> dict[str, int]:
    with _lock:
        return dict(_blocked)


# --- whole-service daily caps (SQLite: survive restarts) -----------------------------------------------------------

def _connect():
    from src.api import community  # imported here: community uses the rules above
    return community.connect()


def used(name: str) -> int:
    with _connect() as db:
        row = db.execute("SELECT count FROM usage WHERE day = ? AND name = ?", (today(), name)).fetchone()
    return row[0] if row else 0


def take(name: str, cap: int) -> bool:
    """Count one use of `name` today if the cap allows it."""
    with _connect() as db:
        db.execute("INSERT OR IGNORE INTO usage (day, name, count) VALUES (?, ?, 0)", (today(), name))
        return db.execute("UPDATE usage SET count = count + 1 WHERE day = ? AND name = ? AND count < ?",
                          (today(), name, cap)).rowcount == 1


def add(name: str) -> None:
    with _connect() as db:
        db.execute("INSERT INTO usage (day, name, count) VALUES (?, ?, 1) "
                   "ON CONFLICT(day, name) DO UPDATE SET count = count + 1", (today(), name))


@contextmanager
def answer_slot() -> Iterator[None]:
    """At most CONCURRENT_ANSWERS answers at once; a request waits up to QUEUE_WAIT_SECONDS for a free slot."""
    if not _answer_slots.acquire(timeout=QUEUE_WAIT_SECONDS):
        raise HTTPException(status_code=503, detail=BUSY_MESSAGE)
    try:
        yield
    finally:
        _answer_slots.release()


QUESTION = Rule("soru", per_minute=QUESTIONS_PER_MINUTE, per_day=QUESTIONS_PER_DAY)
FEEDBACK = Rule("geri_bildirim", per_minute=20)
APPLICATION = Rule("basvuru", per_minute=2, per_day=5)
LOGIN = Rule("giris", per_minute=10, per_day=100)
REPORT = Rule("sikayet", per_minute=10, per_day=50)
