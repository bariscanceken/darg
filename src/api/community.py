"""Verified announcements: organisations apply for an account, post announcements, administrators moderate.

Citizens never log in; they read the feed and can report a post. Organisations (municipality, AFAD provincial office,
Kızılay, hospital…) apply, an administrator checks and approves them. A new organisation's posts wait in the
moderation queue; an organisation marked trusted publishes directly (minutes count in a disaster) and an administrator
can still remove anything. Every post has an expiry: stale disaster information is as harmful as wrong information.
Citizens' reports never take a post down by themselves: every reported post goes to the administrators, who decide
(the project owner's rule; a coordinated report wave must not silence a real announcement).

Data: SQLite (data/app/darg.db, git-ignored). Passwords: salted scrypt. Sessions: random tokens, only their SHA-256
stored. The first administrator is created from the command line (no secret in the code):

    python -m src.api.community create-admin --email you@example.com --name "Ad Soyad"
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from src.api import limits

DB_PATH = Path("data/app/darg.db")
SESSION_DAYS = 30
MAX_VALID_DAYS = 30
LOGIN_WINDOW_SECONDS = 15 * 60
LOGIN_MAX_FAILURES = 5

PostKind = Literal["duyuru", "toplanma", "yardim", "saglik", "uyari"]
KIND_NAMES = {"duyuru": "Duyuru", "toplanma": "Toplanma alanı", "yardim": "Yardım dağıtımı", "saglik": "Sağlık",
              "uyari": "Uyarı"}
OrgType = Literal["kamu", "belediye", "afad", "saglik", "stk", "diger"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('kurum', 'yonetici')),
    status TEXT NOT NULL CHECK (status IN ('beklemede', 'aktif', 'reddedildi', 'askida')),
    org_name TEXT NOT NULL,
    org_type TEXT NOT NULL,
    contact_name TEXT NOT NULL,
    phone TEXT NOT NULL DEFAULT '',
    proof TEXT NOT NULL DEFAULT '',
    trusted INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    reviewed_at TEXT,
    review_note TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    kind TEXT NOT NULL,
    province TEXT,
    district TEXT,
    valid_until TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('beklemede', 'yayinda', 'reddedildi', 'kaldirildi', 'gizlendi')),
    created_at TEXT NOT NULL,
    published_at TEXT,
    review_note TEXT
);
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY,
    post_id INTEGER NOT NULL REFERENCES posts(id),
    device_id TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE (post_id, device_id)
);
CREATE INDEX IF NOT EXISTS posts_feed ON posts (status, valid_until);
CREATE TABLE IF NOT EXISTS usage (
    day TEXT NOT NULL,
    name TEXT NOT NULL,
    count INTEGER NOT NULL,
    PRIMARY KEY (day, name)
);
"""

_init_lock = threading.Lock()
_initialized: set[str] = set()
_failed_logins: dict[str, list[float]] = {}


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    with _init_lock:
        if str(DB_PATH) not in _initialized:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            setup = sqlite3.connect(DB_PATH)  # `with` on a sqlite3 connection commits but does not close it
            try:
                setup.executescript(SCHEMA)
            finally:
                setup.close()
            _initialized.add(str(DB_PATH))
    db = sqlite3.connect(DB_PATH, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    try:
        yield db
        db.commit()
    finally:
        db.close()


# --- passwords and sessions -------------------------------------------------------------------------------------

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt, digest = stored.split("$")
    except ValueError:
        return False
    candidate = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt), n=2**14, r=8, p=1)
    return hmac.compare_digest(candidate.hex(), digest)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def current_user(authorization: str | None = Header(default=None)) -> sqlite3.Row:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Giriş yapmanız gerekiyor.")
    with connect() as db:
        row = db.execute(
            "SELECT users.* FROM sessions JOIN users ON users.id = sessions.user_id "
            "WHERE token_hash = ? AND expires_at > ?", (_token_hash(authorization[7:]), iso(now()))).fetchone()
    if row is None or row["status"] != "aktif":
        raise HTTPException(status_code=401, detail="Oturumunuzun süresi doldu; tekrar giriş yapın.")
    return row


def current_admin(user: sqlite3.Row = Depends(current_user)) -> sqlite3.Row:
    if user["role"] != "yonetici":
        raise HTTPException(status_code=403, detail="Bu işlem yalnız yöneticiler içindir.")
    return user


# --- request bodies ---------------------------------------------------------------------------------------------

class Application(BaseModel):
    email: str = Field(min_length=5, max_length=200, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    sifre: str = Field(min_length=8, max_length=200)
    kurum_adi: str = Field(min_length=2, max_length=200)
    kurum_turu: OrgType
    yetkili_adi: str = Field(min_length=2, max_length=200)
    telefon: str = Field(default="", max_length=40)
    kanit: str = Field(min_length=5, max_length=1000,
                       description="Kurumla bağı gösteren bilgi: resmî web sayfası, görev yeri, kurum e-postası…")


class Login(BaseModel):
    email: str = Field(max_length=200)
    sifre: str = Field(max_length=200)


class NewPost(BaseModel):
    baslik: str = Field(min_length=3, max_length=120)
    metin: str = Field(min_length=5, max_length=2000)
    tur: PostKind
    il: str | None = Field(default=None, max_length=40)
    ilce: str | None = Field(default=None, max_length=60)
    gecerlilik_saat: int = Field(default=72, ge=1, le=MAX_VALID_DAYS * 24)


class Report(BaseModel):
    cihaz_id: str = Field(min_length=8, max_length=64)
    neden: str = Field(default="", max_length=500)


class Decision(BaseModel):
    karar: Literal["onayla", "reddet", "askiya_al"]
    guvenilir: bool = False
    not_: str | None = Field(default=None, alias="not", max_length=500)


class PostDecision(BaseModel):
    karar: Literal["yayinla", "reddet", "kaldir"]
    not_: str | None = Field(default=None, alias="not", max_length=500)


class Trust(BaseModel):
    guvenilir: bool


# --- shaping rows for the app ------------------------------------------------------------------------------------

def account_out(row: sqlite3.Row) -> dict:
    return {"id": row["id"], "email": row["email"], "rol": row["role"], "durum": row["status"],
            "kurum_adi": row["org_name"], "kurum_turu": row["org_type"], "yetkili_adi": row["contact_name"],
            "telefon": row["phone"], "kanit": row["proof"], "guvenilir": bool(row["trusted"]),
            "olusturma": row["created_at"], "inceleme_notu": row["review_note"]}


def post_out(row: sqlite3.Row) -> dict:
    keys = row.keys()
    return {"id": row["id"], "baslik": row["title"], "metin": row["body"], "tur": row["kind"],
            "tur_adi": KIND_NAMES.get(row["kind"], row["kind"]), "il": row["province"], "ilce": row["district"],
            "gecerlilik": row["valid_until"], "durum": row["status"], "olusturma": row["created_at"],
            "yayin": row["published_at"], "inceleme_notu": row["review_note"],
            "kurum_adi": row["org_name"] if "org_name" in keys else None,
            "kurum_guvenilir": bool(row["trusted"]) if "trusted" in keys else None,
            "sikayet": row["report_count"] if "report_count" in keys else None}


POST_SELECT = ("SELECT posts.*, users.org_name, users.trusted, "
               "(SELECT COUNT(*) FROM reports WHERE reports.post_id = posts.id) AS report_count "
               "FROM posts JOIN users ON users.id = posts.user_id ")

# Posts whose reports still need a decision ('gizlendi' only exists in databases from before reports stopped hiding).
def reported_open(table: str = "posts") -> str:
    return f"{table}.status IN ('yayinda', 'gizlendi') AND {table}.id IN (SELECT post_id FROM reports)"

router = APIRouter()


# --- accounts ----------------------------------------------------------------------------------------------------

@router.post("/hesap/basvuru", status_code=201, dependencies=[Depends(limits.APPLICATION)])
def apply(application: Application) -> dict:
    email = application.email.strip().lower()
    with connect() as db:
        if db.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
            raise HTTPException(status_code=409, detail="Bu e-posta ile zaten bir başvuru ya da hesap var.")
        db.execute(
            "INSERT INTO users (email, password_hash, role, status, org_name, org_type, contact_name, phone, proof, "
            "created_at) VALUES (?, ?, 'kurum', 'beklemede', ?, ?, ?, ?, ?, ?)",
            (email, hash_password(application.sifre), application.kurum_adi.strip(), application.kurum_turu,
             application.yetkili_adi.strip(), application.telefon.strip(), application.kanit.strip(), iso(now())))
    return {"durum": "beklemede",
            "mesaj": "Başvurunuz alındı. Yönetici kurumunuzu doğruladıktan sonra giriş yapabilirsiniz."}


@router.post("/hesap/giris", dependencies=[Depends(limits.LOGIN)])
def login(credentials: Login, request: Request) -> dict:
    email = credentials.email.strip().lower()
    # Failures count per e-mail and client address: someone else's wrong passwords cannot lock the owner out.
    attempt = f"{email}|{limits.client_address(request)}"
    recent = [t for t in _failed_logins.get(attempt, []) if time.time() - t < LOGIN_WINDOW_SECONDS]
    if len(recent) >= LOGIN_MAX_FAILURES:
        raise HTTPException(status_code=429, detail="Çok fazla hatalı deneme. Lütfen 15 dakika sonra tekrar deneyin.")
    with connect() as db:
        row = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row is None or not verify_password(credentials.sifre, row["password_hash"]):
            _failed_logins[attempt] = recent + [time.time()]
            raise HTTPException(status_code=401, detail="E-posta ya da şifre hatalı.")
        _failed_logins.pop(attempt, None)
        if row["status"] == "beklemede":
            raise HTTPException(status_code=403, detail="Başvurunuz henüz incelenmedi.")
        if row["status"] != "aktif":
            raise HTTPException(status_code=403, detail="Bu hesap kullanıma kapalı. " + (row["review_note"] or ""))
        token = secrets.token_urlsafe(32)
        db.execute("DELETE FROM sessions WHERE expires_at <= ?", (iso(now()),))
        db.execute("INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                   (_token_hash(token), row["id"], iso(now() + timedelta(days=SESSION_DAYS))))
    return {"token": token, "hesap": account_out(row)}


@router.post("/hesap/cikis")
def logout(authorization: str | None = Header(default=None)) -> dict:
    if authorization and authorization.startswith("Bearer "):
        with connect() as db:
            db.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(authorization[7:]),))
    return {"durum": "cikis"}


@router.get("/hesap/ben")
def me(user: sqlite3.Row = Depends(current_user)) -> dict:
    return account_out(user)


# --- feed and posts ----------------------------------------------------------------------------------------------

@router.get("/akis")
def feed(il: str | None = None, limit: int = 50) -> dict:
    query = POST_SELECT + "WHERE posts.status = 'yayinda' AND posts.valid_until > ?"
    params: list = [iso(now())]
    if il:
        query += " AND (posts.province = ? OR posts.province IS NULL)"  # province-wide plus nationwide posts
        params.append(il)
    query += " ORDER BY posts.published_at DESC LIMIT ?"
    params.append(max(1, min(limit, 200)))
    with connect() as db:
        rows = db.execute(query, params).fetchall()
    return {"gonderiler": [post_out(row) for row in rows]}


@router.post("/gonderiler", status_code=201)
def create_post(post: NewPost, user: sqlite3.Row = Depends(current_user)) -> dict:
    published = bool(user["trusted"]) or user["role"] == "yonetici"
    created = now()
    with connect() as db:
        cursor = db.execute(
            "INSERT INTO posts (user_id, title, body, kind, province, district, valid_until, status, created_at, "
            "published_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user["id"], post.baslik.strip(), post.metin.strip(), post.tur, (post.il or "").strip() or None,
             (post.ilce or "").strip() or None, iso(created + timedelta(hours=post.gecerlilik_saat)),
             "yayinda" if published else "beklemede", iso(created), iso(created) if published else None))
        row = db.execute(POST_SELECT + "WHERE posts.id = ?", (cursor.lastrowid,)).fetchone()
    return {"gonderi": post_out(row),
            "mesaj": "Gönderiniz yayınlandı." if published else "Gönderiniz yönetici onayından sonra yayınlanacak."}


@router.get("/gonderiler/benim")
def my_posts(user: sqlite3.Row = Depends(current_user)) -> dict:
    with connect() as db:
        rows = db.execute(POST_SELECT + "WHERE posts.user_id = ? ORDER BY posts.created_at DESC LIMIT 200",
                          (user["id"],)).fetchall()
    return {"gonderiler": [post_out(row) for row in rows]}


@router.delete("/gonderiler/{post_id}")
def delete_own_post(post_id: int, user: sqlite3.Row = Depends(current_user)) -> dict:
    with connect() as db:
        updated = db.execute("UPDATE posts SET status = 'kaldirildi' WHERE id = ? AND user_id = ?",
                             (post_id, user["id"])).rowcount
    if not updated:
        raise HTTPException(status_code=404, detail="Gönderi bulunamadı.")
    return {"durum": "kaldirildi"}


@router.post("/gonderiler/{post_id}/sikayet", dependencies=[Depends(limits.REPORT)])
def report_post(post_id: int, report: Report) -> dict:
    with connect() as db:
        if not db.execute("SELECT 1 FROM posts WHERE id = ? AND status = 'yayinda'", (post_id,)).fetchone():
            raise HTTPException(status_code=404, detail="Gönderi bulunamadı.")
        db.execute("INSERT OR IGNORE INTO reports (post_id, device_id, reason, created_at) VALUES (?, ?, ?, ?)",
                   (post_id, report.cihaz_id, report.neden.strip(), iso(now())))
    return {"durum": "alindi"}  # reports never take a post down; only an administrator decides


# --- administration ----------------------------------------------------------------------------------------------

@router.get("/yonetim/ozet")
def admin_summary(_: sqlite3.Row = Depends(current_admin)) -> dict:
    with connect() as db:
        count = lambda sql, *args: db.execute(sql, args).fetchone()[0]  # noqa: E731
        return {"bekleyen_basvuru": count("SELECT COUNT(*) FROM users WHERE status = 'beklemede'"),
                "bekleyen_gonderi": count("SELECT COUNT(*) FROM posts WHERE status = 'beklemede'"),
                "gizlenen_gonderi": count("SELECT COUNT(*) FROM posts WHERE status = 'gizlendi'"),
                "sikayetli_gonderi": count(f"SELECT COUNT(*) FROM posts WHERE {reported_open()}"),
                "aktif_kurum": count("SELECT COUNT(*) FROM users WHERE role = 'kurum' AND status = 'aktif'"),
                "yayindaki_gonderi": count("SELECT COUNT(*) FROM posts WHERE status = 'yayinda' AND valid_until > ?",
                                           iso(now()))}


@router.get("/yonetim/kurumlar")
def admin_organisations(durum: str | None = None, _: sqlite3.Row = Depends(current_admin)) -> dict:
    query, params = "SELECT * FROM users WHERE role = 'kurum'", []
    if durum:
        query += " AND status = ?"
        params.append(durum)
    with connect() as db:
        rows = db.execute(query + " ORDER BY created_at DESC", params).fetchall()
    return {"kurumlar": [account_out(row) for row in rows]}


@router.post("/yonetim/kurumlar/{user_id}")
def admin_decide_organisation(user_id: int, decision: Decision, _: sqlite3.Row = Depends(current_admin)) -> dict:
    status = {"onayla": "aktif", "reddet": "reddedildi", "askiya_al": "askida"}[decision.karar]
    with connect() as db:
        updated = db.execute(
            "UPDATE users SET status = ?, trusted = ?, reviewed_at = ?, review_note = ? WHERE id = ? AND role = 'kurum'",
            (status, int(decision.guvenilir and status == "aktif"), iso(now()), decision.not_, user_id)).rowcount
        if status != "aktif":
            db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    if not updated:
        raise HTTPException(status_code=404, detail="Kurum bulunamadı.")
    return {"durum": status}


@router.post("/yonetim/kurumlar/{user_id}/guven")
def admin_trust(user_id: int, trust: Trust, _: sqlite3.Row = Depends(current_admin)) -> dict:
    with connect() as db:
        updated = db.execute("UPDATE users SET trusted = ? WHERE id = ? AND role = 'kurum' AND status = 'aktif'",
                             (int(trust.guvenilir), user_id)).rowcount
    if not updated:
        raise HTTPException(status_code=404, detail="Aktif kurum bulunamadı.")
    return {"guvenilir": trust.guvenilir}


def _with_reasons(db: sqlite3.Connection, rows: list[sqlite3.Row]) -> list[dict]:
    return [{**post_out(row), "sikayet_nedenleri": [r["reason"] for r in db.execute(
        "SELECT reason FROM reports WHERE post_id = ? AND reason != '' ORDER BY id", (row["id"],))]} for row in rows]


@router.get("/yonetim/gonderiler")
def admin_posts(durum: str = "beklemede", _: sqlite3.Row = Depends(current_admin)) -> dict:
    with connect() as db:
        rows = db.execute(POST_SELECT + "WHERE posts.status = ? ORDER BY posts.created_at DESC LIMIT 200",
                          (durum,)).fetchall()
        return {"gonderiler": _with_reasons(db, rows)}


@router.get("/yonetim/sikayetler")
def admin_reported(_: sqlite3.Row = Depends(current_admin)) -> dict:
    """Every post with open reports, most reported first.
    Publishing a post again ("yayinla") settles its reports; removing it takes it down."""
    with connect() as db:
        rows = db.execute(POST_SELECT + f"WHERE {reported_open()} "
                          "ORDER BY report_count DESC, posts.created_at DESC LIMIT 200").fetchall()
        return {"gonderiler": _with_reasons(db, rows)}


@router.post("/yonetim/gonderiler/{post_id}")
def admin_decide_post(post_id: int, decision: PostDecision, _: sqlite3.Row = Depends(current_admin)) -> dict:
    status = {"yayinla": "yayinda", "reddet": "reddedildi", "kaldir": "kaldirildi"}[decision.karar]
    with connect() as db:
        if status == "yayinda":  # published after review: reports so far are settled
            db.execute("DELETE FROM reports WHERE post_id = ?", (post_id,))
        updated = db.execute(
            "UPDATE posts SET status = ?, review_note = ?, published_at = COALESCE(published_at, ?) WHERE id = ?",
            (status, decision.not_, iso(now()) if status == "yayinda" else None, post_id)).rowcount
    if not updated:
        raise HTTPException(status_code=404, detail="Gönderi bulunamadı.")
    return {"durum": status}


# --- command line: first administrator ---------------------------------------------------------------------------

def create_admin(email: str, name: str, password: str) -> None:
    with connect() as db:
        db.execute(
            "INSERT INTO users (email, password_hash, role, status, org_name, org_type, contact_name, created_at, "
            "trusted) VALUES (?, ?, 'yonetici', 'aktif', 'DARG Yönetim', 'diger', ?, ?, 1) "
            "ON CONFLICT(email) DO UPDATE SET password_hash = excluded.password_hash, role = 'yonetici', "
            "status = 'aktif', trusted = 1",
            (email.strip().lower(), hash_password(password), name, iso(now())))


def main() -> None:
    parser = argparse.ArgumentParser(description="DARG yönetici hesabı oluşturur ya da şifresini yeniler.")
    sub = parser.add_subparsers(dest="command", required=True)
    admin = sub.add_parser("create-admin")
    admin.add_argument("--email", required=True)
    admin.add_argument("--name", required=True)
    args = parser.parse_args()
    password = getpass.getpass("Şifre (en az 8 karakter): ")
    if len(password) < 8 or password != getpass.getpass("Şifre (tekrar): "):
        raise SystemExit("Şifre en az 8 karakter olmalı ve iki giriş aynı olmalı.")
    create_admin(args.email, args.name, password)
    print(f"Yönetici hazır: {args.email}")


if __name__ == "__main__":
    main()
