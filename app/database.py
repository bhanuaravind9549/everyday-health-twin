"""SQLite schema and helpers for users, daily entries, and reminder prefs."""
import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "health_twin.db"

# Columns we collect from the user each day (subset of full model features)
CHECKIN_FIELDS = [
    "sleep_hours",
    "sleep_quality",
    "mood_score",
    "stress_level",
    "steps_count",
    "exercise_minutes",
    "diet_quality",
    "screen_time_hours",
    "work_hours",
    "energy_level",
]


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            sleep_hours REAL,
            sleep_quality REAL,
            mood_score REAL,
            stress_level REAL,
            steps_count REAL,
            exercise_minutes REAL,
            diet_quality REAL,
            screen_time_hours REAL,
            work_hours REAL,
            energy_level REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, date),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS reminder_prefs (
            user_id INTEGER PRIMARY KEY,
            remind_at TEXT NOT NULL,
            enabled INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()


def create_user(name: str, email: str = "") -> int:
    conn = get_conn()
    cur = conn.execute("INSERT INTO users (name, email) VALUES (?, ?)", (name.strip(), email.strip()))
    conn.commit()
    uid = cur.lastrowid
    conn.close()
    return uid


def get_user(user_id: int) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT id, name, email, created_at FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "name": row[1], "email": row[2] or "", "created_at": row[3]}


def upsert_entry(user_id: int, day: date, **kwargs) -> None:
    conn = get_conn()
    day_str = day.isoformat()
    existing = conn.execute("SELECT id FROM entries WHERE user_id = ? AND date = ?", (user_id, day_str)).fetchone()
    fields = {f: kwargs[f] for f in CHECKIN_FIELDS if f in kwargs and kwargs[f] is not None}
    if existing:
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(
            f"UPDATE entries SET {set_clause} WHERE user_id = ? AND date = ?",
            list(fields.values()) + [user_id, day_str],
        )
    else:
        cols = ["user_id", "date"] + list(fields.keys())
        placeholders = ", ".join("?" * len(cols))
        conn.execute(f"INSERT INTO entries ({', '.join(cols)}) VALUES ({placeholders})", [user_id, day_str] + list(fields.values()))
    conn.commit()
    conn.close()


def get_entries(user_id: int, limit_days: int = 365) -> list[dict]:
    conn = get_conn()
    cutoff = (date.today() - __import__("datetime").timedelta(days=limit_days)).isoformat()
    rows = conn.execute(
        "SELECT date, sleep_hours, sleep_quality, mood_score, stress_level, steps_count, "
        "exercise_minutes, diet_quality, screen_time_hours, work_hours, energy_level FROM entries "
        "WHERE user_id = ? AND date >= ? ORDER BY date DESC",
        (user_id, cutoff),
    ).fetchall()
    conn.close()
    return [
        {
            "date": r[0],
            "sleep_hours": r[1],
            "sleep_quality": r[2],
            "mood_score": r[3],
            "stress_level": r[4],
            "steps_count": r[5],
            "exercise_minutes": r[6],
            "diet_quality": r[7],
            "screen_time_hours": r[8],
            "work_hours": r[9],
            "energy_level": r[10],
        }
        for r in rows
    ]


def has_logged_today(user_id: int) -> bool:
    conn = get_conn()
    n = conn.execute("SELECT 1 FROM entries WHERE user_id = ? AND date = ?", (user_id, date.today().isoformat())).fetchone()
    conn.close()
    return n is not None


def set_reminder(user_id: int, remind_at: str, enabled: bool = True) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO reminder_prefs (user_id, remind_at, enabled) VALUES (?, ?, ?) "
        "ON CONFLICT(user_id) DO UPDATE SET remind_at=?, enabled=?",
        (user_id, remind_at, 1 if enabled else 0, remind_at, 1 if enabled else 0),
    )
    conn.commit()
    conn.close()


def get_reminder(user_id: int) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT remind_at, enabled FROM reminder_prefs WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return {"remind_at": row[0], "enabled": bool(row[1])}
