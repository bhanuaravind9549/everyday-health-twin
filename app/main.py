"""
Everyday Health Twin – FastAPI backend.
Manual check-in, reminders, insight (LLM or rule-based), and demo mode.
"""
import os
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.database import (
    CHECKIN_FIELDS,
    create_user,
    get_entries,
    get_reminder,
    get_user,
    has_logged_today,
    init_db,
    set_reminder,
    upsert_entry,
)

app = FastAPI(title="Everyday Health Twin")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

FRONTEND = ROOT / "app" / "static"


# ---------- Pydantic models ----------
class RegisterBody(BaseModel):
    name: str
    email: str


class CheckinBody(BaseModel):
    date: str | None = None  # yyyy-mm-dd, default today
    sleep_hours: float | None = None
    sleep_quality: float | None = None
    mood_score: float | None = None
    stress_level: float | None = None
    steps_count: float | None = None
    exercise_minutes: float | None = None
    diet_quality: float | None = None
    screen_time_hours: float | None = None
    work_hours: float | None = None
    energy_level: float | None = None


class RemindBody(BaseModel):
    remind_at: str  # e.g. "20:00"
    enabled: bool = True


# ---------- Startup ----------
@app.on_event("startup")
def startup():
    init_db()


# ---------- Routes ----------
@app.get("/")
def root():
    if FRONTEND.exists() and (FRONTEND / "index.html").exists():
        return FileResponse(FRONTEND / "index.html")
    return {"app": "Everyday Health Twin", "docs": "/docs"}


@app.post("/api/register")
def register(body: RegisterBody):
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="Name required")
    if not body.email.strip():
        raise HTTPException(status_code=400, detail="Email required for reminders")
    user_id = create_user(body.name, body.email.strip())
    user = get_user(user_id)
    return {"user_id": user_id, "user": user}


@app.get("/api/user/{user_id}")
def user_info(user_id: int):
    u = get_user(user_id)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    logged_today = has_logged_today(user_id)
    reminder = get_reminder(user_id)
    return {"user": u, "logged_today": logged_today, "reminder": reminder}


@app.post("/api/checkin/{user_id}")
def checkin(user_id: int, body: CheckinBody):
    if get_user(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    day = date.today() if not body.date else date.fromisoformat(body.date)
    kwargs = {f: getattr(body, f) for f in CHECKIN_FIELDS if getattr(body, f) is not None}
    upsert_entry(user_id, day, **kwargs)
    return {"ok": True, "date": day.isoformat()}


@app.get("/api/entries/{user_id}")
def entries(user_id: int, days: int = 365):
    if get_user(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    return get_entries(user_id, limit_days=days)


@app.post("/api/remind/{user_id}")
def remind(user_id: int, body: RemindBody):
    if get_user(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    set_reminder(user_id, body.remind_at, body.enabled)
    return {"ok": True, "remind_at": body.remind_at, "enabled": body.enabled}


@app.get("/api/insight/{user_id}")
def insight(user_id: int, last_n_days: int = 14):
    if get_user(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    entries_list = get_entries(user_id, limit_days=last_n_days)
    if not entries_list:
        raise HTTPException(status_code=400, detail="No entries yet. Log at least a few days to get an insight.")
    # Build summary and generate insight (rule-based or LLM)
    summary = _entries_to_summary(entries_list, user_id)
    text = _generate_insight(summary)
    return {"insight": text, "summary": summary}


def _entries_to_summary(entries_list: list[dict], user_id: int) -> dict:
    df = pd.DataFrame(entries_list)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    n = len(df)
    stats = {}
    for c in ["sleep_hours", "sleep_quality", "mood_score", "stress_level", "steps_count", "exercise_minutes", "diet_quality", "screen_time_hours", "work_hours", "energy_level"]:
        if c in df.columns:
            s = df[c].dropna()
            if len(s):
                stats[c] = round(float(s.mean()), 2)
    actuals = {}
    for c in ["mood_score", "stress_level"]:
        if c in df.columns and len(df[c].dropna()):
            actuals[c] = round(float(df[c].iloc[-1]), 2)
            actuals[c + "_avg"] = round(float(df[c].mean()), 2)
    # Health twin index: simple composite (e.g. mood*10 + (10-stress)*5 + sleep_quality*2)
    if "mood_score" in df.columns and "stress_level" in df.columns and "sleep_quality" in df.columns:
        m = df["mood_score"].fillna(5).iloc[-1]
        s = df["stress_level"].fillna(5).iloc[-1]
        sq = df["sleep_quality"].fillna(5).iloc[-1]
        hti = m * 5 + (10 - s) * 3 + sq * 2
        actuals["health_twin_index"] = round(hti, 2)
    warnings = []
    good = []
    if stats.get("sleep_hours") is not None:
        if stats["sleep_hours"] < 6:
            warnings.append("Consistently low sleep — try to get 7+ hours.")
        elif stats["sleep_hours"] >= 7:
            good.append("Good sleep habits.")
    if stats.get("exercise_minutes") is not None:
        if stats["exercise_minutes"] < 20:
            warnings.append("More movement could help mood and stress.")
        elif stats["exercise_minutes"] >= 30:
            good.append("Solid exercise routine.")
    if stats.get("stress_level") is not None and stats["stress_level"] >= 6:
        warnings.append("Stress is often high — small breaks and boundaries help.")
    if stats.get("mood_score") is not None and stats["mood_score"] < 4:
        warnings.append("Mood has been low — sleep and exercise can help.")
    date_range = f"{df['date'].min().date()} to {df['date'].max().date()}" if n else ""
    return {
        "user_id": user_id,
        "n_days": n,
        "date_range": date_range,
        "actuals": actuals,
        "predictions": {},
        "stats": stats,
        "warnings": warnings,
        "good": good,
    }


def _generate_insight(summary: dict) -> str:
    """Rule-based insight so it works without OpenAI. Optionally call LLM if key set."""
    try:
        from insights import get_insight_for_user
        if os.environ.get("OPENAI_API_KEY"):
            return get_insight_for_user(summary)
    except Exception:
        pass
    # Rule-based fallback
    lines = []
    if summary.get("good"):
        lines.append("You're doing well on: " + ", ".join(summary["good"]) + ".")
    if summary.get("warnings"):
        lines.append("Worth paying attention to: " + "; ".join(summary["warnings"]))
    if not lines:
        lines.append("Keep logging daily. After a few more days we'll have a clearer picture and can suggest small tweaks.")
    lines.append("Small steps that often help: protect sleep, move a bit every day, and take short breaks when stress is high.")
    return " ".join(lines)


# ---------- Demo (sample data) ----------
SAMPLE_INSIGHT = """Your last two weeks show a solid foundation: sleep is in a good range and your stress is manageable. A few small tweaks could nudge things even better.

Consider building in 20–30 minutes of movement most days—even a brisk walk counts. It’s one of the most reliable ways to support mood and energy. If screen time creeps up, try a short wind-down without devices before bed; it often improves sleep quality.

Keep logging daily so your Health Twin can spot patterns and give you more tailored suggestions over time."""


@app.get("/api/demo")
def demo():
    """Return sample data for 'Try with sample data'.

    Prefer a small predictions_sample.csv checked into the repo so GitHub stays light.
    If that is missing, fall back to a larger local predictions.csv or df_sample_5k.csv
    if present on the machine.
    """
    fallback = {
        "user": {"id": "demo", "name": "Sample User"},
        "entries": [],
        "insight": SAMPLE_INSIGHT,
        "summary": {},
    }
    try:
        # Try small, repo-friendly sample first; then optional local full data.
        candidates = ["predictions_sample.csv", "predictions.csv", "df_sample_5k.csv"]
        pred_path = None
        for name in candidates:
            p = ROOT / "models" / name
            if p.exists():
                pred_path = p
                break
        if pred_path is None:
            return fallback

        df = pd.read_csv(pred_path, nrows=100_000)
        if "user_id" not in df.columns or "date" not in df.columns:
            return fallback

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        # Users with at least 7 days of data (so we can show trends)
        counts = df.groupby("user_id").size()
        eligible = counts[counts >= 7].index.tolist()
        if not eligible:
            return fallback

        uid = random.choice(eligible)
        u = df[df["user_id"] == uid].sort_values("date").tail(30)

        cols = [
            "date",
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
        cols = [c for c in cols if c in u.columns]
        entries_list = u[cols].copy()
        entries_list["date"] = entries_list["date"].dt.strftime("%Y-%m-%d")
        entries_list = entries_list.to_dict("records")

        summary = _entries_to_summary(entries_list, f"demo_{uid}")
        text = _generate_insight(summary)
        return {
            "user": {"id": f"demo_{uid}", "name": f"Sample User #{uid}"},
            "entries": entries_list,
            "insight": text,
            "summary": summary,
        }
    except Exception:
        return fallback
