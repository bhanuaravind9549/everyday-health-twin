"""
LLM-powered health insights: suggestions, warnings for bad lifestyle patterns, and encouragement.
Uses past + predicted mood, stress, health_twin_index plus lifestyle features to generate
personalized text via an LLM (OpenAI).

Usage:
  Set OPENAI_API_KEY in environment, then:
  python insights.py --input models/predictions.csv --output models/insights.json
  python insights.py --input models/predictions.csv --user-id 42 --output models/insight_42.json
  python insights.py --input models/predictions.csv --limit 5 --output models/insights_sample.json
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent

# Lifestyle columns we summarize and use for pattern detection (must exist or be ignored)
LIFESTYLE_COLS = [
    "sleep_hours",
    "sleep_quality",
    "exercise_minutes",
    "steps_count",
    "screen_time_hours",
    "diet_quality",
    "caffeine_mg",
    "alcohol_units",
    "work_hours",
    "outdoor_time_minutes",
    "social_interactions",
    "stress_level",
    "mood_score",
    "energy_level",
]
PRED_COLS = ["mood_score_pred", "stress_level_pred", "health_twin_index_pred"]
TARGET_LABELS = {"mood_score": "Mood (higher better)", "stress_level": "Stress (lower better)", "health_twin_index": "Health Twin Index (higher better)"}


def _normalize_user_id(value):
    """Normalize user_id for robust matching across int/float/string inputs."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except ValueError:
        return s


def load_predictions(path: Path) -> pd.DataFrame:
    """Load predictions CSV and parse dates."""
    df = pd.read_csv(path, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def get_user_df(df: pd.DataFrame, user_id, last_n_days: int = 14) -> pd.DataFrame:
    """Get one user's rows, sorted by date, limited to last_n_days (by calendar)."""
    req_id = _normalize_user_id(user_id)
    u = df[df["user_id"].map(_normalize_user_id) == req_id].copy()
    if u.empty:
        return u
    u = u.sort_values("date").dropna(subset=["date"])
    if u.empty:
        return u
    max_date = u["date"].max()
    cutoff = max_date - pd.Timedelta(days=last_n_days)
    return u[u["date"] >= cutoff]


def compute_patterns(udf: pd.DataFrame) -> dict:
    """Compute simple pattern flags and stats from user's recent data."""
    if udf.empty or len(udf) == 0:
        return {"warnings": [], "good": [], "stats": {}}

    stats = {}
    warnings = []
    good = []

    # Averages over available columns
    for col in LIFESTYLE_COLS:
        if col not in udf.columns:
            continue
        s = udf[col].dropna()
        if len(s) == 0:
            continue
        avg = float(s.mean())
        stats[col] = round(avg, 2)
        if col == "sleep_hours":
            if avg < 6:
                warnings.append("Consistently low sleep (under 6 hours) — increases stress and hurts mood and long-term health.")
            elif avg >= 7:
                good.append("Good average sleep (7+ hours).")
        elif col == "sleep_quality":
            if avg < 4:
                warnings.append("Low sleep quality — consider wind-down routine and limiting screens before bed.")
            elif avg >= 6:
                good.append("Sleep quality is solid.")
        elif col == "exercise_minutes":
            if avg < 20:
                warnings.append("Very little exercise — even 20–30 min daily can improve mood and stress.")
            elif avg >= 30:
                good.append("Regular exercise — great for mood and stress.")
        elif col == "steps_count":
            if avg < 4000:
                warnings.append("Low step count — moving more can improve energy and mood.")
            elif avg >= 7000:
                good.append("Good daily steps — keep it up.")
        elif col == "screen_time_hours":
            if avg > 8:
                warnings.append("High screen time — can worsen sleep and stress; try breaks and evening limits.")
            elif avg <= 5:
                good.append("Reasonable screen time.")
        elif col == "diet_quality":
            if avg < 4:
                warnings.append("Diet quality is low — nutrition strongly affects mood and energy.")
            elif avg >= 6:
                good.append("Diet quality is good.")
        elif col == "caffeine_mg":
            if avg > 400:
                warnings.append("High caffeine — can disrupt sleep and increase anxiety.")
        elif col == "alcohol_units":
            if avg > 2:
                warnings.append("Regular alcohol above guidelines — can affect sleep and mood.")
        elif col == "work_hours":
            if avg > 9:
                warnings.append("Long work hours — linked to higher stress and lower mood; try boundaries.")
        elif col == "outdoor_time_minutes":
            if avg < 15:
                warnings.append("Little time outdoors — even 15–20 min can help mood and stress.")
            elif avg >= 30:
                good.append("Decent outdoor time — helpful for mood.")
        elif col == "social_interactions":
            if avg < 2:
                warnings.append("Few social interactions — connection supports mood and stress.")
            elif avg >= 4:
                good.append("Good social connection.")
        elif col == "stress_level":
            if avg >= 6:
                warnings.append("Stress is often high — consider relaxation, boundaries, or support.")
            elif avg <= 4:
                good.append("Stress levels are manageable.")
        elif col == "mood_score":
            if avg < 4:
                warnings.append("Mood has been low — lifestyle changes and support can help.")
            elif avg >= 6:
                good.append("Mood has been generally good.")

    return {"warnings": warnings, "good": good, "stats": stats}


def build_user_summary(udf: pd.DataFrame, user_id) -> dict:
    """Build a summary dict for one user: recent stats, patterns, and predictions."""
    preds = {}
    for c in PRED_COLS:
        if c in udf.columns:
            s = udf[c].dropna()
            if len(s) > 0:
                preds[c] = round(float(s.iloc[-1]), 2)
            if len(s) > 1:
                preds[c + "_avg_recent"] = round(float(s.mean()), 2)

    actuals = {}
    for col in ["mood_score", "stress_level", "health_twin_index"]:
        if col in udf.columns:
            s = udf[col].dropna()
            if len(s) > 0:
                actuals[col] = round(float(s.iloc[-1]), 2)
                actuals[col + "_avg"] = round(float(s.mean()), 2)

    patterns = compute_patterns(udf)
    n_days = len(udf)
    date_range = f"{udf['date'].min().date()} to {udf['date'].max().date()}" if "date" in udf.columns and len(udf) else ""

    return {
        "user_id": user_id,
        "n_days": n_days,
        "date_range": date_range,
        "actuals": actuals,
        "predictions": preds,
        "stats": patterns["stats"],
        "warnings": patterns["warnings"],
        "good": patterns["good"],
    }


def summary_to_text(summary: dict) -> str:
    """Turn summary dict into a concise text block for the LLM."""
    lines = [
        f"User ID: {summary['user_id']}",
        f"Data: last {summary['n_days']} days ({summary['date_range']}).",
        "",
        "Recent averages (lifestyle):",
    ]
    for k, v in summary.get("stats", {}).items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Actual outcomes (recent):")
    for k, v in summary.get("actuals", {}).items():
        if not k.endswith("_avg"):
            lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Predicted values (from model):")
    for k, v in summary.get("predictions", {}).items():
        if not k.endswith("_avg_recent"):
            lines.append(f"  - {k}: {v}")
    if summary.get("warnings"):
        lines.append("")
        lines.append("Detected concerns:")
        for w in summary["warnings"]:
            lines.append(f"  - {w}")
    if summary.get("good"):
        lines.append("")
        lines.append("Positive patterns:")
        for g in summary["good"]:
            lines.append(f"  - {g}")
    return "\n".join(lines)


def build_prompt(summary: dict) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the LLM."""
    system = """You are a supportive, evidence-based health coach. Your job is to write a short, personalized health insight for one person based on their lifestyle data and predicted mood, stress, and health scores.

Guidelines:
- Suggest 2–4 concrete, actionable steps to improve mood score and Health Twin Index and to lower stress (e.g. sleep, exercise, diet, screen time, boundaries).
- If there are clear bad lifestyle patterns (e.g. poor sleep, no exercise, high stress), warn them in a caring way and briefly explain how continuing these habits can affect their mood, stress, and long-term health.
- If something is going well, encourage them to keep it up.
- Keep the tone warm and concise (about 150–250 words). Write in second person ("you"). Do not mention "User ID" or raw numbers in the final message; use them only to tailor the message."""

    text = summary_to_text(summary)
    user = f"""Here is this person's summarized data and our model's predictions:\n\n{text}\n\nWrite the personalized health insight (suggestions, any warnings about bad habits and long-term effects, and encouragement where appropriate)."""

    return system, user


def call_openai(system: str, user: str, api_key: str | None = None, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI Chat Completions and return the assistant message."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not set. Set OPENAI_API_KEY in the environment or pass --api-key."
        )
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install the OpenAI package: pip install openai")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=500,
        temperature=0.6,
    )
    return resp.choices[0].message.content or ""


def call_ollama(system: str, user: str, model: str = "llama3.1:8b") -> str:
    """Call a local Ollama model via HTTP and return the response text."""
    prompt = system + "\n\n" + user
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")
    data = resp.json()
    return data.get("response", "")


def get_insight_for_user(
    summary: dict,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
) -> str:
    """Build prompt from summary, call LLM, return insight text."""
    system, user = build_prompt(summary)
    provider = (provider or "openai").lower()
    if provider == "ollama":
        return call_ollama(system, user, model=model)
    # default: openai
    return call_openai(system, user, api_key=api_key, model=model)


def run_insights(
    input_path: Path,
    output_path: Path | None = None,
    user_id=None,
    limit: int | None = None,
    last_n_days: int = 14,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
) -> list[dict]:
    """
    Load predictions, optionally filter by user_id or limit users, generate LLM insight per user.
    Returns list of {user_id, summary, insight}. If output_path, writes JSON there.
    """
    df = load_predictions(input_path)
    if "user_id" not in df.columns:
        raise ValueError("Predictions CSV must have a 'user_id' column.")

    user_ids = df["user_id"].dropna().unique().tolist()
    if user_id is not None:
        req_id = _normalize_user_id(user_id)
        user_ids = df.loc[
            df["user_id"].map(_normalize_user_id) == req_id, "user_id"
        ].dropna().unique().tolist()
        if not user_ids:
            sample_ids = df["user_id"].dropna().astype(str).head(10).tolist()
            raise ValueError(
                f"No data for user_id={user_id}. Example user_ids in file: {sample_ids}"
            )
    if limit is not None:
        user_ids = user_ids[: limit]

    results = []
    for i, uid in enumerate(user_ids):
        udf = get_user_df(df, uid, last_n_days=last_n_days)
        if udf.empty:
            continue
        summary = build_user_summary(udf, uid)
        try:
            insight = get_insight_for_user(
                summary,
                api_key=api_key,
                model=model,
                provider=provider,
            )
        except Exception as e:
            insight = f"[LLM error: {e}]"
        results.append({"user_id": uid, "summary": summary, "insight": insight})
        if output_path and (i + 1) % 10 == 0:
            _save_results(results, output_path)

    if output_path:
        _save_results(results, output_path)
        print(f"Wrote insights for {len(results)} users to {output_path}")
    return results


def _save_results(results: list[dict], output_path: Path) -> None:
    """Save results to JSON (summary can be large; we keep it for now)."""
    out = []
    for r in results:
        out.append({"user_id": r["user_id"], "insight": r["insight"], "summary": r["summary"]})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM health insights from predictions (suggestions, warnings, encouragement)."
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Predictions CSV (with user_id, date, features, *_pred).")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output JSON path (one object per user with insight + summary).")
    parser.add_argument("--user-id", type=str, default=None, help="Generate insight for this user only.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of users to process (for testing).")
    parser.add_argument("--last-n-days", type=int, default=14, help="Use last N days of data per user (default: 14).")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY).")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name. For provider=openai this is an OpenAI model (default: gpt-4o-mini). "
             "For provider=ollama this is a local Ollama model name (e.g. llama3.1:8b).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "ollama"],
        help="LLM provider: 'openai' (API) or 'ollama' (local). Default: openai.",
    )
    args = parser.parse_args()

    results = run_insights(
        input_path=args.input,
        output_path=args.output,
        user_id=args.user_id,
        limit=args.limit,
        last_n_days=args.last_n_days,
        api_key=args.api_key,
        model=args.model,
        provider=args.provider,
    )
    for r in results:
        print(f"\n--- User {r['user_id']} ---\n{r['insight']}\n")


if __name__ == "__main__":
    main()
