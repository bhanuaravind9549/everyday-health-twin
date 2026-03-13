# Everyday Health Twin – Web app

Manual daily check-in, in-app reminder, and “Try with sample data” demo.

## Run the app

From the **project root** (so `app` and `models` are on the path):

```bash
cd "C:\Study\Everyday Health Twin Project"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open: **http://localhost:8000**

- **Register** with your name (and optional email).
- **Log today**: sleep, mood, stress, steps, exercise, diet, screen time, work hours, energy.
- **Remind me to log daily**: set a time; we show an in-app banner if you haven’t logged yet when you open the app.
- **Get my insight**: after 3–7 days, fetches a short personalized message (rule-based by default; uses OpenAI if `OPENAI_API_KEY` is set).
- **Try with sample data**: small button bottom-right; shows a sample insight without logging (uses `models/predictions.csv` if present).

## Data

- SQLite DB: `data/health_twin.db` (users, entries, reminder prefs).
- Demo uses a synthetic user from `models/predictions.csv` (run `predict.py` first if you want demo to work).
