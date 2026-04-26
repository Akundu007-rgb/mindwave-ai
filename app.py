"""
MindWave Flask Application
===========================
Full-stack mental health platform with:
- User auth (register / login / logout)
- NLP analysis endpoint (emotion, sentiment, risk)
- Per-user journal entries stored in SQLite
- Dynamic dashboard data API
"""

import os, json, hashlib, re, datetime, sqlite3
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, g)
import joblib
import numpy as np

# ── APP SETUP ────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "mindwave_secret_key_change_in_production_2026"

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
# Always create instance/ folder on startup so SQLite can write the DB file
_INST_DIR = os.path.join(BASE_DIR, "instance")
os.makedirs(_INST_DIR, exist_ok=True)
DB_PATH   = os.path.join(_INST_DIR, "mindwave.db")

# ── LOAD NLP MODELS ──────────────────────────────────────────────────────
print("Loading NLP models...")
try:
    vectorizer    = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    emotion_clf   = joblib.load(os.path.join(MODEL_DIR, "emotion_classifier.pkl"))
    sentiment_clf = joblib.load(os.path.join(MODEL_DIR, "sentiment_classifier.pkl"))
    risk_clf      = joblib.load(os.path.join(MODEL_DIR, "risk_classifier.pkl"))
    encoders      = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    with open(os.path.join(MODEL_DIR, "model_meta.json")) as f:
        model_meta = json.load(f)
    print("✅ NLP models loaded.")
except Exception as e:
    print(f"❌ Model load error: {e}")
    vectorizer = emotion_clf = sentiment_clf = risk_clf = encoders = None
    model_meta = {}

# ── DATABASE ─────────────────────────────────────────────────────────────
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db: db.close()

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT UNIQUE NOT NULL,
            email     TEXT UNIQUE NOT NULL,
            password  TEXT NOT NULL,
            created   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS journals (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            text         TEXT NOT NULL,
            emotion      TEXT,
            sentiment    TEXT,
            risk_level   TEXT,
            wellness_score INTEGER,
            created      TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS checkins (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            mood_score   INTEGER NOT NULL,
            sleep_hours  REAL,
            stress_level INTEGER,
            notes        TEXT,
            created      TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """)
        db.commit()

# ── NLP ANALYSIS ─────────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def analyze_text(text):
    """Run all NLP models on input text, return structured result."""
    if not vectorizer:
        return {"error": "Models not loaded"}

    clean = preprocess(text)
    X = vectorizer.transform([clean])

    emotion_enc   = emotion_clf.predict(X)[0]
    sentiment_enc = sentiment_clf.predict(X)[0]
    risk_enc      = risk_clf.predict(X)[0]

    emotion   = encoders["emotion"].inverse_transform([emotion_enc])[0]
    sentiment = encoders["sentiment"].inverse_transform([sentiment_enc])[0]
    risk      = encoders["risk"].inverse_transform([risk_enc])[0]

    # Get probabilities
    emotion_probs   = emotion_clf.predict_proba(X)[0]
    sentiment_probs = sentiment_clf.predict_proba(X)[0]
    risk_probs      = risk_clf.predict_proba(X)[0]

    emotion_dist = {
        encoders["emotion"].inverse_transform([i])[0]: round(float(p) * 100, 1)
        for i, p in enumerate(emotion_probs)
    }

    # Wellness score: positive sentiment + low risk = high score
    risk_penalty = {"low": 0, "medium": -20, "high": -45}
    sentiment_bonus = {"positive": 20, "neutral": 0, "negative": -15}
    base = 65
    wellness = max(10, min(100, base + sentiment_bonus.get(sentiment, 0) + risk_penalty.get(risk, 0) + int(np.random.normal(0, 5))))

    return {
        "emotion":       emotion,
        "sentiment":     sentiment,
        "risk_level":    risk,
        "wellness_score": wellness,
        "emotion_distribution": emotion_dist,
        "confidence": {
            "emotion":   round(float(emotion_probs.max()) * 100, 1),
            "sentiment": round(float(sentiment_probs.max()) * 100, 1),
            "risk":      round(float(risk_probs.max()) * 100, 1),
        },
        "recommendations": get_recommendations(emotion, risk, sentiment)
    }

def get_recommendations(emotion, risk, sentiment):
    recs = {
        "anxiety":    ["Try 4-7-8 breathing", "Ground yourself with 5-4-3-2-1 senses", "Limit caffeine today", "Write your worries down"],
        "depression": ["Go outside for 10 minutes", "Text one person you trust", "Do one small task to build momentum", "Avoid isolating"],
        "stress":     ["Take a 5-minute break every hour", "Prioritize your top 3 tasks only", "Try progressive muscle relaxation", "Hydrate and eat"],
        "hopeful":    ["Channel this into journaling your goals", "Share your positivity with someone", "Plan your next step forward"],
        "calm":       ["Maintain your routine", "This is a great time to reflect and plan", "Practice gratitude"],
        "anger":      ["Take 10 deep breaths before responding", "Physical exercise helps release anger", "Write it out before saying it aloud"],
        "loneliness": ["Reach out to one person today", "Join an online community", "Volunteer — connection through purpose helps"],
    }
    high_risk_addons = ["Consider speaking with a mental health professional", "Contact a crisis line if needed: iCall 9152987821"]

    result = recs.get(emotion, ["Practice mindfulness", "Journal your thoughts", "Talk to someone you trust"])
    if risk == "high":
        result = result[:2] + high_risk_addons
    return result

# ── AUTH HELPERS ──────────────────────────────────────────────────────────
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def current_user():
    """Return the logged-in user row, or None if not logged in / user not found."""
    if "user_id" not in session:
        return None
    try:
        db = get_db()
        return db.execute("SELECT * FROM users WHERE id=?", (session["user_id"],)).fetchone()
    except Exception:
        return None

def login_required(f):
    from functools import wraps
    @wraps(f)
    def wrapped(*args, **kwargs):
        # 1. No session at all → go to login
        if "user_id" not in session:
            return redirect(url_for("login"))
        # 2. Session exists but user was deleted (e.g. DB reset while cookie stayed)
        #    This is what causes "NoneType is not subscriptable" on user["id"]
        user = current_user()
        if user is None:
            session.clear()   # wipe the stale cookie so they get a clean login
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapped

# ── ROUTES: AUTH ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    user = current_user()
    return render_template("index.html", user=user)

@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm", "")

        if not all([username, email, password]):
            error = "All fields are required."
        elif password != confirm:
            error = "Passwords do not match."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        else:
            db = get_db()
            existing = db.execute("SELECT id FROM users WHERE email=? OR username=?", (email, username)).fetchone()
            if existing:
                error = "Email or username already exists."
            else:
                db.execute(
                    "INSERT INTO users (username, email, password, created) VALUES (?,?,?,?)",
                    (username, email, hash_password(password), datetime.datetime.now().isoformat())
                )
                db.commit()
                user = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
                session["user_id"] = user["id"]
                return redirect(url_for("dashboard"))

    return render_template("register.html", error=error)

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    error = None
    if request.method == "POST":
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE email=? AND password=?",
                          (email, hash_password(password))).fetchone()
        if user:
            session["user_id"] = user["id"]
            return redirect(url_for("dashboard"))
        else:
            error = "Invalid email or password."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# ── ROUTES: MAIN PAGES ────────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    db   = get_db()

    # Last 14 journal entries for chart
    journals = db.execute(
        "SELECT * FROM journals WHERE user_id=? ORDER BY created DESC LIMIT 30",
        (user["id"],)
    ).fetchall()

    # Last 14 checkins
    checkins = db.execute(
        "SELECT * FROM checkins WHERE user_id=? ORDER BY created DESC LIMIT 14",
        (user["id"],)
    ).fetchall()

    # Streak
    streak = compute_streak(user["id"], db)

    # Latest wellness score
    latest_wellness = journals[0]["wellness_score"] if journals else 65
    avg_wellness = int(sum(j["wellness_score"] for j in journals) / len(journals)) if journals else 65

    # Emotion distribution from last 30 entries
    emotion_counts = {}
    for j in journals:
        e = j["emotion"] or "unknown"
        emotion_counts[e] = emotion_counts.get(e, 0) + 1

    return render_template("dashboard.html",
        user=user,
        journals=journals,
        checkins=checkins,
        streak=streak,
        latest_wellness=latest_wellness,
        avg_wellness=avg_wellness,
        emotion_counts=json.dumps(emotion_counts),
        journal_count=len(journals),
    )

@app.route("/journal", methods=["GET", "POST"])
@login_required
def journal():
    user = current_user()
    db   = get_db()
    result = None

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if len(text) < 10:
            return render_template("journal.html", user=user, error="Please write at least a sentence.", recent=[])

        analysis = analyze_text(text)
        db.execute(
            "INSERT INTO journals (user_id, text, emotion, sentiment, risk_level, wellness_score, created) VALUES (?,?,?,?,?,?,?)",
            (user["id"], text, analysis["emotion"], analysis["sentiment"],
             analysis["risk_level"], analysis["wellness_score"],
             datetime.datetime.now().isoformat())
        )
        db.commit()
        result = analysis

    recent = db.execute(
        "SELECT * FROM journals WHERE user_id=? ORDER BY created DESC LIMIT 10",
        (user["id"],)
    ).fetchall()
    return render_template("journal.html", user=user, result=result, recent=recent)

@app.route("/checkin", methods=["GET", "POST"])
@login_required
def checkin():
    user = current_user()
    db   = get_db()
    saved = False

    if request.method == "POST":
        mood_score   = int(request.form.get("mood_score", 5))
        sleep_hours  = float(request.form.get("sleep_hours", 7))
        stress_level = int(request.form.get("stress_level", 5))
        notes        = request.form.get("notes", "").strip()
        db.execute(
            "INSERT INTO checkins (user_id, mood_score, sleep_hours, stress_level, notes, created) VALUES (?,?,?,?,?,?)",
            (user["id"], mood_score, sleep_hours, stress_level, notes, datetime.datetime.now().isoformat())
        )
        db.commit()
        saved = True

    recent = db.execute(
        "SELECT * FROM checkins WHERE user_id=? ORDER BY created DESC LIMIT 7",
        (user["id"],)
    ).fetchall()
    return render_template("checkin.html", user=user, saved=saved, recent=recent)

@app.route("/history")
@login_required
def history():
    user = current_user()
    db   = get_db()
    journals = db.execute(
        "SELECT * FROM journals WHERE user_id=? ORDER BY created DESC",
        (user["id"],)
    ).fetchall()
    checkins = db.execute(
        "SELECT * FROM checkins WHERE user_id=? ORDER BY created DESC",
        (user["id"],)
    ).fetchall()
    return render_template("history.html", user=user, journals=journals, checkins=checkins)

@app.route("/features")
def features():
    return render_template("features.html", user=current_user())

@app.route("/about")
def about():
    return render_template("about.html", user=current_user())

@app.route("/comparison")
def comparison():
    return render_template("comparison.html", user=current_user())

# ── ASSESSMENT ROUTES ─────────────────────────────────────────────────────
QUESTIONS = [
    {
        "id": 1,
        "text": "Over the past 2 weeks, how often have you felt down, depressed, or hopeless?",
        "category": "mood",
        "options": [
            {"label": "Not at all", "score": 0},
            {"label": "Several days", "score": 1},
            {"label": "More than half the days", "score": 2},
            {"label": "Nearly every day", "score": 3},
        ]
    },
    {
        "id": 2,
        "text": "How often have you had little interest or pleasure in doing things you normally enjoy?",
        "category": "anhedonia",
        "options": [
            {"label": "Not at all", "score": 0},
            {"label": "Several days", "score": 1},
            {"label": "More than half the days", "score": 2},
            {"label": "Nearly every day", "score": 3},
        ]
    },
    {
        "id": 3,
        "text": "How would you describe your sleep quality over the past week?",
        "category": "sleep",
        "options": [
            {"label": "Very good — I sleep well consistently", "score": 0},
            {"label": "Fairly good — occasional disruptions", "score": 1},
            {"label": "Fairly bad — often tired in the morning", "score": 2},
            {"label": "Very bad — I struggle to sleep most nights", "score": 3},
        ]
    },
    {
        "id": 4,
        "text": "How often do you feel overwhelmed by worry or anxiety?",
        "category": "anxiety",
        "options": [
            {"label": "Rarely or never", "score": 0},
            {"label": "Occasionally — I can manage it", "score": 1},
            {"label": "Often — it interferes with daily life", "score": 2},
            {"label": "Almost always — it feels uncontrollable", "score": 3},
        ]
    },
    {
        "id": 5,
        "text": "How connected do you feel to friends, family, or your community?",
        "category": "social",
        "options": [
            {"label": "Very connected — strong support network", "score": 0},
            {"label": "Somewhat connected — some good relationships", "score": 1},
            {"label": "Slightly disconnected — feeling isolated", "score": 2},
            {"label": "Very isolated — rarely feel understood", "score": 3},
        ]
    },
    {
        "id": 6,
        "text": "How well are you able to concentrate on tasks, work, or studies?",
        "category": "concentration",
        "options": [
            {"label": "Very well — sharp and focused", "score": 0},
            {"label": "Fairly well — occasional difficulty", "score": 1},
            {"label": "With difficulty — often lose focus", "score": 2},
            {"label": "Severely impaired — can barely concentrate", "score": 3},
        ]
    },
    {
        "id": 7,
        "text": "How often do you take care of yourself (exercise, nutrition, hobbies, rest)?",
        "category": "selfcare",
        "options": [
            {"label": "Daily — self-care is a priority", "score": 0},
            {"label": "A few times a week — mostly consistent", "score": 1},
            {"label": "Rarely — hard to find motivation", "score": 2},
            {"label": "Never — I've stopped taking care of myself", "score": 3},
        ]
    },
]

def compute_assessment_result(answers):
    """answers = list of int scores (0–3) for each question."""
    total   = sum(answers)
    max_val = len(QUESTIONS) * 3   # 21
    pct     = total / max_val      # 0.0 – 1.0

    if pct < 0.25:
        level, label, color = "low",    "Thriving",         "green"
        desc = ("Your responses suggest you are in a good mental space right now. "
                "Keep nurturing your positive habits — consistent journaling, regular check-ins, "
                "and staying connected with people you trust will help maintain this.")
        recs = ["Keep a daily gratitude journal", "Maintain your sleep schedule",
                "Share your wellbeing strategies with a friend", "Schedule a monthly mental health check-in"]
    elif pct < 0.50:
        level, label, color = "medium", "Moderate Stress",  "yellow"
        desc = ("You are experiencing some stress and low mood. This is very manageable with the right tools. "
                "Our AI has detected patterns consistent with moderate anxiety or mild low mood. "
                "Regular journaling and the coping strategies below can make a real difference.")
        recs = ["Try 10 minutes of mindfulness daily", "Write a journal entry every evening",
                "Reach out to one person you trust this week", "Reduce screen time before bed"]
    elif pct < 0.75:
        level, label, color = "high",   "Elevated Concern", "orange"
        desc = ("Your responses indicate elevated distress. You may be experiencing significant anxiety, "
                "low mood, or burnout. We strongly recommend speaking with a mental health professional. "
                "MindWave will prepare an AI-generated summary of your patterns for them.")
        recs = ["Book an appointment with a counsellor or therapist",
                "Use MindWave's daily journal to track your mood",
                "Practice grounding: name 5 things you can see right now",
                "Reach out to a crisis line if things feel urgent"]
    else:
        level, label, color = "crisis", "High Priority",    "red"
        desc = ("We are concerned about your wellbeing. Your responses suggest significant distress. "
                "Please reach out to a professional or crisis service immediately. "
                "You are not alone — help is available right now.")
        recs = ["Call iCall immediately: 9152987821",
                "Contact Vandrevala Foundation: 1860-2662-345",
                "Tell someone you trust how you are feeling today",
                "If in immediate danger, call emergency services (112)"]

    wellness_score = max(5, min(100, round((1 - pct) * 100)))
    category_scores = {}
    cats = [q["category"] for q in QUESTIONS]
    for i, cat in enumerate(cats):
        category_scores[cat] = answers[i] if i < len(answers) else 0

    return {
        "level":            level,
        "label":            label,
        "color":            color,
        "desc":             desc,
        "recommendations":  recs,
        "wellness_score":   wellness_score,
        "total_score":      total,
        "max_score":        max_val,
        "pct":              round(pct * 100),
        "category_scores":  category_scores,
    }

@app.route("/assessment", methods=["GET", "POST"])
@login_required
def assessment():
    user   = current_user()
    result = None
    error  = None

    if request.method == "POST":
        answers = []
        valid   = True
        for q in QUESTIONS:
            val = request.form.get(f"q{q['id']}")
            if val is None:
                valid = False
                error = f"Please answer all questions (missed question {q['id']})."
                break
            answers.append(int(val))

        if valid:
            result = compute_assessment_result(answers)
            db     = get_db()
            # Save assessment as a special journal entry so it appears in history
            summary_text = (
                f"[Assessment] Score: {result['total_score']}/{result['max_score']} "
                f"({result['pct']}%) — {result['label']}. "
                f"Mood:{answers[0]} Anhedonia:{answers[1]} Sleep:{answers[2]} "
                f"Anxiety:{answers[3]} Social:{answers[4]} Focus:{answers[5]} Selfcare:{answers[6]}"
            )
            # Map assessment level to emotion/risk
            level_map = {
                "low":    ("hopeful",    "low",    result["wellness_score"]),
                "medium": ("stress",     "medium", result["wellness_score"]),
                "high":   ("anxiety",    "high",   result["wellness_score"]),
                "crisis": ("depression", "high",   result["wellness_score"]),
            }
            emotion, risk, ws = level_map[result["level"]]
            sentiment = "positive" if result["level"] == "low" else (
                        "neutral"  if result["level"] == "medium" else "negative")
            db.execute(
                "INSERT INTO journals (user_id, text, emotion, sentiment, risk_level, wellness_score, created) VALUES (?,?,?,?,?,?,?)",
                (user["id"], summary_text, emotion, sentiment, risk, ws,
                 datetime.datetime.now().isoformat())
            )
            db.commit()

    # Previous assessments (assessment entries in journal)
    db = get_db()
    prev_assessments = db.execute(
        "SELECT * FROM journals WHERE user_id=? AND text LIKE '[Assessment]%' ORDER BY created DESC LIMIT 5",
        (user["id"],)
    ).fetchall()

    return render_template("assessment.html",
        user=user,
        questions=QUESTIONS,
        result=result,
        error=error,
        prev_assessments=prev_assessments,
    )

# ── ROUTES: API ───────────────────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
@login_required
def api_analyze():
    data = request.get_json()
    text = data.get("text", "")
    if len(text) < 5:
        return jsonify({"error": "Text too short"}), 400
    return jsonify(analyze_text(text))

@app.route("/api/dashboard_data")
@login_required
def api_dashboard_data():
    user = current_user()
    db   = get_db()

    journals = db.execute(
        "SELECT wellness_score, emotion, sentiment, risk_level, created FROM journals WHERE user_id=? ORDER BY created DESC LIMIT 14",
        (user["id"],)
    ).fetchall()

    checkins = db.execute(
        "SELECT mood_score, sleep_hours, stress_level, created FROM checkins WHERE user_id=? ORDER BY created DESC LIMIT 14",
        (user["id"],)
    ).fetchall()

    mood_trend = [{"date": r["created"][:10], "score": r["wellness_score"]} for r in reversed(journals)]
    checkin_trend = [{"date": r["created"][:10], "mood": r["mood_score"], "sleep": r["sleep_hours"], "stress": r["stress_level"]} for r in reversed(checkins)]

    emotion_counts = {}
    for j in journals:
        e = j["emotion"] or "unknown"
        emotion_counts[e] = emotion_counts.get(e, 0) + 1

    avg_wellness = int(sum(j["wellness_score"] for j in journals) / len(journals)) if journals else 0
    avg_mood     = round(sum(c["mood_score"] for c in checkins) / len(checkins), 1) if checkins else 0
    avg_sleep    = round(sum(c["sleep_hours"] for c in checkins) / len(checkins), 1) if checkins else 0

    return jsonify({
        "mood_trend": mood_trend,
        "checkin_trend": checkin_trend,
        "emotion_counts": emotion_counts,
        "stats": {
            "avg_wellness": avg_wellness,
            "avg_mood": avg_mood,
            "avg_sleep": avg_sleep,
            "journal_count": len(list(db.execute("SELECT id FROM journals WHERE user_id=?", (user["id"],)))),
            "streak": compute_streak(user["id"], db)
        }
    })

# ── HELPERS ───────────────────────────────────────────────────────────────
def compute_streak(user_id, db):
    rows = db.execute(
        "SELECT created FROM journals WHERE user_id=? UNION SELECT created FROM checkins WHERE user_id=? ORDER BY created DESC",
        (user_id, user_id)
    ).fetchall()

    dates = sorted(set(r["created"][:10] for r in rows), reverse=True)
    if not dates:
        return 0

    streak = 0
    today  = datetime.date.today()
    for i, d in enumerate(dates):
        check = today - datetime.timedelta(days=i)
        if str(check) == d:
            streak += 1
        else:
            break
    return streak

# ── MAIN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    print("🚀 MindWave running at http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
