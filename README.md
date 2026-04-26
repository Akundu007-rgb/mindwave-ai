# MindWave — AI Mental Health Platform
### IdeaJam 2026 · Problem Statement #16

---

## What is MindWave?

MindWave is a full-stack AI-powered mental health platform built with:
- **Python + Flask** backend
- **NLP models** trained with scikit-learn (TF-IDF + Logistic Regression)
- **SQLite** database with per-user data isolation
- **User authentication** (register / login / logout)
- **Dynamic dashboard** with Chart.js visualizations
- **Responsive frontend** (HTML/CSS/JS + Jinja2 templates)

---

## Project Structure

```
mindwave_app/
├── app.py                  ← Flask application (main entry point)
├── requirements.txt        ← Python dependencies
├── model/
│   ├── train_model.py      ← NLP model training script
│   ├── tfidf_vectorizer.pkl
│   ├── emotion_classifier.pkl
│   ├── sentiment_classifier.pkl
│   ├── risk_classifier.pkl
│   ├── label_encoders.pkl
│   └── model_meta.json
├── instance/
│   └── mindwave.db         ← SQLite database (auto-created on first run)
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── journal.html
│   ├── checkin.html
│   ├── history.html
│   ├── features.html
│   └── about.html
└── static/
    ├── css/main.css
    └── js/main.js
```

---

## How to Run in VS Code

### Step 1 — Prerequisites
Make sure you have **Python 3.9+** installed. Check with:
```bash
python --version
```

### Step 2 — Open in VS Code
```
File → Open Folder → select the mindwave_app folder
```

### Step 3 — Open Terminal in VS Code
```
Terminal → New Terminal   (or Ctrl + `)
```

### Step 4 — Create a Virtual Environment
```bash
python -m venv venv
```

Activate it:
- **Windows:**  `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

You should see `(venv)` in your terminal prompt.

### Step 5 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 6 — Train the NLP Models
**This is required before running the app.**
```bash
python model/train_model.py
```
This trains 3 classifiers and saves `.pkl` files in the `model/` folder.
You'll see model accuracy reports printed in the terminal.

### Step 7 — Run the Flask App
```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## How It Works

### NLP Pipeline
1. User writes a journal entry
2. Text is preprocessed (lowercased, cleaned)
3. **TF-IDF vectorizer** converts text to feature vectors (5000 features, 1-3 ngrams)
4. Three classifiers run in parallel:
   - **Emotion classifier** → anxiety / depression / stress / hopeful / calm / anger / loneliness
   - **Sentiment classifier** → positive / neutral / negative
   - **Risk classifier** → low / medium / high
5. A **wellness score** (0–100) is computed from the outputs
6. Evidence-based **recommendations** are generated based on emotion + risk
7. All results are stored per-user in SQLite

### Database Schema
- `users` — id, username, email, hashed_password, created
- `journals` — id, user_id, text, emotion, sentiment, risk_level, wellness_score, created
- `checkins` — id, user_id, mood_score, sleep_hours, stress_level, notes, created

### Dashboard (Dynamic)
- Pulls real data from `/api/dashboard_data` endpoint
- Wellness trend chart — from journal entries
- Emotion distribution donut — from all entries
- Check-in trend bar chart — from daily check-ins
- All charts update automatically with new entries

---

## Pages

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Landing page + live NLP demo (login required) |
| Register | `/register` | Create a new account |
| Login | `/login` | Login to your account |
| Dashboard | `/dashboard` | Personal dynamic dashboard with charts |
| Journal | `/journal` | Write entries, get instant AI analysis |
| Check-in | `/checkin` | Log daily mood, sleep, and stress |
| History | `/history` | Full history of all entries |
| Features | `/features` | Platform feature overview |
| About | `/about` | Project info, tech stack, ethics |

---

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/analyze` | POST | ✅ | Analyze text with NLP models |
| `/api/dashboard_data` | GET | ✅ | Get user's chart data as JSON |

---

## Recommended VS Code Extensions
- **Python** (Microsoft) — syntax highlighting + IntelliSense
- **Pylance** — type checking
- **Flask Snippets** — Flask template shortcuts
- **SQLite Viewer** — browse your database visually

---

## Important Notice

MindWave is an **educational research prototype** built for IdeaJam 2026.
It is **not a medical device** and should not replace professional mental health care.

**Crisis Resources (India):**
- iCall: **9152987821**
- Vandrevala Foundation: **1860-2662-345**
- NIMHANS: **080-46110007**
