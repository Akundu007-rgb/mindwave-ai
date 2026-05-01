<h1 align="center">🧠 MindWave — AI Mental Health Platform</h1>

<p align="center">
  <b>Write. Analyze. Heal.</b><br>
  AI-powered journaling and mental health intelligence system.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python">
  <img src="https://img.shields.io/badge/Flask-WebApp-black?logo=flask">
  <img src="https://img.shields.io/badge/NLP-TF--IDF-purple">
  <img src="https://img.shields.io/badge/ML-Logistic%20Regression-orange">
  <img src="https://img.shields.io/badge/Database-SQLite-green?logo=sqlite">
  <img src="https://img.shields.io/badge/Charts-Chart.js-yellow">
</p>

---

# What is MindWave?

MindWave is a full-stack AI-powered mental health platform that helps users write journal entries, track daily check-ins, analyze emotions using NLP models, and visualize wellness insights through an interactive dashboard.

It is built with:

- Python and Flask backend
- NLP models using scikit-learn
- TF-IDF vectorization
- Logistic Regression classifiers
- SQLite database
- User authentication
- Chart.js dashboard
- HTML, CSS, JavaScript, and Jinja2 templates

---

# Key Features

- AI-powered journal analysis
- Emotion classification
- Sentiment detection
- Risk level prediction
- Wellness score generation
- Personalized recommendations
- Daily mood, sleep, and stress check-ins
- Dynamic dashboard with visual charts
- User registration and login
- Per-user data isolation
- Responsive frontend design

---

# Screenshots

## Landing Page

![Landing Page](assets/screenshots/landing.png)

## Journal Page

![Journal Page](assets/screenshots/journal.png)

## NLP Analysis Result

![NLP Analysis](assets/screenshots/analysis.png)

## Mental Health Assessment

![Assessment Page](assets/screenshots/assessment.png)

## Dashboard

![Dashboard](assets/screenshots/dashboard.png)

## Daily Check-in

![Check-in Page](assets/screenshots/checkin.png)

## History Page

![History Page](assets/screenshots/history.png)

---

# System Architecture

```text
User Input
|
v
Flask Backend
|
v
NLP Processing Pipeline
|
v
TF-IDF Vectorizer
|
v
Machine Learning Models
|
v
Prediction Engine
|
v
SQLite Database
|
v
Dashboard API
|
v
Chart.js Frontend
```

---

# Tech Stack

```text
Backend: Python, Flask
Frontend: HTML, CSS, JavaScript, Jinja2
Machine Learning: scikit-learn
NLP: TF-IDF Vectorizer
Database: SQLite
Visualization: Chart.js
Authentication: Flask Sessions, Hashed Passwords
```

---

# Project Structure

```text
mindwave_app/
|
|-- app.py
|-- requirements.txt
|
|-- model/
|   |-- train_model.py
|   |-- tfidf_vectorizer.pkl
|   |-- emotion_classifier.pkl
|   |-- sentiment_classifier.pkl
|   |-- risk_classifier.pkl
|   |-- label_encoders.pkl
|   |-- model_meta.json
|
|-- instance/
|   |-- mindwave.db
|
|-- templates/
|   |-- base.html
|   |-- index.html
|   |-- login.html
|   |-- register.html
|   |-- dashboard.html
|   |-- journal.html
|   |-- checkin.html
|   |-- history.html
|   |-- features.html
|   |-- about.html
|
|-- static/
|   |-- css/
|   |   |-- main.css
|   |
|   |-- js/
|       |-- main.js
|
|-- assets/
    |-- screenshots/
        |-- landing.png
        |-- journal.png
        |-- analysis.png
        |-- assessment.png
        |-- dashboard.png
        |-- checkin.png
        |-- history.png
```

---

# NLP Pipeline

```text
1. User writes a journal entry.
2. Text is cleaned and preprocessed.
3. TF-IDF converts the text into numerical features.
4. Emotion classifier predicts the emotional state.
5. Sentiment classifier predicts positive, neutral, or negative tone.
6. Risk classifier predicts low, medium, or high risk.
7. Wellness score is generated.
8. Personalized recommendations are created.
9. Results are stored in SQLite.
10. Dashboard updates using saved user data.
```

---

# Machine Learning Models

```text
Emotion Classifier:
Detects emotional state such as anxiety, depression, stress, hopeful, calm, anger, and loneliness.

Sentiment Classifier:
Predicts whether the journal text is positive, neutral, or negative.

Risk Classifier:
Detects whether the mental health risk is low, medium, or high.

Algorithm Used:
Logistic Regression

Feature Extraction:
TF-IDF Vectorizer
```

---

# Database Schema

```text
users table:
- id
- username
- email
- hashed_password
- created

journals table:
- id
- user_id
- text
- emotion
- sentiment
- risk_level
- wellness_score
- created

checkins table:
- id
- user_id
- mood_score
- sleep_hours
- stress_level
- notes
- created
```

---

# Dashboard Features

```text
The dashboard provides real-time visual insights using user data.

Features:
- Wellness score trend
- Emotion distribution chart
- Daily check-in trends
- Recent journal entries
- Activity statistics
- Latest wellness status

Dashboard data is powered through:
/api/dashboard_data
```

---

# Authentication and Security

```text
Security features:
- User registration
- User login
- User logout
- Session-based authentication
- Hashed password storage
- Per-user data separation
- Protected dashboard and journal routes
```

---

# How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/Akundu007-rgb/mindwave-ai.git
cd mindwave-ai
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

## 3. Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Train NLP Models

```bash
python model/train_model.py
```

## 6. Run Flask App

```bash
python app.py
```

## 7. Open in Browser

```text
http://127.0.0.1:5000
```

---

# Pages and Routes

```text
Home:
Route: /
Description: Landing page and project introduction

Register:
Route: /register
Description: Create a new account

Login:
Route: /login
Description: Login to existing account

Dashboard:
Route: /dashboard
Description: User analytics dashboard

Journal:
Route: /journal
Description: Write and analyze journal entries

Check-in:
Route: /checkin
Description: Track mood, sleep, and stress

History:
Route: /history
Description: View past journal entries and check-ins

Features:
Route: /features
Description: Platform feature overview

About:
Route: /about
Description: Project details, tech stack, and ethics
```

---

# API Endpoints

```text
/api/analyze
Method: POST
Authentication: Required
Description: Analyze journal text using NLP models

/api/dashboard_data
Method: GET
Authentication: Required
Description: Fetch dashboard chart data as JSON
```

---

# Future Enhancements

```text
- BERT-based emotion detection
- Mobile app version
- Cloud deployment
- Doctor consultation module
- Emergency support integration
- PDF wellness report generation
- Advanced analytics
- Multilingual support
```

---

# Ethics and Disclaimer

```text
MindWave is an educational and research-based prototype.

It is not a medical device and should not be used as a replacement for professional mental health care.

The predictions are generated using machine learning models and may not always be accurate.

For serious mental health concerns, users should contact a qualified professional.
```

---

# Crisis Resources India

```text
iCall: 9152987821
Vandrevala Foundation: 1860-2662-345
NIMHANS: 080-46110007
```

---

# Author

```text
Anirban Kundu
GitHub: https://github.com/Akundu007-rgb
```

---

# Support

```text
If you like this project, consider giving it a star on GitHub.
```

<p align="center">
  Made with ❤️ using AI for better mental health.
</p>
