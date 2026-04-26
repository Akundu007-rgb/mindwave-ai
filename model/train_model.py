"""
MindWave NLP Model Trainer
===========================
Trains a multi-output NLP classifier on mental health text data.
Uses TF-IDF + Logistic Regression (production-ready, no GPU needed).

Outputs:
  - model/emotion_classifier.pkl  : emotion label classifier
  - model/sentiment_classifier.pkl: sentiment (positive/negative/neutral)
  - model/risk_classifier.pkl     : distress risk level (low/medium/high)
  - model/tfidf_vectorizer.pkl    : shared TF-IDF vectorizer
  - model/label_encoders.pkl      : label encoders for each task
"""

import os, json, joblib, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# ── 1. DATASET ──────────────────────────────────────────────────────────────
# Curated dataset covering: anxiety, depression, stress, hopeful, calm, anger
# Each sample has: text, emotion, sentiment, risk_level
# Dataset modeled after public mental health corpora (DAIC-WOZ style,
# Reddit SuicideWatch, CLPsych shared tasks) — reproduced synthetically
# for training purposes.

DATASET = [
    # ── ANXIETY ──
    {"text": "I can't stop worrying about everything. My heart races and I feel like something terrible is about to happen.", "emotion": "anxiety", "sentiment": "negative", "risk": "medium"},
    {"text": "The panic attacks are getting worse. I can't breathe properly and my hands won't stop shaking.", "emotion": "anxiety", "sentiment": "negative", "risk": "high"},
    {"text": "I keep checking the locks over and over. I know it's irrational but I just cannot stop myself.", "emotion": "anxiety", "sentiment": "negative", "risk": "medium"},
    {"text": "Social situations terrify me. I rehearse conversations for hours and still freeze up when it actually happens.", "emotion": "anxiety", "sentiment": "negative", "risk": "medium"},
    {"text": "I've been catastrophizing everything at work. My boss said one thing and I spent the whole night convinced I was getting fired.", "emotion": "anxiety", "sentiment": "negative", "risk": "medium"},
    {"text": "The constant what-ifs are exhausting. I can't enjoy anything because my mind is always preparing for disaster.", "emotion": "anxiety", "sentiment": "negative", "risk": "medium"},
    {"text": "I feel a constant knot in my stomach. Eating has become difficult because of the persistent nausea from anxiety.", "emotion": "anxiety", "sentiment": "negative", "risk": "medium"},
    {"text": "My mind races at night and I can't fall asleep. Anxiety about tomorrow keeps me awake until 3am.", "emotion": "anxiety", "sentiment": "negative", "risk": "medium"},
    {"text": "I'm scared to leave the house. Every time I try, the fear becomes overwhelming and I have to turn back.", "emotion": "anxiety", "sentiment": "negative", "risk": "high"},
    {"text": "I feel a bit anxious before presentations but I usually manage to get through them okay.", "emotion": "anxiety", "sentiment": "neutral", "risk": "low"},
    {"text": "Had a small panic moment this morning but breathing exercises really helped calm me down.", "emotion": "anxiety", "sentiment": "neutral", "risk": "low"},
    {"text": "Worried about the interview tomorrow but I've prepared well and feel cautiously optimistic.", "emotion": "anxiety", "sentiment": "neutral", "risk": "low"},
    {"text": "Sometimes I overthink but I'm learning to catch myself and redirect my thoughts.", "emotion": "anxiety", "sentiment": "positive", "risk": "low"},

    # ── DEPRESSION ──
    {"text": "I haven't gotten out of bed in three days. Everything feels completely pointless and I don't know why I bother.", "emotion": "depression", "sentiment": "negative", "risk": "high"},
    {"text": "I used to love painting. Now I look at my brushes and feel nothing. The joy is just gone.", "emotion": "depression", "sentiment": "negative", "risk": "high"},
    {"text": "I feel like a burden to everyone around me. They'd probably be better off without me.", "emotion": "depression", "sentiment": "negative", "risk": "high"},
    {"text": "Nothing tastes good anymore. I eat just to survive but there's no pleasure in it.", "emotion": "depression", "sentiment": "negative", "risk": "medium"},
    {"text": "Crying for no reason again. I was just sitting there and the tears started and wouldn't stop.", "emotion": "depression", "sentiment": "negative", "risk": "medium"},
    {"text": "I feel completely empty inside. Not sad exactly, just hollow. Like there's no one home.", "emotion": "depression", "sentiment": "negative", "risk": "high"},
    {"text": "Getting out of bed takes every ounce of energy I have. By the time I'm up, I'm already exhausted.", "emotion": "depression", "sentiment": "negative", "risk": "medium"},
    {"text": "I've been canceling plans with friends for months. It's easier than pretending to be okay.", "emotion": "depression", "sentiment": "negative", "risk": "medium"},
    {"text": "The future looks completely dark. I can't imagine things ever getting better.", "emotion": "depression", "sentiment": "negative", "risk": "high"},
    {"text": "Feeling down today but I know these moods pass. I reached out to my therapist.", "emotion": "depression", "sentiment": "neutral", "risk": "low"},
    {"text": "Had a rough week emotionally but journaling has been helping me process things.", "emotion": "depression", "sentiment": "neutral", "risk": "low"},
    {"text": "Still struggling but I noticed I smiled genuinely for the first time in a while today.", "emotion": "depression", "sentiment": "positive", "risk": "low"},

    # ── STRESS ──
    {"text": "Work deadlines are crushing me. I'm working 14 hour days and I still can't keep up.", "emotion": "stress", "sentiment": "negative", "risk": "medium"},
    {"text": "I feel like I'm drowning in responsibilities. The to-do list never ends and I'm falling apart.", "emotion": "stress", "sentiment": "negative", "risk": "medium"},
    {"text": "My head is constantly pounding. The stress from the project is giving me daily migraines.", "emotion": "stress", "sentiment": "negative", "risk": "medium"},
    {"text": "I've been snapping at my family because of work stress. I hate that I'm taking it out on them.", "emotion": "stress", "sentiment": "negative", "risk": "medium"},
    {"text": "Juggling school, work, and family is impossible. I feel like I'm failing at all three.", "emotion": "stress", "sentiment": "negative", "risk": "medium"},
    {"text": "The pressure is constant. Even on weekends I can't switch off because Monday is always looming.", "emotion": "stress", "sentiment": "negative", "risk": "medium"},
    {"text": "Deadlines at work are tight but my team is supportive and we're managing well together.", "emotion": "stress", "sentiment": "neutral", "risk": "low"},
    {"text": "Stressful day but a good workout helped. I'm learning better coping strategies slowly.", "emotion": "stress", "sentiment": "neutral", "risk": "low"},
    {"text": "Work is busy but I'm proud of what we've accomplished this quarter.", "emotion": "stress", "sentiment": "positive", "risk": "low"},
    {"text": "Stress is high but manageable. Breathing exercises before meetings have made a real difference.", "emotion": "stress", "sentiment": "positive", "risk": "low"},

    # ── HOPEFUL / POSITIVE ──
    {"text": "I started therapy last month and I'm already seeing improvements. Learning to challenge my negative thoughts.", "emotion": "hopeful", "sentiment": "positive", "risk": "low"},
    {"text": "Today was a good day. I went for a walk, called a friend, and actually felt present in the moment.", "emotion": "hopeful", "sentiment": "positive", "risk": "low"},
    {"text": "Recovery isn't linear but I'm making progress. Six months ago I couldn't have written this.", "emotion": "hopeful", "sentiment": "positive", "risk": "low"},
    {"text": "Meditation has genuinely changed my relationship with anxiety. I feel more grounded now.", "emotion": "hopeful", "sentiment": "positive", "risk": "low"},
    {"text": "I'm learning that it's okay to ask for help. Opened up to a colleague today and felt lighter.", "emotion": "hopeful", "sentiment": "positive", "risk": "low"},
    {"text": "Exercise has become my anchor. Even a 20-minute walk shifts my mood significantly.", "emotion": "hopeful", "sentiment": "positive", "risk": "low"},
    {"text": "I set small goals today and achieved them. Progress feels possible again.", "emotion": "hopeful", "sentiment": "positive", "risk": "low"},
    {"text": "After months of struggling, I finally had a week where I felt like myself again.", "emotion": "hopeful", "sentiment": "positive", "risk": "low"},
    {"text": "Gratitude journaling sounds cheesy but writing three good things every night really works for me.", "emotion": "hopeful", "sentiment": "positive", "risk": "low"},

    # ── CALM / STABLE ──
    {"text": "Feeling balanced today. Sleep was good, ate well, and took breaks throughout the day.", "emotion": "calm", "sentiment": "positive", "risk": "low"},
    {"text": "Mindfulness practice is becoming second nature. I feel more centered than I have in years.", "emotion": "calm", "sentiment": "positive", "risk": "low"},
    {"text": "Nothing major happening emotionally. Just steady and present.", "emotion": "calm", "sentiment": "neutral", "risk": "low"},
    {"text": "Had a peaceful morning. Coffee, reading, gentle music. Good way to start the day.", "emotion": "calm", "sentiment": "positive", "risk": "low"},
    {"text": "Feeling content. Not euphoric, just quietly satisfied with where things are.", "emotion": "calm", "sentiment": "positive", "risk": "low"},
    {"text": "Everything feels manageable right now. Clear head, good sleep, connected with friends.", "emotion": "calm", "sentiment": "positive", "risk": "low"},

    # ── ANGER ──
    {"text": "I'm furious and I don't know what to do with it. Everything is making me rage.", "emotion": "anger", "sentiment": "negative", "risk": "medium"},
    {"text": "I snapped at my partner over something trivial. The anger came out of nowhere and scared me.", "emotion": "anger", "sentiment": "negative", "risk": "medium"},
    {"text": "I feel this burning resentment that won't go away. It's poisoning everything.", "emotion": "anger", "sentiment": "negative", "risk": "medium"},
    {"text": "Had an argument but talked it through. Anger was valid but I expressed it constructively.", "emotion": "anger", "sentiment": "neutral", "risk": "low"},

    # ── LONELINESS ──
    {"text": "Surrounded by people but completely alone. No one really sees me.", "emotion": "loneliness", "sentiment": "negative", "risk": "high"},
    {"text": "It's been weeks since anyone checked in on me. The silence is deafening.", "emotion": "loneliness", "sentiment": "negative", "risk": "high"},
    {"text": "I moved to a new city and I don't know anyone. The isolation is crushing.", "emotion": "loneliness", "sentiment": "negative", "risk": "medium"},
    {"text": "Reached out to an old friend today. Small connection but it helped enormously.", "emotion": "loneliness", "sentiment": "positive", "risk": "low"},
    {"text": "Joined a community group. First session was awkward but hopeful.", "emotion": "loneliness", "sentiment": "positive", "risk": "low"},

    # ── CRISIS INDICATORS ──
    {"text": "I've been thinking that everyone would be better off without me. I'm tired of fighting.", "emotion": "depression", "sentiment": "negative", "risk": "high"},
    {"text": "I don't see a point in continuing. The pain is too much and I don't know how to make it stop.", "emotion": "depression", "sentiment": "negative", "risk": "high"},
    {"text": "I've been researching ways to hurt myself. I don't know what's stopping me.", "emotion": "depression", "sentiment": "negative", "risk": "high"},
    {"text": "Wrote a note today but then deleted it. I'm scared of my own thoughts right now.", "emotion": "depression", "sentiment": "negative", "risk": "high"},

    # ── MIXED / COMPLEX ──
    {"text": "Good days and bad days. Today was somewhere in the middle — flat but not despairing.", "emotion": "depression", "sentiment": "neutral", "risk": "low"},
    {"text": "Anxiety is real but I'm managing it better with the tools from therapy.", "emotion": "anxiety", "sentiment": "positive", "risk": "low"},
    {"text": "Stressed about money but I made a budget today which helped me feel more in control.", "emotion": "stress", "sentiment": "neutral", "risk": "low"},
    {"text": "Relationship issues causing a lot of emotional turbulence. Working through it in couples therapy.", "emotion": "stress", "sentiment": "neutral", "risk": "low"},
    {"text": "Grieving my dog who passed away. The sadness comes in waves but it's okay to mourn.", "emotion": "depression", "sentiment": "neutral", "risk": "low"},
    {"text": "Feeling nervous about a medical test result. Trying to stay grounded and not catastrophize.", "emotion": "anxiety", "sentiment": "neutral", "risk": "low"},
    {"text": "Had a breakdown at work. Cried in the bathroom. But asked for help afterward.", "emotion": "depression", "sentiment": "neutral", "risk": "medium"},
    {"text": "Celebrated six months without a panic attack today. Still anxious but much more in control.", "emotion": "hopeful", "sentiment": "positive", "risk": "low"},
    {"text": "Setback in my recovery this week. Trying not to see it as failure but as information.", "emotion": "hopeful", "sentiment": "neutral", "risk": "medium"},
    {"text": "My therapist said I've made real progress. Hard to see it from inside but I believe her.", "emotion": "hopeful", "sentiment": "positive", "risk": "low"},
]

# ── 2. AUGMENT DATA (simple augmentation for richer training) ─────────────
def augment_text(text):
    """Simple word-level augmentation."""
    synonyms = {
        "terrible": "awful", "exhausted": "drained", "scared": "frightened",
        "angry": "furious", "sad": "sorrowful", "happy": "joyful",
        "worried": "anxious", "tired": "fatigued", "hopeful": "optimistic",
        "calm": "peaceful", "lonely": "isolated", "help": "support"
    }
    words = text.split()
    augmented = []
    for w in words:
        w_lower = w.lower().rstrip('.,!?')
        if w_lower in synonyms and np.random.random() > 0.6:
            augmented.append(synonyms[w_lower])
        else:
            augmented.append(w)
    return " ".join(augmented)

augmented_data = []
for item in DATASET:
    augmented_data.append(item)
    # Add 2 augmented versions per sample
    for _ in range(2):
        aug_item = item.copy()
        aug_item["text"] = augment_text(item["text"])
        augmented_data.append(aug_item)

df = pd.DataFrame(augmented_data)
print(f"Dataset size: {len(df)} samples")
print(f"Emotion distribution:\n{df['emotion'].value_counts()}\n")

# ── 3. PREPROCESSING ─────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(preprocess)

# ── 4. ENCODE LABELS ──────────────────────────────────────────────────────
le_emotion   = LabelEncoder()
le_sentiment = LabelEncoder()
le_risk      = LabelEncoder()

df["emotion_enc"]   = le_emotion.fit_transform(df["emotion"])
df["sentiment_enc"] = le_sentiment.fit_transform(df["sentiment"])
df["risk_enc"]      = le_risk.fit_transform(df["risk"])

# ── 5. TF-IDF VECTORIZER ──────────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    min_df=1,
    analyzer="word"
)

X = vectorizer.fit_transform(df["clean_text"])

# ── 6. TRAIN MODELS ───────────────────────────────────────────────────────
def train_and_report(X, y, label_encoder, name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    labels = label_encoder.classes_
    print(f"\n{'='*50}")
    print(f"  {name} Model Report")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))
    return clf

emotion_clf   = train_and_report(X, df["emotion_enc"],   le_emotion,   "Emotion")
sentiment_clf = train_and_report(X, df["sentiment_enc"], le_sentiment, "Sentiment")
risk_clf      = train_and_report(X, df["risk_enc"],      le_risk,      "Risk Level")

# ── 7. SAVE MODELS ────────────────────────────────────────────────────────
save_dir = os.path.dirname(os.path.abspath(__file__))

joblib.dump(vectorizer,    os.path.join(save_dir, "tfidf_vectorizer.pkl"))
joblib.dump(emotion_clf,   os.path.join(save_dir, "emotion_classifier.pkl"))
joblib.dump(sentiment_clf, os.path.join(save_dir, "sentiment_classifier.pkl"))
joblib.dump(risk_clf,      os.path.join(save_dir, "risk_classifier.pkl"))
joblib.dump({
    "emotion":   le_emotion,
    "sentiment": le_sentiment,
    "risk":      le_risk
}, os.path.join(save_dir, "label_encoders.pkl"))

# Save class info
meta = {
    "emotions":   list(le_emotion.classes_),
    "sentiments": list(le_sentiment.classes_),
    "risk_levels":list(le_risk.classes_),
    "dataset_size": len(df)
}
with open(os.path.join(save_dir, "model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\n✅ All models saved successfully!")
print(f"   Emotions: {meta['emotions']}")
print(f"   Sentiments: {meta['sentiments']}")
print(f"   Risk levels: {meta['risk_levels']}")
