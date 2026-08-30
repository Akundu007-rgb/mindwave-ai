"""
MindWave NLP Model Trainer — v2 (Kaggle Dataset Edition)
=========================================================

REAL KAGGLE DATASETS USED:
─────────────────────────────────────────────────────────
1. Emotion Detection from Text
   URL  : https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text
   File : tweet_emotions.csv
   Cols : tweet_id, sentiment, content
   Size : ~40,000 tweets labelled with 13 emotions
   Use  : Emotion classifier (we map 13 → 7 of our categories)

2. Sentiment140 (Twitter Sentiment)
   URL  : https://www.kaggle.com/datasets/kazanova/sentiment140
   File : training.1600000.processed.noemoticon.csv
   Cols : target(0/4), id, date, flag, user, text
   Use  : Sentiment analyser (positive / negative)
   Note : We sample 5000 rows per class for speed

3. Suicide and Depression Detection
   URL  : https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch
   File : Suicide_Detection.csv
   Cols : text, class  (suicide / non-suicide)
   Use  : Risk level predictor (high risk = suicide class)

HOW TO USE KAGGLE DATASETS:
─────────────────────────────────────────────────────────
1. Go to kaggle.com → create a free account
2. Download each dataset CSV listed above
3. Place them inside:   mindwave_app/data/
   - data/tweet_emotions.csv
   - data/sentiment140.csv         (rename the downloaded file)
   - data/suicide_detection.csv    (rename if needed)
4. Run:  python model/train_model.py

If you do NOT have the Kaggle files yet, the script falls back
to the built-in synthetic dataset automatically so the app still
works while you gather the real data.
─────────────────────────────────────────────────────────
"""

import os, json, re, joblib, warnings
import numpy  as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model             import LogisticRegression
from sklearn.model_selection          import train_test_split, cross_val_score
from sklearn.metrics                  import classification_report, accuracy_score
from sklearn.preprocessing            import LabelEncoder
from sklearn.pipeline                 import Pipeline

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = BASE_DIR   # save .pkl files right here in model/

# ─────────────────────────────────────────────────────────────────────────────
# 1.  BUILT-IN SYNTHETIC DATASET  (fallback when Kaggle files are absent)
# ─────────────────────────────────────────────────────────────────────────────
SYNTHETIC = [
    # ANXIETY
    {"text":"I can't stop worrying, my heart races all the time",          "emotion":"anxiety",    "sentiment":"negative","risk":"medium"},
    {"text":"The panic attacks are getting worse, I can't breathe",        "emotion":"anxiety",    "sentiment":"negative","risk":"high"},
    {"text":"I keep checking locks over and over, I know it's irrational", "emotion":"anxiety",    "sentiment":"negative","risk":"medium"},
    {"text":"Social situations terrify me, I rehearse for hours",          "emotion":"anxiety",    "sentiment":"negative","risk":"medium"},
    {"text":"Constant what-ifs are exhausting, always preparing for disaster","emotion":"anxiety", "sentiment":"negative","risk":"medium"},
    {"text":"I feel a knot in my stomach every morning before work",       "emotion":"anxiety",    "sentiment":"negative","risk":"medium"},
    {"text":"My mind races at night, can't sleep because of anxiety",      "emotion":"anxiety",    "sentiment":"negative","risk":"medium"},
    {"text":"Scared to leave the house, fear becomes overwhelming",        "emotion":"anxiety",    "sentiment":"negative","risk":"high"},
    {"text":"Had a small panic moment but breathing exercises helped",     "emotion":"anxiety",    "sentiment":"neutral", "risk":"low"},
    {"text":"Worried about tomorrow but I've prepared well",               "emotion":"anxiety",    "sentiment":"neutral", "risk":"low"},
    # DEPRESSION
    {"text":"I haven't gotten out of bed in three days, everything feels pointless","emotion":"depression","sentiment":"negative","risk":"high"},
    {"text":"I used to love painting, now I feel nothing when I look at my brushes","emotion":"depression","sentiment":"negative","risk":"high"},
    {"text":"I feel like a burden to everyone, they'd be better off without me",    "emotion":"depression","sentiment":"negative","risk":"high"},
    {"text":"Crying for no reason again, tears won't stop",                "emotion":"depression", "sentiment":"negative","risk":"medium"},
    {"text":"I feel completely empty inside, hollow, like nobody home",    "emotion":"depression", "sentiment":"negative","risk":"high"},
    {"text":"Getting out of bed takes every ounce of energy I have",       "emotion":"depression", "sentiment":"negative","risk":"medium"},
    {"text":"Canceling plans again, easier than pretending to be okay",    "emotion":"depression", "sentiment":"negative","risk":"medium"},
    {"text":"The future looks completely dark, nothing will ever improve", "emotion":"depression", "sentiment":"negative","risk":"high"},
    {"text":"I've been thinking everyone would be better off without me",  "emotion":"depression", "sentiment":"negative","risk":"high"},
    {"text":"Had a rough week but journaling helped me process things",    "emotion":"depression", "sentiment":"neutral", "risk":"low"},
    {"text":"Feeling down but I know these moods pass, called my therapist","emotion":"depression","sentiment":"neutral", "risk":"low"},
    # STRESS
    {"text":"Work deadlines are crushing me, 14 hour days and still behind","emotion":"stress",    "sentiment":"negative","risk":"medium"},
    {"text":"Juggling school work and family, failing at all three",       "emotion":"stress",     "sentiment":"negative","risk":"medium"},
    {"text":"Head is constantly pounding, stress giving me daily migraines","emotion":"stress",   "sentiment":"negative","risk":"medium"},
    {"text":"I've been snapping at my family because of work stress",      "emotion":"stress",     "sentiment":"negative","risk":"medium"},
    {"text":"Even on weekends I can't switch off, Monday always looming",  "emotion":"stress",     "sentiment":"negative","risk":"medium"},
    {"text":"Deadlines tight but team is supportive, managing well",       "emotion":"stress",     "sentiment":"neutral", "risk":"low"},
    {"text":"Stressful day but a workout helped, learning better coping",  "emotion":"stress",     "sentiment":"neutral", "risk":"low"},
    {"text":"Work is busy but proud of what we accomplished this quarter",  "emotion":"stress",     "sentiment":"positive","risk":"low"},
    # HOPEFUL
    {"text":"Started therapy and already seeing improvements, learning to challenge negative thoughts","emotion":"hopeful","sentiment":"positive","risk":"low"},
    {"text":"Today was a good day, went for a walk and felt present",      "emotion":"hopeful",    "sentiment":"positive","risk":"low"},
    {"text":"Recovery isn't linear but I'm making progress, six months ago I couldn't write this","emotion":"hopeful","sentiment":"positive","risk":"low"},
    {"text":"Meditation has changed my relationship with anxiety, more grounded now","emotion":"hopeful","sentiment":"positive","risk":"low"},
    {"text":"Learning it's okay to ask for help, opened up to a colleague and felt lighter","emotion":"hopeful","sentiment":"positive","risk":"low"},
    {"text":"After months of struggling finally had a week feeling like myself","emotion":"hopeful","sentiment":"positive","risk":"low"},
    {"text":"Celebrated six months without a panic attack today",          "emotion":"hopeful",    "sentiment":"positive","risk":"low"},
    # CALM
    {"text":"Feeling balanced today, sleep was good, ate well, took breaks","emotion":"calm",      "sentiment":"positive","risk":"low"},
    {"text":"Mindfulness is becoming second nature, more centered than in years","emotion":"calm", "sentiment":"positive","risk":"low"},
    {"text":"Nothing major happening emotionally, just steady and present","emotion":"calm",       "sentiment":"neutral", "risk":"low"},
    {"text":"Had a peaceful morning, coffee reading gentle music",         "emotion":"calm",       "sentiment":"positive","risk":"low"},
    {"text":"Everything feels manageable, clear head good sleep",          "emotion":"calm",       "sentiment":"positive","risk":"low"},
    # ANGER
    {"text":"I'm furious and don't know what to do with this rage",        "emotion":"anger",      "sentiment":"negative","risk":"medium"},
    {"text":"Snapped at my partner over something trivial, scared me",     "emotion":"anger",      "sentiment":"negative","risk":"medium"},
    {"text":"Burning resentment won't go away, poisoning everything",      "emotion":"anger",      "sentiment":"negative","risk":"medium"},
    {"text":"Had an argument but talked it through constructively",        "emotion":"anger",      "sentiment":"neutral", "risk":"low"},
    # LONELINESS
    {"text":"Surrounded by people but completely alone, no one really sees me","emotion":"loneliness","sentiment":"negative","risk":"high"},
    {"text":"Weeks since anyone checked in on me, silence is deafening",   "emotion":"loneliness", "sentiment":"negative","risk":"high"},
    {"text":"Moved to a new city, don't know anyone, isolation is crushing","emotion":"loneliness","sentiment":"negative","risk":"medium"},
    {"text":"Reached out to an old friend today, small connection helped enormously","emotion":"loneliness","sentiment":"positive","risk":"low"},
    # CRISIS
    {"text":"I don't see a point in continuing, the pain is too much",     "emotion":"depression", "sentiment":"negative","risk":"high"},
    {"text":"Been researching ways to hurt myself, don't know what's stopping me","emotion":"depression","sentiment":"negative","risk":"high"},
    {"text":"Wrote a goodbye note, deleted it, scared of my own thoughts", "emotion":"depression", "sentiment":"negative","risk":"high"},
]

# ─────────────────────────────────────────────────────────────────────────────
# 2.  EMOTION LABEL MAP  (Kaggle tweet_emotions → our 7 categories)
# ─────────────────────────────────────────────────────────────────────────────
EMOTION_MAP = {
    # Kaggle label       → our label
    "worry"             : "anxiety",
    "fear"              : "anxiety",
    "anger"             : "anger",
    "sadness"           : "depression",
    "love"              : "hopeful",
    "happiness"         : "hopeful",
    "fun"               : "hopeful",
    "enthusiasm"        : "hopeful",
    "relief"            : "calm",
    "neutral"           : "calm",
    "boredom"           : "depression",
    "hate"              : "anger",
    "empty"             : "loneliness",
    "lonely"            : "loneliness",
    "surprise"          : "calm",
    "joy"               : "hopeful",
    "disgust"           : "anger",
}

# ─────────────────────────────────────────────────────────────────────────────
# 3.  PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", " ", text)   # URLs, mentions, hashtags
    text = re.sub(r"[^a-z\s']",                " ", text)   # keep only letters + apostrophe
    text = re.sub(r"\s+",                      " ", text).strip()
    return text

# ─────────────────────────────────────────────────────────────────────────────
# 4.  DATA AUGMENTATION  (synonym swap)
# ─────────────────────────────────────────────────────────────────────────────
SYNONYMS = {
    "terrible":"awful","exhausted":"drained","scared":"frightened","angry":"furious",
    "sad":"sorrowful","happy":"joyful","worried":"anxious","tired":"fatigued",
    "hopeful":"optimistic","calm":"peaceful","lonely":"isolated","help":"support",
    "bad":"poor","good":"great","difficult":"hard","afraid":"scared",
}

def augment(text: str, n: int = 2) -> list:
    results = [text]
    words   = text.split()
    for _ in range(n):
        aug = [SYNONYMS.get(w.lower().rstrip(".,!?"), w) for w in words]
        results.append(" ".join(aug))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 5.  LOAD KAGGLE DATASETS  (graceful fallback if files missing)
# ─────────────────────────────────────────────────────────────────────────────
def load_emotion_dataset() -> pd.DataFrame:
    """Load Kaggle 'Emotion Detection from Text' dataset."""
    path = os.path.join(DATA_DIR, "tweet_emotions.csv")
    if not os.path.exists(path):
        print("  ℹ  tweet_emotions.csv not found — using synthetic data")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        # Columns: tweet_id, sentiment, content
        df = df[["sentiment", "content"]].dropna()
        df.columns = ["raw_emotion", "text"]
        df["emotion"] = df["raw_emotion"].str.lower().map(EMOTION_MAP)
        df = df.dropna(subset=["emotion"])
        df = df[df["emotion"].isin(["anxiety","depression","stress","hopeful","calm","anger","loneliness"])]
        # Balance: max 800 per class
        df = df.groupby("emotion").apply(lambda x: x.sample(min(len(x), 800), random_state=42)).reset_index(drop=True)
        print(f"  ✅ Loaded {len(df)} rows from tweet_emotions.csv")
        print(f"     Distribution: {df['emotion'].value_counts().to_dict()}")
        return df[["text","emotion"]]
    except Exception as e:
        print(f"  ⚠  Could not load tweet_emotions.csv: {e}")
        return pd.DataFrame()

def load_sentiment_dataset() -> pd.DataFrame:
    """Load Kaggle Sentiment140 dataset."""
    path = os.path.join(DATA_DIR, "sentiment140.csv")
    if not os.path.exists(path):
        print("  ℹ  sentiment140.csv not found — using synthetic data")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="latin-1", header=None,
                         names=["target","id","date","flag","user","text"])
        df = df[["target","text"]].dropna()
        df["sentiment"] = df["target"].map({0:"negative", 4:"positive"})
        df = df.dropna(subset=["sentiment"])
        # Sample 5000 per class
        df = df.groupby("sentiment").apply(lambda x: x.sample(min(len(x), 5000), random_state=42)).reset_index(drop=True)
        print(f"  ✅ Loaded {len(df)} rows from sentiment140.csv")
        return df[["text","sentiment"]]
    except Exception as e:
        print(f"  ⚠  Could not load sentiment140.csv: {e}")
        return pd.DataFrame()

def load_risk_dataset() -> pd.DataFrame:
    """Load Kaggle Suicide and Depression Detection dataset."""
    path = os.path.join(DATA_DIR, "suicide_detection.csv")
    if not os.path.exists(path):
        print("  ℹ  suicide_detection.csv not found — using synthetic data")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path).dropna()
        # Columns: text, class  (suicide / non-suicide)
        df.columns = [c.lower().strip() for c in df.columns]
        df["risk"] = df["class"].map({"suicide":"high","non-suicide":"low"})
        df = df.dropna(subset=["risk"])
        # Balance: 3000 per class
        df = df.groupby("risk").apply(lambda x: x.sample(min(len(x), 3000), random_state=42)).reset_index(drop=True)
        print(f"  ✅ Loaded {len(df)} rows from suicide_detection.csv")
        return df[["text","risk"]]
    except Exception as e:
        print(f"  ⚠  Could not load suicide_detection.csv: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# 6.  BUILD TRAINING DATAFRAMES
# ─────────────────────────────────────────────────────────────────────────────
def build_emotion_df() -> pd.DataFrame:
    kaggle = load_emotion_dataset()
    # Always include synthetic (manually crafted, clinically grounded)
    synth_rows = [{"text": t, "emotion": r["emotion"]}
                  for r in SYNTHETIC
                  for t in augment(r["text"])]
    synth_df = pd.DataFrame(synth_rows)
    if kaggle.empty:
        df = synth_df
    else:
        df = pd.concat([kaggle, synth_df], ignore_index=True)
    df["text"] = df["text"].apply(preprocess)
    df = df[df["text"].str.len() > 5].drop_duplicates("text")
    print(f"  Emotion training size: {len(df)}")
    return df

def build_sentiment_df() -> pd.DataFrame:
    kaggle = load_sentiment_dataset()
    synth_rows = [{"text": t, "sentiment": r["sentiment"]}
                  for r in SYNTHETIC
                  for t in augment(r["text"])]
    synth_df = pd.DataFrame(synth_rows)
    if kaggle.empty:
        df = synth_df
    else:
        df = pd.concat([kaggle, synth_df], ignore_index=True)
    df["text"] = df["text"].apply(preprocess)
    df = df[df["text"].str.len() > 5].drop_duplicates("text")
    # Add "neutral" class from synthetic
    neutral = pd.DataFrame([
        {"text": preprocess(r["text"]), "sentiment": "neutral"}
        for r in SYNTHETIC if r["sentiment"] == "neutral"
    ])
    df = pd.concat([df, neutral], ignore_index=True)
    print(f"  Sentiment training size: {len(df)}")
    return df

def build_risk_df() -> pd.DataFrame:
    kaggle = load_risk_dataset()
    synth_rows = []
    for r in SYNTHETIC:
        for t in augment(r["text"]):
            synth_rows.append({"text": t, "risk": r["risk"]})
    synth_df = pd.DataFrame(synth_rows)
    if kaggle.empty:
        df = synth_df
    else:
        # Kaggle only has high/low — add medium from synthetic
        medium = synth_df[synth_df["risk"] == "medium"]
        df     = pd.concat([kaggle, synth_df, medium], ignore_index=True)
    df["text"] = df["text"].apply(preprocess)
    df = df[df["text"].str.len() > 5].drop_duplicates("text")
    print(f"  Risk training size: {len(df)}")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 7.  TRAIN ONE CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
def train_classifier(df: pd.DataFrame, text_col: str, label_col: str,
                     name: str, label_encoder: LabelEncoder):
    print(f"\n{'─'*55}")
    print(f"  Training: {name}")
    print(f"{'─'*55}")

    X_text = df[text_col].values
    le     = label_encoder
    y      = le.fit_transform(df[label_col].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features = 8000,
            ngram_range  = (1, 3),
            sublinear_tf = True,
            min_df       = 1,
            analyzer     = "word",
        )),
        ("clf", LogisticRegression(
            max_iter     = 1000,
            C            = 1.5,
            class_weight = "balanced",
            solver       = "lbfgs",
            random_state = 42,
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    labels = le.classes_
    print(f"\n  Test Accuracy: {acc*100:.1f}%\n")
    print(classification_report(y_test, y_pred,
                                 target_names=labels,
                                 zero_division=0))

    # 3-fold CV score
    cv = cross_val_score(pipeline, X_text, y, cv=3, scoring="accuracy")
    print(f"  3-Fold CV Accuracy: {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%")

    return pipeline, acc

# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  MindWave NLP Model Trainer v2  (Kaggle Edition)")
    print("="*55)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,  exist_ok=True)

    # Label encoders
    le_emotion   = LabelEncoder()
    le_sentiment = LabelEncoder()
    le_risk      = LabelEncoder()

    # ── Emotion ──────────────────────────────────────────────
    print("\n📊 Building emotion dataset...")
    df_e = build_emotion_df()
    clf_emotion, acc_e = train_classifier(
        df_e, "text", "emotion", "Emotion Classifier", le_emotion)

    # ── Sentiment ────────────────────────────────────────────
    print("\n📊 Building sentiment dataset...")
    df_s = build_sentiment_df()
    clf_sentiment, acc_s = train_classifier(
        df_s, "text", "sentiment", "Sentiment Analyser", le_sentiment)

    # ── Risk ─────────────────────────────────────────────────
    print("\n📊 Building risk dataset...")
    df_r = build_risk_df()
    clf_risk, acc_r = train_classifier(
        df_r, "text", "risk", "Risk Level Predictor", le_risk)

    # ── Save ─────────────────────────────────────────────────
    print("\n💾 Saving models...")
    joblib.dump(clf_emotion,    os.path.join(MODEL_DIR, "emotion_classifier.pkl"))
    joblib.dump(clf_sentiment,  os.path.join(MODEL_DIR, "sentiment_classifier.pkl"))
    joblib.dump(clf_risk,       os.path.join(MODEL_DIR, "risk_classifier.pkl"))
    joblib.dump({
        "emotion"  : le_emotion,
        "sentiment": le_sentiment,
        "risk"     : le_risk,
    }, os.path.join(MODEL_DIR, "label_encoders.pkl"))

    meta = {
        "version"        : "2.0",
        "emotions"       : list(le_emotion.classes_),
        "sentiments"     : list(le_sentiment.classes_),
        "risk_levels"    : list(le_risk.classes_),
        "emotion_acc"    : round(acc_e, 4),
        "sentiment_acc"  : round(acc_s, 4),
        "risk_acc"       : round(acc_r, 4),
        "emotion_samples": len(df_e),
        "sentiment_samples": len(df_s),
        "risk_samples"   : len(df_r),
        "uses_kaggle"    : {
            "tweet_emotions"    : os.path.exists(os.path.join(DATA_DIR, "tweet_emotions.csv")),
            "sentiment140"      : os.path.exists(os.path.join(DATA_DIR, "sentiment140.csv")),
            "suicide_detection" : os.path.exists(os.path.join(DATA_DIR, "suicide_detection.csv")),
        }
    }
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "="*55)
    print(f"  ✅  ALL MODELS SAVED TO model/")
    print(f"  Emotion    accuracy : {acc_e*100:.1f}%")
    print(f"  Sentiment  accuracy : {acc_s*100:.1f}%")
    print(f"  Risk       accuracy : {acc_r*100:.1f}%")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
