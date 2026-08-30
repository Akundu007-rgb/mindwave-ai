# MindWave — Tests & Kaggle Dataset Guide

## Test Results
- test_models.py  : 61 tests  ✅ all pass
- test_auth.py    : 42 tests  ✅ all pass
- test_api.py     : 62 tests  ✅ all pass
- TOTAL           : 165 tests ✅ all pass

## Run Tests
```bash
cd mindwave_app
pip install pytest
python model/train_model.py    # required first
pytest tests/ -v
```

## Kaggle Datasets — Download & Place in data/

### Dataset 1 — Emotion Classifier (Model 1)
Name  : Emotion Detection from Text
URL   : https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text
File  : tweet_emotions.csv
Place : mindwave_app/data/tweet_emotions.csv
Size  : ~40,000 tweets, 13 emotion labels
Use   : Trains the 7-category emotion classifier

### Dataset 2 — Sentiment Analyser (Model 2)
Name  : Sentiment140
URL   : https://www.kaggle.com/datasets/kazanova/sentiment140
File  : training.1600000.processed.noemoticon.csv
Rename: save as mindwave_app/data/sentiment140.csv
Size  : 1.6M tweets, positive/negative labels
Use   : Trains the sentiment analyser

### Dataset 3 — Risk Predictor (Model 3)
Name  : Suicide and Depression Detection
URL   : https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch
File  : Suicide_Detection.csv
Rename: save as mindwave_app/data/suicide_detection.csv
Size  : ~232,000 Reddit posts, suicide/non-suicide
Use   : Trains the risk level predictor

## After Downloading

1. Place all 3 CSV files in mindwave_app/data/
2. Run: python model/train_model.py
3. The script auto-detects Kaggle files and uses them
4. Expected accuracy with Kaggle data:
   - Emotion    : ~70-80%
   - Sentiment  : ~82-88%
   - Risk       : ~88-93%

## Without Kaggle Files
The trainer falls back to the built-in 51-sample synthetic
dataset automatically. App works but accuracy is lower.

## Test Coverage
tests/test_models.py
  TestModelFiles        - all 5 pkl files exist + meta json valid
  TestPreprocessing     - 9 text cleaning tests
  TestVectorizer        - transform shape + vocabulary
  TestEmotionClassifier - valid labels, proba shape, batch prediction
  TestSentimentClassifier - positive/negative/neutral coverage
  TestRiskClassifier    - low/medium/high coverage
  TestWellnessScore     - range, boundary, formula tests
  TestAnalyzeText       - full integration: keys, ranges, recs

tests/test_auth.py
  TestRegistration      - valid, duplicate, mismatch, short pw
  TestLogin             - correct/wrong credentials, empty fields
  TestLogout            - redirect, session clear
  TestProtectedRoutes   - 5 routes × 2 (auth/unauth) = 10 tests
  TestStaleSession      - the NoneType bug regression tests
  TestCurrentUser       - hash correctness, None when not logged in

tests/test_api.py
  TestPublicPages       - 9 public route tests
  TestAnalyzeAPI        - 14 /api/analyze tests
  TestDashboardAPI      - 8 /api/dashboard_data tests
  TestJournalEndpoint   - 7 journal POST/GET tests
  TestCheckinEndpoint   - 5 check-in tests
  TestAssessmentEndpoint- 11 assessment + compute_result tests
  TestHistoryEndpoint   - 4 history + user isolation tests
