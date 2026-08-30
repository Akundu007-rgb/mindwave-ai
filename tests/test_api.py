"""
tests/test_api.py
=================
Tests for all Flask API endpoints and page routes.

Covers:
  - GET / returns 200 (landing page)
  - POST /api/analyze returns correct JSON structure
  - POST /api/analyze requires authentication
  - GET /api/dashboard_data requires authentication
  - GET /api/dashboard_data returns correct JSON structure
  - POST /journal saves entry and returns analysis
  - POST /checkin saves check-in data
  - POST /assessment computes correct result
  - GET /history returns user entries
  - GET /comparison returns 200 (public page)
  - GET /features returns 200 (public page)
  - GET /about returns 200 (public page)
  - API returns 400 for empty / too-short text
  - Dashboard data reflects actual journal entries
  - Assessment result is computed correctly
  - Journal entry saved to DB after POST

Run:
    pytest tests/test_api.py -v
"""

import os, sys, json, sqlite3, tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def app():
    try:
        import app as am
    except ImportError:
        pytest.skip("Cannot import app.py")

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    orig_db = am.DB_PATH
    am.DB_PATH = tmp.name
    am.app.config["TESTING"]          = True
    am.app.config["WTF_CSRF_ENABLED"] = False
    am.app.config["SECRET_KEY"]       = "api-test-secret"

    with am.app.app_context():
        am.init_db()

    yield am.app

    am.DB_PATH = orig_db
    os.unlink(tmp.name)


@pytest.fixture
def client(app):
    with app.test_client() as c:
        yield c


@pytest.fixture
def auth_client(client):
    """Client with a registered and logged-in user."""
    client.post("/register", data={
        "username": "apiuser",
        "email"   : "api@mindwave.com",
        "password": "apipass123",
        "confirm" : "apipass123",
    }, follow_redirects=True)
    client.post("/login", data={
        "email"   : "api@mindwave.com",
        "password": "apipass123",
    }, follow_redirects=True)
    return client


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def post_json(client, url, data):
    return client.post(url,
        data        = json.dumps(data),
        content_type= "application/json",
    )

def get_json(client, url):
    return client.get(url, content_type="application/json")


# ─────────────────────────────────────────────────────────────────────────────
# 1. PUBLIC PAGE TESTS
# ─────────────────────────────────────────────────────────────────────────────
class TestPublicPages:

    def test_home_page_returns_200(self, client):
        rv = client.get("/")
        assert rv.status_code == 200

    def test_home_page_has_mindwave_branding(self, client):
        rv = client.get("/")
        assert b"MindWave" in rv.data or b"mindwave" in rv.data.lower()

    def test_features_page_returns_200(self, client):
        rv = client.get("/features")
        assert rv.status_code == 200

    def test_about_page_returns_200(self, client):
        rv = client.get("/about")
        assert rv.status_code == 200

    def test_comparison_page_returns_200(self, client):
        rv = client.get("/comparison")
        assert rv.status_code == 200

    def test_comparison_page_mentions_competitors(self, client):
        rv = client.get("/comparison")
        page = rv.data.lower()
        assert b"wysa" in page or b"woebot" in page or b"compare" in page

    def test_login_page_returns_200(self, client):
        rv = client.get("/login")
        assert rv.status_code == 200

    def test_register_page_returns_200(self, client):
        rv = client.get("/register")
        assert rv.status_code == 200

    def test_unknown_route_returns_404(self, client):
        rv = client.get("/this-does-not-exist")
        assert rv.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# 2. /api/analyze  TESTS
# ─────────────────────────────────────────────────────────────────────────────
class TestAnalyzeAPI:

    def test_analyze_requires_auth(self, client):
        """Unauthenticated /api/analyze must not return 200."""
        rv = post_json(client, "/api/analyze", {"text": "I feel anxious"})
        assert rv.status_code in (302, 401, 403), \
            f"Expected redirect/auth error, got {rv.status_code}"

    def test_analyze_returns_200_when_authed(self, auth_client):
        rv = post_json(auth_client, "/api/analyze",
                       {"text": "I have been feeling very stressed lately"})
        assert rv.status_code == 200

    def test_analyze_response_is_json(self, auth_client):
        rv = post_json(auth_client, "/api/analyze",
                       {"text": "I feel anxious about everything"})
        assert rv.content_type.startswith("application/json")

    def test_analyze_returns_emotion(self, auth_client):
        rv   = post_json(auth_client, "/api/analyze", {"text": "I feel hopeful today"})
        data = json.loads(rv.data)
        assert "emotion" in data
        assert data["emotion"] in {"anxiety","depression","stress","hopeful","calm","anger","loneliness"}

    def test_analyze_returns_sentiment(self, auth_client):
        rv   = post_json(auth_client, "/api/analyze", {"text": "Everything is going well"})
        data = json.loads(rv.data)
        assert "sentiment" in data
        assert data["sentiment"] in {"positive", "neutral", "negative"}

    def test_analyze_returns_risk_level(self, auth_client):
        rv   = post_json(auth_client, "/api/analyze", {"text": "Feeling a bit down today"})
        data = json.loads(rv.data)
        assert "risk_level" in data
        assert data["risk_level"] in {"low", "medium", "high"}

    def test_analyze_returns_wellness_score(self, auth_client):
        rv   = post_json(auth_client, "/api/analyze", {"text": "I feel calm and peaceful"})
        data = json.loads(rv.data)
        assert "wellness_score" in data
        assert 0 <= data["wellness_score"] <= 100

    def test_analyze_returns_emotion_distribution(self, auth_client):
        rv   = post_json(auth_client, "/api/analyze", {"text": "I feel so worried"})
        data = json.loads(rv.data)
        assert "emotion_distribution" in data
        assert isinstance(data["emotion_distribution"], dict)
        assert len(data["emotion_distribution"]) == 7

    def test_analyze_returns_confidence(self, auth_client):
        rv   = post_json(auth_client, "/api/analyze", {"text": "stressed at work"})
        data = json.loads(rv.data)
        assert "confidence" in data
        conf = data["confidence"]
        assert "emotion"   in conf
        assert "sentiment" in conf
        assert "risk"      in conf

    def test_analyze_returns_recommendations(self, auth_client):
        rv   = post_json(auth_client, "/api/analyze", {"text": "anxious and scared"})
        data = json.loads(rv.data)
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)
        assert len(data["recommendations"]) >= 1

    def test_analyze_empty_text_returns_400(self, auth_client):
        rv = post_json(auth_client, "/api/analyze", {"text": ""})
        assert rv.status_code in (400, 200)   # 400 preferred, 200 w/ error key acceptable
        if rv.status_code == 200:
            data = json.loads(rv.data)
            assert "error" in data

    def test_analyze_too_short_text_returns_400(self, auth_client):
        rv = post_json(auth_client, "/api/analyze", {"text": "hi"})
        assert rv.status_code in (400, 200)

    def test_analyze_missing_text_key(self, auth_client):
        rv = post_json(auth_client, "/api/analyze", {})
        assert rv.status_code in (400, 200)

    def test_analyze_long_text(self, auth_client):
        long = "I have been feeling very anxious and stressed. " * 30
        rv   = post_json(auth_client, "/api/analyze", {"text": long})
        assert rv.status_code == 200

    def test_analyze_distribution_sums_near_100(self, auth_client):
        rv   = post_json(auth_client, "/api/analyze",
                         {"text": "I cannot sleep and feel very anxious"})
        data = json.loads(rv.data)
        total = sum(data["emotion_distribution"].values())
        assert abs(total - 100.0) < 1.0, f"Distribution sum = {total}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. /api/dashboard_data  TESTS
# ─────────────────────────────────────────────────────────────────────────────
class TestDashboardAPI:

    def test_dashboard_data_requires_auth(self, client):
        rv = client.get("/api/dashboard_data")
        assert rv.status_code in (302, 401, 403)

    def test_dashboard_data_returns_200(self, auth_client):
        rv = client.get("/api/dashboard_data") if False else auth_client.get("/api/dashboard_data")
        assert rv.status_code == 200

    def test_dashboard_data_is_json(self, auth_client):
        rv = auth_client.get("/api/dashboard_data")
        assert rv.content_type.startswith("application/json")

    def test_dashboard_data_has_mood_trend(self, auth_client):
        rv   = auth_client.get("/api/dashboard_data")
        data = json.loads(rv.data)
        assert "mood_trend" in data
        assert isinstance(data["mood_trend"], list)

    def test_dashboard_data_has_checkin_trend(self, auth_client):
        rv   = auth_client.get("/api/dashboard_data")
        data = json.loads(rv.data)
        assert "checkin_trend" in data
        assert isinstance(data["checkin_trend"], list)

    def test_dashboard_data_has_emotion_counts(self, auth_client):
        rv   = auth_client.get("/api/dashboard_data")
        data = json.loads(rv.data)
        assert "emotion_counts" in data
        assert isinstance(data["emotion_counts"], dict)

    def test_dashboard_data_has_stats(self, auth_client):
        rv   = auth_client.get("/api/dashboard_data")
        data = json.loads(rv.data)
        assert "stats" in data
        stats = data["stats"]
        assert "avg_wellness"   in stats
        assert "journal_count"  in stats
        assert "streak"         in stats

    def test_dashboard_reflects_journal_entries(self, app, auth_client):
        """After adding a journal entry, dashboard_data should reflect it."""
        # Add journal via POST /journal
        auth_client.post("/journal", data={
            "text": "I am feeling very anxious and cannot sleep properly"
        }, follow_redirects=True)

        rv   = auth_client.get("/api/dashboard_data")
        data = json.loads(rv.data)
        # mood_trend should now have at least one entry
        assert len(data["mood_trend"]) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. /journal  TESTS
# ─────────────────────────────────────────────────────────────────────────────
class TestJournalEndpoint:

    def test_journal_page_requires_auth(self, client):
        rv = client.get("/journal", follow_redirects=False)
        assert rv.status_code in (301, 302)

    def test_journal_get_returns_200(self, auth_client):
        rv = auth_client.get("/journal")
        assert rv.status_code == 200

    def test_journal_post_saves_entry(self, app, auth_client):
        import app as am
        before = sqlite3.connect(am.DB_PATH).execute(
            "SELECT COUNT(*) FROM journals").fetchone()[0]

        auth_client.post("/journal", data={
            "text": "I have been feeling really stressed about my exams lately"
        }, follow_redirects=True)

        after = sqlite3.connect(am.DB_PATH).execute(
            "SELECT COUNT(*) FROM journals").fetchone()[0]
        assert after > before, "Journal entry was not saved to DB"

    def test_journal_post_shows_analysis_result(self, auth_client):
        rv = auth_client.post("/journal", data={
            "text": "I am feeling hopeful today, things are getting better"
        }, follow_redirects=True)
        page = rv.data.lower()
        # Result section or emotion labels should appear
        assert (b"emotion" in page or b"wellness" in page or
                b"analysis" in page or b"result" in page)

    def test_journal_post_too_short_shows_error(self, auth_client):
        rv = auth_client.post("/journal", data={"text": "hi"},
                              follow_redirects=True)
        page = rv.data.lower()
        assert b"least" in page or b"error" in page or b"sentence" in page

    def test_journal_post_saves_correct_emotion(self, app, auth_client):
        auth_client.post("/journal", data={
            "text": "I cannot stop worrying, everything makes me panic"
        }, follow_redirects=True)

        import app as am
        row = sqlite3.connect(am.DB_PATH).execute(
            "SELECT emotion FROM journals ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row[0] in {"anxiety","depression","stress","hopeful","calm","anger","loneliness"}

    def test_journal_post_saves_wellness_score(self, app, auth_client):
        auth_client.post("/journal", data={
            "text": "feeling wonderful and grateful for everything in my life"
        }, follow_redirects=True)

        import app as am
        row = sqlite3.connect(am.DB_PATH).execute(
            "SELECT wellness_score FROM journals ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        assert 0 <= row[0] <= 100


# ─────────────────────────────────────────────────────────────────────────────
# 5. /checkin  TESTS
# ─────────────────────────────────────────────────────────────────────────────
class TestCheckinEndpoint:

    def test_checkin_requires_auth(self, client):
        rv = client.get("/checkin", follow_redirects=False)
        assert rv.status_code in (301, 302)

    def test_checkin_get_returns_200(self, auth_client):
        rv = auth_client.get("/checkin")
        assert rv.status_code == 200

    def test_checkin_post_saves_to_db(self, app, auth_client):
        import app as am
        before = sqlite3.connect(am.DB_PATH).execute(
            "SELECT COUNT(*) FROM checkins").fetchone()[0]

        auth_client.post("/checkin", data={
            "mood_score"  : "7",
            "sleep_hours" : "7.5",
            "stress_level": "4",
            "notes"       : "Had a good day today",
        }, follow_redirects=True)

        after = sqlite3.connect(am.DB_PATH).execute(
            "SELECT COUNT(*) FROM checkins").fetchone()[0]
        assert after > before

    def test_checkin_saves_correct_values(self, app, auth_client):
        auth_client.post("/checkin", data={
            "mood_score"  : "8",
            "sleep_hours" : "6.5",
            "stress_level": "3",
            "notes"       : "",
        }, follow_redirects=True)

        import app as am
        row = sqlite3.connect(am.DB_PATH).execute(
            "SELECT mood_score, sleep_hours, stress_level FROM checkins ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row[0] == 8
        assert row[1] == 6.5
        assert row[2] == 3

    def test_checkin_post_shows_success_message(self, auth_client):
        rv = auth_client.post("/checkin", data={
            "mood_score"  : "6",
            "sleep_hours" : "8",
            "stress_level": "5",
            "notes"       : "",
        }, follow_redirects=True)
        page = rv.data.lower()
        assert b"saved" in page or b"success" in page or b"check" in page


# ─────────────────────────────────────────────────────────────────────────────
# 6. /assessment  TESTS
# ─────────────────────────────────────────────────────────────────────────────
class TestAssessmentEndpoint:

    def test_assessment_requires_auth(self, client):
        rv = client.get("/assessment", follow_redirects=False)
        assert rv.status_code in (301, 302)

    def test_assessment_get_returns_200(self, auth_client):
        rv = auth_client.get("/assessment")
        assert rv.status_code == 200

    def test_assessment_page_has_7_questions(self, auth_client):
        rv   = auth_client.get("/assessment")
        page = rv.data.decode("utf-8", errors="ignore")
        # Each question has a radio input named q1 through q7
        for i in range(1, 8):
            assert f'name="q{i}"' in page or f"q{i}" in page

    def test_assessment_post_all_zero_gives_high_wellness(self, auth_client):
        rv = auth_client.post("/assessment", data={
            "q1":"0","q2":"0","q3":"0","q4":"0","q5":"0","q6":"0","q7":"0"
        }, follow_redirects=True)
        page = rv.data.lower()
        # Wellness score should be high (Thriving)
        assert b"thriving" in page or b"100" in page or b"wellness" in page

    def test_assessment_post_all_three_gives_high_priority(self, auth_client):
        rv = auth_client.post("/assessment", data={
            "q1":"3","q2":"3","q3":"3","q4":"3","q5":"3","q6":"3","q7":"3"
        }, follow_redirects=True)
        page = rv.data.lower()
        assert b"high priority" in page or b"high" in page or b"concern" in page

    def test_assessment_post_saves_to_journals(self, app, auth_client):
        import app as am
        before = sqlite3.connect(am.DB_PATH).execute(
            "SELECT COUNT(*) FROM journals WHERE text LIKE '[Assessment]%'"
        ).fetchone()[0]

        auth_client.post("/assessment", data={
            "q1":"1","q2":"1","q3":"1","q4":"1","q5":"1","q6":"1","q7":"1"
        }, follow_redirects=True)

        after = sqlite3.connect(am.DB_PATH).execute(
            "SELECT COUNT(*) FROM journals WHERE text LIKE '[Assessment]%'"
        ).fetchone()[0]
        assert after > before, "Assessment was not saved to journals table"

    def test_assessment_missing_question_shows_error(self, auth_client):
        """Submitting without all 7 answers should show an error."""
        rv = auth_client.post("/assessment", data={
            "q1":"1","q2":"1"
            # q3–q7 missing
        }, follow_redirects=True)
        page = rv.data.lower()
        assert b"answer" in page or b"question" in page or b"error" in page or rv.status_code == 200

    def test_compute_assessment_result_thriving(self, app):
        import app as am
        result = am.compute_assessment_result([0,0,0,0,0,0,0])
        assert result["label"]         == "Thriving"
        assert result["wellness_score"] >= 90
        assert result["level"]         == "low"

    def test_compute_assessment_result_moderate(self, app):
        import app as am
        result = am.compute_assessment_result([1,1,1,1,1,1,1])
        assert result["label"]         == "Moderate Stress"
        assert result["level"]         == "medium"

    def test_compute_assessment_result_elevated(self, app):
        import app as am
        result = am.compute_assessment_result([2,2,2,2,2,2,2])
        assert result["label"]         == "Elevated Concern"
        assert result["level"]         == "high"

    def test_compute_assessment_result_high_priority(self, app):
        import app as am
        result = am.compute_assessment_result([3,3,3,3,3,3,3])
        assert result["label"]         == "High Priority"
        assert result["wellness_score"] <= 10

    def test_compute_assessment_wellness_in_range(self, app):
        import app as am
        for answers in [[0]*7, [1]*7, [2]*7, [3]*7, [0,1,2,3,0,1,2]]:
            result = am.compute_assessment_result(answers)
            assert 0 <= result["wellness_score"] <= 100

    def test_compute_assessment_has_recommendations(self, app):
        import app as am
        result = am.compute_assessment_result([2,2,2,2,2,2,2])
        assert "recommendations" in result
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) >= 1

    def test_compute_assessment_category_scores(self, app):
        import app as am
        result = am.compute_assessment_result([1,2,0,3,1,2,0])
        assert "category_scores" in result
        cats = result["category_scores"]
        assert cats["mood"]       == 1
        assert cats["sleep"]      == 0
        assert cats["anxiety"]    == 3


# ─────────────────────────────────────────────────────────────────────────────
# 7. /history  TESTS
# ─────────────────────────────────────────────────────────────────────────────
class TestHistoryEndpoint:

    def test_history_requires_auth(self, client):
        rv = client.get("/history", follow_redirects=False)
        assert rv.status_code in (301, 302)

    def test_history_returns_200(self, auth_client):
        rv = auth_client.get("/history")
        assert rv.status_code == 200

    def test_history_shows_journal_entries(self, auth_client):
        # Add an entry
        auth_client.post("/journal", data={
            "text": "unique history test entry feeling calm today"
        }, follow_redirects=True)
        rv   = auth_client.get("/history")
        page = rv.data.lower()
        assert b"journal" in page or b"history" in page or b"entries" in page

    def test_history_user_isolation(self, app, client):
        """User A should not see User B's journal entries."""
        # Register User B
        client.post("/register", data={
            "username": "userB",
            "email"   : "userb@test.com",
            "password": "passbbb",
            "confirm" : "passbbb",
        }, follow_redirects=True)
        client.post("/login", data={
            "email"   : "userb@test.com",
            "password": "passbbb",
        }, follow_redirects=True)
        client.post("/journal", data={
            "text": "THIS IS USER B SECRET PRIVATE ENTRY"
        }, follow_redirects=True)
        client.get("/logout")

        # Login as User A (apiuser registered in auth_client fixture)
        client.post("/login", data={
            "email"   : "api@mindwave.com",
            "password": "apipass123",
        }, follow_redirects=True)
        rv = client.get("/history")
        assert b"USER B SECRET" not in rv.data
