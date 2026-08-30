"""
tests/conftest.py
=================
Shared pytest configuration and fixtures available
to all test files automatically.
"""

import os, sys, tempfile, sqlite3
import pytest

# Make sure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "integration: marks tests requiring trained models")
    config.addinivalue_line("markers", "auth: marks authentication-related tests")


@pytest.fixture(scope="session")
def models_trained():
    """Skip entire session if models aren't trained."""
    model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
    required  = ["emotion_classifier.pkl", "sentiment_classifier.pkl",
                  "risk_classifier.pkl",   "tfidf_vectorizer.pkl",
                  "label_encoders.pkl"]
    missing   = [f for f in required
                 if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        pytest.skip(f"Run 'python model/train_model.py' first. Missing: {missing}")
    return True


@pytest.fixture(scope="session")
def temp_db_path():
    """Session-scoped temp database path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture(scope="session")
def flask_app(temp_db_path):
    """Session-scoped Flask test app."""
    try:
        import app as am
    except ImportError:
        pytest.skip("Cannot import app.py — make sure you're running from project root")

    original_db = am.DB_PATH
    am.DB_PATH  = temp_db_path
    am.app.config.update({
        "TESTING"         : True,
        "WTF_CSRF_ENABLED": False,
        "SECRET_KEY"      : "conftest-secret-key",
    })
    with am.app.app_context():
        am.init_db()

    yield am.app

    am.DB_PATH = original_db


@pytest.fixture
def test_client(flask_app):
    """Fresh test client per test."""
    with flask_app.test_client() as c:
        yield c


@pytest.fixture
def logged_in_client(test_client):
    """Test client with a pre-registered + logged-in user."""
    test_client.post("/register", data={
        "username": "conftest_user",
        "email"   : "conftest@mindwave.com",
        "password": "conftest_pass",
        "confirm" : "conftest_pass",
    }, follow_redirects=True)
    test_client.post("/login", data={
        "email"   : "conftest@mindwave.com",
        "password": "conftest_pass",
    }, follow_redirects=True)
    return test_client
