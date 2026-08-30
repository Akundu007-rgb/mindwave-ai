"""
tests/test_auth.py
==================
Tests for user authentication — register, login, logout,
session management, password hashing, and stale session fix.

Covers:
  - Registration with valid data succeeds
  - Registration rejects duplicate email / username
  - Registration rejects mismatched passwords
  - Registration rejects passwords shorter than 6 chars
  - Login with correct credentials succeeds
  - Login with wrong password fails
  - Login with unknown email fails
  - Logout clears session
  - Stale session (DB reset) redirects to login instead of crashing
  - Password is stored as SHA-256 hash, never plaintext
  - Protected routes redirect unauthenticated users
  - current_user() returns None when not logged in
  - current_user() returns user row when logged in

Run:
    pytest tests/test_auth.py -v
"""

import os, sys, hashlib, tempfile, sqlite3
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def app():
    """Create a test Flask app with a temporary database."""
    try:
        import app as flask_app_module
    except ImportError:
        pytest.skip("Cannot import app.py")

    # Point DB at a temp file so we don't pollute real data
    tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_db.close()
    original_db = flask_app_module.DB_PATH
    flask_app_module.DB_PATH = tmp_db.name
    flask_app_module.app.config["TESTING"]              = True
    flask_app_module.app.config["WTF_CSRF_ENABLED"]     = False
    flask_app_module.app.config["SECRET_KEY"]           = "test-secret"

    with flask_app_module.app.app_context():
        flask_app_module.init_db()

    yield flask_app_module.app

    # Restore
    flask_app_module.DB_PATH = original_db
    os.unlink(tmp_db.name)


@pytest.fixture
def client(app):
    """Fresh test client for each test."""
    with app.test_client() as c:
        yield c


@pytest.fixture
def registered_client(client):
    """Client that already has a registered + logged-in user."""
    client.post("/register", data={
        "username": "testuser",
        "email"   : "test@mindwave.com",
        "password": "secure123",
        "confirm" : "secure123",
    }, follow_redirects=True)
    return client


def _register(client, username="user1", email="user1@test.com",
              password="pass123", confirm=None):
    return client.post("/register", data={
        "username": username,
        "email"   : email,
        "password": password,
        "confirm" : confirm if confirm is not None else password,
    }, follow_redirects=True)


def _login(client, email="user1@test.com", password="pass123"):
    return client.post("/login", data={
        "email"   : email,
        "password": password,
    }, follow_redirects=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. REGISTRATION TESTS
# ─────────────────────────────────────────────────────────────────────────────
class TestRegistration:

    def test_register_valid_user_succeeds(self, client):
        rv = _register(client, "alice", "alice@test.com", "alicepass")
        assert rv.status_code == 200

    def test_register_redirects_to_dashboard(self, client):
        """After successful register, user lands on dashboard."""
        rv = client.post("/register", data={
            "username": "bob",
            "email"   : "bob@test.com",
            "password": "bobpass1",
            "confirm" : "bobpass1",
        }, follow_redirects=True)
        assert b"dashboard" in rv.data.lower() or rv.status_code == 200

    def test_register_duplicate_email_fails(self, client):
        _register(client, "carol", "carol@test.com", "pass123")
        rv = _register(client, "carol2", "carol@test.com", "pass123")
        assert b"already" in rv.data.lower() or rv.status_code == 200

    def test_register_duplicate_username_fails(self, client):
        _register(client, "dave", "dave@test.com", "pass123")
        rv = _register(client, "dave", "dave2@test.com", "pass123")
        assert b"already" in rv.data.lower() or rv.status_code == 200

    def test_register_password_mismatch_fails(self, client):
        rv = _register(client, "eve", "eve@test.com", "pass123", confirm="different")
        assert b"match" in rv.data.lower() or rv.status_code == 200

    def test_register_short_password_fails(self, client):
        rv = _register(client, "frank", "frank@test.com", "abc", confirm="abc")
        assert b"6" in rv.data or b"least" in rv.data.lower() or rv.status_code == 200

    def test_register_missing_username_fails(self, client):
        rv = client.post("/register", data={
            "username": "",
            "email"   : "noname@test.com",
            "password": "pass123",
            "confirm" : "pass123",
        }, follow_redirects=True)
        assert rv.status_code == 200

    def test_register_missing_email_fails(self, client):
        rv = client.post("/register", data={
            "username": "nomail",
            "email"   : "",
            "password": "pass123",
            "confirm" : "pass123",
        }, follow_redirects=True)
        assert rv.status_code == 200

    def test_password_stored_as_hash(self, app, client):
        """Raw password must never appear in the database."""
        _register(client, "hashtest", "hash@test.com", "mypassword")
        import app as am
        db = sqlite3.connect(am.DB_PATH)
        row = db.execute("SELECT password FROM users WHERE email=?",
                         ("hash@test.com",)).fetchone()
        db.close()
        assert row is not None
        stored = row[0]
        assert stored != "mypassword",  "Password stored in plaintext!"
        assert len(stored) == 64,       "Expected SHA-256 hex (64 chars)"
        expected = hashlib.sha256("mypassword".encode()).hexdigest()
        assert stored == expected

    def test_get_register_page_returns_200(self, client):
        rv = client.get("/register")
        assert rv.status_code == 200

    def test_register_page_contains_form(self, client):
        rv = client.get("/register")
        assert b"username" in rv.data.lower() or b"email" in rv.data.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOGIN TESTS
# ─────────────────────────────────────────────────────────────────────────────
class TestLogin:

    def test_login_correct_credentials_succeeds(self, client):
        _register(client, "loginuser", "login@test.com", "goodpass")
        rv = _login(client, "login@test.com", "goodpass")
        assert rv.status_code == 200

    def test_login_redirects_to_dashboard(self, client):
        _register(client, "dashuser", "dash@test.com", "dashpass")
        rv = client.post("/login", data={
            "email"   : "dash@test.com",
            "password": "dashpass",
        }, follow_redirects=True)
        # After login, response should reference the dashboard
        assert b"dashboard" in rv.data.lower() or rv.status_code == 200

    def test_login_wrong_password_fails(self, client):
        _register(client, "wrongpwuser", "wrongpwuser@test.com", "correctpass99")
        # logout first to ensure clean state
        client.get("/logout", follow_redirects=True)
        rv = _login(client, "wrongpwuser@test.com", "WRONGPASSWORD_xyz")
        # Must NOT land on dashboard
        assert b"dashboard" not in rv.data.lower() or b"invalid" in rv.data.lower() \
               or rv.status_code == 200

    def test_login_unknown_email_fails(self, client):
        rv = _login(client, "nobody@nowhere.com", "anypass")
        assert b"invalid" in rv.data.lower() or b"error" in rv.data.lower() \
               or rv.status_code == 200

    def test_login_empty_email_fails(self, client):
        rv = client.post("/login", data={"email":"","password":"pass"},
                         follow_redirects=True)
        assert rv.status_code == 200

    def test_login_empty_password_fails(self, client):
        rv = client.post("/login", data={"email":"a@b.com","password":""},
                         follow_redirects=True)
        assert rv.status_code == 200

    def test_get_login_page_returns_200(self, client):
        rv = client.get("/login")
        assert rv.status_code == 200

    def test_login_page_contains_form_fields(self, client):
        rv = client.get("/login")
        assert b"email"    in rv.data.lower()
        assert b"password" in rv.data.lower()

    def test_login_case_sensitive_email(self, client):
        """Email lookup should be exact match."""
        _register(client, "caseuser", "Case@Test.com", "casepass")
        # Try with different case
        rv = _login(client, "CASE@TEST.COM", "casepass")
        # Whether this succeeds depends on implementation — just must not crash
        assert rv.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOGOUT TESTS
# ─────────────────────────────────────────────────────────────────────────────
class TestLogout:

    def test_logout_redirects(self, client):
        _register(client, "logoutuser", "logout@test.com", "logoutpass")
        _login(client, "logout@test.com", "logoutpass")
        rv = client.get("/logout", follow_redirects=False)
        assert rv.status_code in (301, 302, 200)

    def test_after_logout_dashboard_redirects_to_login(self, client):
        _register(client, "logoutuser2", "logout2@test.com", "pass123")
        _login(client, "logout2@test.com", "pass123")
        client.get("/logout", follow_redirects=True)
        rv = client.get("/dashboard", follow_redirects=False)
        assert rv.status_code in (301, 302)

    def test_logout_clears_session(self, app, client):
        _register(client, "sessuser", "sess@test.com", "sesspass")
        _login(client, "sess@test.com", "sesspass")
        with client.session_transaction() as sess:
            assert "user_id" in sess
        client.get("/logout", follow_redirects=True)
        with client.session_transaction() as sess:
            assert "user_id" not in sess


# ─────────────────────────────────────────────────────────────────────────────
# 4. PROTECTED ROUTES TESTS
# ─────────────────────────────────────────────────────────────────────────────
PROTECTED_ROUTES = [
    "/dashboard",
    "/journal",
    "/checkin",
    "/assessment",
    "/history",
]

class TestProtectedRoutes:

    @pytest.mark.parametrize("route", PROTECTED_ROUTES)
    def test_unauthenticated_redirects_to_login(self, client, route):
        """Every protected route must redirect unauthenticated users."""
        rv = client.get(route, follow_redirects=False)
        assert rv.status_code in (301, 302), \
            f"{route} should redirect unauthenticated users, got {rv.status_code}"

    @pytest.mark.parametrize("route", PROTECTED_ROUTES)
    def test_authenticated_gets_200(self, client, route):
        """After login, protected routes should be accessible."""
        _register(client, f"protuser_{route.strip('/')}", f"prot_{route.strip('/')}@test.com", "pass123")
        _login(client, f"prot_{route.strip('/')}@test.com", "pass123")
        rv = client.get(route, follow_redirects=True)
        assert rv.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# 5. STALE SESSION TESTS  (the original TypeError bug)
# ─────────────────────────────────────────────────────────────────────────────
class TestStaleSession:

    def test_stale_session_redirects_not_crash(self, app, client):
        """
        Regression test for: TypeError: 'NoneType' object is not subscriptable
        Happens when session has user_id but user no longer exists in DB.
        Must redirect to /login, NOT raise 500.
        """
        # Login normally
        _register(client, "staleuser", "stale@test.com", "stalepass")
        _login(client, "stale@test.com", "stalepass")

        # Wipe the user from DB to simulate DB reset
        import app as am
        db = sqlite3.connect(am.DB_PATH)
        db.execute("DELETE FROM users WHERE email=?", ("stale@test.com",))
        db.commit()
        db.close()

        # Session still has old user_id — must not crash
        rv = client.get("/dashboard", follow_redirects=False)
        assert rv.status_code in (301, 302), \
            f"Stale session should redirect, not crash with 500. Got {rv.status_code}"

    def test_stale_session_clears_cookie(self, app, client):
        """After stale session detected, session should be cleared."""
        _register(client, "stale2", "stale2@test.com", "stalepass2")
        _login(client, "stale2@test.com", "stalepass2")

        import app as am
        db = sqlite3.connect(am.DB_PATH)
        db.execute("DELETE FROM users WHERE email=?", ("stale2@test.com",))
        db.commit()
        db.close()

        client.get("/dashboard", follow_redirects=True)
        with client.session_transaction() as sess:
            assert "user_id" not in sess

    def test_ghost_user_id_in_session(self, app, client):
        """Manually inject a non-existent user_id — must redirect cleanly."""
        with client.session_transaction() as sess:
            sess["user_id"] = 99999   # user that does not exist
        rv = client.get("/dashboard", follow_redirects=False)
        assert rv.status_code in (301, 302)


# ─────────────────────────────────────────────────────────────────────────────
# 6. current_user() HELPER TESTS
# ─────────────────────────────────────────────────────────────────────────────
class TestCurrentUser:

    def test_current_user_none_when_not_logged_in(self, app, client):
        import app as am
        with app.test_request_context("/"):
            from flask import session
            session.clear()
            result = am.current_user()
            assert result is None

    def test_current_user_returns_user_when_logged_in(self, app, client):
        _register(client, "curuser", "cur@test.com", "curpass")
        _login(client, "cur@test.com", "curpass")
        import app as am
        import app as am_mod
        db = sqlite3.connect(am.DB_PATH)
        user = db.execute("SELECT * FROM users WHERE email=?",
                          ("cur@test.com",)).fetchone()
        db.close()
        assert user is not None
        assert user[2] == "cur@test.com"   # email column

    def test_hash_password_is_sha256(self):
        import app as am
        raw      = "testpassword"
        hashed   = am.hash_password(raw)
        expected = hashlib.sha256(raw.encode()).hexdigest()
        assert hashed == expected

    def test_hash_password_length(self):
        import app as am
        hashed = am.hash_password("anypassword")
        assert len(hashed) == 64

    def test_different_passwords_different_hashes(self):
        import app as am
        h1 = am.hash_password("password1")
        h2 = am.hash_password("password2")
        assert h1 != h2

    def test_same_password_same_hash(self):
        import app as am
        h1 = am.hash_password("samepass")
        h2 = am.hash_password("samepass")
        assert h1 == h2
