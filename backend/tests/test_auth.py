"""Authentication: password hashing, login, session gating, logout.

Runs in MockLLM mode (no Gemini) with an isolated temp user store so it never
touches the real users.json or the network. Set up BEFORE importing main so the
app picks up the env.
"""

import importlib
import os
import tempfile

import pytest

os.environ["MOCK_LLM"] = "true"
os.environ["GEMINI_API_KEY"] = ""
os.environ["SESSION_SECRET"] = "test-secret-key"
os.environ["USERS_DB"] = tempfile.mktemp(suffix="_users.json")

from fastapi.testclient import TestClient  # noqa: E402

import auth  # noqa: E402
import config  # noqa: E402

# Ensure a fresh settings + user store under the test env.
config.get_settings.cache_clear()
auth.reset_user_store()

import main  # noqa: E402

DEMO = {"email": "admin@test.com", "password": "admin123"}

JD = "Backend Engineer\nRequired skills: Python, Java\nRequires 2-8 years.\n"
RESUME = (
    "Aarav Sharma\nSoftware Engineer\naarav@example.com\n"
    "Skills\n- Python, Java\n"
    "Experience\nEngineer, Acme (2020 - Present)\n- Built services in Python and Java.\n"
    "Education\nB.Tech CS (2018)\n"
)


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def logged_in_client():
    c = TestClient(main.app)
    r = c.post("/api/auth/login", data=DEMO)
    assert r.status_code == 200
    return c


# --- password hashing ------------------------------------------------------

def test_password_is_hashed_not_plaintext():
    h = auth.hash_password("admin123")
    assert "admin123" not in h
    assert h.startswith("pbkdf2_sha256$")
    assert auth.verify_password("admin123", h)
    assert not auth.verify_password("wrong", h)


def test_demo_account_seeded_with_hash():
    store = auth.get_user_store(config.get_settings())
    stored = store.users.get("admin@test.com")
    assert stored and "admin123" not in stored      # only the hash is stored
    assert store.verify("admin@test.com", "admin123")
    assert store.verify("ADMIN@test.com", "admin123")  # case-insensitive email


# --- login success / failure ----------------------------------------------

def test_login_success(client):
    r = client.post("/api/auth/login", data=DEMO)
    assert r.status_code == 200
    assert r.json()["email"] == "admin@test.com"
    # session cookie set
    assert main.settings.session_cookie in r.cookies or client.cookies


def test_login_failure_wrong_password(client):
    r = client.post("/api/auth/login",
                    data={"email": "admin@test.com", "password": "nope"})
    assert r.status_code == 401
    # no session established
    assert client.get("/api/auth/me").status_code == 401


def test_login_failure_unknown_user(client):
    r = client.post("/api/auth/login",
                    data={"email": "ghost@test.com", "password": "admin123"})
    assert r.status_code == 401


# --- protected routes ------------------------------------------------------

def test_screen_blocked_without_login(client):
    r = client.post("/api/screen", data={"job_description": JD},
                    files={"resume": ("r.txt", RESUME, "text/plain")})
    assert r.status_code == 401


def test_bulk_blocked_without_login(client):
    r = client.post("/api/screen-bulk", data={"job_description": JD},
                    files={"resumes": ("r.txt", RESUME, "text/plain")})
    assert r.status_code == 401


def test_me_blocked_without_login(client):
    assert client.get("/api/auth/me").status_code == 401


def test_dashboard_redirects_to_login_when_anonymous(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/login"


# --- access after login ----------------------------------------------------

def test_me_after_login(logged_in_client):
    r = logged_in_client.get("/api/auth/me")
    assert r.status_code == 200 and r.json()["email"] == "admin@test.com"


def test_single_screen_works_after_login(logged_in_client):
    r = logged_in_client.post("/api/screen", data={"job_description": JD},
                              files={"resume": ("r.txt", RESUME, "text/plain")})
    assert r.status_code == 200
    body = r.json()
    assert body["role_title"] == "Backend Engineer"
    assert body["verdict"] in ("FIT", "UNFIT")


def test_bulk_screen_and_csv_work_after_login(logged_in_client):
    files = [
        ("resumes", ("a.txt", RESUME, "text/plain")),
        ("resumes", ("b.txt", RESUME.replace("Aarav Sharma", "Bob Lee"), "text/plain")),
    ]
    r = logged_in_client.post("/api/screen-bulk",
                              data={"job_description": JD}, files=files)
    assert r.status_code == 200
    assert r.json()["total"] == 2 and r.json()["succeeded"] == 2

    # CSV export also gated + works after login
    r_csv = logged_in_client.post(
        "/api/screen-bulk",
        data={"job_description": JD, "response_format": "csv"}, files=files,
    )
    assert r_csv.status_code == 200
    assert "text/csv" in r_csv.headers["content-type"]
    assert r_csv.text.splitlines()[0].startswith("Rank,Candidate,Score")


def test_dashboard_served_when_logged_in(logged_in_client):
    r = logged_in_client.get("/", follow_redirects=False)
    assert r.status_code == 200
    assert "JobFit" in r.text or "Job" in r.text


# --- logout blocks access again -------------------------------------------

def test_logout_then_blocked(logged_in_client):
    assert logged_in_client.get("/api/auth/me").status_code == 200
    assert logged_in_client.post("/api/auth/logout").status_code == 200
    # session cleared -> protected routes blocked again
    assert logged_in_client.get("/api/auth/me").status_code == 401
    r = logged_in_client.post("/api/screen", data={"job_description": JD},
                              files={"resume": ("r.txt", RESUME, "text/plain")})
    assert r.status_code == 401


# --- session validation ----------------------------------------------------

def test_session_isolated_between_clients(logged_in_client):
    # A separate client without the session cookie is not authenticated.
    other = TestClient(main.app)
    assert other.get("/api/auth/me").status_code == 401
