"""Authentication: secure password hashing + a small user store.

Passwords are hashed with PBKDF2-HMAC-SHA256 (stdlib) using a per-user random
salt — plaintext passwords are never stored. The store persists to a JSON file
and seeds a demo account on first run. Session handling itself lives in main.py
(Starlette SessionMiddleware); this module only owns credentials.
"""

import hashlib
import hmac
import json
import secrets
from pathlib import Path

from config import Settings
from logging_config import get_logger

log = get_logger(__name__)

_ALGO = "pbkdf2_sha256"
_ITERATIONS = 200_000
_BACKEND_DIR = Path(__file__).resolve().parent


def hash_password(password: str, salt: str | None = None) -> str:
    """Return a self-describing hash string: algo$iterations$salt$hash."""
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _ITERATIONS
    )
    return f"{_ALGO}${_ITERATIONS}${salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time verify a password against a stored hash string."""
    try:
        algo, iterations, salt, expected = stored.split("$")
        if algo != _ALGO:
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iterations)
        )
        return hmac.compare_digest(dk.hex(), expected)
    except (ValueError, AttributeError):
        return False


class UserStore:
    """Email → password-hash store, persisted as JSON. Emails are case-insensitive."""

    def __init__(self, path: str, seed: tuple[str, str] | None = None):
        p = Path(path)
        self.path = p if p.is_absolute() else _BACKEND_DIR / p
        self.users: dict[str, str] = {}
        self._load()
        if seed:
            email, password = seed
            if email and email.lower() not in self.users:
                self.add_user(email, password)
                log.info("Seeded demo account: %s", email.lower())

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.users = json.loads(self.path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                log.warning("Could not read user store at %s; starting empty.", self.path)
                self.users = {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.users, indent=2), "utf-8")

    def add_user(self, email: str, password: str) -> None:
        self.users[email.strip().lower()] = hash_password(password)
        self._save()

    def verify(self, email: str, password: str) -> bool:
        stored = self.users.get((email or "").strip().lower())
        return bool(stored) and verify_password(password, stored)


_store: UserStore | None = None


def get_user_store(settings: Settings) -> UserStore:
    global _store
    if _store is None:
        _store = UserStore(
            settings.users_db, seed=(settings.demo_email, settings.demo_password)
        )
    return _store


def reset_user_store() -> None:
    """Drop the cached store (tests)."""
    global _store
    _store = None
