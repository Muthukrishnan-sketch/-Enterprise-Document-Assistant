"""
auth_store.py
==============
Handles user accounts and login: who can sign in, and which
department folders they're allowed to see. Stored in the same
lightweight SQLite pattern as chat_store.py, but in its OWN database
file (users.db) - kept separate on purpose, so nothing that resets
or corrupts chat_history.db could ever accidentally touch user
accounts, or vice versa.

PASSWORD SECURITY, EXPLAINED (this matters - read this before editing):

Passwords are NEVER stored as plain text - only a salted hash is
stored, in the `password_hash` column. Two things make this safe:

1. SALT: a random value generated per user (`secrets.token_hex(16)`)
   that gets mixed into the password before hashing. Without a salt,
   two users who happen to pick the same password would end up with
   the IDENTICAL stored hash - and an attacker with a precomputed
   table of common password->hash pairs ("rainbow table") could crack
   every account using that password at once. A random salt per user
   means the same password hashes to a completely different value for
   every account, so that shortcut doesn't work.

2. PBKDF2 (not a plain hash): a single round of a fast hash function
   like SHA-256 is deliberately too cheap - modern hardware can try
   billions of guesses per second against it. PBKDF2 repeats the hash
   thousands of times on purpose (`PBKDF2_ITERATIONS` below), which
   makes each individual guess meaningfully slower. A real login still
   feels instant (one hash), but an attacker trying millions of
   guesses against a stolen database takes dramatically longer.

This file has no idea what a department folder's CONTENTS are, or
what a RAG pipeline is - it only knows a username maps to a password
hash and a list of allowed folder names. app.py is what actually
ENFORCES this, by reading a logged-in user's folder list and passing
it into rag_core.answer_question_conversational() as `allowed_folders`
- this file only stores the rule, it doesn't apply it.
"""

import sqlite3
import json
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager

DB_FILE = "users.db"
PBKDF2_ITERATIONS = 200_000  # deliberately slow - see module docstring
MAX_LOGIN_ATTEMPTS = 5        # failed attempts allowed before lockout
LOGIN_LOCKOUT_MINUTES = 15    # how long a lockout lasts
REMEMBER_TOKEN_DAYS = 30      # how long a "remember me" login lasts


class LoginLockedError(Exception):
    """Raised by verify_login() when an account has too many recent
    failed attempts. The caller should show a clear "try again later"
    message instead of treating this like an ordinary wrong password -
    see app.py's login form."""
    pass


@contextmanager
def _connect(db_file=DB_FILE):
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_file=DB_FILE):
    """Creates every table this file needs, if they don't exist yet.
    Safe to call every time the app starts."""
    with _connect(db_file) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                folders TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS login_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                attempted_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS remember_tokens (
                token_hash TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
        """)


def _hash_password(password, salt_hex):
    """PBKDF2-HMAC-SHA256 with the given salt - see the module
    docstring above for why this specific approach was chosen."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        PBKDF2_ITERATIONS
    ).hex()


def create_user(username, password, folders, db_file=DB_FILE):
    """
    `folders`: the string "ALL" for an admin who can see every
    department (including any new ones added later), or a list of
    folder name strings (e.g. ["HR_Policies", "Finance"]) for a
    restricted user.

    Raises ValueError if the username already exists - usernames must
    be unique, and this is checked explicitly rather than relying on
    a raw database error, so the caller gets a clear message instead
    of a stack trace.
    """
    username = username.strip()
    if not username:
        raise ValueError("Username cannot be empty.")
    if not password:
        raise ValueError("Password cannot be empty.")

    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    folders_json = json.dumps(folders)
    now = datetime.now(timezone.utc).isoformat()

    with _connect(db_file) as conn:
        existing = conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            raise ValueError(f"User '{username}' already exists.")
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, folders, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (username, password_hash, salt, folders_json, now)
        )


def _record_failed_attempt(username, db_file=DB_FILE):
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_file) as conn:
        conn.execute(
            "INSERT INTO login_attempts (username, attempted_at) VALUES (?, ?)",
            (username, now)
        )


def _recent_failed_attempt_count(username, db_file=DB_FILE):
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=LOGIN_LOCKOUT_MINUTES)).isoformat()
    with _connect(db_file) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM login_attempts WHERE username = ? AND attempted_at > ?",
            (username, cutoff)
        ).fetchone()
    return row["c"]


def _clear_failed_attempts(username, db_file=DB_FILE):
    with _connect(db_file) as conn:
        conn.execute("DELETE FROM login_attempts WHERE username = ?", (username,))


def verify_login(username, password, db_file=DB_FILE):
    """
    Returns {"username": ..., "folders": "ALL" or [...]} if the
    username/password combination is correct, or None if it isn't -
    the caller can't tell from a None return alone whether the
    username didn't exist or the password was wrong, which is
    intentional: telling an attacker "that username doesn't exist" is
    itself useful information you don't want to hand out for free.

    Raises LoginLockedError instead of checking the password at all
    if this username has MAX_LOGIN_ATTEMPTS or more failures within
    the last LOGIN_LOCKOUT_MINUTES - this is what turns "someone can
    try a password a million times" into "someone can try it 5 times
    every 15 minutes," without needing to touch the password check
    logic itself at all.
    """
    username = username.strip()

    if _recent_failed_attempt_count(username, db_file) >= MAX_LOGIN_ATTEMPTS:
        raise LoginLockedError(
            f"Too many failed attempts for '{username}'. Try again in a few minutes."
        )

    with _connect(db_file) as conn:
        row = conn.execute(
            "SELECT username, password_hash, salt, folders FROM users WHERE username = ?",
            (username,)
        ).fetchone()

    if row is None:
        _record_failed_attempt(username, db_file)
        return None

    expected_hash = _hash_password(password, row["salt"])
    # secrets.compare_digest instead of `==` - a plain string comparison
    # exits as soon as it finds a mismatched character, which means
    # comparing against a MOSTLY-correct guess takes microscopically
    # longer than comparing against a totally wrong one. Measuring
    # that timing difference over many attempts is a real attack
    # technique. compare_digest always takes the same time regardless.
    if not secrets.compare_digest(expected_hash, row["password_hash"]):
        _record_failed_attempt(username, db_file)
        return None

    _clear_failed_attempts(username, db_file)
    return {"username": row["username"], "folders": json.loads(row["folders"])}


def list_users(db_file=DB_FILE):
    with _connect(db_file) as conn:
        rows = conn.execute(
            "SELECT username, folders, created_at FROM users ORDER BY created_at ASC"
        ).fetchall()
    return [
        {"username": r["username"], "folders": json.loads(r["folders"]), "created_at": r["created_at"]}
        for r in rows
    ]


def delete_user(username, db_file=DB_FILE):
    with _connect(db_file) as conn:
        conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.execute("DELETE FROM remember_tokens WHERE username = ?", (username,))
        conn.execute("DELETE FROM login_attempts WHERE username = ?", (username,))


def update_user_folders(username, folders, db_file=DB_FILE):
    with _connect(db_file) as conn:
        conn.execute(
            "UPDATE users SET folders = ? WHERE username = ?",
            (json.dumps(folders), username)
        )


def user_count(db_file=DB_FILE):
    with _connect(db_file) as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()
    return row["c"]


# =====================================================================
# REMEMBER ME
#
# Streamlit has no built-in browser cookie API, so this project can't
# set a real httponly session cookie the way a typical web app would.
# Instead, a random token is placed in the page's URL via
# st.query_params (see app.py) - visiting that exact URL again logs
# the browser back in automatically, without retyping a password.
#
# HONEST SECURITY CAVEAT, worth knowing: because the token lives in
# the URL rather than an httponly cookie, anyone who has that URL -
# not just the account owner - can use it to sign in as that user.
# It should be treated with the same caution as a password-reset
# link: don't paste a URL containing ?remember_token=... into a chat,
# email, or screenshot. A real production deployment would replace
# this with a proper httponly session cookie instead.
# =====================================================================

def _hash_token(token):
    """Tokens are hashed at rest the same way passwords are, but with
    a fast hash (not PBKDF2) - a remember-token is long and randomly
    generated, not human-chosen, so it isn't vulnerable to guessing
    the way a password is. Hashing it here only protects against a
    stolen DATABASE FILE handing out ready-to-use tokens directly."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_remember_token(username, db_file=DB_FILE):
    """Generates a new random token, stores its hash (never the raw
    value) against this user with an expiry date, and returns the RAW
    token - this is the only moment it exists outside the database."""
    token = secrets.token_urlsafe(32)
    token_hash = _hash_token(token)
    expires_at = (datetime.now(timezone.utc) + timedelta(days=REMEMBER_TOKEN_DAYS)).isoformat()
    with _connect(db_file) as conn:
        conn.execute(
            "INSERT INTO remember_tokens (token_hash, username, expires_at) VALUES (?, ?, ?)",
            (token_hash, username, expires_at)
        )
    return token


def verify_remember_token(token, db_file=DB_FILE):
    """Returns {"username":..., "folders":...} if the token is valid,
    unexpired, and its account still exists - else None. Possessing a
    valid token IS the credential here, the same way a session cookie
    would be; no password is re-checked."""
    if not token:
        return None
    token_hash = _hash_token(token)
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_file) as conn:
        row = conn.execute(
            "SELECT username, expires_at FROM remember_tokens WHERE token_hash = ?",
            (token_hash,)
        ).fetchone()
        if row is None or row["expires_at"] < now:
            return None
        user_row = conn.execute(
            "SELECT folders FROM users WHERE username = ?", (row["username"],)
        ).fetchone()
    if user_row is None:
        return None  # the account was deleted after this token was issued
    return {"username": row["username"], "folders": json.loads(user_row["folders"])}


def revoke_remember_token(token, db_file=DB_FILE):
    """Called on logout, so a remember-me token can't keep working
    after someone has explicitly signed out."""
    if not token:
        return
    with _connect(db_file) as conn:
        conn.execute("DELETE FROM remember_tokens WHERE token_hash = ?", (_hash_token(token),))


# =====================================================================
# PASSWORD RESET
#
# No email-sending infrastructure exists in this project (a real
# "forgot password" link needs SMTP credentials and a mail server - a
# meaningfully heavier, separate feature this project deliberately
# doesn't take on). The realistic equivalent for an internally-run
# tool like this one: an admin resets it directly, and any logged-in
# user can change their own.
# =====================================================================

def admin_reset_password(username, new_password, db_file=DB_FILE):
    """Sets a new password for `username` directly, without needing
    their OLD password - that's the whole point, for someone who's
    locked out or forgot it. Exactly why this must only ever be
    reachable from an admin-only part of the UI, never by a regular
    user resetting someone else's account."""
    if not new_password:
        raise ValueError("New password cannot be empty.")
    salt = secrets.token_hex(16)
    password_hash = _hash_password(new_password, salt)
    with _connect(db_file) as conn:
        existing = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        if not existing:
            raise ValueError(f"User '{username}' does not exist.")
        conn.execute(
            "UPDATE users SET password_hash = ?, salt = ? WHERE username = ?",
            (password_hash, salt, username)
        )
    _clear_failed_attempts(username, db_file)  # a fresh password shouldn't inherit an old lockout


def change_own_password(username, current_password, new_password, db_file=DB_FILE):
    """Self-service password change. Re-verifies the CURRENT password
    first, through the same verify_login() used at sign-in - so
    someone with a hijacked browser session, but who doesn't actually
    know the password, can't quietly lock the real owner out by
    changing it. This also means repeated wrong guesses here count
    against the normal login rate-limit, on purpose - otherwise this
    form would be a side door around it."""
    if not new_password:
        raise ValueError("New password cannot be empty.")

    if verify_login(username, current_password, db_file) is None:
        raise ValueError("Current password is incorrect.")

    salt = secrets.token_hex(16)
    password_hash = _hash_password(new_password, salt)
    with _connect(db_file) as conn:
        conn.execute(
            "UPDATE users SET password_hash = ?, salt = ? WHERE username = ?",
            (password_hash, salt, username)
        )