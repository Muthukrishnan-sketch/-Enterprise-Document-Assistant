"""
chat_store.py
=============
Persists chat conversations to a local SQLite database so they
survive closing the browser tab or restarting the app - unlike
st.session_state, which lives only in memory and resets every time
the Streamlit process restarts.

WHY SQLITE INSTEAD OF A JSON FILE:
A single JSON file works fine for one conversation, but once there
are many conversations, saving one new message means rewriting the
ENTIRE file every time, and "show me the 5 most recent conversations"
means loading and parsing everything just to sort it. SQLite is a
single file too (no server to install or run), but the `sqlite3`
module that reads and writes it ships with Python itself - zero new
dependencies - and it lets us ask for exactly the rows we need
("messages for conversation X, in order") without touching the rest.
This is also a small, honest step in the same direction a real
production system would eventually take with something like Postgres
- not a dead end that has to be thrown away later.

WHY THIS IS A SEPARATE FILE FROM rag_core.py:
This file has no idea what a RAG pipeline, an embedding, or a chunk
is - it only knows how to store and retrieve conversations and
messages as plain data. rag_core.py has no idea this file exists.
app.py is the only place that uses both together. Keeping storage and
retrieval-generation logic apart means either one can change without
risking breaking the other.

THREAD SAFETY NOTE: every function below opens a fresh connection,
uses it, and closes it (see _connect()) rather than keeping one
connection open for the whole app. SQLite connections aren't safe to
share across threads, and Streamlit can run different browser
sessions on different threads - opening a short-lived connection per
operation sidesteps that entirely, at the cost of a small amount of
overhead per call that's irrelevant at this scale.
"""

import sqlite3
import json
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager

DB_FILE = "chat_history.db"


@contextmanager
def _connect(db_file=DB_FILE):
    """Opens a SQLite connection, yields it for use, then always
    closes it - used as `with _connect() as conn:` so a connection
    never leaks even if a query raises an error partway through."""
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row  # lets us read columns by name, like a dict
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_file=DB_FILE):
    """Creates the two tables this app needs, if they don't already
    exist. Safe to call every time the app starts - CREATE TABLE IF
    NOT EXISTS does nothing on the second and every later run."""
    with _connect(db_file) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT,
                searched_for TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        # Lightweight migration: this column didn't exist before login
        # was added to the project. ALTER TABLE ADD COLUMN is safe to
        # attempt every time - if the column is already there, SQLite
        # raises "duplicate column name", which is simply ignored.
        # Any conversation created before this migration will have
        # username=NULL, which means it belongs to no one and will
        # never show up in anyone's list - effectively orphaned, but
        # harmless (see README section 17).
        try:
            conn.execute("ALTER TABLE conversations ADD COLUMN username TEXT")
        except sqlite3.OperationalError:
            pass


def create_conversation(username, title="New conversation", db_file=DB_FILE):
    """Creates a new, empty conversation owned by `username` and
    returns its id (a UUID string, so IDs never collide even across
    different machines)."""
    conversation_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_file) as conn:
        conn.execute(
            "INSERT INTO conversations (id, username, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, username, title, now, now)
        )
    return conversation_id


def list_conversations(username, limit=50, db_file=DB_FILE):
    """Returns `username`'s past conversations, most recently updated
    first - what the sidebar's conversation list is built from.
    Sending a new message bumps a conversation's updated_at (see
    save_message), so the conversation you're actively using always
    sorts to the top. Only ever returns conversations belonging to
    this exact username - this is the enforcement point that keeps
    one user's conversation list from ever showing another user's
    chats, including their citations."""
    with _connect(db_file) as conn:
        rows = conn.execute(
            "SELECT id, title, updated_at FROM conversations WHERE username = ? ORDER BY updated_at DESC LIMIT ?",
            (username, limit)
        ).fetchall()
    return [dict(row) for row in rows]


def rename_conversation(conversation_id, title, db_file=DB_FILE):
    with _connect(db_file) as conn:
        conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))


def delete_conversation(conversation_id, username, db_file=DB_FILE):
    """Deletes a conversation - but only if it actually belongs to
    `username`. This check exists because deletion is destructive and
    cheap to guard: even if a conversation_id ever leaked or was
    guessed, this stops it from being deleted by anyone but its
    owner. (load_messages(), being read-only and only ever called by
    app.py with an ID it already fetched from that same user's own
    list_conversations() result, doesn't re-check ownership itself -
    see README section 17 for that trust boundary made explicit.)"""
    with _connect(db_file) as conn:
        owned = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND username = ?",
            (conversation_id, username)
        ).fetchone()
        if not owned:
            return False
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return True


def save_message(conversation_id, role, content, sources=None, searched_for=None, db_file=DB_FILE):
    """
    Saves one chat turn. `sources`, if given, is the list of
    (score, entry) tuples rag_core.py returns - stored as JSON so the
    colored citation cards can be rebuilt exactly when this
    conversation is reopened later. Only what's needed to redisplay
    is kept - the 768-number embedding vector inside each `entry` is
    dropped, since it's useless once retrieval has already happened
    and would otherwise make the database file much larger for no
    benefit. The conversation's updated_at is bumped so it sorts to
    the top of the sidebar list.
    """
    now = datetime.now(timezone.utc).isoformat()
    sources_json = None
    if sources:
        trimmed = [
            {
                "score": score,
                "folder": entry["folder"],
                "filename": entry["filename"],
                "path": entry["path"],
                "chunk_index": entry["chunk_index"],
                "text": entry["text"],
            }
            for score, entry in sources
        ]
        sources_json = json.dumps(trimmed)

    with _connect(db_file) as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, sources_json, searched_for, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (conversation_id, role, content, sources_json, searched_for, now)
        )
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))


def load_messages(conversation_id, db_file=DB_FILE):
    """
    Returns this conversation's messages in the exact shape app.py's
    st.session_state.messages already uses, so loading a saved
    conversation and continuing a brand new one work identically from
    the UI's point of view - the chat rendering code doesn't need to
    know or care which one it's looking at.
    """
    with _connect(db_file) as conn:
        rows = conn.execute(
            "SELECT role, content, sources_json, searched_for FROM messages "
            "WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,)
        ).fetchall()

    messages = []
    for row in rows:
        sources = []
        if row["sources_json"]:
            for s in json.loads(row["sources_json"]):
                score = s.pop("score")
                sources.append((score, s))
        messages.append({
            "role": row["role"],
            "content": row["content"],
            "sources": sources,
            "searched_for": row["searched_for"],
        })
    return messages


def title_from_message(text, max_len=48):
    """Turns a user's first message into a short conversation title,
    the same way ChatGPT-style apps auto-title a new chat."""
    text = " ".join(text.split())  # collapse newlines/repeated whitespace
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."