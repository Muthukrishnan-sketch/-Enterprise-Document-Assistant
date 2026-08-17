"""
app.py
======
THIS is the demo you show your TL. Run it with:

    streamlit run app.py

It's a web UI wrapped around the same rag_core.py engine used by
1_ingest.py and 2_ask.py.

VISUAL DESIGN, ON PURPOSE:
This app's whole point is "answers traced back to a department
folder" - so the design makes that literal instead of just correct.
Every source document belongs to one of the 5 department folders
(see FOLDER_STYLES below), each with its own color, shown inline in
the answer, in a legend strip, and on each retrieved-chunk card.

SEARCH MODE, ON PURPOSE:
Instead of four separate sliders as the first thing you see, the
sidebar leads with three preset buttons - Precise / Balanced / Broad
- each one a complete, named combination of chunk size, overlap,
top-K, and relevance threshold (see PRESETS below). Clicking one
re-indexes automatically if the chunk settings actually changed. The
underlying sliders still exist for manual fine-tuning, tucked under
"Advanced" so they don't compete with the presets for attention.

CONVERSATIONAL, ON PURPOSE:
The main area is a real chat transcript (st.chat_message /
st.chat_input), not a single question-and-answer form. Follow-up
questions like "what about its warranty?" are rewritten into a
standalone search query using recent chat history BEFORE retrieval
runs - see rag_core.condense_question() and
rag_core.answer_question_conversational(). Each assistant turn shows
what was actually searched for, so you can demonstrate that follow-
ups are genuinely being understood in context, not just answered
from memory with no grounding.

UI SECTIONS IN THIS FILE (search for these comments to jump around):
  # ===== AUTH: LOGIN GATE =====          <- nothing below runs until logged in
  # ===== SIDEBAR: ACCOUNT =====          <- signed-in-as, logout, admin user management
  # ===== VISUAL IDENTITY =====                <- colors, fonts, badges
  # ===== SEARCH PRESETS =====                  <- the 3 named modes
  # ===== SIDEBAR: PIPELINE SETTINGS =====      <- preset buttons + advanced sliders
  # ===== SIDEBAR: INDEX STATUS =====           <- "Indexed X docs into Y chunks" + upload new documents
  # ===== MAIN: CHAT TRANSCRIPT =====           <- replays the conversation so far
  # ===== MAIN: CHAT INPUT =====                <- the bottom-pinned input + new-turn logic

CHANGING THE UI:
  - Page title / icon / layout -> st.set_page_config(...) below
  - Overall color theme -> .streamlit/config.toml
  - Department colors / add a new department -> FOLDER_STYLES below
  - Preset values / add a 4th preset -> PRESETS below
  - Fonts -> the @import + font-family lines inside CUSTOM_CSS below
"""

import os
import re
import html
from datetime import datetime, timezone
import streamlit as st
import rag_core
import chat_store
import auth_store

st.set_page_config(page_title="Enterprise Document Assistant", page_icon="🗂", layout="wide")

# ===== VISUAL IDENTITY =====

FOLDER_STYLES = {
    "HR_Policies":       {"color": "#4FA79B", "label": "HR"},
    "IT_Security":       {"color": "#5C7CFA", "label": "IT"},
    "Finance":           {"color": "#C99A44", "label": "FINANCE"},
    "Remote_Work":       {"color": "#7FA66B", "label": "REMOTE"},
    "Technical_Manuals": {"color": "#B5651D", "label": "TECH"},
}
DEFAULT_FOLDER_STYLE = {"color": "#6B7280", "label": "DOC"}


def folder_style(folder_name):
    return FOLDER_STYLES.get(folder_name, DEFAULT_FOLDER_STYLE)


def relative_time(iso_string):
    """Turns a stored ISO timestamp into a short, human string like
    '5m ago' or '3d ago' for the conversation list in the sidebar."""
    try:
        dt = datetime.fromisoformat(iso_string)
    except (ValueError, TypeError):
        return ""
    seconds = (datetime.now(timezone.utc) - dt).total_seconds()
    if seconds < 60:
        return "just now"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m ago"
    hours = int(minutes // 60)
    if hours < 24:
        return f"{hours}h ago"
    days = int(hours // 24)
    if days < 7:
        return f"{days}d ago"
    return dt.strftime("%b %d")


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Inter:wght@400;500;600&display=swap');

.stApp { font-family: 'Inter', sans-serif; }
.stApp h1, .stApp h2, .stApp h3 { font-family: 'Fraunces', Georgia, serif !important; }

.folder-legend { display:flex; gap:16px; flex-wrap:wrap; margin: 2px 0 20px 0; }
.folder-legend-item { display:flex; align-items:center; gap:6px; font-size:0.78rem; color:#8B93A3; }
.folder-dot { width:8px; height:8px; border-radius:50%; display:inline-block; flex-shrink:0; }

.cite-badge { display:inline-block; padding:1px 8px; border-radius:9px; font-size:0.70rem;
              font-weight:700; margin:0 1px; letter-spacing:0.02em; white-space:nowrap; }

.source-card { border-left:3px solid #6B7280; background:rgba(255,255,255,0.03);
               padding:12px 16px; border-radius:8px; margin-bottom:12px; }
.source-card-header { display:flex; align-items:center; gap:10px; margin-bottom:8px; flex-wrap:wrap; }
.source-folder-badge { font-size:0.68rem; font-weight:700; padding:2px 9px; border-radius:9px; letter-spacing:0.03em; }
.source-path { font-weight:600; }
.source-meta { color:#8B93A3; font-size:0.78rem; }
.source-text { color:#C7CBD3; font-size:0.85rem; line-height:1.55; margin-bottom:8px; white-space:pre-wrap; }
.relevance-track { background:rgba(255,255,255,0.08); border-radius:4px; height:5px; width:100%; overflow:hidden; }
.relevance-fill { height:100%; border-radius:4px; }
.relevance-label { color:#8B93A3; font-size:0.7rem; margin-top:4px; }

.preset-summary { font-size:0.78rem; color:#8B93A3; background:rgba(255,255,255,0.03);
                   border-radius:6px; padding:8px 10px; margin:8px 0 4px 0; line-height:1.6; }
.preset-summary b { color:#EDEDEA; }
.preset-desc { font-size:0.78rem; color:#8B93A3; margin:2px 0 10px 0; line-height:1.5; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ===== AUTH: LOGIN GATE =====
# Nothing below this block ever runs for a request that isn't logged
# in - st.stop() at the bottom of the "not logged in" branch halts
# the script right there, so the sidebar, the chat, the document
# upload, all of it stays completely unreached, not just hidden.
auth_store.init_db()

if "user" not in st.session_state:
    st.session_state.user = None

# Auto-login via a "remember me" token sitting in the URL, if one is
# present and still valid - see auth_store.py's REMEMBER ME section
# for the honest tradeoff of using the URL instead of a real httponly
# cookie, which Streamlit has no built-in API for.
if st.session_state.user is None:
    remembered_token = st.query_params.get("remember_token")
    if remembered_token:
        remembered_user = auth_store.verify_remember_token(remembered_token)
        if remembered_user:
            st.session_state.user = remembered_user
        else:
            # stale/expired/revoked - drop it so it isn't retried every reload
            del st.query_params["remember_token"]

if st.session_state.user is None:
    st.title("🗂 Enterprise Document Assistant")
    st.caption("Sign in to continue.")

    if auth_store.user_count() == 0:
        st.info(
            "No user accounts exist yet - create the first one below. It "
            "automatically becomes an admin (sees every department). This "
            "form only ever appears when there are zero accounts - once one "
            "exists, every account after it is created from the **Manage "
            "users** panel by an existing admin, not here."
        )
        with st.form("bootstrap_admin_form"):
            first_username = st.text_input("Choose a username")
            first_password = st.text_input("Choose a password", type="password")
            first_password_confirm = st.text_input("Confirm password", type="password")
            bootstrap_submitted = st.form_submit_button("Create admin account", type="primary")

        if bootstrap_submitted:
            if not first_username or not first_password:
                st.error("Username and password are both required.")
            elif first_password != first_password_confirm:
                st.error("Passwords don't match.")
            else:
                try:
                    auth_store.create_user(first_username, first_password, "ALL")
                    st.success(f"Admin account '{first_username}' created - log in below.")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

        st.caption(
            "Running this locally instead? You can also run "
            "`python manage_users.py` in a terminal to do the same thing."
        )
        st.stop()

    with st.form("login_form"):
        login_username = st.text_input("Username")
        login_password = st.text_input("Password", type="password")
        remember_me = st.checkbox(
            "Remember me on this device",
            help="Keeps you signed in via a token in this page's URL. "
                 "Don't share that URL with anyone else - see the README."
        )
        submitted = st.form_submit_button("Log in", type="primary")

    if submitted:
        locked_out = False
        try:
            user = auth_store.verify_login(login_username, login_password)
        except auth_store.LoginLockedError as e:
            st.error(str(e))
            locked_out = True
            user = None

        if user:
            st.session_state.user = user
            if remember_me:
                token = auth_store.create_remember_token(user["username"])
                st.query_params["remember_token"] = token
            st.rerun()
        elif not locked_out:
            st.error("Incorrect username or password.")

    st.stop()


def user_allowed_folders():
    """Returns None if the current user is an admin (no restriction -
    matches rag_core.retrieve()'s convention exactly), otherwise the
    list of folder names they're allowed to see."""
    folders = st.session_state.user["folders"]
    return None if folders == "ALL" else folders


def is_admin():
    return st.session_state.user["folders"] == "ALL"


def visible_folder_styles():
    """FOLDER_STYLES filtered down to whatever the current user can
    actually access - used for the legend, the upload folder picker,
    and anywhere else a folder NAME might otherwise leak to someone
    who isn't supposed to know it exists, not just its contents."""
    allowed = user_allowed_folders()
    if allowed is None:
        return FOLDER_STYLES
    allowed_set = set(allowed)
    return {name: style for name, style in FOLDER_STYLES.items() if name in allowed_set}


def render_folder_legend():
    items_html = "".join(
        f'<span class="folder-legend-item">'
        f'<span class="folder-dot" style="background:{style["color"]}"></span>'
        f'{html.escape(folder)}</span>'
        for folder, style in visible_folder_styles().items()
    )
    st.markdown(f'<div class="folder-legend">{items_html}</div>', unsafe_allow_html=True)


def linkify_citations(answer_text, sources):
    def _replace(match):
        numbers = [n.strip() for n in match.group(1).split(",") if n.strip().isdigit()]
        if not numbers:
            return match.group(0)
        badges = []
        for num_str in numbers:
            n = int(num_str)
            if 1 <= n <= len(sources):
                _, entry = sources[n - 1]
                style = folder_style(entry["folder"])
                badges.append(
                    f'<span class="cite-badge" style="background:{style["color"]}26;'
                    f'color:{style["color"]};border:1px solid {style["color"]}80;">'
                    f'{style["label"]} · {n}</span>'
                )
            else:
                badges.append(html.escape(match.group(0)))
        return " ".join(badges)

    return re.sub(r"\[([\d,\s]+)\]", _replace, answer_text)


def render_source_card(rank, score, entry):
    style = folder_style(entry["folder"])
    pct = max(2, min(100, round(score * 100)))
    card_html = f'''
<div class="source-card" style="border-left-color:{style["color"]}">
  <div class="source-card-header">
    <span class="source-folder-badge" style="background:{style["color"]}26;color:{style["color"]};border:1px solid {style["color"]}80;">{style["label"]}</span>
    <span class="source-path">[{rank}] {html.escape(entry["path"])}</span>
    <span class="source-meta">chunk {entry["chunk_index"]}</span>
  </div>
  <div class="source-text">{html.escape(entry["text"])}</div>
  <div class="relevance-track"><div class="relevance-fill" style="width:{pct}%;background:{style["color"]}"></div></div>
  <div class="relevance-label">relevance {score:.3f}</div>
</div>
'''
    st.markdown(card_html, unsafe_allow_html=True)


# ===== SEARCH PRESETS =====
# Each preset is a complete, named combination of the 4 tuning knobs.
# Add a 4th preset by adding another entry here + its name to PRESET_ORDER.
PRESETS = {
    "Precise": {
        "chunk_size": 300, "chunk_overlap": 30, "top_k": 2, "min_relevance": 0.35,
        "icon": "🎯",
        "description": "Small, tight chunks and a high relevance bar. Short, "
                        "high-confidence answers - more likely to say \"I don't "
                        "know\" than guess.",
    },
    "Balanced": {
        "chunk_size": 500, "chunk_overlap": 50, "top_k": 3, "min_relevance": 0.15,
        "icon": "⚖️",
        "description": "A reasonable middle ground for most questions. The "
                        "default starting point.",
    },
    "Broad": {
        "chunk_size": 800, "chunk_overlap": 100, "top_k": 6, "min_relevance": 0.05,
        "icon": "🌐",
        "description": "Larger chunks, more sources, a low relevance bar. "
                        "Less likely to miss something relevant, at the cost "
                        "of noisier context.",
    },
}
PRESET_ORDER = ["Precise", "Balanced", "Broad"]


def matching_preset_name():
    """If the current chunk_size/overlap/top_k/min_relevance exactly match
    one of the named presets, return its name - otherwise "Custom" (this
    happens once someone drags a slider under Advanced)."""
    for name, preset in PRESETS.items():
        if (st.session_state.chunk_size == preset["chunk_size"] and
                st.session_state.chunk_overlap == preset["chunk_overlap"] and
                st.session_state.top_k == preset["top_k"] and
                st.session_state.min_relevance == preset["min_relevance"]):
            return name
    return "Custom"


# -----------------------------------------------------------------
# st.session_state is Streamlit's memory across reruns, but it's lost
# every time the app restarts or a browser tab closes. chat_store.py
# is what makes conversations survive that - see below.
# -----------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "index_log" not in st.session_state:
    st.session_state.index_log = []
if "indexed_chunk_size" not in st.session_state:
    st.session_state.indexed_chunk_size = None
if "indexed_chunk_overlap" not in st.session_state:
    st.session_state.indexed_chunk_overlap = None

chat_store.init_db()  # creates the tables on first run; does nothing on every run after

if "conversation_id" not in st.session_state:
    # First load since the app started (or the browser tab reopened
    # after being closed) - resume the most recently used conversation
    # if one exists on disk, rather than starting from a blank slate.
    existing = chat_store.list_conversations(st.session_state.user["username"], limit=1)
    if existing:
        st.session_state.conversation_id = existing[0]["id"]
        st.session_state.messages = chat_store.load_messages(st.session_state.conversation_id)
    else:
        st.session_state.conversation_id = chat_store.create_conversation(st.session_state.user["username"])
        st.session_state.messages = []
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = PRESETS["Balanced"]["chunk_size"]
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = PRESETS["Balanced"]["chunk_overlap"]
if "top_k" not in st.session_state:
    st.session_state.top_k = PRESETS["Balanced"]["top_k"]
if "min_relevance" not in st.session_state:
    st.session_state.min_relevance = PRESETS["Balanced"]["min_relevance"]
if "index_error" not in st.session_state:
    st.session_state.index_error = None


def run_indexing(chunk_size, chunk_overlap):
    """Re-run the full offline pipeline with the given settings.

    Incremental: if chunk_size/overlap match what's already indexed,
    unchanged documents are reused instead of re-embedded from
    scratch - see rag_core.build_vector_store()'s previous_store
    parameter. A change to chunk_size/overlap invalidates every old
    chunk boundary even for text that didn't change, so reuse is
    correctly skipped in that case (previous_store=None forces a
    full rebuild, which is the right behavior there, not a bug)."""
    st.session_state.index_log = []

    same_chunking_params = (
        st.session_state.vector_store is not None and
        st.session_state.indexed_chunk_size == chunk_size and
        st.session_state.indexed_chunk_overlap == chunk_overlap
    )
    previous_store = st.session_state.vector_store if same_chunking_params else None

    with st.spinner("Indexing documents..."):
        vector_store = rag_core.build_vector_store(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            previous_store=previous_store,
            progress_callback=lambda msg: st.session_state.index_log.append(msg)
        )
    st.session_state.vector_store = vector_store
    st.session_state.indexed_chunk_size = chunk_size
    st.session_state.indexed_chunk_overlap = chunk_overlap


def apply_preset(name):
    """Set all 4 knobs from a named preset. Top-K and min relevance apply
    instantly (query-time). Chunk size/overlap need a fresh index if they
    actually changed - so this re-indexes automatically when needed,
    exactly once per click (never on every rerun)."""
    preset = PRESETS[name]
    st.session_state.chunk_size = preset["chunk_size"]
    st.session_state.chunk_overlap = preset["chunk_overlap"]
    st.session_state.top_k = preset["top_k"]
    st.session_state.min_relevance = preset["min_relevance"]

    needs_reindex = (
        st.session_state.indexed_chunk_size != preset["chunk_size"] or
        st.session_state.indexed_chunk_overlap != preset["chunk_overlap"]
    )
    if needs_reindex:
        try:
            run_indexing(preset["chunk_size"], preset["chunk_overlap"])
            st.session_state.index_error = None
        except Exception as e:
            st.session_state.index_error = str(e)


# ===== SIDEBAR: ACCOUNT =====
with st.sidebar:
    role_label = "admin - sees every department" if is_admin() else ", ".join(user_allowed_folders()) or "no folders yet"
    st.caption(f"Signed in as **{st.session_state.user['username']}**  ·  {role_label}")
    if st.button("Log out", use_container_width=True):
        # Revoke the remember-me token (if any) so it can't be reused
        # after an explicit sign-out, and clear it from the URL too -
        # otherwise it would just auto-log the next visit right back in.
        auth_store.revoke_remember_token(st.query_params.get("remember_token"))
        if "remember_token" in st.query_params:
            del st.query_params["remember_token"]
        # Only clear what's specific to THIS user's session - the
        # shared FAISS index stays cached, since it isn't user-specific
        # and rebuilding it on every login/logout would be wasteful.
        st.session_state.user = None
        st.session_state.pop("conversation_id", None)
        st.session_state.pop("messages", None)
        st.rerun()

    with st.expander("🔑 Change my password"):
        current_pw = st.text_input("Current password", type="password", key="change_pw_current")
        new_pw = st.text_input("New password", type="password", key="change_pw_new")
        confirm_pw = st.text_input("Confirm new password", type="password", key="change_pw_confirm")
        if st.button("Update password", use_container_width=True):
            if not current_pw or not new_pw:
                st.warning("Both current and new password are required.")
            elif new_pw != confirm_pw:
                st.warning("New password and confirmation don't match.")
            else:
                try:
                    auth_store.change_own_password(
                        st.session_state.user["username"], current_pw, new_pw
                    )
                    st.success("Password updated.")
                except ValueError as e:
                    st.error(str(e))

    if is_admin():
        with st.expander("🛡️ Manage users (admin)"):
            st.markdown("**Existing users**")
            for u in auth_store.list_users():
                access = "ALL" if u["folders"] == "ALL" else (", ".join(u["folders"]) or "(none)")
                st.caption(f"**{u['username']}** — {access}")

            st.divider()
            st.markdown("**Create new user**")
            new_username = st.text_input("Username", key="new_user_username")
            new_password = st.text_input("Password", type="password", key="new_user_password")
            new_is_admin = st.checkbox("Admin (sees every department)", key="new_user_is_admin")
            new_folders = "ALL" if new_is_admin else st.multiselect(
                "Folder access", options=sorted(FOLDER_STYLES.keys()), key="new_user_folders"
            )
            if st.button("Create user", use_container_width=True):
                if not new_username or not new_password:
                    st.warning("Username and password are both required.")
                else:
                    try:
                        auth_store.create_user(new_username, new_password, new_folders)
                        st.success(f"Created user '{new_username}'.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

            st.divider()
            st.markdown("**Reset a user's password**")
            all_usernames = [u["username"] for u in auth_store.list_users()]
            if all_usernames:
                reset_target = st.selectbox("User", options=all_usernames, key="reset_pw_target")
                reset_new_pw = st.text_input("New password for them", type="password", key="reset_pw_value")
                if st.button("Reset their password", use_container_width=True):
                    if not reset_new_pw:
                        st.warning("Enter a new password first.")
                    else:
                        try:
                            auth_store.admin_reset_password(reset_target, reset_new_pw)
                            st.success(f"Password reset for '{reset_target}'. Tell them the new password directly - there's no email flow.")
                        except ValueError as e:
                            st.error(str(e))

            st.divider()
            st.markdown("**Delete a user**")
            deletable = [u["username"] for u in auth_store.list_users()
                         if u["username"] != st.session_state.user["username"]]
            if deletable:
                user_to_delete = st.selectbox("Select user", options=deletable, key="user_to_delete")
                if st.button("Delete this user", use_container_width=True):
                    auth_store.delete_user(user_to_delete)
                    st.success(f"Deleted '{user_to_delete}'.")
                    st.rerun()
            else:
                st.caption("No other users to delete.")


# ===== SIDEBAR: CONVERSATIONS =====
with st.sidebar:
    st.header("💬 Conversations")
    if st.button("+ New conversation", use_container_width=True, type="primary"):
        st.session_state.conversation_id = chat_store.create_conversation(st.session_state.user["username"])
        st.session_state.messages = []
        st.rerun()

    for conv in chat_store.list_conversations(st.session_state.user["username"]):
        is_active = conv["id"] == st.session_state.conversation_id
        row_left, row_right = st.columns([5, 1])
        with row_left:
            label = ("🟢 " if is_active else "") + conv["title"]
            if st.button(label, key=f"conv_{conv['id']}", use_container_width=True):
                st.session_state.conversation_id = conv["id"]
                st.session_state.messages = chat_store.load_messages(conv["id"])
                st.rerun()
            st.caption(relative_time(conv["updated_at"]))
        with row_right:
            if st.button("🗑️", key=f"del_{conv['id']}"):
                chat_store.delete_conversation(conv["id"], st.session_state.user["username"])
                if is_active:
                    remaining = chat_store.list_conversations(st.session_state.user["username"], limit=1)
                    if remaining:
                        st.session_state.conversation_id = remaining[0]["id"]
                        st.session_state.messages = chat_store.load_messages(remaining[0]["id"])
                    else:
                        st.session_state.conversation_id = chat_store.create_conversation(st.session_state.user["username"])
                        st.session_state.messages = []
                st.rerun()


# ===== SIDEBAR: PIPELINE SETTINGS =====
with st.sidebar:
    st.header("Search mode")
    st.caption("Each mode sets chunk size, overlap, top-K, and the relevance bar together.")

    active_preset = matching_preset_name()
    cols = st.columns(3)
    for col, name in zip(cols, PRESET_ORDER):
        with col:
            is_active = active_preset == name
            if st.button(
                f'{PRESETS[name]["icon"]} {name}',
                key=f"preset_btn_{name}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                apply_preset(name)
                st.rerun()

    label = active_preset if active_preset != "Custom" else "Custom (manual)"
    st.markdown(f"**{label} mode**")
    if active_preset in PRESETS:
        st.markdown(f'<div class="preset-desc">{PRESETS[active_preset]["description"]}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="preset-summary">'
        f'chunk size <b>{st.session_state.chunk_size}</b> · '
        f'overlap <b>{st.session_state.chunk_overlap}</b> · '
        f'top-K <b>{st.session_state.top_k}</b> · '
        f'min relevance <b>{st.session_state.min_relevance:.2f}</b>'
        f'</div>',
        unsafe_allow_html=True
    )

    if st.session_state.index_error:
        st.error("Couldn't re-index just now - Google's servers may be briefly overloaded.")
        with st.expander("Technical details"):
            st.code(st.session_state.index_error)

    with st.expander("⚙️ Advanced: fine-tune manually"):
        new_chunk_size = st.slider("Chunk size (characters)", 100, 1000, st.session_state.chunk_size, step=50)
        new_chunk_overlap = st.slider("Chunk overlap (characters)", 0, 300, st.session_state.chunk_overlap, step=10)
        st.caption("Chunk size / overlap change the vector database itself - click Re-index below to apply.")
        st.session_state.chunk_size = new_chunk_size
        st.session_state.chunk_overlap = new_chunk_overlap

        if st.button("🔄 Re-index documents", use_container_width=True):
            try:
                run_indexing(st.session_state.chunk_size, st.session_state.chunk_overlap)
                st.session_state.index_error = None
            except Exception as e:
                st.session_state.index_error = str(e)

        st.divider()
        st.session_state.top_k = st.slider("Top-K chunks retrieved", 1, 10, st.session_state.top_k)
        st.session_state.min_relevance = st.slider(
            "Minimum relevance score", 0.0, 1.0, st.session_state.min_relevance, step=0.01
        )
        st.caption("Top-K and minimum relevance apply instantly - no re-indexing needed.")

# Auto-index on first load so the demo isn't sitting empty
if st.session_state.vector_store is None:
    try:
        run_indexing(st.session_state.chunk_size, st.session_state.chunk_overlap)
    except Exception as e:
        st.error(
            "Couldn't reach Google's servers to build the index. This is "
            "usually temporary - click a search mode or **Re-index documents** "
            "under Advanced to try again."
        )
        with st.expander("Technical details"):
            st.code(str(e))
        st.stop()

# ===== SIDEBAR: INDEX STATUS =====
with st.sidebar:
    st.divider()
    if is_admin():
        visible_entries = list(st.session_state.vector_store)
    else:
        allowed = set(user_allowed_folders())
        visible_entries = [e for e in st.session_state.vector_store if e["folder"] in allowed]
    n_chunks = len(visible_entries)
    n_docs = len({e["path"] for e in visible_entries})
    st.success(f"Indexed {n_docs} documents into {n_chunks} chunks.")
    with st.expander("Indexing log"):
        for line in st.session_state.index_log:
            st.text(line)

    st.divider()
    with st.expander("📤 Upload documents"):
        existing_folders = sorted(visible_folder_styles().keys())
        folder_options = existing_folders + ["+ New folder..."] if is_admin() else existing_folders

        if not folder_options:
            st.info("You don't have upload access to any folder yet - ask an admin to grant access.")
            target_folder = None
        else:
            folder_choice = st.selectbox(
                "Department folder",
                options=folder_options,
                key="upload_folder_choice"
            )
            if folder_choice == "+ New folder...":
                target_folder = st.text_input("New folder name", placeholder="e.g. Legal").strip()
            else:
                target_folder = folder_choice

        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "docx", "xlsx", "txt", "eml", "msg", "db", "sqlite", "sqlite3"],
            accept_multiple_files=True,
            key="doc_uploader"
        )
        st.caption("A file with the same name as an existing one will overwrite it.")

        if st.button("Save and index", use_container_width=True):
            if not target_folder:
                st.warning("Pick or type a department folder first.")
            elif not uploaded_files:
                st.warning("Choose at least one file first.")
            else:
                dest_dir = os.path.join(rag_core.DOCS_FOLDER, target_folder)
                os.makedirs(dest_dir, exist_ok=True)
                for uf in uploaded_files:
                    dest_path = os.path.join(dest_dir, uf.name)
                    with open(dest_path, "wb") as f:
                        f.write(uf.getbuffer())
                st.success(f"Saved {len(uploaded_files)} file(s) to {target_folder}/ - indexing now...")
                try:
                    run_indexing(st.session_state.chunk_size, st.session_state.chunk_overlap)
                    st.session_state.index_error = None
                except Exception as e:
                    st.session_state.index_error = str(e)
                st.rerun()

st.title("🗂 Enterprise Document Assistant")
render_folder_legend()
st.caption("A conversational assistant over your company documents. Every fact is tagged to the department folder it came from.")


DECLINE_PHRASES = (
    "i don't know", "i do not know", "i couldn't find", "i could not find",
    "i'm not sure", "i am not sure", "i don't have", "i do not have",
)


def answer_used_sources(answer_text):
    """True if the answer actually references a numbered source like
    [1] or [1, 2] - i.e. the LLM genuinely grounded its response in
    the retrieved context, not just declined to answer. Reuses the
    exact same citation pattern linkify_citations() looks for, so
    this check and what actually renders as a colored badge can never
    disagree with each other.

    Checked FIRST: does the answer plainly open with a decline
    phrase? Models occasionally tack on a stray [1]-style bracket out
    of habit even while declining, despite the system prompt (in
    rag_core.py) explicitly telling them not to - this catch is a
    safety net for that, not the primary fix. Rely on both, not one."""
    lowered = answer_text.strip().lower()
    if any(lowered.startswith(phrase) for phrase in DECLINE_PHRASES):
        return False
    return bool(re.search(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", answer_text))


def render_assistant_message(content, sources, searched_for=None):
    """Renders one assistant turn: the answer with colored citation
    badges, an optional 'what we actually searched for' note (useful
    for showing that follow-up questions are being understood in
    context), and the retrieved-chunk source cards.

    The source panel's label depends on whether the answer actually
    cited anything. Retrieval and generation are two separate steps -
    a chunk can pass the relevance bar and still not actually answer
    the question, and the LLM is instructed to say so rather than
    force a citation. Without this distinction, "I don't know" next
    to "3 source(s)" reads as a contradiction. It isn't one - but it
    needs to say so, not make someone guess."""
    answer_html = linkify_citations(content, sources)
    st.markdown(answer_html, unsafe_allow_html=True)

    if searched_for:
        st.caption(f"🔍 Searched for: *{searched_for}*")

    n_sources = len(sources)
    if n_sources == 0:
        st.info("No chunk cleared the minimum relevance score. Try a broader search mode.")
    elif answer_used_sources(content):
        with st.expander(f"📎 {n_sources} source(s)"):
            for i, (score, entry) in enumerate(sources, start=1):
                render_source_card(i, score, entry)
    else:
        with st.expander(f"👀 {n_sources} chunk(s) considered, but none answered the question"):
            st.caption(
                "These passed the relevance bar and were shown to the model, "
                "but it didn't cite any of them - shown here for transparency, "
                "not because they back the answer above."
            )
            for i, (score, entry) in enumerate(sources, start=1):
                render_source_card(i, score, entry)


# ===== MAIN: CHAT TRANSCRIPT =====
if not st.session_state.messages:
    st.info("Ask anything about the indexed documents below - then try a follow-up like \"what about its warranty?\" to see conversational retrieval in action.")

for msg in st.session_state.messages:
    avatar = "🗂" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant":
            render_assistant_message(msg["content"], msg.get("sources", []), msg.get("searched_for"))
        else:
            st.markdown(msg["content"])

# ===== MAIN: CHAT INPUT =====
# st.chat_input renders as a box pinned to the bottom of the page,
# regardless of where in the script it's called - that's what gives
# the ChatGPT-like feel instead of a form you scroll down to.
prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    is_first_message = len(st.session_state.messages) == 0

    st.session_state.messages.append({"role": "user", "content": prompt})
    chat_store.save_message(st.session_state.conversation_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🗂"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                history_so_far = st.session_state.messages[:-1]  # everything before this new message
                answer, sources, searched_for = rag_core.answer_question_conversational(
                    prompt,
                    st.session_state.vector_store,
                    chat_history=history_so_far,
                    top_k=st.session_state.top_k,
                    min_relevance=st.session_state.min_relevance,
                    allowed_folders=user_allowed_folders()
                )
                render_assistant_message(answer, sources, searched_for)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "searched_for": searched_for
                })
                chat_store.save_message(
                    st.session_state.conversation_id, "assistant", answer,
                    sources=sources, searched_for=searched_for
                )
                if is_first_message:
                    chat_store.rename_conversation(
                        st.session_state.conversation_id,
                        chat_store.title_from_message(prompt)
                    )
            except Exception as e:
                st.error(
                    "Google's servers were briefly unavailable and every retry/"
                    "fallback model failed too - this is rare. Just ask again "
                    "in a few seconds."
                )
                with st.expander("Technical details"):
                    st.code(str(e))