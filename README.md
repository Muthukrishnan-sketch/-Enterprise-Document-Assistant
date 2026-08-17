# Enterprise Document Assistant — RAG Demo

A working (not scripted) retrieval-augmented-generation assistant that
answers questions from your own documents and cites the exact file
**and folder** each fact came from.

```
Documents -> Chunking -> Embedding -> Vector DB        (offline, once)
Question  -> Embed & search -> Top chunks -> LLM answer (online, every question)
```

---

## 1. What's in this folder

| File | What it is | Do you edit it? |
|---|---|---|
| `documents/` | Sample company files, organized into department folders (`HR_Policies/`, `IT_Security/`, `Finance/`, `Remote_Work/`, `Technical_Manuals/`) | Yes — replace with real files later |
| `rag_core.py` | The engine: chunking, embedding, retrieval, answer generation | Only when tuning behavior |
| `chat_store.py` | Persists conversations to SQLite — knows nothing about RAG | Rarely |
| `1_ingest.py` | Terminal command to build the index | Rarely |
| `2_ask.py` | Terminal command to ask questions (no browser) | Rarely |
| `app.py` | **The web demo you show your TL** | Yes — this is the UI |
| `requirements.txt` | List of Python packages needed | No |
| `faiss_index.bin` | Generated automatically after indexing — the searchable vector index (FAISS's own binary format) | Never by hand |
| `chunk_metadata.json` | Generated alongside it — folder/file/text for each vector, in matching order | Never by hand |
| `chat_history.db` | Generated automatically — every saved conversation (SQLite) | Never by hand |

Everything in `rag_core.py` is used by all three of `1_ingest.py`, `2_ask.py`,
and `app.py`, so the chunking/retrieval logic only exists in one place.

---

## 2. Setup (do this once)

This project uses **Google's Gemini API**, not OpenAI — Gemini has a
genuinely free tier that needs only a Google account, no credit card.

1. **Install Python 3.9+** if you don't have it: check with `python --version`
2. **Open a terminal in this folder** (`rag-demo/`)
3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
4. **Get a free API key:**
   - Go to `aistudio.google.com/apikey` and sign in with a Google account
   - Click **Create API key** — no billing setup required
   - Copy the key (starts with `AIza...`) immediately, it's shown once
5. **Set it as an environment variable** (never paste it directly into the
   code — that's how keys leak):

   macOS / Linux:
   ```
   export GEMINI_API_KEY="AIza...your key..."
   ```
   Windows (PowerShell):
   ```
   $env:GEMINI_API_KEY="AIza...your key..."
   ```
   This only lasts for your current terminal session — you'll need to set
   it again each time you open a new terminal. (For a permanent setting,
   search "set environment variable permanently" for your OS.)

**Free tier limits:** 1,500 requests/day and generous per-minute limits on
the Flash model this project uses — indexing these 6 sample documents
(~20 chunks) and asking dozens of questions won't come close to hitting
that. If you ever do see a rate-limit error, wait a few seconds and retry.

---

## 3. Running it

**Option A — the demo you show your TL (recommended):**
```
streamlit run app.py
```
This opens a browser tab. It indexes the sample documents automatically
on first load, then you can ask questions and adjust the sliders live.

**Option B — terminal only, for quick testing:**
```
python 1_ingest.py
python 2_ask.py
```

Try asking:
- "How many days can I work from home?"
- "What's the warranty on the X200 pump?"
- "How much can I claim for home office equipment?"
- "What's the CEO's personal phone number?" (should say it doesn't know — nothing in the documents answers this, which is the point)

---

## 4. How a question actually gets answered (walk-through)

Say you ask: **"How many vacation days carry over to next year?"**

1. `rag_core.embed_text()` sends that question to Gemini's embedding model,
   getting back a vector — a list of 768 numbers representing its meaning.
2. `rag_core.retrieve()` compares that vector to every chunk's stored vector
   using `cosine_similarity()` — pure math, no keyword matching happens
   anywhere in this project.
3. The chunk from `HR_Policies/leave_policy.txt` about carry-forward limits
   scores highest, clears the minimum relevance bar, and gets kept.
4. `rag_core.build_prompt()` labels that chunk `[1] HR_Policies/leave_policy.txt (chunk 0)`
   and hands it to the LLM along with your question.
5. The LLM is instructed to answer **only** from that text and to cite `[1]`.
6. `app.py` displays the answer, then the "Retrieved chunks" panel shows
   you exactly which folder, file, and chunk backed it up — this is the
   part your earlier attempts were missing.

---

## 5. Tuning guide — what happens if you change each knob

| Setting | Where | Increase it → | Decrease it → |
|---|---|---|---|
| `CHUNK_SIZE` | sidebar slider, or `rag_core.DEFAULT_CHUNK_SIZE` | Each chunk holds more surrounding context, but a match becomes less precise (the LLM gets more text to sift through, some irrelevant) | Chunks are more precise/focused, but a single fact can get split across two chunks and be missed |
| `CHUNK_OVERLAP` | sidebar slider, or `rag_core.DEFAULT_CHUNK_OVERLAP` | Safer against cutting a sentence at a chunk boundary; more stored chunks (slightly more indexing cost) | Cheaper/faster indexing, but higher risk that a fact sitting right on a boundary gets cut in half |
| `TOP_K` | sidebar slider, or `rag_core.DEFAULT_TOP_K` | LLM sees more candidate chunks — better for questions needing multiple facts, but more noise and cost per question | Faster/cheaper, but a relevant chunk might get left out if it ranked 4th and TOP_K=3 |
| `MIN_RELEVANCE` | sidebar slider, or `rag_core.DEFAULT_MIN_RELEVANCE` | Stricter filter — fewer false-positive sources, but a real answer can get filtered out if it scores just under the bar | More permissive — catches weaker matches, but risks feeding the LLM irrelevant context |
| `EMBEDDING_DIM` | `rag_core.py` top of file | Larger (up to 3072, Gemini's full size) = slightly better semantic precision, more storage per chunk | `768` (current default) = ~25% the storage of full-size, near-identical retrieval quality — a good default for a small demo |
| `CHAT_MODEL` | `rag_core.py` top of file | A Pro-tier Gemini model = stronger reasoning over retrieved context, but not on the free tier | `gemini-2.5-flash` (current default) = free tier, fast, fine for straightforward Q&A over policy documents |
| `temperature` | `rag_core.generate_answer()` | Higher = more varied/creative wording between runs | `0` (current default) = same question gives the same answer every time — what you want for policy answers |
| The system prompt | `rag_core.build_prompt()` | Removing "answer ONLY using the context" is what causes hallucination — the LLM would start guessing from its general training knowledge instead of your documents | N/A |

**Important distinction for your test:** chunk size and overlap change
*what gets stored* — they require clicking **Re-index**. Top-K and
minimum relevance only change *what gets picked at question time* — they
apply instantly, no re-indexing needed. If asked "why does one need a
button and the other doesn't," that's the answer.

---

## 6. Customizing the UI (`app.py`)

Every UI section in `app.py` is marked with a comment like
`# ===== SIDEBAR: PIPELINE SETTINGS =====`. Search for these to jump
around. Examples of easy changes:

- **Page title / browser tab icon:** edit the `st.set_page_config(...)` call near the top
- **App heading:** edit the string inside `st.title("📄 Enterprise Document Assistant")`
- **Button label:** edit the string inside `st.button("Ask", type="primary")`
- **Default slider values:** change the numbers in `rag_core.DEFAULT_CHUNK_SIZE` etc.
- **Add a new sidebar option:** add another `st.slider(...)` or `st.selectbox(...)`
  call inside the `with st.sidebar:` block, near the existing ones

---

## 7. How this differs from your two earlier attempts

- **The HTML mockup** had a JavaScript object called `CONVOS` with
  hand-written questions and hand-written answers — asking anything not
  in that list just fell through to a generic "I don't know" message.
  Nothing was actually being read, embedded, or searched.
- **The Streamlit screenshot version** was doing real retrieval, but its
  citations only showed the filename (`leave_policy.txt`), not where that
  file actually lives. This version adds the folder path all the way
  through (`rag_core.py` → `entry["path"]` → the citation and the
  "Retrieved chunks" panel), because in a real 5-million-document company,
  "which folder is this in" is exactly what someone needs to actually go
  find the source file.

---

## 8. Talking points: how this scales to 5 million documents

You won't build this at full scale by Friday, and no one expects you to —
but you should be able to explain what changes:

- **Vector storage:** ✅ done — `rag_core.py` uses FAISS (`faiss.IndexFlatIP`)
  instead of a Python loop comparing every chunk by hand. See section 16 for
  exactly what this is and isn't ready for yet.
- **Ingestion:** still runs as a single `for` loop, not a batch/parallel job
  across many workers — and re-embeds every document on every re-index,
  rather than only the ones that actually changed. Fine at 24 chunks, not
  fine at 5 million.
- **File types:** ✅ done — PDF, Word, and Excel each have their own
  text-extraction step (section 13). Email is the one named format from the
  original brief still untouched.
- **Access control:** still not built. Every chunk would need permission
  metadata, and `retrieve()` would need to filter by the asking user's
  access before ranking results — this needs real authentication to exist
  first, which this project doesn't have yet.
- **Cost:** embedding 5 million documents costs real money and time — this
  is usually why teams start with a scoped pilot (one department, a few
  thousand documents) exactly like this demo, then expand.

---

## 9. Troubleshooting

| Problem | Likely cause |
|---|---|
| Authentication / permission error mentioning API key | `GEMINI_API_KEY` isn't set in this terminal session — redo step 4/5 in Setup |
| `429` / rate-limit error during indexing | You've hit the free tier's per-minute limit — wait ~30-60 seconds and click Re-index again |
| App shows a red "Couldn't reach Google's servers" message | This is expected, graceful behavior, not a bug — see section 10 below |

## 10. Built-in resilience: retries and model fallback

Google's servers occasionally return a `503 UNAVAILABLE` ("high demand")
error — this happens to every user, on every pricing tier, and has nothing
to do with your key or your code. `rag_core.py` handles this automatically:

- **Retry with backoff**: on a `503` or `429` error, the call waits 1
  second, then 2, then gives up and moves on — instead of failing on the
  very first blip.
- **Model fallback**: if `CHAT_MODEL` (`gemini-2.5-flash`) keeps failing
  after retries, `generate_answer()` automatically tries each model in
  `CHAT_MODEL_FALLBACKS` in turn — different models rarely get overloaded
  at the exact same moment.
- **Graceful failure**: if every model fails anyway (rare), `app.py` shows
  a calm on-screen message instead of crashing with a raw traceback — an
  important detail for a live demo in front of your TL.

This is a genuinely good thing to be able to explain in your review: it's
the difference between a script that works once on your laptop and a
system built to stay up when a dependency has a bad moment.
| `ModuleNotFoundError: No module named 'streamlit'` | Run `pip install -r requirements.txt` again, check you're in the right terminal/environment |
| App says "Indexed 0 documents into 0 chunks" | You're running `streamlit run app.py` from the wrong folder — `cd` into `rag-demo/` first |
| Answers seem generic / not from your documents | Check the "Retrieved chunks" panel — if it's empty, lower "Minimum relevance score" in the sidebar |

## 11. Visual design

The UI now has a deliberate identity instead of default Streamlit
styling, built around the one thing this app actually proves: every
fact is traceable to a department folder. That shows up in three
places, all driven by one color per folder:

- **Inline in the answer itself** — `[1]` becomes a small colored
  badge like `HR · 1`, so grounding is visible in the sentence, not
  just in a panel below it
- **A legend strip** under the title, so the colors mean something
  before you've asked anything
- **The "Retrieved chunks" cards** — a colored left edge + badge +
  relevance meter per source

**To add a new department color** (e.g. if you add a `Legal/` folder
later): open `app.py`, find `FOLDER_STYLES` near the top, and add a
line like `"Legal": {"color": "#9B6BD9", "label": "LEGAL"},`. Any
folder not listed there still works automatically — it just renders
in neutral gray instead of getting its own color.

**Overall color theme** (background, accent color) lives in
`.streamlit/config.toml`, using Streamlit's own supported theming —
edit the hex values there to change the base palette without touching
`app.py` at all.

One honest limitation worth knowing for your review: Streamlit only
allows so much visual customization without fighting the framework.
The fonts and colors are real and load reliably; the actual widgets
(sliders, buttons, the text input box) keep Streamlit's native shapes
since restyling those relies on targeting Streamlit's internal HTML
structure, which isn't a stable, documented API and can break on a
Streamlit version upgrade. That's a reasonable, deliberate trade-off
for a working demo, not an oversight.

## 12. Conversational chat (follow-up questions)

`app.py` is now a real chat interface, not a single question/answer
form — built with Streamlit's native `st.chat_message` / `st.chat_input`,
not custom HTML, so it behaves reliably (scrolling, auto-focus, message
bubbles) without fighting the framework.

**The problem a chat interface creates:** in a single-turn Q&A, every
question is self-contained. In a real conversation, it usually isn't —
if you just asked about the X200 pump and then ask *"what's the warranty
on it?"*, the word "it" carries all the meaning. Plain embedding search
has no idea what "it" refers to, so retrieval quality collapses after
the first message unless something fixes this.

**The fix — `rag_core.condense_question()`:** before every follow-up
question is searched, a fast LLM call rewrites it into a fully
standalone version using the last few turns of conversation — *"what's
the warranty on it?"* becomes *"what is the warranty on the X200
pump?"*. **That** rewritten version is what actually gets embedded and
searched, not your literal words. The first message in any conversation
skips this step entirely (there's no history yet to need rewriting),
so it doesn't waste an extra API call.

Each assistant reply shows a small **"🔍 Searched for: ..."** line —
that's the condensed query, shown on purpose so you can point at it
live and prove the follow-up was genuinely understood in context, not
just answered from the model's general memory with nothing behind it.

**What this costs:** roughly two model calls per follow-up turn instead
of one (condense, then generate) — still trivial on the free tier for a
demo like this, but worth knowing as a real cost driver at scale.

**Code map for this feature:**
| Function | File | Role |
|---|---|---|
| `condense_question()` | `rag_core.py` | Rewrites a follow-up into a standalone search query |
| `build_prompt()` | `rag_core.py` | Now optionally includes recent chat history for natural tone |
| `answer_question_conversational()` | `rag_core.py` | Ties condense → retrieve → generate together for the chat UI |
| `answer_question()` | `rag_core.py` | Unchanged, single-turn — still used by `1_ingest.py`/`2_ask.py` |
| chat transcript loop + `st.chat_input` | `app.py` | Renders history, handles new turns |

**Try this for your review:** ask *"Tell me about the X200 pump"*, then
follow up with *"What's its warranty?"* with no other words — then show
the "Searched for" line under the second answer as proof the system
resolved "its" correctly.

## 13. Multi-format ingestion (PDF, Word, Excel)

`documents/` is no longer limited to `.txt` files. `rag_core.py` now
recognizes four formats, each with its own text-extraction step
before your existing chunking pipeline runs unchanged:

| Format | Library | How it's turned into text |
|---|---|---|
| `.txt` | built-in | Read as-is, unchanged from before |
| `.pdf` | `pdfplumber` | Extracted page by page, with a `[Page N]` marker before each page's text |
| `.docx` | `python-docx` | Paragraphs in reading order, plus tables flattened into `Header: value \| Header: value` rows |
| `.xlsx` | `openpyxl` | Every sheet's rows turned into `Header: value` lines using row 1 as column headers, one `[Sheet: Name]` block per sheet |

Three real sample files were added to prove this end-to-end, not just
in theory: `HR_Policies/benefits_enrollment_guide.docx` (paragraphs +
a table), `Finance/approved_vendors.xlsx` (a spreadsheet), and
`IT_Security/data_retention_policy.pdf` (a 2-page PDF). Re-index and
ask *"What plans are available for benefits enrollment?"* or *"Which
vendor has the largest annual spend?"* to see them retrieved and cited
exactly like the original `.txt` files.

**Why tables and spreadsheets get flattened instead of kept as
grids:** embeddings work on prose-like text, not 2D structure. A raw
table pasted into a chunk loses its row/column meaning once it's just
characters. Turning each row into a self-contained sentence like
`"Vendor Name: Acme | Category: Office Supplies | Contract Expiry:
2027-03-01"` keeps every fact anchored to its label, so a question
about "Acme's contract expiry" can match that single row directly.

**What happens to a file that can't be read:** it's skipped, not
crashed on — a corrupt PDF, a scanned PDF with no real text layer, or
a format you haven't `pip install`-ed the library for all show up as
a `⚠️ Skipped ...` line in the sidebar's "Indexing log" expander,
with a reason. A bad file no longer takes down the whole indexing run
or silently vanishes without explanation.

**Known limitation, worth knowing for your review:** `pdfplumber`
only reads the text *layer* already embedded in a PDF. A scanned
document (a photo of a page, with no real text underneath) will
extract as empty and get skipped — reading those requires OCR, which
is a reasonable "next phase" answer if asked, not something silently
broken here.

**Adding a 5th format later:** write one `extract_text_from_X()`
function following the same pattern as the four above, then add one
line to the `EXTRACTORS` dictionary near the top of `rag_core.py` —
`load_documents()` itself never needs to change.

## 14. Uploading documents from the browser

You no longer need PyCharm or the file system to add a document —
the sidebar now has a **📤 Upload documents** section:

1. Pick an existing department folder from the dropdown, or choose
   **+ New folder...** and type a name (e.g. `Legal`) to create one
   on the spot
2. Choose one or more `.pdf`, `.docx`, `.xlsx`, or `.txt` files
3. Click **Save and index**

The file is saved into `documents/<that folder>/` on disk (the exact
same place a file placed there manually would go), then the app
re-indexes automatically using whatever chunk size/overlap is
currently active — no separate re-index click needed.

**Note:** if a new folder name isn't already in `FOLDER_STYLES`
(`app.py`), it still works — it just renders in neutral gray instead
of getting its own color. Add it to `FOLDER_STYLES` afterward if you
want it colored (see section 11).

**A file with the same name as an existing one is overwritten** —
there's no version history, so re-uploading a corrected policy simply
replaces the old one on next index.

## 15. Persistent conversations (survives closing the tab / restarting)

Conversations no longer live only in `st.session_state`, which is
wiped every time the browser tab closes or the Streamlit process
restarts. They're now saved to a local SQLite database,
`chat_history.db`, generated automatically the first time you run
the app (same idea as `faiss_index.bin` - not something you create
by hand).

**Why SQLite instead of a JSON file:** a JSON file works for one
conversation, but appending a single new message means rewriting the
*entire* file, and "give me the 5 most recent conversations" means
loading and parsing everything just to sort it. SQLite is still a
single file — no server to install or run — but `sqlite3` ships with
Python itself (zero new dependencies), and it lets the app ask for
exactly the rows it needs.

**What changed in the UI:** the sidebar now opens with a **💬
Conversations** section instead of jumping straight to search
settings — a "+ New conversation" button, and a clickable list of
past conversations (auto-titled from each one's first message, like
ChatGPT), each with a small 🗑️ to delete it. The 🟢 marks whichever
conversation is currently open.

**What happens on a fresh app load:** rather than starting blank,
the app resumes whatever conversation was most recently active. Ask
a question, close the tab, reopen `localhost:8501` — your last
conversation, with full history and citations, is exactly where you
left it.

**Code map:**
| Function | File | Role |
|---|---|---|
| `init_db()`, `create_conversation()`, `list_conversations()`, `save_message()`, `load_messages()`, `delete_conversation()`, `rename_conversation()`, `title_from_message()` | `chat_store.py` | All persistence — knows nothing about RAG |
| Conversation sidebar section, save-on-every-turn calls | `app.py` | Wires the chat UI to `chat_store.py` |
| *(untouched)* | `rag_core.py` | Has no idea `chat_store.py` exists |

**What's stored per message:** role, text, the condensed "searched
for" query, and a trimmed copy of each cited source (folder, file,
chunk index, score, text) — enough to redraw the exact same colored
citation cards later. The 768-number embedding vector is deliberately
*not* stored per message, since it's only needed during retrieval
and would otherwise bloat the database for no benefit.

**Known limitation, worth knowing for your review:** this is a
single shared database file with no user accounts — every person
running this app on this machine sees the same conversation list.
Multi-user separation would need real authentication first, which
is the same "not built yet" gap already flagged for access control
in section 8.

## 16. Real vector search with FAISS — HNSW, parallel, incremental

Retrieval no longer works by comparing your question against every chunk
in a Python `for` loop. `rag_core.py` uses **FAISS** (`faiss-cpu`), the
same vector-search library used inside many production RAG systems. Three
separate things about *how* indexing and search work changed — each is
worth being able to explain on its own.

**1. Sub-linear search — `IndexHNSWFlat`, not `IndexFlatIP`.** This is a
real algorithmic change, not just a faster implementation of the same
algorithm. The earlier version (`IndexFlatIP`) compared a query against
*every* stored vector, always — correct, but inherently linear in
collection size. HNSW builds a navigable graph over the vectors at index
time (`HNSW_M`, `HNSW_EF_CONSTRUCTION`), so a search only has to visit a
small, roughly constant-size neighborhood of that graph
(`HNSW_EF_SEARCH`), not the whole collection. That's what stops search
time from growing linearly with document count. The honest tradeoff:
HNSW is **approximate** — it can very occasionally miss the single truest
best match in exchange for not scanning everything. At this project's
scale that's not noticeable; the chosen parameters (M=32,
efConstruction=200, efSearch=64) are standard, widely-used defaults that
hold up into the millions of vectors, not numbers tuned to look good on
a small demo. `retrieve()` also bounds how many raw candidates it asks
FAISS for (`HNSW_SEARCH_CANDIDATES=200`) before applying
`min_relevance`/`allowed_folders` filtering — asking an approximate
index for "every vector, ranked" would defeat the entire point of using
it. See the detailed docstring on `retrieve()` for the specific edge
case this creates (a chunk relevant to a heavily-restricted user could
theoretically rank outside that bound at a scale far beyond this
project's) and the real production fix (FAISS's own ID-based
pre-filtering) — a known next step, not implemented here.

**2. Parallel embedding.** Each `embed_text()` call is a network round
trip to Gemini — embedding chunks one at a time means indexing time
scales directly with chunk count. `build_vector_store()` now embeds up
to `DEFAULT_MAX_EMBED_WORKERS` (5) chunks **concurrently** via
`ThreadPoolExecutor`, so wall-clock indexing time is roughly (chunks ÷ 5)
round trips instead of (chunks) round trips. Higher isn't free — too many
concurrent requests can itself trigger the free tier's rate limit, which
is why this defaults to a modest number, not the highest one that works.

**3. Incremental re-indexing.** Re-indexing no longer means re-embedding
every document from scratch every time. Each document's extracted text
gets hashed (`text_hash`); if a document's hash matches what it was last
time, its existing chunks are **reused** — their actual embedding vectors
are pulled back out of the previous FAISS index via `.reconstruct()`,
not recomputed — and the embedding API is never called for that document
at all. Only new or genuinely changed documents get (re-)embedded. This
is what makes uploading one new document fast: with 12 documents already
indexed and one new upload, only that one new document is embedded, not
all 13. `app.py`'s `run_indexing()` only enables this when chunk
size/overlap haven't changed from the last index — a different chunk
size shifts every chunk boundary even for unchanged text, so reuse is
correctly skipped in that case, forcing a full rebuild.

**Storage is still split into two files**, unchanged from before:
`faiss_index.bin` (just the vectors, FAISS's own binary format) and
`chunk_metadata.json` (folder, filename, path, chunk text — in the same
order as the vectors, so result #7 always lines up with `metadata[7]`).

**Why normalization still matters:** every embedding is L2-normalized
(`normalize_vector()`) before being added to the index, and the question
is normalized the same way before searching — `IndexHNSWFlat` with
`METRIC_INNER_PRODUCT` computes a raw inner product, not cosine
similarity, but the inner product of two *normalized* vectors is
mathematically identical to cosine similarity. **Getting the metric
argument right matters more than it looks:** `IndexHNSWFlat`'s default
metric is L2 (Euclidean) distance, not inner product — omitting
`faiss.METRIC_INNER_PRODUCT` explicitly would silently flip every
relevance score's meaning (lower would mean "more similar" instead of
higher), breaking every `min_relevance` threshold in the app without
throwing a single error.

**If asked "is this ready for 5 million documents":** the honest answer
is that the *algorithm* now genuinely is (sub-linear search, incremental
ingestion) — but running it at that scale also needs enough RAM to hold
millions of vectors and their HNSW graph in memory, which is a hardware/
deployment question, not something the code alone resolves. No laptop
demo can prove that part either way.

## 17. Access control per department folder

There's now a real login system, and it actually restricts what a user
can retrieve — not just what the UI chooses to display.

**First-time setup — creating the first admin account:**

You can't log into the app to create a user before any user exists.
Run this once, in a terminal (not the browser):
```
python manage_users.py
```
Follow the prompts to create an admin account (an admin sees every
department, including any added later). After that, log into `app.py`
normally, and an admin can create every other account from inside the
app itself — sidebar → **🛡️ Manage users** — no terminal needed again.

**What a restricted (non-admin) user experiences:**
- Only sees the department folders they've been granted, everywhere —
  the color legend, the upload folder picker, the "Indexed N documents"
  count
- Their conversations are private to them — a different user logging in
  sees their own conversation list, not anyone else's
- Asking a question **only ever retrieves chunks from folders they're
  allowed to see** — a document from a folder they can't access is
  never fetched in the first place, so it can't leak into an answer

**Why enforcement happens in `retrieve()`, not the UI:** the tempting,
easier approach would be to just hide restricted folders from the
sidebar and call it done. That's decoration, not security — anyone who
noticed a folder's citation format could still ask about it directly,
and if retrieval itself doesn't check permissions, they'd get an
answer anyway. Every user's question — restricted or not — passes
through the exact same `rag_core.retrieve()` function; the only
difference is whether `allowed_folders` is `None` (admin, no
restriction) or a specific list. A chunk outside that list is filtered
out before it's ever assembled into a prompt, not after.

**Passwords:** hashed with PBKDF2-HMAC-SHA256 (200,000 iterations) and
a random salt per user — see the detailed explanation at the top of
`auth_store.py`. Two users who happen to pick the same password get
completely different stored hashes. Login failures don't reveal
whether the username or the password was wrong, on purpose.

**Rate limiting on login attempts:** after 5 failed attempts for a
username within 15 minutes, further attempts are rejected outright —
**including the correct password** — until the window passes. That last
part is deliberate and worth understanding: if a correct password were
still accepted during lockout, the lockout would do nothing to stop
someone guessing, since the real owner logging in wouldn't reset an
attacker's guess count either. `verify_login()` raises `LoginLockedError`
distinctly from a plain wrong-password `None`, so the UI can show "try
again in a few minutes" instead of a misleading "wrong password."

**"Remember me":** Streamlit has no built-in browser cookie API, so
this doesn't use a real httponly session cookie the way a typical web
app would. Checking "Remember me" instead places a random token in the
page's URL (`st.query_params`) — revisiting that same URL logs back in
automatically for 30 days, without retyping a password. **Real security
tradeoff, not a minor detail:** because the token lives in the URL
rather than an httponly cookie, anyone who has that exact URL can sign
in as that user, the same as a password-reset link. Don't paste a URL
containing `?remember_token=...` into a chat, email, or screenshot. A
real production deployment should replace this with a proper httponly
cookie instead — this is a reasonable, honest solution for a
Streamlit-based internal tool, not the final word on the approach.

**Password reset:** no email-based "forgot password" flow — that needs
real SMTP infrastructure this project doesn't have, and deliberately
doesn't take on. What exists instead, in the **Manage users** panel:
an admin can reset any user's password directly (they don't need the
old one — that's the point, for someone locked out or who forgot it),
and any logged-in user can change their own password after re-entering
their current one first (so a hijacked-but-unlocked browser tab can't
quietly lock the real owner out).

**Honest limitations that remain, worth knowing for your review:**
- `users.db` isn't encrypted at rest. The password hashes inside it are
  safe even if someone reads the file directly (that's the whole point
  of hashing + salting), but this is still a single local file with no
  disk-level encryption, same as `chat_history.db` and `faiss_index.bin`.
- Rate limiting is per-username, not per-IP or per-device — someone
  could still spread guesses across many different usernames without
  ever triggering one account's lockout. A real production system would
  add IP-based limiting too.
- No CAPTCHA, no audit log of who logged in when or which admin action
  was taken by whom, no 2FA. Reasonable gaps for an internal tool at
  this stage, not gaps a real production deployment should carry
  forward unexamined.

## 18. Email and database ingestion

Both named data sources from the original brief that were still
untouched are now supported — same pattern as everything else:
each format gets its own `extract_text_from_X()` function, and the
chunking/embedding/retrieval pipeline after that point never needs
to know or care where the text came from.

| Format | Library | Notes |
|---|---|---|
| `.eml` | Python's built-in `email` module | The universal, portable email format — every client, including Outlook, can export or save messages this way. Prefers the plain-text body; falls back to stripping HTML tags if the email is HTML-only. |
| `.msg` | `extract-msg` (third-party) | Outlook's own binary format. See the caveat below. |
| `.db` / `.sqlite` / `.sqlite3` | Python's built-in `sqlite3` module | Every table's rows become "Column: value" lines, the same idea as the Excel handler — see section 13. |

**Two real sample emails and a sample database** were added to prove
this end-to-end: `IT_Security/password_reset_notice.eml` (plain
text), `HR_Policies/benefits_deadline_update.eml` (HTML-only, on
purpose — to actually exercise the tag-stripping path, not just the
easy case), and `IT_Security/it_asset_inventory.db` (a small SQLite
database with a 4-row asset table). Re-index and try *"What are the
password requirements?"* or *"Who has the Dell Latitude laptop?"*

**Be precise about the database scope, for your review:** this reads
a SQLite *file* sitting in `documents/`, the same way every other
format here works. It does **not** connect to a live database server
— Postgres, MySQL, SQL Server. That would mean handling connection
strings and credentials, a meaningfully different and more sensitive
feature than "read this file," and one this project deliberately
doesn't take on. The honest answer, if asked: a real company would
export or snapshot the relevant tables from their live database to a
`.db` file on a schedule, and drop that file in here the same way
they'd drop in an exported spreadsheet.

**Honest gap, worth stating plainly: `.msg` was not tested the way
everything else in this project was.** Every other extractor here —
PDF, Word, Excel, `.eml`, SQLite — was verified by generating a real
file and running the actual extraction code against it before it
shipped. `.msg` is Outlook's proprietary binary format; producing a
realistic one requires Outlook itself, which wasn't available while
building this. The code is written carefully against `extract-msg`'s
documented API, but it's unverified. **Test it yourself** before
relying on it: export one real email from Outlook as `.msg`, drop it
in a department folder, re-index, and confirm it doesn't show up in
the sidebar's "Indexing log" as a `⚠️ Skipped` entry.