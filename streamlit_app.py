"""
streamlit_app.py — My Baba And I RAG Application
3-column layout: History | Chat | Sources
"""

import logging
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import streamlit as st
from rag_pipeline import load_embedding_model, load_collection, get_answer
from auth import sign_up, login
from chat_store import save_message, load_sessions, load_session_messages, new_session_id, format_session_label

st.set_page_config(page_title="My Baba And I — Q&A", page_icon="🙏", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    .stApp { background-color: #FAF7F2; }
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }

    /* Left and right column dark background */
    [data-testid="column"]:first-child {
        background-color: #2C1A0E;
        border-radius: 14px;
        padding: 1rem !important;
        min-height: 90vh;
    }
   [data-testid="column"]:first-child * {
        color: #F5E6D3 !important;
    }
    [data-testid="column"]:first-child .username-text {
        color: #000000 !important;
    }
    [data-testid="column"]:last-child {
        background-color: #2C1A0E;
        border-radius: 14px;
        padding: 1rem !important;
        min-height: 90vh;
    }
    [data-testid="column"]:last-child * {
        color: #F5E6D3 !important;
    }

    /* All buttons in left column */
    [data-testid="column"]:first-child .stButton button {
        background-color: transparent !important;
        color: #F5E6D3 !important;
        border: none !important;
        text-align: left !important;
        font-size: 0.82rem !important;
        border-radius: 6px !important;
        padding: 0.3rem 0.6rem !important;
        width: 100% !important;
    }
    [data-testid="column"]:first-child .stButton button:hover {
        background-color: #3D2410 !important;
    }

    /* Scrollable container inside left col */
    [data-testid="column"]:first-child [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #2C1A0E !important;
        border: 1px solid #4D3020 !important;
        border-radius: 8px !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #FFFFFF;
        border: 1px solid #E8DDD0;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: #2C1A0E !important;
    }

    /* Page badges */
    .page-badge {
        display: inline-block;
        background-color: #C4A882;
        color: #2C1A0E !important;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 4px 4px 4px 0;
    }

    /* Panel titles */
    .panel-title {
        color: #C4A882 !important;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 1px solid #4D3020;
        padding-bottom: 0.4rem;
        margin-bottom: 0.8rem;
    }
    .panel-text { color: #A08070 !important; font-size: 0.82rem; }

    /* Welcome box */
    .welcome-box {
        background: linear-gradient(135deg, #FFF8F0, #FFF2E6);
        border: 1px solid #E8DDD0;
        border-radius: 14px;
        padding: 2.5rem;
        text-align: center;
        margin: 2rem auto;
    }

    /* Disclaimer */
    .disclaimer {
        text-align: center;
        font-size: 0.75rem;
        color: #B0968A;
        margin-top: 1rem;
        padding-top: 0.8rem;
        border-top: 1px solid #E8DDD0;
    }

    /* Auth inputs */
    input[type="text"], input[type="password"] {
        background-color: #FFFFFF !important;
        color: #2C1A0E !important;
    }
    [data-testid="stTextInput"] label { color: #2C1A0E !important; }
    [data-testid="stTabs"] button { color: #2C1A0E !important; }
    h4 { color: #2C1A0E !important; }
    .stCaption p { color: #8B6F5C !important; }
    div[data-testid="stTabs"] { max-width: 440px; margin: 0 auto; }
</style>
""", unsafe_allow_html=True)

# ── Cached resources ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_resources():
    model      = load_embedding_model()
    collection = load_collection()
    return model, collection

# ── Session state ──────────────────────────────────────────────────────────────
if "logged_in"    not in st.session_state: st.session_state.logged_in    = False
if "username"     not in st.session_state: st.session_state.username     = None
if "messages"     not in st.session_state: st.session_state.messages     = []
if "sources"      not in st.session_state: st.session_state.sources      = []
if "session_id"   not in st.session_state: st.session_state.session_id   = new_session_id()
if "sessions"     not in st.session_state: st.session_state.sessions     = []
if "show_history" not in st.session_state: st.session_state.show_history = True

# ── Auth screen ────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    st.markdown(
        "<div style='text-align:center;padding:2.5rem 0 1.5rem'>"
        "<div style='font-size:3rem'>🙏</div>"
        "<div style='font-size:2rem;font-weight:700;color:#2C1A0E'>My Baba And I</div>"
        "</div>", unsafe_allow_html=True)

    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:
        tab_login, tab_signup = st.tabs(["🔑  Login", "✨  Sign Up"])
        with tab_login:
            st.caption("Login to continue your journey")
            st.markdown("")
            lu = st.text_input("Username", key="lu", placeholder="Enter your username")
            lp = st.text_input("Password", type="password", key="lp", placeholder="Enter your password")
            st.markdown("")
            if st.button("Login", use_container_width=True, type="primary", key="login_btn"):
                if not lu or not lp:
                    st.error("Please enter your username and password.")
                else:
                    r = login(lu, lp)
                    if r["success"]:
                        st.session_state.logged_in  = True
                        st.session_state.username   = r["username"]
                        st.session_state.session_id = new_session_id()
                        st.session_state.sessions   = load_sessions(r["username"])
                        st.session_state.messages   = []
                        st.session_state.sources    = []
                        st.rerun()
                    else:
                        st.error(r["error"])

        with tab_signup:
            st.markdown("#### Create an account 🌟")
            st.caption("Join the spiritual education community")
            st.markdown("")
            su = st.text_input("Choose a username", key="su", placeholder="At least 3 characters")
            sp = st.text_input("Choose a password", type="password", key="sp", placeholder="At least 6 characters")
            sc = st.text_input("Confirm password", type="password", key="sc", placeholder="Re-enter your password")
            st.markdown("")
            if st.button("Sign Up", use_container_width=True, type="primary", key="signup_btn"):
                if sp != sc:
                    st.error("Passwords do not match.")
                else:
                    r = sign_up(su, sp)
                    if r["success"]:
                        st.success("✅ Account created! Please switch to the Login tab.")
                    else:
                        st.error(r["error"])
    st.stop()

# ── Logged in ──────────────────────────────────────────────────────────────────
model, collection = load_resources()
left_col, chat_col, right_col = st.columns([1, 2.5, 1])

# ── LEFT COLUMN ────────────────────────────────────────────────────────────────
with left_col:
    # Username
   st.markdown(
        "<div style='font-size:0.95rem;font-weight:600;margin-bottom:0.3rem;color:#1a73e8'>👤 "
        + st.session_state.username + "</div>",
        unsafe_allow_html=True
    )
    st.markdown("<hr style='border-color:#4D3020;margin:0.3rem 0 0.8rem'>", unsafe_allow_html=True)

    # Toggle button
    toggle_icon = "▼" if st.session_state.show_history else "▶"
    if st.button(f"🕐 Chat History {toggle_icon}", use_container_width=True, key="toggle_history"):
        st.session_state.show_history = not st.session_state.show_history
        st.rerun()

    # Scrollable history list
    if st.session_state.show_history:
        if not st.session_state.sessions:
            st.markdown("<p class='panel-text'>Your past chats will appear here.</p>", unsafe_allow_html=True)
        else:
            with st.container(height=350):
                for session in st.session_state.sessions:
                    label     = format_session_label(session)
                    is_active = session["session_id"] == st.session_state.session_id
                    btn_label = ("▶ " if is_active else "") + label
                    if st.button(btn_label, key="hist_" + session["session_id"], use_container_width=True):
                        st.session_state.session_id = session["session_id"]
                        st.session_state.messages   = load_session_messages(session["session_id"])
                        st.session_state.sources    = []
                        st.rerun()

    st.markdown("<hr style='border-color:#4D3020;margin:0.8rem 0'>", unsafe_allow_html=True)

    if st.button("➕  New Chat", use_container_width=True, key="new_chat"):
        st.session_state.session_id = new_session_id()
        st.session_state.messages   = []
        st.session_state.sources    = []
        st.rerun()

    if st.button("🚪  Logout", use_container_width=True, key="logout"):
        st.session_state.logged_in  = False
        st.session_state.username   = None
        st.session_state.messages   = []
        st.session_state.sources    = []
        st.session_state.sessions   = []
        st.session_state.session_id = new_session_id()
        st.rerun()

# ── MIDDLE COLUMN — chat ───────────────────────────────────────────────────────
with chat_col:
    st.markdown(
        "<div style='text-align:center;padding:1rem 0;border-bottom:1px solid #E8DDD0;margin-bottom:1.2rem'>"
        "<div style='font-size:1.8rem;font-weight:700;color:#2C1A0E'>🙏 My Baba And I</div>"
        "<div style='font-size:0.9rem;color:#8B6F5C;font-style:italic'>A spiritual companion — ask questions, explore the teachings</div>"
        "</div>", unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown(
            "<div class='welcome-box'>"
            "<div style='font-size:2.5rem;margin-bottom:0.8rem'>📚</div>"
            "<div style='font-size:1.2rem;font-weight:600;color:#2C1A0E;margin-bottom:0.5rem'>Welcome to the Book Q&A</div>"
            "<div style='font-size:0.9rem;color:#8B6F5C;line-height:1.6'>"
            "Ask any question about <em>My Baba And I</em>.<br>"
            "Answers are drawn directly from the pages of the book,<br>"
            "with source page numbers shown on the right."
            "</div></div>", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask a question about the book..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the book..."):
                answer, chunks = get_answer(question, collection, model)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.sources = chunks
        save_message(st.session_state.username, question, answer, st.session_state.session_id)
        st.session_state.sessions = load_sessions(st.session_state.username)
        st.rerun()

    st.markdown(
        "<div class='disclaimer'>Answers are based solely on \"My Baba And I\". Always refer to the book for complete context.</div>",
        unsafe_allow_html=True)

# ── RIGHT COLUMN — source pages ────────────────────────────────────────────────
with right_col:
    st.markdown("<div class='panel-title'>📖 Source Pages</div>", unsafe_allow_html=True)

    if not st.session_state.sources:
        st.markdown("<p class='panel-text'>Page references will appear here after you ask a question.</p>", unsafe_allow_html=True)
    else:
        pages = sorted(set(c["book_page"] for c in st.session_state.sources))
        badges = "".join("<span class='page-badge'>Page " + str(p) + "</span>" for p in pages)
        st.markdown(badges, unsafe_allow_html=True)