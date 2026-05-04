"""
chat_store.py — Save and load chat history using Supabase.
Each chat session has a unique session_id so history can be browsed by session.
"""

import os
import uuid
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

def get_secret(key):
    try:
        import streamlit as st
        return st.secrets[key]
    except Exception:
        return os.getenv(key)

SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")

def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def new_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())

def save_message(username: str, question: str, answer: str, session_id: str) -> bool:
    """Save a Q&A pair to chat history with session ID."""
    try:
        supabase = get_supabase()
        supabase.table("chat_history").insert({
            "username"  : username,
            "question"  : question,
            "answer"    : answer,
            "session_id": session_id
        }).execute()
        return True
    except Exception as e:
        print(f"Failed to save message: {e}")
        return False

def load_session_messages(session_id: str) -> list:
    """
    Load all messages for a specific session.
    Returns list of {"role": "user"/"assistant", "content": "..."} dicts.
    """
    try:
        supabase = get_supabase()
        result = supabase.table("chat_history").select("*").eq(
            "session_id", session_id
        ).order("created_at", desc=False).execute()

        messages = []
        for row in result.data:
            messages.append({"role": "user",      "content": row["question"]})
            messages.append({"role": "assistant",  "content": row["answer"]})
        return messages
    except Exception as e:
        print(f"Failed to load session messages: {e}")
        return []

def load_sessions(username: str) -> list:
    """
    Load all unique sessions for a user, ordered most recent first.
    Returns list of {"session_id": "...", "first_question": "...", "created_at": "..."} dicts.
    """
    try:
        supabase = get_supabase()
        result = supabase.table("chat_history").select("*").eq(
            "username", username
        ).order("created_at", desc=False).execute()

        # Group by session_id, keep first question and latest timestamp
        sessions = {}
        for row in result.data:
            sid = row.get("session_id")
            if not sid:
                continue
            if sid not in sessions:
                sessions[sid] = {
                    "session_id"    : sid,
                    "first_question": row["question"],
                    "created_at"    : row["created_at"]
                }
            else:
                sessions[sid]["created_at"] = row["created_at"]

        # Sort by most recent first
        sorted_sessions = sorted(
            sessions.values(),
            key=lambda x: x["created_at"],
            reverse=True
        )
        return sorted_sessions
    except Exception as e:
        print(f"Failed to load sessions: {e}")
        return []

def load_history(username: str) -> list:
    """
    Load the most recent session's messages for a user.
    Used on login to restore the last chat.
    """
    sessions = load_sessions(username)
    if not sessions:
        return []
    return load_session_messages(sessions[0]["session_id"])

def format_session_label(session: dict) -> str:
    """Format a session label for display in the history panel."""
    question = session["first_question"]
    label    = question[:40] + "..." if len(question) > 40 else question
    try:
        dt   = datetime.fromisoformat(session["created_at"].replace("Z", "+00:00"))
        date = dt.strftime("%b %d")
    except Exception:
        date = ""
    return f"{label}  ({date})"