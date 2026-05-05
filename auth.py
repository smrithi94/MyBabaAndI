"""
auth.py — User authentication using Supabase.

Handles:
  - User signup (with hashed password)
  - User login (verify password hash)
"""

import os
import hashlib
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

def get_secret(key):
    try:
        import streamlit as st
        return st.secrets[key]
    except Exception:
        return os.getenv(key)

def get_supabase():
    url = get_secret("SUPABASE_URL")
    key = get_secret("SUPABASE_KEY")
    return create_client(url, key)
    
def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def sign_up(username: str, password: str) -> dict:
    """
    Register a new user.
    Returns {"success": True} or {"success": False, "error": "..."}
    """
    if not username or not password:
        return {"success": False, "error": "Username and password cannot be empty."}

    if len(username) < 3:
        return {"success": False, "error": "Username must be at least 3 characters."}

    if len(password) < 6:
        return {"success": False, "error": "Password must be at least 6 characters."}

    try:
        supabase = get_supabase()
        supabase.table("users").insert({
            "username": username.strip().lower(),
            "password_hash": hash_password(password)
        }).execute()
        return {"success": True}
    except Exception as e:
        error = str(e)
        if "duplicate" in error.lower() or "unique" in error.lower():
            return {"success": False, "error": "Username already taken. Please choose another."}
        return {"success": False, "error": f"Signup failed: {error}"}

def login(username: str, password: str) -> dict:
    """
    Verify login credentials.
    Returns {"success": True} or {"success": False, "error": "..."}
    """
    if not username or not password:
        return {"success": False, "error": "Please enter your username and password."}

    try:
        supabase = get_supabase()
        result = supabase.table("users").select("*").eq(
            "username", username.strip().lower()
        ).execute()

        if not result.data:
            return {"success": False, "error": "Username not found."}

        user = result.data[0]
        if user["password_hash"] != hash_password(password):
            return {"success": False, "error": "Incorrect password."}

        return {"success": True, "username": user["username"]}
    except Exception as e:
        return {"success": False, "error": f"Login failed: {str(e)}"}