"""
rag_pipeline.py — Shared RAG logic used by step3_query.py and streamlit_app.py.

Any changes to the model, prompt, or search settings should be made here.
"""

import os
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ── Load environment variables ─────────────────────────────────────────────────
load_dotenv()

def get_secret(key):
    """Works for both local .env and Streamlit Cloud secrets."""
    try:
        import streamlit as st
        return st.secrets[key]
    except Exception:
        return os.getenv(key)

# ── Configuration ──────────────────────────────────────────────────────────────
CHROMA_HOST     = "api.trychroma.com"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL      = "meta-llama/llama-4-scout-17b-16e-instruct"
TOP_K           = 2    # Number of pages to retrieve per question

# ── Load resources ─────────────────────────────────────────────────────────────

def load_embedding_model():
    """Load and return the sentence transformer embedding model."""
    print(f"🤖 Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("   ✅ Model loaded")
    return model

def load_collection():
    CHROMA_API_KEY  = get_secret("CHROMA_API_KEY")
    CHROMA_TENANT   = get_secret("CHROMA_TENANT")
    CHROMA_DATABASE = get_secret("CHROMA_DATABASE")
    COLLECTION_NAME = get_secret("CHROMA_COLLECTION") or "spiritual-book"
    client = chromadb.HttpClient(
        ssl=True,
        host=CHROMA_HOST,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
        headers={"x-chroma-token": CHROMA_API_KEY}
    )
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection

# ── Core RAG functions ─────────────────────────────────────────────────────────

def search(question: str, collection, model: SentenceTransformer):
    """Embed the question and retrieve the top K most relevant pages."""
    question_embedding = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text"      : results["documents"][0][i],
            "pdf_page"  : results["metadatas"][0][i]["pdf_page"],
            "book_page" : results["metadatas"][0][i]["book_page"],
            "distance"  : results["distances"][0][i]
        })
    return chunks

def build_prompt(question: str, chunks: list) -> str:
    """Build the prompt to send to the LLM."""
    context = "\n\n---\n\n".join(
        [f"[Page {c['book_page']}]: {c['text']}" for c in chunks]
    )
    return f"""You are a helpful assistant for a spiritual education class based on the book "My Baba and I".
Answer the question directly using only the passages provided.
Do not start with phrases like "According to the book" or "The passages say".
Do not use bullet points or lists unless the question specifically asks for steps.
Only say "I could not find that in the book" if the passages are completely unrelated to the question.

PASSAGES FROM THE BOOK:
{context}

QUESTION:
{question}

ANSWER:"""

def ask_groq(prompt: str) -> str:
    """Send the prompt to Groq and return the answer."""
    GROQ_API_KEY    = get_secret("GROQ_API_KEY")
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024
    )
    return response.choices[0].message.content.strip()

def get_answer(question: str, collection, model) -> tuple:
    """
    Full RAG pipeline — search, build prompt, ask Groq.
    Returns (answer, chunks) so callers can show source passages too.
    If no relevant answer found, returns empty chunks so sidebar stays clean.
    """
    chunks = search(question, collection, model)
    prompt = build_prompt(question, chunks)
    answer = ask_groq(prompt)

    # If LLM couldn't find the answer, don't show source passages
    if "could not find" in answer.lower():
        return answer, []

    return answer, chunks