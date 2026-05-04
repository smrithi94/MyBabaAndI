"""
step2_ingest.py — Embed chunks and upload to ChromaDB Cloud.

This is Step 2 of the RAG pipeline. It:
  1. Reads chunks.json (created by step1_extract.py)
  2. Generates embeddings using sentence-transformers (free, runs locally)
  3. Uploads chunks + embeddings to ChromaDB Cloud

Make sure your .env file has these values set:
    CHROMA_API_KEY=your-api-key
    CHROMA_TENANT=your-tenant-id
    CHROMA_DATABASE=your-database-name
    CHROMA_COLLECTION=spiritual-book   (optional, defaults to "spiritual-book")

Usage:
    python step2_ingest.py
    python step2_ingest.py --reset    # Wipe existing collection and re-upload
"""

import os
import json
import argparse
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ── Load environment variables ─────────────────────────────────────────────────
load_dotenv()

CHROMA_HOST      = os.getenv("CHROMA_HOST", "api.trychroma.com")
CHROMA_API_KEY   = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT    = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE  = os.getenv("CHROMA_DATABASE")
COLLECTION_NAME  = os.getenv("CHROMA_COLLECTION", "spiritual-book")

EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
CHUNKS_FILE      = "chunks.json"
BATCH_SIZE       = 50

# ── Validate environment ───────────────────────────────────────────────────────

def validate_env():
    missing = []
    if not CHROMA_API_KEY:  missing.append("CHROMA_API_KEY")
    if not CHROMA_TENANT:   missing.append("CHROMA_TENANT")
    if not CHROMA_DATABASE: missing.append("CHROMA_DATABASE")
    if missing:
        print(f"\n❌ Missing environment variables in your .env file:")
        for m in missing:
            print(f"   {m}")
        print("\n   Please add them to your .env file and try again.")
        return False
    return True

# ── Step 1: Load chunks from JSON ──────────────────────────────────────────────

def load_chunks(chunks_file: str):
    print(f"\n📂 Loading chunks from {chunks_file}...")
    if not os.path.exists(chunks_file):
        print(f"   ❌ {chunks_file} not found!")
        print(f"   Please run step1_extract.py first.")
        return None

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"   ✅ Loaded {len(chunks)} chunks")
    return chunks

# ── Step 2: Connect to ChromaDB Cloud ─────────────────────────────────────────

def connect_to_chroma():
    print(f"\n🔗 Connecting to ChromaDB Cloud...")
    try:
        client = chromadb.HttpClient(
            ssl=True,
            host=CHROMA_HOST,
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
            headers={"x-chroma-token": CHROMA_API_KEY}
        )
        # Test connection
        client.heartbeat()
        print(f"   ✅ Connected to ChromaDB Cloud")
        return client
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        print(f"   Check your API key, tenant ID, and database name in .env")
        return None

# ── Step 3: Get or create collection ──────────────────────────────────────────

def get_collection(client, reset: bool):
    print(f"\n📦 Setting up collection '{COLLECTION_NAME}'...")

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"   🗑️  Deleted existing collection")
        except Exception:
            pass  # Collection may not exist yet

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    existing = collection.count()
    if existing > 0 and not reset:
        print(f"   ⚠️  Collection already has {existing} chunks.")
        print(f"   Use --reset flag to wipe and re-upload.")
        return None

    print(f"   ✅ Collection ready")
    return collection

# ── Step 4: Generate embeddings ────────────────────────────────────────────────

def generate_embeddings(chunks: list):
    print(f"\n🤖 Loading embedding model: {EMBEDDING_MODEL}")
    print(f"   (This downloads the model on first run — may take a minute)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"   ✅ Model loaded")

    print(f"\n🔢 Generating embeddings for {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    print(f"   ✅ Embeddings generated")
    return embeddings

# ── Step 5: Upload to ChromaDB Cloud ──────────────────────────────────────────

def upload_chunks(collection, chunks: list, embeddings):
    print(f"\n☁️  Uploading to ChromaDB Cloud in batches of {BATCH_SIZE}...")
    total = len(chunks)

    for i in range(0, total, BATCH_SIZE):
        batch_chunks     = chunks[i : i + BATCH_SIZE]
        batch_embeddings = embeddings[i : i + BATCH_SIZE]

        collection.add(
            ids        = [f"chunk_{c['chunk_id']}" for c in batch_chunks],
            documents  = [c["text"] for c in batch_chunks],
            embeddings = [e.tolist() for e in batch_embeddings],
            metadatas  = [{"pdf_page": c["pdf_page"], "book_page": c["book_page"], "length": c["length"]} for c in batch_chunks]
        )

        uploaded = min(i + BATCH_SIZE, total)
        print(f"   Uploaded {uploaded}/{total} chunks...")

    print(f"   ✅ All {total} chunks uploaded!")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Embed and upload chunks to ChromaDB Cloud")
    parser.add_argument("--reset", action="store_true", help="Wipe existing collection and re-upload")
    args = parser.parse_args()

    print("="*60)
    print("  STEP 2 — Embed & Upload to ChromaDB Cloud")
    print("="*60)

    # Validate env vars
    if not validate_env():
        return

    # Load chunks
    chunks = load_chunks(CHUNKS_FILE)
    if not chunks:
        return

    # Connect to ChromaDB
    client = connect_to_chroma()
    if not client:
        return

    # Get or create collection
    collection = get_collection(client, reset=args.reset)
    if not collection:
        return

    # Generate embeddings
    embeddings = generate_embeddings(chunks)

    # Upload
    upload_chunks(collection, chunks, embeddings)

    print("\n" + "="*60)
    print("🎉 Step 2 complete!")
    print(f"   Collection '{COLLECTION_NAME}' now has {collection.count()} chunks in ChromaDB Cloud.")
    print(f"   Next: run step3_query.py to test questions against your book!")
    print("="*60)

if __name__ == "__main__":
    main()