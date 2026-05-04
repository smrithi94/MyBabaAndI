"""
ingest.py — One-time PDF ingestion script for the Spiritual Education RAG app.

What this script does:
  1. Reads your book PDF
  2. Cleans and chunks the text
  3. Embeds each chunk using sentence-transformers
  4. Uploads everything to ChromaDB Cloud

Run this once (or whenever the book content changes):
    python ingest.py --pdf "your_book.pdf"
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Loads variables from .env into environment
import re
import argparse
import pymupdf  # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List

# ── Configuration ────────────────────────────────────────────────────────────
# Fill these in or set as environment variables



CHROMA_HOST       = os.getenv("CHROMA_HOST", "api.trychroma.com")
CHROMA_API_KEY    = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT     = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE   = os.getenv("CHROMA_DATABASE")
COLLECTION_NAME   = os.getenv("CHROMA_COLLECTION")

EMBEDDING_MODEL   = "all-MiniLM-L6-v2"  # Fast, free, good quality
CHUNK_SIZE        = 500   # Characters per chunk
CHUNK_OVERLAP     = 100   # Overlap between chunks to preserve context
BATCH_SIZE        = 50    # Number of chunks to upload at once

# ── Step 1: Extract text from PDF ────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> List[dict]:
    """Extract text page by page from the PDF."""
    print(f"📖 Opening PDF: {pdf_path}")
    doc = pymupdf.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():  # Skip blank pages
            pages.append({
                "page": page_num,
                "text": text
            })

    print(f"   ✅ Extracted text from {len(pages)} pages")
    return pages


# ── Step 2: Clean text ───────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove noise like extra whitespace, page numbers, and odd characters."""
    text = re.sub(r'\n{3,}', '\n\n', text)       # Collapse excessive newlines
    text = re.sub(r'[ \t]+', ' ', text)           # Collapse spaces/tabs
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Remove lone page numbers
    text = text.strip()
    return text


# ── Step 3: Chunk text ───────────────────────────────────────────────────────

def chunk_text(pages: List[dict], chunk_size: int, overlap: int) -> List[dict]:
    """
    Split page text into overlapping chunks.
    Each chunk carries its source page number as metadata.
    """
    chunks = []

    for page in pages:
        text = clean_text(page["text"])
        page_num = page["page"]

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at a sentence boundary for cleaner chunks
            if end < len(text):
                last_period = chunk.rfind('.')
                if last_period > chunk_size // 2:
                    end = start + last_period + 1
                    chunk = text[start:end]

            if chunk.strip():
                chunks.append({
                    "text": chunk.strip(),
                    "page": page_num
                })

            start = end - overlap  # Overlap with next chunk

    print(f"   ✅ Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks


# ── Step 4: Embed chunks ─────────────────────────────────────────────────────

def embed_chunks(chunks: List[dict], model: SentenceTransformer) -> List[dict]:
    """Generate embeddings for each chunk."""
    print(f"🔢 Embedding {len(chunks)} chunks (this may take a few minutes)...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()

    print("   ✅ Embeddings generated")
    return chunks


# ── Step 5: Upload to ChromaDB Cloud ─────────────────────────────────────────

def upload_to_chroma(chunks: List[dict], collection) -> None:
    """Upload chunks, embeddings, and metadata to ChromaDB Cloud in batches."""
    print(f"☁️  Uploading to ChromaDB Cloud in batches of {BATCH_SIZE}...")

    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]

        ids         = [f"chunk_{i + j}" for j, _ in enumerate(batch)]
        documents   = [c["text"] for c in batch]
        embeddings  = [c["embedding"] for c in batch]
        metadatas   = [{"page": c["page"]} for c in batch]

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"   Uploaded {min(i + BATCH_SIZE, total)}/{total} chunks...")

    print(f"   ✅ All {total} chunks uploaded successfully!")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF book into ChromaDB Cloud")
    parser.add_argument("--pdf", required=True, help="Path to your PDF file")
    parser.add_argument("--reset", action="store_true", help="Delete existing collection and re-ingest")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"❌ PDF file not found: {args.pdf}")
        return

    # Connect to ChromaDB Cloud
    '''print("🔗 Connecting to ChromaDB Cloud...")
    client = chromadb.HttpClient(
        ssl=True,
        host=CHROMA_HOST,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
        headers={"x-chroma-token": CHROMA_API_KEY}
    )

    # Handle collection reset
    if args.reset:
        print(f"🗑️  Deleting existing collection '{COLLECTION_NAME}'...")
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # Collection may not exist yet

    # Get or create collection (without specifying embedding function — we provide our own)
    print(f"📦 Getting/creating collection '{COLLECTION_NAME}'...")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Cosine similarity for text
    )

    existing_count = collection.count()
    if existing_count > 0 and not args.reset:
        print(f"⚠️  Collection already has {existing_count} chunks.")
        print("   Use --reset flag to delete and re-ingest. Exiting.")
        return

    # Load embedding model
    print(f"🤖 Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("   ✅ Model loaded")'''

    # Run the pipeline
    pages  = extract_text_from_pdf(args.pdf)
    chunks = chunk_text(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    #chunks = embed_chunks(chunks, model)
   #upload_to_chroma(chunks, collection)

    #print("\n🎉 Ingestion complete!")
    #print(f"   Collection '{COLLECTION_NAME}' now has {collection.count()} chunks ready for querying.")


if __name__ == "__main__":
    main()