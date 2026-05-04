 """
step3_query.py — Test the RAG pipeline in the terminal.

Usage:
    python step3_query.py
"""

from rag_pipeline import load_embedding_model, load_collection, get_answer

def main():
    print("="*60)
    print("  STEP 3 — RAG Query Test")
    print("="*60)

    # Load resources (shared from rag_pipeline.py)
    model      = load_embedding_model()
    collection = load_collection()

    print("\n💬 Ask a question about the book. Type 'exit' to quit.\n")

    while True:
        print("─"*60)
        question = input("Your question: ").strip()

        if question.lower() in ("exit", "quit", "q"):
            print("\nGoodbye! 👋")
            break

        if not question:
            print("Please enter a question.")
            continue

        print("\n🔍 Searching book...")
        answer, chunks = get_answer(question, collection, model)

        # Show source pages
        pages = sorted(set(c["book_page"] for c in chunks))
        print(f"   Found relevant content on pages: {pages}")

        # Print answer
        print("\n" + "="*60)
        print("ANSWER:")
        print("="*60)
        print(answer)

        # Show source passages
        print(f"\n📖 Source passages used (pages {pages}):")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n  [{i}] Book page {chunk['book_page']} (PDF page {chunk['pdf_page']}):")
            print(f"  {chunk['text'][:200]}...")

        print()

if __name__ == "__main__":
    main()