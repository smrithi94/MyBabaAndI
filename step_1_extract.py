"""
step1_extract.py — Extract text page-wise from your PDF into a local JSON file.

This is Step 1 of the RAG pipeline. It:
  1. Reads your PDF page by page
  2. Skips specified pages (front matter, table of contents, index, etc.)
  3. Stops at a configured page
  4. Cleans each page's text
  5. Saves one entry per page to chunks.json

No API keys or internet connection needed for this step!

Usage:
    python step1_extract.py --pdf "MyBabaAndI.pdf"
"""

import re
import os
import json
import argparse
import pymupdf  # PyMuPDF

# ── Configuration ─────────────────────────────────────────────────────────────
PAGE_OFFSET   = 17    # PDF page - book page (PDF page 58 = Book page 41)
OUTPUT_FILE   = "chunks.json"
STOP_AT_PAGE  = 299   # Stop processing after this PDF page number

# Pages to skip (table of contents, index, blank pages, etc.)
SKIP_PAGES = {
    1, 2, 3, 4, 5, 6,                        # Front matter
    11, 12, 13, 14, 15, 16, 17, 18, 19,      # Table of contents
    82, 83,                                   # Skip
    195, 196, 200, 201,                       # Skip
    246, 247, 250, 251,                       # Skip
    259, 260, 261, 262, 263, 264, 265,        # Skip
    278, 279, 280, 281                        # Skip
}

# ── Step 1: Extract text page by page ─────────────────────────────────────────

def extract_pages(pdf_path: str):
    print(f"\n📖 Opening PDF: {pdf_path}")
    doc = pymupdf.open(pdf_path)

    total_pages = len(doc)
    print(f"   Total pages in PDF : {total_pages}")
    print(f"   Stopping at page   : {STOP_AT_PAGE}")
    print(f"   Pages to skip      : {len(SKIP_PAGES)}")

    pages = []
    skipped = []

    for page_num in range(total_pages):
        pdf_page  = page_num + 1               # 1-based PDF page number
        book_page = pdf_page - PAGE_OFFSET     # Actual printed book page number

        # Stop at configured page
        if pdf_page > STOP_AT_PAGE:
            print(f"   ⏹️  Stopped at PDF page {STOP_AT_PAGE} (book page {STOP_AT_PAGE - PAGE_OFFSET})")
            break

        # Skip specified pages
        if pdf_page in SKIP_PAGES:
            skipped.append(pdf_page)
            continue

        page = doc[page_num]
        text = clean_text(page.get_text("text"))

        if text:  # Skip blank pages
            pages.append({
                "chunk_id"  : len(pages),
                "pdf_page"  : pdf_page,
                "book_page" : book_page,
                "text"      : text,
                "length"    : len(text)
            })

    print(f"   ⏭️  Skipped pages      : {skipped}")
    print(f"   ✅ Pages extracted    : {len(pages)}")
    return pages

# ── Step 2: Clean text ────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    return text.strip()

# ── Step 3: Save to JSON ──────────────────────────────────────────────────────

def save_to_json(pages: list, output_file: str):
    print(f"\n💾 Saving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Saved {len(pages)} pages to {output_file}")
    print(f"   Open {output_file} in any text editor to inspect the results!")

# ── Step 4: Print sample ──────────────────────────────────────────────────────

def print_sample(pages: list):
    print("\n" + "="*60)
    print("🔍 SAMPLE — First 3 pages")
    print("="*60)
    for page in pages[:3]:
        print(f"\nBook page {page['book_page']} (PDF page {page['pdf_page']}) | {page['length']} chars")
        print("─"*60)
        print(page["text"][:400])
        if page["length"] > 400:
            print("...")
    print("\n" + "="*60)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract PDF pages into chunks.json")
    parser.add_argument("--pdf", required=True, type=str, help="Path to your PDF file")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"\n❌ PDF not found: '{args.pdf}'")
        print(f"   Files in current folder:")
        for f in os.listdir("."):
            print(f"     {f}")
        return

    # Run the pipeline
    pages = extract_pages(args.pdf)
    print_sample(pages)
    save_to_json(pages, OUTPUT_FILE)

    print("\n🎉 Step 1 complete!")
    print(f"   Each page of your book is now saved as one entry in '{OUTPUT_FILE}'.")
    print(f"   Next: run python step2_ingest.py --reset")

if __name__ == "__main__":
    main()
