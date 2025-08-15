"""Robust PDF → plain-text extractor using pdfplumber.

Usage::
    python -m wba.pdf_to_text input.pdf > output.json
"""
import sys, json, re
from pathlib import Path
import pdfplumber

CLEAN_RE = re.compile(r'\s+')

def extract(path: Path) -> list[str]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            txt = p.extract_text(x_tolerance=2, y_tolerance=2) or ""
            txt = CLEAN_RE.sub(" ", txt).strip()
            if txt:
                pages.append(txt)
    return pages

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: pdf_to_text.py <file.pdf>")
    pdf_in = Path(sys.argv[1])
    pages = extract(pdf_in)
    print(json.dumps(
        [{"page_num": i+1, "content": t} for i, t in enumerate(pages)],
        ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
