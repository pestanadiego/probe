"""
one-time preprocessing script: convert PDF datasheets to plain text.

usage:
    python scripts/pdf_to_text.py                      # converts all PDFs in corpus/datasheets/pdf/
    python scripts/pdf_to_text.py path/to/file.pdf     # converts a single PDF
"""

import os
import sys

from pdfminer.high_level import extract_text


PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus", "datasheets", "pdf")
TEXT_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus", "datasheets", "text")


def pdf_to_text(pdf_path: str, output_dir: str) -> str:
    """
    Extract plain text from a PDF and write it to output_dir/<stem>.txt.

    Returns the path to the written .txt file.
    """
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"converting {pdf_path} → {stem}.txt")
    output_path = os.path.join(output_dir, f"{stem}.txt")

    print(f"extracting: {pdf_path}")
    text = extract_text(pdf_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    size_kb = os.path.getsize(output_path) // 1024
    print(f"  wrote {size_kb} KB → {output_path}")
    return output_path


def convert_directory(input_dir: str, output_dir: str):
    """Convert all PDFs found in input_dir into output_dir."""
    pdf_files = [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        print(f"no PDF files found in {input_dir}")
        return
    for pdf_path in pdf_files:
        pdf_to_text(pdf_path, output_dir)


if __name__ == "__main__":
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(TEXT_DIR, exist_ok=True)

    if len(sys.argv) == 2:
        path = sys.argv[1]
        if not os.path.isfile(path):
            print(f"error: file not found: {path}")
            sys.exit(1)
        pdf_to_text(path, TEXT_DIR)
    else:
        convert_directory(PDF_DIR, TEXT_DIR)
