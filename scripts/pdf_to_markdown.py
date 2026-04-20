#!/usr/bin/env python3
"""
Convert PDF files to Markdown using pymupdf4llm.

Usage:
    python scripts/pdf_to_markdown.py                         # all PDFs in references/SURVEY/
    python scripts/pdf_to_markdown.py path/to/file.pdf        # single file
    python scripts/pdf_to_markdown.py path/to/dir/            # all PDFs in a directory

Output files are written alongside the source PDFs with _text.md suffix.
"""

import sys
import pathlib
import pymupdf4llm

DEFAULT_DIR = pathlib.Path(__file__).parent.parent / \
    "CODE_LAYERS/layer14_speculative_decoding/hierarchical/references/SURVEY"


def convert(pdf_path: pathlib.Path) -> pathlib.Path:
    out_path = pdf_path.with_name(pdf_path.stem + "_text.md")
    print(f"  Converting: {pdf_path.name}")
    md = pymupdf4llm.to_markdown(str(pdf_path))
    out_path.write_text(md, encoding="utf-8")
    size_kb = len(md) // 1024
    print(f"  → {out_path.name}  ({size_kb} KB, {len(md.splitlines())} lines)")
    return out_path


def main():
    if len(sys.argv) == 1:
        target = DEFAULT_DIR
    else:
        target = pathlib.Path(sys.argv[1])

    if target.is_file() and target.suffix == ".pdf":
        pdfs = [target]
    elif target.is_dir():
        pdfs = sorted(target.glob("*.pdf"))
    else:
        print(f"Error: {target} is not a PDF file or directory.")
        sys.exit(1)

    if not pdfs:
        print(f"No PDF files found in {target}")
        sys.exit(0)

    print(f"\nConverting {len(pdfs)} PDF(s) in {target.resolve()}\n")
    results = []
    for pdf in pdfs:
        try:
            out = convert(pdf)
            results.append((pdf.name, out.name, "OK"))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((pdf.name, "-", f"FAILED: {e}"))

    print("\n--- Summary ---")
    for src, dst, status in results:
        print(f"  {status:6}  {src} → {dst}")


if __name__ == "__main__":
    main()
