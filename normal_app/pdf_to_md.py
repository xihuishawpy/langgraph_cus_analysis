#!/usr/bin/env python3
"""Convert a PDF to a Markdown-like text file using PyMuPDF."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import fitz  # PyMuPDF


def convert_pdf_to_markdown(pdf_path: Path, output_path: Path | None = None) -> Path:
    """Convert *pdf_path* to Markdown text saved at *output_path*.

    Each page is emitted with a `## 第 N 页` heading, mirroring the ad-hoc
    structure we tested interactively. The resulting file path is returned.
    """

    pdf_path = pdf_path.expanduser().resolve()
    if output_path is None:
        output_path = pdf_path.with_suffix(".md")
    else:
        output_path = output_path.expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    parts: list[str] = [f"# {pdf_path.stem} (PyMuPDF)", ""]
    try:
        for idx, page in enumerate(doc, start=1):
            text = (page.get_text("text") or "").strip()
            if not text:
                continue
            parts.append(f"## 第 {idx} 页")
            parts.append("")
            parts.append(text)
    finally:
        doc.close()

    output_path.write_text("\n\n".join(parts), encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output Markdown path (defaults to <pdf>.md)",
    )
    args = parser.parse_args(argv)

    output_path = convert_pdf_to_markdown(args.pdf, args.output)
    print(f"Converted {args.pdf} -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
