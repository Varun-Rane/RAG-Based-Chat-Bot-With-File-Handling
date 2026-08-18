"""
07_chunking_simple.py

A very simple, dependency-free chunker for beginners.

It groups paragraphs into chunks by approximate character size.
"""

from typing import List


def chunk_text(markdown_text: str, max_chars: int = 2000) -> List[str]:
    """Split markdown text into chunks by paragraphs.

    Args:
        markdown_text: Full document text in markdown or plain text.
        max_chars: Approximate maximum characters per chunk.

    Returns:
        A list of text chunks (strings).
    """
    paragraphs = [p.strip() for p in markdown_text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    cur = ""

    for p in paragraphs:
        if cur and len(cur) + len(p) + 2 > max_chars:
            chunks.append(cur.strip())
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p

    if cur:
        chunks.append(cur.strip())

    return chunks


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python 07_chunking_simple.py <markdown-file>")
    else:
        path = Path(sys.argv[1])
        text = path.read_text(encoding="utf-8")
        ch = chunk_text(text)
        print(f"Produced {len(ch)} chunks. First chunk preview:\n")
        print(ch[0][:1000])
