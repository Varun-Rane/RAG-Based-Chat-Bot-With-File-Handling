"""
06_extraction_simple.py

Beginner-friendly document extraction helpers.

Provides `extract_to_markdown(source)` which converts a local
file path or URL to a markdown string using `docling`.
"""

from docling.document_converter import DocumentConverter


def extract_to_markdown(source: str) -> str:
    """Convert a file path or URL to markdown text.

    Args:
        source: Local path or URL to the document.

    Returns:
        Markdown string containing the document text.
    """
    converter = DocumentConverter()
    result = converter.convert(source)
    return result.document.export_to_markdown()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python 06_extraction_simple.py <file-or-url>")
    else:
        src = sys.argv[1]
        md = extract_to_markdown(src)
        print(md[:2000])
