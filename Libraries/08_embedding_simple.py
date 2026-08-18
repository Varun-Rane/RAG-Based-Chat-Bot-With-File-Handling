"""
08_embedding_simple.py

Simple wrappers to compute embeddings and index them in lancedb.

Functions:
- `get_embedder()` -> returns a sentence-transformers embedder
- `index_text_chunks(chunks, db_path)` -> stores text+embedding in lancedb
- `search(query, k)` -> returns top-k texts matching query
"""

from typing import List, Optional
import lancedb
from lancedb.embeddings import get_registry


def get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    return get_registry().get("sentence-transformers").create(name=model_name)


def index_text_chunks(chunks: List[str], db_path: str = "embedding_db") -> None:
    if not chunks:
        return

    embedder = get_embedder()
    db = lancedb.connect(db_path)

    # simple schema: text + embedding
    table = db.create_table("chunks", schema={"text": str, "embedding": list}, mode="overwrite")

    rows = []
    for text in chunks:
        emb = embedder.compute_source_embeddings([text])[0]
        rows.append({"text": text, "embedding": emb})

    table.add(rows)


def search(query: str, k: int = 5, db_path: str = "embedding_db") -> Optional[List[str]]:
    try:
        embedder = get_embedder()
        qvec = embedder.compute_query_embeddings([query])[0]
        db = lancedb.connect(db_path)
        table = db.open_table("chunks")
        results = table.search(qvec).limit(k).to_pandas()
        return [row["text"] for _, row in results.iterrows()]
    except Exception:
        return None


if __name__ == "__main__":
    # demo: run `python 08_embedding_simple.py` after you've created chunks
    print("Import this module and call `index_text_chunks()` with your chunks.")
