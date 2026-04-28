import pandas as pd
import lancedb
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from typing import List
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector


converter = DocumentConverter()
result = converter.convert("https://arxiv.org/pdf/2408.09869")

chunker = HybridChunker(
    max_tokens=500,
    merge_peers=True
)

chunks = list(chunker.chunk(dl_doc=result.document))

data = []
for i, chunk in enumerate(chunks, start=1):
    data.append({
        "Chunk No": i,
        "Text": chunk.text[:300],   # preview
        "Length": len(chunk.text)
    })
df = pd.DataFrame(data)
print(df)

db = lancedb.connect("embedding_db")

func = get_registry().get("sentence-transformers").create(
    name="all-MiniLM-L6-v2"
)

class ChunkMetaData(LanceModel):
    filename : str | None = None
    page_numbers : List[int] | None = None
    title : str | None = None

class ChunkData(LanceModel):
    text: str
    embedding: Vector(func.ndims()) # type: ignore
    metadata: ChunkMetaData
    
table = db.create_table("chunks", schema =ChunkData, mode = "overwrite")

processed_chunks = [
    {
        "text": chunk.text,
        "embedding": func.compute_source_embeddings([chunk.text])[0],
        "metadata": {
            "filename": chunk.meta.origin.filename,
            "page_numbers": [
                page_no
                for page_no in sorted(
                    set(
                        prov.page_no
                        for item in chunk.meta.doc_items
                        for prov in item.prov
                    )
                )
            ] or None,
            "title": chunk.meta.headings[0]
            if chunk.meta.headings else None,
        },
    }
    for chunk in chunks
]

table.add(processed_chunks)
print(table.count_rows())
print(table.to_pandas().head())