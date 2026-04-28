import pandas as pd
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

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
