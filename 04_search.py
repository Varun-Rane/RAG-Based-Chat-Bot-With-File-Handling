import lancedb 
import pandas as pd
from lancedb.embeddings import get_registry

db = lancedb.connect("embedding_db")

table = db.open_table("chunks")

func = get_registry().get("sentence-transformers").create(
    name="all-MiniLM-L6-v2"
)

query = "What is Docling"

query_vector = func.compute_query_embeddings([query])[0]

results = (
    table.search(query_vector)
    .limit(5)
    .to_pandas()
)

print("\nTop Search Results:\n")

for i, row in results.iterrows():
    print("=" * 100)
    print(f"Result #{i+1}")
    print()

    text = row["text"].replace("\n", " ").strip()

    print(text[:900])
    print()
