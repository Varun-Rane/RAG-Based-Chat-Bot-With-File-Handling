from sentence_transformers import SentenceTransformer
import pandas as pd

text = """Chunking is the process of segmenting text into smaller, manageable portions based on length, structure or semantic meaning. It allows vector search to focus on precise information rather than entire documents. Understanding different chunking methods helps improve retrieval accuracy and model performance in Retrieval Augmented Generation pipelines.
1. Fixed-Size Chunking: Splits text into equal-sized segments based on characters or tokens.
2. Recursive Character Splitter: Splits text using multiple fallback rules to preserve structure.
3. Token-Based Chunking: Splits text based on model token limits.
4. Sentence or Semantic Chunking: Groups text based on meaning or sentence boundaries.
5. Document-Based Chunking: Breaks structured documents into logical sections.
Chunk overlap refers to the technique of including a small portion of text from the end of one chunk at the beginning of the next chunk. This helps maintain continuity between chunks and prevents important information from being lost when text is split. It is especially useful when sentences or ideas span across multiple chunks. 
Choosing the right chunk size depends on the type of document and the use case. If chunks are too large, the model may include unnecessary data. If chunk is too small, it may lose essential meaning. Some recommended chunk sizes in LangChain are:
Installing LangChain for chunking utilities.
Reading the input text file. """

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode(text)
print(embedding)
df = pd.DataFrame(embedding).T
print(df)
