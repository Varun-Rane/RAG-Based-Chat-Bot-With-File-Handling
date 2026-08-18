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

# chunk_size = 500
# chunks = []

# for i in range(0, len(text), chunk_size):
#     chunk = text[i:i + chunk_size]
#     chunks.append(chunk)

# for index, chunk in enumerate(chunks, start=1):
#     print(f"/n==== Chunk {index} ====/n")
#     print(chunk)

# Chunking using Hugging Face Tokenizer 
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

chunk_size = 100
tokens = tokenizer.encode(text)
chunks = []

for i in range(0, len(tokens), chunk_size):
    chunk_tokens = tokens[i:i + chunk_size]
    chunk_text = tokenizer.decode(chunk_tokens)
    chunks.append(chunk_text)
    
for index, chunk in enumerate(chunks, start=1):
    print(f"\n==== Chunk {index} ====\n")
    print(chunk)