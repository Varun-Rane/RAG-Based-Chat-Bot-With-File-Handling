# Python For AI — Embedding & Chatbot Pipeline

A small pipeline for extracting, chunking, embedding, searching, and chatting over text sources using vector embeddings and a simple chatbot interface.

## Project structure

- `01_extraction.py` — extract raw text (source-specific).
- `02_chunking.py` — split extracted text into chunks.
- `03_embedding.py` — compute embeddings for chunks and write to `embedding_db/`.
- `04_search.py` — search the embedding DB for relevant chunks.
- `05_chat.py` — simple chat demo using the search pipeline.
- `06_updated_chatbot.py` — enhanced chatbot (main file open in editor).
- `embedding_db/` — local embedding store (Lance DB directory).
- `utils/` — helper modules (`sitemap.py`, `tokenizer.py`).

## Requirements

Install dependencies from `requirements.txt` (recommended to use a virtualenv).

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\Activate.ps1 # Windows PowerShell
pip install -r requirements.txt
```

## Typical workflow

1. Run `01_extraction.py` to pull raw text from sources.
2. Run `02_chunking.py` to break text into smaller chunks.
3. Run `03_embedding.py` to compute embeddings and populate `embedding_db/`.
4. Use `04_search.py` to query the DB and inspect results.
5. Run `05_chat.py` or `06_updated_chatbot.py` to interact with the chatbot.

Example (Linux/macOS):

```bash
python 01_extraction.py
python 02_chunking.py
python 03_embedding.py
python 04_search.py --query "your question"
python 06_updated_chatbot.py
```

On Windows PowerShell, activate the venv then run the same `python` commands.

## Data

- The embeddings and chunk data are stored under `embedding_db/chunks.lance/`.
- Keep backups of any important DB files before deleting or re-running the embedding step.

## Utilities

- `utils/sitemap.py` — sitemap helpers used by extraction.
- `utils/tokenizer.py` — tokenization helpers for chunking/embedding.

## Notes & Tips

- The repo uses a simple, linear pipeline; you can re-run only the steps that changed (e.g., re-run `03_embedding.py` after changing chunking).
- If you use a cloud or external LLM/embedding provider, ensure API keys are set as environment variables before running scripts.

## License

This project has no explicit license file. Add one if you intend to publish or share.

---

If you'd like, I can (a) add a quick example dataset and script to verify the pipeline, or (b) create a CONTRIBUTING or LICENSE file next.
