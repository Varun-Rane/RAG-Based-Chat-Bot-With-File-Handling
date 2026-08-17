# 🤖 AI Chatbot Pro — Embedding & Chatbot Pipeline

A powerful pipeline for extracting, chunking, embedding, searching, and chatting over text sources using vector embeddings with a beautiful, modern UI.

## ✨ What's New (Enhanced Version!)

🎨 **Beautiful Modern UI** — Professional gradient design with smooth animations
💬 **3 Chat Modes** — Q&A, Summarization, Explanation
📁 **Document Manager** — Upload PDFs, URLs, or text directly
📊 **Analytics Dashboard** — Track queries and usage statistics
📤 **One-Click Export** — Download conversations as Markdown
⚙️ **Advanced Settings** — Adjustable similarity thresholds and more

👉 **[🚀 Quick Start Guide](QUICKSTART.md)** | **[📋 Full Features](FEATURES.md)**

## Project structure

### 🆕 Enhanced Apps (New!)
- **`app_enhanced.py`** — Main chat interface with beautiful UI, analytics, and 3 chat modes
- **`documents_manager.py`** — Upload and manage documents with progress tracking

### Original Pipeline
- `01_extraction.py` — extract raw text (source-specific).
- `02_chunking.py` — split extracted text into chunks.
- `03_embedding.py` — compute embeddings for chunks and write to `embedding_db/`.
- `04_search.py` — search the embedding DB for relevant chunks.
- `05_chat.py` — simple chat demo using the search pipeline.
- `05_updated_chatbot.py` — enhanced chatbot variant.
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

**Important:** Create a `.env` file with your API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

## 🚀 Quick Start (3 Steps!)

### Option 1: Document Manager (Easiest - Recommended!)
```bash
streamlit run documents_manager.py
```
✅ Upload documents through the browser
✅ Process automatically
✅ No command-line pipeline needed
✅ Chat immediately

### Option 2: Enhanced Chat (After Pipeline)
```bash
streamlit run app_enhanced.py
```
✅ Beautiful modern interface
✅ Advanced search controls
✅ Query analytics
✅ Export conversations

### Option 3: Original Pipeline (Manual)
```bash
python 01_extraction.py
python 02_chunking.py
python 03_embedding.py
python 05_chat.py
```

## 💻 Running the Full System

### First Time Setup:
```bash
# Activate environment
.\.venv\Scripts\Activate.ps1    # Windows
source .venv/bin/activate       # macOS/Linux

# Start Document Manager
streamlit run documents_manager.py
```

Then:
1. Go to `http://localhost:8501`
2. Upload your first document
3. Open new terminal tab

### Second terminal (for chat):
```bash
streamlit run app_enhanced.py
```

Both apps now work together! 🎉

## Features Breakdown

### Enhanced Chat App (`app_enhanced.py`)
- 🎨 Beautiful gradient UI with smooth animations
- 💬 Three response modes: Q&A, Summarize, Explain
- 📌 Source attribution with metadata
- 📊 Analytics dashboard with query history
- 💾 Export conversations as Markdown
- ⚙️ Adjustable search parameters
- 🔐 Secure local storage

### Document Manager (`documents_manager.py`)
- 📤 Upload PDFs, URLs, or text content
- 📊 Real-time statistics and charts
- 🗑️ Delete documents easily
- 📈 Chunk distribution analysis
- ⏱️ Progress tracking for processing
- 🎯 Document metadata viewer

## 📊 Database

- The embeddings and chunk data are stored under `embedding_db/chunks.lance/`.
- Keep backups of any important DB files before deleting or re-running the embedding step.
- Database automatically updates when you upload new documents.

## 🛠️ Utilities

- `utils/sitemap.py` — sitemap helpers used by extraction.
- `utils/tokenizer.py` — tokenization helpers for chunking/embedding.

## 🎯 Typical Workflow

## 🎯 Typical Workflow

### New Way (Recommended!)
```
1. Run: streamlit run documents_manager.py
2. Upload document (PDF/URL/Text)
3. Run: streamlit run app_enhanced.py
4. Chat with your documents
5. Export conversation
```

### Original Way (Still Works!)
```
1. python 01_extraction.py (get data)
2. python 02_chunking.py (split text)
3. python 03_embedding.py (create vectors)
4. python 05_chat.py (start chatting)
```

## 🎓 Learn More

- **[🚀 Quick Start Guide](QUICKSTART.md)** — Get up and running in 5 minutes
- **[📋 Full Features Documentation](FEATURES.md)** — Detailed feature guide and roadmap
- **[📝 Configuration Guide](CONFIG.md)** — Advanced settings and customization

## 🌟 What Can You Do?

✅ Upload any PDF document
✅ Extract text from web pages
✅ Paste text content
✅ Ask questions about documents
✅ Summarize content
✅ Explain concepts
✅ View source attribution
✅ Export conversations
✅ Track analytics
✅ Manage multiple documents

## 🔧 Tech Stack

- **UI Framework:** Streamlit
- **Vector Database:** LanceDB
- **LLM:** Groq API (Mixtral-8x7b)
- **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)
- **Document Processing:** Docling
- **Language:** Python 3.8+

## 🐛 Troubleshooting

**Issue:** "No documents in database"
```bash
# Solution: Run the document manager to upload
streamlit run documents_manager.py
```

**Issue:** "GROQ_API_KEY not set"
```
Solution: Create .env file with:
GROQ_API_KEY=your_key_here
```

**Issue:** "Module not found"
```bash
Solution: Install dependencies
pip install -r requirements.txt
```

For more troubleshooting, see [FEATURES.md](FEATURES.md#-troubleshooting)

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| [QUICKSTART.md](QUICKSTART.md) | 30-second setup guide |
| [FEATURES.md](FEATURES.md) | Complete feature list and roadmap |
| [CONFIG.md](CONFIG.md) | Configuration and advanced settings |
| README.md | This file |

## 🎨 UI Features

- **Dark-friendly design** with gradient colors
- **Responsive layout** for all screen sizes
- **Interactive charts** for analytics
- **Source attribution** with detailed metadata
- **Progress tracking** for document uploads
- **One-click export** for conversations
- **Session analytics** with real-time stats

## 🚀 Tips & Tricks

1. **Adjust Similarity Threshold** — Lower = more results, Higher = more relevant
2. **Choose Chat Mode** — Q&A for questions, Summarize for overview, Explain for learning
3. **View Sources** — Expand source box to see exact document references
4. **Export Early** — Don't lose important conversations
5. **Monitor Analytics** — Check query history in the Analytics tab

## 💡 Next Steps

1. ✅ Install dependencies
2. ✅ Create `.env` file with API key
3. ✅ Run `streamlit run documents_manager.py`
4. ✅ Upload your first document
5. ✅ Open chat app and start asking questions
6. ✅ Check [FEATURES.md](FEATURES.md) for advanced usage

## 📜 License

This project has no explicit license file. Add one if you intend to publish or share.

## 🤝 Contributing

Feel free to enhance and customize! Some ideas:
- Add new chat modes
- Integrate with more LLM providers
- Add real-time collaboration
- Create a mobile app
- Add authentication

See [FEATURES.md](FEATURES.md) for a complete roadmap of suggested features.

---

**Ready to get started?** → [Open QUICKSTART.md](QUICKSTART.md) 🚀

Questions? Check [FEATURES.md](FEATURES.md) for detailed documentation! 💡
