# 🚀 Quick Start Guide - AI Chatbot Pro

## ⚡ 30-Second Setup

### Step 1: Install Dependencies
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### Step 2: Set Environment Variable
Create `.env` file in the project root:
```
GROQ_API_KEY=your_api_key_here
```

### Step 3: Choose Your Path

#### 🎯 Path A: Quick Demo (Recommended for first use)
```bash
streamlit run documents_manager.py
```
- Upload your first document through the UI
- Auto-processes everything
- No command-line scripts needed

#### 🎯 Path B: Manual Pipeline (For power users)
```bash
# Only needed once
python 01_extraction.py
python 02_chunking.py
python 03_embedding.py

# Then use the chat
streamlit run app_enhanced.py
```

---

## 📊 What You Get

### With `documents_manager.py`:
✅ Upload PDF files
✅ Add web URLs
✅ Paste text content
✅ View document stats
✅ Delete documents
✅ Progress tracking

### With `app_enhanced.py`:
✅ Beautiful chat interface
✅ 3 chat modes (Q&A, Summarize, Explain)
✅ View source documents
✅ Query history
✅ Export conversations
✅ Analytics dashboard

---

## 🎮 First Time Usage

1. **Open Document Manager**
   ```bash
   streamlit run documents_manager.py
   ```

2. **Upload Your First Document**
   - Choose: PDF / URL / Text
   - Click Process
   - Wait for completion (watch progress bar)

3. **Open Chat App**
   ```bash
   streamlit run app_enhanced.py
   ```

4. **Start Asking Questions**
   - Type a question about your document
   - Select chat mode (Q&A, Summarize, or Explain)
   - Click Send
   - View response with sources

5. **Explore Features**
   - Check Analytics tab for stats
   - Try different chat modes
   - Export your conversation
   - Upload more documents

---

## 📁 File Purpose Guide

| File | Purpose |
|------|---------|
| `app_enhanced.py` | 🎨 Main chat interface with analytics |
| `documents_manager.py` | 📁 Upload & manage documents |
| `01_extraction.py` | 📄 Extract text from sources |
| `02_chunking.py` | ✂️ Split text into chunks |
| `03_embedding.py` | 🔗 Create embeddings |
| `04_search.py` | 🔍 Search embeddings |
| `05_chat.py` | 💬 Basic chat (original) |

---

## 🎯 Common Tasks

### Upload a PDF
```
documents_manager.py → Upload Document → PDF File → Choose file → Process
```

### Ask a Question
```
app_enhanced.py → Type question → Choose mode → Click Send
```

### View Document Stats
```
documents_manager.py → Manage Documents → See all statistics
```

### Export Conversation
```
app_enhanced.py → Sidebar → Download Conversation
```

### Delete a Document
```
documents_manager.py → Manage Documents → Delete → Confirm
```

---

## 🔧 Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "No documents found"
✅ Use `documents_manager.py` to upload
✅ Or run the extraction pipeline first

### API Key Error
✅ Check your `.env` file
✅ Verify `GROQ_API_KEY` is set
✅ Get key from [console.groq.com](https://console.groq.com)

### Slow Performance
✅ Reduce "Number of results" slider
✅ Increase "Similarity threshold"
✅ Use smaller documents

---

## 💡 Pro Tips

1. **Use Text Mode for Testing**
   - Paste content directly in Document Manager
   - Great for testing without uploading files

2. **Adjust Similarity Threshold**
   - Lower (0.3) = More context
   - Higher (0.7) = More precise

3. **Try All Chat Modes**
   - Q&A: For specific questions
   - Summarize: For overview
   - Explain: For learning

4. **Export Often**
   - Download conversations as you go
   - Backup important chats

5. **Monitor Analytics**
   - Track your query history
   - See which chat mode works best

---

## 🎓 Example Workflows

### Workflow 1: Analyze Research Paper
```
1. Open documents_manager.py
2. Upload PDF research paper
3. Open app_enhanced.py
4. Ask: "What is the main contribution?"
5. Ask: "Summarize the methodology"
6. Ask: "What are the limitations?"
7. Export conversation
```

### Workflow 2: Learn from Documentation
```
1. Paste web documentation as text
2. Ask: "Explain [topic] in simple terms"
3. Ask: "Give me examples of [feature]"
4. Ask: "How do I implement [feature]?"
5. Save useful responses
```

### Workflow 3: Batch Upload
```
1. Upload Document 1
2. Upload Document 2
3. Upload Document 3
4. Ask cross-document questions
5. Check "Documents" tab for coverage
6. Download full conversation
```

---

## 📈 Feature Comparison

| Feature | Original | Enhanced |
|---------|----------|----------|
| Chat Interface | ✓ Basic | ✓✓ Beautiful |
| Document Upload | ✗ Manual | ✓ UI-based |
| Multiple Chat Modes | ✗ | ✓ 3 modes |
| Analytics | ✗ | ✓ Dashboard |
| Source Attribution | ✓ Basic | ✓✓ Enhanced |
| Export | ✗ | ✓ Download |
| Document Manager | ✗ | ✓ Full UI |
| Settings Panel | ✗ | ✓ Advanced |

---

## 🌟 What Makes This Special

✨ **No More Command Line**: Everything in the browser
✨ **Beautiful Design**: Modern, professional UI
✨ **Multiple Modes**: Q&A, Summarization, Explanation
✨ **Full Analytics**: Track your usage
✨ **Easy Exports**: Download conversations
✨ **Drag & Drop**: Upload files easily
✨ **Real-time Progress**: See what's happening
✨ **Smart Search**: Adjustable parameters

---

## 🎯 Next: Suggest Features

Ideas for future versions:
- ✨ User authentication
- ✨ Real-time collaboration
- ✨ Mobile app
- ✨ API endpoints
- ✨ Slack integration
- ✨ Advanced search
- ✨ Team workspaces
- ✨ Document versioning

See `FEATURES.md` for complete feature roadmap!

---

## 📞 Stuck?

1. Check the troubleshooting section above
2. Read `FEATURES.md` for detailed info
3. Check `.env` file settings
4. Verify all dependencies installed
5. Check Groq API quota

---

## 🎉 Ready to Go!

You're all set! Pick your path above and start exploring! 🚀

**Questions?** Check the documentation files in the project.

Happy chatting! 💬
