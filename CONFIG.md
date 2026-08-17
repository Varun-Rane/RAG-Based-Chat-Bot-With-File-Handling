# ⚙️ Configuration & Advanced Settings Guide

## Environment Setup

### .env File Configuration

Create a `.env` file in your project root:

```bash
# Required
GROQ_API_KEY=gsk_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Optional (defaults provided)
# HUGGINGFACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXX
# OPENAI_API_KEY=sk_XXXXXXXXXXXXXXXXXXXX
```

### Get Your API Keys

1. **Groq API Key:**
   - Visit: https://console.groq.com/
   - Sign up or login
   - Create new API key
   - Copy to `.env` file

2. **HuggingFace (Optional):**
   - Visit: https://huggingface.co/settings/tokens
   - Create new access token
   - Add to `.env` if using local models

## Streamlit Configuration

### Create `streamlit_config.toml`

Create `.streamlit/config.toml` for custom settings:

```toml
[theme]
primaryColor = "#6366f1"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[client]
showErrorDetails = true
toolbarMode = "viewer"

[logger]
level = "info"

[client]
maxUploadSize = 200  # MB

[server]
maxUploadSize = 200  # MB
```

## App Configuration

### app_enhanced.py Settings

#### In-App Sidebar Settings:

```
Chat Mode:
- Q&A (default)
- Summarize
- Explain

Number of Results:
- Min: 1
- Max: 10
- Default: 5
- Affects: How many document chunks to use

Similarity Threshold:
- Min: 0.0
- Max: 1.0
- Default: 0.3
- Lower = More results
- Higher = More relevant results
```

### documents_manager.py Settings

#### Processing Parameters:

```python
# Edit in documents_manager.py
CHUNK_SIZE = 500        # Tokens per chunk
MERGE_PEERS = True      # Merge similar chunks
MAX_UPLOAD_SIZE = 200   # MB
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

## LanceDB Configuration

### Database Location

```python
# Default: embedding_db/
db = lancedb.connect("embedding_db")

# Custom location:
db = lancedb.connect("/custom/path/embedding_db")
```

### Database Maintenance

```bash
# Backup your database
cp -r embedding_db embedding_db_backup

# Check database stats
python -c "
import lancedb
db = lancedb.connect('embedding_db')
table = db.open_table('chunks')
print(f'Chunks: {len(table.search().to_pandas())}')
"
```

## Embedding Model Configuration

### Change Embedding Model

Currently: `all-MiniLM-L6-v2` (fast, small, good)

#### Alternative Models:

```python
# In app_enhanced.py or documents_manager.py
embedder = get_registry().get("sentence-transformers").create(
    # Option 1: Better quality but slower
    name="all-mpnet-base-v2"

    # Option 2: Faster but less accurate
    name="all-mini-lm-l6-v2"

    # Option 3: Multi-lingual
    name="paraphrase-multilingual-MiniLM-L12-v2"
)
```

## LLM Configuration

### Change Groq Model

Currently: `mixtral-8x7b-32768`

```python
# In get_chat_response function
message = client.messages.create(
    model="mixtral-8x7b-32768",      # Current
    # Alternatives:
    # model="llama-2-70b-chat",
    # model="gemma-7b-it",

    messages=[...],
    max_tokens=1024,
    temperature=0.7
)
```

#### Available Models:

```
mixtral-8x7b-32768  (Fast, good balance)
llama-2-70b-chat    (Larger, better quality)
gemma-7b-it         (Smaller, faster)
```

### Adjust Response Parameters

```python
# In get_chat_response function

# Temperature: 0.0 = deterministic, 1.0 = creative
temperature=0.7

# Max tokens: Higher = longer responses
max_tokens=1024

# Top-p: Diversity of responses
# top_p=0.9
```

## Database Optimization

### Chunking Strategy

```python
# In documents_manager.py
chunker = HybridChunker(
    max_tokens=500,      # Decrease for more chunks
    merge_peers=True     # Merge similar chunks
)

# Smaller chunks:
# Pros: More precise search results
# Cons: More vectors, slower search, higher cost

# Larger chunks:
# Pros: Fewer vectors, faster
# Cons: Less precise search
```

### Search Parameters

```python
# In search_context function

# Number of results
num_results=5          # Range: 1-10

# Similarity threshold
similarity_threshold=0.3  # Range: 0.0-1.0
```

## Performance Tuning

### Optimize for Speed

```python
# Reduce search scope
num_results=3              # Instead of 5
similarity_threshold=0.6   # Instead of 0.3

# Smaller chunks
max_tokens=250            # Instead of 500

# Faster embedding model
name="all-mini-lm-l6-v2"
```

### Optimize for Quality

```python
# Increase search scope
num_results=10            # Instead of 5
similarity_threshold=0.2  # Instead of 0.3

# Larger chunks
max_tokens=750            # Instead of 500

# Better embedding model
name="all-mpnet-base-v2"
```

### Optimize for Cost

```python
# Use cheaper model
model="gemma-7b-it"       # Instead of mixtral

# Reduce API calls
num_results=3             # Fewer chunks
similarity_threshold=0.5  # Only top results

# Process in batches
# Combine multiple queries
```

## Security Configuration

### Protect API Keys

```bash
# Good: Use .env file (in .gitignore)
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Bad: Hardcoding
# api_key = "gsk_xxxx"

# Bad: Storing in code
# os.environ["GROQ_API_KEY"] = "gsk_xxxx"
```

### .gitignore Setup

```
# .gitignore
.env
.env.local
embedding_db/
*.db
*.log
__pycache__/
.streamlit/secrets.toml
```

### Use Streamlit Secrets (Production)

```bash
# Instead of .env, use Streamlit secrets
# ~/.streamlit/secrets.toml
GROQ_API_KEY = "gsk_xxxx"
```

Access in app:
```python
api_key = st.secrets["GROQ_API_KEY"]
```

## Advanced Features

### Custom Chat Prompts

Edit in `get_chat_response()`:

```python
prompts = {
    "qa": """Your custom Q&A prompt here...""",
    "summarize": """Your custom summarize prompt...""",
    "explain": """Your custom explain prompt...""",
}
```

### Custom Styling

Edit CSS in apps:

```python
st.markdown("""
    <style>
    /* Your custom CSS */
    .chat-message {
        /* Your styles */
    }
    </style>
""", unsafe_allow_html=True)
```

### Session State Customization

```python
# Add custom session variables
if "my_custom_var" not in st.session_state:
    st.session_state.my_custom_var = initial_value

# Use them
st.session_state.my_custom_var
```

## Deployment Configuration

### Streamlit Cloud

Create `streamlit_config.toml`:

```toml
[client]
showErrorDetails = false

[logger]
level = "warning"
```

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-key"
```

### Docker Configuration

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app_enhanced.py"]
```

### Environment Variables (Production)

```bash
# Linux/macOS
export GROQ_API_KEY="gsk_xxxx"
streamlit run app_enhanced.py

# Windows PowerShell
$env:GROQ_API_KEY="gsk_xxxx"
streamlit run app_enhanced.py

# Docker
docker run -e GROQ_API_KEY="gsk_xxxx" myapp:latest
```

## Monitoring & Logging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Track Performance

```python
import time

start = time.time()
# ... your code ...
elapsed = time.time() - start
print(f"Execution time: {elapsed:.2f}s")
```

### Monitor Database

```python
db = init_db()
table = db.open_table("chunks")

# Get stats
stats = {
    "total_chunks": len(table.search().to_pandas()),
    "database_size": "estimate using file system",
}
print(stats)
```

## Troubleshooting Configuration

### Issue: Model not found

```python
# Solution: Check available models
from lancedb.embeddings import get_registry
registry = get_registry().get("sentence-transformers")
print(registry.list_available())
```

### Issue: Out of memory

```python
# Solutions:
1. Reduce max_tokens (250 instead of 500)
2. Reduce num_results (3 instead of 5)
3. Use smaller embedding model
4. Process smaller documents
```

### Issue: Rate limited

```python
# Solutions:
import time

# Add delay between API calls
time.sleep(1)

# Or:
message = client.messages.create(
    model="gemma-7b-it",  # Cheaper model
    ...
)
```

## Best Practices

✅ **DO:**
- Use `.env` file for secrets
- Add `.env` to `.gitignore`
- Use Streamlit secrets in production
- Monitor API usage
- Backup your database
- Test configuration before production
- Use appropriate chunk sizes
- Monitor performance metrics

❌ **DON'T:**
- Hardcode API keys
- Commit secrets to git
- Use huge chunk sizes
- Set threshold too low
- Use too many results
- Ignore error messages
- Skip backups
- Use default passwords

## Configuration Presets

### Quick Start Preset
```
temperature=0.7
max_tokens=1024
num_results=5
similarity_threshold=0.3
max_chunk_tokens=500
```

### Production Preset
```
temperature=0.5
max_tokens=512
num_results=5
similarity_threshold=0.4
max_chunk_tokens=500
```

### Budget Preset
```
temperature=0.5
max_tokens=256
num_results=3
similarity_threshold=0.6
model="gemma-7b-it"
```

### Quality Preset
```
temperature=0.3
max_tokens=2048
num_results=10
similarity_threshold=0.2
model="mixtral-8x7b-32768"
embedding_model="all-mpnet-base-v2"
```

## Further Reading

- [Streamlit Configuration](https://docs.streamlit.io/library/advanced-features/configuration)
- [LanceDB Documentation](https://lancedb.com/docs)
- [Groq API Documentation](https://console.groq.com/docs)
- [Sentence-Transformers Models](https://www.sbert.net/docs/pretrained_models.html)

---

**Need help?** Check the main README or FEATURES.md for more information! 🚀
