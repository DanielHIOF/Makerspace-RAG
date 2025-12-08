# Makerspace RAG - Hogskolen i Ostfold

An AI-powered assistant for Hogskolen i Ostfold's Makerspace, using Retrieval-Augmented Generation (RAG) to answer questions about equipment, safety, and maker techniques.

## Features

- **AI Chat Interface** - Natural language conversations about 3D printing, laser cutting, electronics, and more
- **Hybrid Search** - Combines TF-IDF and semantic embeddings for accurate context retrieval
- **Multi-language Support** - Norwegian (default) and English responses
- **Adaptive Explanations** - Beginner, normal, and expert modes
- **Component Inventory** - Track and search makerspace components
- **Admin Panel** - Manage knowledge base and upload documents
- **Local LLM** - Uses Ollama for privacy-focused, offline AI

## Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed and running
- MariaDB/MySQL (optional, SQLite for development)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/makerspace-rag.git
cd makerspace-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Pull required Ollama models
ollama pull llama3
ollama pull mxbai-embed-large

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings

# Run the application
python launcher.py
```

The application will open automatically at `http://localhost:5000`

## Project Structure

```
makerspace-rag/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── config.py            # Environment configuration
│   ├── extensions.py        # Flask extensions (db, login)
│   ├── models/
│   │   ├── component.py     # Component model
│   │   └── user.py          # Admin user model
│   ├── routes/
│   │   ├── public.py        # Chat, status, health endpoints
│   │   ├── auth.py          # Login/logout
│   │   ├── admin.py         # Admin panel
│   │   └── api.py           # REST API for components
│   ├── services/
│   │   ├── search_service.py      # Hybrid TF-IDF + embedding search
│   │   ├── embedding_service.py   # Ollama embeddings
│   │   ├── llm_service.py         # Ollama chat integration
│   │   ├── knowledge_service.py   # JSON knowledge management
│   │   └── query_classifier.py    # Query analysis with confidence
│   ├── static/
│   │   ├── css/             # Stylesheets
│   │   └── js/              # JavaScript modules
│   └── templates/           # Jinja2 templates
├── frontend/               # React frontend (for integration)
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── hooks/          # Custom React hooks
│   │   ├── services/       # API services
│   │   └── styles/         # CSS styles
│   └── package.json        # npm dependencies
├── knowledge/               # JSON knowledge files
├── vault.txt               # Text knowledge base
├── launcher.py             # Unified launcher script
├── run.py                  # Development entry point
└── requirements.txt
```

## React Frontend

The project includes a React frontend for easy integration with other React applications.

### Development

```bash
# Start Flask backend
python run.py

# In another terminal, start React dev server
cd frontend
npm install
npm run dev
```

The React app runs on `http://localhost:3000` and proxies API requests to Flask.

### Production Build

```bash
cd frontend
npm run build
```

The build outputs to `app/static/react/` and Flask will automatically serve it.

### Integration

The React components can be imported into other React projects:

```jsx
import { ChatProvider, useChat } from './hooks/useChat';
import { ChatArea, InputArea, Header } from './components';
```

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
# Flask
FLASK_ENV=development
SECRET_KEY=your-secret-key

# Database
DB_HOST=localhost
DB_PORT=3306
DB_USER=makerspace
DB_PASSWORD=your-password
DB_NAME=makerspace_rag

# Ollama
OLLAMA_HOST=http://127.0.0.1:11434
LLM_MODEL=llama3
EMBEDDING_MODEL=mxbai-embed-large

# Search
USE_EMBEDDINGS=true
HYBRID_ALPHA=0.5

# Admin
ADMIN_USERNAME=admin
ADMIN_PASSWORD=change-this
```

### Search Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_EMBEDDINGS` | `true` | Enable semantic embeddings for hybrid search |
| `HYBRID_ALPHA` | `0.5` | Balance between TF-IDF (0) and embeddings (1) |

## Usage

### Chat Commands

Use slash commands to customize responses:

| Command | Description |
|---------|-------------|
| `/nybegynner` | Simple explanations for beginners |
| `/ekspert` | Technical, detailed responses |
| `/english` | Respond in English |
| `/prusa` | Focus on Prusa 3D printers |
| `/laser` | Focus on laser cutting |
| `/cnc` | Focus on CNC milling |
| `/elektronikk` | Focus on electronics |
| `/lodding` | Focus on soldering |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat interface |
| `/chat` | POST | Send chat message |
| `/status` | GET | System status |
| `/health` | GET | Health check |
| `/admin` | GET | Admin panel (auth required) |
| `/api/components` | GET/POST | Component CRUD |
| `/api/components/<id>` | GET/PUT/DELETE | Single component |

### Example Chat Request

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "hvordan bruker jeg prusa 3d printer?", "history": []}'
```

## Architecture

### Hybrid Search

The system uses a hybrid search approach:

1. **TF-IDF** - Fast keyword-based search with Norwegian-English query expansion
2. **Semantic Embeddings** - Ollama mxbai-embed-large (1024 dimensions)
3. **Weighted Combination** - Configurable alpha parameter

```
Query -> [TF-IDF Search] -> Normalized Scores ─┐
      -> [Embedding Search] -> Normalized Scores ├─> Combined -> Top-K
                                                 │
Score = (1 - alpha) * TF-IDF + alpha * Embedding ┘
```

### Query Classification

Queries are classified with confidence scores:

| Category | Description | Priority |
|----------|-------------|----------|
| FEILSOKING | Troubleshooting problems | Highest |
| OPPLARING | Learning/tutorials | High |
| VERKTOY_HMS | Equipment/safety info | Medium |
| GENERELL | General queries | Default |

### Knowledge Sources

1. **vault.txt** - Main text knowledge base (chunked, ~5700 chunks)
2. **knowledge/*.json** - Structured equipment, safety rules, room info
3. **Database** - Component inventory (MariaDB/SQLite)

## Development

### Running in Development

```bash
# With debug mode
python run.py

# Or use the unified launcher
python launcher.py
```

### Testing Services

```bash
# Test the classifier
python -c "
from app.services.query_classifier import QueryClassifier
result = QueryClassifier.analyze('hvordan bruker jeg prusa?')
print(result.to_dict())
"

# Test search service
python -c "
from app.services.search_service import get_search_service
ss = get_search_service()
results = ss.search('3d printing filament')
print(f'Found {len(results)} results')
"

# Test embedding service
python -c "
from app.services.embedding_service import get_embedding_service
es = get_embedding_service()
emb = es.get_embedding('test')
print(f'Embedding: {len(emb)} dimensions')
"
```

### Adding Knowledge

1. **Text content**: Add to `vault.txt` (one chunk per line)
2. **Structured data**: Add JSON files to `knowledge/`
3. **Via Admin Panel**: Upload PDFs or paste text directly

## Deployment

### Production Checklist

- [ ] Set `FLASK_ENV=production`
- [ ] Generate secure `SECRET_KEY`
- [ ] Configure MariaDB instead of SQLite
- [ ] Set strong `ADMIN_PASSWORD`
- [ ] Use WSGI server (gunicorn/waitress)
- [ ] Enable HTTPS via reverse proxy
- [ ] Pre-generate embeddings before first request

### Running with Gunicorn (Linux)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 "app:create_app()"
```

### Running with Waitress (Windows)

```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:create_app
```

## Models Used

| Model | Purpose | Size |
|-------|---------|------|
| llama3 | Main chat model | ~4GB |
| llama3.2:1b | Conversation compression | ~1GB |
| mxbai-embed-large | Semantic embeddings | ~670MB |

## Performance Notes

- **First startup**: Slow if `USE_EMBEDDINGS=true` (generates embeddings for ~5700 chunks)
- **Subsequent startups**: Fast (embeddings cached in `embeddings_cache.json`)
- **Disable embeddings**: Set `USE_EMBEDDINGS=false` for faster startup (TF-IDF only)

## Troubleshooting

### Ollama not running
```bash
# Start Ollama
ollama serve

# Check status
curl http://localhost:11434/api/tags
```

### Database connection issues
```bash
# For development, use SQLite (default)
# Set FLASK_ENV=development

# For production with MariaDB
net start MariaDB  # Windows
sudo systemctl start mariadb  # Linux
```

### Slow first request
The first request loads the search index and builds embeddings. Set `USE_EMBEDDINGS=false` to skip embedding generation.

## License

MIT License

## Acknowledgments

- Hogskolen i Ostfold Makerspace
- Ollama team for local LLM infrastructure
- Flask and scikit-learn communities
