# Contributing to Makerspace RAG

Thank you for your interest in contributing to the Makerspace RAG project!

## Getting Started

### Development Setup

1. Fork and clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and configure
5. Run in development mode:
   ```bash
   python run.py
   ```

### Project Architecture

```
app/
├── routes/      # Flask blueprints (API endpoints)
├── services/    # Business logic (search, LLM, classification)
├── models/      # SQLAlchemy models
├── static/      # CSS/JS assets
└── templates/   # Jinja2 HTML templates
```

## Code Style

### Python
- Follow PEP 8
- Use type hints where practical
- Document functions with docstrings
- Keep functions focused and small

### JavaScript
- Use modern ES6+ syntax
- Prefer `const` over `let`, avoid `var`
- Use meaningful variable names

### CSS
- Use CSS variables for colors/spacing
- Follow BEM-like naming for components
- Keep selectors simple

## Making Changes

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `refactor/description` - Code refactoring

### Commit Messages

Use clear, descriptive commit messages:

```
Add hybrid search with configurable alpha

- Implement embedding service using Ollama
- Add TF-IDF + embedding combination
- Configure via USE_EMBEDDINGS and HYBRID_ALPHA
```

### Pull Requests

1. Create a feature branch from `main`
2. Make your changes
3. Test locally
4. Submit PR with description of changes
5. Address review feedback

## Adding Features

### New Search Features

Update `app/services/search_service.py`:
- Add new methods to `SearchService` class
- Update `get_stats()` to include new metrics
- Add configuration via environment variables

### New API Endpoints

1. Add route in appropriate blueprint (`app/routes/`)
2. Add service methods if needed
3. Update API documentation in README

### New UI Components

1. Add CSS to `app/static/css/components.css`
2. Add JS to appropriate file in `app/static/js/`
3. Update templates as needed

## Testing

### Manual Testing

```bash
# Test classifier
python -c "
from app.services.query_classifier import QueryClassifier
result = QueryClassifier.analyze('test query')
print(result.to_dict())
"

# Test search
python -c "
from app.services.search_service import get_search_service
ss = get_search_service()
print(ss.get_stats())
"
```

### API Testing

```bash
# Health check
curl http://localhost:5000/health

# Status
curl http://localhost:5000/status

# Chat
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test", "history": []}'
```

## Knowledge Base

### Adding Text Knowledge

Add to `vault.txt`:
- One chunk per line
- Keep chunks focused (200-500 words)
- Include relevant keywords

### Adding Structured Knowledge

Create JSON files in `knowledge/`:
```json
{
  "topic": {
    "item_name": {
      "description": "...",
      "keywords": ["..."]
    }
  }
}
```

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase
