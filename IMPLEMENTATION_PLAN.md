# Makerspace-RAG Implementation Plan

## Project Overview

**Project:** Makerspace RAG System for Høgskolen i Østfold  
**Purpose:** AI-powered Q&A assistant for makerspace topics (3D printing, laser cutting, electronics, etc.)  
**Developer:** Daniel Nilsen Johansen  
**Status:** ✅ COMPLETE - Ready for deployment

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MAKERSPACE RAG SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   User      │───▶│  Flask App  │───▶│  TF-IDF Search      │ │
│  │  (Browser)  │    │  (app.py)   │    │  (scikit-learn)     │ │
│  └─────────────┘    └──────┬──────┘    └──────────┬──────────┘ │
│                            │                      │             │
│                            ▼                      ▼             │
│                     ┌─────────────┐        ┌─────────────┐     │
│                     │   Ollama    │        │  vault.txt  │     │
│                     │  (llama3)   │        │ (4666 lines)│     │
│                     └─────────────┘        └─────────────┘     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  UNIFIED LAUNCHER (launcher.py / START.bat)                     │
│  - Auto-starts Ollama if not running                            │
│  - Verifies/pulls required models                               │
│  - Starts Flask server                                          │
│  - Opens browser automatically                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
C:\Food-E\Makerspace-RAG\
├── START.bat              # ⭐ Double-click to launch (Windows)
├── launcher.py            # ⭐ Unified launcher script
├── app.py                 # Main Flask web application
├── simple_rag.py          # CLI version with semantic embeddings
├── upload.py              # Tkinter GUI for document uploads
├── vault.txt              # Knowledge base (4666+ chunks)
├── requirements.txt       # Dependencies (COMPLETE)
├── README.md              # Basic documentation
├── IMPLEMENTATION_PLAN.md # This file
├── templates/
│   ├── index.html         # Chat interface
│   ├── admin.html         # Admin panel
│   └── login.html         # Login page
└── uploads/               # Temporary upload folder
```

---

## Quick Start

### Option 1: Double-click (Windows)
```
Double-click START.bat
```

### Option 2: Command Line
```bash
cd C:\Food-E\Makerspace-RAG
python launcher.py
```

### What happens:
1. ✅ Checks Python dependencies
2. ✅ Checks if Ollama is running (starts it if not)
3. ✅ Verifies llama3 model is installed (pulls if needed)
4. ✅ Loads knowledge base
5. ✅ Starts Flask web server
6. ✅ Opens browser to http://localhost:5000/

---

## Features

### Chat Interface (/)
- Ask questions about 3D printing, laser cutting, electronics, etc.
- Three explanation levels: Beginner (1), Normal (2), Expert (3)
- TF-IDF search retrieves relevant knowledge chunks
- Ollama llama3 generates contextual responses

### Admin Panel (/admin)
- **Login:** admin / makerspace2024
- Upload documents (PDF, TXT, MD, JSON)
- Paste text directly
- Automatic chunking with smart deduplication
- View recent chunks
- Reload search index

### Health Check (/health)
- Check Ollama status and models
- Check knowledge base status
- Check search index status

---

## Implementation Status

### ✅ Phase 1: Dependencies & Configuration - COMPLETE
- [x] Updated requirements.txt with all dependencies
- [x] Environment variable support for credentials
- [x] Proper dependency checking in launcher

### ✅ Phase 2: Ollama Setup - COMPLETE  
- [x] Auto-detection of Ollama running state
- [x] Auto-start Ollama if not running
- [x] Model verification and auto-pull

### ✅ Phase 3: Unified Launcher - COMPLETE
- [x] launcher.py - Single entry point
- [x] START.bat - Windows double-click launcher
- [x] Auto browser opening
- [x] Graceful error handling

### ✅ Phase 4: Health Monitoring - COMPLETE
- [x] /health endpoint with full system status
- [x] /status endpoint for quick checks
- [x] Console logging for debugging

### ✅ Phase 5: Admin Panel - COMPLETE
- [x] File upload with PDF support
- [x] Direct text input
- [x] Smart chunking algorithm
- [x] Duplicate detection (exact + similarity)
- [x] Search index reload

### ✅ Phase 6: PDF Extraction Improvements - COMPLETE
- [x] PyMuPDF4LLM integration (best for RAG - extracts markdown)
- [x] pdfplumber fallback (good table extraction)
- [x] PyPDF2 last resort (basic extraction)
- [x] Improved chunking with header awareness
- [x] Chunk overlap for context continuity
- [x] Filters out header-only chunks

### ✅ Phase 7: UI & System Prompt Improvements - COMPLETE
- [x] Clean, modern chat interface design
- [x] Removed stats clutter from main page
- [x] Language selection (Auto/Norwegian/English)
- [x] Simplified level selection (Beginner/Normal/Expert)
- [x] Improved system prompt structure
- [x] Quick suggestion buttons
- [x] Smooth animations and loading states

### ✅ Phase 8: Branding & Theme - COMPLETE
- [x] Makerspace orange/yellow brand colors (#E5A124)
- [x] Lightbulb logo SVG in header
- [x] Dark/Light mode toggle with persistence
- [x] Cleaner button design (numbers for levels, text for languages)
- [x] Brand bar accent at bottom
- [x] Admin panel updated to match branding
- [x] Professional, consistent look across all pages

---

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | No | Chat interface |
| `/chat` | POST | No | Send message, get response |
| `/status` | GET | No | Quick status check |
| `/health` | GET | No | Detailed health check |
| `/login` | GET/POST | No | Admin login page |
| `/logout` | GET | Yes | Logout admin |
| `/admin` | GET | Yes | Admin panel |
| `/upload` | POST | Yes | Upload files |
| `/add-text` | POST | Yes | Add text directly |
| `/reload` | POST | Yes | Reload search index |
| `/recent` | GET | Yes | Get recent chunks |
| `/stats` | GET | Yes | Get vault statistics |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Ollama not found" | Install Ollama from https://ollama.ai |
| "Model not found" | Launcher auto-pulls models, or run: `ollama pull llama3` |
| "Connection refused" | Check Ollama is running: `ollama list` |
| Missing dependencies | Run: `pip install -r requirements.txt` |
| Port 5000 in use | Edit FLASK_PORT in launcher.py |
| Slow responses | First response is slowest; subsequent are faster |
| PDF only grabs headers | Install pymupdf4llm: `pip install pymupdf4llm` |
| Poor PDF table extraction | Install pdfplumber: `pip install pdfplumber` |

---

## Configuration

Edit `launcher.py` to change:
```python
OLLAMA_PORT = 11434      # Ollama API port
FLASK_PORT = 5000        # Web server port
REQUIRED_MODELS = ['llama3']  # Required Ollama models
```

Edit environment variables (optional):
```
SECRET_KEY=your-secret-key
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your-password
```

---

## Future Enhancements (Out of Scope)

- [ ] Hybrid search (TF-IDF + semantic embeddings)
- [ ] Chat history persistence  
- [ ] Multi-user support
- [ ] Docker containerization
- [ ] GPU acceleration
- [ ] Norwegian language model fine-tuning

---

## Phase 9: Query Classification & Context Awareness - IN PROGRESS

### Implemented
- [x] Query classification into 3 categories:
  - **FEILSOKING** - Troubleshooting (noe fungerer ikke)
  - **OPPLARING** - Training/how-to (hvordan gjør jeg X)
  - **VERKTOY_HMS** - Equipment info & safety rules
- [x] Tool detection (3D printer, laser, electronics, etc.)
- [x] Category-specific system prompt instructions
- [x] Context labeling based on category
- [x] Logging of detected category and tool in console
- [x] **Tool-filtered search** - Only return chunks relevant to detected tool
- [x] **Query expansion (NO→EN)** - Expand Norwegian queries with English synonyms for better TF-IDF matching
- [x] **Granular tool detection** - Separate detection for lodding, arduino, raspberry vs general elektronikk

### Tool Detection (priority order - specific first)
1. **lodding**: lodd, solder, tinn, flux, kolbe
2. **arduino**: arduino, uno, mega, nano, sketch
3. **raspberry**: raspberry, gpio, raspbian
4. **3d_printer**: prusa, filament, pla, nozzle, extruder, slicer
5. **laserkutter**: laser, gravering, epilog, kutte, fokus
6. **vinylkutter**: vinyl, cricut, sticker, folie
7. **tekstil**: sy, symaskin, stoff, broderi
8. **cnc**: cnc, fres, mill, router
9. **elektronikk**: krets, circuit, breadboard, pcb (catch-all)

### Query Expansion (Norwegian → English)
Expands Norwegian terms with English synonyms before TF-IDF search:
- `lodding` → `solder soldering how to solder beginner guide`
- `hvordan` → `how to guide tutorial`
- `fungerer ikke` → `not working problem error troubleshoot`
- `byggeplate` → `bed build plate adhesion`
- `løsner` → `warping adhesion detach lifting`

### Search Parameters
- **top_k**: 5 chunks max
- **max_chunk_chars**: 1200 per chunk
- **max_total_chars**: 4000 total context
- **tool_filter**: Only include chunks matching detected tool keywords

### Future Work
- [ ] Tag vault chunks with categories during upload
- [ ] Semantic search fallback for complex queries
- [ ] Multi-language vault support

---

*Document Created: 2025-06-02*  
*Last Updated: 2025-12-02 - Tool-filtered search, query expansion, granular tool detection*
