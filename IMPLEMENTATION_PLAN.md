# Makerspace-RAG Implementation Plan

## Project Overview

**Project:** Makerspace RAG System for Høgskolen i Østfold  
**Purpose:** AI-powered Q&A assistant for makerspace topics (3D printing, laser cutting, electronics, etc.)  
**Developer:** Patrick  
**Status:** ✅ ACTIVE DEVELOPMENT

---

## Recent Changes

### ✅ 2025-12-03: Fast PDF Import with Accept/Decline
**Problem:** PDF-ekstrahering tar for lang tid (30-60 sek) pga PyMuPDF4LLM markdown-konvertering
**Løsning:** Rask PyPDF2-ekstrahering → Forhåndsvisning → Valgfri AI-strukturering → Godkjenn/Avvis

Ny arkitektur:
```
PDF → PyPDF2 (rask) → Forhåndsvisning → [Valgfritt: AI strukturering] → Godkjenn/Avvis
                           ↓                      ↓
                    Redigér manuelt          LLM forbedrer
```

Backend endringer (`app.py`):
- `extract_pdf_fast()` - Ny funksjon som kun bruker PyPDF2 for rask ekstrahering
- `/extract-pdf` - Nytt endpoint for umiddelbar tekstekstrahering
- `/enhance-pdf` - Nytt endpoint for valgfri AI-strukturering
- Legacy `/summarize-pdf` oppdatert til å bruke rask ekstrahering

Frontend endringer (`templates/admin.html`):
- Ny "PDF Import (Rask Ekstrahering)" seksjon erstatter "Smart PDF Import"
- To-stegs arbeidsflyt:
  1. Rå tekst forhåndsvisning med redigeringsmulighet
  2. Valgfri AI-strukturering med egen forhåndsvisning
- Knapper: "Strukturer med AI", "Godkjenn rå tekst", "Tilbake til rå", "Avbryt"
- Viser ekstraksjonstid, sidetall, tegntall

Forbedringer:
- Ekstraksjonstid: ~2-5 sek (ned fra 30-60 sek)
- Brukeren ser resultatet umiddelbart
- Valgfri AI-forbedring (ikke påkrevd)
- Mulighet til å redigere før godkjenning
- Kan avvise og prøve på nytt

### ✅ 2025-12-03: Clickable Links in Chat Responses
**Problem:** RAG skal integreres på en webside og må kunne vise klikkbare lenker
**Løsning:** Frontend link-parsing + backend test-case

Frontend endringer (`templates/index.html`):
- `formatMessage()` oppdatert til å håndtere:
  - Markdown links: `[tekst](url)` → klikkbar lenke med tekst
  - Rå URLs: `https://example.com` → klikkbar lenke
- Ny CSS-klasse `.chat-link` for lenke-styling

Backend endringer (`app.py`):
- Easter egg test: "green apples" eller "grønne epler" trigger test-respons med lenke til vg.no
- Bypass LLM for å teste link-funksjonalitet isolert

Test:
```
Bruker: "Green apples are good"
Bot: Ja, grønne epler er kjempegode! 🍏
     [Les mer på VG](https://www.vg.no)
     https://www.vg.no
```

Neste steg:
- Integrere lenker fra knowledge base (ressurser.json)
- La LLM inkludere relevante lenker i svar

### ✅ 2025-12-03: Incremental Conversation Compression
**Problem:** Komprimering av 44 meldinger på én gang gir lang ventetid
**Løsning:** Inkrementell komprimering hver 6. melding

Konfigurasjon:
- `INCREMENTAL_COMPRESS_EVERY = 6` - Komprimer hver 6. melding (3 utvekslinger)
- `RECENT_MESSAGES_KEEP = 10` - Behold alltid siste 10 meldinger i full tekst

Flyt:
```
Melding 1-6:   [full] [full] [full] [full] [full] [full]
Melding 7:    Komprimer 1-6 → summary, behold 7-10 full
Melding 13:   Komprimer 7-12 → oppdater summary, behold 11-16 full
...osv
```

Filer endret:
- `app.py`: 
  - `summarize_messages()` - Inkrementell oppsummering med eksisterende kontekst
  - `ask_llm()` returnerer nå tuple: (response, updated_summary)
  - `/chat` endpoint mottar og returnerer summary
- `templates/index.html`:
  - `conversationSummary` variabel
  - Sender summary med hver request
  - Oppdaterer summary fra response

Modeller:
- Hovedsvar: `llama3`
- Komprimering: `llama3.2:1b` (rask, 1.3GB)

### ✅ 2025-12-03: Extended Conversation Memory
**Problem:** Samtalehistorikk begrenset til 12 meldinger (6 utvekslinger)
**Løsning:** Økt til 40 meldinger (20 utvekslinger)

Endringer:
- `templates/index.html`: `conversationHistory` limit 12 → 40
- `app.py`: `history_limit` slice [-12:] → [-40:]

Nå kan chatbotten huske ~20 meldingsutvekslinger i en samtale.

---

## Current Architecture (v2)

```
┌─────────────────────────────────────────────────────────────────┐
│                    MAKERSPACE RAG SYSTEM v2                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            STRUCTURED KNOWLEDGE (JSON)                   │   │
│  │  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌───────────┐    │   │
│  │  │ utstyr   │ │ regler   │ │  rom   │ │ ressurser │    │   │
│  │  │  .json   │ │  .json   │ │ .json  │ │   .json   │    │   │
│  │  └──────────┘ └──────────┘ └────────┘ └───────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌───────────────────┐   │
│  │   Query     │───▶│  Classifier │───▶│  Context Builder  │   │
│  │  (Bruker)   │    │  + Tool Det │    │  JSON + TF-IDF    │   │
│  └─────────────┘    └──────┬──────┘    └─────────┬─────────┘   │
│                            │                      │             │
│                            ▼                      ▼             │
│                     ┌─────────────┐        ┌─────────────┐     │
│                     │   Ollama    │        │  vault.txt  │     │
│                     │  (llama3)   │◀───────│  (cleaned)  │     │
│                     └─────────────┘        └─────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
C:\Food-E\Makerspace-RAG\
├── START.bat              # Double-click to launch (Windows)
├── launcher.py            # Unified launcher script
├── app.py                 # Main Flask web application
├── vault.txt              # Document chunks
├── requirements.txt       # Dependencies
├── IMPLEMENTATION_PLAN.md # This file
│
├── knowledge/             # Structured knowledge base
│   ├── utstyr.json        # Equipment inventory
│   ├── regler.json        # HMS/Safety rules  
│   ├── rom.json           # Room information
│   └── ressurser.json     # External learning resources
│
├── data/                  # 🆕 NEW: Persistent data storage
│   └── conversations.db   # SQLite database for chat history
│
├── templates/
│   ├── index.html         # Chat interface
│   ├── admin.html         # Admin panel
│   └── login.html         # Login page
│
├── static/
│   └── makerspace-logo.png
│
└── uploads/               # Document uploads
```

---

## Implementation Status

### ✅ Phase 1-9: COMPLETE
- Core RAG functionality
- Web interface with JSON knowledge
- Admin panel
- Query classification, tool detection, query expansion
- Smart PDF import
- Tilgangsnivåer for utstyr

### 🔄 Phase 10: Conversation Memory - IN PROGRESS

#### Problem Statement
Nåværende system har samtalehistorikk kun i browser-minnet:
- Forsvinner ved refresh/lukking
- Ingen persistens mellom sesjoner
- Begrenset til 12 meldinger
- Ingen mulighet for å se/gjenoppta tidligere samtaler

#### Solution: Session-Based Persistent Memory

**Arkitektur:**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Browser       │────▶│   Flask API     │────▶│   SQLite DB     │
│   (session_id)  │◀────│   /chat         │◀────│   conversations │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

#### Implementation Checklist

##### Backend (app.py)
- [ ] **10.1 Database Setup**
  - [ ] Create `data/` directory
  - [ ] Initialize SQLite database with schema:
    ```sql
    CREATE TABLE conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        title TEXT
    );
    
    CREATE TABLE messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL,
        role TEXT NOT NULL,  -- 'user' or 'assistant'
        content TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
    );
    
    CREATE INDEX idx_session ON conversations(session_id);
    CREATE INDEX idx_conv_id ON messages(conversation_id);
    ```

- [ ] **10.2 Database Helper Functions**
  - [ ] `init_db()` - Create tables if not exist
  - [ ] `create_conversation(session_id)` - Start new conversation
  - [ ] `add_message(conv_id, role, content)` - Save message
  - [ ] `get_conversation_history(conv_id, limit=20)` - Retrieve messages
  - [ ] `get_recent_conversations(session_id, limit=10)` - List conversations
  - [ ] `get_conversation_by_id(conv_id)` - Get specific conversation
  - [ ] `auto_generate_title(conv_id)` - Generate title from first message

- [ ] **10.3 API Endpoints**
  - [ ] `POST /chat` - Modify to save messages and return conv_id
  - [ ] `GET /conversations` - List user's recent conversations
  - [ ] `GET /conversations/<id>` - Get full conversation
  - [ ] `POST /conversations/new` - Start fresh conversation
  - [ ] `DELETE /conversations/<id>` - Delete conversation (optional)

- [ ] **10.4 Session Management**
  - [ ] Generate UUID session_id for anonymous users
  - [ ] Store session_id in cookie (httponly, 30 days expiry)
  - [ ] Pass session_id with all chat requests

##### Frontend (index.html)
- [ ] **10.5 Session Handling**
  - [ ] Check for existing session_id in cookie on load
  - [ ] Generate new session_id if none exists
  - [ ] Send session_id with all API requests

- [ ] **10.6 Conversation UI**
  - [ ] Add sidebar/drawer for conversation history
  - [ ] "Ny samtale" button creates new conversation
  - [ ] Click on previous conversation to load it
  - [ ] Show conversation title (auto-generated from first message)
  - [ ] Visual indicator for active conversation

- [ ] **10.7 Message Persistence**
  - [ ] On page load: fetch current conversation or start new
  - [ ] Display previous messages from database
  - [ ] Auto-scroll to bottom on load
  - [ ] Save messages immediately on send/receive

##### Configuration
- [ ] **10.8 Memory Settings**
  - [ ] `CONVERSATION_HISTORY_LIMIT = 20` - Messages sent to LLM
  - [ ] `CONVERSATIONS_PER_USER = 50` - Max stored per session
  - [ ] `MESSAGE_RETENTION_DAYS = 30` - Auto-cleanup old conversations
  - [ ] Add cleanup cron job / background task

#### Database Schema Details

```python
# In app.py - new section after imports

import sqlite3
import uuid
from pathlib import Path

DATABASE_PATH = Path('data/conversations.db')

def get_db():
    """Get database connection."""
    DATABASE_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables."""
    conn = get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            title TEXT
        );
        
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
        
        CREATE INDEX IF NOT EXISTS idx_session ON conversations(session_id);
        CREATE INDEX IF NOT EXISTS idx_conv_id ON messages(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_updated ON conversations(updated_at DESC);
    ''')
    conn.commit()
    conn.close()
    print("  Database initialized")
```

#### API Response Format

```python
# POST /chat response
{
    "response": "AI svar her...",
    "conversation_id": 42,
    "message_count": 5
}

# GET /conversations response
{
    "conversations": [
        {
            "id": 42,
            "title": "3D-printing med PLA",
            "created_at": "2025-12-03T10:30:00",
            "updated_at": "2025-12-03T10:45:00",
            "message_count": 8
        },
        ...
    ]
}

# GET /conversations/<id> response
{
    "id": 42,
    "title": "3D-printing med PLA",
    "messages": [
        {"role": "user", "content": "Hvordan...", "timestamp": "..."},
        {"role": "assistant", "content": "Du kan...", "timestamp": "..."}
    ]
}
```

#### Frontend Storage Strategy

```javascript
// Session management
function getOrCreateSession() {
    let sessionId = localStorage.getItem('makerspace_session');
    if (!sessionId) {
        sessionId = crypto.randomUUID();
        localStorage.setItem('makerspace_session', sessionId);
    }
    return sessionId;
}

// Current conversation tracking
let currentConversationId = null;
let sessionId = getOrCreateSession();

// Load conversation on startup
async function loadCurrentConversation() {
    const savedConvId = localStorage.getItem('current_conversation');
    if (savedConvId) {
        await loadConversation(savedConvId);
    }
}
```

#### UI Mockup - Conversation Sidebar

```
┌─────────────────────────────────────────────────────────────┐
│ [Logo] MAKERSPACE                    [🔄 Ny] [☀️] [⚙️]     │
├──────────────┬──────────────────────────────────────────────┤
│ SAMTALER     │                                              │
│              │     Hva skal vi lage i dag?                  │
│ ▶ 3D-print.. │                                              │
│   Laser mat..│     [3D-Printing] [Laser] [Elektronikk]      │
│   Arduino pr │                                              │
│   Filament.. │                                              │
│              │                                              │
│              │                                              │
│              │                                              │
├──────────────┴──────────────────────────────────────────────┤
│ [Nivå: 1 2 3] [    Skriv spørsmål...    ] [Send] [NO/EN]   │
└─────────────────────────────────────────────────────────────┘
```

#### Priority Order
1. **Backend first** - Database + API (10.1-10.4)
2. **Basic persistence** - Save/load messages (10.5, 10.7)
3. **UI enhancement** - Conversation list (10.6)
4. **Cleanup** - Auto-delete old conversations (10.8)

#### Estimated Effort
- Backend: 2-3 timer
- Frontend basic: 1-2 timer
- Frontend UI: 2-3 timer
- Testing: 1 time
- **Total: ~8 timer**

---

## Previous Phases (Completed)

### ✅ Phase 9: Knowledge Restructuring
- Created structured JSON knowledge base
- Rebuilt vault.txt with educational content
- Integrated JSON with context builder
- Smart query routing (inventory vs other)

### ✅ Phase 8: Smart PDF Import
- AI-assisted PDF summarization
- Admin review and approval workflow

### ✅ Phase 7: Equipment Access Levels
- Added access_level field to utstyr.json
- 5 tilgangsnivåer: course_makerspace, course_fablab, certification_required, request_required, staff_only

---

## LLM Configuration

```python
model='llama3'
options={'temperature': 0.7, 'num_predict': 500}
```

---

## Context Building Strategy

When a user asks a question:

1. **Classify query** → FEILSOKING | OPPLARING | VERKTOY_HMS | GENERELL
2. **Detect tool** → 3d_printer | laserkutter | cnc | lodding | etc.
3. **Load conversation history** → Last 20 messages from database (NEW!)
4. **Build context**:
   - If tool detected → Include equipment JSON entry
   - If HMS question → Include relevant rules
   - Add TF-IDF search results from vault.txt
5. **Send to LLM** with conversation history + context

---

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | No | Chat interface |
| `/chat` | POST | No | Send message, get response |
| `/conversations` | GET | No | List recent conversations (NEW) |
| `/conversations/<id>` | GET | No | Get conversation details (NEW) |
| `/conversations/new` | POST | No | Start new conversation (NEW) |
| `/status` | GET | No | Quick status check |
| `/health` | GET | No | Detailed health check |
| `/equipment` | GET | No | List all equipment |
| `/admin` | GET | Yes | Admin panel |
| `/upload` | POST | Yes | Upload files |
| `/reload` | POST | Yes | Reload search index |

---

*Document Created: 2025-06-02*  
*Last Updated: 2025-12-03 - Phase 10: Conversation Memory*
