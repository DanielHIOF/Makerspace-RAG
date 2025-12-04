# Makerspace-RAG Implementation Plan

## Project Overview

**Project:** Makerspace RAG System for Høgskolen i Østfold  
**Purpose:** AI-powered Q&A assistant for makerspace topics (3D printing, laser cutting, electronics, etc.)  
**Developer:** Patrick  
**Status:** ✅ ACTIVE DEVELOPMENT

---

## Recent Changes

### ✅ 2025-12-04: Simplified PDF Extraction (OCR-only)
**Problem:** Kompleks PDF-pipeline med 4 metoder (pymupdf4llm → pdfplumber → PyPDF2 → OCR) var overkill
**Løsning:** Fjernet alle andre metoder, kun OCR (EasyOCR) beholdt

Før (172 linjer):
```
1. Detect if image-based
2. If image → OCR
3. Else → pymupdf4llm → pdfplumber → PyPDF2 → OCR fallback
```

Etter (28 linjer):
```
1. OCR all PDFs with EasyOCR
```

Fordeler:
- Konsistent output for alle PDF-typer
- Enklere kode, lettere å vedlikeholde
- Fungerer like bra for tekst-PDFer og skannede dokumenter

Fjernet:
- `count_meaningful_chars()` funksjon
- pymupdf4llm extraction
- pdfplumber extraction  
- PyPDF2 extraction
- Image-based detection logic

Beholdt:
- `extract_pdf_fast()` - Entry point (kaller OCR)
- `extract_pdf_ocr()` - EasyOCR med norsk/engelsk støtte

### ✅ 2025-12-04: Norwegian Default + Natural Component Responses
**Problem 1:** LLM svarte på engelsk selv når bruker stilte spørsmål på norsk
**Problem 2:** LLM hallusinerte lokasjoner som "near the Weller WE1010"
**Problem 3:** LLM byttet ut komponentnavn - ga "termistorer" når bruker spurte om "motstander"
**Problem 4:** LLM kopierte "@"-formatet fra konteksten i stedet for å skrive naturlig
**Problem 5:** Inkonsistent formatering med * kulepunkt, **bold**, _italic_ osv.

**Løsning:**
1. **Norsk som default språk** - fjernet "auto"-modus
   
2. **Forbedret search_components():**
   - Ekstraherer spesifikke søketermer (motstand, resistor, etc.)
   - Viser kategori i output for bedre kontekst
   
3. **Naturlig språk-instruksjoner** i system prompt:
   - GODT eksempel: "Vi har motstander på Komponentvegg, blant annet 10Ω, 15Ω og 100Ω."
   - DÅRLIG eksempel: "10Ω @ Komponentvegg, 15Ω @ Komponentvegg..."
   
4. **Formaterings-regler** i system prompt:
   - Bruk "-" for kulepunkt (ikke *)
   - Forbud mot **bold** og *italic* (rendres ikke)
   - Nummererte lister OK når rekkefølge betyr noe
   - Links OK: [tekst](url)

**Endrede filer:**
- `app.py`: 
  - `detect_language()` - default 'norwegian'
  - `search_components()` - smartere term-matching
  - System prompt med FORMATERING-seksjon

### ✅ 2025-12-04: Separate Components System (components.json)
**Problem:** Komponenter og utstyr er fundamentalt forskjellige - komponenter er deler du bruker i prosjekter, utstyr er maskiner du bruker
**Løsning:** Egen components.json fil med dedikert XLSX-import og søkefunksjonalitet

Ny filstruktur:
```
knowledge/
├── utstyr.json      # Maskiner, verktøy (3D-printere, laserkuttere, etc.)
├── components.json  # Komponenter, deler (motstander, Arduino, skruer, etc.)
├── regler.json
├── rom.json
└── ressurser.json
```

components.json kategorier:
- electronics: Motstander, kondensatorer, IC-er
- sensors: Temperatur, bevegelse, lys
- modules: Arduino, ESP32, displays
- mechanical: Motorer, tannhjul, lagre
- fasteners: Skruer, muttere, bolter
- consumables: Ledning, loddetinn, tape
- other: Annet

Nye funksjoner i app.py:
- `search_components(query)` - Søk etter komponenter på navn/lokasjon/keywords
- `get_all_components_summary()` - Oversikt over alle komponenter
- `is_component_query(query)` - Detekterer komponent-spørsmål
- `check_component_duplicates()` - Duplikatsjekk for XLSX-import

Chat-integrasjon:
- Automatisk deteksjon av komponent-spørsmål
- Søker i components.json før TF-IDF vault-søk
- Viser matchende komponenter med lokasjon og antall

### ✅ 2025-12-04: XLSX Import for Component Lists (openpyxl)
**Problem:** Makerspace har komponentlister i Excel som må importeres
**Løsning:** Dedikert XLSX-import med automatisk duplikatfiltrering

Ny funksjonalitet:
- **Upload XLSX** → Parser rader som komponentoppføringer
- **Auto column mapping** - Gjenkjenner kolonner: navn, lokasjon, antall, kategori, ID
- **Duplikatsjekk** - Filtrerer ut eksisterende (samme ID, eller samme navn+lokasjon)
- **Preview med checkboxes** - Velg hvilke elementer som skal legges til
- **Kategori-valg** - electronics, sensors, modules, mechanical, fasteners, consumables

Backend endpoints:
- `POST /extract-xlsx` - Parser fil og returnerer preview med duplikatinfo
- `POST /approve-xlsx` - Legger til valgte elementer i components.json

Column mappings (norsk/engelsk):
- id: id, product_id, produktid, varenr, sku
- name: name, navn, product, produkt, description, component
- location: location, lokasjon, sted, rom, shelf, hylle, drawer, skuff
- quantity: quantity, antall, qty, count, stk, pcs
- category: category, kategori, type, gruppe
- notes: notes, notater, kommentar, remarks

Duplikatlogikk:
1. Eksakt match: Samme ID + navn + lokasjon → Skip
2. Samme ID → Skip (uavhengig av navn/lokasjon)
3. Samme navn + lokasjon → Skip (uavhengig av ID)

Nye avhengigheter:
- openpyxl>=3.1.0

### ✅ 2025-12-04: Smart OCR-First Detection for Image PDFs
**Problem:** OCR var sist i pipeline, 300-char threshold lot garbage passere
**Løsning:** Detekter image-baserte PDFer FØR tekst-ekstrahering

Ny deteksjonslogikk:
```
chars_per_page < 200 AND images_per_page > 1 → IMAGE-BASED → OCR first
chars_per_page < 100 → IMAGE-BASED → OCR first
```

Console output:
```
[PDF-DETECT] 2 pages, 4 images, 62 text chars
[PDF-DETECT] ~31 chars/page, ~2.0 images/page → IMAGE-BASED
[PDF-OCR] Detected image-based PDF. Using OCR first...
```

Også økt threshold fra 300 til 500 meaningful chars.

### ✅ 2025-12-04: OCR Fallback for Image-Based PDFs (easyocr)
**Problem:** HiØF HMS-dokumenter har tekst bakt inn som bilder, ikke ekte tekst
**Diagnostikk viste:** "Images on page: 72, Simple text length: 31 chars" - innholdet er bilder!

Løsning: Automatisk OCR-fallback når vanlig ekstrahering feiler

Extraction pipeline (i rekkefølge):
1. **pymupdf4llm** - Beste for komplekse layouts
2. **pdfplumber** - God for enkle tabeller
3. **PyPDF2** - Grunnleggende ekstrahering
4. **easyocr** - OCR for skannede/bildebaserte PDFer

OCR-implementasjon (`extract_pdf_ocr()`):
- Bruker easyocr (ren Python, ingen Tesseract nødvendig)
- Språkstøtte: Norsk ('no') og Engelsk ('en')
- Konverterer PDF-sider til bilder via PyMuPDF (2x zoom for kvalitet)
- Filtrerer lav-konfidens resultater (< 0.3)
- CPU-modus for kompatibilitet

Auto-trigger: Hvis < 200 tegn reelt innhold etter de 3 første metodene

Nye avhengigheter (requirements.txt):
- easyocr>=1.7.0
- pdf2image>=1.16.0

MERK: Første kjøring laster ned ~100MB OCR-modeller

### ✅ 2025-12-04: Quality-First PDF Extraction (pymupdf4llm)
**Problem:** Both PyPDF2 and pdfplumber fail to extract table CONTENT - they see structure but miss the actual text
**Eksempel:** HMS-dokument tabell viste bare headers, alle 1.1, 1.2, 1.3 rader forsvant

Root cause: PDF har visuell tabell (linjer) men tekst er ikke strukturelt inne i celler

Løsning: Prioriter pymupdf4llm som er designet for RAG og håndterer komplekse layouts

Extraction priority:
1. **pymupdf4llm** - Konverterer til markdown, håndterer tabeller, best kvalitet (tregere)
2. **pdfplumber** - Fallback for enklere PDFer med tabeller
3. **PyPDF2** - Siste utvei, raskest men ofte mister innhold

Trade-off: Aksepterer tregere ekstrahering (~10-30s) for faktisk fungerende innhold

### ✅ 2025-12-04: Smart PDF Table Extraction (pdfplumber)
**Problem:** PyPDF2 butchers tables - only extracts headers, loses all cell content
**Eksempel:** HMS-dokument med tabell ble til "1 Verneutstyr og sikkerhet" uten innhold

Før (PyPDF2):
```
Nr Handling Ansvar Kommentarer
1 Verneutstyr og sikkerhet
2 Oppstart og bruk
```

Etter (pdfplumber):
```
Nr | Handling | Ansvar | Kommentarer
1 | Verneutstyr og sikkerhet | |
1.1 | Maskinen må brukes av personer som har godkjent kompetanse... | Bruker | Ta kontakt med labingeniøren...
1.2 | Bruk av verneutstyr må vurderes av bruker... | Bruker | Manglende bruk kan medføre bortvisning
```

Backend endringer (`app.py`):
- `extract_pdf_fast()` fullstendig omskrevet:
  - Primær: pdfplumber med `extract_tables()` for tabelldeteksjon
  - Konverterer tabeller til lesbar tekst med `|` separatorer
  - Kombinerer tabelltekst med vanlig tekst fra `extract_text()`
  - Fallback: PyPDF2 hvis pdfplumber feiler eller gir for lite tekst
  - Smart logging: viser hvilken metode som brukes

Avhengigheter:
- pdfplumber>=0.10.0 (allerede i requirements.txt)

### ✅ 2025-12-04: Category-Aware Smart Import (llama3 8B)
**Problem:** All content dumps into vault.txt. No way to create structured JSON entries for utstyr, regler, rom, ressurser.
**Løsning:** Category selection + template-aware AI prompts + smart file routing

Ny arkitektur:
```
PDF → Extract → [Context] + [Category] → AI (llama3) → Template-specific output
                                ↓
                     ┌─────────────────────────────────────┐
                     │  vault   → vault.txt (tekst)        │
                     │  utstyr  → knowledge/utstyr.json    │
                     │  regler  → knowledge/regler.json    │
                     │  rom     → knowledge/rom.json       │
                     │  ressurser → knowledge/ressurser.json│
                     └─────────────────────────────────────┘
```

Backend endringer (`app.py`):
- `CATEGORY_TEMPLATES` - Nye promptmaler for hver kategori:
  - `vault`: Strukturert tekst med --- NIVÅ: Tittel --- format
  - `utstyr`: JSON med id, name, location, access_level, difficulty, materials, keywords
  - `regler`: JSON med id, priority, rule_no/en, applies_to
  - `rom`: JSON med id, name_no/en, building, floor, features, access
  - `ressurser`: JSON med title, url, language, level, description_no/en
- `/enhance-pdf` oppdatert:
  - Mottar `category` parameter (vault, utstyr, regler, rom, ressurser)
  - Bruker `llama3` (8B) i stedet for `llama3.2:1b` for bedre kvalitet
  - Kategori-spesifikke prompts med eksempel-output
  - Returnerer `output_type` (text/json) og `target_file`
- `/approve-summary` oppdatert:
  - Mottar `category` parameter
  - For `vault`: Appender til vault.txt som før
  - For JSON-kategorier: Parser JSON og merger med eksisterende fil
  - Smart kategorisering basert på keywords
  - Oppdaterer `last_updated` metadata
  - Kaller `load_json_knowledge()` for umiddelbar oppdatering

Frontend endringer (`templates/admin.html`):
- Ny kategori-dropdown: "Hvor skal innholdet lagres?"
  - 📚 Generell kunnskap (vault.txt)
  - 🔧 Utstyr (utstyr.json)
  - ⚠️ Regler (regler.json)
  - 🏠 Rom (rom.json)
  - 🔗 Ressurser (ressurser.json)
- `currentCategory` variabel for state management
- Kategori sendes med både `/enhance-pdf` og `/approve-summary`
- Viser målfil i enhanceInfo og success-melding
- Reset av kategori etter godkjenning

Prompts per kategori:
- Alle prompts har detaljerte eksempler på forventet JSON-struktur
- Inkluderer alle gyldige verdier (access_levels, priorities, etc.)
- Instruerer AI om å kun returnere gyldig JSON, ingen forklaring
- Post-processing fjerner markdown code blocks fra JSON-output

Forbedringer:
- Strukturert data direkte inn i kunnskapsbasen
- AI kan nå slå opp spesifikke felt (tilgangsnivå, lokasjon, etc.)
- Bedre kvalitet med llama3 8B vs 1b
- JSON-validering før lagring
- Automatisk merging med eksisterende data

### ✅ 2025-12-04: Fast PDF Import with Accept/Decline
**Problem:** PDF-ekstrahering tar for lang tid (30-60 sek) pga PyMuPDF4LLM markdown-konvertering
**Løsning:** Rask PyPDF2-ekstrahering → Forhåndsvisning → Valgfri AI-strukturering → Godkjenn/Avvis

Ny arkitektur:
```
PDF → PyPDF2 (rask) → Forhåndsvisning → [Beskriv dokument] → AI strukturering → Godkjenn/Avvis
                           ↓                      ↓
                    Redigér manuelt        Kontekst-bevisst prompt
```

Backend endringer (`app.py`):
- `extract_pdf_fast()` - Ny funksjon som kun bruker PyPDF2 for rask ekstrahering
- `/extract-pdf` - Nytt endpoint for umiddelbar tekstekstrahering
- `/enhance-pdf` - Nytt endpoint med kontekst-bevisst AI-strukturering
  - Mottar `text` + `context` (hva dokumentet handler om)
  - Bruker kontekst i prompt for bedre output
- Legacy `/summarize-pdf` oppdatert til å bruke rask ekstrahering

Frontend endringer (`templates/admin.html`):
- Ny "PDF Import (Rask Ekstrahering)" seksjon erstatter "Smart PDF Import"
- Kontekst-input: "Hva handler dokumentet om?" med hurtigvalg-knapper
  - 📖 Manual, ⚠️ HMS, 🔧 Feilsøking, ⚙️ Innstillinger
- To-stegs arbeidsflyt:
  1. Rå tekst forhåndsvisning + kontekstbeskrivelse
  2. AI-strukturering med kontekst-bevisst prompt
- Knapper: "Strukturer med AI", "Godkjenn rå tekst", "Tilbake til rå", "Avbryt"
- Validering: Krever kontekst før AI-behandling

Forbedringer:
- Ekstraksjonstid: ~2-5 sek (ned fra 30-60 sek)
- AI-strukturering: ~10-20 sek (ned fra 60-120 sek)
  - Bruker `llama3.2:1b` i stedet for `llama3` (5-10x raskere)
  - Redusert max_chars: 6000 (ned fra 12000)
  - Redusert num_predict: 1200 (ned fra 2500)
  - Kortere, mer fokusert prompt
- Live timer i UI: "AI jobber... 5s" → brukeren ser at det fungerer
- Kontekst gir LLM retning → mye bedre output
- Brukeren ser resultatet umiddelbart
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
│  │  │ utstyr   │ │ compo-   │ │ regler │ │ ressurser │    │   │
│  │  │  .json   │ │ nents    │ │ .json  │ │   .json   │    │   │
│  │  │(machines)│ │  .json   │ │ (HMS)  │ │  (links)  │    │   │
│  │  └──────────┘ └──────────┘ └────────┘ └───────────┘    │   │
│  │                    ┌────────┐                           │   │
│  │                    │  rom   │                           │   │
│  │                    │ .json  │                           │   │
│  │                    └────────┘                           │   │
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
│   ├── utstyr.json        # Equipment inventory (machines, tools)
│   ├── components.json    # Components inventory (resistors, Arduino, etc.)
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

1. **Detect language** → Norwegian (default) | English (if `/english` or `/en`)
2. **Detect level** → Nybegynner | Normal | Ekspert
3. **Classify query** → FEILSOKING | OPPLARING | VERKTOY_HMS | GENERELL
4. **Check if inventory query** → "hva har dere", "which equipment", etc.
5. **Check if component query** → "motstander", "Arduino", "sensorer", etc.
6. **Detect tool** → 3d_printer | laserkutter | cnc | lodding | etc.
7. **Build context**:
   - If component query → `search_components()` fra components.json
   - If inventory query → Equipment list fra utstyr.json (kun JSON, ikke vault)
   - If tool detected → Include equipment JSON entry + HMS regler
   - Add TF-IDF search results from vault.txt
8. **Load conversation history** → Last 10 messages + compressed summary
9. **Build system prompt** med:
   - Base role (Makerspace veileder)
   - Kontekst fra JSON/vault
   - Ferdighetsnivå instruksjoner
   - Språk instruksjoner
   - Formaterings-regler (- for kulepunkt, ingen bold/italic)
   - Komponent-spesifikke instruksjoner (naturlig språk, bruk lokasjoner)
10. **Send to LLM** (llama3) med conversation history + context

---

## API Endpoints

### Public Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat interface |
| `/chat` | POST | Send message, get AI response |
| `/status` | GET | Quick status check (index loaded?) |
| `/health` | GET | Detailed health check (Ollama, vault, JSON) |

### Authentication
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/login` | GET/POST | Admin login page |
| `/logout` | GET | Logout admin |

### Admin - Content Management (Protected)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin` | GET | Admin panel dashboard |
| `/upload` | POST | Upload files (TXT, PDF, JSON, MD) |
| `/add-text` | POST | Add raw text to vault |
| `/stats` | GET | Vault statistics |
| `/recent` | GET | Recent vault chunks |
| `/reload` | POST | Reload TF-IDF index |

### Admin - PDF Import (Protected)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/extract-pdf` | POST | Fast PDF text extraction (PyPDF2/pdfplumber/OCR) |
| `/enhance-pdf` | POST | AI-strukturering med kategori og kontekst |
| `/approve-summary` | POST | Godkjenn og lagre til vault/JSON |
| `/summarize-pdf` | POST | Legacy endpoint (backwards compat) |

### Admin - Component Import (Protected)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/extract-xlsx` | POST | Parse Excel-fil, returner preview med duplikatsjekk |
| `/approve-xlsx` | POST | Godkjenn valgte komponenter til components.json |

---

## Configuration

### Environment Variables
```bash
SECRET_KEY=your-secret-key          # Flask session key (default: makerspace-secret-key-change-in-production)
ADMIN_USERNAME=admin                 # Admin login username (default: admin)
ADMIN_PASSWORD=your-password         # Admin login password (default: makerspace2024)
```

### Application Constants (app.py)
```python
# File handling
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'json', 'md', 'csv', 'html', 'htm', 'xlsx'}
VAULT_FILE = 'vault.txt'
CHUNK_SIZE = 1000

# Conversation memory
INCREMENTAL_COMPRESS_EVERY = 6      # Compress every 6 messages (3 exchanges)
RECENT_MESSAGES_KEEP = 10           # Always keep last 10 messages in full

# LLM settings
MODEL_MAIN = 'llama3'               # Primary model for responses
MODEL_COMPRESS = 'llama3.2:1b'      # Fast model for compression
TEMPERATURE = 0.7
MAX_TOKENS = 500
```

### Knowledge Files (knowledge/)
| File | Purpose | Format |
|------|---------|--------|
| `utstyr.json` | Maskiner og verktøy | JSON med categories → items |
| `components.json` | Elektroniske komponenter | JSON med categories → components |
| `regler.json` | HMS-regler | JSON med general_rules + tool_rules |
| `rom.json` | Rom-informasjon | JSON med rooms array |
| `ressurser.json` | Eksterne lenker | JSON med resources array |

---

## Known Issues / Limitations

### Current Bugs
- **None tracked** - Opprett GitHub issues for nye bugs

### Known Limitations
1. **Ingen persistent samtalehistorikk** - Forsvinner ved browser refresh (Phase 10 planlagt)
2. **CPU-only** - Ollama kjører på CPU, trege svar (2-5 min på svak hardware)
3. **Maks 16MB uploads** - Kan justeres i config
4. **Norsk OCR** - EasyOCR norsk modell er ikke perfekt på håndskrift
5. **Ingen bruker-auth** - Chat er åpen, kun admin er beskyttet

### Edge Cases
- Tomme PDF-er gir ingen feilmelding (returnerer tom tekst)
- Veldig lange samtaler kan overstige context window
- Excel-filer med komplekse formler importerer kun verdier

---

## Testing

### Manual Testing Checklist

#### Chat Interface
- [ ] Norsk spørsmål → Norsk svar
- [ ] `/english` prefix → English response
- [ ] Komponent-spørsmål → Riktig lokasjon fra JSON
- [ ] Utstyr-spørsmål → Liste fra utstyr.json
- [ ] HMS-spørsmål → Sikkerhetsinfo inkludert
- [ ] Lang samtale → Komprimering fungerer

#### Admin Panel
- [ ] Login med riktig passord → OK
- [ ] Login med feil passord → Avvist
- [ ] PDF upload → Tekst ekstrahert
- [ ] XLSX upload → Komponenter vist med duplikatsjekk
- [ ] Godkjenn til vault → Tekst lagt til
- [ ] Godkjenn til JSON → Strukturert data lagret
- [ ] Reload index → Ny data søkbar

#### API Health
```bash
# Quick status
curl http://localhost:5000/status

# Detailed health
curl http://localhost:5000/health

# Test chat
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Har dere 3D-printere?"}'
```

### Test Queries
```
# Komponenter
"Har dere motstander?"          → Skal liste fra Komponentvegg
"Hvor finner jeg Arduino?"      → Skal gi lokasjon fra components.json

# Utstyr
"Hvilke 3D-printere har dere?"  → Liste fra utstyr.json
"Hva slags laserkutter har dere?" → Epilog/andre fra JSON

# HMS
"Er det farlig å laserkutte PVC?" → ADVARSEL om klorgass
"Hva må jeg ha på meg ved lodding?" → Verneutstyr info

# Nivåer
"/nybegynner Hva er 3D-printing?" → Enkel forklaring
"/ekspert Forklar FDM vs SLA"     → Teknisk dybde
```

---

## Deployment

### Development (Local)
```bash
# 1. Start Ollama
ollama serve

# 2. Pull required models
ollama pull llama3
ollama pull llama3.2:1b

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python app.py
# eller
python launcher.py
# eller double-click START.bat (Windows)
```

### Production Recommendations
1. **Sett environment variables**:
   ```bash
   export SECRET_KEY="$(openssl rand -hex 32)"
   export ADMIN_PASSWORD="strong-password-here"
   ```

2. **Bruk gunicorn** (ikke Flask dev server):
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Reverse proxy** (nginx):
   ```nginx
   location / {
       proxy_pass http://127.0.0.1:5000;
       proxy_set_header Host $host;
       proxy_set_header X-Real-IP $remote_addr;
   }
   ```

4. **GPU for Ollama** (anbefalt):
   - NVIDIA 40-series: ~10-30 sek responstid
   - CPU-only: ~2-5 min responstid

---

## Error Handling

### Ollama Connection
```python
# Sjekk i /health endpoint
try:
    ollama.list()
    ollama_status = "connected"
except:
    ollama_status = "disconnected"
```

**Hvis Ollama er nede:**
- Chat returnerer feilmelding til bruker
- Admin PDF-enhancing feiler med error
- Health endpoint viser "ollama: disconnected"

### File Processing Errors
| Error | Handling |
|-------|----------|
| PDF OCR fails | Returnerer error med detaljer |
| OCR timeout | Returnerer partial results |
| XLSX parse error | Returnerer error med detaljer |
| Invalid JSON in upload | Validation error til bruker |

### Common Issues
1. **"Ollama not found"** → `ollama serve` ikke kjørt
2. **"Model not found"** → `ollama pull llama3` 
3. **Trege svar** → CPU-mode, vurder GPU
4. **Tomt svar** → Sjekk vault.txt ikke er tom
5. **Feil språk** → Sjekk at `/english` ikke er i query

---

## Performance Benchmarks

### Response Times (Typical)
| Operation | CPU (i5) | GPU (RTX 4070) |
|-----------|----------|----------------|
| Chat response | 60-180s | 5-15s |
| PDF extraction | 2-10s | 2-10s |
| AI enhancement | 30-90s | 5-20s |
| XLSX parsing | <1s | <1s |
| TF-IDF search | <100ms | <100ms |

### Startup Times
| Component | Time |
|-----------|------|
| Flask app | <1s |
| Load vault.txt | <1s |
| Build TF-IDF index | 1-3s |
| Load JSON knowledge | <1s |
| **Total cold start** | **2-5s** |

### Memory Usage
| Component | RAM |
|-----------|-----|
| Flask app | ~50MB |
| TF-IDF index | ~10-50MB (depends on vault size) |
| Ollama llama3 | ~4-8GB |
| EasyOCR models | ~100MB (loaded on demand) |

---

*Document Created: 2025-06-02*  
*Last Updated: 2025-12-04 - Norwegian default, natural component responses, formatting rules*
