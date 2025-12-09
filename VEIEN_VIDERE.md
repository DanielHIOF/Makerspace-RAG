# Veien Videre - Utviklingsplan for Makerspace RAG

Dette dokumentet beskriver planlagte funksjoner, arkitekturforslag og veiledning for videre utvikling av Makerspace RAG.

---

## Innholdsfortegnelse

1. [Prioriterte Forbedringer](#prioriterte-forbedringer)
2. [Koblingsdiagrammer](#koblingsdiagrammer)
3. [Stemmeinteraksjon](#stemmeinteraksjon)
4. [Database-distribusjon](#database-distribusjon)
5. [Infrastruktur og Skalering](#infrastruktur-og-skalering)
6. [Frontend-forbedringer](#frontend-forbedringer)
7. [AI-forbedringer](#ai-forbedringer)
8. [Sikkerhet](#sikkerhet)
9. [Utviklerguide](#utviklerguide)

---

## Prioriterte Forbedringer

### Høy prioritet

| Funksjon | Beskrivelse | Kompleksitet |
|----------|-------------|--------------|
| Database-bunting | Inkludere database med exe | Medium |
| Auto-import | Importer JSON til DB ved oppstart | Lav |
| Koblingsdiagrammer | Vis kretsdiagrammer for komponenter | Høy |
| Bedre feilhåndtering | Bruker vennlige feilmeldinger | Lav |

### Medium prioritet

| Funksjon | Beskrivelse | Kompleksitet |
|----------|-------------|--------------|
| Stemmeinput | Snakk til chatboten | Medium |
| Tekst-til-tale | Boten leser opp svar | Medium |
| Offline-modus | Fungerer uten internett | Medium |
| Multi-bruker | Flere samtidige brukere | Høy |

### Lav prioritet

| Funksjon | Beskrivelse | Kompleksitet |
|----------|-------------|--------------|
| Mobil-app | Native app for telefon | Høy |
| Cloud-hosting | Kjøre i skyen | Medium |
| Flerspråklig UI | Engelsk/norsk grensesnitt | Lav |

---

## Koblingsdiagrammer

### Oversikt

Koblingsdiagrammer (circuit diagrams) viser hvordan elektroniske komponenter skal kobles sammen. Dette er svært nyttig for Makerspace-brukere.

### Implementasjonsalternativer

#### Alternativ 1: Statiske SVG-bilder (Anbefalt for start)

**Fordeler:**
- Enkelt å implementere
- Rask lasting
- Fungerer offline

**Implementasjon:**

1. Lag SVG-filer for vanlige kretser i `app/static/diagrams/`
2. Legg til referanse i `components.json`:

```json
{
  "navn": "LED med motstand",
  "kategori": "Grunnkrets",
  "diagram": "/static/diagrams/led-resistor.svg",
  "komponenter": ["LED", "330Ω motstand", "Strømforsyning"],
  "beskrivelse": "Enkel LED-krets med strømbegrensende motstand"
}
```

3. Vis diagram i frontend:

```jsx
// frontend/src/components/CircuitDiagram.jsx
import React from 'react';

const CircuitDiagram = ({ src, alt }) => {
  return (
    <div className="circuit-diagram">
      <img src={src} alt={alt} />
      <p className="caption">{alt}</p>
    </div>
  );
};

export default CircuitDiagram;
```

#### Alternativ 2: Interaktive diagrammer med Circuit.js

**Fordeler:**
- Interaktiv simulering
- Brukeren kan eksperimentere

**Implementasjon:**

1. Integrer [CircuitJS](https://www.falstad.com/circuit/) som iframe
2. Lagre kretsdefinisjoner som tekstfiler
3. Last inn i simulator ved klikk

```html
<iframe
  src="https://www.falstad.com/circuit/circuitjs.html?ctz=<encoded-circuit>"
  width="100%"
  height="400">
</iframe>
```

#### Alternativ 3: Fritzing-eksport

**Fordeler:**
- Profesjonelle diagrammer
- Breadboard-visning

**Implementasjon:**

1. Lag kretser i [Fritzing](https://fritzing.org/)
2. Eksporter som SVG eller PNG
3. Lagre i `app/static/diagrams/`

### Datastruktur for diagrammer

Utvid `components.json`:

```json
{
  "navn": "Arduino LED-prosjekt",
  "kategori": "Prosjekt",
  "diagrammer": [
    {
      "type": "schematic",
      "url": "/static/diagrams/arduino-led-schematic.svg",
      "beskrivelse": "Koblingsskjema"
    },
    {
      "type": "breadboard",
      "url": "/static/diagrams/arduino-led-breadboard.png",
      "beskrivelse": "Breadboard-oppsett"
    },
    {
      "type": "interactive",
      "url": "https://wokwi.com/share/...",
      "beskrivelse": "Interaktiv simulering"
    }
  ],
  "pinout": {
    "Arduino D13": "LED anode (+)",
    "GND": "LED katode (-) via 330Ω"
  }
}
```

### Backend-endringer

Legg til i `app/services/search_service.py`:

```python
def get_circuit_diagrams(component_name: str) -> list:
    """Hent koblingsdiagrammer for en komponent."""
    # Søk i components.json
    # Returner liste med diagram-URLs
    pass
```

Legg til API-endepunkt i `app/routes/api.py`:

```python
@api.route('/diagrams/<component>')
def get_diagrams(component):
    diagrams = search_service.get_circuit_diagrams(component)
    return jsonify(diagrams)
```

---

## Stemmeinteraksjon

### Stemme-til-tekst (Speech-to-Text)

#### Alternativ 1: Web Speech API (Enklest)

**Fordeler:**
- Innebygd i nettleseren
- Ingen server-side prosessering
- Fungerer på Chrome, Edge, Safari

**Frontend-implementasjon:**

```jsx
// frontend/src/hooks/useSpeechRecognition.js
import { useState, useEffect } from 'react';

const useSpeechRecognition = () => {
  const [transcript, setTranscript] = useState('');
  const [isListening, setIsListening] = useState(false);

  useEffect(() => {
    if (!('webkitSpeechRecognition' in window)) {
      console.warn('Speech recognition ikke støttet');
      return;
    }

    const recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'nb-NO'; // Norsk bokmål

    recognition.onresult = (event) => {
      const current = event.resultIndex;
      const transcript = event.results[current][0].transcript;
      setTranscript(transcript);
    };

    recognition.onend = () => setIsListening(false);

    if (isListening) {
      recognition.start();
    }

    return () => recognition.stop();
  }, [isListening]);

  const startListening = () => setIsListening(true);
  const stopListening = () => setIsListening(false);

  return { transcript, isListening, startListening, stopListening };
};

export default useSpeechRecognition;
```

**Bruk i ChatInput-komponent:**

```jsx
import useSpeechRecognition from '../hooks/useSpeechRecognition';

const ChatInput = ({ onSend }) => {
  const { transcript, isListening, startListening, stopListening } = useSpeechRecognition();
  const [message, setMessage] = useState('');

  useEffect(() => {
    if (transcript) {
      setMessage(transcript);
    }
  }, [transcript]);

  return (
    <div className="chat-input">
      <input
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Skriv eller snakk..."
      />
      <button
        onClick={isListening ? stopListening : startListening}
        className={isListening ? 'recording' : ''}
      >
        🎤
      </button>
      <button onClick={() => onSend(message)}>Send</button>
    </div>
  );
};
```

#### Alternativ 2: Whisper (Lokal, offline)

For bedre norsk støtte og offline-funksjonalitet:

1. Installer whisper.cpp eller faster-whisper
2. Kjør lokalt på server

```python
# app/services/speech_service.py
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cuda")

def transcribe_audio(audio_path: str) -> str:
    segments, info = model.transcribe(audio_path, language="no")
    return " ".join([segment.text for segment in segments])
```

### Tekst-til-tale (Text-to-Speech)

#### Alternativ 1: Web Speech API

```jsx
const speakResponse = (text) => {
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'nb-NO';
  utterance.rate = 0.9;
  window.speechSynthesis.speak(utterance);
};
```

#### Alternativ 2: Coqui TTS (Lokal)

For bedre norsk uttale:

```python
# app/services/tts_service.py
from TTS.api import TTS

tts = TTS(model_name="tts_models/no/cv/vits")

def generate_speech(text: str, output_path: str):
    tts.tts_to_file(text=text, file_path=output_path)
    return output_path
```

---

## Database-distribusjon

### Problem

MariaDB er en separat installasjon som brukeren må håndtere selv.

### Løsning 1: Auto-import fra JSON (Anbefalt)

Modifiser `launcher.py` til å importere data ved første kjøring:

```python
# installer/launcher.py

def import_initial_data():
    """Importer data fra JSON-filer til database."""
    from app import create_app
    from app.extensions import db
    from app.models.component import Component
    import json

    app = create_app()
    with app.app_context():
        # Sjekk om data allerede finnes
        if Component.query.count() > 0:
            print_status("Database har allerede data", "OK")
            return

        # Importer fra components.json
        json_path = PROJECT_ROOT / 'knowledge' / 'components.json'
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                components = json.load(f)

            for comp in components:
                db_comp = Component(
                    name=comp.get('navn', comp.get('name')),
                    category=comp.get('kategori', comp.get('category')),
                    description=comp.get('beskrivelse', ''),
                    quantity=comp.get('antall', 0),
                    location=comp.get('plassering', '')
                )
                db.session.add(db_comp)

            db.session.commit()
            print_status(f"Importert {len(components)} komponenter", "OK")
```

### Løsning 2: SQLite i stedet for MariaDB

For enklere distribusjon, bytt til SQLite:

1. Endre `app/config.py`:

```python
# SQLite (filbasert, ingen server nødvendig)
SQLALCHEMY_DATABASE_URI = 'sqlite:///makerspace.db'
```

2. Inkluder databasefilen i exe:

```python
# installer/makerspace_rag.spec
app_data = [
    ...
    (str(PROJECT_ROOT / 'makerspace.db'), '.'),
]
```

**Fordeler med SQLite:**
- Ingen server å installere
- Enkelt å bundle med exe
- Fungerer offline

**Ulemper:**
- Ikke egnet for mange samtidige brukere
- Begrenset for store datamengder

### Løsning 3: Inkluder SQL-dump

Lag en SQL-fil med all data:

```sql
-- installer/initial_data.sql
INSERT INTO components (name, category, quantity, location) VALUES
('Arduino Uno', 'Mikrokontroller', 10, 'Skuff B2'),
('Raspberry Pi 4', 'Mikrokontroller', 5, 'Skuff B3'),
...
```

Kjør ved oppstart:

```python
def import_sql_dump():
    sql_file = SCRIPT_DIR / 'initial_data.sql'
    if sql_file.exists():
        run_sql_script(mariadb_dir, sql_file, password)
```

---

## Infrastruktur og Skalering

### Nåværende arkitektur

```
[Bruker] → [Flask Web Server] → [Ollama LLM]
                ↓
           [MariaDB]
```

### Anbefalt arkitektur for produksjon

```
[Brukere] → [Nginx/Traefik] → [Flask (Gunicorn)]
                                    ↓
                              [Redis Cache]
                                    ↓
                              [Ollama LLM]
                              [MariaDB/PostgreSQL]
```

### Docker-oppsett

Lag `docker-compose.yml`:

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DB_HOST=db
      - OLLAMA_HOST=http://ollama:11434
    depends_on:
      - db
      - ollama

  db:
    image: mariadb:10.11
    environment:
      MYSQL_ROOT_PASSWORD: rootpassword
      MYSQL_DATABASE: makerspace_rag
      MYSQL_USER: makerspace
      MYSQL_PASSWORD: makerspace2024
    volumes:
      - db_data:/var/lib/mysql

  ollama:
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

volumes:
  db_data:
  ollama_data:
```

### Caching-strategi

Legg til Redis-caching for embeddings:

```python
# app/services/embedding_service.py
import redis
import json
import hashlib

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_embedding_cached(text: str) -> list:
    cache_key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"

    # Sjekk cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Generer ny embedding
    embedding = generate_embedding(text)

    # Lagre i cache (24 timer)
    redis_client.setex(cache_key, 86400, json.dumps(embedding))

    return embedding
```

---

## Frontend-forbedringer

### Planlagte komponenter

#### 1. Komponentvisning med diagrammer

```jsx
// frontend/src/components/ComponentCard.jsx
const ComponentCard = ({ component }) => {
  return (
    <div className="component-card">
      <h3>{component.navn}</h3>
      <p>{component.beskrivelse}</p>

      <div className="specs">
        <span>Antall: {component.antall}</span>
        <span>Plassering: {component.plassering}</span>
      </div>

      {component.diagram && (
        <CircuitDiagram
          src={component.diagram}
          alt={`Koblingsskjema for ${component.navn}`}
        />
      )}

      {component.pinout && (
        <PinoutTable pinout={component.pinout} />
      )}
    </div>
  );
};
```

#### 2. Forbedret chat med kodeblokker

```jsx
// Støtte for syntax highlighting i svar
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';

const MessageContent = ({ content }) => {
  // Parse markdown og vis kodeblokker
  const parts = content.split(/(```\w*\n[\s\S]*?\n```)/g);

  return parts.map((part, i) => {
    if (part.startsWith('```')) {
      const [, lang, code] = part.match(/```(\w*)\n([\s\S]*?)\n```/);
      return (
        <SyntaxHighlighter language={lang || 'text'} key={i}>
          {code}
        </SyntaxHighlighter>
      );
    }
    return <p key={i}>{part}</p>;
  });
};
```

#### 3. Mørk modus

```css
/* frontend/src/styles/dark-mode.css */
:root {
  --bg-primary: #ffffff;
  --text-primary: #1a1a1a;
  --accent: #007bff;
}

[data-theme="dark"] {
  --bg-primary: #1a1a1a;
  --text-primary: #f0f0f0;
  --accent: #4da6ff;
}

body {
  background-color: var(--bg-primary);
  color: var(--text-primary);
}
```

---

## AI-forbedringer

### 1. Kontekstvindu-optimalisering

```python
# app/services/llm_service.py

def optimize_context(context: list, max_tokens: int = 4000) -> list:
    """Prioriter mest relevant kontekst innenfor token-grense."""
    # Sorter etter relevans-score
    sorted_context = sorted(context, key=lambda x: x['score'], reverse=True)

    total_tokens = 0
    optimized = []

    for item in sorted_context:
        item_tokens = len(item['text'].split()) * 1.3  # Estimat
        if total_tokens + item_tokens > max_tokens:
            break
        optimized.append(item)
        total_tokens += item_tokens

    return optimized
```

### 2. Hybrid søk med re-ranking

```python
def hybrid_search(query: str, top_k: int = 10) -> list:
    # Semantisk søk
    semantic_results = embedding_search(query, top_k * 2)

    # Nøkkelord-søk
    keyword_results = keyword_search(query, top_k * 2)

    # Kombiner med Reciprocal Rank Fusion
    combined = reciprocal_rank_fusion(semantic_results, keyword_results)

    # Re-rank med kryss-enkoder
    reranked = cross_encoder_rerank(query, combined[:top_k * 2])

    return reranked[:top_k]
```

### 3. Streaming-svar

```python
# app/routes/api.py

@api.route('/chat/stream', methods=['POST'])
def chat_stream():
    data = request.json

    def generate():
        for chunk in llm_service.stream_response(data['message']):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"

    return Response(generate(), mimetype='text/event-stream')
```

---

## Sikkerhet

### Anbefalte forbedringer

1. **Rate limiting**
```python
from flask_limiter import Limiter
limiter = Limiter(app, key_func=get_remote_address)

@api.route('/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat():
    ...
```

2. **Input-validering**
```python
from bleach import clean

def sanitize_input(text: str) -> str:
    return clean(text, tags=[], strip=True)
```

3. **HTTPS i produksjon**
```nginx
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:5000;
    }
}
```

---

## Utviklerguide

### Legge til ny funksjonalitet

#### 1. Ny API-rute

```python
# app/routes/api.py

@api.route('/ny-funksjon', methods=['POST'])
def ny_funksjon():
    data = request.json
    # Implementer logikk
    return jsonify({'status': 'success'})
```

#### 2. Ny service

```python
# app/services/ny_service.py

class NyService:
    def __init__(self):
        pass

    def prosesser(self, data):
        # Implementer forretningslogikk
        return resultat
```

#### 3. Ny React-komponent

```jsx
// frontend/src/components/NyKomponent.jsx
import React from 'react';
import './NyKomponent.css';

const NyKomponent = ({ prop1, prop2 }) => {
  return (
    <div className="ny-komponent">
      {/* Innhold */}
    </div>
  );
};

export default NyKomponent;
```

### Testing

```bash
# Kjør Python-tester
pytest tests/

# Kjør frontend-tester
cd frontend && npm test

# Lint
flake8 app/
cd frontend && npm run lint
```

### Git workflow

1. Lag ny branch: `git checkout -b feature/ny-funksjon`
2. Gjør endringer og commit
3. Push: `git push -u origin feature/ny-funksjon`
4. Lag Pull Request på GitHub

---

## Kontakt og support

For spørsmål om videre utvikling, kontakt utviklingsteamet ved Høgskolen i Østfold.
