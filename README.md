# Makerspace RAG

En RAG (Retrieval-Augmented Generation) chatbot for Makerspace ved Høgskolen i Østfold. Gir veiledning om digitalt fabrikasjonsutstyr, HMS-regler og komponentlager.

---

## Innholdsfortegnelse

1. [Hva er Makerspace RAG?](#hva-er-makerspace-rag)
2. [Funksjoner](#funksjoner)
3. [Systemkrav](#systemkrav)
4. [Hurtigstart (Exe)](#hurtigstart-exe)
5. [Installasjon for utviklere](#installasjon-for-utviklere)
6. [Brukerveiledning](#brukerveiledning)
7. [Administrasjonspanel](#administrasjonspanel)
8. [Kunnskapsbase](#kunnskapsbase)
9. [Database](#database)
10. [Konfigurasjon](#konfigurasjon)
11. [Feilsøking](#feilsøking)
12. [Prosjektstruktur](#prosjektstruktur)

---

## Hva er Makerspace RAG?

Makerspace RAG er en intelligent chatbot som bruker kunstig intelligens for å hjelpe brukere av Makerspace. Systemet kombinerer:

- **Lokal LLM (Ollama)**: Kjører AI-modellen lokalt på din maskin - ingen data sendes til skyen
- **RAG-teknologi**: Henter relevant informasjon fra kunnskapsbasen for å gi presise svar
- **Semantisk søk**: Forstår meningen bak spørsmålet, ikke bare nøkkelord

### Hvordan det fungerer

1. Bruker stiller et spørsmål i chatten
2. Systemet søker i kunnskapsbasen etter relevant informasjon
3. AI-modellen genererer et svar basert på den hentede informasjonen
4. Svaret vises i chatten med kilder og referanser

---

## Funksjoner

### Chatfunksjoner

| Funksjon | Beskrivelse |
|----------|-------------|
| **Kontekstbevisst chat** | Husker tidligere meldinger i samtalen |
| **Flerspråklig** | Støtter norsk og engelsk |
| **Ferdighetsnivå** | Tilpasser svar til nybegynner eller ekspert |
| **Kildeangivelse** | Viser hvor informasjonen kommer fra |

### Utstyrsveiledning

- Bruksanvisninger for alle maskiner
- HMS-regler og sikkerhetsinstrukser
- Feilsøking og vedlikehold
- Materialguider

### Komponentlager

- Søk etter elektroniske komponenter
- Se lagerstatus og plassering
- Tekniske spesifikasjoner
- Koblingsdiagrammer

### Administrasjon

- Last opp nye dokumenter
- Rediger kunnskapsbasen
- Administrer komponenter
- Se bruksstatistikk

---

## Systemkrav

### Minimumskrav

| Komponent | Krav |
|-----------|------|
| **OS** | Windows 10/11 |
| **RAM** | 8 GB |
| **Lagring** | 10 GB ledig plass |
| **CPU** | 4 kjerner |

### Anbefalte krav (for best ytelse)

| Komponent | Anbefalt |
|-----------|----------|
| **OS** | Windows 11 |
| **RAM** | 16 GB eller mer |
| **Lagring** | SSD med 20 GB ledig |
| **GPU** | NVIDIA GPU med 8+ GB VRAM |

### GPU-akselerasjon (Sterkt anbefalt)

For rask AI-respons anbefales det sterkt å bruke en NVIDIA GPU:

**Fordeler med GPU:**
- 5-10x raskere svar fra AI
- Bedre håndtering av lange dokumenter
- Støtte for større AI-modeller

**Kompatible GPUer:**
- NVIDIA RTX 3060 eller nyere (anbefalt)
- NVIDIA GTX 1660 eller nyere (minimum)
- Krever CUDA-støtte

**Slik aktiverer du GPU:**
1. Installer [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Ollama oppdager GPU automatisk
3. Verifiser med: `ollama run llama3 "test"`

---

## Hurtigstart (Exe)

For sluttbrukere som bare vil kjøre programmet:

### Steg 1: Start programmet

```
dist\MakerspaceRAG\MakerspaceRAG.exe
```

### Steg 2: Første gangs oppsett

Ved første oppstart vil programmet:

1. **Sjekke Ollama** - Installerer automatisk hvis mangler
2. **Laste ned AI-modeller** - llama3 og mxbai-embed-large (~4 GB)
3. **Konfigurere database** - Velg lokal eller ekstern database
4. **Starte nettleser** - Åpner automatisk http://localhost:5000

### Steg 3: Databasevalg

Du får valget mellom:

**[1] Lokal database** - MariaDB på samme maskin
- Enklest oppsett
- Krever MariaDB installert

**[2] Ekstern database** - MariaDB på annen maskin
- For delt tilgang
- Oppgi IP, brukernavn og passord

---

## Installasjon for utviklere

### Forutsetninger

Installer følgende før du begynner:

1. **Python 3.10+**: [python.org](https://python.org)
2. **Node.js 18+**: [nodejs.org](https://nodejs.org)
3. **MariaDB**: [mariadb.org](https://mariadb.org)
4. **Git**: [git-scm.com](https://git-scm.com)

### Steg 1: Klon prosjektet

```bash
git clone <repository-url>
cd Makerspace-RAG
```

### Steg 2: Python-avhengigheter

```bash
pip install -r requirements.txt
```

### Steg 3: Database-oppsett

**Automatisk (anbefalt):**
```bash
python installer/setup_database.py
```

**Manuelt:**
```sql
CREATE DATABASE makerspace_rag CHARACTER SET utf8mb4;
CREATE USER 'makerspace'@'localhost' IDENTIFIED BY 'makerspace2024';
GRANT ALL PRIVILEGES ON makerspace_rag.* TO 'makerspace'@'localhost';
FLUSH PRIVILEGES;
```

### Steg 4: Importer komponentdata (VIKTIG - kun første gang!)

Dette scriptet importerer alle komponenter fra JSON til databasen:

```bash
python installer/import_data.py
```

Velg alternativ 1 for direkte import. Scriptet hopper over hvis data allerede finnes.

### Steg 5: Ollama og AI-modeller

```bash
# Installer Ollama fra ollama.com, deretter:
ollama pull llama3
ollama pull mxbai-embed-large
```

### Steg 6: Bygg React-frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

### Steg 7: Start applikasjonen

```bash
python run.py
```

Åpne http://localhost:5000 i nettleseren.

---

## Brukerveiledning

### Chatte med boten

**Slik bruker du chatten:**

1. Skriv spørsmålet ditt i tekstfeltet nederst
2. Trykk Enter eller klikk Send-knappen
3. Vent på svar (kan ta noen sekunder)
4. Les svaret og eventuelle kilder

**Tips for gode spørsmål:**

| Bra spørsmål | Dårlig spørsmål |
|--------------|-----------------|
| "Hvordan bruker jeg laserkutteren Epilog?" | "Laser" |
| "Hvilke sikkerhetstiltak gjelder for 3D-printing?" | "Regler" |
| "Har dere 10k ohm motstander på lager?" | "Motstand" |

### Velge ferdighetsnivå

Klikk på innstillinger-ikonet for å velge:

- **Nybegynner**: Detaljerte forklaringer, steg-for-steg instrukser
- **Ekspert**: Kortfattede svar, teknisk språk

### Søk etter komponenter

1. Skriv "søk etter [komponent]" i chatten
2. Eller bruk søkefeltet i komponentoversikten
3. Resultatene viser lagerstatus og plassering

---

## Administrasjonspanel

Gå til http://localhost:5000/admin

**Standard innlogging:**
- Brukernavn: `admin`
- Passord: `makerspace2024`

### Funksjoner i admin

#### 1. Last opp dokumenter

Støttede formater:
- PDF (.pdf)
- Tekstfiler (.txt, .md)
- HTML (.html)
- Excel (.xlsx)

**Slik laster du opp:**
1. Klikk "Last opp dokument"
2. Velg fil(er)
3. Dokumentene behandles og legges til kunnskapsbasen

#### 2. Administrer komponenter

- Legg til nye komponenter
- Rediger eksisterende
- Oppdater lagerstatus
- Slett utgåtte komponenter

#### 3. Se statistikk

- Antall spørsmål
- Populære emner
- Responstider

---

## Kunnskapsbase

Kunnskapsbasen er kjernen i RAG-systemet. Den inneholder all informasjon boten kan svare på.

### Filstruktur

```
knowledge/
   components.json      # Elektroniske komponenter
   utstyr.json          # Maskiner og utstyr
   regler.json          # HMS og sikkerhet
   rom.json             # Rominfo og åpningstider
   ressurser.json       # Lenker og ressurser
   prosessflyt.json     # Arbeidsflyter
   prosjektideer.json   # Prosjektforslag
```

### Legge til ny informasjon

#### Metode 1: Via admin-panelet (anbefalt)

1. Gå til Admin > Kunnskapsbase
2. Klikk "Legg til innhold"
3. Fyll ut skjema med tittel, kategori og innhold
4. Klikk Lagre

#### Metode 2: Redigere JSON-filer direkte

Åpne relevant JSON-fil i `knowledge/`-mappen:

**Eksempel - Legge til nytt utstyr i utstyr.json:**

```json
{
  "navn": "Prusa MK4",
  "kategori": "3D-printer",
  "beskrivelse": "Avansert FDM 3D-printer med automatisk kalibrering",
  "plassering": "Rom 101",
  "hms": "Krever opplæring før bruk. Ikke berør varm dyse.",
  "bruksanvisning": "1. Slå på printeren...",
  "materialer": ["PLA", "PETG", "ASA"]
}
```

**Eksempel - Legge til ny komponent i components.json:**

```json
{
  "navn": "LED 5mm Rød",
  "kategori": "LED",
  "beskrivelse": "Standard 5mm rød LED, 20mA, 2V",
  "antall": 500,
  "plassering": "Skuff A3",
  "datablad": "https://...",
  "pinout": {
    "anode": "Lang pinne (+)",
    "katode": "Kort pinne (-)"
  }
}
```

#### Metode 3: Last opp dokumenter

1. Legg PDF/tekstfiler i `uploads/`-mappen
2. Kjør: `python -c "from app.services.knowledge_service import process_uploads; process_uploads()"`
3. Dokumentene blir automatisk chunked og indeksert

### Vault.txt - Prosesserte chunks

Filen `vault.txt` inneholder alle tekstbiter som boten søker i. Hver linje er en "chunk" på ~1000 tegn.

**Regenerere vault.txt:**
```bash
python -c "from app.services.knowledge_service import rebuild_vault; rebuild_vault()"
```

### Oppdatere embeddings

Etter store endringer i kunnskapsbasen:

```bash
python -c "from app.services.embedding_service import rebuild_embeddings; rebuild_embeddings()"
```

---

## Database

### Tabellstruktur

| Tabell | Beskrivelse |
|--------|-------------|
| `users` | Adminbrukere |
| `components` | Elektroniske komponenter |
| `chat_history` | Samtalelogg |
| `documents` | Opplastede dokumenter |

### Legge til data i databasen

#### Via Python-shell

```python
from app import create_app
from app.extensions import db
from app.models.component import Component

app = create_app()
with app.app_context():
    # Legg til ny komponent
    komponent = Component(
        name="Arduino Uno",
        category="Mikrokontroller",
        quantity=10,
        location="Skuff B2"
    )
    db.session.add(komponent)
    db.session.commit()
```

#### Via SQL direkte

```sql
INSERT INTO components (name, category, quantity, location)
VALUES ('Arduino Uno', 'Mikrokontroller', 10, 'Skuff B2');
```

### Slette data

```python
# Slett spesifikk komponent
Component.query.filter_by(name='Arduino Uno').delete()
db.session.commit()

# Slett alle i en kategori
Component.query.filter_by(category='Utgått').delete()
db.session.commit()
```

### Ekstern database (nettverkstilgang)

For å kjøre databasen på en annen maskin:

**På database-serveren:**

1. Kjør setup-scriptet:
   ```bash
   python installer/setup_database.py
   ```

2. Åpne brannmur:
   ```cmd
   netsh advfirewall firewall add rule name="MariaDB" dir=in action=allow protocol=tcp localport=3306
   ```

**På applikasjons-serveren:**

Rediger `.env`:
```env
DB_HOST=192.168.1.100
DB_PORT=3306
DB_USER=makerspace
DB_PASSWORD=makerspace2024
DB_NAME=makerspace_rag
```

---

## Konfigurasjon

### Miljøvariable (.env)

Kopier `.env.example` til `.env` og tilpass:

```env
# Database
DB_HOST=localhost
DB_PORT=3306
DB_USER=makerspace
DB_PASSWORD=makerspace2024
DB_NAME=makerspace_rag

# Flask
FLASK_ENV=production
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
SECRET_KEY=endre-dette-til-noe-sikkert

# Ollama/AI
OLLAMA_HOST=http://127.0.0.1:11434
LLM_MODEL=llama3
EMBEDDING_MODEL=mxbai-embed-large

# Admin
ADMIN_USERNAME=admin
ADMIN_PASSWORD=endre-dette-passordet
```

### Endre AI-modell

For å bruke en annen modell:

1. Last ned modellen: `ollama pull <modellnavn>`
2. Oppdater `.env`: `LLM_MODEL=<modellnavn>`
3. Start applikasjonen på nytt

**Anbefalte modeller:**
- `llama3` - God balanse mellom kvalitet og hastighet
- `llama3:70b` - Beste kvalitet (krever kraftig GPU)
- `mistral` - Rask, god på norsk

### Endre port

Rediger i `.env`:
```env
FLASK_PORT=8080
```

---

## Feilsøking

### Vanlige problemer

#### "Kan ikke koble til database"

1. Sjekk at MariaDB kjører: `net start MariaDB`
2. Verifiser tilkobling: `mysql -u makerspace -p makerspace_rag`
3. Sjekk brannmur for port 3306

#### "Ollama ikke funnet"

1. Installer Ollama fra [ollama.com](https://ollama.com)
2. Start Ollama: `ollama serve`
3. Verifiser: `ollama list`

#### "AI svarer sakte"

1. Sjekk GPU-bruk: `nvidia-smi`
2. Verifiser at CUDA er installert
3. Bruk en mindre modell: `LLM_MODEL=llama3:8b`

#### "Finner ikke relevante svar"

1. Sjekk at informasjonen finnes i kunnskapsbasen
2. Regenerer embeddings
3. Prøv mer spesifikke spørsmål

### Logger

Logger finnes i:
- `app.log` - Applikasjonslogg
- Konsollvinduet ved kjøring

---

## Prosjektstruktur

```
Makerspace-RAG/
   app/                          # Flask-applikasjon
      __init__.py                # App factory
      config.py                  # Konfigurasjon
      extensions.py              # Flask-utvidelser
      models/                    # Databasemodeller
         component.py            # Komponent-modell
         user.py                 # Bruker-modell
      routes/                    # API-endepunkter
         api.py                  # Chat API
         admin.py                # Admin-ruter
         auth.py                 # Innlogging
         public.py               # Offentlige sider
      services/                  # Forretningslogikk
         llm_service.py          # AI-integrasjon
         search_service.py       # Søk og RAG
         embedding_service.py    # Vektorembeddings
         knowledge_service.py    # Kunnskapsbase
      static/react/              # Bygget frontend

   frontend/                     # React-kildekode
      src/
         components/             # React-komponenter
         pages/                  # Sidekomponenter
         styles/                 # CSS-filer

   installer/                    # Installasjonsskript
      launcher.py                # Hovedlauncher
      setup_database.py          # DB-oppsett
      setup_ollama.py            # Ollama-oppsett
      makerspace_rag.spec        # PyInstaller-konfig

   knowledge/                    # Kunnskapsbase (JSON)
   uploads/                      # Opplastede dokumenter
   dist/MakerspaceRAG/           # Bygget exe

   .env                          # Konfigurasjon (ikke i git)
   .env.example                  # Eksempelkonfig
   requirements.txt              # Python-avhengigheter
   run.py                        # Utviklings-entrypoint
   vault.txt                     # Prosesserte tekstbiter
```

---

## Bygge exe fra kildekode

```bash
pip install pyinstaller
python -m PyInstaller installer/makerspace_rag.spec --noconfirm
```

Resultat: `dist/MakerspaceRAG/MakerspaceRAG.exe`

---

## Lisens

Dette prosjektet er utviklet for Høgskolen i Østfold - kun for internbruk.
