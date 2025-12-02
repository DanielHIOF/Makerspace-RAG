"""
Makerspace RAG - Unified Web Application
Chat interface + Protected Admin panel with authentication
"""

import os
import re
import json
import time
import traceback
import ollama
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
import PyPDF2


# =============================================================================
# Text Chunking
# =============================================================================
def text_to_chunks_improved(text, chunk_size=800, overlap=100):
    """Split text into chunks with overlap for better context."""
    if not text or not text.strip():
        return []
    
    # Clean the text
    text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce multiple newlines
    text = re.sub(r' {2,}', ' ', text)  # Reduce multiple spaces
    
    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If paragraph alone is too big, split it by sentences
        if len(para) > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < chunk_size:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
        else:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Filter out very short chunks
    chunks = [c for c in chunks if len(c) > 50]
    
    return chunks


app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'makerspace-secret-key-change-in-production')

# =============================================================================
# Configuration
# =============================================================================
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'json', 'md', 'csv', 'html', 'htm'}
VAULT_FILE = 'vault.txt'
CHUNK_SIZE = 1000

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Admin credentials - use environment variables in production
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD_HASH = generate_password_hash(os.environ.get('ADMIN_PASSWORD', 'makerspace2024'))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =============================================================================
# Flask-Login Setup
# =============================================================================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access the admin panel.'
login_manager.login_message_category = 'error'


class AdminUser(UserMixin):
    def __init__(self, user_id):
        self.id = user_id


@login_manager.user_loader
def load_user(user_id):
    if user_id == 'admin':
        return AdminUser('admin')
    return None


# =============================================================================
# RAG System - TF-IDF Keyword Search (Fast, no embeddings needed)
# =============================================================================
vault_content = []
tfidf_vectorizer = None
tfidf_matrix = None


def load_vault():
    """Load vault content and build TF-IDF index."""
    global vault_content, tfidf_vectorizer, tfidf_matrix
    
    print("Loading knowledge base...")
    vault_content = []
    if os.path.exists(VAULT_FILE):
        with open(VAULT_FILE, "r", encoding='utf-8') as f:
            vault_content = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(vault_content)} knowledge chunks")
    
    if not vault_content:
        print("No data in knowledge base yet.")
        tfidf_vectorizer = None
        tfidf_matrix = None
        return
    
    # Build TF-IDF index (instant!)
    print("Building search index...")
    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),  # Unigrams and bigrams
        max_df=0.95,         # Ignore terms in >95% of docs
        min_df=1,            # Keep all terms
        stop_words=None      # Keep Norwegian stop words
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(vault_content)
    print(f"Search index ready! ({tfidf_matrix.shape[1]} terms)")


def get_tool_filter_keywords(tool):
    """Get keywords that chunks must contain for a specific tool."""
    filters = {
        '3d_printer': ['3d', 'print', 'printer', 'prusa', 'filament', 'pla', 'abs', 'petg', 
                       'nozzle', 'dyse', 'extruder', 'byggeplate', 'slicer', 'infill'],
        'laserkutter': ['laser', 'kutt', 'graver', 'epilog', 'watt', 'fokus',
                        'akryl', 'mdf', 'speed', 'power', 'dpi', 'engrav'],
        'lodding': ['lodd', 'solder', 'soldering', 'flux', 'kolbe', 'tinning'],
        'arduino': ['arduino', 'uno', 'mega', 'nano', 'sketch', 'atmega'],
        'raspberry': ['raspberry', 'gpio', 'raspbian'],
        'elektronikk': ['krets', 'circuit', 'breadboard', 'motstand', 'resistor', 
                        'kondensator', 'capacitor', 'transistor', 'diode',
                        'pcb', 'volt', 'amp', 'ohm', 'multimeter'],
        'vinylkutter': ['vinyl', 'cricut', 'silhouette', 'klistremerke', 'folie'],
        'tekstil': ['symaskin', 'stoff', 'fabric', 'broderi', 'embroid'],
        'cnc': ['cnc', 'fres', 'mill', 'router', 'carve', 'spindel']
    }
    return filters.get(tool, [])


def expand_query(query):
    """Expand Norwegian query with English synonyms for better TF-IDF matching."""
    expansions = {
        # Lodding / Soldering
        'lodd': 'solder soldering',
        'lodde': 'solder soldering iron',
        'lodder': 'solder soldering how to',
        'lodding': 'solder soldering iron tip flux',
        'tinn': 'solder tin lead-free',
        'loddekolbe': 'soldering iron station tip',
        'kolbe': 'soldering iron',
        'fluss': 'flux rosin',
        'loddetinn': 'solder wire',
        
        # 3D printing
        'printer': 'printer printing 3d print',
        '3d': '3d print printer printing',
        'printe': 'print printing 3d',
        'skrive ut': 'print printing',
        'byggeplate': 'bed build plate adhesion first layer',
        'dyse': 'nozzle hotend extruder clog',
        'filament': 'filament pla abs petg material',
        'ekstrudere': 'extrude extruder extrusion',
        'lag': 'layer height layers',
        'feste': 'adhesion bed stick',
        'løsner': 'warping adhesion lifting detach bed',
        'stringing': 'stringing oozing retraction',
        'tett': 'clogged clog jam nozzle',
        'varme': 'temperature heat bed nozzle',
        'temp': 'temperature heat',
        'slicer': 'slicer slicing cura prusaslicer',
        'stl': 'stl file model',
        'infill': 'infill density fill',
        'support': 'support supports overhang',
        'raft': 'raft brim skirt adhesion',
        'brim': 'brim skirt adhesion',
        
        # Laser
        'laser': 'laser cutter cutting engraving',
        'laserkutter': 'laser cutter cutting engraving epilog',
        'kutte': 'cut cutting speed power',
        'kutter': 'cut cutter cutting',
        'gravere': 'engrave engraving etch',
        'gravering': 'engraving engrave etch',
        'fokus': 'focus height z-offset distance',
        'brenner': 'burn burning fire power',
        'akryl': 'acrylic plexiglass',
        'pleksiglass': 'acrylic plexiglass',
        'tre': 'wood mdf plywood',
        'mdf': 'mdf wood',
        'hastighet': 'speed velocity',
        'styrke': 'power watt strength',
        'watt': 'watt power',
        
        # Electronics
        'krets': 'circuit board pcb',
        'motstand': 'resistor resistance ohm',
        'kondensator': 'capacitor',
        'transistor': 'transistor',
        'diode': 'diode led',
        'led': 'led light diode',
        'arduino': 'arduino uno mega microcontroller',
        'raspberry': 'raspberry pi gpio',
        'breadboard': 'breadboard prototype',
        'multimeter': 'multimeter volt amp ohm',
        
        # General actions
        'hvordan': 'how to guide tutorial steps',
        'bruke': 'use using operate',
        'bruker': 'use using how to',
        'starte': 'start begin power on',
        'slå på': 'power on turn on start',
        'slå av': 'power off turn off stop',
        'fungerer ikke': 'not working problem error fix',
        'virker ikke': 'not working broken error',
        'feil': 'error problem issue fix',
        'problem': 'problem error issue troubleshoot',
        'hjelp': 'help guide tutorial',
        'starter ikke': 'not starting power error',
        'stopper': 'stopping stops error freeze',
        'krasjer': 'crash error freeze',
        
        # Safety / HMS
        'sikkerhet': 'safety safe danger warning',
        'hms': 'safety health environment',
        'fare': 'danger hazard warning',
        'verneutstyr': 'safety equipment protection ppe',
        'briller': 'glasses goggles safety',
        'hansker': 'gloves protection',
        'brann': 'fire burn safety',
        
        # Materials
        'materiale': 'material settings',
        'plast': 'plastic pla abs petg',
        'metall': 'metal aluminum steel',
        'stoff': 'fabric textile cloth',
    }
    
    expanded = query
    query_lower = query.lower()
    
    for no_term, en_terms in expansions.items():
        if no_term in query_lower:
            expanded += ' ' + en_terms
    
    return expanded


def search_vault(query, top_k=3, max_chunk_chars=600, max_total_chars=1800, tool_filter=None):
    """Find most relevant chunks using TF-IDF similarity with optional tool filtering."""
    if tfidf_vectorizer is None or tfidf_matrix is None or len(vault_content) == 0:
        return []
    
    # Remove level commands from query
    clean_query = re.sub(r'/(nybegynner|ny|middels|avansert|ekspert|beginner|new|intermediate|advanced|expert)\s*', '', query)
    
    # Expand Norwegian query with English synonyms
    expanded_query = expand_query(clean_query)
    
    # Transform query to TF-IDF vector
    query_vector = tfidf_vectorizer.transform([expanded_query])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get indices sorted by similarity
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Get tool filter keywords if tool specified
    filter_keywords = get_tool_filter_keywords(tool_filter) if tool_filter else []
    
    # Filter and collect results
    results = []
    total_chars = 0
    
    for i in sorted_indices:
        if similarities[i] <= 0:
            continue
            
        chunk = vault_content[i]
        chunk_lower = chunk.lower()
        
        # If tool filter is set, chunk must contain at least one tool keyword
        if filter_keywords:
            if not any(kw in chunk_lower for kw in filter_keywords):
                continue
        
        # Truncate long chunks
        if len(chunk) > max_chunk_chars:
            chunk = chunk[:max_chunk_chars] + "..."
        
        # Check total size limit
        if total_chars + len(chunk) > max_total_chars:
            break
            
        results.append(chunk)
        total_chars += len(chunk)
        
        if len(results) >= top_k:
            break
    
    return results


def detect_level(query):
    """Detect explanation level from query (supports both Norwegian and English)."""
    query_lower = query.lower()
    
    levels = {
        '/nybegynner': ('nybegynner', "Forklar som om jeg ikke vet noe. Bruk enkle ord, ingen faguttrykk, steg-for-steg med eksempler."),
        '/beginner': ('beginner', "Explain like I know nothing. Use simple words, no jargon, step-by-step with examples."),
        '/ekspert': ('ekspert', "Anta dyp ekspertise. Diskuter på profesjonelt nivå med teori og teknisk presisjon."),
        '/expert': ('expert', "Assume deep expertise. Discuss at professional level with theory and precision."),
    }
    
    for cmd, (level, instruction) in levels.items():
        if cmd in query_lower:
            return level, instruction
    
    return 'normal', "Use clear, practical explanations suitable for someone with basic familiarity."


def detect_language(query):
    """Detect language preference from query."""
    query_lower = query.lower()
    
    if '/norsk' in query_lower or '/no' in query_lower:
        return 'norwegian', "Du MÅ svare på norsk. Bruk norsk språk i hele svaret."
    elif '/english' in query_lower or '/en' in query_lower:
        return 'english', "You MUST respond in English. Use English throughout your entire response."
    
    return 'auto', "Respond in the same language as the question."


def classify_query(query):
    """Classify query into: VERKTOY_HMS, OPPLARING, or FEILSOKING."""
    query_lower = query.lower()
    
    # Feilsøking - noe fungerer ikke
    feilsoking_keywords = [
        'fungerer ikke', 'virker ikke', 'feil', 'error', 'problem', 'stopper',
        'stuck', 'fastkjørt', 'løsner', 'warping', 'stringing', 'clogged',
        'tett', 'brenner', 'kutter ikke', 'printer ikke', 'henger', 'crashed',
        'mislykkes', 'failed', 'hvorfor', 'what went wrong', 'help',
        'ikke riktig', 'dårlig', 'skjev', 'bøyd', 'smelter', 'knekker'
    ]
    
    # Opplæring - hvordan gjøre noe
    opplaring_keywords = [
        'hvordan', 'how to', 'how do', 'steg for steg', 'step by step',
        'guide', 'tutorial', 'lære', 'learn', 'begynne', 'start',
        'første gang', 'first time', 'introduksjon', 'intro', 'basics',
        'grunnleggende', 'eksempel', 'example', 'vise meg', 'show me',
        'forklare', 'explain', 'instruksjon', 'instruction', 'bruke',
        'use', 'lage', 'make', 'create', 'designe', 'design', 'slicing',
        'slice', 'eksportere', 'export', 'importere', 'import', 'settings',
        'innstillinger', 'parametere', 'parameters'
    ]
    
    # Verktøy og HMS - hva finnes, regler, sikkerhet
    verktoy_hms_keywords = [
        'hva slags', 'what kind', 'hvilke', 'which', 'har dere', 'do you have',
        'finnes', 'available', 'utstyr', 'equipment', 'maskin', 'machine',
        'sikkerhet', 'safety', 'hms', 'regler', 'rules', 'fare', 'danger',
        'forbudt', 'forbidden', 'tillatt', 'allowed', 'lov til', 'permitted',
        'verneutstyr', 'protection', 'kan jeg bruke', 'can i use',
        'materiale', 'material', 'type', 'modell', 'model', 'spesifikasjoner',
        'specs', 'kapasitet', 'capacity', 'størrelse', 'size', 'maks', 'max',
        'åpningstider', 'opening hours', 'booking', 'reservere', 'reserve'
    ]
    
    # Check for troubleshooting first (highest priority if something is wrong)
    if any(kw in query_lower for kw in feilsoking_keywords):
        return 'FEILSOKING'
    
    # Check for learning/how-to
    if any(kw in query_lower for kw in opplaring_keywords):
        return 'OPPLARING'
    
    # Check for equipment/HMS info
    if any(kw in query_lower for kw in verktoy_hms_keywords):
        return 'VERKTOY_HMS'
    
    # Default to general guidance
    return 'GENERELL'


def detect_tool(query):
    """Detect which tool/equipment the query is about. More specific tools first."""
    query_lower = query.lower()
    
    # Order matters! More specific tools first, general categories last
    tools = [
        # Specific electronics tools first
        ('lodding', ['lodd', 'solder', 'tinn', 'flux', 'kolbe', 'iron', 'soldering']),
        ('arduino', ['arduino', 'uno', 'mega', 'nano', 'sketch', 'ide', 'atmega']),
        ('raspberry', ['raspberry', 'pi', 'gpio', 'raspbian', 'rpi']),
        
        # Main tools
        ('3d_printer', ['3d print', '3d-print', 'printer', 'prusa', 'filament', 'pla', 'abs', 
                        'petg', 'nozzle', 'dyse', 'extruder', 'bed', 'byggeplate', 'slicer']),
        ('laserkutter', ['laser', 'laserkutt', 'gravering', 'engraving', 'epilog', 
                         'kutte', 'gravere', 'fokus', 'watt']),
        ('vinylkutter', ['vinyl', 'cricut', 'silhouette', 'sticker', 'klistremerke', 'folie']),
        ('tekstil', ['sy', 'sew', 'symaskin', 'stoff', 'fabric', 'broderi', 'embroid']),
        ('cnc', ['cnc', 'fres', 'mill', 'router', 'carve']),
        
        # General electronics last (catch-all)
        ('elektronikk', ['krets', 'circuit', 'breadboard', 'motstand', 'resistor', 
                         'kondensator', 'capacitor', 'led', 'pcb', 'multimeter', 
                         'volt', 'amp', 'ohm', 'elektronikk', 'electronics']),
    ]
    
    for tool, keywords in tools:
        if any(kw in query_lower for kw in keywords):
            return tool
    
    return None


def ask_llm(query, context):
    """Send query + context to llama3."""
    level, level_instruction = detect_level(query)
    language, language_instruction = detect_language(query)
    
    # Clean query of all command prefixes
    clean_query = re.sub(r'/(nybegynner|beginner|ekspert|expert|norsk|no|english|en)\s*', '', query, flags=re.IGNORECASE).strip()
    
    # Classify the query
    query_category = classify_query(clean_query)
    detected_tool = detect_tool(clean_query)
    
    # Simple category hints
    category_hints = {
        'FEILSOKING': 'Brukeren har et PROBLEM. Hjelp med feilsøking.',
        'OPPLARING': 'Brukeren vil LÆRE. Forklar steg-for-steg.',
        'VERKTOY_HMS': 'Spørsmål om UTSTYR/SIKKERHET. Vær presis.',
        'GENERELL': 'Generelt spørsmål. Vær hjelpsom.'
    }
    
    hint = category_hints.get(query_category, '')
    tool_hint = f" (Verktøy: {detected_tool})" if detected_tool else ""
    
    # Keep context short
    context_text = context[:2000] if context else ""
    
    prompt = f"""Du er makerspace-assistenten ved HiØF. {hint}{tool_hint}

{level_instruction}

KONTEKST:
{context_text}

SPØRSMÅL: {clean_query}

Svar kort og praktisk på norsk."""
    
    try:
        response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    except Exception as e:
        error_msg = str(e)
        print(f"  [ERROR] OLLAMA ERROR: {error_msg}")
        raise e


# =============================================================================
# Admin Helper Functions
# =============================================================================
def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_vault_stats():
    """Get statistics about the vault."""
    if not os.path.exists(VAULT_FILE):
        return {'chunks': 0, 'size': 0, 'size_kb': 0}
    
    with open(VAULT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = [l.strip() for l in content.split('\n') if l.strip()]
    
    return {
        'chunks': len(lines),
        'size': len(content),
        'size_kb': round(len(content) / 1024, 2)
    }


def get_recent_chunks(n=10):
    """Get the last n chunks from the vault."""
    if not os.path.exists(VAULT_FILE):
        return []
    
    with open(VAULT_FILE, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines[-n:] if len(lines) >= n else lines


def text_to_chunks(text, chunk_size=CHUNK_SIZE, overlap=100):
    """Split text into chunks with semantic awareness and optional overlap.
    
    Features:
    - Keeps headers with their content
    - Respects paragraph boundaries
    - Adds overlap between chunks for context continuity
    - Handles markdown formatting from PDF extraction
    """
    if not text:
        return []
    
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    
    if not text:
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    
    # Split into sections by headers (markdown style from pymupdf4llm)
    # Pattern matches: # Header, ## Header, ### Header, etc.
    header_pattern = r'(^|\n)(#{1,6}\s+[^\n]+)'
    
    # Split text into sections, keeping headers
    sections = re.split(r'\n(?=#{1,6}\s)', text)
    
    current_chunk = ""
    previous_overlap = ""
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        # Check if this section starts with a header
        header_match = re.match(r'^(#{1,6}\s+[^\n]+)\n?(.*)$', section, re.DOTALL)
        
        if header_match:
            header = header_match.group(1)
            content = header_match.group(2).strip()
            
            # If header + content fits, add together
            combined = f"{header}\n{content}" if content else header
            
            if len(current_chunk) + len(combined) + 2 <= chunk_size:
                current_chunk += ("\n\n" + combined if current_chunk else combined)
            else:
                # Save current chunk with overlap from previous
                if current_chunk:
                    final_chunk = previous_overlap + current_chunk if previous_overlap else current_chunk
                    chunks.append(final_chunk.strip())
                    # Keep last part for overlap
                    previous_overlap = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                
                current_chunk = combined
        else:
            # No header - treat as regular paragraphs
            paragraphs = re.split(r'\n\n+', section)
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                if len(para) > chunk_size:
                    # Save current chunk first
                    if current_chunk:
                        final_chunk = previous_overlap + current_chunk if previous_overlap else current_chunk
                        chunks.append(final_chunk.strip())
                        previous_overlap = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                        current_chunk = ""
                    
                    # Split long paragraph by sentences
                    sentences = re.split(r'(?<=[.!?:;])\s+', para)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        
                        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                            current_chunk += (" " + sentence if current_chunk else sentence)
                        else:
                            if current_chunk:
                                final_chunk = previous_overlap + current_chunk if previous_overlap else current_chunk
                                chunks.append(final_chunk.strip())
                                previous_overlap = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                            
                            # Force split very long sentences
                            if len(sentence) > chunk_size:
                                words = sentence.split()
                                current_chunk = ""
                                for word in words:
                                    if len(current_chunk) + len(word) + 1 <= chunk_size:
                                        current_chunk += (" " + word if current_chunk else word)
                                    else:
                                        if current_chunk:
                                            final_chunk = previous_overlap + current_chunk if previous_overlap else current_chunk
                                            chunks.append(final_chunk.strip())
                                            previous_overlap = ""
                                        current_chunk = word
                            else:
                                current_chunk = sentence
                else:
                    if len(current_chunk) + len(para) + 2 <= chunk_size:
                        current_chunk += ("\n\n" + para if current_chunk else para)
                    else:
                        if current_chunk:
                            final_chunk = previous_overlap + current_chunk if previous_overlap else current_chunk
                            chunks.append(final_chunk.strip())
                            previous_overlap = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                        current_chunk = para
    
    # Don't forget the last chunk
    if current_chunk.strip():
        final_chunk = previous_overlap + current_chunk if previous_overlap else current_chunk
        chunks.append(final_chunk.strip())
    
    # Filter out empty and very short chunks, also remove pure header-only chunks
    filtered = []
    for c in chunks:
        c = c.strip()
        if not c or len(c) < 20:
            continue
        # Skip if it's just a header with no content
        if re.match(r'^#{1,6}\s+[^\n]+$', c) and len(c) < 100:
            continue
        filtered.append(c)
    
    return filtered


def process_pdf(file_path):
    """Extract text from PDF file using best available method."""
    text = ''
    
    # Try PyMuPDF4LLM first (best for RAG)
    try:
        import pymupdf4llm
        print(f"  [PDF] Using PyMuPDF4LLM for extraction...")
        
        # Get page chunks with metadata
        chunks = pymupdf4llm.to_markdown(
            file_path,
            page_chunks=True,  # Get chunks per page
            write_images=False,  # Don't save images
            force_text=True  # Extract text even from image-heavy pages
        )
        
        # Combine all page texts
        for chunk in chunks:
            page_text = chunk.get('text', '')
            if page_text:
                text += page_text + "\n\n"
        
        if text.strip():
            print(f"  [OK] Extracted {len(text)} chars from {len(chunks)} pages")
            return text
            
    except ImportError:
        print("  [WARN] pymupdf4llm not installed, trying pdfplumber...")
    except Exception as e:
        print(f"  [WARN] PyMuPDF4LLM failed: {e}, trying pdfplumber...")
    
    # Try pdfplumber as fallback (good for tables)
    try:
        import pdfplumber
        print(f"  [PDF] Using pdfplumber for extraction...")
        
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
                
                # Also extract tables as text
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        for row in table:
                            if row:
                                row_text = ' | '.join(str(cell) if cell else '' for cell in row)
                                text += row_text + "\n"
                        text += "\n"
        
        if text.strip():
            print(f"  [OK] Extracted {len(text)} chars with pdfplumber")
            return text
            
    except ImportError:
        print("  [WARN] pdfplumber not installed, trying PyPDF2...")
    except Exception as e:
        print(f"  [WARN] pdfplumber failed: {e}, trying PyPDF2...")
    
    # Last resort: PyPDF2 (often misses content)
    try:
        print(f"  [PDF] Using PyPDF2 for extraction (may miss content)...")
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        
        if text.strip():
            print(f"  [WARN] Extracted {len(text)} chars with PyPDF2 (quality may be poor)")
            return text
            
    except Exception as e:
        print(f"  [ERROR] PyPDF2 failed: {e}")
        raise
    
    return text


def process_file(file_path, ext):
    """Process file based on extension and return text content."""
    if ext == 'pdf':
        return process_pdf(file_path)
    elif ext in ('txt', 'md'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == 'json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False)
    elif ext == 'csv':
        # Convert CSV to readable text format
        import csv
        text_parts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if headers:
                text_parts.append("Columns: " + ", ".join(headers))
                text_parts.append("")
                for row in reader:
                    # Create readable row: "Header1: Value1, Header2: Value2"
                    row_text = ", ".join(f"{h}: {v}" for h, v in zip(headers, row) if v.strip())
                    if row_text:
                        text_parts.append(row_text)
        return "\n".join(text_parts)
    elif ext in ('html', 'htm'):
        # Extract text from HTML, stripping tags
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        # Simple HTML tag removal (basic approach)
        import re as html_re
        # Remove script and style content
        text = html_re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=html_re.DOTALL | html_re.IGNORECASE)
        text = html_re.sub(r'<style[^>]*>.*?</style>', '', text, flags=html_re.DOTALL | html_re.IGNORECASE)
        # Remove HTML tags
        text = html_re.sub(r'<[^>]+>', ' ', text)
        # Clean up whitespace
        text = html_re.sub(r'\s+', ' ', text).strip()
        # Decode HTML entities
        import html
        text = html.unescape(text)
        return text
    return ''


def append_to_vault(chunks):
    """Append chunks to the vault file."""
    with open(VAULT_FILE, 'a', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk.strip() + '\n')
    return len(chunks)


def get_existing_chunks():
    """Load existing vault chunks as a set for deduplication."""
    if not os.path.exists(VAULT_FILE):
        return set()
    with open(VAULT_FILE, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())


def normalize_for_comparison(text):
    """Normalize text for similarity comparison."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text


def get_word_set(text):
    """Get set of words from normalized text."""
    normalized = normalize_for_comparison(text)
    return set(normalized.split())


def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def find_similar_chunk(new_chunk, existing_chunks, threshold=0.85):
    """Check if new chunk is too similar to any existing chunk.
    Returns the similar chunk if found, None otherwise."""
    new_words = get_word_set(new_chunk)
    new_normalized = normalize_for_comparison(new_chunk)
    
    for existing in existing_chunks:
        existing_normalized = normalize_for_comparison(existing)
        
        # Exact match after normalization
        if new_normalized == existing_normalized:
            return existing
        
        # Jaccard similarity check
        existing_words = get_word_set(existing)
        similarity = jaccard_similarity(new_words, existing_words)
        if similarity >= threshold:
            return existing
    
    return None


def deduplicate_chunks(new_chunks, similarity_threshold=0.85):
    """Remove duplicates from new chunks, checking against vault and within batch.
    Returns tuple: (unique_chunks, duplicate_count, similar_count)"""
    existing_chunks = get_existing_chunks()
    
    unique_chunks = []
    duplicates = 0
    similar = 0
    seen_normalized = set()  # Track within-batch duplicates
    
    for chunk in new_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        
        normalized = normalize_for_comparison(chunk)
        
        # Check exact duplicate within batch
        if normalized in seen_normalized:
            duplicates += 1
            continue
        
        # Check exact duplicate in vault
        if chunk in existing_chunks:
            duplicates += 1
            continue
        
        # Check similarity against existing chunks
        similar_chunk = find_similar_chunk(chunk, existing_chunks, similarity_threshold)
        if similar_chunk:
            similar += 1
            continue
        
        # Check similarity within this batch
        similar_in_batch = find_similar_chunk(chunk, unique_chunks, similarity_threshold)
        if similar_in_batch:
            similar += 1
            continue
        
        unique_chunks.append(chunk)
        seen_normalized.add(normalized)
    
    return unique_chunks, duplicates, similar


# =============================================================================
# Routes - Public
# =============================================================================
@app.route('/')
def index():
    """Main chat interface (public)."""
    stats = get_vault_stats()
    return render_template('index.html', stats=stats)


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'response': 'Please enter a message.'})
    
    print(f"\n{'='*60}")
    print(f"[{timestamp}] [CHAT] NY MELDING MOTTATT")
    print(f"{'='*60}")
    print(f"  Sporsmal: {message[:100]}{'...' if len(message) > 100 else ''}")
    
    # Classify the query
    query_category = classify_query(message)
    detected_tool = detect_tool(message)
    print(f"  Kategori: {query_category}")
    if detected_tool:
        print(f"  Verktoy: {detected_tool}")
    
    # Search for relevant context (filtered by detected tool)
    print(f"\n[{timestamp}] [SEARCH] Soker i kunnskapsbasen...")
    if detected_tool:
        print(f"  [FILTER] Filtrerer pa verktoy: {detected_tool}")
    search_start = time.time()
    relevant_chunks = search_vault(message, tool_filter=detected_tool)
    search_time = time.time() - search_start
    print(f"  [OK] Fant {len(relevant_chunks)} relevante biter ({search_time:.2f}s)")
    
    if relevant_chunks:
        for i, chunk in enumerate(relevant_chunks[:3]):
            preview = chunk[:80].replace('\n', ' ')
            print(f"    {i+1}. {preview}...")
    
    context = "\n\n".join(relevant_chunks) if relevant_chunks else "No relevant information found in knowledge base."
    
    # Get response from LLM
    print(f"\n[{timestamp}] [LLM] Sender til Ollama (llama3)...")
    print(f"  [WAIT] Venter pa svar (dette kan ta 10-30 sek)...")
    llm_start = time.time()
    
    try:
        response = ask_llm(message, context)
        llm_time = time.time() - llm_start
        print(f"  [OK] Svar mottatt! ({llm_time:.1f}s)")
        print(f"\n[{timestamp}] [SENT] SVAR SENDT TIL BRUKER")
        print(f"  Lengde: {len(response)} tegn")
        print(f"{'='*60}\n")
        return jsonify({'response': response})
    except Exception as e:
        llm_time = time.time() - llm_start
        error_msg = str(e)
        print(f"\n  [ERROR] FEIL etter {llm_time:.1f}s!")
        print(f"  [ERROR] Type: {type(e).__name__}")
        print(f"  [ERROR] Melding: {error_msg}")
        print(f"\n  [TRACE] FULL TRACEBACK:")
        print("-" * 40)
        traceback.print_exc()
        print("-" * 40)
        print(f"\n  [TIP] Sjekk at Ollama kjorer: 'ollama list'")
        print(f"{'='*60}\n")
        return jsonify({'response': f'Feil: {error_msg}. Sjekk terminalen for detaljer.'})


@app.route('/status')
def status():
    """Check if search index is loaded."""
    return jsonify({
        'loaded': tfidf_matrix is not None,
        'chunks': len(vault_content)
    })


@app.route('/health')
def health():
    """Comprehensive health check endpoint."""
    import requests
    
    # Check Ollama
    ollama_ok = False
    ollama_models = []
    try:
        resp = requests.get('http://127.0.0.1:11434/api/tags', timeout=5)
        if resp.status_code == 200:
            ollama_ok = True
            data = resp.json()
            ollama_models = [m['name'] for m in data.get('models', [])]
    except:
        pass
    
    # Check vault
    vault_ok = os.path.exists(VAULT_FILE) and len(vault_content) > 0
    
    # Check search index
    search_ok = tfidf_matrix is not None
    
    # Overall health
    all_ok = ollama_ok and vault_ok and search_ok
    
    return jsonify({
        'status': 'healthy' if all_ok else 'degraded',
        'ollama': {
            'running': ollama_ok,
            'models': ollama_models
        },
        'knowledge_base': {
            'loaded': vault_ok,
            'chunks': len(vault_content)
        },
        'search_index': {
            'ready': search_ok,
            'terms': tfidf_matrix.shape[1] if search_ok else 0
        }
    })


# =============================================================================
# Routes - Authentication
# =============================================================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Admin login page."""
    if current_user.is_authenticated:
        return redirect(url_for('admin'))
    
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        
        if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
            user = AdminUser('admin')
            login_user(user)
            flash('Logged in successfully!', 'success')
            
            next_page = request.args.get('next')
            return redirect(next_page or url_for('admin'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """Log out admin user."""
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))


# =============================================================================
# Routes - Admin (Protected)
# =============================================================================
@app.route('/admin')
@login_required
def admin():
    """Admin panel for managing knowledge base."""
    stats = get_vault_stats()
    recent = get_recent_chunks(5)
    return render_template('admin.html', stats=stats, recent_chunks=recent)


@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Handle single or multiple file uploads to knowledge base."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    files = request.files.getlist('file')
    
    if not files or all(f.filename == '' for f in files):
        return jsonify({'success': False, 'error': 'No files selected'}), 400
    
    results = []
    total_chunks = 0
    total_duplicates = 0
    total_similar = 0
    errors = []
    
    for file in files:
        if file.filename == '':
            continue
            
        if not allowed_file(file.filename):
            errors.append(f'{file.filename}: type not allowed')
            continue
        
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            ext = filename.rsplit('.', 1)[1].lower()
            text = process_file(file_path, ext)
            
            chunks = text_to_chunks_improved(text)
            
            if not chunks:
                os.remove(file_path)
                errors.append(f'{filename}: no text extracted')
                continue
            
            # Deduplicate chunks
            unique_chunks, duplicates, similar = deduplicate_chunks(chunks)
            total_duplicates += duplicates
            total_similar += similar
            
            if unique_chunks:
                added_count = append_to_vault(unique_chunks)
                total_chunks += added_count
                
                detail = f'{added_count} new'
                if duplicates or similar:
                    detail += f' ({duplicates} dup, {similar} similar skipped)'
                results.append(f'{filename}: {detail}')
            else:
                results.append(f'{filename}: all {len(chunks)} chunks already exist')
            
            os.remove(file_path)  # Clean up
            
        except Exception as e:
            errors.append(f'{filename}: {str(e)}')
    
    if not results and errors:
        return jsonify({
            'success': False,
            'error': '; '.join(errors)
        }), 400
    
    message = f'Added {total_chunks} new chunks from {len(results)} file(s)'
    if total_duplicates or total_similar:
        message += f' | Skipped: {total_duplicates} duplicates, {total_similar} similar'
    if errors:
        message += f' | Errors: {len(errors)}'
    
    return jsonify({
        'success': True,
        'message': message,
        'chunks_added': total_chunks,
        'duplicates_skipped': total_duplicates,
        'similar_skipped': total_similar,
        'files_processed': results,
        'errors': errors,
        'stats': get_vault_stats()
    })


@app.route('/add-text', methods=['POST'])
@login_required
def add_text():
    """Add text directly to the vault."""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'success': False, 'error': 'No text provided'}), 400
    
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'success': False, 'error': 'Empty text'}), 400
    
    try:
        chunks = text_to_chunks_improved(text)
        
        if not chunks:
            return jsonify({'success': False, 'error': 'Could not create chunks'}), 400
        
        # Deduplicate chunks
        unique_chunks, duplicates, similar = deduplicate_chunks(chunks)
        
        if not unique_chunks:
            return jsonify({
                'success': True,
                'message': f'No new content. {duplicates} duplicates, {similar} similar chunks already exist.',
                'chunks_added': 0,
                'duplicates_skipped': duplicates,
                'similar_skipped': similar,
                'stats': get_vault_stats()
            })
        
        added_count = append_to_vault(unique_chunks)
        
        message = f'Added {added_count} new chunks'
        if duplicates or similar:
            message += f' (skipped {duplicates} dup, {similar} similar)'
        
        return jsonify({
            'success': True,
            'message': message,
            'chunks_added': added_count,
            'duplicates_skipped': duplicates,
            'similar_skipped': similar,
            'stats': get_vault_stats()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/stats')
@login_required
def stats():
    """Get vault statistics (admin only)."""
    return jsonify(get_vault_stats())


@app.route('/recent')
@login_required
def recent():
    """Get recent chunks (admin only)."""
    n = request.args.get('n', 10, type=int)
    chunks = get_recent_chunks(n)
    return jsonify({'chunks': chunks, 'count': len(chunks)})


@app.route('/reload', methods=['POST'])
@login_required
def reload():
    """Reload search index after adding new content."""
    load_vault()
    return jsonify({'success': True, 'message': 'Søkeindeks oppdatert', 'chunks': len(vault_content)})


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Makerspace RAG - Unified Web Application")
    print("=" * 60)
    print(f"\n  Vault: {VAULT_FILE}")
    stats = get_vault_stats()
    print(f"  Current chunks: {stats['chunks']}")
    print(f"\n  Public:  http://localhost:5000/")
    print(f"  Admin:   http://localhost:5000/admin")
    print(f"  Login:   admin / makerspace2024")
    print("=" * 60 + "\n")
    
    # Load embeddings on startup
    load_vault()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
