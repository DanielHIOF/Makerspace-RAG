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

# Structured knowledge from JSON files
knowledge_utstyr = {}
knowledge_regler = {}
knowledge_rom = {}
knowledge_ressurser = {}


def load_json_knowledge():
    """Load structured knowledge from JSON files."""
    global knowledge_utstyr, knowledge_regler, knowledge_rom, knowledge_ressurser
    
    knowledge_dir = 'knowledge'
    
    try:
        with open(os.path.join(knowledge_dir, 'utstyr.json'), 'r', encoding='utf-8') as f:
            knowledge_utstyr = json.load(f)
        print(f"  Loaded utstyr.json")
    except Exception as e:
        print(f"  Warning: Could not load utstyr.json: {e}")
        knowledge_utstyr = {}
    
    try:
        with open(os.path.join(knowledge_dir, 'regler.json'), 'r', encoding='utf-8') as f:
            knowledge_regler = json.load(f)
        print(f"  Loaded regler.json")
    except Exception as e:
        print(f"  Warning: Could not load regler.json: {e}")
        knowledge_regler = {}
    
    try:
        with open(os.path.join(knowledge_dir, 'rom.json'), 'r', encoding='utf-8') as f:
            knowledge_rom = json.load(f)
        print(f"  Loaded rom.json")
    except Exception as e:
        print(f"  Warning: Could not load rom.json: {e}")
        knowledge_rom = {}
    
    try:
        with open(os.path.join(knowledge_dir, 'ressurser.json'), 'r', encoding='utf-8') as f:
            knowledge_ressurser = json.load(f)
        print(f"  Loaded ressurser.json")
    except Exception as e:
        print(f"  Warning: Could not load ressurser.json: {e}")
        knowledge_ressurser = {}


def get_equipment_context(tool_type):
    """Get structured equipment info for a detected tool type."""
    if not knowledge_utstyr or 'categories' not in knowledge_utstyr:
        return ""
    
    # Map tool detection to JSON category
    tool_to_category = {
        '3d_printer': '3d_printing',
        'laserkutter': 'laser_cutting',
        'cnc': 'cnc',
        'lodding': 'electronics',
        'elektronikk': 'electronics',
        'vinylkutter': None,  # Not in JSON yet
        'tekstil': None,
        'arduino': 'electronics',
        'raspberry': 'electronics'
    }
    
    category = tool_to_category.get(tool_type)
    if not category or category not in knowledge_utstyr['categories']:
        return ""
    
    cat_data = knowledge_utstyr['categories'][category]
    equipment_list = cat_data.get('equipment', [])
    
    if not equipment_list:
        return ""
    
    # Build context string
    lines = [f"UTSTYR ({cat_data.get('name_no', category)}):"]
    for eq in equipment_list:
        lines.append(f"- {eq.get('name', 'Ukjent')}")
        if eq.get('location'):
            lines.append(f"  Lokasjon: {eq['location']}")
        if eq.get('difficulty'):
            lines.append(f"  Nivå: {eq['difficulty']}")
        if eq.get('requires_training'):
            lines.append(f"  Krever opplæring: Ja")
        if eq.get('materials'):
            lines.append(f"  Materialer: {', '.join(eq['materials'])}")
    
    return "\n".join(lines)


def get_all_equipment_by_access():
    """Get all equipment organized by access level - for 'what can I use' questions."""
    if not knowledge_utstyr or 'categories' not in knowledge_utstyr:
        return ""
    
    # Get access level definitions
    access_levels = knowledge_utstyr.get('access_levels', {})
    
    # Collect equipment by access level
    by_access = {}
    for cat_key, cat_data in knowledge_utstyr['categories'].items():
        for eq in cat_data.get('equipment', []):
            access = eq.get('access_level', 'unknown')
            if access not in by_access:
                by_access[access] = []
            by_access[access].append({
                'name': eq.get('name', 'Ukjent'),
                'location': eq.get('location', ''),
                'difficulty': eq.get('difficulty', ''),
                'certifier': eq.get('certifier', ''),
                'notes': eq.get('notes', '')
            })
    
    # Build context string in order of access level
    lines = ["TILGANGSNIVÅER FOR UTSTYR VED MAKERSPACE HiØF:", ""]
    
    access_order = ['course_makerspace', 'course_fablab', 'certification_required', 'request_required', 'staff_only']
    access_names = {
        'course_makerspace': '1. MakerSpace-kurs (D1-044) - Etter fullført kurs kan du bruke:',
        'course_fablab': '2. FabLab HMS-kurs (D1-043) - Krever HMS-kurs:',
        'certification_required': '3. Sertifisering påkrevd - Kontakt labingeniør:',
        'request_required': '4. Må hentes fra labansvarlig:',
        'staff_only': '5. Kun personale:'
    }
    
    for access in access_order:
        if access in by_access:
            lines.append(access_names.get(access, access))
            for eq in by_access[access]:
                line = f"  - {eq['name']}"
                if eq['location']:
                    line += f" ({eq['location']})"
                lines.append(line)
            lines.append("")
    
    lines.append("VIKTIG: Uten opplæring kan du IKKE bruke noe utstyr.")
    lines.append("Start med: MakerSpace introduksjonskurs for å få tilgang til 3D-printere, lodding, osv.")
    lines.append("Mer info: https://www.hiof.no/iio/itk/om/labber/makerspace/arrangementer/")
    
    return "\n".join(lines)


def get_safety_rules_context(tool_type=None, include_general=True):
    """Get HMS/safety rules for a tool type or general rules."""
    if not knowledge_regler:
        return ""
    
    lines = []
    
    # General rules
    if include_general and 'general_rules' in knowledge_regler:
        gen_rules = knowledge_regler['general_rules'].get('rules', [])
        critical_rules = [r for r in gen_rules if r.get('priority') == 'critical']
        if critical_rules:
            lines.append("VIKTIGE SIKKERHETSREGLER:")
            for r in critical_rules[:3]:  # Top 3 critical
                lines.append(f"⚠️ {r.get('rule_no', '')}")
    
    # Equipment-specific rules
    if tool_type and 'equipment_specific' in knowledge_regler:
        tool_to_category = {
            '3d_printer': '3d_printing',
            'laserkutter': 'laser_cutting',
            'cnc': 'cnc',
            'lodding': 'soldering',
            'elektronikk': 'soldering'
        }
        
        category = tool_to_category.get(tool_type)
        if category and category in knowledge_regler['equipment_specific']:
            cat_rules = knowledge_regler['equipment_specific'][category]
            rules = cat_rules.get('rules', [])
            
            if rules:
                lines.append(f"\nREGLER FOR {cat_rules.get('name_no', category).upper()}:")
                for r in rules[:4]:  # Top 4 rules
                    priority_icon = "🔴" if r.get('priority') == 'critical' else "🟡"
                    lines.append(f"{priority_icon} {r.get('rule_no', '')}")
            
            # Forbidden materials for laser
            if category == 'laser_cutting':
                forbidden = cat_rules.get('forbidden_materials', [])
                if forbidden:
                    lines.append("\n⛔ FORBUDTE MATERIALER:")
                    for f in forbidden[:4]:
                        lines.append(f"- {f.get('material')}: {f.get('reason_no', '')}")
    
    return "\n".join(lines)


def get_room_context(room_id=None):
    """Get room information."""
    if not knowledge_rom or 'rooms' not in knowledge_rom:
        return ""
    
    lines = ["LOKALER:"]
    for room_key, room in knowledge_rom['rooms'].items():
        lines.append(f"- {room.get('id', '')}: {room.get('name_no', '')} - {room.get('description_no', '')}")
    
    # Add contact info
    if 'contacts' in knowledge_rom:
        contacts = knowledge_rom['contacts']
        if 'lab_responsible' in contacts:
            lines.append(f"\nLabansvarlig: Morgan Waage (kontor D1-060B)")
        if 'student_assistants' in contacts:
            lines.append("Studentassistenter tilgjengelig i åpningstider")
    
    return "\n".join(lines)


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
        '/nybegynner': ('nybegynner', "NYBEGYNNER - Forklar som til en som aldri har gjort dette før. Bruk enkle ord, unngå faguttrykk, gi steg-for-steg instruksjoner med eksempler. Vær tålmodig og grundig."),
        '/beginner': ('beginner', "BEGINNER - Explain as if to someone who has never done this before. Use simple words, avoid jargon, give step-by-step instructions with examples."),
        '/ekspert': ('ekspert', "EKSPERT - Anta at brukeren har dyp teknisk kunnskap. Bruk presise faguttrykk, diskuter på profesjonelt nivå, inkluder tekniske detaljer og teori. Vær konsis."),
        '/expert': ('expert', "EXPERT - Assume deep technical knowledge. Use precise terminology, discuss at professional level, include technical details and theory. Be concise."),
    }
    
    for cmd, (level, instruction) in levels.items():
        if cmd in query_lower:
            return level, instruction
    
    return 'normal', "NORMAL - Bruk klare, praktiske forklaringer. Balanse mellom enkelhet og presisjon. Forklar faguttrykk kort hvis du bruker dem."


def detect_language(query):
    """Detect language preference from query."""
    query_lower = query.lower()
    
    if '/norsk' in query_lower or '/no' in query_lower:
        return 'norwegian', "Du MÅ svare på norsk. Bruk norsk språk i hele svaret."
    elif '/english' in query_lower or '/en' in query_lower:
        return 'english', "You MUST respond in English. Use English throughout your entire response."
    
    return 'auto', "Svar på SAMME språk som brukeren skriver. Norsk spørsmål = norsk svar. English question = English answer."


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


def is_inventory_query(query):
    """Detect if query is asking for a list/inventory of equipment.
    These should use JSON data primarily, not vault search."""
    query_lower = query.lower()
    
    inventory_patterns = [
        'liste over', 'list of', 'hvilke', 'which',
        'hva har dere', 'what do you have', 'har dere',
        'vis meg alle', 'show me all', 'alle',
        'hva slags', 'what kind', 'typer',
        'oversikt', 'overview', 'inventory',
        'tilgjengelig', 'available', 'finnes'
    ]
    
    return any(p in query_lower for p in inventory_patterns)


# =============================================================================
# Conversation Compression
# =============================================================================
INCREMENTAL_COMPRESS_EVERY = 6  # Compress every 6 new messages (3 exchanges)
RECENT_MESSAGES_KEEP = 10       # Always keep last 10 messages in full

def summarize_messages(messages, existing_summary=""):
    """Compress messages into a summary, incorporating existing summary."""
    if not messages:
        return existing_summary
    
    # Build conversation text for summarization
    conversation_text = ""
    for msg in messages:
        role = "Bruker" if msg.get('role') == 'user' else "Assistent"
        content = msg.get('content', '')[:200]  # Truncate long messages
        conversation_text += f"{role}: {content}\n"
    
    # Include existing summary in prompt if available
    existing_context = ""
    if existing_summary:
        existing_context = f"\nTIDLIGERE KONTEKST:\n{existing_summary}\n"
    
    prompt = f"""Lag en KORT oppsummering (2-3 setninger) av samtalen.
Behold: hovedtema, viktige beslutninger, spesifikke detaljer (utstyr, innstillinger, problemer).
{existing_context}
NYE MELDINGER:
{conversation_text}

OPPSUMMERING:"""

    try:
        response = ollama.chat(
            model='llama3.2:1b',  # Fast small model for compression
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.3, 'num_predict': 100}
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"  [WARN] Compression failed: {e}")
        return existing_summary


def ask_llm(query, context, is_inventory=False, conversation_history=None, existing_summary=""):
    """Send query + context to llama3 with conversation history support.
    Returns tuple: (response_text, updated_summary)
    """
    level, level_instruction = detect_level(query)
    language, language_instruction = detect_language(query)
    
    # Clean query of all command prefixes
    clean_query = re.sub(r'/(nybegynner|beginner|ekspert|expert|norsk|no|english|en)\s*', '', query, flags=re.IGNORECASE).strip()
    
    # Classify the query
    query_category = classify_query(clean_query)
    detected_tool = detect_tool(clean_query)
    
    tool_hint = f" (Verktøy: {detected_tool})" if detected_tool else ""
    
    # Keep context short
    context_text = context[:2000] if context else ""
    
    # Base role description
    base_role = """Du er en veileder og ekspert på Digital Fabrikasjon og HMS (Helse, Miljø og Sikkerhet) ved Makerspace, Høgskolen i Østfold.

Din rolle er å:
- Veilede studenter trygt gjennom bruk av utstyr (3D-printing, laserkutting, CNC, elektronikk, lodding)
- ALLTID prioritere sikkerhet - minn om HMS-regler når relevant
- Gi praktiske, handlingsrettede råd
- Tilpasse forklaringer til brukerens ferdighetsnivå"""

    # Build system prompt based on query type
    if is_inventory:
        system_prompt = f"""{base_role}

TILGJENGELIG UTSTYR:
{context_text}

FERDIGHETSNIVÅ: {level_instruction}

SPRÅK: {language_instruction}

REGLER FOR SVAR:
- List kun utstyret som er relevant
- Maks 2-3 linjer per utstyr (navn, lokasjon, nivå)
- Nevn eventuelle HMS-krav eller opplæringskrav
- Avslutt med: "Vil du vite mer om noe av dette?"
- VIKTIG: Husk samtalehistorikken"""
    else:
        system_prompt = f"""{base_role}{tool_hint}

RELEVANT INFORMASJON:
{context_text}

FERDIGHETSNIVÅ: {level_instruction}

SPRÅK: {language_instruction}

REGLER FOR SVAR:
- Svar KORT (2-4 setninger) men informativt
- Gi ETT konkret tips eller neste steg
- Hvis relevant: minn om HMS/sikkerhet (verneutstyr, farlige materialer, osv.)
- Still et oppfølgingsspørsmål for å holde samtalen i gang
- VIKTIG: Husk hva dere har snakket om tidligere i samtalen"""
    
    # Build messages array for Ollama
    messages = [{'role': 'system', 'content': system_prompt}]
    
    # Handle conversation history with incremental compression
    updated_summary = existing_summary
    
    if conversation_history:
        total_messages = len(conversation_history)
        
        # Add existing summary as context
        if existing_summary:
            messages.append({
                'role': 'system', 
                'content': f"TIDLIGERE I SAMTALEN:\n{existing_summary}"
            })
        
        # Only keep last RECENT_MESSAGES_KEEP messages in full
        recent_messages = conversation_history[-RECENT_MESSAGES_KEEP:] if total_messages > RECENT_MESSAGES_KEEP else conversation_history
        
        # Check if we need to compress (every INCREMENTAL_COMPRESS_EVERY messages)
        messages_since_last_compress = total_messages % INCREMENTAL_COMPRESS_EVERY
        if total_messages >= INCREMENTAL_COMPRESS_EVERY and messages_since_last_compress == 0:
            # Get the messages to compress (the ones just before recent)
            compress_start = max(0, total_messages - RECENT_MESSAGES_KEEP - INCREMENTAL_COMPRESS_EVERY)
            compress_end = total_messages - RECENT_MESSAGES_KEEP
            if compress_end > compress_start:
                to_compress = conversation_history[compress_start:compress_end]
                print(f"  [COMPRESS] Inkrementell komprimering av {len(to_compress)} meldinger")
                updated_summary = summarize_messages(to_compress, existing_summary)
        
        # Add recent messages in full
        for msg in recent_messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role in ('user', 'assistant') and content:
                messages.append({'role': role, 'content': content})
    
    # Add current query as the last user message
    messages.append({'role': 'user', 'content': clean_query})
    
    try:
        response = ollama.chat(
            model='llama3',
            messages=messages,
            options={'temperature': 0.7, 'num_predict': 500}
        )
        return response['message']['content'], updated_summary
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
    """Handle chat messages with conversation history support."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    data = request.get_json()
    message = data.get('message', '').strip()
    conversation_history = data.get('history', [])  # Get conversation history
    existing_summary = data.get('summary', '')  # Get existing summary from frontend
    
    if not message:
        return jsonify({'response': 'Please enter a message.'})
    
    # =========================================================================
    # EASTER EGG: Green apples test for link functionality
    # =========================================================================
    if 'green apples' in message.lower() or 'grønne epler' in message.lower():
        print(f"\n[{timestamp}] [EASTER EGG] Green apples detected!")
        test_response = """Ja, grønne epler er kjempegode! 🍏

De er sprø, friske og fulle av smak. Granny Smith er en klassiker.

Vil du lese mer om frukt og helse? Sjekk ut denne artikkelen: [Les mer på VG](https://www.vg.no)

Du kan også besøke siden direkte: https://www.vg.no"""
        return jsonify({
            'response': test_response,
            'summary': existing_summary
        })
    
    print(f"\n{'='*60}")
    print(f"[{timestamp}] [CHAT] NY MELDING MOTTATT")
    print(f"{'='*60}")
    print(f"  Sporsmal: {message[:100]}{'...' if len(message) > 100 else ''}")
    print(f"  Historikk: {len(conversation_history)} meldinger")
    if existing_summary:
        print(f"  Summary: {existing_summary[:50]}...")
    
    # Build combined context from history for better tool detection
    # If current message is short/vague, use history for context
    combined_context = message
    if len(message.split()) < 5 and conversation_history:
        # Include last few messages for context detection
        recent_messages = [msg.get('content', '') for msg in conversation_history[-4:]]
        combined_context = ' '.join(recent_messages) + ' ' + message
        print(f"  [CONTEXT] Kombinerer med historikk for bedre verktøydeteksjon")
    
    # Classify the query (use combined context for better detection)
    query_category = classify_query(combined_context)
    detected_tool = detect_tool(combined_context)
    inventory_query = is_inventory_query(message)  # Keep this on current message only
    
    print(f"  Kategori: {query_category}")
    if detected_tool:
        print(f"  Verktoy: {detected_tool}")
    if inventory_query:
        print(f"  Type: INVENTAR-SPØRSMÅL (bruker kun JSON)")
    
    # Build context from multiple sources
    context_parts = []
    
    # 1. For inventory queries without specific tool: show ALL equipment by access level
    if inventory_query and not detected_tool:
        all_equipment_ctx = get_all_equipment_by_access()
        if all_equipment_ctx:
            context_parts.append(all_equipment_ctx)
            print(f"  [JSON] Lagt til komplett utstyrsoversikt etter tilgangsnivå")
    
    # 2. Structured JSON context based on detected tool
    elif detected_tool:
        equipment_ctx = get_equipment_context(detected_tool)
        if equipment_ctx:
            context_parts.append(equipment_ctx)
            print(f"  [JSON] Lagt til utstyrskontekst for {detected_tool}")
    
    # 3. Safety rules - SKIP for inventory queries
    if not inventory_query and (query_category == 'VERKTOY_HMS' or detected_tool):
        safety_ctx = get_safety_rules_context(tool_type=detected_tool, include_general=(query_category == 'VERKTOY_HMS'))
        if safety_ctx:
            context_parts.append(safety_ctx)
            print(f"  [JSON] Lagt til sikkerhetsregler")
    
    # 3. Room info for location questions
    if 'rom' in message.lower() or 'hvor' in message.lower() or 'lokasjon' in message.lower():
        room_ctx = get_room_context()
        if room_ctx:
            context_parts.append(room_ctx)
            print(f"  [JSON] Lagt til rominfo")
    
    # 4. TF-IDF search - SKIP for inventory queries (JSON is enough)
    if not inventory_query:
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
            context_parts.append("DOKUMENTASJON:\n" + "\n\n".join(relevant_chunks))
    
    context = "\n\n".join(context_parts) if context_parts else "Ingen relevant informasjon funnet."
    
    # Get response from LLM
    print(f"\n[{timestamp}] [LLM] Sender til Ollama (llama3)...")
    print(f"  [WAIT] Venter pa svar...")
    llm_start = time.time()
    
    try:
        response, updated_summary = ask_llm(
            message, context, 
            is_inventory=inventory_query, 
            conversation_history=conversation_history,
            existing_summary=existing_summary
        )
        llm_time = time.time() - llm_start
        print(f"  [OK] Svar mottatt! ({llm_time:.1f}s)")
        print(f"\n[{timestamp}] [SENT] SVAR SENDT TIL BRUKER")
        print(f"  Lengde: {len(response)} tegn")
        print(f"{'='*60}\n")
        return jsonify({
            'response': response,
            'summary': updated_summary  # Return updated summary to frontend
        })
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
# Fast PDF Import - Quick extraction with optional AI enhancement
# =============================================================================
def extract_pdf_fast(file_path):
    """Fast PDF text extraction using PyPDF2 only. Returns text quickly."""
    text = ''
    try:
        print(f"  [PDF-FAST] Using PyPDF2 for quick extraction...")
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        print(f"  [PDF-FAST] Extracted {len(text)} chars from {total_pages} pages")
        return text, total_pages
    except Exception as e:
        print(f"  [PDF-FAST] Error: {e}")
        raise


@app.route('/extract-pdf', methods=['POST'])
@login_required
def extract_pdf():
    """Fast PDF extraction - returns raw text immediately for preview."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Ingen fil lastet opp'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Ingen fil valgt'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'success': False, 'error': 'Kun PDF-filer støttes'}), 400
    
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        print(f"[FAST EXTRACT] Processing {filename}...")
        start_time = time.time()
        
        pdf_text, page_count = extract_pdf_fast(file_path)
        
        os.remove(file_path)
        
        elapsed = time.time() - start_time
        print(f"[FAST EXTRACT] Done in {elapsed:.2f}s")
        
        if not pdf_text or len(pdf_text.strip()) < 50:
            return jsonify({'success': False, 'error': 'Kunne ikke trekke ut tekst fra PDF'}), 400
        
        return jsonify({
            'success': True,
            'filename': filename,
            'text': pdf_text,
            'char_count': len(pdf_text),
            'page_count': page_count,
            'extract_time': round(elapsed, 2)
        })
        
    except Exception as e:
        print(f"[FAST EXTRACT] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/enhance-pdf', methods=['POST'])
@login_required
def enhance_pdf():
    """Use LLM to structure/summarize extracted PDF text."""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'success': False, 'error': 'Ingen tekst å behandle'}), 400
    
    pdf_text = data.get('text', '').strip()
    
    if len(pdf_text) < 50:
        return jsonify({'success': False, 'error': 'For lite tekst å behandle'}), 400
    
    # Truncate if too long
    max_chars = 12000
    truncated = False
    if len(pdf_text) > max_chars:
        pdf_text = pdf_text[:max_chars]
        truncated = True
        print(f"[ENHANCE] Truncated to {max_chars} chars")
    
    print(f"[ENHANCE] Sending {len(pdf_text)} chars to LLM...")
    
    prompt = f"""Du er en ekspert på å lage strukturert dokumentasjon for et Makerspace.

Les følgende dokumentasjon og lag strukturerte kunnskapsoppføringer.

REGLER:
1. Behold VIKTIG teknisk informasjon (innstillinger, prosedyrer, sikkerhet)
2. Fjern unødvendig fyll og gjentakelser
3. Skriv konsist og praktisk
4. Bruk norsk språk (tekniske termer kan være på engelsk)
5. Skill mellom ulike emner med tomme linjer

DOKUMENTASJON:
{pdf_text}

STRUKTURERT OUTPUT:"""

    try:
        start_time = time.time()
        response = ollama.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.3, 'num_predict': 2000}
        )
        summary = response['message']['content']
        elapsed = time.time() - start_time
        print(f"[ENHANCE] LLM done in {elapsed:.1f}s, generated {len(summary)} chars")
        
        return jsonify({
            'success': True,
            'enhanced_text': summary,
            'original_chars': len(pdf_text),
            'enhanced_chars': len(summary),
            'truncated': truncated,
            'enhance_time': round(elapsed, 1)
        })
    except Exception as e:
        print(f"[ENHANCE] LLM Error: {e}")
        return jsonify({'success': False, 'error': f'LLM feil: {str(e)}'}), 500


# Legacy endpoint - kept for backwards compatibility
@app.route('/summarize-pdf', methods=['POST'])
@login_required
def summarize_pdf():
    """Legacy: Upload PDF, extract text, and use LLM to create structured vault entries."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Ingen fil lastet opp'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Ingen fil valgt'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'success': False, 'error': 'Kun PDF-filer støttes'}), 400
    
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        print(f"[SMART IMPORT] Extracting text from {filename}...")
        pdf_text, _ = extract_pdf_fast(file_path)  # Use fast extraction
        
        os.remove(file_path)
        
        if not pdf_text or len(pdf_text.strip()) < 100:
            return jsonify({'success': False, 'error': 'Kunne ikke trekke ut nok tekst fra PDF'}), 400
        
        max_chars = 15000
        if len(pdf_text) > max_chars:
            pdf_text = pdf_text[:max_chars] + "\n\n[... dokumentet fortsetter ...]"
            print(f"[SMART IMPORT] Truncated to {max_chars} chars")
        
        print(f"[SMART IMPORT] Extracted {len(pdf_text)} chars, sending to LLM...")
        
        prompt = f"""Du er en ekspert på å lage strukturert dokumentasjon for et Makerspace.

Les følgende dokumentasjon og lag strukturerte kunnskapsoppføringer i vårt vault-format.

VAULT-FORMAT REGLER:
1. Bruk "--- TITTEL ---" for hver seksjon
2. Start med nivå: NYBEGYNNER, INTERMEDIATE, ADVANCED, eller EKSPERT
3. Skriv konsist og praktisk - fokuser på HVA brukeren må vite
4. Inkluder feilsøking hvis relevant
5. Bruk norsk språk (tekniske termer kan være på engelsk)
6. Hver oppføring bør være 3-10 linjer

EKSEMPEL FORMAT:
--- NYBEGYNNER: Oppstartsprosedyre [Utstyrsnavn] ---
1. Slå på hovedstrøm
2. Vent til systemet starter
3. [flere steg...]

--- INTERMEDIATE: Vanlige innstillinger ---
Speed: 50-100 for [materiale]
Power: 80% for [tykkelse]

--- FEILSØKING: [Problem] ---
Symptom: Beskrivelse av problemet
Årsak: Hvorfor det skjer
Løsning: Hvordan fikse det

DOKUMENTASJON Å BEHANDLE:
{pdf_text}

GENERER VAULT-OPPFØRINGER:"""

        # Call LLM
        try:
            response = ollama.chat(
                model='llama3',  # Use llama3 for better instruction following
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3}  # Lower temp for more consistent output
            )
            summary = response['message']['content']
            print(f"[SMART IMPORT] LLM generated {len(summary)} chars")
        except Exception as e:
            # Fallback to normistral if llama3 not available
            print(f"[SMART IMPORT] llama3 failed, trying normistral: {e}")
            try:
                response = ollama.chat(
                    model='hf.co/norallm/normistral-7b-warm-instruct',
                    messages=[{'role': 'user', 'content': prompt}]
                )
                summary = response['message']['content']
            except Exception as e2:
                return jsonify({'success': False, 'error': f'LLM feil: {str(e2)}'}), 500
        
        return jsonify({
            'success': True,
            'filename': filename,
            'original_chars': len(pdf_text),
            'summary': summary
        })
        
    except Exception as e:
        print(f"[SMART IMPORT] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/approve-summary', methods=['POST'])
@login_required  
def approve_summary():
    """Add approved LLM-generated summary to vault."""
    data = request.get_json()
    
    if not data or 'content' not in data:
        return jsonify({'success': False, 'error': 'Ingen innhold å legge til'}), 400
    
    content = data.get('content', '').strip()
    
    if not content:
        return jsonify({'success': False, 'error': 'Tomt innhold'}), 400
    
    try:
        # Append directly to vault (already formatted)
        with open(VAULT_FILE, 'a', encoding='utf-8') as f:
            f.write('\n\n')  # Separator
            f.write(content)
            f.write('\n')
        
        # Count approximate "chunks" added (sections)
        sections = content.count('---')
        
        return jsonify({
            'success': True,
            'message': f'Lagt til ~{sections//2} seksjoner i kunnskapsbasen',
            'stats': get_vault_stats()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


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
    load_json_knowledge()
    load_vault()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
