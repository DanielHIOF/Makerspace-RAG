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

# Database imports
from models import db, Component, search_components_db, get_all_hylleplasser
from db_config import DATABASE_URI


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
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'json', 'md', 'csv', 'html', 'htm', 'xlsx'}
VAULT_FILE = 'vault.txt'
CHUNK_SIZE = 1000

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,  # Check connection before using (auto-reconnect)
    'pool_recycle': 300,    # Recycle connections after 5 minutes
    'pool_size': 5,         # Number of connections to keep
    'max_overflow': 10,     # Allow up to 10 extra connections when busy
}
db.init_app(app)

# Initialize database tables on app startup (always runs)
with app.app_context():
    try:
        db.create_all()
        # Test the connection
        db.session.execute(db.text('SELECT 1'))
        print("  [DB] MariaDB connected and tables ready")
    except Exception as e:
        print(f"  [DB] ERROR: {e}")
        print("  [DB] Make sure MariaDB is running: net start MariaDB")

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
knowledge_components = {}
knowledge_kodeeksempler = {}
knowledge_prosessflyt = {}
knowledge_prosjektskalering = {}
knowledge_prosjektideer = {}


def load_json_knowledge():
    """Load structured knowledge from JSON files."""
    global knowledge_utstyr, knowledge_regler, knowledge_rom, knowledge_ressurser, knowledge_components, knowledge_kodeeksempler, knowledge_prosessflyt, knowledge_prosjektskalering, knowledge_prosjektideer
    
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
    
    try:
        with open(os.path.join(knowledge_dir, 'components.json'), 'r', encoding='utf-8') as f:
            knowledge_components = json.load(f)
        print(f"  Loaded components.json")
    except Exception as e:
        print(f"  Warning: Could not load components.json: {e}")
        knowledge_components = {}
    
    try:
        with open(os.path.join(knowledge_dir, 'kodeeksempler.json'), 'r', encoding='utf-8') as f:
            knowledge_kodeeksempler = json.load(f)
        print(f"  Loaded kodeeksempler.json")
    except Exception as e:
        print(f"  Warning: Could not load kodeeksempler.json: {e}")
        knowledge_kodeeksempler = {}
    
    try:
        with open(os.path.join(knowledge_dir, 'prosessflyt.json'), 'r', encoding='utf-8') as f:
            knowledge_prosessflyt = json.load(f)
        print(f"  Loaded prosessflyt.json")
    except Exception as e:
        print(f"  Warning: Could not load prosessflyt.json: {e}")
        knowledge_prosessflyt = {}
    
    try:
        with open(os.path.join(knowledge_dir, 'prosjektskalering.json'), 'r', encoding='utf-8') as f:
            knowledge_prosjektskalering = json.load(f)
        print(f"  Loaded prosjektskalering.json")
    except Exception as e:
        print(f"  Warning: Could not load prosjektskalering.json: {e}")
        knowledge_prosjektskalering = {}
    
    try:
        with open(os.path.join(knowledge_dir, 'prosjektideer.json'), 'r', encoding='utf-8') as f:
            knowledge_prosjektideer = json.load(f)
        print(f"  Loaded prosjektideer.json")
    except Exception as e:
        print(f"  Warning: Could not load prosjektideer.json: {e}")
        knowledge_prosjektideer = {}


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


def search_components(query):
    """Search for components by name or location - NOW USES DATABASE."""
    results = search_components_db(query, limit=20)
    
    if not results:
        return ""
    
    lines = [f"KOMPONENTER FUNNET ({len(results)} treff):"]
    lines.append("VIKTIG: List DISSE komponentene, ikke andre!")
    lines.append("")
    
    for comp in results[:15]:
        line = f"- {comp.name} @ {comp.hylleplass}"
        if comp.antall and comp.antall > 0:
            line += f" ({comp.antall} stk)"
        if comp.restock:
            line += " [TRENGER RESTOCK]"
        lines.append(line)
    
    if len(results) > 15:
        lines.append(f"... og {len(results) - 15} flere")
    
    return "\n".join(lines)


def get_all_components_summary():
    """Get a summary of all available components - NOW USES DATABASE."""
    total = Component.query.count()
    
    if total == 0:
        return "Ingen komponenter registrert ennå."
    
    lines = ["TILGJENGELIGE KOMPONENTER VED MAKERSPACE HiØF:", ""]
    
    # Group by hylleplass
    locations = db.session.query(Component.hylleplass).distinct().order_by(Component.hylleplass).all()
    
    shown = 0
    for loc in locations[:10]:  # Show first 10 locations
        loc_name = loc[0]
        comps = Component.query.filter(Component.hylleplass == loc_name).limit(5).all()
        if comps:
            lines.append(f"{loc_name}:")
            for comp in comps:
                lines.append(f"  - {comp.name}")
                shown += 1
            count = Component.query.filter(Component.hylleplass == loc_name).count()
            if count > 5:
                lines.append(f"  ... og {count - 5} flere")
            lines.append("")
    
    lines.append(f"Totalt: {total} komponenter på {len(locations)} hylleplasser.")
    lines.append("Spør om spesifikke komponenter for mer info!")
    
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
    """Expand Norwegian query with English synonyms and equipment names for better TF-IDF matching."""
    # First, add equipment-specific keywords from JSON
    equipment_keywords = []
    if knowledge_utstyr and 'categories' in knowledge_utstyr:
        query_lower = query.lower()
        for category_name, category_data in knowledge_utstyr['categories'].items():
            for equipment in category_data.get('equipment', []):
                equipment_name = equipment.get('name', '').lower()
                equipment_id = equipment.get('id', '').lower()
                keywords = equipment.get('keywords_no', []) + equipment.get('keywords_en', [])
                
                # If query mentions this equipment, boost with all its keywords
                if (equipment_name in query_lower or 
                    equipment_id in query_lower or
                    any(kw.lower() in query_lower for kw in keywords)):
                    equipment_keywords.extend(keywords)
                    # Add equipment name variations
                    if 'prusa' in equipment_name:
                        equipment_keywords.extend(['prusa', 'prusa3d', 'prusaslicer'])
                    if 'mini' in equipment_name:
                        equipment_keywords.append('mini')
                    if 'mk3' in equipment_name or 'mk3s' in equipment_name:
                        equipment_keywords.extend(['mk3', 'mk3s', 'mk3s+'])
                    if 'ultimaker' in equipment_name:
                        equipment_keywords.extend(['ultimaker', 'cura'])
                    if 'voron' in equipment_name:
                        equipment_keywords.extend(['voron', 'corexy'])
    
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
        
        # 3D printing - enhanced with Prusa-specific terms
        'prusa': 'prusa prusa3d prusaslicer prusa mini prusa mk3 prusa mk3s prusa i3',
        'printer': 'printer printing 3d print prusa ultimaker voron',
        '3d': '3d print printer printing prusa prusaslicer',
        'printe': 'print printing 3d prusa',
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
        'slicer': 'slicer slicing cura prusaslicer prusa slicer',
        'prusaslicer': 'prusaslicer prusa slicer slicing',
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
    
    # Add equipment keywords first (high priority)
    if equipment_keywords:
        expanded += ' ' + ' '.join(set(equipment_keywords))
    
    # Then add general expansions
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
    
    # Boost chunks that contain specific equipment names
    if knowledge_utstyr and 'categories' in knowledge_utstyr:
        query_lower = query.lower()
        equipment_boost_terms = []
        
        # Extract equipment names from query
        for category_name, category_data in knowledge_utstyr['categories'].items():
            for equipment in category_data.get('equipment', []):
                equipment_name = equipment.get('name', '').lower()
                equipment_id = equipment.get('id', '').lower()
                
                # Check if query mentions this equipment
                if (equipment_name in query_lower or 
                    equipment_id in query_lower or
                    any(kw.lower() in query_lower for kw in equipment.get('keywords_no', []) + equipment.get('keywords_en', []))):
                    # Add equipment name and variations to boost terms
                    equipment_boost_terms.append(equipment_name)
                    equipment_boost_terms.append(equipment_id)
                    # Add name parts (e.g., "prusa mini" -> boost "prusa" and "mini")
                    equipment_boost_terms.extend(equipment_name.split())
        
        # Boost similarity scores for chunks containing equipment terms
        if equipment_boost_terms:
            for i in range(len(similarities)):
                chunk_lower = vault_content[i].lower()
                boost_count = sum(1 for term in equipment_boost_terms if term in chunk_lower)
                if boost_count > 0:
                    # Boost by up to 0.3 (30%) based on number of matching terms
                    boost = min(0.3, boost_count * 0.1)
                    similarities[i] += boost
    
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


def detect_category_mode(query):
    """Detect category mode from slash commands (/prusa, /cnc, /laser, etc.)."""
    query_lower = query.lower()
    
    category_modes = {
        '/prusa': {
            'tool_filter': '3d_printer',
            'instruction': 'FOKUS PÅ PRUSA: Brukeren spør om Prusa-printere spesifikt. Prioriter informasjon om Prusa Mini+, Prusa MK3s, PrusaSlicer, og Prusa-spesifikke innstillinger. Ignorer informasjon om andre printermerker med mindre det er direkte relevant.',
            'boost_keywords': ['prusa', 'prusaslicer', 'prusa mini', 'prusa mk3', 'prusa mk3s']
        },
        '/3d': {
            'tool_filter': '3d_printer',
            'instruction': 'FOKUS PÅ 3D-PRINTING: Brukeren spør om 3D-printing generelt. Inkluder informasjon om alle typer 3D-printere, filament, slicer-programmer, og 3D-printing prosesser.',
            'boost_keywords': ['3d', 'print', 'printer', 'filament', 'slicer']
        },
        '/laser': {
            'tool_filter': 'laserkutter',
            'instruction': 'FOKUS PÅ LASERKUTTING: Brukeren spør om laserkutting. Prioriter informasjon om Epilog, Glowforge, laserkutting-prosesser, materialer, og laserkutting-innstillinger.',
            'boost_keywords': ['laser', 'kutt', 'graver', 'epilog', 'glowforge']
        },
        '/cnc': {
            'tool_filter': 'cnc',
            'instruction': 'FOKUS PÅ CNC-FRESING: Brukeren spør om CNC-fresing. Prioriter informasjon om Wegstr CNC, Avid CNC, CNC-fresing prosesser, og CNC-innstillinger.',
            'boost_keywords': ['cnc', 'fres', 'wegstr', 'avid', 'mill', 'router']
        },
        '/elektronikk': {
            'tool_filter': 'elektronikk',
            'instruction': 'FOKUS PÅ ELEKTRONIKK: Brukeren spør om elektronikk. Prioriter informasjon om Arduino, Raspberry Pi, komponenter, kretser, og elektronikk-prosjekter.',
            'boost_keywords': ['arduino', 'raspberry', 'elektronikk', 'krets', 'komponent']
        },
        '/lodding': {
            'tool_filter': 'lodding',
            'instruction': 'FOKUS PÅ LODDING: Brukeren spør om lodding. Prioriter informasjon om loddeutstyr, loddeteknikker, flux, og lodding-prosesser.',
            'boost_keywords': ['lodd', 'solder', 'loddekolbe', 'flux']
        }
    }
    
    for cmd, config in category_modes.items():
        if cmd in query_lower:
            return config
    
    return None


def detect_language(query):
    """Detect language preference from query. Default is Norwegian."""
    query_lower = query.lower()
    
    # Explicit English request
    if '/english' in query_lower or '/en' in query_lower:
        return 'english', "You MUST respond in English. Use English throughout your entire response."
    
    # Default to Norwegian (explicit /norsk or /no also works)
    return 'norwegian', "Du MÅ svare på NORSK. Bruk norsk språk i hele svaret. ALDRI svar på engelsk med mindre brukeren eksplisitt ber om det med /english"


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


def is_component_query(query):
    """Detect if query is asking about components/parts."""
    query_lower = query.lower()
    
    component_patterns = [
        'komponent', 'component', 'deler', 'parts',
        'motstand', 'resistor', 'kondensator', 'capacitor',
        'led', 'diode', 'transistor', 'ic', 'chip',
        'sensor', 'modul', 'module', 'arduino', 'esp32', 'esp8266',
        'raspberry', 'motor', 'servo', 'relay', 'relé',
        'kabel', 'wire', 'ledning', 'connector', 'kontakt',
        'skrue', 'screw', 'mutter', 'nut', 'bolt',
        'loddetinn', 'solder', 'tape', 'lim', 'glue',
        'har dere', 'finnes det', 'hvor finner jeg',
        'elektronikk-deler', 'electronics parts'
    ]
    
    return any(p in query_lower for p in component_patterns)


def detect_code_example_query(query):
    """Detect if user wants code example."""
    query_lower = query.lower()
    
    code_patterns = [
        r'kode.*eksempel', r'code.*example',
        r'hvordan.*koble', r'how.*connect',
        r'pin.*kobling', r'pin.*connection',
        r'arduino.*kode', r'esp32.*kode',
        r'vis.*kode', r'show.*code',
        r'koblingsskjema', r'wiring.*diagram',
        r'koblingsdiagram', r'wiring.*diagram',
        r'hvordan.*bruke', r'how.*use',
        r'eksempel.*kode', r'example.*code',
        r'vis.*diagram', r'show.*diagram',
        r'koble.*til', r'connect.*to'
    ]
    
    return any(re.search(p, query_lower, re.IGNORECASE) for p in code_patterns)


def generate_visualization_with_submodel(prompt, visualization_type='wiring_diagram'):
    """Generate visualization using sub-model (llama3.2:1b for speed) with concrete examples.
    Returns Mermaid diagram code or None if generation fails.
    """
    try:
        # Konkrete eksempler for hver diagramtype
        examples = {
            'wiring_diagram': """Eksempel på korrekt Mermaid wiring diagram:
graph LR
    A[Arduino Pin 9] -->|TRIG| B[HC-SR04]
    C[Arduino Pin 10] -->|ECHO| B
    D[5V] -->|VCC| B
    E[GND] -->|GND| B

Regler:
- Start med "graph LR" eller "graph TD"
- Bruk [tekst] for noder
- Bruk -->|label| for piler med tekst
- Hver linje skal være en kobling eller node-definisjon""",

            'process_flow': """Eksempel på korrekt Mermaid flowchart:
flowchart TD
    A[Design i CAD] --> B[Eksporter STL]
    B --> C[Importer i PrusaSlicer]
    C --> D{Forste lag OK?}
    D -->|Ja| E[Start print]
    D -->|Nei| F[Justér innstillinger]
    F --> C
    E --> G[Fjern print]
    style A fill:#E5A124
    style G fill:#4CAF50

Regler:
- Start med "flowchart TD" (top-down)
- Bruk [tekst] for prosesser
- Bruk {tekst} for beslutningspunkter
- Bruk -->|label| for piler med tekst
- Hver linje skal være en kobling eller node-definisjon""",

            'scaling_diagram': """Eksempel på korrekt Mermaid scaling diagram:
flowchart TD
    A[Prototype: 3D-print] --> B{Antall stk?}
    B -->|1-10| C[3D-print flere]
    B -->|10-100| D[Silikonform + resin]
    B -->|100+| E[Injeksjonsstøping]
    C --> F[Kostnad: Lav<br/>Tid: Rask]
    D --> G[Kostnad: Medium<br/>Tid: Medium]
    E --> H[Kostnad: Høy oppstart<br/>Tid: Lang oppstart]

Regler:
- Start med "flowchart TD"
- Bruk {tekst} for beslutningspunkter basert på antall
- Bruk -->|label| for piler med betingelser
- Hver linje skal være en kobling eller node-definisjon""",

            'printer_comparison': """Eksempel på korrekt Mermaid printer comparison:
graph LR
    A[Prusa Mini+<br/>Volume: 18x18x18cm<br/>Materials: PLA, PETG<br/>Level: Beginner]
    B[Prusa MK3s+<br/>Volume: 25x21x21cm<br/>Materials: PLA, PETG, TPU<br/>Level: Intermediate]
    A -.->|Sammenligning| B

Regler:
- Start med "graph LR" (left-right)
- Bruk [tekst<br/>tekst] for multi-line noder
- Bruk -.-> for sammenligningspiler
- Hver linje skal være en node eller kobling""",

            'mindmap': """Eksempel på korrekt Mermaid mindmap:
mindmap
  root((Arduino Prosjekter))
    Sensorer
      Temperatur
      Bevegelse
      Lys
    Aktuatorer
      Servo
      Motor
      LED
    Kombinert
      Værsstasjon
      Robot

Regler:
- Start med "mindmap"
- Bruk root((tekst)) for rot-node
- Bruk innrykk (2 spaces) for underkategorier
- Hver linje skal være en node eller underkategori"""
        }
        
        # Hent eksempel for denne diagramtypen
        example = examples.get(visualization_type, examples['wiring_diagram'])
        
        system_prompt = f"""Du er en ekspert på å generere Mermaid-diagrammer for makerspace-prosjekter.

VIKTIG: Du skal KUN returnere Mermaid-diagram kode, ingen forklaring eller annen tekst.

Her er et eksempel på korrekt syntaks for {visualization_type}:

{example}

VIKTIGE REGLER:
1. Start direkte med diagram-typen (graph, flowchart, mindmap) - INGEN markdown code blocks
2. Ikke inkluder ```mermaid eller ``` i responsen
3. Følg eksemplet nøyaktig når det gjelder syntaks
4. Hver linje skal være en node-definisjon eller kobling
5. Bruk norske navn hvor relevant

Returner KUN diagram-koden, start direkte med diagram-typen."""

        response = ollama.chat(
            model='llama3.2:1b',  # Fast small model for visualizations
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            options={'temperature': 0.2, 'num_predict': 400}  # Lower temperature for more consistent output
        )
        
        diagram_code = response['message']['content'].strip()
        
        # Clean up - remove markdown code blocks if present
        diagram_code = re.sub(r'```mermaid\s*\n?', '', diagram_code)
        diagram_code = re.sub(r'```\s*$', '', diagram_code)
        diagram_code = diagram_code.strip()
        
        # Remove any leading/trailing whitespace or newlines
        diagram_code = diagram_code.strip()
        
        return diagram_code
    except Exception as e:
        print(f"  [WARN] Sub-model visualization generation failed: {e}")
        return None


def search_code_examples(query):
    """Search for components/sensors that match the query - searches in components.json."""
    if not knowledge_components:
        return None
    
    query_lower = query.lower()
    
    # Search in components.json for sensors/modules
    if 'categories' in knowledge_components:
        # Check sensors category
        sensors_cat = knowledge_components.get('categories', {}).get('sensorer', {})
        sensors_list = sensors_cat.get('components', [])
        
        for sensor in sensors_list:
            name = sensor.get('name', '').lower()
            keywords = [k.lower() for k in sensor.get('keywords_no', []) + sensor.get('keywords_en', [])]
            
            if (name in query_lower or 
                any(kw in query_lower for kw in keywords) or
                any(kw in query_lower for kw in name.split())):
                return sensor.get('id'), sensor
        
        # Check modules category
        modules_cat = knowledge_components.get('categories', {}).get('moduler', {})
        modules_list = modules_cat.get('components', [])
        
        for module in modules_list:
            name = module.get('name', '').lower()
            keywords = [k.lower() for k in module.get('keywords_no', []) + module.get('keywords_en', [])]
            
            if (name in query_lower or 
                any(kw in query_lower for kw in keywords) or
                any(kw in query_lower for kw in name.split())):
                return module.get('id'), module
    
    return None


def generate_code_example_with_diagram(sensor_data, board_type='arduino', query=""):
    """Generate code example with wiring diagram using LLM sub-model.
    sensor_data: Component data from components.json
    board_type: 'arduino' or 'esp32'
    query: Original user query for context
    """
    if not sensor_data:
        return None
    
    sensor_name = sensor_data.get('name', 'sensor')
    sensor_location = sensor_data.get('location', '')
    
    # Build context for LLM
    context = f"""Komponent: {sensor_name}
Lokasjon: {sensor_location}
Brukerens spørsmål: {query}
Board type: {board_type.upper()}

Generer:
1. Et komplett Arduino/ESP32 kodeeksempel for å bruke denne sensoren
2. Et Mermaid-diagram som viser koblingsskjemaet (wiring diagram)
3. Liste over nødvendige biblioteker
4. Pin-koblinger

Format:
- Kodeeksempel skal være komplett og fungerende
- Mermaid-diagram skal vise fysisk kobling mellom board og sensor
- Bruk norsk for kommentarer og beskrivelser"""

    try:
        # Use main LLM to generate code example and get component info
        code_prompt = f"""Du er en ekspert på Arduino/ESP32 programmering.

Komponent: {sensor_name}
Board: {board_type.upper()}

Generer et komplett, fungerende kodeeksempel for å bruke denne sensoren.
Inkluder:
- Nødvendige includes/biblioteker
- Pin-definisjoner
- Setup() funksjon
- Loop() funksjon med sensor-avlesning
- Kommentarer på norsk

Returner KUN koden, ingen forklaring."""

        code_response = ollama.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': code_prompt}],
            options={'temperature': 0.5, 'num_predict': 500}
        )
        code = code_response['message']['content'].strip()
        
        # Clean code - remove markdown if present
        code = re.sub(r'```\w*\n?', '', code)
        code = re.sub(r'```\s*$', '', code)
        code = code.strip()
        
        # Generate wiring diagram with sub-model
        diagram_prompt = f"""Lag et Mermaid-diagram som viser koblingsskjemaet for:
- {sensor_name} sensor
- {board_type.upper()} mikrokontroller

Diagrammet skal vise:
- Fysisk kobling mellom {board_type.upper()} og sensoren
- Pin-koblinger (DATA, VCC, GND)
- Retning på signaler

Bruk graph LR (left-right) format med labels på koblingene."""

        diagram = generate_visualization_with_submodel(diagram_prompt, 'wiring_diagram')
        
        # Build response
        response_parts = []
        response_parts.append(f"Her er kodeeksempel og koblingsskjema for {sensor_name}:")
        response_parts.append("")
        
        # Wiring diagram
        if diagram:
            response_parts.append("**Koblingsskjema:**")
            response_parts.append("")
            response_parts.append("```mermaid")
            # Split diagram on \n to ensure proper line breaks for Mermaid
            diagram_lines = diagram.split('\n')
            response_parts.extend(diagram_lines)
            response_parts.append("```")
            response_parts.append("")
        
        # Code example
        if code:
            response_parts.append("**Kodeeksempel:**")
            response_parts.append("")
            response_parts.append(f"```{board_type}")
            response_parts.append(code)
            response_parts.append("```")
            response_parts.append("")
        
        # Add location info
        if sensor_location:
            response_parts.append(f"**Lokasjon:** {sensor_location}")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        print(f"  [ERROR] Failed to generate code example: {e}")
        return None


def detect_process_flow_query(query):
    """Detect if user wants process flow visualization."""
    query_lower = query.lower()
    
    process_patterns = [
        r'prosess.*flyt', r'process.*flow',
        r'steg.*for.*steg', r'step.*by.*step',
        r'hvordan.*prosess', r'how.*process',
        r'prosess.*3d', r'process.*3d',
        r'prosess.*laser', r'process.*laser',
        r'feils.*k', r'troubleshoot',
        r'hva.*er.*stegene', r'what.*are.*steps'
    ]
    
    return any(re.search(p, query_lower, re.IGNORECASE) for p in process_patterns)


def generate_process_flow(process_type, level='beginner', query=""):
    """Generate process flow diagram using LLM sub-model."""
    process_names = {
        '3d_printing': '3D-printing',
        'laser_cutting': 'Laserkutting',
        'troubleshooting_3d': 'Feilsøking 3D-printer'
    }
    
    process_name = process_names.get(process_type, process_type)
    
    # Build prompt for LLM
    prompt = f"""Lag et Mermaid flowchart-diagram som viser prosessflyt for {process_name}.

Nivå: {level}

Prosessflytet skal vise:
- Alle hovedsteg i prosessen
- Beslutningspunkter (diamant-form)
- Returløkker hvis noe går galt
- Start og slutt-punkter

Bruk flowchart TD (top-down) format.
For {process_type}:
"""
    
    if process_type == '3d_printing':
        prompt += """- Design i CAD
- Eksporter STL
- Importer i PrusaSlicer
- Slice til G-code
- Last til printer
- Start print
- Monitor første lag
- Fjern print"""
    elif process_type == 'laser_cutting':
        prompt += """- Design i Inkscape
- Forbered materiale
- Still inn fokus
- Start kutting
- Følg med hele tiden
- Fjern materiale"""
    elif process_type == 'troubleshooting_3d':
        prompt += """- Identifiser problem
- Sjekk første lag
- Juster innstillinger
- Prøv igjen"""
    
    try:
        diagram = generate_visualization_with_submodel(prompt, 'process_flow')
        
        if not diagram:
            return None
        
        # Build response
        response_parts = []
        response_parts.append(f"**Prosessflyt for {process_name}:**")
        response_parts.append("")
        
        # Process flow diagram
        response_parts.append("**Prosessflyt:**")
        response_parts.append("")
        response_parts.append("```mermaid")
        diagram_lines = diagram.split('\n')
        response_parts.extend(diagram_lines)
        response_parts.append("```")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        print(f"  [ERROR] Failed to generate process flow: {e}")
        return None


def detect_scaling_query(query):
    """Detect if user asks about project scaling."""
    query_lower = query.lower()
    
    scaling_patterns = [
        r'skalere.*prosjekt', r'scale.*project',
        r'prototype.*produksjon', r'prototype.*production',
        r'hvordan.*skalere', r'how.*scale',
        r'fra.*til.*stk', r'from.*to.*pcs',
        r'masseproduksjon', r'mass.*production',
        r'produksjon.*metode', r'production.*method'
    ]
    
    return any(re.search(p, query_lower, re.IGNORECASE) for p in scaling_patterns)


def generate_scaling_diagram(project_type, target_quantity=None, query=""):
    """Generate scaling diagram using LLM sub-model."""
    project_names = {
        '3d_printing': '3D-printing',
        'laser_cutting': 'Laserkutting'
    }
    
    project_name = project_names.get(project_type, project_type)
    
    # Build prompt for LLM
    prompt = f"""Lag et Mermaid flowchart-diagram som viser skaleringsveier fra prototype til produksjon for {project_name}.

Diagrammet skal vise:
- Startpunkt: Prototype (3D-print eller laserkutting)
- Beslutningspunkt basert på antall stk
- Forskjellige skaleringsveier:
  * 1-10 stk: Fortsett med samme metode
  * 10-100 stk: Medium-skala produksjon (silikonform, CNC, etc.)
  * 100+ stk: Masseproduksjon (injeksjonsstøping, stansing, etc.)
- Kostnad og tidsestimater for hver vei

Bruk flowchart TD (top-down) format med beslutningsdiamanter."""

    if target_quantity:
        prompt += f"\n\nBrukeren trenger {target_quantity} stk - hvilken vei anbefaler du?"
    
    try:
        diagram = generate_visualization_with_submodel(prompt, 'scaling_diagram')
        
        if not diagram:
            return None
        
        # Build response
        response_parts = []
        response_parts.append(f"**Skaleringsveier for {project_name}:**")
        response_parts.append("")
        
        # Scaling diagram
        response_parts.append("**Skaleringsveier:**")
        response_parts.append("")
        response_parts.append("```mermaid")
        diagram_lines = diagram.split('\n')
        response_parts.extend(diagram_lines)
        response_parts.append("```")
        response_parts.append("")
        
        # Add recommendation if quantity specified
        if target_quantity:
            try:
                qty = int(target_quantity)
                if qty <= 10:
                    recommendation = "Fortsett med samme metode (3D-print/laserkutting)"
                elif qty <= 100:
                    recommendation = "Vurder medium-skala produksjon (silikonform/CNC)"
                else:
                    recommendation = "Vurder masseproduksjon (injeksjonsstøping/stansing)"
                
                response_parts.append(f"**Anbefaling for {target_quantity} stk:**")
                response_parts.append(f"{recommendation}")
            except:
                pass
        
        return "\n".join(response_parts)
        
    except Exception as e:
        print(f"  [ERROR] Failed to generate scaling diagram: {e}")
        return None


def detect_printer_comparison_query(query):
    """Detect if user wants printer comparison."""
    query_lower = query.lower()
    
    comparison_patterns = [
        r'sammenlign.*printer', r'compare.*printer',
        r'forskjell.*printer', r'difference.*printer',
        r'hvilken.*printer', r'which.*printer',
        r'printer.*vs', r'printer.*versus',
        r'best.*printer', r'best.*for'
    ]
    
    return any(re.search(p, query_lower, re.IGNORECASE) for p in comparison_patterns)


def generate_printer_comparison(printer_ids, query=""):
    """Generate comparison diagram using LLM sub-model."""
    if not knowledge_utstyr or 'categories' not in knowledge_utstyr:
        return None
    
    printers = []
    category = knowledge_utstyr.get('categories', {}).get('3d_printing', {})
    equipment_list = category.get('equipment', [])
    
    # Find printers by ID
    for printer_id in printer_ids:
        for eq in equipment_list:
            if eq.get('id') == printer_id:
                printers.append(eq)
                break
    
    if len(printers) < 2:
        return None
    
    printer_names = [p.get('name', '') for p in printers]
    
    # Build context for LLM
    printer_info = []
    for printer in printers:
        info = f"{printer.get('name', '')}: Build volume {printer.get('build_volume', 'N/A')}, Materialer: {', '.join(printer.get('materials', []))}, Vanskelighetsnivå: {printer.get('difficulty', 'N/A')}"
        printer_info.append(info)
    
    prompt = f"""Lag et Mermaid-diagram som sammenligner disse 3D-printerne:

{chr(10).join(printer_info)}

Diagrammet skal vise:
- Forskjeller i build volume
- Material-kompatibilitet
- Vanskelighetsnivå
- Anbefaling basert på bruksområde

Bruk graph LR (left-right) format med noder for hver printer."""

    try:
        diagram = generate_visualization_with_submodel(prompt, 'printer_comparison')
        
        if not diagram:
            return None
        
        # Build response
        response_parts = []
        response_parts.append(f"**Sammenligning: {' vs '.join(printer_names)}**")
        response_parts.append("")
        
        # Comparison diagram
        response_parts.append("**Sammenligning:**")
        response_parts.append("")
        response_parts.append("```mermaid")
        diagram_lines = diagram.split('\n')
        response_parts.extend(diagram_lines)
        response_parts.append("```")
        response_parts.append("")
        
        # Detailed comparison
        response_parts.append("**Detaljert sammenligning:**")
        for printer in printers:
            name = printer.get('name', '')
            response_parts.append(f"**{name}:**")
            response_parts.append(f"- Build volume: {printer.get('build_volume', 'N/A')}")
            response_parts.append(f"- Materialer: {', '.join(printer.get('materials', []))}")
            response_parts.append(f"- Vanskelighetsnivå: {printer.get('difficulty', 'N/A')}")
            response_parts.append(f"- Tilgangsnivå: {printer.get('access_level', 'N/A')}")
            if printer.get('notes'):
                response_parts.append(f"- Notater: {printer.get('notes')}")
            response_parts.append("")
        
        # Recommendation
        response_parts.append("**Anbefaling:**")
        beginner_printers = [p for p in printers if p.get('difficulty') == 'beginner']
        if beginner_printers:
            response_parts.append(f"For nybegynnere: **{beginner_printers[0].get('name', '')}**")
        elif printers:
            response_parts.append(f"For avanserte brukere: **{printers[0].get('name', '')}**")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        print(f"  [ERROR] Failed to generate printer comparison: {e}")
        return None


def detect_idea_query(query):
    """Detect if user asks for project ideas."""
    query_lower = query.lower()
    
    idea_patterns = [
        r'ide.*prosjekt', r'project.*idea',
        r'hva.*kan.*lage', r'what.*can.*make',
        r'prosjekt.*ide', r'idea.*for',
        r'forslag.*prosjekt', r'suggest.*project',
        r'idemyldring', r'brainstorm'
    ]
    
    return any(re.search(p, query_lower, re.IGNORECASE) for p in idea_patterns)


def generate_idea_mindmap(technology, difficulty_level=None, query=""):
    """Generate mind map diagram using LLM sub-model."""
    tech_names = {
        'arduino': 'Arduino',
        '3d_printing': '3D-printing',
        'laser_cutting': 'Laserkutting'
    }
    
    tech_name = tech_names.get(technology, technology)
    
    # Build prompt for LLM
    prompt = f"""Lag et Mermaid mind map-diagram med prosjektideer for {tech_name}.

Mind mapet skal vise:
- Sentralt tema: {tech_name} Prosjekter
- Kategorier av ideer (sensorer, aktuatorer, kombinert, etc.)
- Spesifikke prosjekt-ideer under hver kategori
- Vanskelighetsnivå hvor relevant

Bruk mindmap format med root node og underkategorier."""

    if difficulty_level:
        prompt += f"\n\nFokuser på {difficulty_level} nivå prosjekter."
    
    try:
        diagram = generate_visualization_with_submodel(prompt, 'mindmap')
        
        if not diagram:
            return None
        
        # Build response
        response_parts = []
        response_parts.append(f"**Prosjektideer for {tech_name}:**")
        response_parts.append("")
        
        # Mind map diagram
        response_parts.append("**Prosjektideer:**")
        response_parts.append("")
        response_parts.append("```mermaid")
        diagram_lines = diagram.split('\n')
        response_parts.extend(diagram_lines)
        response_parts.append("```")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        print(f"  [ERROR] Failed to generate idea mindmap: {e}")
        return None


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
    category_mode = detect_category_mode(query)
    
    # Clean query of all command prefixes
    clean_query = re.sub(r'/(nybegynner|beginner|ekspert|expert|norsk|no|english|en|prusa|3d|laser|cnc|elektronikk|lodding)\s*', '', query, flags=re.IGNORECASE).strip()
    
    # Add category mode instruction if present
    category_instruction = ""
    if category_mode:
        category_instruction = category_mode.get('instruction', '')
    
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
        category_section = f"\n\nKATEGORI-MODUS:\n{category_instruction}" if category_instruction else ""
        
        system_prompt = f"""{base_role}{tool_hint}

RELEVANT INFORMASJON:
{context_text}

FERDIGHETSNIVÅ: {level_instruction}

SPRÅK: {language_instruction}{category_section}

KRITISK FOR KOMPONENTER:
- Bruk informasjonen fra "KOMPONENTER FUNNET" men SKRIV NATURLIG
- IKKE bruk "@" eller list-format fra konteksten
- GODT: "Vi har motstander på Komponentvegg, blant annet 10Ω, 15Ω og 100Ω."
- DÅRLIG: "10Ω @ Komponentvegg, 15Ω @ Komponentvegg..."
- Nevn lokasjonen ÉN gang, så list noen eksempler

FORMATERING:
- Bruk "-" for kulepunkt (ikke *)
- ALDRI bruk **bold** eller *italic* - det rendres ikke riktig
- Nummererte lister (1. 2. 3.) er OK når rekkefølge betyr noe
- Links er OK: [tekst](url)

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
    
    # Detect category mode from slash commands (e.g., /prusa, /cnc)
    category_mode = detect_category_mode(message)
    category_instruction = ""
    if category_mode:
        # Override detected_tool with category mode tool_filter
        detected_tool_from_category = category_mode.get('tool_filter')
        category_instruction = category_mode.get('instruction', '')
        print(f"  [CATEGORY] Kategori-modus aktivert: {detected_tool_from_category}")
    
    # Classify the query (use combined context for better detection)
    query_category = classify_query(combined_context)
    if not category_mode:  # Only auto-detect tool if no category mode set
        detected_tool = detect_tool(combined_context)
    else:
        detected_tool = detected_tool_from_category
    
    inventory_query = is_inventory_query(message)  # Keep this on current message only
    component_query = is_component_query(message)  # Check for component questions
    code_example_query = detect_code_example_query(message)  # Check for code example requests
    process_flow_query = detect_process_flow_query(message)  # Check for process flow requests
    scaling_query = detect_scaling_query(message)  # Check for scaling requests
    printer_comparison_query = detect_printer_comparison_query(message)  # Check for printer comparison
    idea_query = detect_idea_query(message)  # Check for project ideas
    
    print(f"  Kategori: {query_category}")
    if detected_tool:
        print(f"  Verktoy: {detected_tool}")
    if inventory_query:
        print(f"  Type: INVENTAR-SPØRSMÅL (bruker kun JSON)")
    if component_query:
        print(f"  Type: KOMPONENT-SPØRSMÅL (søker i components.json)")
    if code_example_query:
        print(f"  Type: KODEEKSEMPEL-SPØRSMÅL (søker i components.json, genererer med LLM)")
    if process_flow_query:
        print(f"  Type: PROSESSFLYT-SPØRSMÅL (søker i prosessflyt.json)")
    if scaling_query:
        print(f"  Type: PROSJEKTSKALERING-SPØRSMÅL (søker i prosjektskalering.json)")
    if printer_comparison_query:
        print(f"  Type: PRINTER-SAMMENLIGNING (søker i utstyr.json)")
    if idea_query:
        print(f"  Type: PROSJEKTIDEER-SPØRSMÅL (søker i prosjektideer.json)")
    
    # Build context from multiple sources
    context_parts = []
    
    # 0. For component queries: search components.json
    if component_query:
        # Try to find specific component
        comp_ctx = search_components(message)
        if comp_ctx:
            context_parts.append(comp_ctx)
            print(f"  [JSON] Fant komponenter som matcher spørringen")
        else:
            # No specific match, show summary
            comp_summary = get_all_components_summary()
            if comp_summary:
                context_parts.append(comp_summary)
                print(f"  [JSON] Lagt til komponentoversikt")
    
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
        if category_mode:
            print(f"  [BOOST] Bruker kategori-boost keywords: {category_mode.get('boost_keywords', [])}")
        search_start = time.time()
        # If category mode is active, boost search with category keywords
        if category_mode and category_mode.get('boost_keywords'):
            # Add boost keywords to query for better matching
            boosted_query = message + ' ' + ' '.join(category_mode.get('boost_keywords', []))
            relevant_chunks = search_vault(boosted_query, tool_filter=detected_tool)
        else:
            relevant_chunks = search_vault(message, tool_filter=detected_tool)
        search_time = time.time() - search_start
        print(f"  [OK] Fant {len(relevant_chunks)} relevante biter ({search_time:.2f}s)")
        
        if relevant_chunks:
            for i, chunk in enumerate(relevant_chunks[:3]):
                preview = chunk[:80].replace('\n', ' ')
                print(f"    {i+1}. {preview}...")
            context_parts.append("DOKUMENTASJON:\n" + "\n\n".join(relevant_chunks))
    
    # Check for code example request - return directly if found
    if code_example_query:
        code_result = search_code_examples(message)
        if code_result:
            sensor_id, sensor_data = code_result
            # Detect board type from query
            board_type = 'esp32' if 'esp32' in message.lower() else 'arduino'
            print(f"  [CODE] Genererer kodeeksempel med LLM for {sensor_id} ({board_type})")
            code_response = generate_code_example_with_diagram(sensor_data, board_type, message)
            if code_response:
                print(f"  [CODE] Returnerer kodeeksempel for {sensor_id} ({board_type})")
                return jsonify({
                    'response': code_response,
                    'summary': existing_summary
                })
        else:
            # If no specific sensor found but user asked for wiring diagram, generate generic one
            if 'koblingsdiagram' in message.lower() or 'koblingsskjema' in message.lower() or 'wiring' in message.lower():
                print(f"  [CODE] Ingen spesifikk sensor funnet, genererer generisk koblingsdiagram")
                # Try to extract sensor name from query
                query_lower = message.lower()
                sensor_name = "sensor"
                if 'hc-sr04' in query_lower or 'ultralyd' in query_lower:
                    sensor_name = "HC-SR04 Ultralydsensor"
                elif 'dht11' in query_lower or 'temperatur' in query_lower:
                    sensor_name = "DHT11 Temperatursensor"
                elif 'pir' in query_lower or 'bevegelse' in query_lower:
                    sensor_name = "PIR Bevegelsessensor"
                
                board_type = 'esp32' if 'esp32' in query_lower else 'arduino'
                
                # Generate just the wiring diagram
                diagram_prompt = f"""Lag et Mermaid-diagram som viser koblingsskjemaet for:
- {sensor_name}
- {board_type.upper()} mikrokontroller

Diagrammet skal vise:
- Fysisk kobling mellom {board_type.upper()} og sensoren
- Pin-koblinger (DATA, VCC, GND)
- Retning på signaler

Bruk graph LR (left-right) format med labels på koblingene."""
                
                diagram = generate_visualization_with_submodel(diagram_prompt, 'wiring_diagram')
                if diagram:
                    response_text = f"Her er koblingsskjema for {sensor_name} til {board_type.upper()}:\n\n"
                    response_text += "**Koblingsskjema:**\n\n"
                    response_text += "```mermaid\n"
                    response_text += diagram
                    response_text += "\n```"
                    print(f"  [CODE] Returnerer generisk koblingsdiagram")
                    return jsonify({
                        'response': response_text,
                        'summary': existing_summary
                    })
    
    # Check for process flow request - return directly if found
    if process_flow_query:
        # Detect process type from query
        process_type = None
        query_lower = message.lower()
        if '3d' in query_lower or 'print' in query_lower:
            process_type = '3d_printing'
        elif 'laser' in query_lower or 'kutt' in query_lower:
            process_type = 'laser_cutting'
        elif 'feils' in query_lower or 'troubleshoot' in query_lower or 'problem' in query_lower:
            process_type = 'troubleshooting_3d'
        
        if process_type:
            # Detect level from query
            level = 'beginner' if 'nybegynner' in query_lower or 'beginner' in query_lower else 'normal'
            process_response = generate_process_flow(process_type, level, message)
            if process_response:
                print(f"  [PROCESS] Returnerer prosessflyt for {process_type} ({level})")
                return jsonify({
                    'response': process_response,
                    'summary': existing_summary
                })
    
    # Check for scaling request - return directly if found
    if scaling_query:
        # Detect project type from query
        project_type = None
        query_lower = message.lower()
        if '3d' in query_lower or 'print' in query_lower:
            project_type = '3d_printing'
        elif 'laser' in query_lower or 'kutt' in query_lower:
            project_type = 'laser_cutting'
        
        if project_type:
            # Try to extract target quantity from query
            qty_match = re.search(r'(\d+)\s*(stk|pcs|enheter|units)', query_lower)
            target_quantity = qty_match.group(1) if qty_match else None
            
            scaling_response = generate_scaling_diagram(project_type, target_quantity, message)
            if scaling_response:
                print(f"  [SCALING] Returnerer skaleringsdiagram for {project_type}")
                return jsonify({
                    'response': scaling_response,
                    'summary': existing_summary
                })
    
    # Check for printer comparison request - return directly if found
    if printer_comparison_query:
        query_lower = message.lower()
        printer_ids = []
        
        # Try to extract printer names/IDs from query
        if 'mini' in query_lower or 'prusa-mini' in query_lower:
            printer_ids.append('prusa-mini')
        if 'mk3' in query_lower or 'prusa-mk3s' in query_lower:
            printer_ids.append('prusa-mk3s')
        if 'ultimaker' in query_lower:
            printer_ids.append('ultimaker-3-extended')
        if 'voron' in query_lower:
            if '0.1' in query_lower or '0' in query_lower:
                printer_ids.append('voron-0.1')
            else:
                printer_ids.append('voron-2.4r2')
        
        # If no specific printers mentioned, compare common ones
        if not printer_ids:
            printer_ids = ['prusa-mini', 'prusa-mk3s']
        
        if len(printer_ids) >= 2:
            comparison_response = generate_printer_comparison(printer_ids[:3], message)  # Max 3 printers
            if comparison_response:
                print(f"  [COMPARISON] Returnerer sammenligning for {printer_ids}")
                return jsonify({
                    'response': comparison_response,
                    'summary': existing_summary
                })
    
    # Check for project ideas request - return directly if found
    if idea_query:
        query_lower = message.lower()
        technology = None
        
        # Detect technology from query
        if 'arduino' in query_lower or 'elektronikk' in query_lower or 'sensor' in query_lower:
            technology = 'arduino'
        elif '3d' in query_lower or 'print' in query_lower:
            technology = '3d_printing'
        elif 'laser' in query_lower or 'kutt' in query_lower:
            technology = 'laser_cutting'
        
        # Default to arduino if no specific technology
        if not technology:
            technology = 'arduino'
        
        # Detect difficulty level
        difficulty = None
        if 'nybegynner' in query_lower or 'beginner' in query_lower:
            difficulty = 'beginner'
        elif 'ekspert' in query_lower or 'advanced' in query_lower:
            difficulty = 'advanced'
        
        idea_response = generate_idea_mindmap(technology, difficulty, message)
        if idea_response:
            print(f"  [IDEAS] Returnerer prosjektideer for {technology}")
            return jsonify({
                'response': idea_response,
                'summary': existing_summary
            })
    
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
            
            # Auto-detect file type and handle accordingly
            if ext == 'xlsx':
                # XLSX files should use the extract-xlsx endpoint, not generic upload
                # For now, skip them here - they'll be handled by the frontend
                errors.append(f'{filename}: XLSX files should be uploaded via the Excel import section')
                os.remove(file_path)
                continue
            elif ext == 'pdf':
                # PDF files can be processed here or via extract-pdf endpoint
                # For generic upload, extract text and chunk it
                text = process_file(file_path, ext)
            else:
                # Other file types (txt, json, md, etc.)
                text = process_file(file_path, ext)
            
            if not text:
                os.remove(file_path)
                errors.append(f'{filename}: no text extracted')
                continue
            
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
# Fast PDF Import - OCR-based extraction
# =============================================================================
def extract_pdf_fast(file_path):
    """
    PDF text extraction using OCR (EasyOCR).
    
    All PDFs are processed with OCR for consistent results.
    Works well for both scanned documents and text-based PDFs.
    """
    import fitz  # PyMuPDF
    
    # Get page count
    doc = fitz.open(file_path)
    total_pages = len(doc)
    doc.close()
    
    print(f"  [PDF] Extracting {total_pages} pages using OCR...")
    
    try:
        text = extract_pdf_ocr(file_path)
        if text and len(text.strip()) > 50:
            print(f"  [PDF] OCR extracted {len(text)} chars from {total_pages} pages")
            return text, total_pages
        else:
            raise Exception("OCR returned insufficient text")
    except Exception as e:
        print(f"  [PDF] OCR failed: {e}")
        raise Exception(f"PDF extraction failed: {e}")


def extract_pdf_ocr(file_path):
    """
    OCR extraction for image-based PDFs using EasyOCR.
    Converts PDF pages to images and runs OCR on each.
    """
    import fitz  # PyMuPDF for PDF to image
    import easyocr
    import numpy as np
    from PIL import Image
    import io
    
    print(f"  [OCR] Initializing EasyOCR (first run downloads models ~100MB)...")
    
    # Initialize reader with Norwegian and English
    reader = easyocr.Reader(['no', 'en'], gpu=False)  # CPU mode for compatibility
    
    doc = fitz.open(file_path)
    all_text = []
    
    for page_num, page in enumerate(doc):
        print(f"  [OCR] Processing page {page_num + 1}/{len(doc)}...")
        
        # Render page to image (higher DPI = better OCR)
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to numpy array for easyocr
        img_array = np.array(img)
        
        # Run OCR
        results = reader.readtext(img_array)
        
        # Extract text from results
        page_text = []
        for (bbox, text, confidence) in results:
            if confidence > 0.3:  # Filter low confidence results
                page_text.append(text)
        
        if page_text:
            all_text.append(f"--- Page {page_num + 1} ---")
            all_text.append("\n".join(page_text))
    
    doc.close()
    
    return "\n\n".join(all_text)


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


# =============================================================================
# XLSX Import - Component/Equipment Lists from Excel
# =============================================================================

def parse_xlsx_to_equipment(file_path):
    """
    Parse XLSX file and convert rows to component entries.
    Auto-detects column mapping based on header names.
    """
    import openpyxl
    
    wb = openpyxl.load_workbook(file_path, data_only=True)
    sheet = wb.active
    
    # Get headers from first row
    headers = []
    for cell in sheet[1]:
        headers.append(str(cell.value).lower().strip() if cell.value else '')
    
    print(f"  [XLSX] Found headers: {headers}")
    
    # Column mapping - maps various header names to our component fields
    COLUMN_MAPPINGS = {
        'id': ['id', 'product_id', 'produktid', 'varenr', 'item_id', 'sku', 'part_number', 'partnumber'],
        'name': ['name', 'navn', 'product', 'produkt', 'description', 'beskrivelse', 'item', 'component', 'komponent', 'del'],
        'location': ['location', 'lokasjon', 'sted', 'rom', 'room', 'placement', 'plassering', 'shelf', 'hylle', 'drawer', 'skuff', 'boks', 'box'],
        'category': ['category', 'kategori', 'type', 'gruppe', 'group', 'class', 'klasse'],
        'notes': ['notes', 'notater', 'kommentar', 'comment', 'remarks', 'merknad', 'info'],
    }
    
    # Find which columns map to which fields
    column_map = {}
    for field, possible_names in COLUMN_MAPPINGS.items():
        for i, header in enumerate(headers):
            if header in possible_names:
                column_map[field] = i
                break
    
    print(f"  [XLSX] Column mapping: {column_map}")
    
    # Parse rows
    equipment_list = []
    for row_num, row in enumerate(sheet.iter_rows(min_row=2, values_only=True), start=2):
        # Skip empty rows
        if not any(row):
            continue
        
        # Build component entry
        entry = {
            'id': '',
            'name': '',
            'location': '',
            'category': 'other',
            'notes': '',
            'keywords_no': [],
            'keywords_en': [],
            '_row': row_num  # Track source row for debugging
        }
        
        # Map columns to fields
        for field, col_idx in column_map.items():
            if col_idx < len(row) and row[col_idx] is not None:
                value = str(row[col_idx]).strip()
                entry[field] = value
        
        # Generate ID if not present
        if not entry['id'] and entry['name']:
            # Create slug from name
            import re
            slug = re.sub(r'[^a-z0-9]+', '-', entry['name'].lower())
            slug = slug.strip('-')[:50]
            entry['id'] = slug
        
        # Generate keywords from name
        if entry['name']:
            words = entry['name'].lower().split()
            entry['keywords_no'] = [w for w in words if len(w) > 2]
            entry['keywords_en'] = entry['keywords_no']  # Same for now
        
        # Only add if we have at least a name
        if entry['name']:
            equipment_list.append(entry)
    
    wb.close()
    return equipment_list, headers, column_map


def check_component_duplicates(new_items, existing_components):
    """
    Check for duplicates based on id, name, and location.
    Returns: (items_to_add, duplicates_skipped)
    """
    # Build lookup of existing items: key = (id, name_lower, location_lower)
    existing_keys = set()
    
    for category in existing_components.get('categories', {}).values():
        for item in category.get('components', []):
            key = (
                item.get('id', '').lower(),
                item.get('name', '').lower(),
                item.get('location', '').lower()
            )
            existing_keys.add(key)
    
    items_to_add = []
    duplicates_skipped = []
    
    for item in new_items:
        key = (
            item.get('id', '').lower(),
            item.get('name', '').lower(),
            item.get('location', '').lower()
        )
        
        # Check for exact match (all three match)
        if key in existing_keys:
            duplicates_skipped.append(item)
        else:
            # Also check partial matches (same id OR same name+location)
            is_duplicate = False
            for ex_key in existing_keys:
                # Same ID
                if key[0] and key[0] == ex_key[0]:
                    duplicates_skipped.append({**item, '_duplicate_reason': f'Same ID: {key[0]}'})
                    is_duplicate = True
                    break
                # Same name AND location
                if key[1] and key[2] and key[1] == ex_key[1] and key[2] == ex_key[2]:
                    duplicates_skipped.append({**item, '_duplicate_reason': f'Same name+location: {key[1]} @ {key[2]}'})
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                items_to_add.append(item)
    
    return items_to_add, duplicates_skipped


@app.route('/extract-xlsx', methods=['POST'])
@login_required
def extract_xlsx():
    """Parse XLSX and return equipment entries for preview."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Ingen fil lastet opp'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Ingen fil valgt'}), 400
    
    if not file.filename.lower().endswith('.xlsx'):
        return jsonify({'success': False, 'error': 'Kun XLSX-filer støttes (ikke .xls)'}), 400
    
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        print(f"[XLSX EXTRACT] Processing {filename}...")
        start_time = time.time()
        
        # Parse XLSX
        equipment_list, headers, column_map = parse_xlsx_to_equipment(file_path)
        
        # Load existing components for duplicate check
        components_path = 'knowledge/components.json'
        existing_components = {}
        if os.path.exists(components_path):
            with open(components_path, 'r', encoding='utf-8') as f:
                existing_components = json.load(f)
        
        # Check duplicates
        items_to_add, duplicates_skipped = check_component_duplicates(equipment_list, existing_components)
        
        os.remove(file_path)
        
        elapsed = time.time() - start_time
        print(f"[XLSX EXTRACT] Found {len(equipment_list)} items, {len(items_to_add)} new, {len(duplicates_skipped)} duplicates in {elapsed:.2f}s")
        
        if not equipment_list:
            return jsonify({'success': False, 'error': 'Ingen utstyr funnet i filen. Sjekk at første rad har kolonneoverskrifter.'}), 400
        
        return jsonify({
            'success': True,
            'filename': filename,
            'headers': headers,
            'column_map': column_map,
            'total_rows': len(equipment_list),
            'items_to_add': items_to_add,
            'duplicates_skipped': duplicates_skipped,
            'extract_time': round(elapsed, 2)
        })
        
    except Exception as e:
        print(f"[XLSX EXTRACT] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/approve-xlsx', methods=['POST'])
@login_required
def approve_xlsx():
    """Add approved component items to components.json, grouping by each item's category."""
    data = request.get_json()
    if not data or 'items' not in data:
        return jsonify({'success': False, 'error': 'Ingen komponenter å legge til'}), 400
    
    items = data['items']
    
    if not items:
        return jsonify({'success': False, 'error': 'Listen er tom'}), 400
    
    try:
        components_path = 'knowledge/components.json'
        
        # Load existing
        if os.path.exists(components_path):
            with open(components_path, 'r', encoding='utf-8') as f:
                components_data = json.load(f)
        else:
            components_data = {
                'version': '1.0',
                'last_updated': '',
                'description': 'Komponentoversikt for Makerspace HiØF',
                'categories': {}
            }
        
        # Group items by their category
        by_category = {}
        for item in items:
            cat = item.get('category', 'other').lower().strip()
            if not cat:
                cat = 'other'
            # Normalize category name (replace spaces with underscores)
            cat = cat.replace(' ', '_')
            if cat not in by_category:
                by_category[cat] = []
            # Clean item (remove internal fields)
            clean = {k: v for k, v in item.items() if not k.startswith('_')}
            by_category[cat].append(clean)
        
        # Add items to their respective categories
        for cat, cat_items in by_category.items():
            if cat not in components_data.get('categories', {}):
                components_data['categories'][cat] = {
                    'name_no': cat.replace('_', ' ').title(),
                    'name_en': cat.replace('_', ' ').title(),
                    'components': []
                }
            components_data['categories'][cat]['components'].extend(cat_items)
        
        components_data['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        
        # Save
        with open(components_path, 'w', encoding='utf-8') as f:
            json.dump(components_data, f, indent=2, ensure_ascii=False)
        
        # Reload knowledge
        load_json_knowledge()
        
        # Build summary of what was added
        categories_added = list(by_category.keys())
        total_added = len(items)
        print(f"[XLSX APPROVE] Added {total_added} items to categories: {categories_added}")
        
        return jsonify({
            'success': True,
            'message': f'La til {total_added} komponenter i {len(categories_added)} kategori(er)',
            'items_added': total_added,
            'categories': categories_added
        })
        
    except Exception as e:
        print(f"[XLSX APPROVE] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# Category Templates for Smart Import
# =============================================================================
CATEGORY_TEMPLATES = {
    'utstyr': {
        'file': 'knowledge/utstyr.json',
        'type': 'json',
        'prompt': '''Du skal lage en JSON-oppføring for et UTSTYR/VERKTØY i et Makerspace.

EKSEMPEL PÅ ØNSKET OUTPUT:
{{
  "id": "prusa-mini",
  "name": "Prusa Mini+",
  "location": "D1-044",
  "status": "active",
  "access_level": "course_makerspace",
  "difficulty": "beginner",
  "materials": ["PLA", "PETG"],
  "filament_diameter": "1.75mm",
  "build_volume": "180x180x180mm",
  "notes": "God for nybegynnere",
  "keywords_no": ["3d print", "printer", "prusa"],
  "keywords_en": ["3d print", "printer", "prusa"]
}}

TILGANGSNIVÅER (velg én):
- course_makerspace: Krever MakerSpace-kurs (D1-044)
- course_fablab: Krever FabLab HMS-kurs (D1-043)
- certification_required: Krever sertifisering fra labingeniør
- request_required: Må hentes fra labansvarlig
- staff_only: Kun labingeniører/studentassistenter

VANSKELIGHETSGRAD: beginner, intermediate, advanced

DOKUMENTET:
{text}

KONTEKST: {context}

OUTPUT (kun gyldig JSON, ingen forklaring):'''
    },
    'regler': {
        'file': 'knowledge/regler.json',
        'type': 'json',
        'prompt': '''Du skal lage JSON-oppføringer for HMS/SIKKERHETSREGLER i et Makerspace.

EKSEMPEL PÅ ØNSKET OUTPUT:
[
  {{
    "id": "rule-laser-001",
    "priority": "critical",
    "rule_no": "Følg med på hele jobben - laseren kan starte brann",
    "rule_en": "Monitor the entire job - laser can start fires",
    "applies_to": "laser_cutting"
  }},
  {{
    "id": "rule-laser-002",
    "priority": "high",
    "rule_no": "ALDRI kutt PVC eller vinyl - avgir giftig gass",
    "rule_en": "NEVER cut PVC or vinyl - releases toxic fumes",
    "applies_to": "laser_cutting"
  }}
]

PRIORITET: critical (livstruende), high (alvorlig), medium (viktig), low (anbefalt)
APPLIES_TO: all, 3d_printing, laser_cutting, cnc, electronics, woodworking

DOKUMENTET:
{text}

KONTEKST: {context}

OUTPUT (kun gyldig JSON-array, ingen forklaring):'''
    },
    'rom': {
        'file': 'knowledge/rom.json',
        'type': 'json',
        'prompt': '''Du skal lage en JSON-oppføring for et ROM/LOKALE i et Makerspace.

EKSEMPEL PÅ ØNSKET OUTPUT:
{{
  "id": "D1-044",
  "name_no": "MakerSpace",
  "name_en": "MakerSpace",
  "building": "D",
  "floor": 1,
  "description_no": "Hovedrom for prototyping og elektronikk",
  "description_en": "Main room for prototyping and electronics",
  "equipment_categories": ["3d_printing", "electronics"],
  "features": ["3D-printere", "Loddestasjoner", "Elektronikkarbeidsplasser"],
  "access": {{
    "requires_training": true,
    "booking_required": false,
    "open_hours": "Se booking-system"
  }}
}}

DOKUMENTET:
{text}

KONTEKST: {context}

OUTPUT (kun gyldig JSON, ingen forklaring):'''
    },
    'ressurser': {
        'file': 'knowledge/ressurser.json',
        'type': 'json',
        'prompt': '''Du skal lage JSON-oppføringer for LÆRINGSRESSURSER/LENKER for et Makerspace.

EKSEMPEL PÅ ØNSKET OUTPUT:
[
  {{
    "title": "Prusa Knowledge Base",
    "url": "https://help.prusa3d.com/",
    "language": "en",
    "level": "all",
    "description_no": "Offisiell Prusa dokumentasjon",
    "description_en": "Official Prusa documentation"
  }},
  {{
    "title": "Arduino Getting Started",
    "url": "https://www.arduino.cc/en/Guide",
    "language": "en",
    "level": "beginner",
    "description_no": "Offisiell Arduino begynnerguide",
    "description_en": "Official Arduino getting started guide"
  }}
]

NIVÅER: beginner, intermediate, advanced, all
SPRÅK: no, en

DOKUMENTET:
{text}

KONTEKST: {context}

OUTPUT (kun gyldig JSON-array, ingen forklaring):'''
    },
    'vault': {
        'file': 'vault.txt',
        'type': 'text',
        'prompt': '''Du skal lage strukturert KUNNSKAPSINNHOLD for et Makerspace.

FORMAT:
--- NIVÅ: Tittel ---
Innhold her...

NIVÅER: NYBEGYNNER, INTERMEDIATE, AVANSERT, EKSPERT, FEILSØKING

EKSEMPEL:
--- NYBEGYNNER: Første 3D-print steg-for-steg ---
1. Last ned en 3D-modell fra Printables.com
2. Åpne PrusaSlicer og importer filen
3. Velg riktig printer og materiale
4. Klikk "Slice now" og eksporter G-code
5. Sett kortet i printeren og start

--- FEILSØKING: Print løsner fra platen ---
Symptom: Hjørner løfter seg, warping
Årsaker:
- Skitten plate: Vask med isopropanol
- Feil Z-offset: Juster ned i små steg
- For lav bed-temp: Øk med 5-10°C

DOKUMENTET:
{text}

KONTEKST: {context}

OUTPUT (kun strukturert tekst, bruk norsk språk):'''
    }
}


@app.route('/enhance-pdf', methods=['POST'])
@login_required
def enhance_pdf():
    """Use LLM to structure/summarize extracted PDF text based on category."""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'success': False, 'error': 'Ingen tekst å behandle'}), 400
    
    pdf_text = data.get('text', '').strip()
    doc_context = data.get('context', '').strip()
    category = data.get('category', 'vault').strip()
    
    if len(pdf_text) < 50:
        return jsonify({'success': False, 'error': 'For lite tekst å behandle'}), 400
    
    if category not in CATEGORY_TEMPLATES:
        return jsonify({'success': False, 'error': f'Ukjent kategori: {category}'}), 400
    
    template = CATEGORY_TEMPLATES[category]
    
    # Larger limit for llama3 - it can handle more
    max_chars = 8000
    truncated = False
    if len(pdf_text) > max_chars:
        pdf_text = pdf_text[:max_chars]
        truncated = True
        print(f"[ENHANCE] Truncated to {max_chars} chars")
    
    print(f"[ENHANCE] Category: {category} | Context: '{doc_context}' | Sending {len(pdf_text)} chars to llama3...")
    
    # Build category-specific prompt
    prompt = template['prompt'].format(text=pdf_text, context=doc_context)

    try:
        start_time = time.time()
        # Use llama3 8B for better quality output
        response = ollama.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.3, 'num_predict': 2000}
        )
        summary = response['message']['content']
        elapsed = time.time() - start_time
        print(f"[ENHANCE] LLM done in {elapsed:.1f}s, generated {len(summary)} chars")
        
        # Clean up JSON output if it's a JSON category
        if template['type'] == 'json':
            # Try to extract just the JSON part
            summary = summary.strip()
            if summary.startswith('```json'):
                summary = summary[7:]
            if summary.startswith('```'):
                summary = summary[3:]
            if summary.endswith('```'):
                summary = summary[:-3]
            summary = summary.strip()
        
        return jsonify({
            'success': True,
            'enhanced_text': summary,
            'original_chars': len(pdf_text),
            'enhanced_chars': len(summary),
            'truncated': truncated,
            'enhance_time': round(elapsed, 1),
            'category': category,
            'output_type': template['type'],
            'target_file': template['file']
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
    """Add approved LLM-generated content to appropriate file based on category."""
    data = request.get_json()
    
    if not data or 'content' not in data:
        return jsonify({'success': False, 'error': 'Ingen innhold å legge til'}), 400
    
    content = data.get('content', '').strip()
    category = data.get('category', 'vault').strip()
    
    if not content:
        return jsonify({'success': False, 'error': 'Tomt innhold'}), 400
    
    if category not in CATEGORY_TEMPLATES:
        return jsonify({'success': False, 'error': f'Ukjent kategori: {category}'}), 400
    
    template = CATEGORY_TEMPLATES[category]
    target_file = template['file']
    
    try:
        if template['type'] == 'text':
            # Append to vault.txt
            with open(VAULT_FILE, 'a', encoding='utf-8') as f:
                f.write('\n\n')
                f.write(content)
                f.write('\n')
            
            sections = content.count('---')
            message = f'Lagt til ~{sections//2} seksjoner i vault.txt'
            
        else:
            # Handle JSON categories
            try:
                new_data = json.loads(content)
            except json.JSONDecodeError as je:
                return jsonify({
                    'success': False, 
                    'error': f'Ugyldig JSON: {str(je)}. Prøv å strukturere på nytt.'
                }), 400
            
            # Load existing JSON file
            existing_data = {}
            if os.path.exists(target_file):
                with open(target_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            
            # Merge strategy depends on category
            if category == 'utstyr':
                # Add to appropriate category
                if 'categories' not in existing_data:
                    existing_data['categories'] = {}
                
                # Try to determine category from the new data
                if isinstance(new_data, dict):
                    # Single equipment item - add to a generic category or detect from keywords
                    eq_category = 'other'
                    keywords = new_data.get('keywords_no', []) + new_data.get('keywords_en', [])
                    for kw in keywords:
                        if '3d' in kw.lower() or 'print' in kw.lower():
                            eq_category = '3d_printing'
                            break
                        elif 'laser' in kw.lower():
                            eq_category = 'laser_cutting'
                            break
                        elif 'cnc' in kw.lower():
                            eq_category = 'cnc'
                            break
                        elif 'lodd' in kw.lower() or 'elektro' in kw.lower():
                            eq_category = 'electronics'
                            break
                    
                    if eq_category not in existing_data['categories']:
                        existing_data['categories'][eq_category] = {
                            'name_no': eq_category.replace('_', ' ').title(),
                            'name_en': eq_category.replace('_', ' ').title(),
                            'equipment': []
                        }
                    existing_data['categories'][eq_category]['equipment'].append(new_data)
                    message = f"Lagt til utstyr '{new_data.get('name', 'ukjent')}' i {eq_category}"
                elif isinstance(new_data, list):
                    # Multiple items
                    count = len(new_data)
                    for item in new_data:
                        eq_category = 'other'
                        if eq_category not in existing_data['categories']:
                            existing_data['categories'][eq_category] = {'equipment': []}
                        existing_data['categories'][eq_category]['equipment'].append(item)
                    message = f"Lagt til {count} utstyr"
            
            elif category == 'regler':
                # Add rules to appropriate section
                if 'equipment_specific' not in existing_data:
                    existing_data['equipment_specific'] = {}
                
                if isinstance(new_data, list):
                    for rule in new_data:
                        applies_to = rule.get('applies_to', 'general')
                        if applies_to == 'all':
                            if 'general_rules' not in existing_data:
                                existing_data['general_rules'] = {'rules': []}
                            existing_data['general_rules']['rules'].append(rule)
                        else:
                            if applies_to not in existing_data['equipment_specific']:
                                existing_data['equipment_specific'][applies_to] = {'rules': []}
                            if 'rules' not in existing_data['equipment_specific'][applies_to]:
                                existing_data['equipment_specific'][applies_to]['rules'] = []
                            existing_data['equipment_specific'][applies_to]['rules'].append(rule)
                    message = f"Lagt til {len(new_data)} regler"
                elif isinstance(new_data, dict):
                    existing_data['general_rules'] = existing_data.get('general_rules', {'rules': []})
                    existing_data['general_rules']['rules'].append(new_data)
                    message = "Lagt til 1 regel"
            
            elif category == 'rom':
                # Add room
                if 'rooms' not in existing_data:
                    existing_data['rooms'] = {}
                
                if isinstance(new_data, dict):
                    room_id = new_data.get('id', f"room-{len(existing_data['rooms']) + 1}")
                    existing_data['rooms'][room_id] = new_data
                    message = f"Lagt til rom '{new_data.get('name_no', room_id)}'"
                else:
                    message = "Lagt til rom"
            
            elif category == 'ressurser':
                # Add resources
                if 'resources' not in existing_data:
                    existing_data['resources'] = {}
                
                if isinstance(new_data, list):
                    # Try to categorize resources
                    for res in new_data:
                        res_category = 'general'
                        title = res.get('title', '').lower()
                        if '3d' in title or 'print' in title or 'prusa' in title:
                            res_category = '3d_printing'
                        elif 'laser' in title:
                            res_category = 'laser_cutting'
                        elif 'arduino' in title or 'electron' in title:
                            res_category = 'electronics'
                        
                        if res_category not in existing_data['resources']:
                            existing_data['resources'][res_category] = {'guides': []}
                        if 'guides' not in existing_data['resources'][res_category]:
                            existing_data['resources'][res_category]['guides'] = []
                        existing_data['resources'][res_category]['guides'].append(res)
                    message = f"Lagt til {len(new_data)} ressurser"
                elif isinstance(new_data, dict):
                    if 'general' not in existing_data['resources']:
                        existing_data['resources']['general'] = {'guides': []}
                    existing_data['resources']['general']['guides'].append(new_data)
                    message = "Lagt til 1 ressurs"
            
            # Update metadata
            existing_data['last_updated'] = datetime.now().strftime('%Y-%m-%d')
            
            # Write back to file
            with open(target_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            # Reload JSON knowledge
            load_json_knowledge()
        
        return jsonify({
            'success': True,
            'message': message,
            'target_file': target_file,
            'stats': get_vault_stats()
        })
        
    except Exception as e:
        print(f"[APPROVE] Error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# Components Database API
# =============================================================================

@app.route('/components')
@login_required
def components_page():
    """Admin page for component management."""
    try:
        rendered = render_template('components.html')
        # Log if body is suspiciously short
        if len(rendered) < 1000:
            app.logger.warning(f"Rendered template is very short: {len(rendered)} chars")
            app.logger.warning(f"First 500 chars: {rendered[:500]}")
        
        # Create response and set headers to allow iframe embedding
        from flask import make_response
        response = make_response(rendered)
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'  # Allow same-origin iframes
        response.headers['Content-Security-Policy'] = "frame-ancestors 'self'"
        return response
    except Exception as e:
        app.logger.error(f"Error rendering components.html: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        error_html = f"<html><body><h1>Error</h1><pre>{str(e)}</pre><pre>{traceback.format_exc()}</pre></body></html>"
        from flask import make_response
        response = make_response(error_html, 500)
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        return response


@app.route('/api/db-status', methods=['GET'])
def api_db_status():
    """Check database connection status."""
    try:
        db.session.execute(db.text('SELECT 1'))
        return jsonify({'status': 'connected', 'database': 'MariaDB'})
    except Exception as e:
        return jsonify({'status': 'disconnected', 'error': str(e)}), 503


@app.route('/api/components', methods=['GET'])
@login_required
def api_get_components():
    """Get all components with stats."""
    try:
        query = request.args.get('q', '')

        if query:
            components = search_components_db(query, limit=100)
        else:
            components = Component.query.order_by(Component.hylleplass, Component.name).all()

        # Get stats
        total = Component.query.count()
        locations = db.session.query(Component.hylleplass).distinct().count()
        restock_count = Component.query.filter(Component.restock == True).count()

        return jsonify({
            'components': [c.to_dict() for c in components],
            'total': total,
            'locations': locations,
            'restock_needed': restock_count
        })
    except Exception as e:
        error_msg = str(e)
        if 'Connection refused' in error_msg or 'Can\'t connect' in error_msg:
            return jsonify({'error': 'Database ikke tilgjengelig. Start MariaDB: net start MariaDB'}), 503
        return jsonify({'error': error_msg}), 500


@app.route('/api/components', methods=['POST'])
@login_required
def api_add_component():
    """Add a new component."""
    try:
        data = request.get_json()
        
        if not data.get('name') or not data.get('hylleplass'):
            return jsonify({'error': 'Navn og hylleplass er paakrevd'}), 400
        
        component = Component(
            name=data['name'],
            hylleplass=data['hylleplass'].upper(),
            antall=data.get('antall', 0),
            forbruksvare=data.get('forbruksvare', False),
            restock=data.get('restock', False)
        )
        
        db.session.add(component)
        db.session.commit()
        
        return jsonify({'success': True, 'component': component.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/components/<int:component_id>', methods=['PUT'])
@login_required
def api_update_component(component_id):
    """Update a component."""
    try:
        component = Component.query.get(component_id)
        if not component:
            return jsonify({'error': 'Komponent ikke funnet'}), 404
        
        data = request.get_json()
        
        if 'name' in data:
            component.name = data['name']
        if 'hylleplass' in data:
            component.hylleplass = data['hylleplass'].upper()
        if 'antall' in data:
            component.antall = data['antall']
        if 'forbruksvare' in data:
            component.forbruksvare = data['forbruksvare']
        if 'restock' in data:
            component.restock = data['restock']
        
        db.session.commit()
        
        return jsonify({'success': True, 'component': component.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/components/<int:component_id>', methods=['DELETE'])
@login_required
def api_delete_component(component_id):
    """Delete a component."""
    try:
        component = Component.query.get(component_id)
        if not component:
            return jsonify({'error': 'Komponent ikke funnet'}), 404
        
        db.session.delete(component)
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/hylleplasser', methods=['GET'])
@login_required
def api_get_hylleplasser():
    """Get all unique hylleplass values."""
    try:
        locations = get_all_hylleplasser()
        return jsonify({'hylleplasser': locations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
