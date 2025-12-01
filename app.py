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

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'makerspace-secret-key-change-in-production')

# =============================================================================
# Configuration
# =============================================================================
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'json', 'md'}
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


def search_vault(query, top_k=5):
    """Find most relevant chunks using TF-IDF similarity."""
    if tfidf_vectorizer is None or tfidf_matrix is None or len(vault_content) == 0:
        return []
    
    # Remove level commands from query
    clean_query = re.sub(r'/(nybegynner|ny|middels|avansert|ekspert|beginner|new|intermediate|advanced|expert)\s*', '', query)
    
    # Transform query to TF-IDF vector
    query_vector = tfidf_vectorizer.transform([clean_query])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top-k indices
    top_k = min(top_k, len(similarities))
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Filter out zero-similarity results
    results = [(vault_content[i], similarities[i]) for i in top_indices if similarities[i] > 0]
    
    return [chunk for chunk, score in results]


def detect_level(query):
    """Detect explanation level from query (supports both Norwegian and English)."""
    query_lower = query.lower()
    
    levels = {
        # Norwegian commands
        '/nybegynner': ('nybegynner', "Forklar som om jeg ikke vet noe. Bruk enkle ord, ingen faguttrykk, steg-for-steg med eksempler."),
        '/ny': ('ny', "Forklar med antagelse om grunnleggende kjennskap. Definer faguttrykk når de brukes første gang."),
        '/avansert': ('avansert', "Anta betydelig erfaring. Gå i teknisk dybde, kanttilfeller, optimalisering."),
        '/ekspert': ('ekspert', "Anta dyp ekspertise. Diskuter på profesjonelt nivå med teori og teknisk presisjon."),
        # English commands
        '/beginner': ('beginner', "Explain like I know nothing. Use simple words, no jargon, step-by-step with examples."),
        '/new': ('new', "Explain assuming basic familiarity. Define technical terms when first used."),
        '/advanced': ('advanced', "Assume significant experience. Go into technical depth, edge cases, optimization."),
        '/expert': ('expert', "Assume deep expertise. Discuss at professional level with theory and precision."),
    }
    
    for cmd, (level, instruction) in levels.items():
        if cmd in query_lower:
            return level, instruction
    
    return 'intermediate', "Assume working knowledge. Use standard terminology, focus on practical details."


def ask_llm(query, context):
    """Send query + context to llama3."""
    level, level_instruction = detect_level(query)
    clean_query = re.sub(r'/(nybegynner|ny|middels|avansert|ekspert|beginner|new|intermediate|advanced|expert)\s*', '', query).strip()
    
    prompt = f"""You are a helpful assistant for the Makerspace at Høgskolen i Østfold (HiØF).
You help students and staff with questions about 3D printing, laser cutting, electronics, vinyl cutting, textiles, and maker projects.

IMPORTANT - Explanation Level: {level.upper()}
{level_instruction}

Use the following knowledge base context to answer accurately. If the context doesn't contain relevant information, use your general knowledge but mention that.

=== KNOWLEDGE BASE CONTEXT ===
{context}
=== END CONTEXT ===

Question: {clean_query}

Respond in the same language as the question (Norwegian or English). Be practical and helpful."""
    
    try:
        response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    except Exception as e:
        error_msg = str(e)
        print(f"  ❌ OLLAMA FEIL: {error_msg}")
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


def text_to_chunks(text, chunk_size=CHUNK_SIZE):
    """Split text into chunks using paragraph and sentence boundaries."""
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
    paragraphs = re.split(r'\n\n+', text)
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if len(para) > chunk_size:
            # Save current chunk first
            if current_chunk:
                chunks.append(current_chunk.strip())
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
                        chunks.append(current_chunk.strip())
                    
                    # Force split very long sentences
                    if len(sentence) > chunk_size:
                        words = sentence.split()
                        current_chunk = ""
                        for word in words:
                            if len(current_chunk) + len(word) + 1 <= chunk_size:
                                current_chunk += (" " + word if current_chunk else word)
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = word
                    else:
                        current_chunk = sentence
        else:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += (" " + para if current_chunk else para)
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out empty and very short chunks
    return [c for c in chunks if c and len(c) > 10]


def process_pdf(file_path):
    """Extract text from PDF file."""
    text = ''
    try:
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
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
    print(f"[{timestamp}] 📩 NY MELDING MOTTATT")
    print(f"{'='*60}")
    print(f"  Spørsmål: {message[:100]}{'...' if len(message) > 100 else ''}")
    
    # Search for relevant context
    print(f"\n[{timestamp}] 🔍 Søker i kunnskapsbasen...")
    search_start = time.time()
    relevant_chunks = search_vault(message)
    search_time = time.time() - search_start
    print(f"  ✓ Fant {len(relevant_chunks)} relevante biter ({search_time:.2f}s)")
    
    if relevant_chunks:
        for i, chunk in enumerate(relevant_chunks[:3]):
            preview = chunk[:80].replace('\n', ' ')
            print(f"    {i+1}. {preview}...")
    
    context = "\n\n".join(relevant_chunks) if relevant_chunks else "No relevant information found in knowledge base."
    
    # Get response from LLM
    print(f"\n[{timestamp}] 🤖 Sender til Ollama (llama3)...")
    print(f"  ⏳ Venter på svar (dette kan ta 10-30 sek)...")
    llm_start = time.time()
    
    try:
        response = ask_llm(message, context)
        llm_time = time.time() - llm_start
        print(f"  ✓ Svar mottatt! ({llm_time:.1f}s)")
        print(f"\n[{timestamp}] 📤 SVAR SENDT TIL BRUKER")
        print(f"  Lengde: {len(response)} tegn")
        print(f"{'='*60}\n")
        return jsonify({'response': response})
    except Exception as e:
        llm_time = time.time() - llm_start
        error_msg = str(e)
        print(f"\n  ❌ FEIL etter {llm_time:.1f}s!")
        print(f"  ❌ Type: {type(e).__name__}")
        print(f"  ❌ Melding: {error_msg}")
        print(f"\n  📋 FULL TRACEBACK:")
        print("-" * 40)
        traceback.print_exc()
        print("-" * 40)
        print(f"\n  💡 TIPS: Sjekk at Ollama kjører: 'ollama list'")
        print(f"{'='*60}\n")
        return jsonify({'response': f'Feil: {error_msg}. Sjekk terminalen for detaljer.'})


@app.route('/status')
def status():
    """Check if search index is loaded."""
    return jsonify({
        'loaded': tfidf_matrix is not None,
        'chunks': len(vault_content)
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
            
            chunks = text_to_chunks(text)
            
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
        chunks = text_to_chunks(text)
        
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
