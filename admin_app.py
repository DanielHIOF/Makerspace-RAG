"""
Admin Web Interface for Makerspace RAG
Upload files and manage the knowledge base
"""

import os
import re
import json
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import PyPDF2

app = Flask(__name__)
app.secret_key = 'makerspace-admin-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'json', 'md'}

VAULT_FILE = 'vault.txt'
CHUNK_SIZE = 1000  # Max characters per chunk

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def get_vault_stats():
    """Get statistics about the vault"""
    if not os.path.exists(VAULT_FILE):
        return {'chunks': 0, 'size': 0}
    
    with open(VAULT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = [l.strip() for l in content.split('\n') if l.strip()]
    
    return {
        'chunks': len(lines),
        'size': len(content),
        'size_kb': round(len(content) / 1024, 2)
    }


def get_recent_chunks(n=10):
    """Get the last n chunks from the vault"""
    if not os.path.exists(VAULT_FILE):
        return []
    
    with open(VAULT_FILE, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
        return lines[-n:] if len(lines) >= n else lines


def text_to_chunks(text, chunk_size=CHUNK_SIZE):
    """
    Split text into chunks, trying multiple strategies.
    Each chunk should be under chunk_size characters.
    """
    if not text:
        return []
    
    # Clean up the text - normalize whitespace but preserve some structure
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces/tabs
    text = text.strip()
    
    if not text:
        return []
    
    # If text is short enough, return as single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    
    # Strategy 1: Try splitting by paragraphs first (double newline)
    paragraphs = re.split(r'\n\n+', text)
    
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If paragraph itself is too long, split it further
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
                    
                    # If single sentence is too long, force split it
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
            # Paragraph fits, try to add to current chunk
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += (" " + para if current_chunk else para)
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out empty chunks and very short ones
    chunks = [c for c in chunks if c and len(c) > 10]
    
    return chunks


def process_pdf(file_path):
    """Extract text from PDF file"""
    text = ''
    try:
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            print(f"[PDF] Found {len(pdf_reader.pages)} pages")
            
            for i, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                        print(f"[PDF] Page {i+1}: extracted {len(page_text)} chars")
                except Exception as e:
                    print(f"[PDF] Error on page {i+1}: {e}")
                    
        print(f"[PDF] Total extracted: {len(text)} chars")
    except Exception as e:
        print(f"[PDF] Error reading PDF: {e}")
        raise
    
    return text


def process_txt(file_path):
    """Read text from TXT file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def process_json(file_path):
    """Extract text from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return json.dumps(data, ensure_ascii=False)


def process_md(file_path):
    """Read text from Markdown file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def append_to_vault(chunks):
    """Append chunks to the vault file"""
    with open(VAULT_FILE, 'a', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk.strip() + '\n')
    return len(chunks)


@app.route('/')
def index():
    """Main admin page"""
    stats = get_vault_stats()
    recent = get_recent_chunks(5)
    return render_template('admin.html', stats=stats, recent_chunks=recent)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'success': False, 
            'error': f'File type not allowed. Allowed: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
        }), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text based on file type
        ext = filename.rsplit('.', 1)[1].lower()
        print(f"\n[UPLOAD] Processing {filename} (type: {ext})")
        
        if ext == 'pdf':
            text = process_pdf(file_path)
        elif ext == 'txt':
            text = process_txt(file_path)
        elif ext == 'json':
            text = process_json(file_path)
        elif ext == 'md':
            text = process_md(file_path)
        else:
            text = ''
        
        print(f"[UPLOAD] Extracted {len(text)} characters of text")
        
        # Convert to chunks
        chunks = text_to_chunks(text)
        print(f"[UPLOAD] Created {len(chunks)} chunks")
        
        if not chunks:
            return jsonify({
                'success': False, 
                'error': 'Could not extract any text from the file'
            }), 400
        
        # Append to vault
        added_count = append_to_vault(chunks)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify({
            'success': True,
            'message': f'Successfully added {added_count} chunks from {filename}',
            'chunks_added': added_count,
            'stats': get_vault_stats()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/add-text', methods=['POST'])
def add_text():
    """Add text directly to the vault"""
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
        
        added_count = append_to_vault(chunks)
        
        return jsonify({
            'success': True,
            'message': f'Successfully added {added_count} chunks',
            'chunks_added': added_count,
            'stats': get_vault_stats()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/stats')
def stats():
    """Get vault statistics"""
    return jsonify(get_vault_stats())


@app.route('/recent')
def recent():
    """Get recent chunks"""
    n = request.args.get('n', 10, type=int)
    chunks = get_recent_chunks(n)
    return jsonify({'chunks': chunks, 'count': len(chunks)})


@app.route('/preview', methods=['POST'])
def preview():
    """Preview how text will be chunked"""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'success': False, 'error': 'No text provided'}), 400
    
    text = data.get('text', '').strip()
    chunks = text_to_chunks(text)
    
    return jsonify({
        'success': True,
        'chunks': chunks,
        'count': len(chunks)
    })


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  Makerspace RAG - Admin Interface")
    print("  Add data to the knowledge base")
    print("=" * 50)
    print(f"\nVault: {VAULT_FILE}")
    stats = get_vault_stats()
    print(f"Current chunks: {stats['chunks']}")
    print(f"Open http://localhost:5001 in your browser")
    print("=" * 50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
