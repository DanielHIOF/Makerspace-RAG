"""
Makerspace RAG - Public Routes
Main chat interface and status endpoints
"""

import os
import time
import traceback
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, send_from_directory, current_app

from app.services.search_service import get_search_service
from app.services.knowledge_service import get_knowledge_service
from app.services.llm_service import get_llm_service
from app.services.query_classifier import QueryClassifier

public_bp = Blueprint('public', __name__)

# Check if React build exists
REACT_BUILD_DIR = os.path.join(os.path.dirname(__file__), '..', 'static', 'react')
USE_REACT = os.path.exists(os.path.join(REACT_BUILD_DIR, 'index.html'))


def get_vault_stats():
    """Get statistics about the vault."""
    vault_file = 'vault.txt'
    if not os.path.exists(vault_file):
        return {'chunks': 0, 'size': 0, 'size_kb': 0}

    with open(vault_file, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = [l.strip() for l in content.split('\n') if l.strip()]

    return {
        'chunks': len(lines),
        'size': len(content),
        'size_kb': round(len(content) / 1024, 2)
    }


@public_bp.route('/')
def index():
    """Main chat interface (public)."""
    # Serve React build if it exists
    if USE_REACT:
        return send_from_directory(REACT_BUILD_DIR, 'index.html')

    # Fall back to Jinja template
    stats = get_vault_stats()
    return render_template('index.html', stats=stats)


@public_bp.route('/assets/<path:path>')
def serve_react_assets(path):
    """Serve React build assets from /assets/."""
    if USE_REACT:
        assets_dir = os.path.join(REACT_BUILD_DIR, 'assets')
        return send_from_directory(assets_dir, path)
    return '', 404


@public_bp.route('/admin')
def admin_react():
    """Serve React app for admin route (client-side routing)."""
    if USE_REACT:
        return send_from_directory(REACT_BUILD_DIR, 'index.html')
    # Fall back to Jinja template
    return render_template('admin.html')


@public_bp.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with conversation history support."""
    timestamp = datetime.now().strftime("%H:%M:%S")

    data = request.get_json()
    message = data.get('message', '').strip()
    conversation_history = data.get('history', [])
    existing_summary = data.get('summary', '')

    if not message:
        return jsonify({'response': 'Please enter a message.'})

    # Easter egg
    if 'green apples' in message.lower() or 'grønne epler' in message.lower():
        return jsonify({
            'response': "Ja, grønne epler er kjempegode! 🍏\n\nDe er sprø, friske og fulle av smak.",
            'summary': existing_summary
        })

    print(f"\n{'='*60}")
    print(f"[{timestamp}] [CHAT] NY MELDING MOTTATT")
    print(f"  Sporsmal: {message[:100]}{'...' if len(message) > 100 else ''}")
    print(f"  Historikk: {len(conversation_history)} meldinger")

    # Get services
    search_service = get_search_service()
    knowledge_service = get_knowledge_service()
    llm_service = get_llm_service()

    # Build combined context from history for better tool detection
    combined_context = message
    if len(message.split()) < 5 and conversation_history:
        recent_messages = [msg.get('content', '') for msg in conversation_history[-4:]]
        combined_context = ' '.join(recent_messages) + ' ' + message

    # Analyze query
    analysis = QueryClassifier.analyze(combined_context)
    category_mode = analysis.category_mode

    detected_tool = analysis.tool.tool
    if category_mode:
        detected_tool = category_mode.get('tool_filter', detected_tool)

    inventory_query = analysis.is_inventory
    component_query = analysis.is_component

    print(f"  Kategori: {analysis.category} (confidence: {analysis.classification.confidence})")
    if detected_tool:
        print(f"  Verktoy: {detected_tool} (confidence: {analysis.tool.confidence})")

    # Build context from multiple sources
    context_parts = []

    # Component queries
    if component_query:
        comp_ctx = knowledge_service.search_components(message)
        if comp_ctx:
            context_parts.append(comp_ctx)
            print(f"  [DB] Fant komponenter")
        else:
            comp_summary = knowledge_service.get_all_components_summary()
            if comp_summary:
                context_parts.append(comp_summary)

    # Inventory queries
    if inventory_query and not detected_tool:
        all_equipment = knowledge_service.get_all_equipment_by_access()
        if all_equipment:
            context_parts.append(all_equipment)
            print(f"  [JSON] Lagt til utstyrsoversikt")
    elif detected_tool:
        equipment_ctx = knowledge_service.get_equipment_context(detected_tool)
        if equipment_ctx:
            context_parts.append(equipment_ctx)
            print(f"  [JSON] Lagt til utstyrskontekst for {detected_tool}")

    # Safety rules
    if not inventory_query and (analysis.category == 'VERKTOY_HMS' or detected_tool):
        safety_ctx = knowledge_service.get_safety_rules_context(
            tool_type=detected_tool,
            include_general=(analysis.category == 'VERKTOY_HMS')
        )
        if safety_ctx:
            context_parts.append(safety_ctx)
            print(f"  [JSON] Lagt til sikkerhetsregler")

    # Room info
    if any(word in message.lower() for word in ['rom', 'hvor', 'lokasjon']):
        room_ctx = knowledge_service.get_room_context()
        if room_ctx:
            context_parts.append(room_ctx)

    # TF-IDF search
    if not inventory_query:
        print(f"[{timestamp}] [SEARCH] Soker i kunnskapsbasen...")
        search_start = time.time()

        search_query = message
        if category_mode and category_mode.get('boost_keywords'):
            search_query = message + ' ' + ' '.join(category_mode.get('boost_keywords', []))

        relevant_chunks = search_service.search(search_query, tool_filter=detected_tool)
        search_time = time.time() - search_start
        print(f"  [OK] Fant {len(relevant_chunks)} relevante biter ({search_time:.2f}s)")

        if relevant_chunks:
            context_parts.append("DOKUMENTASJON:\n" + "\n\n".join(relevant_chunks))

    context = "\n\n".join(context_parts) if context_parts else "Ingen relevant informasjon funnet."

    # Get response from LLM
    print(f"[{timestamp}] [LLM] Sender til Ollama...")
    llm_start = time.time()

    try:
        response, updated_summary = llm_service.chat(
            message, context,
            is_inventory=inventory_query,
            conversation_history=conversation_history,
            existing_summary=existing_summary
        )
        llm_time = time.time() - llm_start
        print(f"  [OK] Svar mottatt! ({llm_time:.1f}s)")
        return jsonify({
            'response': response,
            'summary': updated_summary
        })
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        return jsonify({'response': f'Feil: {e}. Sjekk terminalen for detaljer.'})


@public_bp.route('/status')
def status():
    """Check if search index is loaded."""
    search_service = get_search_service()
    stats = search_service.get_stats()
    return jsonify({
        'loaded': stats['has_index'],
        'chunks': stats['chunks'],
        'terms': stats['terms'],
        'embeddings': stats.get('embeddings', 0),
        'hybrid_enabled': stats.get('use_embeddings', False),
        'hybrid_alpha': stats.get('hybrid_alpha', 0.5)
    })


@public_bp.route('/health')
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
    search_service = get_search_service()
    stats = search_service.get_stats()
    vault_ok = stats['chunks'] > 0

    return jsonify({
        'status': 'healthy' if (ollama_ok and vault_ok) else 'degraded',
        'ollama': {
            'running': ollama_ok,
            'models': ollama_models
        },
        'vault': {
            'loaded': vault_ok,
            'chunks': stats['chunks'],
            'terms': stats['terms']
        },
        'search': {
            'embeddings': stats.get('embeddings', 0),
            'hybrid_enabled': stats.get('use_embeddings', False),
            'hybrid_alpha': stats.get('hybrid_alpha', 0.5)
        }
    })
