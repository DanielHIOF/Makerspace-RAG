"""
Makerspace RAG - Admin Routes
Protected admin panel and file management
"""

import os
from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required
from werkzeug.utils import secure_filename

from app.services.search_service import get_search_service

admin_bp = Blueprint('admin', __name__)


def allowed_file(filename):
    """Check if file extension is allowed."""
    allowed = current_app.config.get('ALLOWED_EXTENSIONS', {'txt', 'pdf', 'json', 'md'})
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed


def get_vault_stats():
    """Get statistics about the vault."""
    vault_file = current_app.config.get('VAULT_FILE', 'vault.txt')
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


@admin_bp.route('/admin-legacy')
@login_required
def admin_panel():
    """Legacy admin panel dashboard (Jinja template)."""
    stats = get_vault_stats()
    return render_template('admin.html', stats=stats)


@admin_bp.route('/admin/stats')
@login_required
def admin_stats():
    """Get vault statistics for admin panel (JSON API)."""
    return jsonify(get_vault_stats())


@admin_bp.route('/components')
@login_required
def components_page():
    """Component management page."""
    return render_template('components.html')


@admin_bp.route('/reload', methods=['POST'])
@login_required
def reload_index():
    """Reload the search index."""
    try:
        search_service = get_search_service()
        search_service.load_vault()
        stats = search_service.get_stats()
        return jsonify({
            'success': True,
            'message': f"Lastet inn {stats['chunks']} biter med {stats['terms']} termer"
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@admin_bp.route('/add-text', methods=['POST'])
@login_required
def add_text():
    """Add text directly to vault."""
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'success': False, 'message': 'Ingen tekst mottatt'})

    vault_file = current_app.config.get('VAULT_FILE', 'vault.txt')

    try:
        with open(vault_file, 'a', encoding='utf-8') as f:
            f.write('\n' + text)

        # Reload index
        search_service = get_search_service()
        search_service.load_vault()

        return jsonify({
            'success': True,
            'message': f'La til {len(text)} tegn i kunnskapsbasen'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
