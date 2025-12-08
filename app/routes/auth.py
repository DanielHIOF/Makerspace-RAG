"""
Makerspace RAG - Authentication Routes
Login and logout endpoints
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash

from app.models.user import AdminUser

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/auth/check')
def check_auth():
    """Check if user is authenticated (for React frontend)."""
    return jsonify({
        'authenticated': current_user.is_authenticated
    })


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Admin login page - supports both form and JSON."""
    if request.method == 'POST':
        # Check if JSON request (from React) - check content type header directly
        content_type = request.content_type or ''
        is_json = request.is_json or 'application/json' in content_type

        if is_json:
            try:
                data = request.get_json(force=True)
                password = data.get('password', '') if data else ''

                if check_password_hash(current_app.admin_password_hash, password):
                    user = AdminUser('admin')
                    login_user(user)
                    return jsonify({'success': True})
                else:
                    return jsonify({'success': False, 'error': 'Feil passord'})
            except Exception as e:
                return jsonify({'success': False, 'error': f'Invalid request: {str(e)}'})

        # Form submission (legacy)
        username = request.form.get('username', '')
        password = request.form.get('password', '')

        if username == current_app.config['ADMIN_USERNAME'] and \
           check_password_hash(current_app.admin_password_hash, password):
            user = AdminUser('admin')
            login_user(user)
            flash('Innlogget!', 'success')
            return redirect(url_for('admin.admin_panel'))
        else:
            flash('Feil brukernavn eller passord', 'error')

    return render_template('login.html')


@auth_bp.route('/logout', methods=['GET', 'POST'])
def logout():
    """Logout - supports both redirect and JSON response."""
    logout_user()

    if request.method == 'POST':
        return jsonify({'success': True})

    flash('Du er nå logget ut.', 'success')
    return redirect(url_for('public.index'))
