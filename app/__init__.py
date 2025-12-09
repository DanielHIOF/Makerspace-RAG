"""
Makerspace RAG - Application Factory
Flask application initialization with factory pattern
"""

import os
from flask import Flask, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash

from app.config import config
from app.extensions import db, login_manager
from app.models.user import AdminUser


def create_app(config_name=None):
    """Create and configure the Flask application."""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')

    app = Flask(__name__)

    # Enable CORS for development (React dev server on port 3000)
    CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'])

    # Load configuration
    app.config.from_object(config[config_name])

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)

    # Create upload folder
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Setup user loader for Flask-Login
    @login_manager.user_loader
    def load_user(user_id):
        if user_id == 'admin':
            return AdminUser('admin')
        return None

    # Store password hash for admin authentication
    app.admin_password_hash = generate_password_hash(app.config['ADMIN_PASSWORD'])

    # Initialize database
    with app.app_context():
        try:
            db.create_all()
            db.session.execute(db.text('SELECT 1'))
            print("  [DB] Database connected and tables ready")
        except Exception as e:
            print(f"  [DB] ERROR: {e}")
            print("  [DB] Make sure MariaDB is running: net start MariaDB")

    # Register blueprints
    from app.routes import public_bp, auth_bp, admin_bp, api_bp
    from app.routes.file_routes import file_bp
    app.register_blueprint(public_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(file_bp)

    # Preload search service with embeddings on startup
    print("  [SEARCH] Preloading search service and embeddings...")
    from app.services.search_service import get_search_service
    get_search_service()
    print("  [SEARCH] Search service ready!")

    return app
