"""
Makerspace RAG - Flask Extensions
Centralized extension initialization
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

# Database
db = SQLAlchemy()

# Authentication
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access the admin panel.'
login_manager.login_message_category = 'error'
