"""
Makerspace RAG - Configuration Management
Environment-based configuration with sensible defaults
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_database_uri():
    """Build database URI from environment variables."""
    # Allow full URI override
    if os.environ.get('DATABASE_URI'):
        return os.environ.get('DATABASE_URI')

    # Build from components
    host = os.environ.get('DB_HOST', 'localhost')
    port = os.environ.get('DB_PORT', '3306')
    user = os.environ.get('DB_USER', 'makerspace')
    password = os.environ.get('DB_PASSWORD', 'makerspace2024')
    database = os.environ.get('DB_NAME', 'makerspace_rag')

    return f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}'


class Config:
    """Base configuration."""

    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

    # Database
    SQLALCHEMY_DATABASE_URI = get_database_uri()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'pool_size': 5,
        'max_overflow': 10,
    }

    # Uploads
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_MB', '16')) * 1024 * 1024
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'json', 'md', 'csv', 'html', 'htm', 'xlsx'}

    # RAG Settings
    VAULT_FILE = os.environ.get('VAULT_FILE', 'vault.txt')
    KNOWLEDGE_DIR = os.environ.get('KNOWLEDGE_DIR', 'knowledge')
    CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '1000'))

    # LLM Settings
    OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
    LLM_MODEL = os.environ.get('LLM_MODEL', 'llama3')
    LLM_SMALL_MODEL = os.environ.get('LLM_SMALL_MODEL', 'llama3.2:1b')
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'mxbai-embed-large')

    # Admin Authentication
    ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
    ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'makerspace2024')


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    SQLALCHEMY_ENGINE_OPTIONS = {}  # SQLite doesn't support pool options


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
