# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Makerspace RAG
Build with: pyinstaller installer/makerspace_rag.spec
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(SPECPATH).parent

block_cipher = None

# Collect all app files
app_data = [
    # Flask templates and static files
    (str(PROJECT_ROOT / 'app' / 'templates'), 'app/templates'),
    (str(PROJECT_ROOT / 'app' / 'static'), 'app/static'),

    # Knowledge base
    (str(PROJECT_ROOT / 'vault.txt'), '.'),
    (str(PROJECT_ROOT / 'knowledge'), 'knowledge'),

    # JSON data files
    (str(PROJECT_ROOT / 'utstyr.json'), '.'),
    (str(PROJECT_ROOT / 'sikkerhet.json'), '.'),
    (str(PROJECT_ROOT / 'rom.json'), '.'),

    # Installer scripts and database setup
    (str(PROJECT_ROOT / 'installer' / 'database_setup.sql'), 'installer'),
    (str(PROJECT_ROOT / 'installer' / 'setup_database.py'), 'installer'),
    (str(PROJECT_ROOT / 'installer' / 'setup_ollama.py'), 'installer'),

    # Environment template
    (str(PROJECT_ROOT / '.env.example'), '.'),
]

# Filter out non-existent paths
app_data = [(src, dst) for src, dst in app_data if os.path.exists(src)]

# Hidden imports for Flask and SQLAlchemy
hidden_imports = [
    'flask',
    'flask_cors',
    'flask_login',
    'flask_sqlalchemy',
    'sqlalchemy',
    'sqlalchemy.dialects.mysql',
    'sklearn',
    'sklearn.feature_extraction.text',
    'sklearn.metrics.pairwise',
    'numpy',
    'requests',
    'werkzeug',
    'jinja2',
    'markupsafe',
    'click',
    'itsdangerous',
    'pymysql',
    'dotenv',
    'python-dotenv',
]

a = Analysis(
    [str(PROJECT_ROOT / 'installer' / 'launcher.py')],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=app_data,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MakerspaceRAG',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Show console for status messages
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / 'app' / 'static' / 'makerspace-logo.ico') if (PROJECT_ROOT / 'app' / 'static' / 'makerspace-logo.ico').exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MakerspaceRAG',
)
