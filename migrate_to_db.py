"""
Makerspace RAG - Database Migration Script
Imports existing components.json data into MariaDB database

Run this ONCE:
    python migrate_to_db.py

Creates tables and imports components
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from models import db, Component
from db_config import DATABASE_URI

# Create Flask app for migration
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)


def create_tables():
    """Create database tables"""
    print("Creating database...")
    db.create_all()
    print("[OK] Database created!")


def import_components_json():
    """Import components from knowledge/components.json"""
    json_path = 'knowledge/components.json'
    
    if not os.path.exists(json_path):
        print(f"[ERROR] File not found: {json_path}")
        return 0
    
    print(f"Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both structures
    categories_data = data.get('categories', data)
    
    total_imported = 0
    
    for cat_key, cat_data in categories_data.items():
        if not isinstance(cat_data, dict) or 'components' not in cat_data:
            continue
        
        components_list = cat_data.get('components', [])
        print(f"  {cat_key}: {len(components_list)} components...")
        
        for comp_data in components_list:
            name = comp_data.get('name', 'Unnamed')
            location = comp_data.get('location', 'UKJENT')
            
            # Create component with uppercase hylleplass
            component = Component(
                name=name,
                hylleplass=location.upper(),  # ENFORCE CAPS
                forbruksvare=False,  # Default, can update later
                restock=False,       # Default, can update later
                antall=comp_data.get('quantity', 0)
            )
            
            db.session.add(component)
            total_imported += 1
            
            # Batch commit
            if total_imported % 500 == 0:
                db.session.commit()
                print(f"    ... {total_imported} imported")
    
    db.session.commit()
    return total_imported


def print_stats():
    """Show what we got"""
    total = Component.query.count()
    locations = db.session.query(Component.hylleplass).distinct().count()
    
    print(f"\nDATABASE STATS:")
    print(f"   Components: {total}")
    print(f"   Unique hylleplasser: {locations}")
    
    print(f"\nHYLLEPLASSER:")
    for loc in db.session.query(Component.hylleplass).distinct().order_by(Component.hylleplass).limit(15).all():
        count = Component.query.filter(Component.hylleplass == loc[0]).count()
        print(f"   {loc[0]}: {count} components")


def run_migration():
    """Run full migration"""
    print("=" * 50)
    print("MAKERSPACE DATABASE MIGRATION")
    print("=" * 50)
    print()
    
    create_tables()
    print()
    
    count = import_components_json()
    print(f"\n[OK] Imported {count} components!")
    
    print_stats()
    
    print()
    print("=" * 50)
    print("[OK] DONE! Database: makerspace.db")
    print("=" * 50)


if __name__ == '__main__':
    with app.app_context():
        run_migration()
