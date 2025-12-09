"""
Makerspace RAG - Engangs Data Import Script
Kjør dette scriptet EN gang ved første oppsett for å importere
komponenter fra JSON til databasen.

Bruk:
    python installer/import_data.py

Dette scriptet:
1. Leser knowledge/components.json
2. Importerer alle komponenter til databasen
3. Hopper over hvis data allerede finnes
"""

import os
import sys
import json
from pathlib import Path

# Legg til prosjektrot i path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Last miljøvariabler
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')


def print_status(message, status="INFO"):
    """Print formatert statusmelding."""
    icons = {"INFO": "ℹ", "OK": "✓", "WARN": "!", "ERROR": "✗", "WAIT": "~"}
    icon = icons.get(status, "*")
    print(f"  [{icon}] {message}")


def import_components():
    """Importer komponenter fra JSON til database."""
    from app import create_app
    from app.extensions import db
    from app.models.component import Component

    app = create_app()

    with app.app_context():
        # Sjekk om data allerede finnes
        existing_count = Component.query.count()
        if existing_count > 0:
            print_status(f"Database har allerede {existing_count} komponenter", "WARN")
            response = input("  Vil du overskrive? (ja/nei) [nei]: ").strip().lower()
            if response != 'ja':
                print_status("Avbryter import", "INFO")
                return False

            # Slett eksisterende data
            Component.query.delete()
            db.session.commit()
            print_status("Slettet eksisterende komponenter", "OK")

        # Les JSON-fil
        json_path = PROJECT_ROOT / 'knowledge' / 'components.json'
        if not json_path.exists():
            print_status(f"Fant ikke {json_path}", "ERROR")
            return False

        print_status(f"Leser {json_path}...", "WAIT")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Tell komponenter
        total_components = 0
        imported = 0

        # Iterer gjennom kategorier
        for kategori_key, kategori_data in data.items():
            kategori_navn = kategori_data.get('name_no', kategori_key)
            components = kategori_data.get('components', [])

            print_status(f"Importerer {kategori_navn}: {len(components)} komponenter", "INFO")

            for comp in components:
                total_components += 1

                # Map JSON-felt til database-felt
                name = comp.get('name', comp.get('id', 'Ukjent'))
                location = comp.get('location', 'Ukjent')

                # Opprett komponent
                db_comp = Component(
                    name=name,
                    hylleplass=location.upper() if location else 'UKJENT',
                    kategori=kategori_navn,
                    forbruksvare=comp.get('forbruksvare', False),
                    restock=comp.get('restock', False),
                    antall=comp.get('antall', 0)
                )
                db.session.add(db_comp)
                imported += 1

        # Commit alle endringer
        db.session.commit()

        print_status(f"Importert {imported} av {total_components} komponenter", "OK")
        return True


def generate_sql_dump():
    """Generer SQL INSERT-statements fra JSON."""
    json_path = PROJECT_ROOT / 'knowledge' / 'components.json'
    sql_path = PROJECT_ROOT / 'installer' / 'initial_data.sql'

    if not json_path.exists():
        print_status(f"Fant ikke {json_path}", "ERROR")
        return False

    print_status(f"Genererer SQL-dump...", "WAIT")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sql_lines = [
        "-- Makerspace RAG - Initial Data Import",
        "-- Generert automatisk fra components.json",
        "-- Kjør EN gang ved første oppsett",
        "",
        "-- Slett eksisterende data (valgfritt)",
        "-- DELETE FROM components;",
        "",
        "-- Importer komponenter",
    ]

    count = 0
    for kategori_key, kategori_data in data.items():
        kategori_navn = kategori_data.get('name_no', kategori_key)
        components = kategori_data.get('components', [])

        sql_lines.append(f"\n-- {kategori_navn}")

        for comp in components:
            name = comp.get('name', comp.get('id', 'Ukjent'))
            location = comp.get('location', 'Ukjent')

            # Escape enkle anførselstegn for SQL
            name_escaped = name.replace("'", "''")
            location_escaped = location.replace("'", "''").upper() if location else 'UKJENT'
            kategori_escaped = kategori_navn.replace("'", "''")

            forbruksvare = 1 if comp.get('forbruksvare', False) else 0
            restock = 1 if comp.get('restock', False) else 0
            antall = comp.get('antall', 0)

            sql_lines.append(
                f"INSERT INTO components (name, hylleplass, kategori, forbruksvare, restock, antall) "
                f"VALUES ('{name_escaped}', '{location_escaped}', '{kategori_escaped}', {forbruksvare}, {restock}, {antall});"
            )
            count += 1

    # Skriv til fil
    with open(sql_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sql_lines))

    print_status(f"Lagret {count} INSERT-statements til {sql_path}", "OK")
    return True


def main():
    """Hovedfunksjon."""
    print("\n" + "="*60)
    print("  MAKERSPACE RAG - DATA IMPORT")
    print("="*60)

    print("\n  Velg importmetode:\n")
    print("    [1] Importer direkte til database (anbefalt)")
    print("    [2] Generer SQL-fil (for manuell import)")
    print("    [3] Begge deler")
    print()

    choice = input("  Valg (1/2/3) [1]: ").strip() or "1"

    if choice in ('1', '3'):
        print("\n[Importerer til database...]")
        import_components()

    if choice in ('2', '3'):
        print("\n[Genererer SQL-dump...]")
        generate_sql_dump()

    print("\n" + "="*60)
    print("  FERDIG!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
