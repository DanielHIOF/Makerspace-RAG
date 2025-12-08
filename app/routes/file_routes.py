"""
Makerspace RAG - File Import Routes
PDF extraction, XLSX parsing, and AI enhancement endpoints
"""

import os
import json
import time
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

file_bp = Blueprint('file', __name__)

# Category templates for AI enhancement
CATEGORY_TEMPLATES = {
    'utstyr': {
        'file': 'knowledge/utstyr.json',
        'type': 'json',
        'prompt': '''Du skal lage en JSON-oppføring for et UTSTYR/VERKTØY i et Makerspace.

EKSEMPEL PÅ ØNSKET OUTPUT:
{{
  "id": "prusa-mini",
  "name": "Prusa Mini+",
  "location": "D1-044",
  "status": "active",
  "access_level": "course_makerspace",
  "difficulty": "beginner",
  "materials": ["PLA", "PETG"],
  "keywords_no": ["3d print", "printer", "prusa"],
  "keywords_en": ["3d print", "printer", "prusa"]
}}

DOKUMENTET:
{text}

KONTEKST: {context}

OUTPUT (kun gyldig JSON, ingen forklaring):'''
    },
    'regler': {
        'file': 'knowledge/regler.json',
        'type': 'json',
        'prompt': '''Du skal lage JSON-oppføringer for HMS/SIKKERHETSREGLER.

EKSEMPEL:
[
  {{
    "id": "rule-laser-001",
    "priority": "critical",
    "rule_no": "Følg med på hele jobben",
    "rule_en": "Monitor the entire job",
    "applies_to": "laser_cutting"
  }}
]

DOKUMENTET:
{text}

KONTEKST: {context}

OUTPUT (kun gyldig JSON-array, ingen forklaring):'''
    },
    'vault': {
        'file': 'vault.txt',
        'type': 'text',
        'prompt': '''Du skal lage strukturert KUNNSKAPSINNHOLD for et Makerspace.

FORMAT:
--- NIVÅ: Tittel ---
Innhold her...

NIVÅER: NYBEGYNNER, INTERMEDIATE, AVANSERT, EKSPERT, FEILSØKING

DOKUMENTET:
{text}

KONTEKST: {context}

OUTPUT (kun strukturert tekst, bruk norsk språk):'''
    }
}


def extract_pdf_ocr(file_path):
    """OCR extraction for PDFs using EasyOCR."""
    import fitz
    import easyocr
    import numpy as np
    from PIL import Image

    print(f"  [OCR] Initializing EasyOCR...")
    reader = easyocr.Reader(['no', 'en'], gpu=False)

    doc = fitz.open(file_path)
    all_text = []

    for page_num, page in enumerate(doc):
        print(f"  [OCR] Processing page {page_num + 1}/{len(doc)}...")

        # Render page to image
        mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
        pix = page.get_pixmap(matrix=mat)

        # Convert to numpy array
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_np = np.array(img)

        # Run OCR
        results = reader.readtext(img_np)

        # Extract text
        page_text = ' '.join([r[1] for r in results])
        if page_text.strip():
            all_text.append(f"--- Side {page_num + 1} ---\n{page_text}")

    doc.close()
    return '\n\n'.join(all_text)


def extract_pdf_fast(file_path):
    """PDF text extraction using OCR."""
    import fitz

    doc = fitz.open(file_path)
    total_pages = len(doc)
    doc.close()

    print(f"  [PDF] Extracting {total_pages} pages using OCR...")

    try:
        text = extract_pdf_ocr(file_path)
        if text and len(text.strip()) > 50:
            print(f"  [PDF] OCR extracted {len(text)} chars from {total_pages} pages")
            return text, total_pages
        else:
            raise Exception("OCR returned insufficient text")
    except Exception as e:
        print(f"  [PDF] OCR failed: {e}")
        raise Exception(f"PDF extraction failed: {e}")


def parse_xlsx_to_equipment(file_path):
    """Parse XLSX file and convert rows to component entries."""
    import openpyxl

    wb = openpyxl.load_workbook(file_path, data_only=True)
    sheet = wb.active

    # Get headers
    headers = []
    for cell in sheet[1]:
        headers.append(str(cell.value).lower().strip() if cell.value else '')

    print(f"  [XLSX] Found headers: {headers}")

    # Column mapping
    COLUMN_MAPPINGS = {
        'id': ['id', 'product_id', 'produktid', 'varenr', 'item_id', 'sku'],
        'name': ['name', 'navn', 'product', 'produkt', 'description', 'beskrivelse'],
        'location': ['location', 'lokasjon', 'sted', 'rom', 'room', 'placement'],
        'category': ['category', 'kategori', 'type', 'gruppe', 'group'],
        'notes': ['notes', 'notater', 'kommentar', 'comment', 'remarks'],
    }

    column_map = {}
    for field, possible_names in COLUMN_MAPPINGS.items():
        for i, header in enumerate(headers):
            if header in possible_names:
                column_map[field] = i
                break

    print(f"  [XLSX] Column mapping: {column_map}")

    # Parse rows
    equipment_list = []
    for row_num, row in enumerate(sheet.iter_rows(min_row=2, values_only=True), start=2):
        if not any(row):
            continue

        entry = {
            'id': '',
            'name': '',
            'location': '',
            'category': 'other',
            'notes': '',
            'keywords_no': [],
            'keywords_en': [],
            '_row': row_num
        }

        for field, col_idx in column_map.items():
            if col_idx < len(row) and row[col_idx] is not None:
                entry[field] = str(row[col_idx]).strip()

        # Generate ID if not present
        if not entry['id'] and entry['name']:
            import re
            slug = re.sub(r'[^a-z0-9]+', '-', entry['name'].lower())
            entry['id'] = slug.strip('-')[:50]

        # Generate keywords
        if entry['name']:
            words = entry['name'].lower().split()
            entry['keywords_no'] = [w for w in words if len(w) > 2]
            entry['keywords_en'] = entry['keywords_no']

        if entry['name']:
            equipment_list.append(entry)

    wb.close()
    return equipment_list, headers, column_map


@file_bp.route('/extract-pdf', methods=['POST'])
def extract_pdf():
    """Fast PDF extraction - returns raw text for preview."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Ingen fil lastet opp'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Ingen fil valgt'}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'success': False, 'error': 'Kun PDF-filer støttes'}), 400

    try:
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        print(f"[FAST EXTRACT] Processing {filename}...")
        start_time = time.time()

        pdf_text, page_count = extract_pdf_fast(file_path)

        os.remove(file_path)

        elapsed = time.time() - start_time
        print(f"[FAST EXTRACT] Done in {elapsed:.2f}s")

        if not pdf_text or len(pdf_text.strip()) < 50:
            return jsonify({'success': False, 'error': 'Kunne ikke trekke ut tekst fra PDF'}), 400

        return jsonify({
            'success': True,
            'filename': filename,
            'text': pdf_text,
            'char_count': len(pdf_text),
            'page_count': page_count,
            'extract_time': round(elapsed, 2)
        })

    except Exception as e:
        print(f"[FAST EXTRACT] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@file_bp.route('/extract-xlsx', methods=['POST'])
def extract_xlsx():
    """Parse XLSX and return equipment entries for preview."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Ingen fil lastet opp'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Ingen fil valgt'}), 400

    if not file.filename.lower().endswith('.xlsx'):
        return jsonify({'success': False, 'error': 'Kun XLSX-filer støttes'}), 400

    try:
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        print(f"[XLSX EXTRACT] Processing {filename}...")
        start_time = time.time()

        equipment_list, headers, column_map = parse_xlsx_to_equipment(file_path)

        # Load existing components for duplicate check
        components_path = 'knowledge/components.json'
        existing_components = {}
        if os.path.exists(components_path):
            with open(components_path, 'r', encoding='utf-8') as f:
                existing_components = json.load(f)

        # Simple duplicate check
        existing_ids = set()
        for cat in existing_components.get('categories', {}).values():
            for item in cat.get('components', []):
                existing_ids.add(item.get('id', '').lower())

        items_to_add = [i for i in equipment_list if i.get('id', '').lower() not in existing_ids]
        duplicates_skipped = [i for i in equipment_list if i.get('id', '').lower() in existing_ids]

        os.remove(file_path)

        elapsed = time.time() - start_time
        print(f"[XLSX EXTRACT] Found {len(equipment_list)} items in {elapsed:.2f}s")

        if not equipment_list:
            return jsonify({'success': False, 'error': 'Ingen utstyr funnet i filen'}), 400

        return jsonify({
            'success': True,
            'filename': filename,
            'headers': headers,
            'column_map': column_map,
            'total_rows': len(equipment_list),
            'items_to_add': items_to_add,
            'duplicates_skipped': duplicates_skipped,
            'extract_time': round(elapsed, 2)
        })

    except Exception as e:
        print(f"[XLSX EXTRACT] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@file_bp.route('/approve-xlsx', methods=['POST'])
def approve_xlsx():
    """Add approved component items to components.json."""
    data = request.get_json()
    if not data or 'items' not in data:
        return jsonify({'success': False, 'error': 'Ingen komponenter å legge til'}), 400

    items = data['items']
    if not items:
        return jsonify({'success': False, 'error': 'Listen er tom'}), 400

    try:
        components_path = 'knowledge/components.json'

        if os.path.exists(components_path):
            with open(components_path, 'r', encoding='utf-8') as f:
                components_data = json.load(f)
        else:
            components_data = {
                'version': '1.0',
                'last_updated': '',
                'description': 'Komponentoversikt for Makerspace',
                'categories': {}
            }

        # Group items by category
        by_category = {}
        for item in items:
            cat = item.get('category', 'other').lower().strip().replace(' ', '_') or 'other'
            if cat not in by_category:
                by_category[cat] = []
            clean = {k: v for k, v in item.items() if not k.startswith('_')}
            by_category[cat].append(clean)

        # Add to categories
        for cat, cat_items in by_category.items():
            if cat not in components_data.get('categories', {}):
                components_data['categories'][cat] = {
                    'name_no': cat.replace('_', ' ').title(),
                    'name_en': cat.replace('_', ' ').title(),
                    'components': []
                }
            components_data['categories'][cat]['components'].extend(cat_items)

        components_data['last_updated'] = datetime.now().strftime('%Y-%m-%d')

        with open(components_path, 'w', encoding='utf-8') as f:
            json.dump(components_data, f, indent=2, ensure_ascii=False)

        return jsonify({
            'success': True,
            'message': f'La til {len(items)} komponenter',
            'items_added': len(items),
            'categories': list(by_category.keys())
        })

    except Exception as e:
        print(f"[XLSX APPROVE] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@file_bp.route('/enhance-pdf', methods=['POST'])
def enhance_pdf():
    """Use LLM to structure/summarize extracted PDF text."""
    import ollama

    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'success': False, 'error': 'Ingen tekst å behandle'}), 400

    pdf_text = data.get('text', '').strip()
    doc_context = data.get('context', '').strip()
    category = data.get('category', 'vault').strip()

    if len(pdf_text) < 50:
        return jsonify({'success': False, 'error': 'For lite tekst'}), 400

    if category not in CATEGORY_TEMPLATES:
        return jsonify({'success': False, 'error': f'Ukjent kategori: {category}'}), 400

    template = CATEGORY_TEMPLATES[category]

    # Truncate if needed
    max_chars = 8000
    truncated = len(pdf_text) > max_chars
    if truncated:
        pdf_text = pdf_text[:max_chars]

    print(f"[ENHANCE] Category: {category} | Sending {len(pdf_text)} chars to LLM...")

    prompt = template['prompt'].format(text=pdf_text, context=doc_context)

    try:
        start_time = time.time()
        response = ollama.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.3, 'num_predict': 2000}
        )
        summary = response['message']['content']
        elapsed = time.time() - start_time
        print(f"[ENHANCE] LLM done in {elapsed:.1f}s")

        # Clean up JSON output
        if template['type'] == 'json':
            summary = summary.strip()
            if summary.startswith('```json'):
                summary = summary[7:]
            if summary.startswith('```'):
                summary = summary[3:]
            if summary.endswith('```'):
                summary = summary[:-3]
            summary = summary.strip()

        return jsonify({
            'success': True,
            'enhanced_text': summary,
            'original_chars': len(pdf_text),
            'enhanced_chars': len(summary),
            'truncated': truncated,
            'enhance_time': round(elapsed, 1),
            'category': category,
            'output_type': template['type'],
            'target_file': template['file']
        })
    except Exception as e:
        print(f"[ENHANCE] LLM Error: {e}")
        return jsonify({'success': False, 'error': f'LLM feil: {str(e)}'}), 500


@file_bp.route('/add-text', methods=['POST'])
def add_text():
    """Add text directly to vault."""
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'success': False, 'message': 'Ingen tekst mottatt'})

    try:
        with open('vault.txt', 'a', encoding='utf-8') as f:
            f.write('\n' + text)

        # Reload index
        from app.services.search_service import get_search_service
        search_service = get_search_service()
        search_service.load_vault()

        return jsonify({
            'success': True,
            'message': f'La til {len(text)} tegn i kunnskapsbasen'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@file_bp.route('/approve-summary', methods=['POST'])
def approve_summary():
    """Add approved LLM-generated content to appropriate file."""
    data = request.get_json()

    if not data or 'content' not in data:
        return jsonify({'success': False, 'error': 'Ingen innhold å legge til'}), 400

    content = data.get('content', '').strip()
    category = data.get('category', 'vault').strip()

    if not content:
        return jsonify({'success': False, 'error': 'Tomt innhold'}), 400

    if category not in CATEGORY_TEMPLATES:
        return jsonify({'success': False, 'error': f'Ukjent kategori: {category}'}), 400

    template = CATEGORY_TEMPLATES[category]
    target_file = template['file']

    try:
        if template['type'] == 'text':
            # Append to vault.txt
            with open('vault.txt', 'a', encoding='utf-8') as f:
                f.write('\n\n')
                f.write(content)
                f.write('\n')

            sections = content.count('---')
            message = f'Lagt til ~{sections//2} seksjoner i vault.txt'

        else:
            # Handle JSON categories
            try:
                new_data = json.loads(content)
            except json.JSONDecodeError as je:
                return jsonify({
                    'success': False,
                    'error': f'Ugyldig JSON: {str(je)}'
                }), 400

            # Load existing JSON file
            existing_data = {}
            if os.path.exists(target_file):
                with open(target_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

            # Simple merge - add to list or dict
            if isinstance(new_data, list):
                if 'items' not in existing_data:
                    existing_data['items'] = []
                existing_data['items'].extend(new_data)
            elif isinstance(new_data, dict):
                existing_data.update(new_data)

            with open(target_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)

            message = f'Lagt til data i {target_file}'

        # Reload search index
        from app.services.search_service import get_search_service
        search_service = get_search_service()
        search_service.load_vault()

        return jsonify({
            'success': True,
            'message': message,
            'target_file': target_file
        })

    except Exception as e:
        print(f"[APPROVE] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
