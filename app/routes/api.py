"""
Makerspace RAG - Component API Routes
REST API for component management
"""

from flask import Blueprint, request, jsonify
from flask_login import login_required

from app.extensions import db
from app.models.component import (
    Component, search_components_db, get_all_hylleplasser, get_all_kategorier,
    add_component, update_component, delete_component
)

api_bp = Blueprint('api', __name__)


@api_bp.route('/components', methods=['GET'])
def list_components():
    """List all components or search."""
    query = request.args.get('q', '').strip()
    restock_only = request.args.get('restock', '').lower() == 'true'

    if query:
        components = search_components_db(query, limit=50)
    elif restock_only:
        components = Component.query.filter(Component.restock == True).all()
    else:
        components = Component.query.order_by(Component.hylleplass, Component.name).all()

    return jsonify([c.to_dict() for c in components])


@api_bp.route('/components', methods=['POST'])
@login_required
def create_component():
    """Create a new component."""
    data = request.get_json()

    name = data.get('name', '').strip()
    hylleplass = data.get('hylleplass', '').strip()

    if not name or not hylleplass:
        return jsonify({'error': 'Navn og hylleplass er påkrevd'}), 400

    try:
        component = add_component(
            name=name,
            hylleplass=hylleplass,
            kategori=data.get('kategori', 'Annet'),
            forbruksvare=data.get('forbruksvare', False),
            restock=data.get('restock', False),
            antall=data.get('antall', 0)
        )
        return jsonify(component.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/components/<int:component_id>', methods=['GET'])
def get_component(component_id):
    """Get a single component."""
    component = Component.query.get(component_id)
    if not component:
        return jsonify({'error': 'Komponent ikke funnet'}), 404
    return jsonify(component.to_dict())


@api_bp.route('/components/<int:component_id>', methods=['PUT'])
@login_required
def update_component_route(component_id):
    """Update a component."""
    data = request.get_json()

    component = update_component(
        component_id,
        name=data.get('name'),
        hylleplass=data.get('hylleplass'),
        kategori=data.get('kategori'),
        forbruksvare=data.get('forbruksvare'),
        restock=data.get('restock'),
        antall=data.get('antall')
    )

    if not component:
        return jsonify({'error': 'Komponent ikke funnet'}), 404

    return jsonify(component.to_dict())


@api_bp.route('/components/<int:component_id>', methods=['DELETE'])
@login_required
def delete_component_route(component_id):
    """Delete a component."""
    if delete_component(component_id):
        return jsonify({'success': True})
    return jsonify({'error': 'Komponent ikke funnet'}), 404


@api_bp.route('/hylleplasser', methods=['GET'])
def list_hylleplasser():
    """Get list of all unique shelf locations."""
    return jsonify(get_all_hylleplasser())


@api_bp.route('/kategorier', methods=['GET'])
def list_kategorier():
    """Get list of all unique categories."""
    # Default categories + any custom ones from DB
    default_kategorier = ['Annet', 'Motstand', 'Kondensator', 'LED', 'Diode', 'Transistor',
                          'IC', 'Sensor', 'Kabel', 'Kontakt', 'Bryter', 'Motor', 'Display']
    db_kategorier = get_all_kategorier()
    all_kategorier = sorted(set(default_kategorier + db_kategorier))
    return jsonify(all_kategorier)
