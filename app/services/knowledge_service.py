"""
Makerspace RAG - Knowledge Service
Manages structured knowledge from JSON files
"""

import os
import json
from app.extensions import db
from app.models.component import Component, search_components_db


class KnowledgeService:
    """Manages structured JSON knowledge from knowledge/ directory."""

    def __init__(self, knowledge_dir='knowledge'):
        self.knowledge_dir = knowledge_dir
        self._knowledge = {
            'utstyr': {},
            'regler': {},
            'rom': {},
            'ressurser': {},
            'components': {},
            'kodeeksempler': {},
            'prosessflyt': {},
            'prosjektskalering': {},
            'prosjektideer': {}
        }

    def load_all(self):
        """Load all JSON knowledge files."""
        for name in self._knowledge.keys():
            self._load_file(name)

    def _load_file(self, name):
        """Load a single knowledge file."""
        filepath = os.path.join(self.knowledge_dir, f'{name}.json')
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self._knowledge[name] = json.load(f)
            print(f"  Loaded {name}.json")
        except Exception as e:
            print(f"  Warning: Could not load {name}.json: {e}")
            self._knowledge[name] = {}

    def get(self, name):
        """Get a knowledge dictionary by name."""
        return self._knowledge.get(name, {})

    # Tool type to category mapping
    TOOL_TO_CATEGORY = {
        '3d_printer': '3d_printing',
        'laserkutter': 'laser_cutting',
        'cnc': 'cnc',
        'lodding': 'electronics',
        'elektronikk': 'electronics',
        'vinylkutter': None,
        'tekstil': None,
        'arduino': 'electronics',
        'raspberry': 'electronics'
    }

    TOOL_TO_SAFETY_CATEGORY = {
        '3d_printer': '3d_printing',
        'laserkutter': 'laser_cutting',
        'cnc': 'cnc',
        'lodding': 'soldering',
        'elektronikk': 'soldering'
    }

    def get_equipment_context(self, tool_type):
        """Get structured equipment info for a detected tool type."""
        utstyr = self._knowledge.get('utstyr', {})
        if not utstyr or 'categories' not in utstyr:
            return ""

        category = self.TOOL_TO_CATEGORY.get(tool_type)
        if not category or category not in utstyr['categories']:
            return ""

        cat_data = utstyr['categories'][category]
        equipment_list = cat_data.get('equipment', [])

        if not equipment_list:
            return ""

        lines = [f"UTSTYR ({cat_data.get('name_no', category)}):"]
        for eq in equipment_list:
            lines.append(f"- {eq.get('name', 'Ukjent')}")
            if eq.get('location'):
                lines.append(f"  Lokasjon: {eq['location']}")
            if eq.get('difficulty'):
                lines.append(f"  Nivå: {eq['difficulty']}")
            if eq.get('requires_training'):
                lines.append(f"  Krever opplæring: Ja")
            if eq.get('materials'):
                lines.append(f"  Materialer: {', '.join(eq['materials'])}")

        return "\n".join(lines)

    def get_all_equipment_by_access(self):
        """Get all equipment organized by access level."""
        utstyr = self._knowledge.get('utstyr', {})
        if not utstyr or 'categories' not in utstyr:
            return ""

        # Collect equipment by access level
        by_access = {}
        for cat_key, cat_data in utstyr['categories'].items():
            for eq in cat_data.get('equipment', []):
                access = eq.get('access_level', 'unknown')
                if access not in by_access:
                    by_access[access] = []
                by_access[access].append({
                    'name': eq.get('name', 'Ukjent'),
                    'location': eq.get('location', ''),
                    'difficulty': eq.get('difficulty', ''),
                    'certifier': eq.get('certifier', ''),
                    'notes': eq.get('notes', '')
                })

        # Build context string in order of access level
        lines = ["TILGANGSNIVÅER FOR UTSTYR VED MAKERSPACE HiØF:", ""]

        access_order = ['course_makerspace', 'course_fablab', 'certification_required',
                        'request_required', 'staff_only']
        access_names = {
            'course_makerspace': '1. MakerSpace-kurs (D1-044) - Etter fullført kurs kan du bruke:',
            'course_fablab': '2. FabLab HMS-kurs (D1-043) - Krever HMS-kurs:',
            'certification_required': '3. Sertifisering påkrevd - Kontakt labingeniør:',
            'request_required': '4. Må hentes fra labansvarlig:',
            'staff_only': '5. Kun personale:'
        }

        for access in access_order:
            if access in by_access:
                lines.append(access_names.get(access, access))
                for eq in by_access[access]:
                    line = f"  - {eq['name']}"
                    if eq['location']:
                        line += f" ({eq['location']})"
                    lines.append(line)
                lines.append("")

        lines.append("VIKTIG: Uten opplæring kan du IKKE bruke noe utstyr.")
        lines.append("Start med: MakerSpace introduksjonskurs for å få tilgang til 3D-printere, lodding, osv.")
        lines.append("Mer info: https://www.hiof.no/iio/itk/om/labber/makerspace/arrangementer/")

        return "\n".join(lines)

    def search_components(self, query):
        """Search for components by name or location - uses database."""
        results = search_components_db(query, limit=20)

        if not results:
            return ""

        lines = [f"KOMPONENTER FUNNET ({len(results)} treff):"]
        lines.append("VIKTIG: List DISSE komponentene, ikke andre!")
        lines.append("")

        for comp in results[:15]:
            line = f"- {comp.name} @ {comp.hylleplass}"
            if comp.antall and comp.antall > 0:
                line += f" ({comp.antall} stk)"
            if comp.restock:
                line += " [TRENGER RESTOCK]"
            lines.append(line)

        if len(results) > 15:
            lines.append(f"... og {len(results) - 15} flere")

        return "\n".join(lines)

    def get_all_components_summary(self):
        """Get a summary of all available components - uses database."""
        total = Component.query.count()

        if total == 0:
            return "Ingen komponenter registrert ennå."

        lines = ["TILGJENGELIGE KOMPONENTER VED MAKERSPACE HiØF:", ""]

        # Group by hylleplass
        locations = db.session.query(Component.hylleplass).distinct().order_by(Component.hylleplass).all()

        for loc in locations[:10]:
            loc_name = loc[0]
            comps = Component.query.filter(Component.hylleplass == loc_name).limit(5).all()
            if comps:
                lines.append(f"{loc_name}:")
                for comp in comps:
                    lines.append(f"  - {comp.name}")
                count = Component.query.filter(Component.hylleplass == loc_name).count()
                if count > 5:
                    lines.append(f"  ... og {count - 5} flere")
                lines.append("")

        lines.append(f"Totalt: {total} komponenter på {len(locations)} hylleplasser.")
        lines.append("Spør om spesifikke komponenter for mer info!")

        return "\n".join(lines)

    def get_safety_rules_context(self, tool_type=None, include_general=True):
        """Get HMS/safety rules for a tool type or general rules."""
        regler = self._knowledge.get('regler', {})
        if not regler:
            return ""

        lines = []

        # General rules
        if include_general and 'general_rules' in regler:
            gen_rules = regler['general_rules'].get('rules', [])
            critical_rules = [r for r in gen_rules if r.get('priority') == 'critical']
            if critical_rules:
                lines.append("VIKTIGE SIKKERHETSREGLER:")
                for r in critical_rules[:3]:
                    lines.append(f"⚠️ {r.get('rule_no', '')}")

        # Equipment-specific rules
        if tool_type and 'equipment_specific' in regler:
            category = self.TOOL_TO_SAFETY_CATEGORY.get(tool_type)
            if category and category in regler['equipment_specific']:
                cat_rules = regler['equipment_specific'][category]
                rules = cat_rules.get('rules', [])

                if rules:
                    lines.append(f"\nREGLER FOR {cat_rules.get('name_no', category).upper()}:")
                    for r in rules[:4]:
                        priority_icon = "🔴" if r.get('priority') == 'critical' else "🟡"
                        lines.append(f"{priority_icon} {r.get('rule_no', '')}")

                # Forbidden materials for laser
                if category == 'laser_cutting':
                    forbidden = cat_rules.get('forbidden_materials', [])
                    if forbidden:
                        lines.append("\n⛔ FORBUDTE MATERIALER:")
                        for f in forbidden[:4]:
                            lines.append(f"- {f.get('material')}: {f.get('reason_no', '')}")

        return "\n".join(lines)

    def get_room_context(self, room_id=None):
        """Get room information."""
        rom = self._knowledge.get('rom', {})
        if not rom or 'rooms' not in rom:
            return ""

        lines = ["LOKALER:"]
        for room_key, room in rom['rooms'].items():
            lines.append(f"- {room.get('id', '')}: {room.get('name_no', '')} - {room.get('description_no', '')}")

        # Add contact info
        if 'contacts' in rom:
            contacts = rom['contacts']
            if 'lab_responsible' in contacts:
                lines.append(f"\nLabansvarlig: Morgan Waage (kontor D1-060B)")
            if 'student_assistants' in contacts:
                lines.append("Studentassistenter tilgjengelig i åpningstider")

        return "\n".join(lines)


# Global instance for easy access
_knowledge_service = None


def get_knowledge_service(knowledge_dir='knowledge'):
    """Get or create the knowledge service singleton."""
    global _knowledge_service
    if _knowledge_service is None:
        _knowledge_service = KnowledgeService(knowledge_dir)
        _knowledge_service.load_all()
    return _knowledge_service
