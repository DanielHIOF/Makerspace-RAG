"""
Makerspace RAG - Query Classifier
Classifies and analyzes user queries for routing and context
Enhanced with confidence scores and embedding support
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum


class QueryCategory(Enum):
    """Query classification categories."""
    FEILSOKING = "FEILSOKING"  # Troubleshooting
    OPPLARING = "OPPLARING"    # Learning/tutorials
    VERKTOY_HMS = "VERKTOY_HMS"  # Equipment/safety
    GENERELL = "GENERELL"      # General


@dataclass
class ClassificationResult:
    """Result of query classification with confidence."""
    category: QueryCategory
    confidence: float  # 0.0 to 1.0
    matched_keywords: List[str] = field(default_factory=list)
    secondary_categories: List[Tuple[QueryCategory, float]] = field(default_factory=list)


@dataclass
class ToolDetectionResult:
    """Result of tool detection with confidence."""
    tool: Optional[str]
    confidence: float
    matched_keywords: List[str] = field(default_factory=list)


@dataclass
class QueryAnalysis:
    """Complete analysis of a query."""
    classification: ClassificationResult
    tool: ToolDetectionResult
    level: Tuple[str, str]  # (level_name, instruction)
    language: Tuple[str, str]  # (language, instruction)
    category_mode: Optional[Dict]
    is_inventory: bool
    is_component: bool
    is_code_example: bool
    raw_query: str

    @property
    def category(self) -> str:
        """Backward compatible category accessor."""
        return self.classification.category.value

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'category': self.classification.category.value,
            'category_confidence': self.classification.confidence,
            'matched_keywords': self.classification.matched_keywords,
            'tool': self.tool.tool,
            'tool_confidence': self.tool.confidence,
            'level': self.level[0],
            'language': self.language[0],
            'is_inventory': self.is_inventory,
            'is_component': self.is_component,
            'is_code_example': self.is_code_example
        }


class QueryClassifier:
    """Classifies and analyzes user queries with confidence scoring."""

    # Category modes from slash commands
    CATEGORY_MODES = {
        '/prusa': {
            'tool_filter': '3d_printer',
            'instruction': 'FOKUS PA PRUSA: Brukeren spor om Prusa-printere spesifikt. Prioriter informasjon om Prusa Mini+, Prusa MK3s, PrusaSlicer, og Prusa-spesifikke innstillinger.',
            'boost_keywords': ['prusa', 'prusaslicer', 'prusa mini', 'prusa mk3', 'prusa mk3s']
        },
        '/3d': {
            'tool_filter': '3d_printer',
            'instruction': 'FOKUS PA 3D-PRINTING: Brukeren spor om 3D-printing generelt. Inkluder informasjon om alle typer 3D-printere, filament, slicer-programmer.',
            'boost_keywords': ['3d', 'print', 'printer', 'filament', 'slicer']
        },
        '/laser': {
            'tool_filter': 'laserkutter',
            'instruction': 'FOKUS PA LASERKUTTING: Brukeren spor om laserkutting. Prioriter informasjon om Epilog, Glowforge, laserkutting-prosesser.',
            'boost_keywords': ['laser', 'kutt', 'graver', 'epilog', 'glowforge']
        },
        '/cnc': {
            'tool_filter': 'cnc',
            'instruction': 'FOKUS PA CNC-FRESING: Brukeren spor om CNC-fresing. Prioriter informasjon om Wegstr CNC, Avid CNC, CNC-prosesser.',
            'boost_keywords': ['cnc', 'fres', 'wegstr', 'avid', 'mill', 'router']
        },
        '/elektronikk': {
            'tool_filter': 'elektronikk',
            'instruction': 'FOKUS PA ELEKTRONIKK: Brukeren spor om elektronikk. Prioriter Arduino, Raspberry Pi, komponenter, kretser.',
            'boost_keywords': ['arduino', 'raspberry', 'elektronikk', 'krets', 'komponent']
        },
        '/lodding': {
            'tool_filter': 'lodding',
            'instruction': 'FOKUS PA LODDING: Brukeren spor om lodding. Prioriter loddeutstyr, loddeteknikker, flux.',
            'boost_keywords': ['lodd', 'solder', 'loddekolbe', 'flux']
        }
    }

    # Explanation levels from slash commands
    LEVELS = {
        '/nybegynner': ('nybegynner', "NYBEGYNNER - Forklar som til en som aldri har gjort dette for. Bruk enkle ord, unnga faguttrykk, gi steg-for-steg instruksjoner med eksempler."),
        '/beginner': ('beginner', "BEGINNER - Explain as if to someone who has never done this before. Use simple words, avoid jargon, give step-by-step instructions."),
        '/ekspert': ('ekspert', "EKSPERT - Anta at brukeren har dyp teknisk kunnskap. Bruk presise faguttrykk, diskuter pa profesjonelt niva, inkluder tekniske detaljer."),
        '/expert': ('expert', "EXPERT - Assume deep technical knowledge. Use precise terminology, discuss at professional level, include technical details."),
    }

    # Tool detection patterns with weights (order matters - more specific first)
    TOOLS = [
        ('lodding', ['lodd', 'solder', 'tinn', 'flux', 'kolbe', 'iron', 'soldering'], 1.0),
        ('arduino', ['arduino', 'uno', 'mega', 'nano', 'sketch', 'ide', 'atmega'], 1.0),
        ('raspberry', ['raspberry', 'pi', 'gpio', 'raspbian', 'rpi'], 1.0),
        ('3d_printer', ['3d print', '3d-print', 'printer', 'prusa', 'filament', 'pla', 'abs',
                        'petg', 'nozzle', 'dyse', 'extruder', 'bed', 'byggeplate', 'slicer'], 0.9),
        ('laserkutter', ['laser', 'laserkutt', 'gravering', 'engraving', 'epilog',
                         'kutte', 'gravere', 'fokus', 'watt'], 0.9),
        ('vinylkutter', ['vinyl', 'cricut', 'silhouette', 'sticker', 'klistremerke', 'folie'], 1.0),
        ('tekstil', ['sy', 'sew', 'symaskin', 'stoff', 'fabric', 'broderi', 'embroid'], 1.0),
        ('cnc', ['cnc', 'fres', 'mill', 'router', 'carve'], 0.9),
        ('elektronikk', ['krets', 'circuit', 'breadboard', 'motstand', 'resistor',
                         'kondensator', 'capacitor', 'led', 'pcb', 'multimeter',
                         'volt', 'amp', 'ohm', 'elektronikk', 'electronics'], 0.8),
    ]

    # Classification keywords with weights
    CATEGORY_KEYWORDS = {
        QueryCategory.FEILSOKING: {
            'keywords': [
                'fungerer ikke', 'virker ikke', 'feil', 'error', 'problem', 'stopper',
                'stuck', 'fastkjort', 'losner', 'warping', 'stringing', 'clogged',
                'tett', 'brenner', 'kutter ikke', 'printer ikke', 'henger', 'crashed',
                'mislykkes', 'failed', 'hvorfor', 'what went wrong', 'help',
                'ikke riktig', 'darlig', 'skjev', 'boyd', 'smelter', 'knekker'
            ],
            'weight': 1.2  # Troubleshooting gets boost
        },
        QueryCategory.OPPLARING: {
            'keywords': [
                'hvordan', 'how to', 'how do', 'steg for steg', 'step by step',
                'guide', 'tutorial', 'lare', 'learn', 'begynne', 'start',
                'forste gang', 'first time', 'introduksjon', 'intro', 'basics',
                'grunnleggende', 'eksempel', 'example', 'vise meg', 'show me',
                'forklare', 'explain', 'instruksjon', 'instruction', 'bruke',
                'use', 'lage', 'make', 'create', 'designe', 'design', 'slicing',
                'slice', 'eksportere', 'export', 'importere', 'import', 'settings',
                'innstillinger', 'parametere', 'parameters'
            ],
            'weight': 1.0
        },
        QueryCategory.VERKTOY_HMS: {
            'keywords': [
                'hva slags', 'what kind', 'hvilke', 'which', 'har dere', 'do you have',
                'finnes', 'available', 'utstyr', 'equipment', 'maskin', 'machine',
                'sikkerhet', 'safety', 'hms', 'regler', 'rules', 'fare', 'danger',
                'forbudt', 'forbidden', 'tillatt', 'allowed', 'lov til', 'permitted',
                'verneutstyr', 'protection', 'kan jeg bruke', 'can i use',
                'materiale', 'material', 'type', 'modell', 'model', 'spesifikasjoner',
                'specs', 'kapasitet', 'capacity', 'storrelse', 'size', 'maks', 'max',
                'apningstider', 'opening hours', 'booking', 'reservere', 'reserve'
            ],
            'weight': 1.0
        }
    }

    INVENTORY_PATTERNS = [
        'liste over', 'list of', 'hvilke', 'which',
        'hva har dere', 'what do you have', 'har dere',
        'vis meg alle', 'show me all', 'alle',
        'hva slags', 'what kind', 'typer',
        'oversikt', 'overview', 'inventory',
        'tilgjengelig', 'available', 'finnes'
    ]

    COMPONENT_PATTERNS = [
        'komponent', 'component', 'deler', 'parts',
        'motstand', 'resistor', 'kondensator', 'capacitor',
        'led', 'diode', 'transistor', 'ic', 'chip',
        'sensor', 'modul', 'module', 'arduino', 'esp32', 'esp8266',
        'raspberry', 'motor', 'servo', 'relay', 'rele',
        'kabel', 'wire', 'ledning', 'connector', 'kontakt',
        'skrue', 'screw', 'mutter', 'nut', 'bolt',
        'loddetinn', 'solder', 'tape', 'lim', 'glue',
        'har dere', 'finnes det', 'hvor finner jeg',
        'elektronikk-deler', 'electronics parts'
    ]

    CODE_EXAMPLE_PATTERNS = [
        r'kode.*eksempel', r'code.*example',
        r'hvordan.*koble', r'how.*connect',
        r'pin.*kobling', r'pin.*connection',
        r'arduino.*kode', r'esp32.*kode',
        r'vis.*kode', r'show.*code',
        r'koblingsskjema', r'wiring.*diagram',
        r'koblingsdiagram', r'wiring.*diagram',
        r'hvordan.*bruke', r'how.*use',
        r'eksempel.*kode', r'example.*code',
        r'vis.*diagram', r'show.*diagram',
        r'koble.*til', r'connect.*to'
    ]

    @classmethod
    def classify(cls, query: str) -> ClassificationResult:
        """
        Classify query with confidence score.
        Returns ClassificationResult with primary category and confidence.
        """
        query_lower = query.lower()
        scores = {}
        matched = {}

        # Calculate score for each category
        for category, config in cls.CATEGORY_KEYWORDS.items():
            keywords = config['keywords']
            weight = config['weight']
            matches = [kw for kw in keywords if kw in query_lower]
            # Score based on number of matches and weight
            score = len(matches) * weight
            if matches:
                scores[category] = score
                matched[category] = matches

        # If no matches, return GENERELL with low confidence
        if not scores:
            return ClassificationResult(
                category=QueryCategory.GENERELL,
                confidence=0.3,
                matched_keywords=[],
                secondary_categories=[]
            )

        # Find primary category (highest score)
        sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_category, primary_score = sorted_categories[0]

        # Calculate confidence (normalize by max possible score)
        max_possible = max(len(c['keywords']) * c['weight'] for c in cls.CATEGORY_KEYWORDS.values())
        confidence = min(1.0, (primary_score / max_possible) * 3)  # Scale up, cap at 1.0

        # Get secondary categories with significant scores
        secondary = []
        for cat, score in sorted_categories[1:]:
            if score > primary_score * 0.5:  # At least 50% of primary
                sec_confidence = min(1.0, (score / max_possible) * 3)
                secondary.append((cat, sec_confidence))

        return ClassificationResult(
            category=primary_category,
            confidence=round(confidence, 2),
            matched_keywords=matched.get(primary_category, []),
            secondary_categories=secondary
        )

    @classmethod
    def detect_tool(cls, query: str) -> ToolDetectionResult:
        """Detect which tool/equipment the query is about with confidence."""
        query_lower = query.lower()
        best_tool = None
        best_score = 0
        best_matches = []

        for tool, keywords, weight in cls.TOOLS:
            matches = [kw for kw in keywords if kw in query_lower]
            if matches:
                score = len(matches) * weight
                if score > best_score:
                    best_score = score
                    best_tool = tool
                    best_matches = matches

        if best_tool:
            # Confidence based on number of matching keywords
            confidence = min(1.0, best_score / 3)  # 3+ matches = full confidence
            return ToolDetectionResult(
                tool=best_tool,
                confidence=round(confidence, 2),
                matched_keywords=best_matches
            )

        return ToolDetectionResult(tool=None, confidence=0.0)

    @classmethod
    def detect_level(cls, query: str) -> Tuple[str, str]:
        """Detect explanation level from query."""
        query_lower = query.lower()

        for cmd, (level, instruction) in cls.LEVELS.items():
            if cmd in query_lower:
                return level, instruction

        return 'normal', "NORMAL - Bruk klare, praktiske forklaringer. Balanse mellom enkelhet og presisjon."

    @classmethod
    def detect_language(cls, query: str) -> Tuple[str, str]:
        """Detect language preference from query. Default is Norwegian."""
        query_lower = query.lower()

        if '/english' in query_lower or '/en' in query_lower:
            return 'english', "You MUST respond in English. Use English throughout your entire response."

        return 'norwegian', "Du MA svare pa NORSK. Bruk norsk sprak i hele svaret. ALDRI svar pa engelsk med mindre brukeren eksplisitt ber om det med /english"

    @classmethod
    def detect_category_mode(cls, query: str) -> Optional[Dict]:
        """Detect category mode from slash commands."""
        query_lower = query.lower()

        for cmd, config in cls.CATEGORY_MODES.items():
            if cmd in query_lower:
                return config

        return None

    @classmethod
    def is_inventory_query(cls, query: str) -> bool:
        """Detect if query is asking for a list/inventory of equipment."""
        query_lower = query.lower()
        return any(p in query_lower for p in cls.INVENTORY_PATTERNS)

    @classmethod
    def is_component_query(cls, query: str) -> bool:
        """Detect if query is asking about components/parts."""
        query_lower = query.lower()
        return any(p in query_lower for p in cls.COMPONENT_PATTERNS)

    @classmethod
    def is_code_example_query(cls, query: str) -> bool:
        """Detect if user wants code example."""
        query_lower = query.lower()
        return any(re.search(p, query_lower, re.IGNORECASE) for p in cls.CODE_EXAMPLE_PATTERNS)

    @classmethod
    def analyze(cls, query: str) -> QueryAnalysis:
        """
        Full analysis of a query - returns QueryAnalysis with all detected attributes.
        This is the main entry point for query analysis.
        """
        return QueryAnalysis(
            classification=cls.classify(query),
            tool=cls.detect_tool(query),
            level=cls.detect_level(query),
            language=cls.detect_language(query),
            category_mode=cls.detect_category_mode(query),
            is_inventory=cls.is_inventory_query(query),
            is_component=cls.is_component_query(query),
            is_code_example=cls.is_code_example_query(query),
            raw_query=query
        )

    # Backward compatibility methods
    @classmethod
    def get_category(cls, query: str) -> str:
        """Backward compatible: Get category string."""
        return cls.classify(query).category.value
