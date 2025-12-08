"""
Makerspace RAG - Search Service
Hybrid search using TF-IDF + semantic embeddings
"""

import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional

# Import embedding service (lazy import to avoid circular dependency)
_embedding_service = None


def _get_embedding_service():
    """Lazy import of embedding service."""
    global _embedding_service
    if _embedding_service is None:
        from app.services.embedding_service import get_embedding_service
        _embedding_service = get_embedding_service()
    return _embedding_service


class SearchService:
    """Manages vault content and hybrid search (TF-IDF + semantic embeddings)."""

    # Tool filter keywords for specific equipment
    TOOL_FILTER_KEYWORDS = {
        '3d_printer': ['3d', 'print', 'printer', 'prusa', 'filament', 'pla', 'abs', 'petg',
                       'nozzle', 'dyse', 'extruder', 'byggeplate', 'slicer', 'infill'],
        'laserkutter': ['laser', 'kutt', 'graver', 'epilog', 'watt', 'fokus',
                        'akryl', 'mdf', 'speed', 'power', 'dpi', 'engrav'],
        'lodding': ['lodd', 'solder', 'soldering', 'flux', 'kolbe', 'tinning'],
        'arduino': ['arduino', 'uno', 'mega', 'nano', 'sketch', 'atmega'],
        'raspberry': ['raspberry', 'gpio', 'raspbian'],
        'elektronikk': ['krets', 'circuit', 'breadboard', 'motstand', 'resistor',
                        'kondensator', 'capacitor', 'transistor', 'diode',
                        'pcb', 'volt', 'amp', 'ohm', 'multimeter'],
        'vinylkutter': ['vinyl', 'cricut', 'silhouette', 'klistremerke', 'folie'],
        'tekstil': ['symaskin', 'stoff', 'fabric', 'broderi', 'embroid'],
        'cnc': ['cnc', 'fres', 'mill', 'router', 'carve', 'spindel']
    }

    # Norwegian to English expansions for better matching
    QUERY_EXPANSIONS = {
        # Lodding / Soldering
        'lodd': 'solder soldering',
        'lodde': 'solder soldering iron',
        'lodder': 'solder soldering how to',
        'lodding': 'solder soldering iron tip flux',
        'tinn': 'solder tin lead-free',
        'loddekolbe': 'soldering iron station tip',
        'kolbe': 'soldering iron',
        'fluss': 'flux rosin',
        'loddetinn': 'solder wire',

        # 3D printing
        'prusa': 'prusa prusa3d prusaslicer prusa mini prusa mk3 prusa mk3s prusa i3',
        'printer': 'printer printing 3d print prusa ultimaker voron',
        '3d': '3d print printer printing prusa prusaslicer',
        'printe': 'print printing 3d prusa',
        'skrive ut': 'print printing',
        'byggeplate': 'bed build plate adhesion first layer',
        'dyse': 'nozzle hotend extruder clog',
        'filament': 'filament pla abs petg material',
        'ekstrudere': 'extrude extruder extrusion',
        'lag': 'layer height layers',
        'feste': 'adhesion bed stick',
        'løsner': 'warping adhesion lifting detach bed',
        'stringing': 'stringing oozing retraction',
        'tett': 'clogged clog jam nozzle',
        'varme': 'temperature heat bed nozzle',
        'temp': 'temperature heat',
        'slicer': 'slicer slicing cura prusaslicer prusa slicer',
        'prusaslicer': 'prusaslicer prusa slicer slicing',
        'stl': 'stl file model',
        'infill': 'infill density fill',
        'support': 'support supports overhang',
        'raft': 'raft brim skirt adhesion',
        'brim': 'brim skirt adhesion',

        # Laser
        'laser': 'laser cutter cutting engraving',
        'laserkutter': 'laser cutter cutting engraving epilog',
        'kutte': 'cut cutting speed power',
        'kutter': 'cut cutter cutting',
        'gravere': 'engrave engraving etch',
        'gravering': 'engraving engrave etch',
        'fokus': 'focus height z-offset distance',
        'brenner': 'burn burning fire power',
        'akryl': 'acrylic plexiglass',
        'pleksiglass': 'acrylic plexiglass',
        'tre': 'wood mdf plywood',
        'mdf': 'mdf wood',
        'hastighet': 'speed velocity',
        'styrke': 'power watt strength',
        'watt': 'watt power',

        # Electronics
        'krets': 'circuit board pcb',
        'motstand': 'resistor resistance ohm',
        'kondensator': 'capacitor',
        'transistor': 'transistor',
        'diode': 'diode led',
        'led': 'led light diode',
        'arduino': 'arduino uno mega microcontroller',
        'raspberry': 'raspberry pi gpio',
        'breadboard': 'breadboard prototype',
        'multimeter': 'multimeter volt amp ohm',

        # General actions
        'hvordan': 'how to guide tutorial steps',
        'bruke': 'use using operate',
        'bruker': 'use using how to',
        'starte': 'start begin power on',
        'slå på': 'power on turn on start',
        'slå av': 'power off turn off stop',
        'fungerer ikke': 'not working problem error fix',
        'virker ikke': 'not working broken error',
        'feil': 'error problem issue fix',
        'problem': 'problem error issue troubleshoot',
        'hjelp': 'help guide tutorial',
        'starter ikke': 'not starting power error',
        'stopper': 'stopping stops error freeze',
        'krasjer': 'crash error freeze',

        # Safety / HMS
        'sikkerhet': 'safety safe danger warning',
        'hms': 'safety health environment',
        'fare': 'danger hazard warning',
        'verneutstyr': 'safety equipment protection ppe',
        'briller': 'glasses goggles safety',
        'hansker': 'gloves protection',
        'brann': 'fire burn safety',

        # Materials
        'materiale': 'material settings',
        'plast': 'plastic pla abs petg',
        'metall': 'metal aluminum steel',
        'stoff': 'fabric textile cloth',
    }

    def __init__(self, vault_file='vault.txt', use_embeddings=True, hybrid_alpha=0.5):
        self.vault_file = vault_file
        self.vault_content = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self._knowledge_utstyr = None  # Reference for equipment boost

        # Embedding/hybrid search settings
        self.use_embeddings = use_embeddings
        self.hybrid_alpha = hybrid_alpha  # 0=TF-IDF only, 1=embeddings only
        self.vault_embeddings = []  # Cached embeddings for vault content

    def set_knowledge_reference(self, knowledge_utstyr):
        """Set reference to equipment knowledge for query boosting."""
        self._knowledge_utstyr = knowledge_utstyr

    def load_vault(self, build_embeddings=None):
        """Load vault content and build search indices."""
        print("Loading knowledge base...")
        self.vault_content = []

        if os.path.exists(self.vault_file):
            with open(self.vault_file, "r", encoding='utf-8') as f:
                self.vault_content = [line.strip() for line in f.readlines() if line.strip()]

        print(f"Loaded {len(self.vault_content)} knowledge chunks")

        if not self.vault_content:
            print("No data in knowledge base yet.")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            self.vault_embeddings = []
            return

        # Build TF-IDF index
        print("Building TF-IDF index...")
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1,
            stop_words=None
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.vault_content)
        print(f"TF-IDF index ready! ({self.tfidf_matrix.shape[1]} terms)")

        # Build embeddings if enabled
        should_build_embeddings = build_embeddings if build_embeddings is not None else self.use_embeddings
        if should_build_embeddings:
            self._build_embeddings()

    def get_tool_filter_keywords(self, tool):
        """Get keywords that chunks must contain for a specific tool."""
        return self.TOOL_FILTER_KEYWORDS.get(tool, [])

    def expand_query(self, query):
        """Expand Norwegian query with English synonyms for better TF-IDF matching."""
        equipment_keywords = []

        # Add equipment-specific keywords from JSON if available
        if self._knowledge_utstyr and 'categories' in self._knowledge_utstyr:
            query_lower = query.lower()
            for category_name, category_data in self._knowledge_utstyr['categories'].items():
                for equipment in category_data.get('equipment', []):
                    equipment_name = equipment.get('name', '').lower()
                    equipment_id = equipment.get('id', '').lower()
                    keywords = equipment.get('keywords_no', []) + equipment.get('keywords_en', [])

                    if (equipment_name in query_lower or
                            equipment_id in query_lower or
                            any(kw.lower() in query_lower for kw in keywords)):
                        equipment_keywords.extend(keywords)
                        # Add equipment name variations
                        if 'prusa' in equipment_name:
                            equipment_keywords.extend(['prusa', 'prusa3d', 'prusaslicer'])
                        if 'mini' in equipment_name:
                            equipment_keywords.append('mini')
                        if 'mk3' in equipment_name or 'mk3s' in equipment_name:
                            equipment_keywords.extend(['mk3', 'mk3s', 'mk3s+'])
                        if 'ultimaker' in equipment_name:
                            equipment_keywords.extend(['ultimaker', 'cura'])
                        if 'voron' in equipment_name:
                            equipment_keywords.extend(['voron', 'corexy'])

        expanded = query
        query_lower = query.lower()

        # Add equipment keywords first
        if equipment_keywords:
            expanded += ' ' + ' '.join(set(equipment_keywords))

        # Then add general expansions
        for no_term, en_terms in self.QUERY_EXPANSIONS.items():
            if no_term in query_lower:
                expanded += ' ' + en_terms

        return expanded

    def _build_embeddings(self):
        """Build embeddings for vault content using the embedding service."""
        if not self.vault_content:
            self.vault_embeddings = []
            return

        print("Building semantic embeddings (this may take a while on first run)...")
        embedding_service = _get_embedding_service()

        # Generate embeddings for all vault content
        self.vault_embeddings = embedding_service.get_embeddings_batch(
            self.vault_content,
            show_progress=True
        )

        valid_count = sum(1 for e in self.vault_embeddings if e is not None)
        print(f"Semantic embeddings ready! ({valid_count}/{len(self.vault_content)} chunks)")

    def _semantic_search(self, query: str, top_k: int = 10) -> List[tuple]:
        """Search using semantic embeddings. Returns (index, score) tuples."""
        if not self.vault_embeddings or not any(self.vault_embeddings):
            return []

        embedding_service = _get_embedding_service()
        query_embedding = embedding_service.get_embedding(query)

        if not query_embedding:
            return []

        scores = []
        for i, doc_emb in enumerate(self.vault_embeddings):
            if doc_emb:
                score = embedding_service.cosine_similarity(query_embedding, doc_emb)
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _tfidf_search(self, query: str, top_k: int = 10) -> List[tuple]:
        """Search using TF-IDF. Returns (index, score) tuples."""
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []

        # Expand query for better matching
        expanded_query = self.expand_query(query)
        query_vector = self.tfidf_vectorizer.transform([expanded_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Apply equipment boost if applicable
        self._apply_equipment_boost(query, similarities)

        # Get top results
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        return [(i, similarities[i]) for i in sorted_indices if similarities[i] > 0]

    def _apply_equipment_boost(self, query: str, similarities: np.ndarray):
        """Apply boost to chunks containing mentioned equipment."""
        if not self._knowledge_utstyr or 'categories' not in self._knowledge_utstyr:
            return

        query_lower = query.lower()
        equipment_boost_terms = []

        for category_name, category_data in self._knowledge_utstyr['categories'].items():
            for equipment in category_data.get('equipment', []):
                equipment_name = equipment.get('name', '').lower()
                equipment_id = equipment.get('id', '').lower()

                if (equipment_name in query_lower or
                        equipment_id in query_lower or
                        any(kw.lower() in query_lower for kw in
                            equipment.get('keywords_no', []) + equipment.get('keywords_en', []))):
                    equipment_boost_terms.append(equipment_name)
                    equipment_boost_terms.append(equipment_id)
                    equipment_boost_terms.extend(equipment_name.split())

        if equipment_boost_terms:
            for i in range(len(similarities)):
                chunk_lower = self.vault_content[i].lower()
                boost_count = sum(1 for term in equipment_boost_terms if term in chunk_lower)
                if boost_count > 0:
                    boost = min(0.3, boost_count * 0.1)
                    similarities[i] += boost

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = None) -> List[tuple]:
        """
        Hybrid search combining TF-IDF and semantic embeddings.

        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for embeddings (0=TF-IDF only, 1=embeddings only)
                   If None, uses self.hybrid_alpha

        Returns:
            List of (index, combined_score) tuples
        """
        alpha = alpha if alpha is not None else self.hybrid_alpha

        # Get both search results (fetch more than needed for combination)
        fetch_k = top_k * 3
        tfidf_results = self._tfidf_search(query, fetch_k)
        semantic_results = self._semantic_search(query, fetch_k) if self.use_embeddings else []

        # Combine scores
        combined_scores = {}

        # Normalize and add TF-IDF scores
        if tfidf_results:
            max_tfidf = max(score for _, score in tfidf_results) or 1
            for idx, score in tfidf_results:
                normalized_score = score / max_tfidf
                combined_scores[idx] = combined_scores.get(idx, 0) + (1 - alpha) * normalized_score

        # Normalize and add semantic scores
        if semantic_results:
            max_semantic = max(score for _, score in semantic_results) or 1
            for idx, score in semantic_results:
                normalized_score = score / max_semantic
                combined_scores[idx] = combined_scores.get(idx, 0) + alpha * normalized_score

        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def search(self, query, top_k=3, max_chunk_chars=600, max_total_chars=1800, tool_filter=None, use_hybrid=None):
        """
        Find most relevant chunks using hybrid search (TF-IDF + embeddings).

        Args:
            query: Search query
            top_k: Max number of results
            max_chunk_chars: Max characters per chunk
            max_total_chars: Max total characters for all results
            tool_filter: Optional tool name to filter results
            use_hybrid: Use hybrid search (default: True if embeddings available)

        Returns:
            List of relevant text chunks
        """
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None or len(self.vault_content) == 0:
            return []

        # Remove level commands from query
        clean_query = re.sub(
            r'/(nybegynner|ny|middels|avansert|ekspert|beginner|new|intermediate|advanced|expert)\s*',
            '', query
        )

        # Determine search method
        should_use_hybrid = use_hybrid if use_hybrid is not None else (self.use_embeddings and self.vault_embeddings)

        # Get ranked results
        if should_use_hybrid:
            ranked_results = self.hybrid_search(clean_query, top_k=top_k * 3)  # Get more for filtering
        else:
            ranked_results = self._tfidf_search(clean_query, top_k=top_k * 3)

        # Get tool filter keywords if tool specified
        filter_keywords = self.get_tool_filter_keywords(tool_filter) if tool_filter else []

        # Filter and collect results
        results = []
        total_chars = 0

        for idx, score in ranked_results:
            if score <= 0:
                continue

            chunk = self.vault_content[idx]
            chunk_lower = chunk.lower()

            # If tool filter is set, chunk must contain at least one tool keyword
            if filter_keywords:
                if not any(kw in chunk_lower for kw in filter_keywords):
                    continue

            # Truncate long chunks
            if len(chunk) > max_chunk_chars:
                chunk = chunk[:max_chunk_chars] + "..."

            # Check total size limit
            if total_chars + len(chunk) > max_total_chars:
                break

            results.append(chunk)
            total_chars += len(chunk)

            if len(results) >= top_k:
                break

        return results

    def get_stats(self):
        """Get statistics about the vault and search indices."""
        embedding_count = sum(1 for e in self.vault_embeddings if e is not None) if self.vault_embeddings else 0
        return {
            'chunks': len(self.vault_content),
            'terms': self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0,
            'has_index': self.tfidf_vectorizer is not None,
            'embeddings': embedding_count,
            'use_embeddings': self.use_embeddings,
            'hybrid_alpha': self.hybrid_alpha
        }

    def add_content(self, new_content):
        """Add new content to vault and rebuild index."""
        with open(self.vault_file, 'a', encoding='utf-8') as f:
            f.write('\n' + new_content)
        self.load_vault()  # Rebuild index


# Global instance
_search_service = None


def get_search_service(vault_file='vault.txt', use_embeddings=None, hybrid_alpha=None):
    """
    Get or create the search service singleton.

    Args:
        vault_file: Path to vault.txt
        use_embeddings: Enable semantic embeddings (default: from env USE_EMBEDDINGS or True)
        hybrid_alpha: Weight for embeddings vs TF-IDF (default: from env HYBRID_ALPHA or 0.5)
    """
    global _search_service
    if _search_service is None:
        # Get settings from environment with defaults
        if use_embeddings is None:
            use_embeddings = os.environ.get('USE_EMBEDDINGS', 'true').lower() == 'true'
        if hybrid_alpha is None:
            hybrid_alpha = float(os.environ.get('HYBRID_ALPHA', '0.5'))

        _search_service = SearchService(
            vault_file=vault_file,
            use_embeddings=use_embeddings,
            hybrid_alpha=hybrid_alpha
        )
        _search_service.load_vault()
    return _search_service


def reset_search_service():
    """Reset the search service singleton (for testing)."""
    global _search_service
    _search_service = None
