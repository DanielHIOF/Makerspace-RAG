"""
Makerspace RAG - LLM Service
Handles all Ollama LLM interactions
"""

import re
import ollama
from app.services.query_classifier import QueryClassifier


class LLMService:
    """Handles all Ollama LLM interactions."""

    # Conversation compression settings
    INCREMENTAL_COMPRESS_EVERY = 6  # Compress every 6 new messages
    RECENT_MESSAGES_KEEP = 10       # Always keep last 10 messages in full

    # Base role for the assistant
    BASE_ROLE = """Du er en veileder og ekspert på Digital Fabrikasjon og HMS (Helse, Miljø og Sikkerhet) ved Makerspace, Høgskolen i Østfold.

Din rolle er å:
- Veilede studenter trygt gjennom bruk av utstyr (3D-printing, laserkutting, CNC, elektronikk, lodding)
- ALLTID prioritere sikkerhet - minn om HMS-regler når relevant
- Gi praktiske, handlingsrettede råd
- Tilpasse forklaringer til brukerens ferdighetsnivå"""

    def __init__(self, model='llama3', small_model='llama3.2:1b'):
        self.model = model
        self.small_model = small_model

    def summarize_messages(self, messages, existing_summary=""):
        """Compress messages into a summary using small model."""
        if not messages:
            return existing_summary

        # Build conversation text for summarization
        conversation_text = ""
        for msg in messages:
            role = "Bruker" if msg.get('role') == 'user' else "Assistent"
            content = msg.get('content', '')[:200]
            conversation_text += f"{role}: {content}\n"

        # Include existing summary in prompt if available
        existing_context = ""
        if existing_summary:
            existing_context = f"\nTIDLIGERE KONTEKST:\n{existing_summary}\n"

        prompt = f"""Lag en KORT oppsummering (2-3 setninger) av samtalen.
Behold: hovedtema, viktige beslutninger, spesifikke detaljer (utstyr, innstillinger, problemer).
{existing_context}
NYE MELDINGER:
{conversation_text}

OPPSUMMERING:"""

        try:
            response = ollama.chat(
                model=self.small_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3, 'num_predict': 100}
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"  [WARN] Compression failed: {e}")
            return existing_summary

    def chat(self, query, context, is_inventory=False, conversation_history=None, existing_summary=""):
        """Send query + context to LLM with conversation history support.
        Returns tuple: (response_text, updated_summary)
        """
        level, level_instruction = QueryClassifier.detect_level(query)
        language, language_instruction = QueryClassifier.detect_language(query)
        category_mode = QueryClassifier.detect_category_mode(query)

        # Clean query of all command prefixes
        clean_query = re.sub(
            r'/(nybegynner|beginner|ekspert|expert|norsk|no|english|en|prusa|3d|laser|cnc|elektronikk|lodding)\s*',
            '', query, flags=re.IGNORECASE
        ).strip()

        # Add category mode instruction if present
        category_instruction = ""
        if category_mode:
            category_instruction = category_mode.get('instruction', '')

        # Classify the query
        detected_tool = QueryClassifier.detect_tool(clean_query)
        tool_hint = f" (Verktøy: {detected_tool})" if detected_tool else ""

        # Keep context short
        context_text = context[:2000] if context else ""

        # Build system prompt based on query type
        if is_inventory:
            system_prompt = self._build_inventory_prompt(context_text, level_instruction, language_instruction)
        else:
            system_prompt = self._build_chat_prompt(
                context_text, level_instruction, language_instruction,
                category_instruction, tool_hint
            )

        # Build messages array for Ollama
        messages = [{'role': 'system', 'content': system_prompt}]

        # Handle conversation history with incremental compression
        updated_summary = existing_summary

        if conversation_history:
            total_messages = len(conversation_history)

            # Add existing summary as context
            if existing_summary:
                messages.append({
                    'role': 'system',
                    'content': f"TIDLIGERE I SAMTALEN:\n{existing_summary}"
                })

            # Only keep last RECENT_MESSAGES_KEEP messages in full
            recent_messages = conversation_history[-self.RECENT_MESSAGES_KEEP:] if total_messages > self.RECENT_MESSAGES_KEEP else conversation_history

            # Check if we need to compress
            messages_since_last_compress = total_messages % self.INCREMENTAL_COMPRESS_EVERY
            if total_messages >= self.INCREMENTAL_COMPRESS_EVERY and messages_since_last_compress == 0:
                compress_start = max(0, total_messages - self.RECENT_MESSAGES_KEEP - self.INCREMENTAL_COMPRESS_EVERY)
                compress_end = total_messages - self.RECENT_MESSAGES_KEEP
                if compress_end > compress_start:
                    to_compress = conversation_history[compress_start:compress_end]
                    print(f"  [COMPRESS] Inkrementell komprimering av {len(to_compress)} meldinger")
                    updated_summary = self.summarize_messages(to_compress, existing_summary)

            # Add recent messages in full
            for msg in recent_messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role in ('user', 'assistant') and content:
                    messages.append({'role': role, 'content': content})

        # Add current query as the last user message
        messages.append({'role': 'user', 'content': clean_query})

        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={'temperature': 0.7, 'num_predict': 500}
            )
            return response['message']['content'], updated_summary
        except Exception as e:
            error_msg = str(e)
            print(f"  [ERROR] OLLAMA ERROR: {error_msg}")
            raise e

    def _build_inventory_prompt(self, context_text, level_instruction, language_instruction):
        """Build system prompt for inventory queries."""
        return f"""{self.BASE_ROLE}

TILGJENGELIG UTSTYR:
{context_text}

FERDIGHETSNIVÅ: {level_instruction}

SPRÅK: {language_instruction}

REGLER FOR SVAR:
- List kun utstyret som er relevant
- Maks 2-3 linjer per utstyr (navn, lokasjon, nivå)
- Nevn eventuelle HMS-krav eller opplæringskrav
- Avslutt med: "Vil du vite mer om noe av dette?"
- VIKTIG: Husk samtalehistorikken"""

    def _build_chat_prompt(self, context_text, level_instruction, language_instruction,
                           category_instruction, tool_hint):
        """Build system prompt for chat queries."""
        category_section = f"\n\nKATEGORI-MODUS:\n{category_instruction}" if category_instruction else ""

        return f"""{self.BASE_ROLE}{tool_hint}

RELEVANT INFORMASJON:
{context_text}

FERDIGHETSNIVÅ: {level_instruction}

SPRÅK: {language_instruction}{category_section}

KRITISK FOR KOMPONENTER:
- Bruk informasjonen fra "KOMPONENTER FUNNET" men SKRIV NATURLIG
- IKKE bruk "@" eller list-format fra konteksten
- GODT: "Vi har motstander på Komponentvegg, blant annet 10Ω, 15Ω og 100Ω."
- DÅRLIG: "10Ω @ Komponentvegg, 15Ω @ Komponentvegg..."
- Nevn lokasjonen ÉN gang, så list noen eksempler

FORMATERING:
- Bruk "-" for kulepunkt (ikke *)
- ALDRI bruk **bold** eller *italic* - det rendres ikke riktig
- Nummererte lister (1. 2. 3.) er OK når rekkefølge betyr noe
- Links er OK: [tekst](url)

REGLER FOR SVAR:
- Svar KORT (2-4 setninger) men informativt
- Gi ETT konkret tips eller neste steg
- Hvis relevant: minn om HMS/sikkerhet (verneutstyr, farlige materialer, osv.)
- Still et oppfølgingsspørsmål for å holde samtalen i gang
- VIKTIG: Husk hva dere har snakket om tidligere i samtalen"""

    def generate_with_small_model(self, prompt, max_tokens=200, temperature=0.3):
        """Generate text using the small/fast model."""
        try:
            response = ollama.chat(
                model=self.small_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': temperature, 'num_predict': max_tokens}
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"  [ERROR] Small model error: {e}")
            return None

    def check_ollama_status(self):
        """Check if Ollama is running and models are available."""
        try:
            models = ollama.list()
            # Handle different API response formats
            model_list = models.get('models', []) if isinstance(models, dict) else []
            model_names = []
            for m in model_list:
                if isinstance(m, dict):
                    model_names.append(m.get('name', m.get('model', '')))
                elif hasattr(m, 'name'):
                    model_names.append(m.name)
            return {
                'running': True,
                'models': model_names,
                'has_main_model': any(self.model in m for m in model_names),
                'has_small_model': any(self.small_model in m for m in model_names)
            }
        except Exception as e:
            return {
                'running': False,
                'error': str(e),
                'models': [],
                'has_main_model': False,
                'has_small_model': False
            }


# Global instance
_llm_service = None


def get_llm_service(model='llama3', small_model='llama3.2:1b'):
    """Get or create the LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(model, small_model)
    return _llm_service
