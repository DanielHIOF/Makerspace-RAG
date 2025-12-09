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

    def chat(self, query, context, is_inventory=False, conversation_history=None, existing_summary="", needs_wiring_diagram=False):
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
                category_instruction, tool_hint, needs_wiring_diagram
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

SVARFORMAT - FØLG DETTE NØYAKTIG:
1. Start med en kort intro-setning
2. List utstyr med **fet skrift navn** etterfulgt av kolon og beskrivelse
3. Nevn lokasjon i eget avsnitt
4. Tilby mer hjelp
5. Avslutt med oppfølgingsspørsmål

===== EKSEMPEL 1 - UTSTYRSLISTE =====
BRUKER: Hvilke 3D-printere har dere?
SVAR:
Vi har flere 3D-printere i vår fabrikasjonssal! Her er de mest populære modellene:

**Prusa Mini+**: En beginner-friendly 3D-printer, perfekt for nybegynnere. Lett å bruke og gir gode resultater.

**Prusa MK3s**: En populær modell med stor skiveflate. Kan printe med flere forskjellige materialer.

**Ultimaker 3 Extended**: En intermediate-level printer for deg med litt erfaring.

Disse modellene finner du i D1-044.

Vil du vite mer om noen av disse?
===== SLUTT EKSEMPEL 1 =====

===== EKSEMPEL 2 - MED LISTE =====
BRUKER: Hva er HMS-reglene for lodding?
SVAR:
For lodding må du følge disse HMS-reglene:

- Bruk alltid avtrekk/ventilasjon - lodderøyk er helseskadelig
- La loddekolben avkjøles i holderen, aldri på bordet
- Vask hendene etter lodding - bly er giftig
- Rengjør loddetuppen regelmessig med svamp

Du finner **loddestasjonene** i D1-044. Husk å ta HMS-kurset først.

Har du tatt HMS-kurset for lodding?
===== SLUTT EKSEMPEL 2 =====

KRITISKE REGLER:
- ALDRI bruk emojis (ingen smilefjes, ikoner, symboler)
- ALDRI bruk * for kulepunkter - KUN bruk - (bindestrek)
- Bruk **doble stjerner** rundt utstyrsnavn for fet skrift
- Maks 3-4 setninger per avsnitt
- Avslutt ALLTID med et oppfølgingsspørsmål

FEIL FORMAT (IKKE GJØR DETTE):
* Kulepunkt med stjerne (FEIL)
:sparkles: Emoji (FEIL)
Prusa Mini+ uten fet skrift (FEIL)

RIKTIG FORMAT:
- Kulepunkt med bindestrek (RIKTIG)
**Prusa Mini+**: med fet skrift (RIKTIG)"""

    def _build_chat_prompt(self, context_text, level_instruction, language_instruction,
                           category_instruction, tool_hint, needs_wiring_diagram=False):
        """Build system prompt for chat queries."""
        category_section = f"\n\nKATEGORI-MODUS:\n{category_instruction}" if category_instruction else ""

        # Add wiring diagram instructions if needed
        wiring_section = ""
        if needs_wiring_diagram:
            wiring_section = """

KOBLINGSSKJEMA (VIKTIG!):
Når brukeren spør om å koble komponenter, INKLUDER ALLTID et koblingsskjema i JSON-format.
Bruk denne nøyaktige syntaksen med tre backticks og "wiring-json":

```wiring-json
{
  "board": "arduino_uno",
  "title": "Beskrivende tittel",
  "components": [
    {"type": "led", "id": "led1", "color": "red"},
    {"type": "resistor", "id": "r1", "value": "220"}
  ],
  "connections": [
    {"from": "D13", "to": "r1.1", "color": "orange"},
    {"from": "r1.2", "to": "led1.anode", "color": "orange"},
    {"from": "led1.cathode", "to": "GND", "color": "black"}
  ]
}
```

REGLER FOR KOBLINGSSKJEMA:
- board: "arduino_uno", "arduino_nano", eller "esp32"
- Komponent-typer: led, resistor, button, potentiometer, sensor, servo, motor, relay, buzzer, display, capacitor
- Arduino pins: D0-D13, A0-A5, 5V, 3.3V, GND, VIN, RESET
- Komponent-pins: .1, .2 (for resistor), .anode/.cathode (for LED/diode), .signal/.vcc/.gnd (for sensorer)
- Wire-farger: red, black, orange, yellow, green, blue, purple, white, gray
- INKLUDER ALLTID motstand (220Ω) før LED-er!
- GND og 5V/3.3V må alltid være med når nødvendig"""

        return f"""{self.BASE_ROLE}{tool_hint}

RELEVANT INFORMASJON:
{context_text}

FERDIGHETSNIVÅ: {level_instruction}

SPRÅK: {language_instruction}{category_section}{wiring_section}

SVARFORMAT - FØLG DETTE NØYAKTIG:
1. Start med en kort intro-setning
2. Når du nevner utstyr/verktøy, bruk **fet skrift navn** etterfulgt av kolon og beskrivelse
3. Grupper relatert info i egne avsnitt
4. Nevn HMS/sikkerhet når relevant
5. Avslutt med oppfølgingsspørsmål

===== EKSEMPEL 1 - UTSTYRSSPØRSMÅL =====
BRUKER: Hvilke 3D-printere har dere?
SVAR:
Vi har flere 3D-printere i vår fabrikasjonssal! Her er de mest populære:

**Prusa Mini+**: En beginner-friendly 3D-printer, perfekt for nybegynnere. Lett å bruke og gir gode resultater.

**Prusa MK3s**: En populær modell med stor skiveflate. Kan printe med flere materialer.

Disse finner du i D1-044.

Vil du vite mer om noen av disse?
===== SLUTT EKSEMPEL 1 =====

===== EKSEMPEL 2 - HMS MED LISTE =====
BRUKER: Hva er HMS-reglene for lodding?
SVAR:
For lodding må du følge disse HMS-reglene:

- Bruk alltid avtrekk/ventilasjon - lodderøyk er helseskadelig
- La loddekolben avkjøles i holderen, aldri på bordet
- Vask hendene etter lodding - bly er giftig
- Rengjør loddetuppen regelmessig med svamp

Du finner **loddestasjonene** i D1-044. Husk å ta HMS-kurset først.

Har du tatt HMS-kurset for lodding?
===== SLUTT EKSEMPEL 2 =====

===== EKSEMPEL 3 - KOMPONENTER =====
BRUKER: Har dere motstander?
SVAR:
Ja, vi har **motstander** tilgjengelig på Komponentveggen i D1-044.

Du finner et bredt utvalg verdier:

- 220Ω (for LED-kretser)
- 1kΩ og 10kΩ (for generell bruk)
- 100Ω og 470Ω (for strømbegrensning)

Hva slags prosjekt skal du bruke motstandene til?
===== SLUTT EKSEMPEL 3 =====

===== EKSEMPEL 4 - KORT SVAR =====
BRUKER: Hvor finner jeg laserkutteren?
SVAR:
**Laserkutteren** finner du i D1-043.

Husk at du må ha godkjent HMS-kurs før du kan bruke den. Kurset tar ca. 30 minutter.

Har du tatt laserkutter-kurset?
===== SLUTT EKSEMPEL 4 =====

KRITISKE REGLER:
- ALDRI bruk emojis (ingen smilefjes, ikoner, symboler)
- ALDRI bruk * for kulepunkter - KUN bruk - (bindestrek)
- Bruk **doble stjerner** rundt utstyrsnavn for fet skrift
- Maks 3-4 setninger per avsnitt
- Avslutt ALLTID med et oppfølgingsspørsmål
- Bruk KORREKT norsk rettskrivning (først, ikke forst; gjør, ikke gjor; etc.)

FEIL FORMAT (IKKE GJØR DETTE):
* Kulepunkt med stjerne (FEIL)
:sparkles: Emoji (FEIL)
Prusa Mini+ uten fet skrift (FEIL)
1. Nummerert liste for ikke-sekvensielle ting (FEIL)
forst, gjor, nar (FEIL - mangler ø/å)

RIKTIG FORMAT:
- Kulepunkt med bindestrek (RIKTIG)
**Prusa Mini+**: med fet skrift og kolon (RIKTIG)
først, gjør, når (RIKTIG - korrekt norsk)"""

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
