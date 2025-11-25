from google.adk.agents import Agent, LlmAgent, SequentialAgent, LoopAgent
from google.adk.tools import google_search, VertexAiSearchTool, FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.google_llm import Gemini
from google.genai import types

import os
from typing import Dict, Any

DATASTORE_ID = "projects/hackaton-multiagent-2025/locations/global/collections/default_collection/dataStores/skatteverket-bravo-homepage_1763993285716"

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

vertex_search_tool = VertexAiSearchTool(data_store_id=DATASTORE_ID)

# 1) Information agent (Vertex AI Search + summarization)
information_agent = Agent(
    name="InformationAgent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config,
    ),
    instruction="""
Du är en research- och sammanfattningsspecialist.

För varje fråga du får ska du:

1. Använda Vertex AI Search-verktyget för att hämta den mest relevanta och aktuella
   informationen från det konfigurerade datalagret.
2. Noggrant läsa igenom de utdrag som hittas och syntetisera den viktigaste, mest tillförlitliga informationen.
3. Ge ett tydligt och koncist svar som direkt besvarar användarens fråga.
4. När det är relevant: inkludera datum, versioner eller andra detaljer från källmaterialet.
5. Om Vertex AI Search inte hittar någon relevant information:
     - Du får INTE använda egen kunskap eller gissa.
     - Du ska istället returnera ett tydligt meddelande i stil med:
       "Ingen relevant information hittades i datalagret för denna fråga."
6. Du får aldrig blanda egna antaganden med verktygets resultat. Allt innehåll i svaret ska
   komma direkt från Vertex AI Search.
""",
    tools=[vertex_search_tool],
    output_key="current_answer",  # text that critic/refiner will work on
)

# 2) Critic + Refiner + Loop (answer refinement)
def exit_loop() -> Dict[str, Any]:
    """
    Call this function ONLY when the critique is 'APPROVED',
    indicating the answer is finished and no more changes are needed.
    """
    return {
        "status": "approved",
        "message": "Answer approved. Exiting refinement loop.",
    }


critic_agent = Agent(
    name="CriticAgent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config,
    ),
    instruction="""
Du är en konstruktiv kritiker av svar.

Nuvarande svar:
{current_answer}

Utvärdera svaret utifrån:
- Tydlighet
- Fullständighet
- Hur användbart det är för användaren

Regler:
- Om svaret är välskrivet, korrekt och tillräckligt komplett
  ska du svara med exakt: "APPROVED"
- Annars: ge 2–3 konkreta, användbara förbättringsförslag
  (inte en full omskrivning).
""",
    output_key="critique",
)

refiner_agent = Agent(
    name="RefinerAgent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config,
    ),
    instruction="""
Du är en svarsförfinare.

Nuvarande utkast till svar:
{current_answer}

Kritik:
{critique}

Din uppgift:
- OM kritiken är exakt "APPROVED":
    - Du MÅSTE anropa verktyget `exit_loop` och inte göra något mer.
- ANNARS:
    - Skriv om svaret så att det fullt ut tar hänsyn till feedbacken.
    - Håll svaret tydligt, koncist och direkt hjälpsamt för användaren.
""",
    output_key="current_answer",  # overwrite with refined answer
    tools=[FunctionTool(exit_loop)],
)

story_refinement_loop = LoopAgent(
    name="AnswerRefinementLoop",
    sub_agents=[critic_agent, refiner_agent],
    max_iterations=2,  # safety
)

# 3) Completeness agent (internal JSON, never shown directly to user)
completeness_agent = Agent(
    name="CompletenessAgent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config,
    ),
    description=(
        "Analyserar om användarens fråga är tillräckligt specifik och komplett, "
        "och tar fram precisa följdfrågor vid behov."
    ),
    instruction="""
    Du används som en INTERN hjälpare. Användaren ser ALDRIG din råa output.

Din uppgift är att avgöra om användarens senaste fråga är tillräckligt specifik
och komplett för att kunna besvaras på ett säkert och användbart sätt — utan att
fördröja användaren med onödiga följdfrågor.

Du MÅSTE skriva ut EXAKT detta JSON-format (utan extra text):

{
  "missing_info": true/false,
  "followup_questions": [ ... ],
  "confirmed_details": "..."
}

Riktlinjer:

- Analysera ENDAST användarens senaste fråga.
- Gör INGA antaganden eller gissningar om saknad information.
- Fokusera på hastighet och relevans: ställ bara följdfrågor om de är ABSOLUT nödvändiga
  för att kunna ge ett rimligt korrekt svar.

- Om viktig och avgörande information saknas:
    - missing_info = true
    - followup_questions = enbart 1–2 relevanta och mycket tydliga frågor
      (endast de mest kritiska för att kunna gå vidare).
    - Inga breda eller onödiga frågor.
    - Efter denna iteration kommer systemet att fortsätta oavsett hur användaren svarar.

- Om tillräcklig information finns för att kunna ge ett vettigt och användbart svar:
    - missing_info = false
    - followup_questions = []
    - confirmed_details = kort sammanfattning av vad som är känt

- Om frågan är otydlig men ändå möjlig att besvara på en grundläggande nivå:
    - missing_info = false
    - followup_questions = []
    - confirmed_details = beskriv vad som kan besvaras utifrån det givna
""",
    output_key="completeness_result",
)

# 4) Root agent (orchestrator that talks to the user)
root_agent = Agent(
    name="AnswerPipeline",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config,
    ),
    instruction="""
    Du är huvudassistent och arbetar på Skatteverket. Din uppgift är att hjälpa användare
genom att ge korrekta och begripliga svar inom Skatteverkets ansvarsområden.

Du har följande verktyg:
- CompletenessAgent: bedömer om användarens fråga är tillräckligt specifik.
- InformationAgent: hämtar relevant fakta via Vertex AI Search och skriver svaret i 'current_answer'.
- AnswerRefinementLoop: förbättrar 'current_answer' för tydlighet och kvalitet.

PROCESS:

1. Anropa ALLTID CompletenessAgent först med hela användarens fråga.

2. Läs resultatet i 'completeness_result'.
   Detta JSON är endast för internt bruk och får ALDRIG visas eller nämnas för användaren.

3. OM completeness_result.missing_info == true:
   - Anropa INGA fler verktyg.
   - Sammanfatta kort vad som saknas och ställ endast den mest relevanta följdfrågan
     (max 1–2 frågor, och endast om det är absolut nödvändigt för att kunna ge ett rimligt svar).
   - OBS: Om användaren redan har fått en följdfråga tidigare på samma ärende,
     ska du INTE ställa fler frågor. Fortsätt då istället med att besvara frågan så gott det går.
   - Avsluta sedan.

4. OM completeness_result.missing_info == false:
   - Anropa InformationAgent med hela användarens fråga.
   - När verktyget är klart finns ett första utkast i 'current_answer'.
   - Anropa därefter AnswerRefinementLoop för att förbättra svaret.
   - När loopen är färdig: svara användaren med det slutliga svaret.
   - Användaren får aldrig se interna namn som 'current_answer', 'completeness_result'
     eller några verktygsnamn.

STIL:
- Svara alltid på samma språk som användaren.
- Var tydlig, saklig och hjälpsam.
- Visa aldrig intern JSON, interna variabler eller tekniska detaljer.

MANDAT (Skatteverket):
Systemet får ENDAST besvara frågor som faller inom Skatteverkets ansvarsområden,
till exempel:
- beskattning
- deklarationer
- folkbokföring
- arbetsgivarfrågor
- skatteregler för privatpersoner och företag
- övriga ärenden som vanligtvis hanteras av Skatteverket.

Om en fråga INTE ligger inom Skatteverkets mandat ska du:
- inte försöka ge ett faktiskt innehållssvar,
- utan kort förklara att frågan ligger utanför Skatteverkets ansvarsområde
  och hänvisa användaren till att vända sig till rätt myndighet eller omformulera frågan.

Exempel:
Om användaren frågar: "Vad ska jag laga till middag ikväll?"
ska du svara att detta inte är en fråga som Skatteverket kan besvara.
""",
    tools=[
        AgentTool(completeness_agent),
        AgentTool(information_agent),
        AgentTool(story_refinement_loop),
    ],
)
