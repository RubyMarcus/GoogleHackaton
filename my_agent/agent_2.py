from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.tools import google_search, VertexAiSearchTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.google_llm import Gemini
from google.genai import types

DATASTORE_ID = "projects/hackaton-multiagent-2025/locations/global/collections/default_collection/dataStores/skatteverket-bravo-homepage_1763993285716"

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

completeness_agent = Agent(
    name="CompletenessAgent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config
    ),
    description=(
        "Ensures that no specialist agent or the root agent proceeds with "
        "incomplete, ambiguous, or assumption-based input. "
        "Identifies missing required information and generates precise clarification questions. "
        "Never guesses or infers missing data."
    ),
    instruction=(
        "Your task is to analyze the user's question and the information from the 'information_agent' to determine exactly what information is missing "
        "for a correct and safe answer. "
        "Do not make assumptions. "
        "Don't ask redundant question, or question that already had been answered before."
        "If information is missing, formulate specific follow-up questions. "
        "If all necessary information is present, confirm it and list the details.")
)

information_agent = Agent(
    name="InformationAgent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config
    ),
    instruction="""
You are a research and summarization specialist.

For every user question you receive:

1. Use the `VertexAiSearchTool` tool to look up the most relevant, up-to-date information.
2. Read the search results and synthesize the key information.
3. Return your answer as:
   - Summarize the main findings.
   - Include dates / versions when they matter.
4. Do *not* answer purely from your own knowledge if search results are available;
   ground your answer in what the tool returns.
""",
    tools=[VertexAiSearchTool(data_store_id=DATASTORE_ID)],      # only here, not on root_agent
    #output_key="final_summary", # what root_agent will read
)

root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config
    ),
    description="Top-level assistant that routes questions to InformationAgent.",
    instruction="""
You are the entrypoint assistant.

Behavior:
1. For every user message, call the `information_agent` tool with the user's full question.
2. Wait for the tool result.
3. Then ask the completeness_agent to see if the answer and question is complete, else return with at follow-up question to the user.
3. Take the `final_summary` field from the tool output and present it to the user.
4. Do not answer questions directly yourself; always delegate to `information_agent`.

If the tool call fails, briefly explain the error to the user.
""",
    tools=[AgentTool(information_agent), AgentTool(completeness_agent)],
    #tools=[AgentTool(information_agent)],  # ⬅️ ONLY this tool
    # NOTE: do NOT set tools=[google_search] here
    # and do NOT also list information_agent in sub_agents at the same time.
)
