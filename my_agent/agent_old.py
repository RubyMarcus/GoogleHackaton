from google.adk.agents.llm_agent import Agent
from google.adk.tools import google_search
from google.adk.tools import AgentTool

information_agent = Agent(
    name="InformationAgent",
    model="gemini-2.5-flash",
    # The instruction is modified to request a bulleted list for a clear output format.
    instruction="""You are a specialized information gathering agent. Your only job is to use the
    google_search tool to find relevant information to the user.
    
    Then summarize the information.""",
    tools=[google_search],
    output_key="information_summary",
)


root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='A helpful assistant for user questions. Y',
    instruction="""You are an helpful assitant for user questions. Your goal is to answer the user's query by orchestrating a workflow.
1. First, you MUST call the `information_agent` sub_agent to find relevant information on the topic provided by the user.
2. Next, after receiving the information findings, you MUST call the `summarizer_agent` tool to create a concise summary.
3. Finally, present the final summary clearly to the user as your response.""",
    sub_agents=[information_agent],
    #tools=[AgentTool(information_agent), AgentTool(summarizer_agent)],
    #tools=[google_search]
)

