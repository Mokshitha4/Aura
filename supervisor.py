# supervisor.py
# This file contains the Supervisor class, built with LangGraph.
# It acts as an intelligent router and conductor for a multi-agent system.

# --- SETUP INSTRUCTIONS ---
# 1. Ensure 'aura_brain.py' and 'agent.py' are in the same directory.
# 2. Update your requirements.txt and run 'pip install -r requirements.txt'.
# 3. Set your environment variables: OPENAI_API_KEY, TAVILY_API_KEY, QLOO_API_KEY.
# 4. Run this file directly to see a demonstration: 'python supervisor.py'

import os
import json
import logging
import re
import uuid
from typing import TypedDict, List, Annotated
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from langchain_core.messages import ToolMessage
from langgraph.graph.message import add_messages

# Import our existing brain and memory agent
from aura_brain import AuraBrain
from aura_agent import AuraAgent 

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- INITIALIZE THE SPECIALIST AGENT AND ITS TOOLS ---
try:
    brain = AuraBrain()
    memory_agent = AuraAgent(brain) 
except Exception as e:
    logging.error(f"Could not initialize the core brain/agent. Exiting. Error: {e}")
    exit()

# We now wrap the MemoryAgent's functions as "Tools" that the Supervisor can call.
@tool
def update_knowledge_base(extracted_graph: dict) -> str:
    """Use this tool to save a pre-extracted graph of entities and relationships to the personal knowledge base. The 'extracted_graph' should be a dictionary with 'nodes' and 'edges' keys."""
    return memory_agent.update_knowledge_base(extracted_graph)

@tool
def query_knowledge_base(query: str) -> str:
    """Use this tool when the user asks a question about themselves or information they've previously provided. Searches the personal knowledge base."""
    return memory_agent.query_knowledge_base(query)

@tool
def qloo_enrichment(entity_name: str, entity_type: str) -> str:
    """Use this tool to find culturally related recommendations for a given entity (like a Movie, Book, or Artist) and add them to the knowledge base."""
    return memory_agent.qloo_enrichment(entity_name, entity_type)

# The Supervisor also gets its own tool for external searches using the real Tavily tool.
if not os.getenv("TAVILY_API_KEY"):
    logging.warning("TAVILY_API_KEY not set. External search will be limited.")
    @tool
    def tavily_search(query: str) -> str:
        """Use this to find general, public information about a topic (e.g., a movie, a person, a concept)."""
        logging.info(f"SUPERVISOR TOOL (Fallback): Performing external search for '{query}'")
        return memory_agent.external_search(query)
    tavily_tool = tavily_search
else:
    tavily_tool = TavilySearch(max_results=5, topic="general")
    tavily_tool.name = "tavily_search" 
    tavily_tool.description = "Use this to find general, public information about a topic (e.g., a movie, a person, a concept)."


# --- LANGGRAPH STATE DEFINITION ---
class AgentState(TypedDict):
    input: str
    plan: List[dict] 
    past_steps: Annotated[list, add_messages]
    response: str

# --- LANGGRAPH NODES (The "Workers") ---
def planner_node(state: AgentState):
    """This node creates the initial plan AND extracts the graph if necessary."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # *** FIX APPLIED: Added explicit instruction for handling questions. ***
    prompt = f"""
    You are an intelligent orchestrator for a personal AI assistant named Aura.
    Your job is to analyze the user's input and create a step-by-step plan.
    You have access to: `update_knowledge_base`, `query_knowledge_base`, `tavily_search`, `qloo_enrichment`.

    1.  **Analyze Intent:** Determine if the user is providing information or asking a question.

    2.  **Create a Plan:**
        -   **If providing information:**
            a. Extract entities and relationships into a JSON object. Each node MUST have "content" and "type" keys.
            b. Your first step MUST be a call to `update_knowledge_base`. The 'arg' for this tool MUST be a dictionary with a single key "extracted_graph", whose value is the extracted graph.
            c. **CRITICAL RULE:** If the extracted graph contains a cultural entity (e.g., Movie, Book, Artist), you MUST add a second step to call `qloo_enrichment`. The 'arg' for this tool MUST be a dictionary with "entity_name" and "entity_type" keys (e.g., {{"entity_name": "Oppenheimer", "entity_type": "movie"}}).
        -   **If asking a question:**
            a. Create a plan that starts with `query_knowledge_base`. The 'arg' for this tool MUST be a dictionary with a single key "query", whose value is the user's original question.
            b. If needed, add a second step to use `tavily_search` with a simple string as the 'arg'.

    User Input: "{state['input']}"

    Respond with a single, valid JSON object with a key "plan".
    """
    
    try:
        response = llm.invoke(prompt)
        cleaned_content = response.content.strip().replace("```json", "").replace("```", "").strip()
        plan_data = json.loads(cleaned_content)
        plan = plan_data.get("plan", [])
    except (json.JSONDecodeError, AttributeError) as e:
        logging.error(f"Planner failed to generate a valid JSON plan. Output: {response.content if 'response' in locals() else 'No response'}. Error: {e}")
        plan = []
    return {"plan": plan}

def execution_node(state: AgentState):
    """This node executes the next step in the plan."""
    if not state.get("plan"):
        return {"past_steps": []}

    plan = state["plan"]
    step = plan[0]
    remaining_plan = plan[1:]
    
    tool_name = step["tool"]
    tool_arg = step["arg"]
    
    all_tools = {
        "update_knowledge_base": update_knowledge_base,
        "query_knowledge_base": query_knowledge_base,
        "tavily_search": tavily_tool,
        "qloo_enrichment": qloo_enrichment
    }
    
    result = all_tools[tool_name].invoke(tool_arg)

    tool_message = ToolMessage(
        content=str(result), 
        name=tool_name,
        tool_call_id=str(uuid.uuid4())
    )
    
    return {
        "plan": remaining_plan,
        "past_steps": [tool_message]
    }

def response_synthesizer_node(state: AgentState):
    """This node creates the final, human-readable response."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    prompt = f"""
    You are Aura, a personal AI assistant with a friendly and insightful personality. Your goal is to be a true partner to the user, not just an information retriever. You have just executed a plan to handle a user's request.

    The user's original input was: "{state['input']}"

    Here are the results from the tools you used:
    {state['past_steps']}

    Your task is to synthesize these results into a single, helpful, and conversational response.
    - **Do not just list the tool outputs.** Weave them into a natural narrative.
    - **Connect the dots.** If you found information from multiple sources (e.g., their personal knowledge base and an external search), highlight the interesting connections.
    - **Be proactive.** End your response by proposing a logical next step, asking a clarifying question, or offering to perform a related task. Be creative and show you understand their underlying goal.
    """
    
    response = llm.invoke(prompt)
    return {"response": response.content}

# --- LANGGRAPH CONDITIONAL LOGIC ---
def should_continue(state: AgentState):
    """Determines whether to continue the plan or finish."""
    if state.get("plan") and len(state["plan"]) > 0:
        return "execute"
    else:
        return "respond"

# --- BUILD THE GRAPH ---
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", execution_node)
workflow.add_node("responder", response_synthesizer_node)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "executor")
workflow.add_conditional_edges(
    "executor",
    should_continue,
    {
        "execute": "executor",
        "respond": "responder"
    }
)
workflow.add_edge("responder", END)

# Compile the graph into a runnable app
supervisor_app = workflow.compile()


if __name__ == '__main__':
    print("--- Aura Multi-Agent System Initialized ---")
    print("This system uses LangGraph to orchestrate tasks between a memory agent and external search tools.")
    
    # query = "My friend Sarah recommended the movie Oppenheimer."
    query = "I am thinking of watching Dune. How is it?"
    
    print(f"\n[USER INPUT] '{query}'")
    
    events = supervisor_app.stream({"input": query})
    
    final_response = ""
    for event in events:
        if "planner" in event:
            print("\n--- Planner Node ---")
            print(f"Plan created: {event['planner']['plan']}")
        if "executor" in event:
            print("\n--- Executor Node ---")
            print(f"Executed step, result: {event['executor']['past_steps']}")
        if "responder" in event:
            print("\n--- Responder Node ---")
            final_response = event['responder']['response']

    print("\n-------------------------")
    print(f"[AURA'S FINAL RESPONSE] {final_response}")
    print("-------------------------")
