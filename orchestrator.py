# orchestrator.py
# This file contains the AuraOrchestrator, which acts as a "supervisor" or "router".
# It analyzes user input to determine intent and then delegates tasks to the AuraAgent.

import logging
from aura_agent import AuraAgent
from aura_brain import AuraBrain

class AuraOrchestrator:
    """
    The orchestration layer that directs the flow of information and tasks.
    It sits on top of the AuraAgent and decides which of the agent's tools to use.
    """
    def __init__(self, agent: AuraAgent):
        if not isinstance(agent, AuraAgent):
            raise TypeError("Orchestrator must be initialized with an instance of AuraAgent.")
        self.agent = agent
        self.openai_client = agent.openai_client # Reuse the client
        logging.info("AuraOrchestrator initialized and connected to AuraAgent.")

    def handle_request(self, text: str):
        """
        The main entry point for any user request.
        It determines the user's intent and routes the request to the appropriate function.
        """
        # 1. Determine the user's intent using an LLM call.
        intent = self._determine_intent(text)
        logging.info(f"Determined intent for request: '{intent}'")

        # 2. Route the request based on the determined intent.
        if intent == "ASK_QUESTION":
            # In the tool-based model, we would have the orchestrator call the tool.
            # For this router model, we call the agent's query function directly.
            return self.agent.query_knowledge_base(text) # Assuming answer_query was renamed to query_knowledge_base
        
        elif intent == "ADD_COMPLEX_KNOWLEDGE":
            # *** FIX APPLIED HERE ***
            # The function in the new agent.py is called 'update_knowledge_base'.
            return self.agent.update_knowledge_base(text)
            
        elif intent == "ADD_SIMPLE_NOTE":
            # For simple notes, we bypass the complex extraction and just add a node.
            node_id = self.agent.brain.add_node(text, "Note")
            self.agent.brain.save()
            return f"Got it. I've saved that as a simple note with ID {node_id}."
            
        else: # Fallback for unknown intent
            logging.warning(f"Unknown intent '{intent}'. Defaulting to complex knowledge addition.")
            # *** FIX APPLIED HERE ***
            return self.agent.update_knowledge_base(text)

    def _determine_intent(self, text: str):
        """
        Uses an LLM to classify the user's intent into one of three categories.
        """
        # This prompt is designed to force the LLM to make a clear choice.
        prompt = f"""
        Analyze the user's input and classify its primary intent. Choose one of the following three categories:

        1.  **ADD_SIMPLE_NOTE**: Use this if the input is a short, simple statement, a personal reflection, a fleeting thought, or a direct command to remember something simple.
            Examples: "Remember that I like sci-fi movies.", "Idea: a personal AI assistant.", "The meeting is at 3 PM."

        2.  **ADD_COMPLEX_KNOWLEDGE**: Use this if the input is a longer piece of text, a quote, a paragraph from an article, or a statement that contains multiple distinct facts and relationships that should be broken down.
            Examples: "My friend Sarah, who lives in Tokyo, recommended a great book about sci-fi world-building.", "Dune is a 2021 film directed by Denis Villeneuve."

        3.  **ASK_QUESTION**: Use this if the input is a question asking for information, a summary, or a recommendation from the knowledge base.
            Examples: "What do you know about my travel plans?", "Who is Sarah?", "What should I watch tonight?"

        Based on these definitions, classify the following text. Respond with ONLY the category name (e.g., ADD_SIMPLE_NOTE).

        Text: "{text}"
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0 # We want a deterministic classification
            )
            intent = response.choices[0].message.content.strip()
            # Basic validation to ensure it's one of our expected categories
            if intent in ["ADD_SIMPLE_NOTE", "ADD_COMPLEX_KNOWLEDGE", "ASK_QUESTION"]:
                return intent
            else:
                return "UNKNOWN"
        except Exception as e:
            logging.error(f"Failed to determine intent: {e}")
            return "UNKNOWN" # Default to a safe fallback

# if __name__ == '__main__':
#     # --- DEMONSTRATION of the Orchestrator ---
#     print("--- Initializing the full Aura system (Brain, Agent, Orchestrator) ---")
    
#     # We need to initialize the brain and agent first
#     brain = AuraBrain()
#     agent = AuraAgent(brain)
#     orchestrator = AuraOrchestrator(agent)

#     print("\n--- Testing the Orchestrator's Intent Routing ---")

#     # Example 1: A complex statement
#     text1 = "My friend Sarah, who lives in Tokyo, recommended the movie Dune."
#     print(f"\n[INPUT] '{text1}'")
#     orchestrator.handle_request(text1) # This should trigger ADD_COMPLEX_KNOWLEDGE

#     # Example 2: A question
#     text2 = "Who do I know that lives in Tokyo?"
#     print(f"\n[INPUT] '{text2}'")
#     response2 = orchestrator.handle_request(text2)
#     print(f"[AURA'S RESPONSE] {response2}")

#     # Example 3: A simple note
#     text3 = "Idea: a proactive travel planner."
#     print(f"\n[INPUT] '{text3}'")
#     response3 = orchestrator.handle_request(text3)
#     print(f"[AURA'S RESPONSE] {response3}")
