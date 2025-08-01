# main.py
# This is the simplified API server. Its only job is to receive requests
# and pass them to the orchestrator.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# Import the full system
from aura_brain import AuraBrain
from aura_agent import AuraAgent
from aura_orchestrator import AuraOrchestrator

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- INITIALIZE THE FULL SYSTEM ON STARTUP ---
try:
    brain = AuraBrain()
    agent = AuraAgent(brain)
    orchestrator = AuraOrchestrator(agent)
    logging.info("Full Aura system (Brain, Agent, Orchestrator) is initialized and ready.")
except Exception as e:
    logging.error(f"CRITICAL: Failed to initialize the Aura system. Error: {e}")
    orchestrator = None

# --- FASTAPI APPLICATION ---
app = FastAPI(
    title="Aura Agent API",
    description="A unified endpoint for interacting with the Aura personal AI."
)

# --- API MODELS ---
class UserRequest(BaseModel):
    text: str

# --- API ENDPOINT ---

@app.on_event("shutdown")
def shutdown_event():
    """Ensure the brain is saved when the server shuts down."""
    if orchestrator and orchestrator.agent and orchestrator.agent.brain:
        logging.info("Server shutting down. Saving Aura's brain...")
        orchestrator.agent.brain.save()

@app.post("/handle", summary="Handle any user request")
def handle_user_request(request: UserRequest):
    """
    This single endpoint takes any text from the user, passes it to the
    orchestrator, and returns the final, synthesized response.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Aura system is not available.")
    
    try:
        final_response = orchestrator.handle_request(request.text)
        return {"response": final_response}
    except Exception as e:
        logging.error(f"An unexpected error occurred in the orchestrator: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while handling your request.")
