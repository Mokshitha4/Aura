# main.py
# This is the simplified API server. Its only job is to receive requests
# and pass them to the supervisor.

# --- SETUP INSTRUCTIONS ---
# 1. Make sure 'aura_brain.py', 'agent.py', and 'supervisor.py' are in the same directory.
# 2. Run the server from your terminal:
#    uvicorn main:app --reload

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
# *** FIX APPLIED: Import CORSMiddleware ***
from fastapi.middleware.cors import CORSMiddleware


# Import the compiled LangGraph app and the brain instance from our supervisor file
from supervisor import supervisor_app, brain

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FASTAPI APPLICATION ---
app = FastAPI(
    title="Aura Agent API",
    description="A unified endpoint for interacting with the Aura personal AI."
)

# *** FIX APPLIED: Add CORSMiddleware to handle browser security (CORS) ***
# This allows your frontend (chat.html and the extension) to talk to this backend server.
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "null", # Allows opening chat.html directly as a file
    # Add the origin of your Chrome extension if you know it, but "*" is fine for local dev
    "*" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods, including OPTIONS and POST
    allow_headers=["*"], # Allow all headers
)


# --- API MODELS ---
class UserRequest(BaseModel):
    text: str

# --- API ENDPOINT ---

@app.on_event("shutdown")
def shutdown_event():
    """Ensure the brain is saved when the server shuts down."""
    if brain:
        logging.info("Server shutting down. Saving Aura's brain...")
        brain.save()

@app.post("/handle", summary="Handle any user request")
def handle_user_request(request: UserRequest):
    """
    This single endpoint takes any text from the user, passes it to the
    supervisor, and returns the final, synthesized response.
    """
    if not supervisor_app:
        raise HTTPException(status_code=503, detail="Aura system is not available.")
    
    try:
        # Use .invoke() for a single, final response, which is better for an API
        final_state = supervisor_app.invoke({"input": request.text})
        final_response = final_state.get("response", "Sorry, I had trouble generating a final response.")
        
        return {"response": final_response}
    except Exception as e:
        logging.error(f"An unexpected error occurred in the supervisor graph: {e}")
        # Add more detailed error logging for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal error occurred while handling your request.")

